"""Probe a network camera end to end, before its URL goes anywhere near config_local.py.

Every step a camera can fail at, in the order it fails at them, with the diagnosis attached:
is the host reachable, is RTSP listening at all, does the camera accept these credentials,
which stream paths actually exist on it, what does each one cost to decode, and what
motion_min_area does that resolution want. Read-only -- it opens streams and reads frames,
and writes nothing but the test frames you ask for.

Run it with the project's venv interpreter (a bare `python` on the rig is the system one, which
has no OpenCV -- it says so rather than throwing a traceback):

    .venv\\Scripts\\python.exe tools/camprobe.py 192.168.1.50 --user rig   # CAM_PASS from the env
    python tools/camprobe.py 192.168.1.50 --user rig --password swordfish
    python tools/camprobe.py 192.168.1.50 --user rig --path h264Preview_01_sub
    python tools/camprobe.py --url rtsp://rig:swordfish@192.168.1.50:554/h264Preview_01_sub

With no --path it tries a list of known vendor paths and reports which ones open, so an
unfamiliar camera does not need its manual found first.

Passwords are MASKED in everything this prints. The point of the exercise is a credential you
are about to write down, and this output is exactly what gets pasted into a chat window when
it does not work. Pass --show-secret if you want the literal config line.

SAVE THE TEST FRAME AND LOOK AT IT. The single most useful output here is not a number: it is
the picture. On 2026-08-21 it was a frame of a closet and a vacuum cleaner that revealed the
new camera was still sitting on the floor indoors, which is what decided sub-stream over main.
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import sys
import time
from pathlib import Path

# Force TCP for RTSP before anything opens a stream: UDP packet loss tears H.264 frames, and
# this is the same transport the rig itself uses (backyard_cam.run), so what we measure here is
# what the rig will get.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))               # tools/ sits one below the project root

VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

try:
    import cv2                                   # noqa: E402
    import config                                # noqa: E402
    from cameras import parse_stream_url         # noqa: E402  (one URL parser, not two)
except ModuleNotFoundError as exc:               # pragma: no cover - depends on the interpreter
    # A bare `python` is not this project's interpreter: on the rig it resolves to the system
    # Python, which has no OpenCV. Fail with the command to run instead of a raw traceback --
    # this tool gets reached for while something is already going wrong with a camera, which is
    # the worst moment to also have to debug an environment. (Same spirit as config.py's
    # version check.)
    raise SystemExit(
        f"camprobe needs this project's virtualenv -- no module named {exc.name!r}.\n\n"
        f"Run it with the venv's interpreter rather than a bare `python`:\n\n"
        f"    {VENV_PY} tools/camprobe.py ...\n\n"
        f"(or activate the venv first, which is what the README's commands assume)"
    )

# Stream paths worth trying when you don't know the camera. One entry per (vendor, stream), in
# the order a wildlife rig wants to see them: the cheap sub-stream first, because that is the
# one you will start on. Not exhaustive -- a camera that answers none of these still has its
# path in its own app, usually under "RTSP" or "Advanced".
CANDIDATE_PATHS = [
    ("Reolink, sub",            "h264Preview_01_sub"),
    ("Reolink, main",           "h264Preview_01_main"),
    ("Reolink, main (H.265)",   "h265Preview_01_main"),
    ("Dahua / Amcrest, sub",    "cam/realmonitor?channel=1&subtype=1"),
    ("Dahua / Amcrest, main",   "cam/realmonitor?channel=1&subtype=0"),
    ("Hikvision, sub",          "Streaming/Channels/102"),
    ("Hikvision, main",         "Streaming/Channels/101"),
    ("Tapo / generic, sub",     "stream2"),
    ("Tapo / generic, main",    "stream1"),
    ("Axis",                    "axis-media/media.amp"),
]


def port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    """One TCP connect. A closed port on a camera that answers ping is the signature of an
    RTSP service that was never enabled (or was enabled and not rebooted into)."""
    s = socket.socket()
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def rtsp_challenge(host: str, port: int, path: str, timeout: float = 6.0) -> tuple[str, dict]:
    """Send ONE unauthenticated DESCRIBE and read what the server says.

    A 401 here is the GOOD outcome: it proves an RTSP server is behind the port and tells you
    which auth scheme and realm it wants. Returns (status line, parsed WWW-Authenticate params).
    """
    uri = f"rtsp://{host}:{port}/{path}"
    try:
        s = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        return f"(no connection: {exc})", {}
    try:
        s.sendall((f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: 1\r\n"
                   "User-Agent: camprobe\r\nAccept: application/sdp\r\n\r\n").encode())
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = s.recv(4096)
            if not chunk:
                break
            raw += chunk
    except OSError as exc:
        return f"(no reply: {exc})", {}
    finally:
        s.close()
    text = raw.decode("utf-8", "replace")
    status = text.splitlines()[0].strip() if text.strip() else "(empty reply)"
    params = dict(re.findall(r'(\w+)="([^"]*)"', text))
    if "Digest" in text:
        params["scheme"] = "Digest"
    elif "Basic" in text:
        params["scheme"] = "Basic"
    return status, params


def measure(cap: "cv2.VideoCapture", seconds: float, warmup: float) -> dict:
    """Sustained read rate and decode cost, with the open-time buffer excluded.

    The warmup matters more than it looks: a first burst can be drained buffer rather than live
    decode, which on 2026-08-21 read a 4K stream at "30 fps" for two seconds before settling to
    its real 25. Anything that decides a config value gets measured after the warmup.
    """
    end_warm = time.time() + warmup
    while time.time() < end_warm:
        cap.read()
    frames, first = 0, None
    cpu0, wall0 = time.process_time(), time.time()
    while time.time() - wall0 < seconds:
        ok, frame = cap.read()
        if ok:
            frames += 1
            if first is None:
                first = frame
    wall = max(time.time() - wall0, 1e-6)
    return {"fps": frames / wall, "cpu_pct": (time.process_time() - cpu0) / wall * 100,
            "frames": frames, "frame": first}


def suggested_motion_area(cfg, w: int, h: int) -> int:
    """motion_min_area for a w*h camera, holding the reference camera's TRIGGER FRACTION.

    The gate runs on a downscaled copy but converts blob areas back to full-frame pixels, so
    this knob is always full-frame pixels -- which makes it resolution-dependent and the single
    easiest thing to get wrong when adding a camera. Scaling by pixel count keeps a new camera
    as sensitive, in fraction-of-frame terms, as the one already tuned.
    """
    ref_px = (cfg.frame_width or 1920) * (cfg.frame_height or 1080)
    return max(1, round((cfg.motion_min_area or 800) * (w * h) / ref_px))


def parse_rtsp_url(url: str) -> tuple[str, str, str, int, str] | None:
    """Split rtsp://[user[:pass]@]host[:port]/path into its parts, or None if it isn't one.

    The parsing itself is cameras.parse_stream_url -- the same one the rig uses to decompose a
    camera URL into its DB columns, so a URL that camprobe accepts is a URL the camera list can
    store. This wrapper only narrows it to RTSP and flattens it to the tuple main() wants.
    """
    parts = parse_stream_url(url, schemes=("rtsp",))
    if parts is None:
        return None
    return (parts["username"] or "", parts["password"] or "", parts["url_host"],
            parts["url_port"] or 554, parts["url_path"] or "")


def probe_path(host, port, user, pw, label, path, args, cfg) -> dict | None:
    """Open one stream path, measure it, and report. None if it would not open."""
    url = f"rtsp://{user}:{pw}@{host}:{port}/{path}" if user else f"rtsp://{host}:{port}/{path}"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        print(f"  {label:24} {path:42} --")
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # freshest frame only; what the rig does
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m = measure(cap, args.seconds, args.warmup)
    cap.release()
    area = suggested_motion_area(cfg, w, h)
    # Below a tenth of a core the number is Windows timer granularity, not decode cost.
    cpu = "  <0.1" if m["cpu_pct"] < 0.1 else f"{m['cpu_pct']:6.1f}"
    print(f"  {label:24} {path:42} {w}x{h}  {m['fps']:5.1f} fps  "
          f"{cpu}% of one core  motion_min_area={area}")
    out = {"label": label, "path": path, "url": url, "w": w, "h": h, "area": area, **m}
    if m["frame"] is not None and args.frames:
        d = Path(args.frames)
        d.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", path)[:60]
        p = d / f"camprobe_{host.replace('.', '-')}_{safe_name}.jpg"
        cv2.imwrite(str(p), m["frame"])
        out["frame_path"] = p
        print(f"  {'':24} {'':42} frame -> {p}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Probe a network camera before adding it to config_local.py.")
    ap.add_argument("host", nargs="?", help="camera IP or hostname")
    ap.add_argument("--url", help="a full rtsp:// URL, instead of host/user/password/path")
    ap.add_argument("--user", default=os.environ.get("CAM_USER", ""),
                    help="camera user (a DEDICATED one, not admin). Or set CAM_USER.")
    ap.add_argument("--password", default=os.environ.get("CAM_PASS", ""),
                    help="camera password. Prefer the CAM_PASS environment variable, which "
                         "keeps it out of your shell history.")
    ap.add_argument("--port", type=int, default=554)
    ap.add_argument("--path", help="probe only this stream path (skips the vendor guesses)")
    ap.add_argument("--seconds", type=float, default=8.0, help="measurement window per stream")
    ap.add_argument("--warmup", type=float, default=3.0,
                    help="discarded before measuring, so buffered frames don't inflate the fps")
    ap.add_argument("--frames", default=str(ROOT / "reports" / "camprobe"), metavar="DIR",
                    help="save one test frame per working stream here ('' to skip). LOOK AT THEM.")
    ap.add_argument("--show-secret", action="store_true",
                    help="print the config line with the real password instead of a placeholder")
    args = ap.parse_args()

    cfg = config.CONFIG
    if args.url:
        parsed = parse_rtsp_url(args.url)
        if parsed is None:
            print("Could not parse that URL. Expected rtsp://[user:pass@]host[:port]/path")
            return 2
        user, pw, host, port, path = parsed
        paths = [("from --url", path)]
    else:
        if not args.host:
            ap.error("give a host (or --url)")
        host, port, user, pw = args.host, args.port, args.user, args.password
        paths = [("requested", args.path)] if args.path else CANDIDATE_PATHS

    print(f"\ncamprobe: {host}:{port}"
          + (f"  as user {user!r}" if user else "  (no credentials given)"))

    # --- 1. Is anything listening? -----------------------------------------------------
    if not port_open(host, port):
        print(f"\n  port {port}: CLOSED")
        print("\n  The camera may still be up -- a closed 554 on a camera that answers ping is")
        print("  almost always RTSP being switched OFF rather than a network problem.")
        print("  On Reolink: app -> Settings -> Network -> Advanced -> Server Settings -> tick")
        print("  RTSP, Save, and then REBOOT the camera. The port stays shut until it restarts.")
        return 1
    print(f"  port {port}: open")

    # --- 2. What kind of server, and what auth does it want? ---------------------------
    status, params = rtsp_challenge(host, port, paths[0][1] or "")
    scheme, realm = params.get("scheme", "?"), params.get("realm", "?")
    print(f"  server:      {status}   auth={scheme} realm={realm!r}")
    if "401" in status:
        print("               (a 401 here is CORRECT -- it proves RTSP is answering)")

    # --- 3. Which paths open, and what do they cost? -----------------------------------
    print(f"\n  probing {len(paths)} path(s), {args.warmup:g}s warmup + {args.seconds:g}s measured each:\n")
    working = [r for r in (probe_path(host, port, user, pw, label, path, args, cfg)
                           for label, path in paths) if r]

    if not working:
        print("\n  Nothing opened. Read FFmpeg's own line above -- it names the cause:")
        print("    '401 Unauthorized'  -> the CREDENTIALS. On a cloud-managed camera the app")
        print("                           login is an account with the vendor, NOT a user on the")
        print("                           camera; RTSP only knows the latter. Make a dedicated")
        print("                           camera user in the app and try that.")
        print("    '404 Not Found'     -> the PATH. The credentials are fine; this vendor names")
        print("                           its streams something not in the list above.")
        print("    nothing at all      -> the stream exists but sent no frames, which usually")
        print("                           means that channel is disabled on the camera.")
        return 1

    # --- 4. The line you actually need -------------------------------------------------
    print(f"\n  {len(working)} working stream(s). Ready for config_local.py:\n")
    shown_pw = pw if args.show_secret else "***"
    for r in working:
        url = f"rtsp://{user}:{shown_pw}@{host}:{port}/{r['path']}"
        cost = "<0.1" if r["cpu_pct"] < 0.1 else f'{r["cpu_pct"]:.1f}'
        print(f'    # {r["label"]}: {r["w"]}x{r["h"]}, {r["fps"]:.1f} fps, '
              f'{cost}% of one core')
        print(f'    CameraSpec("<source>", "{url}",')
        print(f'               name="<display name>", frame_width={r["w"]}, '
              f'frame_height={r["h"]},')
        print(f'               motion_min_area={r["area"]}),')
    if not args.show_secret:
        print("\n  (password masked -- put the real one in, or re-run with --show-secret)")
    print(f"\n  motion_min_area above holds the reference camera's trigger fraction: "
          f"{cfg.motion_min_area} px at {cfg.frame_width}x{cfg.frame_height} = "
          f"{cfg.motion_min_area / ((cfg.frame_width or 1920) * (cfg.frame_height or 1080)) * 100:.3f}% "
          f"of frame.")
    if args.frames:
        print(f"  Test frames in {args.frames} -- OPEN THEM before you trust any of these numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
