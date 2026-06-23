"""
Camera image-quality tuning experiment -- repeatable, for after you move the camera.

What it does: grabs a baseline frame at the camera's current (auto) settings, then sweeps a
range of manual EXPOSURE values, scoring each frame for sharpness (focus / motion-blur) and
highlight clipping (blown-out). It writes a labeled contact sheet + a metrics table and
prints a recommended setting. Re-run it whenever you reposition the camera or the light
changes (day vs night) -- each run is archived under tuning/<timestamp>/ so you can compare.

Metrics:
  sharpness  = variance of the Laplacian (higher = sharper; low = blurry/out of focus)
  brightness = mean grey level 0-255 (well-exposed is roughly 90-150)
  blown_pct  = % of pixels clipped to white (>=250) -- the "blown out" number; want it low
  black_pct  = % of pixels crushed to black (<=5)

Note: only one process can open the webcam, so STOP the capture rig first. The detector is
not used here, so this is quick.

Usage:
  python tune.py                         # sweep on the configured camera
  python tune.py --camera-index 1 --exposures "-4,-5,-6,-7,-8"
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
from config import CONFIG

FONT = cv2.FONT_HERSHEY_SIMPLEX


def metrics(frame_bgr) -> dict:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var()),
        "brightness": float(g.mean()),
        "blown_pct": float((g >= 250).mean() * 100.0),
        "black_pct": float((g <= 5).mean() * 100.0),
    }


def open_cam(cfg) -> cv2.VideoCapture:
    # DirectShow is Windows-only; off Windows use CAP_ANY so tune.py runs on Linux/macOS too
    # (matches backyard_cam.capture_backend). The exposure sweep below still assumes a UVC webcam.
    backend = cv2.CAP_DSHOW if (cfg.use_dshow_backend and sys.platform == "win32") else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    return cap


def grab(cap, settle: int = 8):
    """Read several frames so the new setting takes effect, return the last good one."""
    frame = None
    ok = False
    for _ in range(settle):
        ok, frame = cap.read()
        time.sleep(0.03)
    return frame if ok else None


def probe(cap) -> None:
    props = {
        "AUTO_EXPOSURE": cv2.CAP_PROP_AUTO_EXPOSURE, "EXPOSURE": cv2.CAP_PROP_EXPOSURE,
        "GAIN": cv2.CAP_PROP_GAIN, "BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
        "CONTRAST": cv2.CAP_PROP_CONTRAST, "GAMMA": cv2.CAP_PROP_GAMMA,
        "AUTO_WB": cv2.CAP_PROP_AUTO_WB, "WB_TEMP": cv2.CAP_PROP_WB_TEMPERATURE,
        "BACKLIGHT": cv2.CAP_PROP_BACKLIGHT, "FPS": cv2.CAP_PROP_FPS,
        "WIDTH": cv2.CAP_PROP_FRAME_WIDTH, "HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
    }
    print("Camera controls (current values):")
    for name, pid in props.items():
        print(f"  {name:14} {cap.get(pid)}")


def _label(frame, text: str):
    out = frame.copy()
    for i, ln in enumerate(text.split("\n")):
        y = 26 + i * 26
        cv2.putText(out, ln, (10, y), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, ln, (10, y), FONT, 0.7, (40, 255, 255), 1, cv2.LINE_AA)
    return out


def contact_sheet(tiles, cols: int = 3, tile_w: int = 460):
    rs = []
    for t in tiles:
        h, w = t.shape[:2]
        rs.append(cv2.resize(t, (tile_w, max(1, int(h * tile_w / w)))))
    th = max(t.shape[0] for t in rs)
    rows = []
    for i in range(0, len(rs), cols):
        row = rs[i:i + cols]
        row = [cv2.copyMakeBorder(r, 0, th - r.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
               for r in row]
        while len(row) < cols:
            row.append(np.full((th, tile_w, 3), 20, np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> int:
    c = CONFIG
    p = argparse.ArgumentParser(description="Camera image-quality tuning (exposure sweep).")
    p.add_argument("--camera-index", type=int, default=c.camera_index)
    p.add_argument("--width", type=int, default=c.frame_width)
    p.add_argument("--height", type=int, default=c.frame_height)
    p.add_argument("--exposures", default="-3,-4,-5,-6,-7,-8,-9,-10",
                   help="Comma-separated manual EXPOSURE values to sweep (DirectShow ~ log2 seconds).")
    p.add_argument("--manual-flag", type=float, default=0.25,
                   help="Value that puts the camera in MANUAL exposure (0.25 on most UVC cams; some want 1).")
    args = p.parse_args()

    cfg = replace(c, camera_index=args.camera_index, frame_width=args.width, frame_height=args.height)
    cap = open_cam(cfg)
    if not cap.isOpened():
        cap.release()           # the VideoCapture object exists even on a failed open -- free it
        print(f"Could not open camera index {cfg.camera_index}. Is the capture rig still "
              f"running? Stop it first (only one process can hold the webcam).")
        return 1

    outdir = config.ROOT / "tuning" / datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nTuning run -> {outdir}\n")

    tiles, results = [], []
    try:
        probe(cap)

        # 1) Baseline at the camera's current AUTO settings (what the rig uses now).
        base = grab(cap, settle=12)
        if base is not None:
            m = metrics(base)
            results.append(("auto", cap.get(cv2.CAP_PROP_EXPOSURE), m))
            cv2.imwrite(str(outdir / "00_baseline_auto.jpg"), base)
            tiles.append(_label(base, f"AUTO (current)\nsharp {m['sharpness']:.0f}\n"
                                       f"bright {m['brightness']:.0f}\nblown {m['blown_pct']:.1f}%"))
            print(f"\nbaseline AUTO: sharp={m['sharpness']:.0f} bright={m['brightness']:.0f} "
                  f"blown={m['blown_pct']:.2f}% black={m['black_pct']:.2f}%")

        # 2) Sweep manual exposure.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, args.manual_flag)
        time.sleep(0.2)
        try:
            exposures = [float(x) for x in args.exposures.split(",") if x.strip()]
        except ValueError:
            print(f"--exposures must be comma-separated numbers; got '{args.exposures}'.")
            return 1
        print("\nexposure sweep (requested -> actual readback):")
        for e in exposures:
            cap.set(cv2.CAP_PROP_EXPOSURE, e)
            frame = grab(cap, settle=10)
            if frame is None:
                continue
            actual = cap.get(cv2.CAP_PROP_EXPOSURE)
            m = metrics(frame)
            results.append((e, actual, m))
            cv2.imwrite(str(outdir / f"exp_{e:g}.jpg"), frame)
            tiles.append(_label(frame, f"exp={e:g}\nsharp {m['sharpness']:.0f}\n"
                                        f"bright {m['brightness']:.0f}\nblown {m['blown_pct']:.1f}%"))
            print(f"  {e:>6g} -> {actual:>8g}   sharp={m['sharpness']:>6.0f} "
                  f"bright={m['brightness']:>5.0f} blown={m['blown_pct']:>5.2f}%")

        # Restore auto exposure so the next app/rig isn't stuck in manual.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    finally:
        cap.release()    # always release, even if the sweep raised -- else the webcam stays locked

    if tiles:
        sheet = contact_sheet(tiles)
        sheet_path = outdir / "contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet)
    else:
        print("No frames captured.")
        return 1

    # Metrics table on disk.
    with open(outdir / "metrics.csv", "w") as f:
        f.write("exposure,actual,sharpness,brightness,blown_pct,black_pct\n")
        for e, actual, m in results:
            f.write(f"{e},{actual},{m['sharpness']:.1f},{m['brightness']:.1f},"
                    f"{m['blown_pct']:.3f},{m['black_pct']:.3f}\n")

    # Recommend: sharpest frame that is well-exposed (low clipping, mid brightness).
    swept = [(e, m) for e, _, m in results if e != "auto"]
    good = [(e, m) for e, m in swept if m["blown_pct"] < 1.0 and 80 <= m["brightness"] <= 160]
    if good:
        pick = max(good, key=lambda em: em[1]["sharpness"])
        why = "sharpest frame with <1% blown and brightness 80-160"
    elif swept:
        pick = min(swept, key=lambda em: em[1]["blown_pct"])  # at least un-blow it
        why = "least-blown frame (none hit the ideal band -- consider a feeding-spot light)"
    else:
        pick = None

    print(f"\ncontact sheet: {sheet_path}")
    if pick:
        e, m = pick
        print(f"\nRECOMMENDED exposure = {e:g}  ({why})")
        print(f"  -> sharp={m['sharpness']:.0f} bright={m['brightness']:.0f} blown={m['blown_pct']:.2f}%")
        print(f"  Apply it: set `exposure = {e:g}` in config.py, or run with `--exposure {e:g}`.")
        print("  (One value won't fit both day and night -- re-tune, or we add day/night profiles.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
