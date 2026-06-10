"""
Backyard critter cam -- live glass-door rig (the primary all-species source, day + night).

Pipeline per frame:
    capture (OpenCV/DirectShow)
      -> MOG2 motion gate  (cheap; skip the detector on still frames)
      -> MegaDetector v6   (only on frames with real motion, rate-limited)
      -> for each kept detection: draw box, save crop (+ optional frame), write a SQLite row
      -> live preview window with boxes + confidence; press 'q' to quit.

Everything tunable lives in config.py; the most common knobs are also CLI flags below.
Run `python backyard_cam.py --list-cameras` to find your webcam's index.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2

import clips
import config
import daynight
import db
import stats
import visits
import web
from config import CONFIG
from detector import CudaUnavailableError, Detection, Detector

# ---- Drawing constants -------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLORS = {                      # BGR
    "animal": (60, 220, 60),
    "person": (255, 170, 0),
    "vehicle": (0, 165, 255),
}
DEFAULT_BOX_COLOR = (0, 255, 255)

# Ignore motion for the first few frames so MOG2 can build its background model instead of
# flagging the whole scene as "motion" on startup.
MOTION_WARMUP_FRAMES = 15
# Tolerate this many consecutive bad reads (transient USB hiccups) before we treat the
# camera as disconnected and try a full reopen.
READ_FAIL_TOLERANCE = 10


# ---- Camera ------------------------------------------------------------------------
def apply_camera_settings(cap, settings: dict) -> None:
    """Apply a camera-settings dict to an open capture: 'exposure' (None = auto-expose, a
    number = locked manual), 'gain' (None = leave alone), and any other KEY ->
    cv2.CAP_PROP_<KEY> (e.g. 'BACKLIGHT': 0). Used for both the static base (open_camera) and
    the sun-driven time-of-day profiles (run)."""
    if "exposure" in settings:
        e = settings["exposure"]
        if e is None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # 0.75 = auto on most UVC/DSHOW cams
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual
            cap.set(cv2.CAP_PROP_EXPOSURE, float(e))
    for key, val in settings.items():
        if key == "exposure":
            continue
        if key == "gain":
            if val is not None:
                cap.set(cv2.CAP_PROP_GAIN, float(val))
            continue
        prop = getattr(cv2, f"CAP_PROP_{key}", None)
        if prop is not None:
            cap.set(prop, val)


# Controls surfaced to the dashboard; key -> cv2.CAP_PROP_<NAME>.
_CONTROL_PROPS = {
    "exposure": "EXPOSURE", "gain": "GAIN", "focus": "FOCUS", "brightness": "BRIGHTNESS",
    "contrast": "CONTRAST", "saturation": "SATURATION", "sharpness": "SHARPNESS",
    "gamma": "GAMMA", "wb": "WB_TEMPERATURE", "auto_exposure": "AUTO_EXPOSURE",
    "autofocus": "AUTOFOCUS", "auto_wb": "AUTO_WB",
}


def read_camera_controls(cap) -> dict:
    """Read the camera's current control values so the dashboard sliders show real positions."""
    vals = {}
    for key, prop in _CONTROL_PROPS.items():
        p = getattr(cv2, f"CAP_PROP_{prop}", None)
        try:
            vals[key] = cap.get(p) if p is not None else None
        except Exception:
            vals[key] = None
    return vals


# Slider controls we can safely probe for write-support: skip the auto* toggles, and skip
# 'exposure' (its auto/manual mode makes a blind nudge ambiguous -- assume it's settable).
_PROBE_KEYS = tuple(k for k in _CONTROL_PROPS if not k.startswith("auto") and k != "exposure")


def probe_writable_controls(cap) -> dict:
    """Return {control_key: bool} -- whether this camera actually lets us CHANGE each slider
    control. cv2's cap.set returns False when the driver rejects a property; but some drivers
    ACK a no-op set-to-CURRENT value while still rejecting any real change (this webcam does
    exactly that for FOCUS). So we set a genuinely different value, trying both directions to
    stay in range, then restore the original. The dashboard locks controls that come back
    False. Runs once at startup -- a brief, pre-capture-loop blip on the supported controls."""
    out = {}
    for key in _PROBE_KEYS:
        p = getattr(cv2, f"CAP_PROP_{_CONTROL_PROPS[key]}", None)
        if p is None:
            out[key] = False
            continue
        try:
            cur = cap.get(p)
            out[key] = bool(cap.set(p, cur + 32.0)) or bool(cap.set(p, cur - 32.0))
            cap.set(p, cur)  # restore whatever it was before the probe
        except Exception:
            out[key] = False
    return out


def capture_backend(cfg: config.Config) -> int:
    """Pick the OpenCV capture backend. DirectShow is a Windows-only API, so we only use it on
    Windows (and only when enabled); on Linux/macOS we pass CAP_ANY and let OpenCV choose the
    native backend (V4L2 / AVFoundation). This is what lets the same code run on the Linux box."""
    if cfg.use_dshow_backend and sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def open_camera(cfg: config.Config) -> cv2.VideoCapture:
    """Open the webcam with the configured backend/resolution and warm it up."""
    cap = cv2.VideoCapture(cfg.camera_index, capture_backend(cfg))
    if cfg.frame_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    if cfg.frame_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    if cfg.exposure is not None:  # lock manual exposure (0.25 = manual on most UVC/DSHOW cams)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)
    if cfg.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, cfg.gain)
    for _name, _val in cfg.camera_controls.items():  # static UVC tweaks (e.g. BACKLIGHT off)
        _prop = getattr(cv2, f"CAP_PROP_{_name}", None)
        if _prop is not None:
            cap.set(_prop, _val)
    for _ in range(cfg.camera_warmup_frames):  # let exposure / auto-WB settle
        cap.read()
    return cap


def reopen_camera(cap: cv2.VideoCapture, cfg: config.Config) -> cv2.VideoCapture | None:
    """Camera-disconnect handling: release and retry up to the configured limit.

    Returns a working capture, or None if it never came back (caller shuts down cleanly --
    we never let a disconnect crash the loop with a traceback).
    """
    print("[camera] read failed -- attempting to reopen ...")
    try:
        cap.release()
    except Exception:
        pass
    for attempt in range(1, cfg.reopen_max_retries + 1):
        time.sleep(cfg.reopen_delay_s)
        new_cap = open_camera(cfg)
        if new_cap is not None and new_cap.isOpened():
            ok, _ = new_cap.read()
            if ok:
                print(f"[camera] reopened on attempt {attempt}.")
                return new_cap
            new_cap.release()
        print(f"[camera] reopen attempt {attempt}/{cfg.reopen_max_retries} failed.")
    print("[camera] giving up after exhausting reopen retries.")
    return None


def list_cameras(cfg: config.Config, max_index: int = 8) -> None:
    """Probe indices 0..max_index-1 with the platform's backend and report which ones open."""
    backend = capture_backend(cfg)
    backend_name = "DirectShow" if backend == cv2.CAP_DSHOW else "the default backend"
    print(f"Probing camera indices via {backend_name} (this can take a few seconds)...")
    found = False
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  index {i}: OPEN   {w}x{h}   read_ok={ok}")
            found = True
        cap.release()
    if not found:
        print("  No cameras opened. Is the webcam plugged in / not in use by another app?")


# ---- Stats readout (--stats) -------------------------------------------------------
def print_stats(cfg: config.Config) -> None:
    """Print the shared stats summary (stats.compute_stats) to the console and exit.
    Read-only and WAL-safe, so it's fine to run while the live rig is capturing."""
    s = stats.compute_stats(cfg)
    if s is None:
        print(f"No database yet at {cfg.db_path}. Run the rig to capture some detections first.")
        return
    if s["total_crops"] == 0:
        print(f"{cfg.db_path}: no detections yet.")
        return

    gap = s["gap_minutes"]

    def day(ts):  return ts[:10]
    def hhmm(ts): return ts[11:16]

    lo, hi = s["span"]["start"], s["span"]["end"]
    n_days = len(s["by_day"])
    print()
    print(f"Backyard Critter Cam -- stats   ({Path(s['db_path']).name})")
    print("-" * 66)
    print(f"Detections (crops): {s['total_crops']:<6}  Visits (est, >{gap:g} min gap): {s['total_visits']}")
    print(f"Span: {day(lo)} {hhmm(lo)}  ->  {day(hi)} {hhmm(hi)}   "
          f"({n_days} day{'' if n_days == 1 else 's'})")

    print("\nBy source:")
    for r in s["by_source"]:
        print(f"  {r['source']:<18} {r['crops']:>5} crops   {r['visits']:>4} visits")

    print("\nBy class (detector class now; species fills in at phase 2):")
    for r in s["by_class"]:
        print(f"  {r['name']:<18} {r['crops']:>5} crops")

    print("\nBy day:")
    for d in s["by_day"]:
        cls_str = ", ".join(f"{k}:{v}" for k, v in d["classes"].items())
        print(f"  {d['day']}   {d['crops']:>5} crops   {d['visits']:>4} visits   [{cls_str}]")

    if s["by_hour"]:
        print("\nArrivals by hour (visit starts, local time):")
        peak = max(h["visits"] for h in s["by_hour"])
        for h in s["by_hour"]:
            print(f"  {h['hour']:02d}h  {'#' * max(1, round(20 * h['visits'] / peak))} {h['visits']}")

    print("\nLatest catches:")
    for r in s["latest"][:5]:
        sp = f" {r['species']}" if r["species"] else ""
        print(f"  {day(r['timestamp'])} {hhmm(r['timestamp'])}  "
              f"{r['detection_class']}{sp}  {r['confidence']:.2f}  {r['crop_path']}")

    print()
    print(f"Note: 'visits' is a V1 estimate -- consecutive detections on one source <={gap:g} min")
    print("apart, collapsed into one event. It can't yet separate two animals present at once")
    print("(needs phase-3 individual IDs). Count visits, not crops, for activity stats.")


# ---- Motion gate -------------------------------------------------------------------
class MotionGate:
    """MOG2 background subtractor -> 'largest motion blob area' in pixels.

    MOG2 adapts to gradual outdoor light changes; we drop its shadow pixels (marked 127)
    and de-noise with a blur + morphology so a single flickering pixel never triggers.
    """

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.motion_history,
            varThreshold=cfg.motion_var_threshold,
            detectShadows=cfg.motion_detect_shadows,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update(self, frame_bgr) -> float:
        k = self.cfg.motion_blur_ksize
        src = frame_bgr
        if k and k > 1:
            k = k | 1  # GaussianBlur needs an odd kernel
            src = cv2.GaussianBlur(frame_bgr, (k, k), 0)
        mask = self.bg.apply(src)
        # Keep only strong foreground (255); shadows come back as 127 and are discarded.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        return max(cv2.contourArea(c) for c in contours)


# ---- Saving ------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """Store paths relative to the project root when possible (keeps the DB portable)."""
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


def save_crop(frame_bgr, det: Detection, cfg: config.Config, day: str, stamp: str, idx: int):
    """Crop the (padded, clamped) detection box and write it as a JPEG. Returns the Path."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = det.bbox
    pad_x = (x2 - x1) * cfg.crop_padding
    pad_y = (y2 - y1) * cfg.crop_padding
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    day_dir = cfg.crops_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{stamp}_{idx}_{det.class_name}_{det.confidence:.2f}.jpg"
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    return path


def save_frame(frame_bgr, cfg: config.Config, day: str, stamp: str):
    """Save the full frame (only when cfg.save_full_frame). Returns the Path."""
    day_dir = cfg.frames_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{stamp}.jpg"
    cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    return path


# ---- Preview overlays --------------------------------------------------------------
def draw_detections(frame, detections: list[Detection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = BOX_COLORS.get(det.class_name, DEFAULT_BOX_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        ly = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, ly), (x1 + tw + 6, ly + th + 6), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly + th + 1), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, *, fps: float, motion_area: float, motion: bool,
             saved: int, source: str, model: str, period: str | None = None,
             recording: bool = False) -> None:
    lines = [
        f"{source}  |  {model}" + (f"  |  {period}" if period else ""),
        f"FPS {fps:4.1f}   motion {int(motion_area):>6} px   saved {saved}",
        "q: quit",
    ]
    y = 22
    for text in lines:
        cv2.putText(frame, text, (10, y), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)      # shadow
        cv2.putText(frame, text, (10, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    if motion:  # red "motion" dot, top-right
        cv2.circle(frame, (frame.shape[1] - 20, 20), 8, (0, 0, 255), -1)
    if recording:  # "REC" tag under the motion dot while a behaviour clip is being written
        cv2.putText(frame, "REC", (frame.shape[1] - 52, 48), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "REC", (frame.shape[1] - 52, 48), FONT, 0.6, (0, 0, 255), 1, cv2.LINE_AA)


# ---- Naming subprocess ------------------------------------------------------------
def _naming_pids(tag: str) -> list:
    """PIDs of python processes whose command line carries our unique --tag (the live-naming
    helper and any interpreter the venv launcher re-spawned for it). Windows-only, best-effort."""
    ps = ("Get-CimInstance Win32_Process | "
          f"Where-Object {{ $_.Name -like 'python*' -and $_.CommandLine -like '*{tag}*' }} | "
          "Select-Object -ExpandProperty ProcessId")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=15)
        return [p.strip() for p in out.stdout.splitlines() if p.strip()]
    except Exception:
        return []


def _stop_naming(proc, tag: str | None = None) -> None:
    """Stop the species-naming helper and EVERY process it spawned, on shutdown. On Windows the
    venv's python.exe is a launcher whose real interpreter can run OUTSIDE the launcher's process
    tree, so killing the tree alone can leave it orphaned. So we also match the helper by the
    unique --tag we put on its command line and sweep those, repeating until none remain. Silent
    and best-effort: naming is optional and may already have exited."""
    if proc is None:
        return
    if sys.platform != "win32":
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return
    for _ in range(4):
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        pids = _naming_pids(tag) if tag else []
        if not pids:
            break
        subprocess.run(["taskkill", "/F"] + [a for p in pids for a in ("/PID", p)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.4)


# ---- Main loop ---------------------------------------------------------------------
def run(cfg: config.Config) -> None:
    cfg.crops_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_full_frame:
        cfg.frames_dir.mkdir(parents=True, exist_ok=True)
    if cfg.record_clips:
        cfg.clips_dir.mkdir(parents=True, exist_ok=True)

    conn = db.connect(cfg.db_path)

    # Build the detector first: this resolves the device (a real GPU compute-check for
    # 'cuda'/'auto') and downloads the model weights on first run. With device='cuda' and no
    # usable GPU we fail here with a clear message (before touching the camera).
    print(f"Loading MegaDetector v6 ({cfg.model_version}) on {cfg.device} ...")
    print("  (first run downloads the model weights from Zenodo -- one time only)")
    detector = Detector(cfg.model_version, cfg.device, cfg.min_confidence,
                        classes=cfg.detect_classes)
    if detector.device == "cuda":
        print(f"  detector ready on GPU: {detector.device_name}")
    else:
        print("  detector ready on CPU -- slower per frame, but the motion gate keeps it usable.")

    cap = open_camera(cfg)
    if cap is None or not cap.isOpened():
        conn.close()
        raise RuntimeError(
            f"Could not open camera index {cfg.camera_index}. "
            f"Try `python backyard_cam.py --list-cameras` to find the right index."
        )
    print(f"Camera {cfg.camera_index} open. Watching for critters -- press 'q' to quit.\n")

    # Optional local web dashboard (live stream + stats + camera controls), same process.
    server = None
    frame_buffer = None
    control_bridge = None
    if cfg.serve:
        frame_buffer = web.FrameBuffer()
        control_bridge = web.CameraControlBridge()
        try:
            server = web.start(cfg, frame_buffer, control_bridge)
            print(f"  dashboard: http://{cfg.web_host}:{cfg.web_port}  (open in a browser)\n")
        except OSError as e:
            print(f"  [web] could not start the dashboard on {cfg.web_host}:{cfg.web_port}: {e}")
            print("  [web] continuing without it -- try another port with --port N.\n")
            server, frame_buffer, control_bridge = None, None, None

    # Resizable preview window (drag any edge); KEEPRATIO avoids stretching the feed.
    if cfg.show_preview:
        cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(cfg.window_name, cfg.frame_width, cfg.frame_height)

    # Sun-driven day/night camera profile, re-applied whenever the period changes. Disabled
    # when a manual --exposure/--gain was given (those force the static settings instead).
    profiles_on = (cfg.use_time_of_day_profiles and cfg.exposure is None and cfg.gain is None
                   and cfg.latitude is not None and cfg.longitude is not None)
    active_period = None
    last_profile_check = 0.0
    if profiles_on:
        try:
            st = daynight.sun_times(cfg.latitude, cfg.longitude)
            print("  sun today: " + "  ".join(f"{k} {v.strftime('%H:%M')}" for k, v in st.items()))
        except Exception:
            pass
        active_period = daynight.current_period(cfg.latitude, cfg.longitude)
        apply_camera_settings(cap, cfg.camera_profiles.get(active_period, {}))
        print(f"  camera profile: '{active_period}'\n")

    # Probe once which sliders this camera actually honors (many webcams reject manual FOCUS);
    # the dashboard locks the rest. Only when serving -- it briefly nudges supported controls.
    writable = probe_writable_controls(cap) if control_bridge is not None else {}

    # Species naming runs as a managed CHILD PROCESS (classify.py --watch), not a thread: the
    # BioCLIP model can only be BUILT on a process's main thread -- constructing it in a
    # background thread deadlocks -- so naming gets its own process. That also loads the model
    # in parallel, so the camera and preview still come up instantly. The child shares THIS
    # console (its logs print here; no extra window), and we kill it on shutdown, so there's
    # still one launch and one stop with nothing left running.
    classify_proc = None
    classify_tag = f"backyard-naming-{os.getpid()}"   # unique marker for a clean, total shutdown
    if cfg.classify_live:
        try:
            classify_proc = subprocess.Popen(
                [sys.executable, str(config.ROOT / "classify.py"), "--watch",
                 "--device", str(cfg.classify_device),
                 "--interval", str(cfg.classify_interval_s),
                 "--tag", classify_tag],
                cwd=str(config.ROOT),
            )
            print("  species naming: ON -- a helper is warming up the model (~1-2 min), then it\n"
                  "  names new crops automatically. The dashboard shows when it's ready.\n")
        except Exception as e:
            print(f"  [naming] couldn't start the species-naming helper "
                  f"(detection still works): {e}\n")
            classify_proc = None

    # Optional behaviour-clip recorder (phase 4 capture): records a short video around each
    # visit. Off unless cfg.record_clips / --record-clips. Lives in this (capture) thread.
    recorder = clips.ClipRecorder(cfg, conn) if cfg.record_clips else None
    clip_trigger = cfg.clip_classes or cfg.save_classes   # which classes start a clip
    if recorder is not None:
        print(f"  behaviour clips: ON -- recording short videos to {_rel(cfg.clips_dir)}/ "
              f"(pre {cfg.clip_pre_roll_s:g}s / post {cfg.clip_post_roll_s:g}s, "
              f"cap {cfg.clip_max_s:g}s; triggers on {', '.join(clip_trigger)}).\n")

    gate = MotionGate(cfg)
    last_detect_t = 0.0          # monotonic time of last detector run (for rate limiting)
    last_dets: list[Detection] = []
    last_dets_t = 0.0            # monotonic time the persisted boxes were produced
    total_saved = 0
    frame_count = 0
    read_fails = 0
    fps = 0.0
    fps_t = time.monotonic()
    fps_n = 0
    last_ctrl_pub = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                read_fails += 1
                if read_fails >= READ_FAIL_TOLERANCE:
                    cap = reopen_camera(cap, cfg)
                    read_fails = 0
                    if cap is None:
                        break  # graceful shutdown -- camera never came back
                    active_period = None       # camera reset on reopen -- re-apply profile
                    last_profile_check = 0.0
                continue
            read_fails = 0
            frame_count += 1

            # --- FPS (rolling, ~ once per second) ---
            fps_n += 1
            now = time.monotonic()
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0

            # --- Time-of-day camera profile (sun-driven; checked ~every 30 s) ---
            if profiles_on and now - last_profile_check >= 30.0:
                last_profile_check = now
                period = daynight.current_period(cfg.latitude, cfg.longitude)
                if period != active_period:
                    active_period = period
                    apply_camera_settings(cap, cfg.camera_profiles.get(period, {}))
                    print(f"[camera] sun crossed -- switched to '{period}' profile")

            # --- Live camera controls from the dashboard (web thread -> capture thread) ---
            if control_bridge is not None:
                pending = control_bridge.take_pending()
                if pending:
                    apply_camera_settings(cap, pending)
                if now - last_ctrl_pub >= 1.0:
                    last_ctrl_pub = now
                    snap = read_camera_controls(cap)
                    snap["period"] = active_period
                    snap["writable"] = writable
                    snap["lat"], snap["lon"] = cfg.latitude, cfg.longitude
                    control_bridge.publish(snap)

            # --- Motion gate ---
            motion_area = gate.update(frame)
            motion = (frame_count > MOTION_WARMUP_FRAMES) and (motion_area > cfg.motion_min_area)

            # --- Behaviour clip: buffer pre-roll every frame; write + auto-stop while recording ---
            if recorder is not None:
                recorder.note_frame(frame, now, loop_fps=fps)

            # --- Detector (only on motion, rate-limited) ---
            if motion and (now - last_detect_t) >= cfg.detector_min_interval_s:
                last_detect_t = now
                try:
                    dets = detector.detect(frame)
                except Exception as e:  # never let one bad frame kill the loop
                    print(f"[detector] error on a frame (skipping): {e}")
                    dets = []

                if dets:
                    last_dets, last_dets_t = dets, now
                    saved_dets = [d for d in dets if d.class_name in cfg.save_classes]
                    # Clips trigger on clip_trigger (default = save_classes), independent of what
                    # gets cropped -- so a person can record a test clip without a DB selfie.
                    if recorder is not None:
                        clip_dets = [d for d in dets if d.class_name in clip_trigger]
                        if clip_dets:
                            recorder.note_detection(now, clip_dets)
                    dt = datetime.now().astimezone()
                    iso = dt.isoformat()
                    day = dt.strftime("%Y-%m-%d")
                    stamp = dt.strftime("%Y-%m-%dT%H-%M-%S-") + f"{dt.microsecond // 1000:03d}"
                    h, w = frame.shape[:2]
                    frame_path = None
                    if cfg.save_full_frame:
                        fp = save_frame(frame, cfg, day, stamp)
                        frame_path = _rel(fp) if fp else None

                    for i, det in enumerate(saved_dets):
                        crop = save_crop(frame, det, cfg, day, stamp, i)
                        if crop is None:
                            continue
                        db.insert_detection(
                            conn,
                            timestamp=iso,
                            source=cfg.source,
                            detection_class=det.class_name,
                            confidence=det.confidence,
                            bbox=det.bbox,
                            frame_w=w,
                            frame_h=h,
                            crop_path=_rel(crop),
                            frame_path=frame_path,
                            # species / individual_id stay NULL in V1.
                        )
                        total_saved += 1
                        print(f"  [{iso}] {det.class_name} {det.confidence:.2f} -> {_rel(crop)}")

            # --- Live preview + optional web stream ---
            if cfg.show_preview or server is not None:
                disp = frame.copy()
                show = last_dets if (now - last_dets_t) <= cfg.box_display_ttl_s else []
                draw_detections(disp, show)
                draw_hud(disp, fps=fps, motion_area=motion_area, motion=motion,
                         saved=total_saved, source=cfg.source, model=cfg.model_version,
                         period=active_period,
                         recording=recorder is not None and recorder.recording)
                if server is not None:
                    ok_enc, buf = cv2.imencode(".jpg", disp,
                                               [cv2.IMWRITE_JPEG_QUALITY, cfg.web_jpeg_quality])
                    if ok_enc:
                        frame_buffer.update(buf.tobytes())
                if cfg.show_preview:
                    cv2.imshow(cfg.window_name, disp)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        print("\n'q' pressed -- shutting down.")
                        break
                    # Closing the video window (its X button) shuts the rig down too, so a teen
                    # can just close the window instead of remembering the 'q' key.
                    if cv2.getWindowProperty(cfg.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("\nVideo window closed -- shutting down.")
                        break
    except KeyboardInterrupt:
        print("\nInterrupted -- shutting down.")
    finally:
        if recorder is not None:
            recorder.finalize()                     # flush any clip that was mid-record (writes its DB row)
        _stop_naming(classify_proc, classify_tag)   # kill the helper + any venv-launcher subproc
        if server is not None:
            web.shutdown(server)
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        # Keep the visit ledger fresh: collapse this session's detections into visit events so
        # the dashboard's Behaviour tab is current without a manual `python visits.py` run.
        # A full rebuild is subsecond at this scale; never let it block a clean shutdown.
        try:
            conn.row_factory = sqlite3.Row
            visits.build_visits(conn, cfg.visit_gap_minutes, verbose=False)
            print("  visit ledger refreshed.")
        except Exception as e:
            print(f"  [visits] could not refresh visit events (run `python visits.py`): {e}")
        conn.close()
        clips_note = f"  Recorded {recorder.clips_saved} clip(s)." if recorder is not None else ""
        print(f"Done. Saved {total_saved} detection(s) this session to {cfg.db_path}.{clips_note}")


# ---- CLI ---------------------------------------------------------------------------
def parse_args() -> tuple[config.Config, argparse.Namespace]:
    c = CONFIG
    p = argparse.ArgumentParser(
        description="Backyard critter detection -- V1 live glass-door cam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera-index", type=int, default=c.camera_index)
    p.add_argument("--width", type=int, default=c.frame_width)
    p.add_argument("--height", type=int, default=c.frame_height)
    p.add_argument("--exposure", type=float, default=c.exposure,
                   help="Lock manual exposure (find a value with tune.py). Omit to auto-expose.")
    p.add_argument("--gain", type=float, default=c.gain, help="Lock manual gain. Omit for auto.")
    p.add_argument("--model-version", default=c.model_version,
                   help="MDV6-yolov10-c | MDV6-yolov9-c | MDV6-rtdetr-c | MDV6-yolov10-e | MDV6-yolov9-e")
    p.add_argument("--device", default=c.device, choices=["cuda", "cpu", "auto"],
                   help="Inference device: cuda (default, requires an NVIDIA GPU) | cpu (no GPU "
                        "needed, slower) | auto (GPU if it genuinely runs, else CPU).")
    p.add_argument("--min-confidence", type=float, default=c.min_confidence)
    p.add_argument("--motion-min-area", type=int, default=c.motion_min_area,
                   help="Largest motion blob (px) needed to wake the detector.")
    p.add_argument("--detector-interval", type=float, default=c.detector_min_interval_s,
                   help="Min seconds between detector runs while motion continues.")
    p.add_argument("--db", default=str(c.db_path))
    p.add_argument("--crops-dir", default=str(c.crops_dir))
    p.add_argument("--frames-dir", default=str(c.frames_dir))
    p.add_argument("--save-full-frame", action="store_true", default=c.save_full_frame,
                   help="Also save the full frame for each detection event (default off).")
    p.add_argument("--record-clips", action="store_true", default=c.record_clips,
                   help="Record a short video clip around each visit (phase-4 behaviour capture; "
                        "default off). Crops are still saved alongside.")
    p.add_argument("--clips-dir", default=str(c.clips_dir),
                   help="Where behaviour clips are written (with --record-clips).")
    p.add_argument("--clip-classes", nargs="+", default=None,
                   metavar="CLASS", choices=["animal", "person", "vehicle"],
                   help="Detector classes that trigger a clip (default: same as saved = animal). "
                        "e.g. --clip-classes animal person to also record yourself as a test.")
    p.add_argument("--no-preview", action="store_true",
                   help="Run headless (no window). Quit with Ctrl+C.")
    p.add_argument("--no-classify", dest="classify_live", action="store_false",
                   default=c.classify_live,
                   help="Detection only -- don't name species live in this process. "
                        "(You can still fill species in later with `python classify.py`.)")
    p.add_argument("--list-cameras", action="store_true",
                   help="Probe camera indices and exit.")
    p.add_argument("--stats", action="store_true",
                   help="Print a summary of the database (crops, visits, activity) and exit.")
    p.add_argument("--visit-gap-min", type=float, default=c.visit_gap_minutes,
                   help="Minutes between detections that separates one visit from the next (--stats).")
    p.add_argument("--serve", action="store_true", default=c.serve,
                   help="Also serve a local web dashboard (live stream + stats) in the browser.")
    p.add_argument("--port", type=int, default=c.web_port, help="Web dashboard port (with --serve).")
    p.add_argument("--host", default=c.web_host,
                   help="Web dashboard bind host (default 127.0.0.1; 0.0.0.0 exposes it on the LAN).")
    args = p.parse_args()

    cfg = replace(
        c,
        camera_index=args.camera_index,
        frame_width=args.width,
        frame_height=args.height,
        exposure=args.exposure,
        gain=args.gain,
        model_version=args.model_version,
        device=args.device,
        min_confidence=args.min_confidence,
        motion_min_area=args.motion_min_area,
        detector_min_interval_s=args.detector_interval,
        db_path=Path(args.db),
        crops_dir=Path(args.crops_dir),
        frames_dir=Path(args.frames_dir),
        save_full_frame=args.save_full_frame,
        record_clips=args.record_clips,
        clips_dir=Path(args.clips_dir),
        clip_classes=tuple(args.clip_classes) if args.clip_classes else c.clip_classes,
        show_preview=not args.no_preview,
        visit_gap_minutes=args.visit_gap_min,
        serve=args.serve,
        web_host=args.host,
        web_port=args.port,
        classify_live=args.classify_live,
    )
    return cfg, args


def main() -> int:
    cfg, args = parse_args()
    if args.list_cameras:
        list_cameras(cfg)
        return 0
    if args.stats:
        print_stats(cfg)
        return 0
    try:
        run(cfg)
    except CudaUnavailableError as e:
        print(f"\n[CUDA ERROR]\n{e}")
        return 2
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
