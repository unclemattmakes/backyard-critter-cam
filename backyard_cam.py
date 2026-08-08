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
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import numpy as np

import clips
import config
import daynight
import db
import powerguard
import quality
import stats
import visits
import web
from config import CONFIG
from detector import CudaUnavailableError, Detection, Detector

# The reference-image veto (refimg.py) is OPTIONAL and ships disabled, so a problem importing it
# must cost the shadow feature and nothing else -- capturing animals is the job, and this module
# is the newest code in the tree. The failure is reported once, where it is actionable
# (RefimgShadow.create), not swallowed.
try:
    import refimg
    _REFIMG_IMPORT_ERROR = None
except Exception as _refimg_exc:          # pragma: no cover -- deps are shared with the rig itself
    refimg, _REFIMG_IMPORT_ERROR = None, _refimg_exc

# ---- File logging (so the next post-mortem isn't blind) ----------------------------
# The overnight deaths left no trace: the start-ed console window vanished with the process,
# so there was nothing to read the morning after. We now TEE every line of stdout/stderr to a
# rotating, line-timestamped log file as well as the console -- the live window UX is
# unchanged (everything still prints there), but a death now leaves a dated breadcrumb trail.
# Crash-safe: each line is flushed immediately, so even a hard kill keeps what was printed.
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "backyard_cam.log"


class _Tee:
    """A stdout/stderr stand-in that writes to the real stream AND a logging handler, stamping
    each line with a timestamp in the file copy. Mirrors the file object API just enough for
    print() and tracebacks (write/flush/isatty/fileno + the encoding attr libraries probe).

    The file sink is a safety net, never load-bearing: if the console can't encode a line
    (cp1252 console + a non-cp1252 species label) we degrade instead of dropping it, and if the
    log handler itself starts failing (e.g. disk full) we disable it so logging.handleError
    can't recurse back through this tee and crash the capture app."""

    def __init__(self, real_stream, file_logger):
        self._real = real_stream
        self._log = file_logger
        self._buf = ""                              # accumulate until a newline -> one log record

    def write(self, text: str) -> int:
        # Console first, unchanged (UX preserved). On a cp1252 console a non-cp1252 char raises
        # UnicodeEncodeError -- don't let the live line silently vanish; re-encode with a visible
        # escape so the window still shows something (the file copy keeps the real text).
        try:
            n = self._real.write(text)
            self._real.flush()
        except UnicodeEncodeError:
            try:
                enc = getattr(self._real, "encoding", "utf-8") or "utf-8"
                self._real.buffer.write(text.encode(enc, "backslashreplace"))
                self._real.flush()
            except Exception:
                pass
            n = len(text)
        except Exception:
            n = len(text)
        # Buffer and split on newlines so each console line becomes one timestamped file line.
        self._buf += text
        try:
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._log.info(line)                # the handler timestamps + flushes per line
        except Exception:
            # The file sink failed mid-run (e.g. disk full). Stop feeding it so the failing
            # handler can't re-enter this tee via stderr and recurse the app to death; the
            # console (the real UX) stays alive.
            self._log = _NullLog()
        return n

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception:
            pass

    # Pass-throughs so anything that introspects the stream (cv2, subprocess, faulthandler,
    # isatty checks) still sees a real terminal underneath rather than choking on a bare object.
    def isatty(self) -> bool:
        return getattr(self._real, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._real.fileno()

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")


class _NullLog:
    """A do-nothing stand-in we swap in if the real file sink starts failing, so the tee keeps
    forwarding to the console but never re-enters a broken logging handler."""
    def info(self, *args, **kwargs) -> None:
        pass


def install_file_logging() -> Path | None:
    """Tee stdout + stderr to a rotating, line-timestamped file under logs/. Call once, first
    thing in main(). Returns the log path, or None if it couldn't be set up (which is itself
    printed -- never silent). Idempotent-safe: re-wrapping is harmless but we only call it once."""
    try:
        # Don't let a failing handler raise through logging itself -- our tee already isolates the
        # sink, and we never want a log error to surface as an app crash.
        logging.raiseExceptions = False
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # 5 files x 2 MB ~ a couple of weeks of this rig's chatter; old ones roll off automatically.
        handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                               datefmt="%Y-%m-%dT%H:%M:%S"))
        file_logger = logging.getLogger("backyard_cam.file")
        file_logger.setLevel(logging.INFO)
        file_logger.propagate = False               # our own sink; don't touch the root logger
        file_logger.handlers = [handler]
        sys.stdout = _Tee(sys.stdout, file_logger)
        sys.stderr = _Tee(sys.stderr, file_logger)
        return LOG_FILE
    except Exception as e:
        # Logging is a safety net, never a hard dependency -- if it can't open, say so and run on.
        print(f"  log: WARNING -- could not open the log file ({e}); running without a file log.")
        return None


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


# ---- Keep the machine awake while we watch -----------------------------------------
# Some rigs (e.g. this Acer handheld) only support Modern Standby (S0 low-power idle), not
# S3. Left idle, the box slides into standby within minutes -- which USB-suspends the
# webcam. The capture loop then sees the camera "vanish", and the app silently dies
# overnight. While a run is live we hold a power request so Windows won't idle-sleep
# underneath us.
#
# We learned the hard way that SetThreadExecutionState(ES_SYSTEM_REQUIRED) is NOT
# Modern-Standby-aware: the box still slid into S0 standby (Kernel-Power 506) WHILE we were
# actively capturing. The Modern-Standby-aware mechanism is the Power Request API
# (PowerCreateRequest / PowerSetRequest) with PowerRequestExecutionRequired -- that's the
# request type that actually keeps an S0 box out of the deep DRIPS sub-states that suspend
# USB. So we hold that as the primary, and keep SetThreadExecutionState as a
# belt-and-suspenders secondary (harmless, and helps on classic-S3 machines).
#
# Windows-only; a clean no-op everywhere else. CRITICAL: unlike the old version, a failure
# here is now LOGGED loudly -- a silent failure is exactly what hid the overnight deaths.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001

# POWER_REQUEST_TYPE enum (winnt.h): the value we want is PowerRequestExecutionRequired.
_PowerRequestSystemRequired = 1
_PowerRequestExecutionRequired = 3
# REASON_CONTEXT.Flags: a plain human-readable reason string (vs. a module+resource id).
_POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1

# The live power-request handle, held for the lifetime of a run. Two requests are set on it
# (Execution + System), so allow_system_sleep() clears both before closing it.
_power_request_handle = None


def keep_system_awake() -> bool:
    """Hold a Modern-Standby-aware power request so Windows won't idle-sleep (S0/S3) until
    allow_system_sleep(). Stores the request handle in module state so the release can clear
    it. Returns True if at least the Execution request was set; False on non-Windows or if
    the call failed (and logs WHY -- a silent failure is what hid the overnight deaths)."""
    global _power_request_handle
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32

        # REASON_CONTEXT for the simple-string case: Version (ULONG, 0), Flags (DWORD), and a
        # union whose simple-string arm is an LPWSTR reason. We only ever use the string arm,
        # so model the struct with just that member (the detailed-info arm is larger, but we
        # never set Flags to use it, so Windows never reads past our LPWSTR).
        class _REASON_CONTEXT(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("Flags", wintypes.DWORD),
                ("SimpleReasonString", wintypes.LPWSTR),
            ]

        # CRITICAL: PowerCreateRequest returns a HANDLE. Without restype=HANDLE ctypes assumes
        # c_int and truncates the pointer to 32 bits on 64-bit Python -- the handle then looks
        # valid but PowerSetRequest/CloseHandle silently operate on garbage. Pin it explicitly.
        kernel32.PowerCreateRequest.restype = wintypes.HANDLE
        kernel32.PowerCreateRequest.argtypes = [ctypes.POINTER(_REASON_CONTEXT)]
        kernel32.PowerSetRequest.restype = wintypes.BOOL
        kernel32.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
        # CloseHandle releases the request on the failure path below; pin it too so a large
        # handle value can't raise OverflowError against ctypes' default c_int argument.
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        ctx = _REASON_CONTEXT(
            Version=0,
            Flags=_POWER_REQUEST_CONTEXT_SIMPLE_STRING,
            SimpleReasonString="Backyard Critter Cam is capturing (keep the USB camera awake).",
        )
        handle = kernel32.PowerCreateRequest(ctypes.byref(ctx))
        # INVALID_HANDLE_VALUE (-1) or NULL means the create failed -- read GetLastError() live
        # (windll isn't use_last_error=True, so call the API directly rather than ctypes' cache).
        if not handle or handle == wintypes.HANDLE(-1).value:
            err = kernel32.GetLastError()
            print(f"  power: WARNING -- PowerCreateRequest failed (err {err}); cannot hold a "
                  "power request. The box may idle into Modern Standby and suspend the camera.")
            # Fall through to the SetThreadExecutionState belt-and-suspenders below anyway.
            handle = None

        ok_exec = False
        if handle is not None:
            # PowerRequestExecutionRequired is the Modern-Standby-aware one (keeps an S0 box out
            # of the deep DRIPS states that USB-suspend the cam); PowerRequestSystemRequired
            # covers classic S3 idle-sleep. Set both on the one handle.
            ok_exec = bool(kernel32.PowerSetRequest(handle, _PowerRequestExecutionRequired))
            ok_sys = bool(kernel32.PowerSetRequest(handle, _PowerRequestSystemRequired))
            if not ok_exec:
                err = kernel32.GetLastError()
                print(f"  power: WARNING -- PowerSetRequest(Execution) failed (err {err}); the "
                      "Modern-Standby keep-awake is NOT in effect. The cam may suspend overnight.")
            elif not ok_sys:
                # Execution is the one that matters for this rig; note the S3 miss but don't alarm.
                print("  power: note -- the system-required (S3) request didn't set, but the "
                      "Modern-Standby (execution) request is holding.")
            _power_request_handle = handle if ok_exec else None
            if not ok_exec:
                kernel32.CloseHandle(handle)        # nothing to hold; don't leak the handle

        # Belt-and-suspenders: also assert the legacy execution-state flag. It alone did NOT hold
        # this Modern-Standby box (that's the whole bug), but it's free and helps on S3 machines.
        try:
            kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
        except Exception:
            pass

        return ok_exec
    except Exception as e:
        print(f"  power: WARNING -- could not set a power request ({e}); if the box sleeps it "
              "may suspend the camera and stop the app.")
        return False


def allow_system_sleep() -> None:
    """Release the keep-awake request from keep_system_awake() (restore normal idle/sleep):
    clear both power requests, close the handle, and drop the legacy execution-state flag."""
    global _power_request_handle
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        handle = _power_request_handle
        _power_request_handle = None
        if handle is not None:
            kernel32.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
            kernel32.CloseHandle.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.PowerClearRequest(handle, _PowerRequestExecutionRequired)
            kernel32.PowerClearRequest(handle, _PowerRequestSystemRequired)
            kernel32.CloseHandle(handle)
        kernel32.SetThreadExecutionState(_ES_CONTINUOUS)   # drop the belt-and-suspenders flag
    except Exception:
        pass


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


class CommandedAutoState:
    """The three auto-mode toggles as the rig LAST COMMANDED them -- published to the dashboard
    in place of cap.get(), which is not a truth source for these on this hardware: the glass-door
    cam's driver answers -1.0 for get(AUTOFOCUS)/get(AUTO_EXPOSURE) and 0.0 for get(AUTO_WB)
    whatever the real mode is (measured 2026-08-07: both panel boxes showed manual minutes after
    open_capture asserted auto and with the frame's R/G ratio proving auto-WB was live), and the
    2026-06 cam lied the other way (AUTOFOCUS read 1.0 forever after we set 0). The rig is the
    ONLY writer of these modes -- the driver never flips them on its own -- so the last write IS
    the state. Seeded from open_capture's all-auto baseline resolved against the config locks;
    note() every settings dict that reaches apply_camera_settings (profiles, dashboard posts)
    and the WB watchdog's recoveries, and it stays the state's single honest ledger."""

    def __init__(self, spec: config.CameraSpec, cfg: config.Config):
        exposure = _eff(spec, cfg, "exposure")
        self.state = {"autofocus": 1.0, "auto_wb": 1.0,
                      "auto_exposure": 0.75 if exposure is None else 0.25}
        self.note(_eff(spec, cfg, "camera_controls") or {})

    def note(self, settings: dict) -> None:
        """Record the auto-mode consequences of a settings dict bound for apply_camera_settings."""
        if not settings:
            return
        if "exposure" in settings:      # the profile/dashboard idiom: None = auto, number = lock
            self.state["auto_exposure"] = 0.75 if settings["exposure"] is None else 0.25
        for prop, ours in (("AUTO_EXPOSURE", "auto_exposure"), ("AUTOFOCUS", "autofocus"),
                           ("AUTO_WB", "auto_wb")):
            if settings.get(prop) is not None:
                self.state[ours] = float(settings[prop])


class WhiteBalanceWatchdog:
    """Put a camera back into AUTO white balance when its colour goes red-starved.

    Why this exists: the 2026-07 glass-door cam's MANUAL white balance is broken. Measured
    2026-07-25 by sweeping the live rig, every WB_TEMPERATURE from 2800 K to 6500 K rendered red
    30-66% below green (R/G 0.34-0.70) -- no setting on that axis reaches neutral, because colour
    TEMPERATURE only moves the blue<->amber axis and the fault is on green<->magenta. Its AUTO
    white balance is correct (R/G ~0.98) and recovers ~2 s after AUTO_WB is re-asserted. Since
    the dashboard's WB slider posts {AUTO_WB: 0, WB_TEMPERATURE: v}, one nudge leaves the rig
    cyan-cast for the rest of the session -- and most DAYLIGHT hours of 2026-07-21..25 were in
    that state (R/G 0.57-0.72 against the old camera's 0.96-1.10).

    Scope: this only judges frames bright enough to carry colour (wb_recover_min_luma). The
    camera's NIGHT rendering is a separate open question -- it is strongly green (R/G ~0.85,
    B/G ~0.57) where the old camera under the same yard light was amber (R/G ~2.2) -- but with
    no known-good night reference for this sensor there's nothing to recover TO, so nights are
    left alone rather than "corrected" toward a guess.

    The check is deliberately cheap and deliberately blunt: a strided channel mean every
    `wb_recover_interval_s`, and re-asserting auto when the camera is already in auto costs
    nothing, so being wrong is free. See config.wb_auto_recover for the knobs."""

    def __init__(self, cfg: config.Config, tag: str = ""):
        self.cfg, self.tag = cfg, tag
        self.enabled = bool(cfg.wb_auto_recover)
        self.user_manual = False      # the dashboard asked for manual WB -- stand down
        self._next_check = 0.0
        self._strikes = 0
        self.recoveries = 0            # how many times we've pulled it back this session
        self.last_ratio: float | None = None
        # Wedge tell (2026-07-30): a real manual-WB trap recovers ~2 s after AUTO_WB is
        # re-asserted, so the NEXT judged sample (>= interval later) being bad again means the
        # re-assert didn't take -- that's the USB-wedge signature, not the WB trap. We verify
        # each recovery on its next bright-enough sample and count the consecutive failures;
        # powerguard.WedgeDetector consumes the streak.
        self._verify_pending = False
        self.failed_streak = 0

    def note_settings(self, settings: dict) -> None:
        """Watch dashboard-applied settings for a deliberate manual/auto white-balance choice."""
        if "AUTO_WB" not in (settings or {}):
            return
        val = settings["AUTO_WB"]
        self.user_manual = (val is not None and float(val) < 0.5)
        if self.user_manual:
            self._strikes = 0

    @staticmethod
    def _sample(frame) -> tuple[float, float]:
        """(red/green ratio, luma-ish mean) from a strided sample -- ~1/64th of the pixels."""
        small = frame[::8, ::8]
        b = float(small[:, :, 0].mean())
        g = float(small[:, :, 1].mean())
        r = float(small[:, :, 2].mean())
        return r / max(g, 1e-6), (r + g + b) / 3.0

    def check(self, cap, frame, now: float) -> bool:
        """Sample the frame and, if it's persistently red-starved, restore auto white balance.
        Returns True when it acted. Cheap enough to call every frame -- it self-rate-limits."""
        if not self.enabled or self.user_manual or frame is None:
            return False
        if now < self._next_check:
            return False
        self._next_check = now + self.cfg.wb_recover_interval_s
        try:
            ratio, luma = self._sample(frame)
        except Exception:
            return False
        self.last_ratio = ratio
        if luma < self.cfg.wb_recover_min_luma:
            self._strikes = 0            # too dark to judge colour; a warm night is not a fault
            return False
        # Verify the previous recovery on its first bright-enough follow-up sample: auto WB
        # takes ~2 s, this sample is >= wb_recover_interval_s later, so "still bad" means the
        # re-assert didn't take (the wedge), not that it hasn't settled yet.
        if self._verify_pending:
            self._verify_pending = False
            if ratio < self.cfg.wb_recover_ratio:
                self.failed_streak += 1
                print(f"{self.tag} white balance did NOT recover after the AUTO_WB re-assert "
                      f"(R/G {ratio:.2f}; failed recovery x{self.failed_streak}) -- the "
                      "USB-wedge signature, not the manual-WB trap.")
            else:
                self.failed_streak = 0
        if ratio >= self.cfg.wb_recover_ratio:
            self._strikes = 0
            self.failed_streak = 0
            return False
        self._strikes += 1
        if self._strikes < self.cfg.wb_recover_strikes:
            return False
        self._strikes = 0
        try:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
        except Exception:
            return False
        self.recoveries += 1
        self._verify_pending = True
        print(f"{self.tag} white balance looked red-starved (R/G {ratio:.2f} < "
              f"{self.cfg.wb_recover_ratio:.2f}) -- restored AUTO_WB "
              f"(recovery #{self.recoveries}).")
        return True


def capture_backend(cfg: config.Config) -> int:
    """Pick the OpenCV capture backend. DirectShow is a Windows-only API, so we only use it on
    Windows (and only when enabled); on Linux/macOS we pass CAP_ANY and let OpenCV choose the
    native backend (V4L2 / AVFoundation). This is what lets the same code run on the Linux box."""
    if cfg.use_dshow_backend and sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _eff(spec: config.CameraSpec, cfg: config.Config, field: str):
    """A CameraSpec field's effective value: the per-camera override if set, else the Config
    default. This is what lets a spec stay terse -- only the fields that differ are filled in."""
    v = getattr(spec, field, None)
    return v if v is not None else getattr(cfg, field)


def _backend_for(spec: config.CameraSpec, cfg: config.Config) -> int:
    """OpenCV capture backend for a camera. A stream URL -> FFMPEG (what reads rtsp://, http MJPEG).
    A local index -> DirectShow on Windows (fast init) else the native backend. An explicit
    spec.backend wins; 'dshow' off Windows degrades to the native backend (as capture_backend does)."""
    b = spec.backend
    if b is None:
        b = "ffmpeg" if spec.is_url else ("dshow" if cfg.use_dshow_backend else "any")
    if b == "ffmpeg":
        return cv2.CAP_FFMPEG
    if b == "dshow" and sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def open_capture(spec: config.CameraSpec, cfg: config.Config) -> cv2.VideoCapture | None:
    """Open ONE camera from its CameraSpec, warm it up, and return the VideoCapture (or None if it
    won't open). Handles a local webcam index (DirectShow on Windows) and a networked stream URL
    (FFMPEG backend + a 1-frame buffer, so decoded frames can't pile up into seconds of lag while
    the detector is busy). Exposure/gain/UVC tweaks are applied for local cams; on a network stream
    they're harmless no-ops (the camera, not OpenCV, owns those)."""
    cap = cv2.VideoCapture(spec.src, _backend_for(spec, cfg))
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None
    if spec.is_network:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep only the freshest frame (kills RTSP lag)
        except Exception:
            pass
    def _ask_fourcc():
        if spec.is_network or not cfg.camera_fourcc:
            return
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.camera_fourcc))
        except Exception:
            pass

    # Ask for the compressed format BOTH before and after size/rate: each size/fps set can
    # rebuild the DirectShow media type and silently revert the format to uncompressed YUY2.
    # Measured on the 2026-07 glass-door cam: fourcc-then-size ends up YUY2 @ 4 fps; re-asserting
    # MJPG after size+fps holds MJPG @ 30 fps at 1920x1080 (its only full-speed mode).
    _ask_fourcc()
    w, h = _eff(spec, cfg, "frame_width"), _eff(spec, cfg, "frame_height")
    if w:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    if h:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not spec.is_network and cfg.camera_request_fps:
        cap.set(cv2.CAP_PROP_FPS, float(cfg.camera_request_fps))
    _ask_fourcc()
    # UVC webcams PERSIST control state in hardware across sessions: one manual nudge -- the
    # dashboard's Focus/WB slider (which posts {AUTOFOCUS: 0} / {AUTO_WB: 0} by design), a
    # tune.py experiment, any other app -- and every LATER session silently starts in manual
    # focus / manual white balance. On this cam manual WB is the broken red-starved state the
    # WhiteBalanceWatchdog exists for, and manual focus has no watchdog at all. So every open
    # starts from the known baseline: everything auto. A deliberate lock still wins -- exposure/
    # gain and camera_controls are applied AFTER this, so camera_controls={"AUTOFOCUS": 0}
    # keeps meaning what it says. On a network stream these are the usual harmless no-ops.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
    exposure, gain = _eff(spec, cfg, "exposure"), _eff(spec, cfg, "gain")
    if exposure is None:       # assert auto-expose (don't inherit a persisted manual lock)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # 0.75 = auto on most UVC/DSHOW cams
    else:                      # lock manual exposure (0.25 = manual on most UVC/DSHOW cams)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, gain)
    for _name, _val in (_eff(spec, cfg, "camera_controls") or {}).items():
        _prop = getattr(cv2, f"CAP_PROP_{_name}", None)
        if _prop is not None:
            cap.set(_prop, _val)
    for _ in range(cfg.camera_warmup_frames):  # let exposure / auto-WB settle
        cap.read()
    # Re-assert the resolved auto modes now that frames are FLOWING. Measured 2026-08-07: this
    # driver drops auto-mode sets issued before the capture graph is running -- the panel's
    # mid-stream {AUTOFOCUS: 1}/{AUTO_WB: 1} visibly re-focused and re-balanced the image
    # minutes after open had "asserted" the same thing -- while value props from the same
    # pre-stream phase (CONTRAST 48) demonstrably stick. Same values as above, config locks
    # included, so a deliberate manual lock still wins; no-ops on drivers that already obeyed.
    _ctl = _eff(spec, cfg, "camera_controls") or {}
    cap.set(cv2.CAP_PROP_AUTOFOCUS, float(_ctl.get("AUTOFOCUS", 1.0)))
    cap.set(cv2.CAP_PROP_AUTO_WB, float(_ctl.get("AUTO_WB", 1.0)))
    if exposure is None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(_ctl.get("AUTO_EXPOSURE", 0.75)))
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return cap


def open_camera(cfg: config.Config) -> cv2.VideoCapture | None:
    """Back-compat shim: open the single (or primary) camera from the legacy flat config fields."""
    return open_capture(cfg.camera_specs()[0], cfg)


def reconnect_capture(spec: config.CameraSpec, cfg: config.Config,
                      stop_event: threading.Event) -> cv2.VideoCapture | None:
    """Re-open a camera after a read failure. NEITHER kind ever gives up: it keeps retrying
    until the camera comes back or we're told to stop. A NETWORK camera (the AP, switch, or
    camera itself rebooted) comes back; a LOCAL USB camera does too -- and crucially, on this
    Modern-Standby rig a USB "unplug" is usually the box idle-suspending the cam overnight, so
    when it wakes the device re-enumerates and we recover. The OLD behaviour (local cams gave up
    after cfg.reopen_max_retries) is exactly what killed the app overnight; we don't do that any
    more. Both kinds get a fast initial burst (cfg.reopen_max_retries attempts at
    cfg.reopen_delay_s, for brief USB hiccups) then slow exponential backoff, capped, FOREVER.
    Honours stop_event throughout -- it uses stop_event.wait() (not sleep) so Ctrl+C / 'q' / the
    finally-join end the wait instantly and never hang. Returns a working capture, or None only
    when we're shutting down."""
    tag = f"[{spec.source}]"
    print(f"{tag} read failed -- reconnecting ...")
    attempt, delay = 0, cfg.reopen_delay_s
    while not stop_event.is_set():
        attempt += 1
        cap = open_capture(spec, cfg)
        if cap is not None and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"{tag} reconnected on attempt {attempt}.")
                return cap
            cap.release()
        # Fast burst first (brief USB hiccup / network blip), then back off so we're not hammering
        # a device that's genuinely away (e.g. the box is in standby) -- but keep trying forever.
        if attempt < cfg.reopen_max_retries:
            print(f"{tag} reopen attempt {attempt} failed; retrying in {cfg.reopen_delay_s:.0f}s.")
            stop_event.wait(cfg.reopen_delay_s)
        else:
            print(f"{tag} reopen attempt {attempt} failed; the device looks away -- backing off, "
                  f"retrying in {delay:.0f}s (will keep trying until it returns).")
            stop_event.wait(delay)
            delay = min(30.0, delay * 2)        # exponential backoff, capped at 30 s
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
    """MOG2 background subtractor -> 'largest motion blob area' in FULL-FRAME pixels.

    MOG2 adapts to gradual outdoor light changes; we drop its shadow pixels (marked 127)
    and de-noise with a blur + morphology so a single flickering pixel never triggers.
    The whole chain runs on a copy downscaled to ~cfg.motion_gate_width (its cost scales
    with pixel count -- at 1080p a full-res gate alone held the loop to ~6 fps), and blob
    areas are scaled BACK to full-frame px^2 so motion_min_area / the HUD readout keep
    their familiar units at any capture size.

    The cleaned mask itself is KEPT on `self.mask` (post-threshold, post-morphology,
    pre-contour) rather than thrown away with the rest of the chain. It costs nothing -- the
    array already exists -- and it is the detector-independent evidence policy E needs: every
    pixel that has blobbed recently is marked NOT COVERED in the reference image, so a raccoon
    the DETECTOR missed (measured: a properly certified reference with an undetected raccoon
    walking the wall in it) costs an abstention instead of an erasure. See refimg.py.
    `mask` is at the gate's own downscaled resolution, aspect preserved, so a consumer just
    resizes it to whatever working resolution it uses; the returned AREA is unchanged.
    """

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.motion_history,
            varThreshold=cfg.motion_var_threshold,
            detectShadows=cfg.motion_detect_shadows,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Last cleaned foreground mask (None until the first update()). Read-only for callers.
        self.mask = None

    def update(self, frame_bgr) -> float:
        src = frame_bgr
        area_scale = 1.0   # multiplier back to full-frame px^2
        gw = self.cfg.motion_gate_width
        if gw and frame_bgr.shape[1] > gw:
            s = gw / frame_bgr.shape[1]
            src = cv2.resize(frame_bgr, (gw, max(1, round(frame_bgr.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
            area_scale = 1.0 / (s * s)
        k = self.cfg.motion_blur_ksize
        if k and k > 1:
            k = k | 1  # GaussianBlur needs an odd kernel
            src = cv2.GaussianBlur(src, (k, k), 0)
        mask = self.bg.apply(src)
        # Keep only strong foreground (255); shadows come back as 127 and are discarded.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        self.mask = mask                    # retained for refimg's coverage channel (see docstring)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        return max(cv2.contourArea(c) for c in contours) * area_scale


# ---- Ignore zones (static false-fire spots) -----------------------------------------
def box_iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes; 0.0 for disjoint or degenerate boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def drop_ignored(dets, zones, min_iou: float):
    """Split detections into (kept, dropped) against this camera's ignore zones (the
    dashboard-editable IgnoreZoneStore; config.ignore_zones seeds it).

    A detection is dropped only when its box mostly IS a zone (IoU >= min_iou) -- the
    static-false-fire signature. The IoU gate keeps this surgical: a real animal walking
    THROUGH a zone carries a much bigger box, so its IoU with the zone stays tiny and it is
    kept (measured on the wall-opening zone: the false-fire scores ~0.55, a raccoon-sized
    box over the same spot ~0.09)."""
    if not zones:
        return list(dets), []
    kept, dropped = [], []
    for d in dets:
        (dropped if any(box_iou(d.bbox, z) >= min_iou for z in zones) else kept).append(d)
    return kept, dropped


# ---- Detector census (measurement, always on) ---------------------------------------
class DetectorCensus:
    """How often does the detector run, and how often does it come back EMPTY?

    This is design section 8, item 1 of docs/refimg-design-2026-08-07.md -- named there as the
    biggest unknown in the whole reference-image design. A reference image can only be built from
    a frame the detector RAN on and certified empty, and the offline estimate of how often such a
    frame exists (6.5%) is a floor derived from clip rows, not a measurement: the DB structurally
    cannot record a quiet stretch, because a quiet stretch writes nothing. The live rig is the
    only place the real rate exists, so the loop counts it directly.

    Deliberately NOT gated on cfg.refimg_enabled: this is measurement, not behaviour. It writes
    no rows, touches no detection and changes no decision -- one log line per hour through the
    existing print path (which the file tee already timestamps and rotates).

    `longest_empty_s` is the longest UNBROKEN wall-clock stretch of detector verdicts that all
    came back empty, which is the number that says whether a `refimg_certify_hold_s` window is
    ever available. It is measured on the monotonic clock (a system clock step must not invent a
    six-hour empty run) and is capped at the reporting period, so "longest empty run" always
    means "within this hour".
    """

    def __init__(self, period_s: float = 3600.0):
        self.period_s = float(period_s)
        self.runs = 0                    # detector invocations this period
        self.empty = 0                   # ... of which returned zero boxes
        self.longest_empty_s = 0.0       # longest unbroken empty stretch this period
        self._empty_since: float | None = None
        self._start: float | None = None

    def note(self, now: float, n_boxes: int) -> None:
        """Record one detector invocation. `n_boxes` is the RAW detector output count."""
        if self._start is None:
            self._start = now
        self.runs += 1
        if n_boxes:
            self._empty_since = None
            return
        self.empty += 1
        if self._empty_since is None:
            self._empty_since = now
        else:
            self.longest_empty_s = max(self.longest_empty_s, now - self._empty_since)

    def line(self, now: float) -> str:
        span_min = (now - (self._start if self._start is not None else now)) / 60.0
        pct = (100.0 * self.empty / self.runs) if self.runs else 0.0
        return (f"detector census: {self.runs} run(s), {self.empty} empty ({pct:.0f}%), "
                f"longest empty run {self.longest_empty_s:.0f}s, over {span_min:.0f} min")

    def roll(self, now: float) -> str | None:
        """Returns the hourly line and resets, or None if the period isn't up. Cheap enough to
        call every frame -- a zero-run hour is itself the finding, so silence must not stand in
        for it."""
        if self._start is None:
            self._start = now
            return None
        if now - self._start < self.period_s:
            return None
        text = self.line(now)
        self._start = now
        self.runs = self.empty = 0
        self.longest_empty_s = 0.0
        if self._empty_since is not None:
            self._empty_since = now      # a run spanning the boundary re-measures from here
        return text


# ---- Reference-image veto, SHADOW MODE (refimg.py) -----------------------------------
def _short_age(seconds: float) -> str:
    """A compact age for the HUD: 42s / 7m / 3.1h."""
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s // 60)}m"
    return f"{s / 3600.0:.1f}h"


class RefimgShadow:
    """One camera's wiring into refimg.py's reference-image veto -- and NOTHING ELSE CHANGES.

    SHADOW MODE IS THE WHOLE POINT. When the veto says a box is furniture, the detection is
    still saved exactly as it is today: same row, same crop on disk, same clip trigger, same
    HUD, same species naming. All that happens afterwards is db.record_suppression writing four
    additive columns (suppressed_at / suppressed_by / suppress_ref_id / suppress_detail) that
    NOTHING in this codebase reads. The point is a week of flagged rows Matt can look at on one
    contact sheet (`python refimg.py --review`) before any consumer honours the flag. An erased
    animal writes no row at all and is a silent permanent loss; a wrongly flagged watering can
    is a row a human can clear. That asymmetry decides every tie in here, which is why every
    failure path below abstains, logs, and lets the detection through untouched.

    Wiring, per camera thread:
      observe(...)  once per frame the DETECTOR RAN on (it is already capped at one call per
                    second), carrying that frame, the detector's RAW boxes and MotionGate's
                    retained mask. Passing the raw boxes -- before ignore_zones -- is deliberate:
                    any box at all must reset certification, so a zone we chose to ignore can
                    never help a frame qualify as "empty".
      judge(...)    once per detection that was just written, AFTER its row and crop exist.

    THE PERSISTED REFERENCE IS THE ONE THAT JUDGES. refimg re-certifies on every quiet detector
    frame, so writing each certification would be a PNG and a row per second. Instead a
    reference is written at most every PERSIST_EVERY_S and that stored image is what the veto
    compares against until the next write -- so suppress_ref_id always resolves to the exact
    picture that produced the decision. Only the COVER mask is taken live between writes:
    coverage is the safety gate that protects a missed animal, and a stale one would protect
    nothing.
    """

    # 10 minutes: ~144 references/day/illumination worst case at 320x180 grey (a few MB), against
    # a max reference age of 2 h. Small enough that a judged reference is always fresh, large
    # enough that a quiet night is not a per-second write storm on the capture thread.
    PERSIST_EVERY_S = 600.0

    def __init__(self, cfg: config.Config, source: str, conn, tag: str = ""):
        self.cfg, self.source, self.tag = cfg, source, tag
        # Seed the view epoch from the DB: a restart must not reset the camera's history to 0 and
        # start writing references under an epoch key that other rows already used.
        self.epoch = db.current_view_epoch(conn, source)
        self.view = refimg.ViewWatcher(epoch=self.epoch)
        self.manager = refimg.ReferenceManager(source=source, view=self.view, cfg=cfg)
        self.veto = refimg.ShadowVeto(cfg=cfg)
        # The recurrence ledger persists to a JSON sidecar, NOT the DB -- it updates on every
        # detection in the capture thread, and this project has already killed the rig once with
        # "database is locked" from a worker holding the write lock.
        self.recurrence = refimg.Recurrence(
            path=refimg.store_dir(cfg) / source / "recurrence.json", cfg=cfg).load()
        self.obs = None                  # the last prepared frame (an refimg.Observation)
        self.ref = None                  # the reference the next judge() will use
        self._pinned: dict = {}          # illumination -> the reference written to the DB
        self.n_suppressed = 0
        self.n_judged = 0

    @classmethod
    def create(cls, cfg: config.Config, source: str, conn, tag: str = "") -> "RefimgShadow | None":
        """The shadow veto for this camera, or None when it is off or cannot start. None is the
        no-op path: every call site is `if shadow is not None`, so an unavailable veto is exactly
        today's rig."""
        if not getattr(cfg, "refimg_enabled", False):
            return None
        if refimg is None:
            print(f"{tag} refimg: WARNING -- refimg_enabled is on but refimg.py did not import "
                  f"({_REFIMG_IMPORT_ERROR}); running without the shadow veto.")
            return None
        try:
            shadow = cls(cfg, source, conn, tag)
        except Exception as e:
            print(f"{tag} refimg: WARNING -- could not start the shadow veto ({e}); "
                  "running without it. Detections are unaffected.")
            return None
        print(f"{tag} reference-image veto: SHADOW MODE -- flagging suppression metadata only "
              f"(view epoch {shadow.epoch}). Nothing reads the flag; audit with "
              f"`python refimg.py --review`.")
        return shadow

    # -- per-frame bookkeeping -------------------------------------------------------------
    def observe(self, conn, frame, detections, motion_mask, now: float) -> None:
        """Feed ONE frame the detector ran on. `detections` is that frame's RAW verdict (a list of
        Detection, or of bare boxes -- an empty list means "it ran and found nothing", which is
        what lets a frame certify). `now` is WALL-CLOCK epoch seconds (references are stamped with
        it and recurrence counts calendar days from it), not the loop's monotonic clock. Never
        raises: on any failure the frame is simply not usable and the veto abstains.
        """
        try:
            boxes = [getattr(d, "bbox", d) for d in (detections or ())]
            self.obs = self.manager.observe(frame, detections=boxes,
                                            motion_mask=motion_mask, now=now)
            if self.obs.view_epoch != self.epoch:
                self._note_epoch(conn, self.obs.view_epoch)
            self.ref = self._reference(conn, self.obs)
        except Exception as e:
            self.obs = self.ref = None
            self._warn("preparing the reference", e)

    def _reference(self, conn, obs):
        """The reference to judge against: the persisted image, with a live cover mask."""
        live = self.manager.get(obs.illumination)
        if live is None:
            return None
        pinned = self._pinned.get(obs.illumination)
        if (pinned is None or pinned.view_epoch != live.view_epoch
                or live.captured_at - pinned.captured_at >= self.PERSIST_EVERY_S):
            refimg.save_reference(conn, live, cfg=self.cfg)     # writes the PNGs + row, sets .id
            self._pinned[obs.illumination] = live
            return live
        # Between writes: the image (and id) already on disk, the cover mask as of right now.
        return replace(pinned, cover=live.cover, detail=live.detail)

    def _note_epoch(self, conn, epoch: int) -> None:
        """The camera was seen to MOVE. Record it as a first-class event -- this project's
        standing rule is that a stale hand-measured zone fails SILENTLY, and refimg has already
        dropped every reference and every remembered box for the old view."""
        corr = self.view.changes[-1][1] if self.view.changes else None
        self.epoch = epoch
        self._pinned.clear()
        try:
            recorded = db.bump_view_epoch(conn, self.source, corr=corr)
        except Exception as e:
            self._warn("recording the view epoch", e)
            return
        print(f"{self.tag} refimg: the camera looks MOVED (edge correlation {corr}) -- view epoch "
              f"{recorded}; references retired, suppression pauses until a fresh one certifies.")
        if recorded != epoch:
            print(f"{self.tag} refimg: NOTE -- the DB allocated epoch {recorded} but this rig "
                  f"counted {epoch}; something else has been writing view_epochs for "
                  f"'{self.source}'.")

    # -- per-detection judgement -----------------------------------------------------------
    def judge(self, conn, detection_id: int, det: Detection):
        """Judge ONE detection that has ALREADY been written. Returns the refimg.Decision (or
        None if it could not be evaluated). The row, the crop and every downstream consumer are
        untouched either way -- a SUPPRESS only adds the four metadata columns."""
        if self.obs is None:
            return None
        try:
            box = self.obs.to_working(det.bbox)
            # Recurrence is updated with this box BEFORE the veto reads it (the gate asks "has
            # this spot kept firing", and this firing counts).
            self.recurrence.observe(box, self.obs.ts, epoch=self.obs.view_epoch)
            decision = self.veto.evaluate(box, self.obs, self.ref, self.recurrence)
            self.n_judged += 1
            if decision.suppressed:
                db.record_suppression(conn, detection_id, db.SUPPRESSED_BY_REFIMG_VETO,
                                      ref_id=decision.ref_id, detail_json=decision.to_json())
                self.n_suppressed += 1
                r = decision.recurrence or {}
                print(f"  {self.tag} shadow veto: detection {detection_id} LOOKS LIKE FURNITURE "
                      f"(ref {decision.age_s:.0f}s old, {r.get('events', 0)} firings over "
                      f"{r.get('days', 0)} day(s)) -- flagged only, the row and crop are kept.")
            self.recurrence.save(self.obs.ts)      # throttled + atomic; a no-op most calls
            return decision
        except Exception as e:
            self._warn("judging a box", e)
            return None

    # -- presentation / shutdown -----------------------------------------------------------
    def hud(self) -> str:
        """One short token for the HUD line: 'ref night 7m', or 'ref --' when there is nothing to
        judge against (which is the honest, common answer -- the veto abstains)."""
        ref = self.ref
        if ref is None:
            return "ref --"
        return f"ref {ref.illumination} {_short_age(time.time() - ref.captured_at)}"

    def close(self) -> None:
        """Flush the recurrence ledger on shutdown so a clean stop keeps its evidence."""
        try:
            self.recurrence.save(force=True)
        except Exception as e:
            self._warn("saving the recurrence ledger", e)

    def _warn(self, doing: str, err: Exception) -> None:
        print(f"{self.tag} refimg: error {doing} ({err}) -- abstaining; detections are unaffected.")


# ---- Saving ------------------------------------------------------------------------
# Project-root-relative stored path. Aliased to the shared db helper (kept importable under this
# name because import_trailcam does `from backyard_cam import _rel`).
_rel = db.rel_to_root


def save_crop(frame_bgr, det: Detection, cfg: config.Config, day: str, stamp: str, idx: int):
    """Crop the (padded, clamped) detection box and write it as a JPEG. Returns (Path, quality) --
    the shot-quality score (quality.score_crop) computed from the crop we already have in hand, so
    the dashboard can later lead with the sharpest frame -- or None if the box is degenerate."""
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
    return path, quality.score_crop(crop)


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
             recording: bool = False, warnings: list[str] | None = None,
             ref_status: str | None = None) -> None:
    # ref_status is refimg's one-token state ('ref night 7m' / 'ref --'): whether there is a
    # certified empty reference for the current illumination and how old it is. None (the
    # shipped default, and what a disabled veto passes) leaves the line exactly as it was.
    lines = [
        f"{source}  |  {model}" + (f"  |  {period}" if period else ""),
        f"FPS {fps:4.1f}   motion {int(motion_area):>6} px   saved {saved}"
        + (f"   {ref_status}" if ref_status else ""),
        "q: quit",
    ]
    y = 22
    for text in lines:
        cv2.putText(frame, text, (10, y), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)      # shadow
        cv2.putText(frame, text, (10, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    # Rig warnings (on-battery, wedged camera): bigger, alarm-red, impossible to miss --
    # these are the "walk over and do something" lines (see powerguard.py).
    for text in (warnings or []):
        y += 6
        cv2.putText(frame, text, (10, y), FONT, 0.62, (0, 0, 0), 4, cv2.LINE_AA)      # shadow
        cv2.putText(frame, text, (10, y), FONT, 0.62, (0, 60, 255), 2, cv2.LINE_AA)
        y += 28
    if motion:  # red "motion" dot, top-right
        cv2.circle(frame, (frame.shape[1] - 20, 20), 8, (0, 0, 255), -1)
    if recording:  # "REC" tag under the motion dot while a behaviour clip is being written
        cv2.putText(frame, "REC", (frame.shape[1] - 52, 48), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "REC", (frame.shape[1] - 52, 48), FONT, 0.6, (0, 0, 255), 1, cv2.LINE_AA)


# ---- Multi-camera preview grid -----------------------------------------------------
def _fit_cell(frame, w: int, h: int):
    """Letterbox `frame` into a w x h black cell (preserving aspect). A None frame -- a camera
    that hasn't produced its first frame yet -- becomes a plain black cell, so the grid is stable."""
    cell = np.zeros((h, w, 3), dtype=np.uint8)
    if frame is None or frame.size == 0:
        return cell
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x, y = (w - nw) // 2, (h - nh) // 2
    cell[y:y + nh, x:x + nw] = resized
    return cell


def compose_grid(frames: list):
    """Tile per-camera annotated frames into one image for the native preview window (the web
    dashboard renders its own grid). ONE camera -> its frame as-is, so the single-camera preview is
    pixel-identical to before. Several -> a near-square grid of fixed cells. Returns None only when
    there's nothing to show yet (a single camera with no frame)."""
    n = len(frames)
    if n == 0:
        return None
    if n == 1:
        return frames[0]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cw, ch = 640, 360
    canvas = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = _fit_cell(f, cw, ch)
    return canvas


# ---- Naming subprocess ------------------------------------------------------------
_NAMING_TAG_PREFIX = "backyard-naming-"     # helper --tag = this prefix + the spawning rig's pid
_NAMING_TAG_RE = re.compile(re.escape(_NAMING_TAG_PREFIX) + r"(\d+)")
_RIG_CMDLINE_MARKER = "backyard_cam"        # present however a rig is launched (script or -m)


def _parse_process_rows(text: str) -> list[tuple[int, str]]:
    """Parse 'pid<TAB>command line' lines into (pid, cmdline) pairs. Only the FIRST tab splits
    (a command line may itself contain tabs); lines without a leading pid are dropped."""
    rows = []
    for line in text.splitlines():
        pid_s, _, cmdline = line.partition("\t")
        try:
            rows.append((int(pid_s.strip()), cmdline.strip()))
        except ValueError:
            continue
    return rows


def _python_process_rows(timeout: float = 15) -> list[tuple[int, str]]:
    """(pid, command line) of every live python* process, in one CIM query. Windows-only,
    best-effort: any failure -- including a non-zero exit, whose output could be a TRUNCATED
    table that makes a live rig look dead -- returns [] so callers do nothing at all."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
          'ForEach-Object { "$($_.ProcessId)`t$($_.CommandLine)" }')
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return []
        return _parse_process_rows(out.stdout)
    except Exception:
        return []


def _naming_pids(tag: str) -> list:
    """PIDs of python processes whose command line carries our unique --tag (the live-naming
    helper and any interpreter the venv launcher re-spawned for it). Windows-only, best-effort."""
    return [str(pid) for pid, cmd in _python_process_rows() if tag in cmd]


def _stale_naming_pids(rows: list[tuple[int, str]], my_pid: int) -> list[int]:
    """Which of these python processes are naming helpers whose rig is PROVABLY gone.

    A helper's --tag carries the pid of the rig that spawned it. Two rigs legitimately run
    side by side into the same DB, and Windows reuses pids, so the bar for killing is proof
    of death, not absence of proof of life:
      * tag pid == my_pid: stale. We sweep BEFORE spawning our own helper, so a helper
        already wearing our pid belongs to a dead rig whose pid we happened to inherit.
      * tag pid is not a live python process: stale (rig exited, or its pid was reused by
        something that isn't python).
      * tag pid is a live python whose readable command line is some OTHER script: stale
        (pid reused by e.g. a clipmotion batch).
      * anything else -- a live rig, or a python whose command line we can't read: KEPT.
        A lingering stale helper wastes CPU; killing a live rig's helper stops its naming.
    Both rows of a helper (venv shim + real interpreter) carry the tag, so both come back."""
    cmd_by_pid = dict(rows)
    stale = []
    for pid, cmd in rows:
        if "classify.py" not in cmd:
            continue                        # not a naming helper (a rig, a batch job, a REPL...)
        m = _NAMING_TAG_RE.search(cmd)
        if m is None:
            continue
        rig_pid = int(m.group(1))
        rig_cmd = cmd_by_pid.get(rig_pid)
        if (rig_pid == my_pid
                or rig_cmd is None
                or (rig_cmd and _RIG_CMDLINE_MARKER not in rig_cmd)):
            stale.append(pid)
    return stale


def _sweep_stale_naming() -> None:
    """Reap naming helpers orphaned by a previous rig that died WITHOUT running its finally
    block (taskkill /F, a crash, the OOM/standby deaths). classify.py never exits on its own,
    so a leftover helper plus this run's fresh one means two BioCLIP workers fighting over CPU
    and the SQLite write lock (the documented 'database is locked' capture failure). Windows-
    only, best-effort and silent on failure; the query timeout is tight so a slow WMI never
    holds up startup. MUST run before this rig spawns its own helper (see _stale_naming_pids)."""
    if sys.platform != "win32":
        return
    try:
        me = os.getpid()
        stale = [p for p in _stale_naming_pids(_python_process_rows(timeout=2), me)
                 if p != me and p > 4]      # paranoia: never self/system pids
        if not stale:
            return
        print(f"  naming: reaping {len(stale)} stale helper process(es) left by a dead rig "
              f"(pids {', '.join(map(str, stale))}).")
        subprocess.run(["taskkill", "/F"] + [a for p in stale for a in ("/PID", str(p))],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        pass


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


# ---- Per-camera capture worker -----------------------------------------------------
def _run_camera(spec, cfg, detector, det_lock, frame_buffers, control_bridges,
                latest_frames, latest_lock, stop_event, results, powermon=None, healer=None,
                zone_store=None):
    """One camera's whole capture pipeline, run in its own thread:
        read -> MOG2 motion gate -> (shared) MegaDetector on motion -> save crop + DB row + clip,
        then publish an annotated frame to this camera's web FrameBuffer and the preview grid.
    Each worker owns its OWN capture, motion gate, DB connection and clip recorder, and tags every
    row with spec.source -- so N cameras never interfere. The one shared resource is the detector,
    serialized by det_lock (PyTorch inference isn't guaranteed thread-safe, and the motion gate
    keeps detector calls rare, so the lock almost never contends). Does NO cv2 GUI -- the main
    thread owns the window. Tallies are reported back via results[spec.source]."""
    tag = f"[{spec.source}]"
    fb = frame_buffers.get(spec.source)         # None when not serving the dashboard
    bridge = control_bridges.get(spec.source)   # None when not serving
    eff_motion_min_area = _eff(spec, cfg, "motion_min_area")
    eff_profiles = _eff(spec, cfg, "camera_profiles") or {}
    record = _eff(spec, cfg, "record_clips")
    clip_trigger = cfg.clip_classes or cfg.save_classes
    # Static false-fire spots for THIS camera: a detection boxed ~on one is dropped before
    # drawing, saving, and clip-triggering. Framing-specific by design. These now come from the
    # shared IgnoreZoneStore (DB-backed, dashboard-editable; config.ignore_zones just seeds it),
    # re-read every loop pass below so an edit takes effect on the next frame without a restart.

    conn = db.connect(cfg.db_path)
    cap = None
    recorder = None
    shadow = None                               # refimg shadow veto; stays None when it's off
    saved = ignored = 0
    try:
        cap = open_capture(spec, cfg)
        if cap is None:
            # Couldn't open on the first try -- but don't bail. An IP cam may still be booting, and
            # a local USB cam may be mid-suspend (the box was in Modern Standby when we launched).
            # reconnect_capture retries forever (honouring stop_event), so we recover once it wakes.
            print(f"{tag} could not open camera src={spec.src!r} yet -- will keep trying to connect."
                  + ("" if spec.is_url else "  (if it never opens, `python backyard_cam.py "
                     "--list-cameras` finds the right index)"))
            cap = reconnect_capture(spec, cfg, stop_event)
        if cap is None:
            return                                  # only reached when we're shutting down
        print(f"{tag} open ({spec.src!r}).")

        recorder = clips.ClipRecorder(cfg, conn, source=spec.source) if record else None

        # Sun-driven day/night camera profile -- LOCAL cams only (a network stream exposes no UVC
        # exposure controls), and only when no manual exposure/gain is forcing static settings.
        profiles_on = (_eff(spec, cfg, "use_time_of_day_profiles") and not spec.is_url
                       and _eff(spec, cfg, "exposure") is None and _eff(spec, cfg, "gain") is None
                       and cfg.latitude is not None and cfg.longitude is not None)
        active_period = None
        last_profile_check = 0.0
        # The auto modes as last commanded -- the dashboard's truth source, because this
        # driver's get() lies about them (see CommandedAutoState).
        auto_cmd = CommandedAutoState(spec, cfg)
        if profiles_on:
            active_period = daynight.current_period(cfg.latitude, cfg.longitude)
            apply_camera_settings(cap, eff_profiles.get(active_period, {}))
            auto_cmd.note(eff_profiles.get(active_period, {}))
            print(f"{tag} camera profile: '{active_period}'")

        # Probe writable sliders once (local cams only; a network cam has none we can set).
        writable = {} if spec.is_url else (probe_writable_controls(cap) if bridge is not None else {})

        # Auto-white-balance recovery. A network stream's colour is the camera's own business,
        # so this is local-cams-only like the rest of the UVC handling.
        wb_guard = WhiteBalanceWatchdog(cfg, tag)
        if spec.is_url:
            wb_guard.enabled = False

        # USB-wedge watch (powerguard.py): garbage-frame detection + the device-reset ladder.
        # Local USB cams only -- a network cam can't be pnputil-cycled, and its stream tearing
        # is the transport's problem (rtsp-over-TCP), not a UVC wedge.
        wedge = powerguard.WedgeDetector(cfg, tag, healer=healer)
        if spec.is_url:
            wedge.enabled = False

        gate = MotionGate(cfg)
        # Detector census (measurement, ALWAYS on: design section 8 item 1) + the reference-image
        # veto in shadow mode (off unless cfg.refimg_enabled; None == exactly today's rig).
        census = DetectorCensus()
        shadow = RefimgShadow.create(cfg, spec.source, conn, tag)
        last_detect_t = last_dets_t = last_animal_t = 0.0
        last_dets: list[Detection] = []
        frame_count = read_fails = fps_n = 0
        fps = 0.0
        fps_t = time.monotonic()
        last_ctrl_pub = 0.0
        # Display work (copy + draw + JPEG encode) is capped at display_max_fps and skipped
        # entirely when nothing consumes it -- it's presentation, not capture (see config).
        disp_min_dt = (1.0 / cfg.display_max_fps) if cfg.display_max_fps else 0.0
        last_disp_t = 0.0

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                read_fails += 1
                if read_fails >= READ_FAIL_TOLERANCE:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = reconnect_capture(spec, cfg, stop_event)
                    read_fails = 0
                    if cap is None:
                        break                       # only None when we're shutting down
                    active_period = None            # re-apply the profile after a reconnect
                    last_profile_check = 0.0
                    # open_capture just re-asserted the all-auto baseline on the fresh device.
                    auto_cmd = CommandedAutoState(spec, cfg)
                    # Fresh capture, fresh background model: after a standby wake (or a heal's
                    # device reset) the scene lighting has usually changed, and the OLD model
                    # would read the whole frame as one giant motion blob for a minute -- which
                    # both churns the detector and looks exactly like the wedge signature.
                    # Rebuilding re-learns the scene in seconds; the warmup gate covers the gap.
                    gate = MotionGate(cfg)
                    frame_count = 0
                    wedge.note_reconnect(time.monotonic())
                continue
            read_fails = 0
            frame_count += 1
            # A lock + tuple fetch per frame: the price of dashboard zone edits landing live.
            ignore_zones = zone_store.rects(spec.source) if zone_store is not None else ()

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
                    apply_camera_settings(cap, eff_profiles.get(period, {}))
                    auto_cmd.note(eff_profiles.get(period, {}))
                    print(f"{tag} sun crossed -- switched to '{period}' profile")

            # --- White-balance watchdog (self-rate-limited; see WhiteBalanceWatchdog) ---
            wb_rec = wb_guard.recoveries
            wb_guard.check(cap, frame, now)
            if wb_guard.recoveries != wb_rec:
                auto_cmd.note({"AUTO_WB": 1})   # the watchdog put white balance back to auto

            # --- Live camera controls from the dashboard (web thread -> this capture thread) ---
            if bridge is not None:
                pending = bridge.take_pending()
                if pending and not spec.is_url:
                    apply_camera_settings(cap, pending)
                    auto_cmd.note(pending)
                    wb_guard.note_settings(pending)   # honour a deliberate manual-WB choice
                if now - last_ctrl_pub >= 1.0:
                    last_ctrl_pub = now
                    snap = {} if spec.is_url else read_camera_controls(cap)
                    if not spec.is_url:
                        snap.update(auto_cmd.state)   # commanded truth beats the driver's guess
                    snap["period"] = active_period
                    snap["writable"] = writable
                    # Colour health, so a cyan-cast session is visible instead of just looking
                    # "a bit off": ~1.0 = neutral, well under 1 = the red-starved manual-WB state.
                    snap["wb_ratio"] = (round(wb_guard.last_ratio, 3)
                                        if wb_guard.last_ratio is not None else None)
                    snap["wb_recoveries"] = wb_guard.recoveries
                    # Rig health, for the dashboard's warning strip: on-battery and wedge state
                    # (see powerguard.py -- these are the "walk over and do something" signals).
                    snap["power"] = powermon.snapshot() if powermon is not None else None
                    snap["wedge"] = wedge.snapshot()
                    # Masthead live chip: an animal counts as "on-cam now" if a critter-class
                    # detection fired within the last on_cam_window_s (last_animal_t == 0.0
                    # means none yet this run, published as age None / active False).
                    age_s = (now - last_animal_t) if last_animal_t > 0 else None
                    snap["animal_age_s"] = None if age_s is None else round(age_s, 1)
                    snap["animal_active"] = age_s is not None and age_s <= cfg.on_cam_window_s
                    snap["network"] = spec.is_url   # the dashboard hides sliders for a network cam
                    # The TRUE frame size (what ignore-zone pixel coords are measured in) -- the
                    # zones editor clamps against this rather than trusting a browser's numbers.
                    snap["frame_w"], snap["frame_h"] = int(frame.shape[1]), int(frame.shape[0])
                    # Coarsen the published coords (~10 km) -- a LAN/DNS-rebind client can read this.
                    snap["lat"] = round(cfg.latitude, 1) if cfg.latitude is not None else None
                    snap["lon"] = round(cfg.longitude, 1) if cfg.longitude is not None else None
                    bridge.publish(snap)

            # --- Motion gate (per-camera background model + threshold) ---
            motion_area = gate.update(frame)
            motion = (frame_count > MOTION_WARMUP_FRAMES) and (motion_area > eff_motion_min_area)

            # --- Wedge watch: feed the cheap signals, advance the detect->heal->replug ladder ---
            if wedge.enabled:
                if frame_count > MOTION_WARMUP_FRAMES:
                    fh, fw = frame.shape[:2]
                    wedge.note_motion(now, motion_area / float(max(1, fh * fw)))
                wedge.note_wb(wb_guard.failed_streak)
                wedge.update(now)

            # --- Behaviour clip: buffer pre-roll every frame; write + auto-stop while recording ---
            if recorder is not None:
                recorder.note_frame(frame, now, loop_fps=fps)

            # --- Detector census: one line per hour, whatever the veto is doing (a zero-run hour
            #     is itself the finding, so this is checked every frame, not only on a run) ---
            census_line = census.roll(now)
            if census_line:
                print(f"{tag} {census_line}")

            # --- Detector (only on motion, rate-limited, under the shared-detector lock) ---
            if motion and (now - last_detect_t) >= cfg.detector_min_interval_s:
                last_detect_t = now
                try:
                    with det_lock:
                        dets = detector.detect(frame)
                except Exception as e:  # never let one bad frame kill the loop
                    print(f"{tag} detector error on a frame (skipping): {e}")
                    dets = []
                # Count the RAW verdict (pre ignore-zone), and hand the same raw verdict to the
                # reference manager: a frame can only certify as "empty" if the detector saw
                # NOTHING in it, and a box we chose to ignore is still a box it saw.
                census.note(now, len(dets))
                if shadow is not None:
                    shadow.observe(conn, frame, dets, gate.mask, time.time())
                if dets:
                    # Any MegaDetector hit (even a zone-dropped one) vetoes the pegged-motion
                    # wedge rule: real structure in the frame means big motion is an animal or
                    # a scene change, not UVC garbage. (The 07-30 noise variant that DID score
                    # "animal" is caught by the WB rule instead.)
                    wedge.note_detections(now)
                if dets and ignore_zones:
                    dets, zoned_out = drop_ignored(dets, ignore_zones, cfg.ignore_zone_iou)
                    ignored += len(zoned_out)
                if dets:
                    last_dets, last_dets_t = dets, now
                    saved_dets = [d for d in dets if d.class_name in cfg.save_classes]
                    if saved_dets:
                        last_animal_t = now   # critter on-cam (stamped even if the save path fails)
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
                        s = save_crop(frame, det, cfg, day, stamp, i)
                        if s is None:
                            continue
                        crop_path, crop_q = s
                        # The crop is already on disk, so a transient DB hiccup (classically
                        # "database is locked" while the naming helper holds a long write) must
                        # NOT kill the capture thread -- log it and keep watching. Same spirit as
                        # the detector guard above; staying alive beats one detection row.
                        try:
                            det_id = db.insert_detection(
                                conn, timestamp=iso, source=spec.source,
                                detection_class=det.class_name, confidence=det.confidence,
                                bbox=det.bbox, frame_w=w, frame_h=h,
                                crop_path=_rel(crop_path), frame_path=frame_path, crop_quality=crop_q)
                        except Exception as e:
                            print(f"  {tag} [{iso}] DB write failed, kept crop {_rel(crop_path)}: {e}")
                            continue
                        saved += 1
                        print(f"  {tag} [{iso}] {det.class_name} {det.confidence:.2f} -> {_rel(crop_path)}")
                        # SHADOW VETO, strictly AFTER the row and crop exist: it can only ADD
                        # suppression metadata to a detection that was saved exactly as before.
                        if shadow is not None:
                            shadow.judge(conn, det_id, det)

            # --- Annotate for the dashboard feed + the native preview grid (NO cv2 GUI here) ---
            # Presentation only, so it's rate-capped, and the dashboard side runs ONLY while a
            # stream client / fresh snapshot request is actually watching. Capture never waits.
            want_stream = fb is not None and fb.watched()
            if (want_stream or cfg.show_preview) and (now - last_disp_t) >= disp_min_dt:
                last_disp_t = now
                disp = frame.copy()
                for zx1, zy1, zx2, zy2 in ignore_zones:   # faint outline: "the rig ignores this spot"
                    cv2.rectangle(disp, (int(zx1), int(zy1)), (int(zx2), int(zy2)),
                                  (120, 120, 120), 1)
                    cv2.putText(disp, "ignored", (int(zx1) + 4, int(zy1) + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
                show = last_dets if (now - last_dets_t) <= cfg.box_display_ttl_s else []
                draw_detections(disp, show)
                warns = []
                if powermon is not None and powermon.warning:
                    warns.append(powermon.warning)
                if wedge.message:
                    warns.append(wedge.message)
                draw_hud(disp, fps=fps, motion_area=motion_area, motion=motion, saved=saved,
                         source=spec.display_name, model=cfg.model_version, period=active_period,
                         recording=recorder is not None and recorder.recording, warnings=warns,
                         ref_status=shadow.hud() if shadow is not None else None)
                if want_stream:
                    ok_enc, buf = cv2.imencode(".jpg", disp,
                                               [cv2.IMWRITE_JPEG_QUALITY, cfg.web_jpeg_quality])
                    if ok_enc:
                        fb.update(buf.tobytes())
                if cfg.show_preview:
                    with latest_lock:
                        latest_frames[spec.source] = disp
    except Exception as e:  # never let one camera's crash take down the whole rig
        print(f"{tag} capture thread stopped on error: {e}")
    finally:
        if recorder is not None:
            try:
                recorder.finalize(shutdown=True)    # flush any clip mid-record (writes its DB row)
            except Exception:
                pass
        if shadow is not None:
            shadow.close()                          # flush the recurrence ledger (never raises)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass
        results[spec.source] = {"saved": saved, "ignored": ignored,
                                "clips": recorder.clips_saved if recorder is not None else 0,
                                "suppressed": shadow.n_suppressed if shadow is not None else 0}


# ---- Native preview window (main thread owns all cv2 GUI) ---------------------------
def _preview_loop(cfg, specs, latest_frames, latest_lock, stop_event, threads):
    """Composite each camera's latest annotated frame into a grid, show it, and watch for 'q' or a
    window close. ALL cv2 GUI lives on the main thread (imshow/waitKey aren't thread-safe), so the
    capture workers only ever hand frames over. Sets stop_event when the user quits or every camera
    thread has exited (so a single dead camera still ends the rig instead of hanging on a blank window)."""
    sources = [s.source for s in specs]
    while not stop_event.is_set():
        with latest_lock:
            frames = [latest_frames.get(s) for s in sources]
        grid = compose_grid(frames)
        if grid is not None:
            ps = cfg.preview_scale
            if ps and ps != 1.0:   # shrink BEFORE imshow -- HighGUI re-uploads every tick
                grid = cv2.resize(grid, (max(1, int(grid.shape[1] * ps)),
                                         max(1, int(grid.shape[0] * ps))),
                                  interpolation=cv2.INTER_AREA)
            cv2.imshow(cfg.window_name, grid)
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            print("\n'q' pressed -- shutting down.")
            break
        try:
            # Closing the video window (its X button) shuts the rig down too, so a teen can just
            # close the window instead of remembering the 'q' key.
            if cv2.getWindowProperty(cfg.window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\nVideo window closed -- shutting down.")
                break
        except cv2.error:
            pass
        if not any(t.is_alive() for t in threads):
            print("\nAll cameras stopped -- shutting down.")
            break
    stop_event.set()


# ---- Main loop ---------------------------------------------------------------------
def run(cfg: config.Config) -> None:
    """Drive every configured camera at once: one capture thread each, sharing one MegaDetector and
    one dashboard. Single-camera mode is just the N=1 case (cfg.camera_specs() synthesizes one spec
    from the flat fields), so existing setups behave exactly as before."""
    specs = cfg.camera_specs()
    seen = set()
    for s in specs:                                 # a duplicate source would silently merge two cams
        if s.source in seen:
            raise RuntimeError(
                f"Duplicate camera source '{s.source}' in cfg.cameras -- give each camera a unique source name.")
        seen.add(s.source)

    cfg.crops_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_full_frame:
        cfg.frames_dir.mkdir(parents=True, exist_ok=True)
    if any(_eff(s, cfg, "record_clips") for s in specs):
        cfg.clips_dir.mkdir(parents=True, exist_ok=True)

    # Networked cams read rtsp:// through FFMPEG; force TCP transport (UDP packet loss tears H.264
    # frames). Must be set before any VideoCapture opens a stream -- so set it here, once, up front.
    if any(isinstance(s.src, str) and s.src.lower().startswith("rtsp") for s in specs):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    conn = db.connect(cfg.db_path)                  # ensures schema first; reused for the shutdown rebuild
    server = None
    frame_buffers: dict = {}
    control_bridges: dict = {}
    classify_proc = None
    classify_tag = f"{_NAMING_TAG_PREFIX}{os.getpid()}"   # unique marker for a clean, total shutdown
    # A previous rig that died without its finally (taskkill /F, crash, OOM kill) leaves its
    # helper running forever. Reap those leftovers now, BEFORE our own helper exists.
    _sweep_stale_naming()
    threads: list[threading.Thread] = []
    stop_event = threading.Event()
    latest_frames: dict = {}
    latest_lock = threading.Lock()
    results: dict = {}

    try:
        # Hold the system awake for the lifetime of the run so it can't idle into Modern Standby
        # and USB-suspend the camera out from under us (the cause of the silent overnight deaths).
        if keep_system_awake():
            print("  power: holding the system awake (no idle-sleep) while the cam runs.")
        elif sys.platform == "win32":
            print("  power: WARNING -- could not hold the system awake; if the box sleeps it may "
                  "suspend the camera and stop the app. (Also turn off USB selective suspend.)")

        # Battery watch + wedge self-heal plumbing (powerguard.py). ONE monitor and ONE healer
        # for the rig (one physical box, one hardware-reset budget); each camera worker gets its
        # own WedgeDetector. The immediate poll makes an on-battery LAUNCH warn right away
        # instead of power_poll_s later.
        powermon = powerguard.PowerMonitor(cfg)
        powermon.poll(time.monotonic())
        threading.Thread(target=powermon.monitor, args=(stop_event,),
                         name="powermon", daemon=True).start()
        healer = powerguard.SelfHealer(cfg, "[usb-heal]")
        if sys.platform == "win32" and cfg.wedge_guard:
            if cfg.wedge_self_heal and healer.available():
                print("  wedge guard: ON -- garbage-frame watch + automatic USB device reset "
                      f"(task '{cfg.wedge_heal_task}').")
            elif cfg.wedge_self_heal:
                print("  wedge guard: ON (detect + banner only) -- run setup_selfheal.bat once, "
                      "as admin, to enable automatic USB device resets.")
            else:
                print("  wedge guard: ON (detect + banner only; self-heal disabled in config).")

        # Build the (shared) detector first: resolves the device (a real GPU compute-check for
        # 'cuda'/'auto') and downloads the weights on first run, failing loud with device='cuda'.
        print(f"Loading MegaDetector v6 ({cfg.model_version}) on {cfg.device} ...")
        print("  (first run downloads the model weights from Zenodo -- one time only)")
        detector = Detector(cfg.model_version, cfg.device, cfg.min_confidence, classes=cfg.detect_classes)
        if detector.device == "cuda":
            print(f"  detector ready on GPU: {detector.device_name}")
        else:
            print("  detector ready on CPU -- slower per frame, but the motion gate keeps it usable.")
        det_lock = threading.Lock()

        n = len(specs)
        print(f"  {n} camera{'' if n == 1 else 's'}: "
              + ", ".join(f"{s.display_name} ({s.src!r})" for s in specs))
        if cfg.latitude is not None and cfg.longitude is not None:
            try:
                st = daynight.sun_times(cfg.latitude, cfg.longitude)
                print("  sun today: " + "  ".join(f"{k} {v.strftime('%H:%M')}" for k, v in st.items()))
            except Exception:
                pass
        if any(_eff(s, cfg, "record_clips") for s in specs):
            print(f"  behaviour clips: ON -- short videos to {_rel(cfg.clips_dir)}/<source>/ "
                  f"(pre {cfg.clip_pre_roll_s:g}s / post {cfg.clip_post_roll_s:g}s, cap {cfg.clip_max_s:g}s).")

        # Ignore zones (static false-fire spots): DB-backed and dashboard-editable, one store
        # shared by the web server and every capture thread. config.ignore_zones seeds the table
        # the first time each rectangle is seen; a zone deleted in the UI stays deleted (see
        # db.seed_ignore_zones). Built even with the dashboard off -- the capture threads read it.
        zone_store = web.IgnoreZoneStore.load(cfg)
        zc = zone_store.counts()
        if zc:
            print("  ignore zones: " + ", ".join(f"{n} on {s}" for s, n in sorted(zc.items()))
                  + "  (edit in the dashboard's Instrument Panel)")

        # Optional local web dashboard: one FrameBuffer + control bridge PER camera, so the Live tab
        # can show a grid of feeds and control each camera independently.
        if cfg.serve:
            for s in specs:
                frame_buffers[s.source] = web.FrameBuffer()
                control_bridges[s.source] = web.CameraControlBridge()
            try:
                server = web.start(cfg, frame_buffers, control_bridges, zone_store)
                print(f"  dashboard: http://{cfg.web_host}:{cfg.web_port}  (open in a browser)\n")
            except OSError as e:
                print(f"  [web] could not start the dashboard on {cfg.web_host}:{cfg.web_port}: {e}")
                print("  [web] continuing without it -- try another port with --port N.\n")
                server, frame_buffers, control_bridges = None, {}, {}

        # ONE shared species-naming child (classify.py --watch): it names any crop lacking a species
        # regardless of source, so a single helper covers every camera (one launch, one stop). Runs
        # as a child PROCESS because BioCLIP must be built on a process's main thread (a thread deadlocks).
        # NOTE the process list shows TWO python rows with this helper's command line: the venv's
        # python.exe is a redirector that launches the real interpreter as its child, so every spawn
        # is a 0-CPU shim + one worker (the rig itself appears doubled the same way). That's one
        # helper, not two -- _naming_pids, _stop_naming and the startup sweep all reap both rows.
        if cfg.classify_live:
            try:
                classify_proc = subprocess.Popen(
                    [sys.executable, str(config.ROOT / "classify.py"), "--watch",
                     "--device", str(cfg.classify_device),
                     "--interval", str(cfg.classify_interval_s), "--tag", classify_tag],
                    cwd=str(config.ROOT))
                print("  species naming: ON -- a helper is warming up the model (~1-2 min), then it\n"
                      "  names new crops automatically. The dashboard shows when it's ready.\n")
            except Exception as e:
                print(f"  [naming] couldn't start the species-naming helper "
                      f"(detection still works): {e}\n")
                classify_proc = None

        # One capture thread per camera, all sharing the single detector.
        for s in specs:
            t = threading.Thread(
                target=_run_camera, name=f"cam-{s.source}",
                args=(s, cfg, detector, det_lock, frame_buffers, control_bridges,
                      latest_frames, latest_lock, stop_event, results, powermon, healer,
                      zone_store),
                daemon=True)
            t.start()
            threads.append(t)
        print("Watching for critters -- press 'q' in the window (or close it) to quit.\n")

        if cfg.show_preview:
            # The MAIN thread owns the cv2 window (GUI calls must be single-threaded).
            cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            ps = cfg.preview_scale or 1.0
            cv2.resizeWindow(cfg.window_name,
                             max(1, int(cfg.frame_width * ps)), max(1, int(cfg.frame_height * ps)))
            _preview_loop(cfg, specs, latest_frames, latest_lock, stop_event, threads)
        else:
            # Headless: idle until Ctrl+C or every camera thread has exited.
            while not stop_event.is_set() and any(t.is_alive() for t in threads):
                stop_event.wait(0.3)
    except KeyboardInterrupt:
        print("\nInterrupted -- shutting down.")
    finally:
        stop_event.set()                            # tell every worker (and any reconnect wait) to stop
        allow_system_sleep()                        # let the box idle/sleep normally again
        for t in threads:
            t.join(timeout=10)
        _stop_naming(classify_proc, classify_tag)   # kill the helper + any venv-launcher subproc
        if server is not None:
            web.shutdown(server)
        cv2.destroyAllWindows()
        # Keep the visit ledger fresh: collapse this session's detections into visit events so the
        # dashboard's Behaviour tab is current without a manual `python visits.py`. Best-effort
        # (visits.refresh never raises) and subsecond here.
        visits.refresh(conn, cfg.visit_gap_minutes)
        conn.close()
        total_saved = sum(r.get("saved", 0) for r in results.values())
        total_clips = sum(r.get("clips", 0) for r in results.values())
        total_ignored = sum(r.get("ignored", 0) for r in results.values())
        total_suppressed = sum(r.get("suppressed", 0) for r in results.values())
        clips_note = f"  Recorded {total_clips} clip(s)." if total_clips else ""
        zone_note = (f"  Dropped {total_ignored} ignore-zone false-fire(s)." if total_ignored else "")
        # Shadow mode: these rows were SAVED and merely flagged -- say so, so the number is never
        # mistaken for detections that went missing.
        veto_note = (f"  Flagged {total_suppressed} row(s) as furniture (kept; audit with "
                     f"`python refimg.py --review`)." if total_suppressed else "")
        print(f"Done. Saved {total_saved} detection(s) this session to {cfg.db_path}."
              f"{clips_note}{zone_note}{veto_note}")


# ---- CLI ---------------------------------------------------------------------------
def parse_args() -> tuple[config.Config, argparse.Namespace]:
    c = CONFIG
    p = argparse.ArgumentParser(
        description="Backyard critter detection -- V1 live glass-door cam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera-index", type=int, default=c.camera_index,
                   help="Which webcam (single-camera mode; --list-cameras finds it). For SEVERAL "
                        "cameras at once -- USB + networked -- set cfg.cameras in config_local.py.")
    p.add_argument("--source", default=c.source,
                   help="DB 'source' label for this camera's rows (single-camera mode). Defaults to "
                        "'glass_door_cam'; set it when running a second live rig into the same DB.")
    p.add_argument("--width", type=int, default=c.frame_width)
    p.add_argument("--height", type=int, default=c.frame_height)
    p.add_argument("--exposure", type=float, default=c.exposure,
                   help="Lock manual exposure (find a value with tune.py). Omit to auto-expose.")
    p.add_argument("--gain", type=float, default=c.gain, help="Lock manual gain. Omit for auto.")
    p.add_argument("--model-version", default=c.model_version,
                   help="MDV6-yolov10-c | MDV6-yolov9-c | MDV6-rtdetr-c | MDV6-yolov10-e | MDV6-yolov9-e")
    p.add_argument("--device", default=c.device, choices=["cuda", "cpu", "auto"],
                   help="Inference device: auto (GPU when it genuinely runs, else CPU) | cuda "
                        "(REQUIRE an NVIDIA GPU, fail loud without one) | cpu (no GPU needed, "
                        "slower).")
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
    p.add_argument("--record-clips", action=argparse.BooleanOptionalAction, default=c.record_clips,
                   help="Record a short video clip around each visit (phase-4 behaviour capture; "
                        "ON by default, disk-capped to clips_max_gb). --no-record-clips turns it "
                        "off for a run. Crops are still saved alongside.")
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
        source=args.source,
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
    # Start the file log before anything else so the whole run (including a crash) is captured.
    log_path = install_file_logging()
    if log_path is not None:
        print(f"  log: writing this session to {log_path}")
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
    except Exception:
        # Anything that would otherwise vanish with the window: log the full traceback to the file
        # (via the tee'd stderr) AND surface a nonzero exit so the .bat keeps the window open.
        import traceback
        print("\n[FATAL] uncaught exception -- the run is ending. Traceback:")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
