"""
Central configuration for the backyard-critter detection rig.

This is the ONE place to tune the rig. Every knob the app reads lives here; the CLI
(backyard_cam.py) exposes the common ones as --flags that override these defaults.

V1 scope: the live glass-door USB webcam ("glass_door_cam") -- the PRIMARY rig for all three
species, day AND night (crows by day; a raccoon at dusk; lit mammals at the glass after
dark -- the "glass = mirror at night" worry did not materialize). A second source, the
wider-yard weatherproof trail cam (batch SD-card import, IR at night), plugs into the same
pipeline later via the `source` column. See PLAN.md for the full four-phase roadmap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root = the folder containing this file. All default paths hang off it, so the
# whole rig (code + db + crops) is relocatable as one directory.
ROOT = Path(__file__).resolve().parent

# Where the live species-naming helper (classify.py --watch) writes its status, so the dashboard
# can show "warming up" vs "naming" vs "stopped". A hidden file in the project root.
NAMING_STATUS_FILE = ROOT / ".naming_status.json"


@dataclass
class Config:
    # ---- Camera (OpenCV capture) ------------------------------------------------
    camera_index: int = 1           # USB webcam index; 0 = default device.
    frame_width: int = 1280         # Requested width  (the webcam may snap to the nearest).
    frame_height: int = 720         # Requested height (the webcam may snap to the nearest).
    # CAP_DSHOW = DirectShow backend: fast init on Windows and avoids the slow MSMF path.
    # DirectShow is Windows-only, so this is automatically ignored off-Windows (Linux/macOS get
    # CAP_ANY and OpenCV picks the native backend) -- see capture_backend() in backyard_cam.py.
    use_dshow_backend: bool = True
    camera_warmup_frames: int = 5   # Throwaway reads so exposure / auto-white-balance settle.
    # Manual exposure / gain. None = let the camera auto-expose (default). Set a number to
    # LOCK it -- use tune.py to find good values for the current scene. On Windows/DirectShow
    # EXPOSURE is ~log2 seconds (e.g. -6 ~ 1/64 s); lower = faster shutter = less motion blur
    # but darker. Re-tune after you move the camera; one value won't fit both day and night.
    exposure: float | None = None
    gain: float | None = None
    # Static UVC controls applied on open (NAME -> value, mapped to cv2.CAP_PROP_<NAME>).
    # Tuning (tune.py, 2026-06-08) found backlight-compensation OFF gives slightly crisper,
    # less-washed daytime frames through the glass -- a small but free win. Add more as you
    # find them, e.g. {"BACKLIGHT": 0, "CONTRAST": 40}. Empty {} = leave the camera's defaults.
    camera_controls: dict = field(default_factory=lambda: {"BACKLIGHT": 0})
    reopen_max_retries: int = 30    # On a read failure, how many reopen attempts before giving up.
    reopen_delay_s: float = 1.0     # Wait between reopen attempts (camera disconnect handling).

    # ---- Time-of-day camera profiles (sun-driven) -------------------------------
    # The live app selects a camera profile by SUN position, so "day" vs "night" tracks the
    # seasons automatically (local daylight runs ~8 h in winter to ~16 h in summer; fixed
    # clock hours would be wrong half the year). It re-applies the profile whenever the period
    # flips (around civil dawn / dusk). Tune each profile with tune.py AT THAT TIME OF DAY and
    # paste the values here. A profile is a settings dict: 'exposure' (None = auto-expose; a
    # number = locked, e.g. -7), 'gain' (None = auto), and any UVC control by NAME (e.g.
    # 'BACKLIGHT': 0). Defaults below match current behaviour (auto-expose, backlight off) for
    # BOTH periods, so nothing changes until you tune them. Set use_time_of_day_profiles=False
    # (or pass --exposure/--gain) to fall back to the static exposure/gain/camera_controls above.
    use_time_of_day_profiles: bool = True
    # Camera location for the sun-driven day/night profiles. Left unset here so a private
    # location never enters version control -- set it in config_local.py (gitignored; see
    # config_local.example.py). Unset = sun profiles disabled (the camera just auto-exposes).
    latitude: float | None = None
    longitude: float | None = None
    camera_profiles: dict = field(default_factory=lambda: {
        # Auto-expose BOTH periods so the camera's own auto-exposure adapts continuously through
        # the day -- the image tracks changing light (dawn / overcast / dusk) instead of being
        # frozen. We tried a locked daytime exposure (-8) to kill peak-sun patio clipping, but it
        # pinned the whole ~17 h daylight span at one value tuned for noon -- too dark exactly at
        # the crepuscular hours when most critters show. If midday direct-sun clipping bothers you,
        # bias auto darker (lower BRIGHTNESS, tuned with tune.py) rather than re-locking exposure.
        "day":   {"exposure": None, "gain": None, "BACKLIGHT": 0},
        "night": {"exposure": None, "gain": None, "BACKLIGHT": 0},   # tune with tune.py at night once a light's in
    })

    # ---- Motion gate (MOG2 background subtractor) -------------------------------
    # We only run the (expensive) detector on frames where motion exceeds a threshold.
    # MOG2 adapts to gradual outdoor light changes (drifting sun, passing clouds).
    motion_history: int = 500           # Frames of background "memory".
    motion_var_threshold: float = 16.0  # MOG2 sensitivity (lower = more sensitive).
    motion_detect_shadows: bool = True  # MOG2 marks shadows as gray(127); we drop them.
    motion_blur_ksize: int = 5          # Gaussian blur kernel before subtraction (denoise).
    # Trigger when the LARGEST motion blob exceeds this many pixels. Resolution-dependent:
    # at 1280x720 a visiting crow is comfortably a few hundred+ px. Raise to ignore small
    # twitches (leaves, distant motion); lower to catch smaller / further critters.
    motion_min_area: int = 800

    # ---- Detector (MegaDetector v6, run via Ultralytics) ------------------------
    # MegaDetector v6 (Microsoft AI for Good Lab). The official MDV6 weights are Ultralytics
    # YOLO models, so detector.py loads them directly via Ultralytics on the GPU -- a lean
    # alternative to the heavyweight PytorchWildlife wrapper (which drags in a gradio web UI
    # and an audio stack). Weights auto-download from Zenodo on first use. Valid versions:
    #   MDV6-yolov9-c / MDV6-yolov10-c / MDV6-rtdetr-c  -> compact, fast  (good for live)
    #   MDV6-yolov9-e / MDV6-yolov10-e                  -> heavy, 1280px, more accurate
    # yolov10-c is NMS-free and quick -> sensible default for a live rig on a laptop GPU.
    model_version: str = "MDV6-yolov10-c"
    # 'cuda' (default): require an NVIDIA GPU and fail loud if it can't compute (catches a
    # wrong-arch torch build). 'cpu': no GPU needed, slower -- fine here because the motion gate
    # only wakes the detector on real motion. 'auto': GPU if usable, else CPU. (--device overrides.)
    device: str = "cuda"
    # A detection must score at least this to come back from the detector (passed as
    # MegaDetector's det_conf_thres) -- so weaker boxes are never drawn or saved.
    min_confidence: float = 0.25
    # MegaDetector's coarse classes: 0=animal, 1=person, 2=vehicle. We DRAW every class in
    # the preview (wave a hand and watch it work), but only SAVE the classes listed here.
    # Default = animals only, so you sitting next to the camera don't fill crops/ with selfies.
    save_classes: tuple[str, ...] = ("animal",)
    # Don't run the detector more often than this while motion is continuous. Caps GPU load
    # and stops a lingering crow from writing a near-duplicate row on every single frame.
    detector_min_interval_s: float = 1.0
    # How long the last drawn boxes persist on the preview between detector runs (seconds).
    box_display_ttl_s: float = 1.0

    # ---- Output (crops mandatory, full frame optional) --------------------------
    db_path: Path = ROOT / "backyard.db"
    crops_dir: Path = ROOT / "crops"
    frames_dir: Path = ROOT / "frames"
    save_full_frame: bool = False       # Default OFF per spec; crops are ALWAYS saved.
    crop_padding: float = 0.08          # Expand each box by this fraction before cropping.
    jpeg_quality: int = 95              # 0-100 for saved JPEGs.

    # ---- Identity of this capture source (V1 constant) --------------------------
    # Written verbatim into detections.source. Future sources: 'trail_cam_sd', etc.
    source: str = "glass_door_cam"

    # ---- Live preview -----------------------------------------------------------
    show_preview: bool = True           # Required feature; press 'q' in the window to quit.
    window_name: str = "Backyard Critter Cam"

    # ---- Stats readout (`python backyard_cam.py --stats`) -----------------------
    # Two detections on the same source more than this many minutes apart count as separate
    # "visits". One lingering critter fires many crops, so visits (not crops) are the honest
    # activity unit. This is a V1 estimate -- true per-individual visits arrive with phase 3.
    visit_gap_minutes: float = 5.0

    # ---- Local web dashboard (optional, `--serve`) ------------------------------
    # A stdlib-only (no web framework) local page: live MJPEG stream + stats + crop gallery,
    # served from this same process. Bound to localhost by default -- it shows your camera
    # feed, so only expose it to the LAN deliberately (set web_host = "0.0.0.0").
    serve: bool = False
    web_host: str = "127.0.0.1"
    web_port: int = 8000
    web_jpeg_quality: int = 80   # JPEG quality for the live stream (lighter than saved crops).

    # ---- Live species naming (phase 2, folded into the live rig) -----------------
    # The live rig names each new crop by species ITSELF, in a background thread, so a single
    # process -- and a single window -- does both detection and naming. That's the whole reason
    # you can shut the rig down with one 'q' (or by closing the video window): there's no second
    # classifier console left running. Naming uses the CPU by default so it never competes with
    # the live detector for the GPU. Set classify_live = False (or pass --no-classify) to run
    # detection only; you can still fill species in later with `python classify.py`.
    classify_live: bool = True
    classify_device: str = "cpu"        # 'cpu' (default; no GPU contention) or 'cuda'.
    classify_interval_s: float = 5.0    # Seconds between checks for new crops to name.

    # ---- Timezone convention ----------------------------------------------------
    # Timestamps are stored as LOCAL time WITH the UTC offset, ISO 8601, e.g.
    #     2026-06-07T19:25:59.123456-07:00
    # Produced by datetime.now().astimezone().isoformat(): the wall-clock time you'd read
    # off a window-side clock, but the trailing offset keeps it globally unambiguous and
    # correctly sortable across DST changes. See db.now_local_iso().


# The single shared default instance. backyard_cam.py copies this and applies CLI overrides.
CONFIG = Config()

# Local, untracked overrides for private / machine-specific values (e.g. your camera's real
# latitude/longitude). Copy config_local.example.py to config_local.py (gitignored) and set
# them there, so they never enter version control. Applied last, over the defaults above.
try:
    import config_local  # type: ignore

    config_local.apply(CONFIG)
except ImportError:
    pass
