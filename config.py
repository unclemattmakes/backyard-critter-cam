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

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Built and tested on Python 3.14, but runs on 3.10+. Fail early with a plain message on older
# interpreters rather than later with a cryptic SyntaxError deep in a submodule (a modern f-string,
# say) -- baffling for a non-developer setting this up for the first time. config is imported by
# every entry point, so this one check covers them all.
if sys.version_info < (3, 10):
    raise SystemExit(
        "Backyard Critter Cam needs Python 3.10 or newer (built/tested on 3.14); you're running "
        f"{sys.version.split()[0]}. Install a newer Python and recreate the .venv (see README Setup)."
    )

# Project root = the folder containing this file. All default paths hang off it, so the
# whole rig (code + db + crops) is relocatable as one directory.
ROOT = Path(__file__).resolve().parent

# Where the live species-naming helper (classify.py --watch) writes its status, so the dashboard
# can show "warming up" vs "naming" vs "stopped". A hidden file in the project root.
NAMING_STATUS_FILE = ROOT / ".naming_status.json"


@dataclass
class CameraSpec:
    """One camera in a multi-camera rig.

    The live rig can watch several cameras at once -- a USB webcam at the glass door PLUS
    networked cameras around the yard -- each writing its own `source` into the DB so all the
    downstream phases (species ID, re-ID, behaviour, the dashboard) keep them separate. List them
    in `Config.cameras` (typically in config_local.py); leave it None to stay single-camera (the
    flat `camera_index`/`source` fields below are then used, unchanged).

    Only `source` and `src` are required; every other field defaults to None and INHERITS the
    matching Config value at runtime, so a spec stays terse:

        cfg.cameras = [
            CameraSpec("glass_door_cam", 0, name="Glass door"),              # USB webcam, index 0
            CameraSpec("yard_ir", "rtsp://user:pass@192.168.1.50:554/h264Preview_01_sub",
                       name="Yard (night IR)"),                              # Reolink PoE, sub-stream
            CameraSpec("feeder_esp32", "http://192.168.1.51:81/stream",
                       name="Feeder (ESP32)", motion_min_area=300),         # ESP32-CAM MJPEG
        ]

    `src` is whatever cv2.VideoCapture accepts: an int webcam index, or a URL string -- an RTSP
    URL for an IP/PoE camera (best for night: real IR + one-cable PoE), or an http .../stream
    MJPEG URL for an ESP32-CAM (a fun daytime angle; weak in the dark). A networked camera is
    opened through OpenCV's FFMPEG backend with a 1-frame buffer and reconnected with indefinite
    backoff (a network blip is transient; a USB unplug is not -- see backyard_cam.py)."""
    source: str                              # DB 'source' label; MUST be unique across cameras.
    src: int | str = 0                       # cv2.VideoCapture arg: int webcam index OR stream URL.
    name: str | None = None                  # Display name in the dashboard (defaults to `source`).
    # Everything below: None = inherit the same-named Config field at runtime.
    frame_width: int | None = None
    frame_height: int | None = None
    backend: str | None = None               # 'dshow' | 'ffmpeg' | 'any'; None = auto by src type + OS.
    exposure: float | None = None
    gain: float | None = None
    camera_controls: dict | None = None
    camera_profiles: dict | None = None
    use_time_of_day_profiles: bool | None = None
    # Motion-gate trigger area is RESOLUTION-DEPENDENT (pixels), so a lower-res networked cam wants
    # a smaller value than the 1280x720 default -- set it per camera here. None = inherit Config.
    motion_min_area: int | None = None
    record_clips: bool | None = None

    @property
    def is_url(self) -> bool:
        """True when `src` is a stream URL (a string) rather than an int webcam index."""
        return isinstance(self.src, str)

    @property
    def is_network(self) -> bool:
        """True for a networked stream (rtsp://, http://...): gets the FFMPEG backend, a 1-frame
        buffer, and indefinite-backoff reconnect. A plain local file path (str without '://')
        counts as non-network so a recorded-file test source reconnects like a local device."""
        return isinstance(self.src, str) and "://" in self.src

    @property
    def display_name(self) -> str:
        return self.name or self.source


@dataclass
class Config:
    # ---- Camera (OpenCV capture) ------------------------------------------------
    camera_index: int = 0           # USB webcam index (0 = first/default). Set yours in config_local.py; --list-cameras finds it.
    frame_width: int = 1280         # Requested width  (the webcam may snap to the nearest).
    frame_height: int = 720         # Requested height (the webcam may snap to the nearest).
    # CAP_DSHOW = DirectShow backend: fast init on Windows and avoids the slow MSMF path.
    # DirectShow is Windows-only, so this is automatically ignored off-Windows (Linux/macOS get
    # CAP_ANY and OpenCV picks the native backend) -- see capture_backend() in backyard_cam.py.
    use_dshow_backend: bool = True
    # Pixel-format + frame-rate REQUESTS for local webcams (None = leave the driver's pick).
    # DirectShow builds the capture mode from format+size+rate together, and with no format
    # request it can pick an uncompressed mode at a crawl: the 2026-07-19 replacement cam serves
    # 720p YUY2 at a flat 4 fps day AND night (measured from clip avg_frame_rate), where MJPG
    # runs full speed. Both are polite requests -- the driver picks the nearest supported mode,
    # an unsupported set is ignored, and a network stream ignores them entirely (the camera owns
    # its own encoding there).
    camera_fourcc: str | None = "MJPG"
    camera_request_fps: float | None = 30.0
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

    # ---- Multiple cameras (optional) --------------------------------------------
    # None = single-camera mode: the flat camera_index/source fields above are the one camera.
    # Set a list of CameraSpec to run SEVERAL cameras at once (USB + networked), each on its own
    # capture thread, all sharing one detector and one dashboard (a grid of live feeds). Define it
    # in config_local.py, e.g.:
    #   from config import CameraSpec
    #   cfg.cameras = [CameraSpec("glass_door_cam", 0, name="Glass door"),
    #                  CameraSpec("yard_ir", "rtsp://user:pass@192.168.1.50:554/h264Preview_01_sub",
    #                             name="Yard (night IR)")]
    # Each camera's `source` is written to its detections/clips, so stats/re-ID/behaviour keep the
    # cameras separate automatically. See CameraSpec (above) for the per-camera fields.
    cameras: list | None = None

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
    # Run the motion gate on a DOWNSCALED copy about this many pixels wide (the detector, crops
    # and clips all still see the full-resolution frame). Blur + MOG2 + contours cost scales
    # with pixel count, and at 1920x1080 a full-res gate held the whole capture loop to ~6 fps
    # (measured 2026-07-20). Blob areas are converted BACK to full-frame pixels before the
    # motion_min_area comparison, so that knob keeps meaning what it always meant at any
    # capture size. None/0 = gate at full resolution (the old behaviour).
    motion_gate_width: int | None = 640

    # ---- Detector (MegaDetector v6, run via Ultralytics) ------------------------
    # MegaDetector v6 (Microsoft AI for Good Lab). The official MDV6 weights are Ultralytics
    # YOLO models, so detector.py loads them directly via Ultralytics on the GPU -- a lean
    # alternative to the heavyweight PytorchWildlife wrapper (which drags in a gradio web UI
    # and an audio stack). Weights auto-download from Zenodo on first use. Valid versions:
    #   MDV6-yolov9-c / MDV6-yolov10-c / MDV6-rtdetr-c  -> compact, fast  (good for live)
    #   MDV6-yolov9-e / MDV6-yolov10-e                  -> heavy, 1280px, more accurate
    # yolov10-c is NMS-free and quick -> sensible default for a live rig on a laptop GPU.
    model_version: str = "MDV6-yolov10-c"
    # 'auto' (default): use the GPU when it genuinely computes, else fall back to CPU -- so the rig
    # runs out-of-the-box on a laptop with no NVIDIA GPU. 'cuda': REQUIRE a working GPU and fail
    # loud if it can't compute (catches a wrong-arch torch build -- the Blackwell sm_120 trap); set
    # device='cuda' in config_local.py if you want that strictness. 'cpu': force CPU. (--device overrides.)
    device: str = "auto"
    # A detection must score at least this to come back from the detector (passed as
    # MegaDetector's det_conf_thres) -- so weaker boxes are never drawn or saved.
    min_confidence: float = 0.25
    # Which detector classes to REPORT at all. MegaDetector's coarse classes are 0=animal,
    # 1=person, 2=vehicle. A glass-door cam keeps catching cars on the street as "vehicle",
    # which only clutters the live preview (the saved data is filtered separately by
    # save_classes). Listing a subset here tells Ultralytics to ignore the rest entirely --
    # they're never returned, drawn, or saved, at no extra cost (one YOLO pass scores every
    # class regardless). Keep "person" so the "wave a hand and watch it work" liveness check
    # still draws a box at the door. None = report every class (the old draw-everything behaviour).
    detect_classes: tuple[str, ...] | None = ("animal", "person")
    # Of the classes the detector reports, only these are SAVED (cropped + written to the DB).
    # Default = animals only, so a person at the glass door is boxed in the live view but never
    # fills crops/ with selfies. (Anything not in detect_classes never reaches here to begin with.)
    save_classes: tuple[str, ...] = ("animal",)
    # Don't run the detector more often than this while motion is continuous. Caps GPU load
    # and stops a lingering crow from writing a near-duplicate row on every single frame.
    detector_min_interval_s: float = 1.0
    # How long the last drawn boxes persist on the preview between detector runs (seconds).
    box_display_ttl_s: float = 1.0
    # Ignore zones: persistent STATIC false-fire spots, per camera source. A scene can hold a
    # patch the detector reliably misreads as an animal -- on this rig, a dark opening in the
    # retaining wall scored "animal" 0.25-0.7 on every run for a whole dusk (2026-07-20: 170+
    # junk rows, later force-named as birds by the species step). The motion gate is no defense:
    # it only decides WHEN the detector runs, and anything else moving (a real visitor, dusk
    # light-fade) makes the stateless full-frame detector re-report the patch. List such spots
    # here -- {source: [(x1, y1, x2, y2), ...]} in FULL-RES frame pixels -- and any detection
    # whose box mostly overlaps one (IoU >= ignore_zone_iou) is dropped before drawing, saving,
    # and clip-triggering. The IoU gate keeps it surgical: a real animal passing THROUGH a zone
    # has a much bigger box (IoU stays tiny) and is kept; only a zone-sized box sitting ON the
    # zone is dropped. Zones are framing-specific: set them in config_local.py and re-measure
    # after physically moving a camera (zones are drawn as faint gray boxes on the preview so a
    # stale one is visible). None/{} = no zones.
    ignore_zones: dict | None = None
    ignore_zone_iou: float = 0.45

    # ---- Output (crops mandatory, full frame optional) --------------------------
    db_path: Path = ROOT / "backyard.db"
    crops_dir: Path = ROOT / "crops"
    frames_dir: Path = ROOT / "frames"
    save_full_frame: bool = False       # Default OFF per spec; crops are ALWAYS saved.
    crop_padding: float = 0.15          # Expand each box by this fraction before cropping (more
                                        # breathing room around the animal). Applies to NEW captures
                                        # only -- existing crops were already cut at the old pad.
    jpeg_quality: int = 95              # 0-100 for saved JPEGs.

    # ---- Content backups (backup.py) ---------------------------------------------
    # Where backup.py archives everything the rig GENERATES: per-day zips of clips/ and crops/
    # (clips especially -- clips_max_gb prunes the live folder, the backup outlives it), plus
    # integrity-checked db snapshots. Machine-specific, so set it in config_local.py -- point it
    # at a folder your cloud client syncs (Google Drive / Dropbox / OneDrive) and the upload
    # takes care of itself. None = backup.py refuses to run until told where (or --dest).
    backup_dest: Path | None = None

    # ---- Identity of this capture source (V1 constant) --------------------------
    # Written verbatim into detections.source. Future sources: 'trail_cam_sd', etc.
    source: str = "glass_door_cam"

    # ---- Live preview -----------------------------------------------------------
    show_preview: bool = True           # Required feature; press 'q' in the window to quit.
    window_name: str = "Backyard Critter Cam"
    # Native preview window size relative to the capture resolution. imshow re-uploads the
    # whole image every UI tick, which at full 1080p is a real slice of the main thread; a
    # half-size window is visually identical on a laptop screen and much cheaper. 1.0 = native.
    preview_scale: float = 0.5

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
    # Cap how often the capture thread annotates + JPEG-encodes a display frame (dashboard
    # stream AND native preview share the cap). A 1080p encode is one of the loop's biggest
    # per-frame costs, and above ~12 fps a monitoring view gains nothing. Frames are encoded
    # for the dashboard ONLY while someone is actually watching (a stream client is connected
    # or a snapshot was just requested) -- an unwatched rig spends nothing on encoding. This
    # never throttles capture itself: detection, crops and clips run at full camera rate.
    display_max_fps: float = 12.0
    # When the dashboard is bound to the network (web_host = "0.0.0.0", the LAN launcher), accept
    # connections ONLY from your local network -- loopback + private ranges (192.168.x, 10.x,
    # 172.16-31.x, link-local). A DIRECT request from a public internet address is refused (HTTP
    # 403), and the Host header is validated so a malicious website you visit can't use DNS
    # rebinding to reach the rig from your own browser. That blocks the common exposure paths --
    # but it is NOT a login: anyone already ON your Wi-Fi has full access, INCLUDING label edits.
    # So don't port-forward it or put it on an untrusted network; for real remote access, front it
    # with your own VPN or an authenticating reverse proxy. No effect when bound to localhost
    # (already machine-only). Set False only if you deliberately front it with your own auth/VPN.
    lan_only: bool = True

    # ---- Behaviour clips (phase 4 capture: short video around each visit) --------
    # Stills capture WHO and WHEN; a short VIDEO clip captures HOW -- gait, approach speed,
    # dwell, vigilance, who-defers-to-whom. Motion is the behaviour signal (and a confound-robust
    # second shot at individual ID: a limp reads the same from any angle). The recorder keeps a
    # rolling pre-roll buffer so a clip includes the animal ARRIVING, then writes until
    # clip_post_roll_s after the last detection (or the clip_max_s safety cap). Stills are
    # still saved alongside -- clips are additive, and the crops still feed species ID.
    # ON by default (2026-06-09) so the rig banks motion data for clipmotion.py / gait work by
    # itself; safe to leave on because clips_max_gb prunes the OLDEST clips to a disk budget
    # (measured ~0.44 MB/s of active footage -> a busy night ~0.8 GB; 10 GB = a rolling ~2
    # weeks). Turn off per-run with --no-record-clips, or permanently here.
    record_clips: bool = True
    clips_max_gb: float = 10.0          # disk budget for clips/; oldest pruned past this (0 = no cap)
    # Per-SOURCE disk budgets, overriding clips_max_gb for the named source only. The rolling
    # window is per source because the sources are not equally replaceable: the live rig can
    # always record tomorrow, but a trail-cam clip exists only until the SD card is formatted.
    # With one shared budget, importing a card's videos evicted an equal slice of glass-door
    # footage (oldest-first, source-blind) -- 9 GB of trail cam would have emptied a 10 GB pool.
    # Key by source string exactly as it appears in clips.source; unlisted sources share
    # clips_max_gb. 0 = that source is never pruned.
    clips_max_gb_by_source: dict[str, float] = field(
        default_factory=lambda: {"trail_cam_sd": 15.0})
    clips_dir: Path = ROOT / "clips"
    # Representative frame-crops of clip tracklets (clipembed.py), used as un-blend UI thumbnails.
    # Gitignored cache, like crops/clips; safe to delete (regenerated by `clipembed.py --redo`).
    clip_crops_dir: Path = ROOT / "clip_crops"
    # Which detector classes START/extend a clip. None = same as save_classes (animals), so a
    # person at the glass is boxed live but never recorded. Decoupled from save_classes on
    # purpose: set ("animal", "person") to also capture human approaches WITHOUT saving person
    # crops -- which is exactly how you test the recorder yourself (walk by, no DB selfie).
    clip_classes: tuple[str, ...] | None = None
    clip_pre_roll_s: float = 3.0        # seconds of buffered pre-detection footage to prepend
    clip_post_roll_s: float = 3.0       # keep recording this long after the last detection
    clip_max_s: float = 60.0            # hard cap: a camped-out raccoon can't make a giant file
    clip_fps: float | None = None       # None = measure the live capture rate; or force a value
    # 0 < scale <= 1: downscale recorded frames. 1.0 = full resolution. Lower (e.g. 0.5) cuts BOTH
    # the in-RAM pre-roll buffer (~Nx720p frames) and the file size; bump down if memory is tight.
    clip_scale: float = 1.0
    # Clip video codec. 'h264' records straight to browser-playable H.264 by piping frames to
    # ffmpeg (libx264): ~half the size of mp4v, plays in any <video>, and cv2 still reads it for
    # clipmotion. Falls back to OpenCV's 'mp4v' writer automatically if ffmpeg isn't on PATH. Set
    # to an OpenCV fourcc ('mp4v', 'XVID', ...) to force the cv2 writer instead -- but note mp4v is
    # NOT browser-playable, so the dashboard has to transcode those on the fly (see web.py).
    clip_codec: str = "h264"

    # ---- Individual suggestions (phase 3, the confirm-or-correct loop) -----------
    # The suggestion engine (individuals.py) compares a visit's appearance PROTOTYPE (mean of its
    # best crops' embeddings) against every HUMAN-CONFIRMED visit, nearest-visit-first. Validated
    # on real data 2026-06-11: single crops can't match across sessions (~0.5 ceiling, see
    # reid.py), but visit prototypes recover it (same-animal cross-night matches at 0.83-0.93 vs
    # ~0.3-0.45 between different night-blocks), and two different raccoons IN THE SAME FRAME
    # score only ~0.36-0.42 -- so a high prototype match is meaningful. Every confirmation adds a
    # template, so suggestions sharpen as the cast gets named ("gets better over time").
    reid_proto_top_k: int = 40          # best crops (by crop_quality) averaged into a prototype
    reid_proto_min_crops: int = 3       # fewer embedded crops than this = too thin to suggest on
    reid_suggest_min_conf: float = 0.5  # embedding gate: crops below this confidence don't vote
    # Best-match similarity below this = "possibly someone new" (novelty flag). Set to the eval
    # optimum: eval.py measured same-vs-different individual separation at ROC-AUC 0.81 with the
    # Youden-J best operating point at cosine 0.31 (reports/eval_*.json, reid.separation.
    # best_threshold). The old 0.55 was over-conservative -- it wrongly flagged ~7 of 61 correctly-
    # matched raccoon visits as "possibly someone new". Since suggest-confirm still has a human
    # approve every match, the looser cut mostly just recovers true returnees.
    reid_novel_threshold: float = 0.31
    # >= this many frames with two separated raccoon boxes = a multi-animal visit: it gets a
    # "2+ raccoons" badge and its (blended) prototype never becomes a suggestion template.
    reid_co_presence_min: int = 3
    # CLIP-based co-presence (clipmotion.py tracklets) is a SECOND, more sensitive multi-animal
    # signal: the full-frame-rate clips catch the second raccoon walking apart in moments the
    # sparse live detections miss (validated 2026-06-11: clips flagged 6 pair visits the
    # detection badge missed, all in the pair's time window). A visit counts as multi if THIS
    # many of its clips each hold >= 2 sustained tracklets. 2 (not 1) so a lone fragmentation
    # artifact can't wrongly exclude a good solo template; clip-only flags need corroboration.
    reid_clip_co_presence_min_clips: int = 2
    # CLIP-SPACE appearance match: once a pair member is un-blended (its clip tracklets labelled),
    # those tracklets form a CLIP-space template that finds it in new visits -- the only way a
    # never-solo animal (Elliot) becomes recognizable. Clip vectors sit in a different, lower
    # similarity regime than the still prototypes, so they need their OWN threshold (measured
    # 2026-06-16: same-individual clip<->clip 0.52 / cross-space still<->clip-centroid 0.65, vs
    # ~0.10-0.18 for different individuals -- both separate cleanly at 0.40). Distinct from the
    # still-still novelty cut (reid_novel_threshold 0.55).
    reid_clip_match_threshold: float = 0.40
    # AUTO-ASSIGN (the "review by exception" tier): the nightly batch (run_clipmotion.bat ->
    # individuals.py --auto-assign) names a solo visit AUTOMATICALLY when its best match clears
    # BOTH bars: nearest-confirmed-visit similarity >= reid_auto_threshold AND lead over the
    # runner-up individual >= reid_auto_margin. Auto names are stamped individual_source='auto':
    # they show on tracking surfaces (rollcall, per-individual pages) but NEVER feed the
    # suggestion templates and never ground behaviour links (clipmotion --link is human-only) --
    # a wrong auto name can't teach the matcher or contaminate ground truth. The dashboard queue
    # shows each as "auto: <name>" with one-tap promote (-> human) / reject (-> never re-named).
    # 0.0 DISABLES the pass. Don't guess values: run `python eval.py --reid` -- its auto-assign
    # sweep recommends the max-coverage operating point with ZERO wrong names and ZERO novel-
    # animal false-accepts on your own confirmed corpus (re-run it as the cast grows).
    # Current values = the sweep on 2026-07-17's corpus (113 LOO probes over a 5-raccoon cast,
    # reports/eval_20260718T030508Z.json): named 17/113 (15%) with zero errors. Deliberately
    # conservative -- the confident head of the distribution, not the whole queue.
    reid_auto_threshold: float = 0.76
    reid_auto_margin: float = 0.12

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

    # ---- Non-animal prefilter (general CLIP, runs BEFORE BioCLIP) ----------------
    # BioCLIP is organism-only: it can't say "that's not an animal", so MegaDetector's coarse
    # 'animal' false-fires (a plate of food, a pet bowl, bare ground at the glass door) get forced
    # onto the nearest real species -- historically they piled up as "brown rat" (none real). This
    # gate runs a GENERAL CLIP (open_clip, already installed for BioCLIP) on each crop FIRST and
    # asks the one question BioCLIP can't: "is this even an animal?" Crops it judges non-animal are
    # labelled clipfilter.NONANIMAL_LABEL (in stats._NON_CRITTER, so the digest/dashboard hide
    # them) and skip BioCLIP entirely. Zero-shot -- prompt lists live in clipfilter.py, edit freely.
    nonanimal_filter: bool = True
    # open_clip checkpoint. ViT-B-32 is the smallest sane choice, so it adds little memory next to
    # BioCLIP on the CPU naming helper (~0.6 GB). Heavier (ViT-L-14) is sharper but much larger.
    nonanimal_model: str = "ViT-B-32"
    nonanimal_pretrained: str = "laion2b_s34b_b79k"
    # Reject (label non-animal) only when CLIP puts at least this much probability mass on the
    # non-animal prototype vs the animal one. Deliberately > 0.5 so the gate is conservative -- it
    # should fire only when quite sure, never shave off a genuine (if odd-looking) critter. Tune
    # with `python clipfilter.py --sample` (prints a threshold sweep over the real crops).
    # Validated 2026-06-17 on this DB: 0.50 wrongly hid 5% of human-VERIFIED animals; 0.55+ kept
    # verified animals AND the dog at 0% while still catching ~56% of the food false-fires, so 0.60
    # sits with margin above that cliff. The residual ~2% low-confidence (det 0.25-0.45) collateral
    # is flat across 0.55-0.65 (and some of it is non-animal junk anyway).
    nonanimal_threshold: float = 0.60

    # ---- Per-yard overrides (species & re-ID focus) -----------------------------
    # These default to the Pacific-Northwest backyard starter lists baked into classify.py /
    # clipfilter.py. Override them for YOUR yard in config_local.py -- plain lists of strings, no
    # source edit (so a typo can't break a module import). species_labels is the zero-shot
    # candidate set BioCLIP forces every crop onto; the two prompt lists drive the general-CLIP
    # "is this an animal?" gate. None = use the module's built-in default.
    species_labels: list | None = None
    clip_animal_prompts: list | None = None
    clip_nonanimal_prompts: list | None = None
    # Which species the dashboard's Individuals (re-ID) tab works on by default. 'raccoon' suits a
    # PNW yard; set it to your most-identifiable mammal (opossum, fox, ...) in config_local.py.
    reid_species: str = "raccoon"

    # ---- Timezone convention ----------------------------------------------------
    # Timestamps are stored as LOCAL time WITH the UTC offset, ISO 8601, e.g.
    #     2026-06-07T19:25:59.123456-07:00
    # Produced by datetime.now().astimezone().isoformat(): the wall-clock time you'd read
    # off a window-side clock, but the trailing offset keeps it globally unambiguous and
    # correctly sortable across DST changes. See db.now_local_iso().

    def camera_specs(self) -> "list[CameraSpec]":
        """The cameras to run, as a list of CameraSpec -- ALWAYS at least one. With `cameras` set
        it's returned as-is; otherwise a single spec is synthesized from the flat single-camera
        fields, so the legacy one-camera path is just the N=1 case of the multi-camera one (no
        separate code path, and every existing config keeps working untouched)."""
        if self.cameras:
            return list(self.cameras)
        return [CameraSpec(
            source=self.source, src=self.camera_index, name=None,
            frame_width=self.frame_width, frame_height=self.frame_height,
            backend=("dshow" if self.use_dshow_backend else "any"),
            exposure=self.exposure, gain=self.gain,
            camera_controls=self.camera_controls, camera_profiles=self.camera_profiles,
            use_time_of_day_profiles=self.use_time_of_day_profiles,
            motion_min_area=self.motion_min_area, record_clips=self.record_clips,
        )]


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
