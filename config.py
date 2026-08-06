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

    # ---- Auto-white-balance recovery --------------------------------------------
    # Some webcams have a BROKEN manual white balance. Measured on the 2026-07 glass-door cam
    # (2026-07-25): with AUTO_WB off, EVERY WB_TEMPERATURE from 2800 K to 6500 K renders red
    # ~40% below green (R/G 0.34-0.70 across the whole sweep) -- the cyan/green cast that made
    # daytime crops look washed out and flat. Its AUTO white balance is fine (R/G ~0.98), and
    # re-asserting AUTO_WB=1 pulls it back within ~2 s. But anything that writes a manual WB
    # value drops it into that broken state for the REST of the session -- including the
    # dashboard's own WB slider, which posts {AUTO_WB: 0, WB_TEMPERATURE: v} by design. Half the
    # sampled hours of 2026-07-21..25 sat in the bad state.
    # So: sample the frame's red/green ratio periodically and, when it sits in the red-starved
    # band, put the camera back into auto. Re-asserting auto is a no-op when the camera is
    # already there, so a false positive costs nothing but a log line.
    # A DELIBERATE manual choice is honoured: once the dashboard sends AUTO_WB=0 the watchdog
    # stands down for that camera until auto white balance is switched back on.
    wb_auto_recover: bool = True
    # Trip below this frame-wide red/green ratio. The two states are far apart (good ~0.85-1.10,
    # broken ~0.57-0.72), so anywhere in the gap works; 0.78 sits in the middle of it.
    wb_recover_ratio: float = 0.78
    wb_recover_interval_s: float = 20.0   # How often to sample the ratio (cheap: a strided mean).
    # Don't judge white balance on a frame too dark to carry colour -- a night scene under an
    # amber yard light is LEGITIMATELY far from neutral, and that's not a fault to correct.
    wb_recover_min_luma: float = 40.0
    # Consecutive bad samples before acting, so one odd frame (a red-brown animal filling the
    # view, headlights sweeping the fence) can't trigger it.
    wb_recover_strikes: int = 2

    # ---- Power + USB-wedge guard (see powerguard.py) ----------------------------
    # The keep-awake only wins on AC: on battery, Windows Modern-Standby-naps at the DC idle
    # timeout and suspend-cycles the USB cam until its stream WEDGES -- still delivering frames,
    # but torn garbage (2026-07-29..31, three evenings running). Warn hard while on battery.
    power_warn: bool = True             # console + HUD + dashboard warning while on battery
    power_poll_s: float = 30.0          # how often to sample AC/battery state
    power_warn_repeat_s: float = 600.0  # re-print the console warning this often while on DC
    # Wedge detection. Two observed signatures, either sufficient:
    #  * the WB watchdog's recovery didn't take, repeatedly (R/G stays pinned through AUTO_WB
    #    re-asserts; a real manual-WB trap recovers in ~2 s) -- the 07-30 "striped static";
    #  * the largest motion blob pins near the whole frame for a sustained stretch while
    #    MegaDetector finds NOTHING in it -- the 07-31 "torn slabs". (A close animal filling
    #    the view is excluded: it gets detected, and any detection vetoes the trigger.)
    wedge_guard: bool = True
    wedge_wb_failures: int = 2          # consecutive failed WB recoveries -> wedged
    wedge_motion_frac: float = 0.55     # largest blob >= this fraction of the frame ...
    wedge_motion_sustain_s: float = 60.0  # ... for this long, detection-free -> wedged
    wedge_clear_s: float = 30.0         # signature-free this long -> healthy again
    # Self-heal: fire the elevated scheduled task (registered ONCE by setup_selfheal.bat) that
    # pnputil disable/enable-cycles the camera -- the software version of the unplug/replug that
    # cured the 07-30 wedge. Unregistered/off, the rig still detects + banners the wedge; it
    # just can't fix it alone. Budgeted so a persistent fault can't churn the hardware.
    wedge_self_heal: bool = True
    wedge_heal_task: str = "BackyardCritterCam-UsbReset"
    wedge_heal_verify_s: float = 90.0   # grace after a reset before re-judging the signature
    wedge_heal_max_per_hour: int = 2    # then stop resetting and ask for a human replug

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

    # ---- Species vote for a visit (visits.py) -----------------------------------
    # A visit's species is voted on by its crops. The original vote counts crops, so VOLUME beats
    # CERTAINTY: a static artifact firing 800 crops at 0.3 confidence outvotes a real animal's 100
    # crops at 0.9. Turning this on sums species_confidence instead, over the crops that clear
    # species_vote_min_confidence.
    # SHIPPED OFF, deliberately. Species GATES re-identification -- a raccoon relabelled to opossum
    # drops out of every appearance template and can never be matched to a named individual again --
    # so a mass relabel is not something a clone of this repo should get by surprise. Look before
    # you leap:
    #     python visits.py --species-vote-report      # dry run, read-only, writes nothing
    # then set this in config_local.py if the flipped set looks right for YOUR corpus. (On the
    # author's DB, 2026-08-05: 108 of 2,416 visits move, 12 of them into raccoon.)
    species_vote_confidence_weighted: bool = False
    # Confidence a crop needs to join the weighted vote. Only consulted when the weighted vote is
    # on; if NO crop in a visit clears it, that visit falls back to the ungated weighted vote rather
    # than losing its species.
    species_vote_min_confidence: float = 0.8

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

    # ---- Certified identity reference batches (refcam.py) ------------------------
    # Photos a HUMAN vouched for -- one folder per individual, "this batch is <name>" -- shot on a
    # phone or any other camera, before any model has an opinion. Every individual label in the
    # database was otherwise produced by a human AGREEING with what the embedding proposed, and
    # docs/identity-eval-2026-08-05.md says that circularity caps the trustworthiness of every
    # accuracy number in it. These break the circle, and unlike the appearance embedding (top-1
    # 0.818 -> 0.222 as the probe-to-template gap grows to 21 days) they don't decay.
    # They are GROUND TRUTH FOR EVALUATION AND TRAIT WORK, never re-ID gallery templates: a phone
    # at arm's length in daylight is a different domain from the night webcam, and the maximum
    # measured cross-source similarity in this project is 0.363. refcam.py keeps them in their own
    # tables with their own `source` string and never writes to `detections`.
    #   reference_dir        -- the drop folder: reference/<name>/*.jpg
    #   reference_crops_dir  -- where the OPTIONAL detector crops of those photos are written,
    #                           deliberately NOT crops/ (that tree is the pipeline's, foldered by
    #                           capture date and zipped per-day by backup.py). Regenerable cache.
    reference_dir: Path = ROOT / "reference"
    reference_crops_dir: Path = ROOT / "reference_crops"

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
    # optimum: eval.py's Youden-J best operating point on same-vs-different individual separation
    # (reid.separation.best_threshold in the report it writes). The old 0.55 was over-conservative
    # -- it wrongly flagged ~7 of 61 correctly-matched raccoon visits as "possibly someone new".
    # Since suggest-confirm still has a human approve every match, the looser cut mostly just
    # recovers true returnees. Separation itself got HARDER as the cast grew: ROC-AUC was 0.81
    # over two raccoons, and 0.635 (top-1 0.637 over 113 leave-one-out probes) on the 2026-07-18
    # run with five. That drop is the honest number -- more lookalike raccoons means more
    # confusable pairs, not a regression -- and the best threshold barely moved (0.29). Re-run
    # `python eval.py --reid` on your own corpus rather than trusting either figure.
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
    # still-still novelty cut (reid_novel_threshold 0.31).
    reid_clip_match_threshold: float = 0.40
    # STILL-TRACKLET un-blend (the multi-animal splitter, 2026-07-31): when a visit holds 2+
    # animals, its detections are chained into per-animal TRACKLETS (time + box continuity), each
    # tracklet's crops are averaged into a mini-prototype, and the tracklets are clustered into
    # the individual animals -- with a hard cannot-link constraint (two boxes in the SAME frame are
    # never the same animal, no threshold needed). Measured on this corpus (2026-07-31, n=492/2319
    # raccoon pairs): two co-present raccoons score median 0.19 at crop level while one animal
    # frame-to-frame scores 0.72 -- the same separation that makes clip un-blend work, available
    # even when a visit has no clips (trail-cam photo cycles, pruned footage).
    # Max seconds a tracklet may bridge between saved crops. Conservative on purpose: a fragment
    # re-merges by appearance in the clustering step, but a chain that runs across two animals
    # poisons its prototype and nothing downstream can split it again. 20 s covers the usual
    # still cadence; the 26 s gap that separated Stan's wall-frames on 2026-07-31 stays split.
    reid_track_link_gap_s: float = 20.0
    # A still-tracklet cluster's centroid vs the confirmed-visit templates: suggestions show at or
    # above this cosine. Its own regime, like the clip threshold: a tracklet averages 5-30 crops
    # from ONE session, so it sits between crop-level noise (~0.5 ceiling) and full visit
    # prototypes (0.83+). 0.55 is a provisional cut pending an eval.py sweep -- suggestions are
    # human-confirmed, so a loose value costs a click, not contamination.
    reid_track_match_threshold: float = 0.55
    # AUTO-ASSIGN (the "review by exception" tier): the nightly batch (run_clipmotion.bat ->
    # individuals.py --auto-assign) names a solo visit AUTOMATICALLY when its best match clears
    # BOTH bars: nearest-confirmed-visit similarity >= reid_auto_threshold AND lead over the
    # runner-up individual >= reid_auto_margin. Auto names are stamped individual_source='auto':
    # they show on tracking surfaces (rollcall, per-individual pages) but NEVER feed the
    # suggestion templates and never ground behaviour links (clipmotion --link is human-only) --
    # a wrong auto name can't teach the matcher or contaminate ground truth. The dashboard queue
    # shows each as "auto: <name>" with one-tap promote (-> human) / reject (-> never re-named).
    # 0.0 DISABLES the pass, and that is what SHIPS: an operating point is a property of one
    # corpus, one camera and one cast, so the default must not put machine-written names on a
    # stranger's animals. Don't guess values: run `python eval.py --reid` -- its auto-assign sweep
    # recommends the max-coverage operating point with ZERO wrong names and ZERO novel-animal
    # false-accepts on your own confirmed corpus (re-run it as the cast grows), then set the two
    # bars here. For reference, this author's own numbers are 0.76 / 0.12 -- the sweep on
    # 2026-07-17's corpus (113 LOO probes over a 5-raccoon cast) named 17/113 (15%) with zero
    # errors. Deliberately conservative even there: the confident head of the distribution, not
    # the whole queue. The margin below is inert while the threshold is 0.
    reid_auto_threshold: float = 0.0
    reid_auto_margin: float = 0.12
    # PER-INDIVIDUAL TEMPLATE FLOOR (2026-08-05 identity evaluation, phase A2). auto_assign will
    # not WRITE a name backed by fewer than this many confirmed SOLO visits on the same camera; it
    # skips the visit with reason 'thin_templates' (visible in --auto-assign --dry-run). A thin
    # individual still competes in the ranking -- it can still block an assignment as the
    # runner-up -- it just never gets written. Why: a corpus like this one splits into a few
    # well-observed individuals (tens of confirmed solo visits each) and a tail seen only a handful
    # of times -- and the auto tier had already machine-named visits for two tail individuals off a
    # SINGLE template each. A threshold swept on one visit is not a measurement, and a wrong auto
    # name spends the scarcest thing here: the human's attention on the review queue. 5 ships as the
    # cautious side of that measurement (it excludes the whole thin tail); lower it in
    # config_local.py only with a sweep that says the thin individuals hold up. Inert while
    # reid_auto_threshold is 0. (Names and per-individual counts live in the DB, never here: this
    # file ships to strangers.)
    reid_auto_min_templates: int = 5
    # REVIEW-QUEUE MODES (web.py's "Who is this?" panel). Human confirmations are the only source
    # of templates, so the queue is the multiplier on everything above it -- but a newest-first
    # window reaches a few dozen of hundreds of candidates. The queue therefore offers filters
    # over the whole pool (recent / unreviewed_auto / ambiguous / stale) and paginates. Two knobs,
    # both generic and both safe as shipped:
    # STALE: a visit whose top candidate's newest human template is at least this many days old.
    # 14 comes from the shape of the decay curve, not from one corpus -- leave-one-visit-out top-1
    # measured here fell 0.818 -> 0.482 -> 0.222 as the newest allowed template aged 0 -> 7 -> 21
    # days, so a two-week-old template is already near the majority-class baseline. Re-measure on
    # your own corpus (`python eval.py --reid`) before trusting the exact number.
    reid_queue_stale_days: int = 14
    # AMBIGUOUS: the runner-up margin used to define "the machine can't call this one". Normally
    # reid_auto_margin, so the mode shows exactly what the auto tier refuses -- but that margin is
    # inert while reid_auto_threshold is 0.0 (the shipped default), and this mode is most useful
    # BEFORE the auto tier is turned on. This is the fallback used in that case.
    reid_queue_ambiguous_margin: float = 0.02

    # ---- ANATOMY TRAITS (traits.py) ---------------------------------------------
    # Descriptors measured off the animal's own body rather than off a global appearance embedding.
    # Nothing here is wired into the matcher: traits.py is a standalone extractor with a dry-run
    # CLI, and these knobs only decide when it says "I can't measure that here" instead of
    # returning a number. They are therefore inert as shipped.
    # WHAT IS LEFT, AND WHY SO LITTLE. Four hand audits on 2026-08-06 returned NO-GO on every
    # automatic identity trait proposed here -- tail rings (the tracer traced ears), tail thinness
    # and mask fade (the call flips with the light, not the animal), sitting posture (all three
    # adults do it) and the ear notch (1-3 px at this resolution). What survives is fur shading,
    # which measures a real fur tone and identifies NOBODY: session-blocked leave-one-visit-out
    # top-1 0.309 against a 0.345 majority baseline, inside a nuisance band whose controls score
    # 0.345 and 0.237, on a harness whose positive control reproduces 0.741. Keep it as a prior;
    # do not rank on it.
    # Every value below is a RATIO or a COUNT, except the review-panel resolution bound at the end
    # of this block, which is flagged where it appears. This camera's white balance runs on AUTO
    # (see the watchdog above) and its cameras get repositioned on purpose, so an absolute
    # threshold would silently encode the session.
    # Foreground mask (GrabCut, seeded from the crop's own inscribed ellipse, not a border rect --
    # the crop is a tight detector box, so its border is often animal).
    traits_mask_side: int = 160           # long edge of the internal working image, px
    traits_mask_iterations: int = 3       # GrabCut iterations; 3 is where the mask stops changing
    traits_mask_core_fraction: float = 0.45   # central ellipse seeded as definite foreground
    traits_mask_min_fg_fraction: float = 0.25  # below this the segmentation collapsed to "nothing"
    traits_mask_max_fg_fraction: float = 0.97  # above this it collapsed to "everything"
    # Does the mask outline sit on a real image edge? Median Sobel magnitude along the outline,
    # normalised by the crop's OWN 90th-percentile gradient, so a soft through-glass night frame
    # and a crisp one are judged on the same scale.
    traits_mask_min_edge_agreement: float = 0.25
    # GrabCut initialises its GMMs with k-means, which draws from OpenCV's GLOBAL RNG. Unseeded,
    # the same crop segments differently on every call -- which is how the deleted tail tracer came
    # to publish numbers nobody could reproduce. traits.py pins this before every segmentation.
    traits_rng_seed: int = 20260806
    # TOMBSTONE, 2026-08-06. The traits_tail_* knobs that used to live here are GONE, along with
    # the tracer they configured. It was not merely weak, it was WRONG: of 22 hand-audited traces,
    # 20 followed ears, legs, snouts or mask-bleed rather than tails, and its single
    # highest-confidence trace in the whole corpus followed an EAR while a ringed tail sat in plain
    # view outside the mask. A disabled-but-plausible extractor is worse than none, because the
    # next reader trusts its output. See traits.REMOVED_DISPROVEN, which re-raises the audit if
    # anybody imports the old names.
    # Fur shading. The trait is (L_p50 - L_p5) / (L_p95 - L_p5) inside the mask: where the fur sits
    # between the animal's own near-black (facial mask, shadowed underside) and near-white (muzzle,
    # lit guard hairs). Both references are ON THE ANIMAL and IN THIS CROP, so the auto-WB decision
    # cancels -- measured 2026-08-05 over 138 confirmed visits at corr 0.049 with the crop's R/G
    # and 0.075 with its mean luminance, while the ABSOLUTE version of the same idea scores BELOW
    # the majority baseline. This gate demands both references actually be present: L is 0..1, so
    # 0.15 asks for a 15% black-to-white span before the ratio means anything.
    traits_fur_min_span: float = 0.15
    # Fleck ("grey-flecked hair") texture scale, as a fraction of sqrt(mask area) so it tracks the
    # animal's apparent size rather than the frame. CONFOUNDED BY FOCUS AND MOTION BLUR -- emitted
    # so it can be measured, not because it is trusted.
    traits_fleck_scale: float = 0.02
    # A visit-level trait needs this many crops that cleared the gate before it is reported at all.
    traits_min_crops_per_visit: int = 3
    # BEHAVIOUR AXIS (traits.eating_style) -- DISPLAY / PRIOR ONLY, never a ranker input. Box
    # aspect: wide-and-low = head down at the food, tall = sitting back holding something. Kept
    # here, next to nothing that touches appearance, because box aspect is exactly the quantity an
    # appearance descriptor must never be normalised by.
    traits_eating_min_frames: int = 8
    traits_eating_low_aspect: float = 1.40
    traits_eating_upright_aspect: float = 0.95
    traits_eating_margin: float = 0.25    # how far apart the two fractions must be to call a style

    # ---- REVIEW AXIS (traits.head_review_panel / review_candidates) --------------
    # Added 2026-08-06, after four hand audits returned NO-GO on every automatic identity trait
    # proposed. The ear notch is the only trait among them whose failure is a failure of RESOLUTION
    # rather than of the idea -- a notch is permanent, binary and era-invariant, which is exactly
    # what an appearance embedding measured at 0.222 top-1 across a 21-day gap is not. So the build
    # is a magnifying glass for Matt, not a classifier: these knobs decide which crops are worth
    # his eyes and how far to trust them. NOTHING HERE RANKS ANYTHING, and traits.appearance_vector
    # refuses a review object by type.
    #
    # THE ONE ABSOLUTE-PIXEL NUMBER IN THE TRAITS BLOCK IS HERE AND IT IS DELIBERATE: human
    # readability is a property of the sensor and the optics, not of the yard, so no ratio can
    # express it. Everything else stays a fraction of the crop, so a camera move cannot void it.
    traits_review_ear_height_fraction: float = 0.065  # ear height / bbox height (measured 6-7%)
    # Hand-audited thresholds: an ear margin is MARGINALLY callable at ~25-30 px of ear (reached at
    # ~400 px of bbox height on this rig) and COMFORTABLY callable at ~50-70 px. Below the floor
    # traits.head_review_panel returns None -- not "no notch". A hidden ear is not an intact ear.
    traits_review_min_feature_px: float = 25.0
    traits_review_good_feature_px: float = 50.0       # readability reaches 1.0 here
    traits_review_head_band: float = 0.35             # top share of the crop the panel shows
    traits_review_target_px: int = 480                # rendered panel height, LANCZOS
    traits_review_max_magnification: float = 8.0      # past this it is upsampling, not detail
    # Reject a box that fills this share of the FRAME. Measured, not guessed: ordering the June
    # corpus by box height returns full-frame boxes first, one containing no animal at all and one
    # holding a raccoon that occupies a tenth of it. Their height is the FRAME's height, so the
    # readability estimate above would be pure fiction. A fraction of the frame, so it survives a
    # camera move.
    traits_review_max_frame_fraction: float = 0.50
    # STATIC FURNITURE REJECT. Two audits found the same trap from opposite ends: the five largest
    # boxes in a confirmed 2026-08-06 raccoon visit contain NO ANIMAL (a dark shrub re-detected at
    # 0.26-0.31 confidence over 17 minutes), and another confirmed visit's contact sheet was
    # dominated by a glass jar. Self-calibrating on purpose -- boxes are compared to EACH OTHER,
    # never to a stored zone, because the trail cam moved 384 px between two cycles and a
    # hand-drawn zone would have failed silently.
    # IT GROUPS ON PIXELS, NOT ON BOXES. Box geometry was tried twice and thrown away twice: plain
    # IoU FRAGMENTED one shrub into sub-threshold clusters, because the detector emits nested boxes
    # on it (a measured pair scores IoU 0.50 and containment 1.00), and four animal-free crops
    # reached the top of a review sheet; loosening to containment then MERGED that shrub with 120
    # genuine raccoon frames. Correlating the crops asks the question directly and needs no overlap
    # constant at all -- so there isn't one here.
    traits_static_min_detections: int = 5
    # A frozen scene is present for the WHOLE visit by definition, so the span test is a SHARE of
    # the visit's own span -- self-calibrating, no clock constant. An absolute 10 minutes was tried
    # first and silently exempted short visits: a 7-minute visit cannot contain a 10-minute group,
    # so all four of its review candidates came back furniture (opened, confirmed empty). The small
    # absolute floor stays underneath, to stop a 20-second visit of a briefly-still animal reading
    # as furniture -- real animal runs decorrelate at 0.235-0.551 across minutes but were measured
    # as low as 0.054 across a few consecutive frames.
    traits_static_min_span_fraction: float = 0.5
    traits_static_min_span_minutes: float = 2.0
    # Pixel confirmation: 1 - median normalised cross-correlation over small grey thumbnails.
    # Measured on this corpus -- animal-free clusters 0.005-0.058 (the shrub, and the glass jar the
    # posture audit found), real animal runs 0.235-0.551. The order of magnitude is the signal; the
    # cut sits in the empty middle, not on a knife edge.
    traits_static_max_dissimilarity: float = 0.10
    traits_static_thumb_side: int = 48

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

    # Individual names belonging to the humans in your household. The dashboard will let you name
    # yourself as an individual (handy -- it stops the re-ID queue offering you up as a new raccoon),
    # and that name is then a label like any other. makingof_export.py refuses to publish any crop
    # carrying one, on top of the generic person/people/human labels. Set it in config_local.py so
    # your name never rides along in a commit; a label-based filter can't catch a person the
    # detector mislabelled, so still review exported crops by eye.
    privacy_deny_names: tuple = ()

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
except ModuleNotFoundError as exc:
    # Having no config_local.py is the normal case, so that one is silent. But a bare
    # `except ImportError` also swallowed a ModuleNotFoundError raised from INSIDE the file:
    # one typo'd import in the user's own overrides and every override vanished without a
    # word, which is a miserable thing to debug. Only the absent module itself passes.
    if exc.name != "config_local":
        raise
else:
    config_local.apply(CONFIG)
