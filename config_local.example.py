"""Local, untracked config overrides.

Copy this file to `config_local.py` (which is gitignored) and set any private or
machine-specific values there. config.py imports it automatically after building CONFIG,
so whatever you set here overrides the defaults in config.py -- and never enters git.
"""


def apply(cfg):
    # Your camera's location, used only for the sun-driven day/night camera profiles.
    # Leave these unset (delete the lines) to disable sun profiles and simply auto-expose.
    cfg.latitude = 0.0    # decimal degrees, e.g. 40.7128
    cfg.longitude = 0.0   # decimal degrees, e.g. -74.0060

    # Power-user option (the default 'auto' already uses the GPU when it works, else CPU). Set
    # 'cuda' to REQUIRE a working NVIDIA GPU and fail loud if a wrong torch build can't use it,
    # rather than quietly running slow on the CPU -- handy on the main rig so a broken GPU shows.
    # cfg.device = "cuda"

    # Which webcam to open if index 0 isn't your USB cam. Find yours with:
    #   python backyard_cam.py --list-cameras
    # cfg.camera_index = 1

    # ---- Several cameras at once (USB + networked) ----------------------------------
    # Leave cfg.cameras unset for the single glass-door webcam above. To watch MORE of the yard,
    # add networked cameras here: each runs on its own capture thread, all share one detector, and
    # the dashboard shows a live grid. Each camera's `source` is written to its rows, so stats /
    # re-ID / behaviour keep the cameras separate automatically. (Tip for a nocturnal yard: a PoE
    # IR camera is the night workhorse; an ESP32-CAM is a fun, cheap DAYTIME angle -- weak in the dark.)
    #
    # from config import CameraSpec
    # cfg.cameras = [
    #     # The existing glass-door USB webcam (keep it as the primary).
    #     CameraSpec("glass_door_cam", 0, name="Glass door"),
    #     # A Reolink (or any RTSP/ONVIF) PoE camera -- use the lower-res SUB-stream for the motion
    #     # gate so decoding stays cheap. Put your camera's user/pass and LAN IP in the URL.
    #     CameraSpec("yard_ir", "rtsp://user:pass@192.168.1.50:554/h264Preview_01_sub",
    #                name="Yard (night IR)"),
    #     # An ESP32-CAM streaming MJPEG over HTTP (Arduino CameraWebServer). Lower-res, so give it a
    #     # smaller motion_min_area than the 1280x720 default (the trigger is in pixels).
    #     CameraSpec("feeder_esp32", "http://192.168.1.51:81/stream",
    #                name="Feeder (ESP32)", frame_width=640, frame_height=480, motion_min_area=300),
    # ]

    # Retune the species list + re-ID focus for YOUR yard (no source edit needed). The defaults are
    # a Pacific-Northwest backyard set; replace with your region's animals.
    # cfg.species_labels = ["raccoon", "red fox", "white-tailed deer", "American crow", "blue jay"]
    # cfg.reid_species = "raccoon"   # the species the dashboard's Individuals tab works on

    # A STATIC spot the detector keeps misreading as an animal (a hard shadow, a dark opening)?
    # List it per camera: a detection boxed ~exactly there is dropped before saving or
    # clip-triggering, while a real animal walking through the spot has a much bigger box and is
    # kept. Boxes are (x1, y1, x2, y2) in full-resolution frame pixels -- read them off the
    # repeated rows in the DB, or the preview. Full story at `ignore_zones` in config.py.
    # cfg.ignore_zones = {"glass_door_cam": [(1127, 595, 1234, 701)]}

    # Where backup.py archives the rig's generated content (clips/crops/db). Point it at a
    # folder your cloud client syncs (Google Drive, Dropbox, OneDrive) so uploads are automatic;
    # schedule `python backup.py` weekly. Clips especially want this: the live clips/ folder is
    # a rolling window (clips_max_gb) -- the backup is what outlives the pruning.
    # from pathlib import Path
    # cfg.backup_dest = Path(r"C:\cloud-synced-folder\backyard")
