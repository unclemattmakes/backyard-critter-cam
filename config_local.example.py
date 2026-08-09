"""Local, untracked config overrides.

Copy this file to `config_local.py` (which is gitignored) and set any private or
machine-specific values there. config.py imports it automatically after building CONFIG,
so whatever you set here overrides the defaults in config.py -- and never enters git.
"""


def apply(cfg):
    # Every override below is commented out, so a straight copy of this file changes nothing.
    # This `pass` is what keeps that legal Python -- a function whose whole body is comments
    # won't import. Leave it; uncomment the lines you actually want.
    pass

    # Your camera's location. It drives the sun-driven day/night camera profiles and the
    # dashboard's dawn/dusk buckets; left unset (as below) both simply switch off -- the rig
    # auto-exposes and the stats stop splitting Day from Night. Don't leave a placeholder 0/0
    # in: nothing treats it as "unset", and the Gulf of Guinea puts dawn at 22:40 local.
    # cfg.latitude = 40.7128    # decimal degrees
    # cfg.longitude = -74.0060  # decimal degrees, negative for west

    # The individual names you've given the humans in your household. Name yourself in the
    # dashboard and the re-ID queue stops offering you up as a new raccoon -- but that name is
    # then a label like any other, so list it here and makingof_export.py will never publish a
    # crop carrying it. Belongs in this untracked file, not in config.py. These names are also
    # folded into the non-critter denylist automatically, so you never show up as a rare species.
    # cfg.privacy_deny_names = ("yourname", "housemate")

    # Your yard's own additions to the non-critter denylist -- correction labels for YOUR
    # furniture and false triggers, in your own words or language ("bin", "wheelbarrow",
    # "la scopa"). They extend the built-in set (never replace it) and disappear from every
    # insight surface: species counts, Rarely Seen, the Behaviour notes, calendar glyphs.
    # cfg.non_critter_labels = ("bin", "wheelbarrow")

    # Household viewing vs curating (OFF unless set). With a token, devices that haven't entered
    # it (dashboard footer, once per browser) become VIEWERS: they read and play everything and
    # can log "who's here" as reviewable testimony, but every label/settings write is refused by
    # the server. Localhost is always the operator. Without this line, every device on your
    # Wi-Fi can edit labels -- fine alone, risky with houseguests' phones on the network.
    # cfg.operator_token = "pick-a-phrase"

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
    # The easy way: open the dashboard's Instrument Panel -> "Ignored Spots" and drag a box over
    # it -- no config edit, no restart. A detection boxed ~exactly there is dropped before saving
    # or clip-triggering, while a real animal walking through the spot has a much bigger box and
    # is kept. This config field still works but only SEEDS the dashboard's table (once per exact
    # rectangle; a spot deleted in the dashboard stays deleted). Boxes are (x1, y1, x2, y2) in
    # full-resolution frame pixels. Full story at `ignore_zones` in config.py.
    # cfg.ignore_zones = {"glass_door_cam": [(1127, 595, 1234, 701)]}

    # Running a SECOND source (a trail cam's SD card, another rig)? Give it its own slice of the
    # clips/ disk budget. Without this all sources share one pool pruned oldest-first, so importing
    # a full card evicts an equal amount of your live footage -- which is the wrong trade, because
    # the live rig can record tomorrow and the card gets formatted. Key by the source string
    # exactly as it appears in clips.source; a source you don't list shares the global
    # cfg.clips_max_gb. 0 = never pruned.
    # cfg.clips_max_gb_by_source = {"trail_cam_sd": 15.0}

    # Where backup.py archives the rig's generated content (clips/crops/db). Point it at a
    # folder your cloud client syncs (Google Drive, Dropbox, OneDrive) so uploads are automatic;
    # schedule `python backup.py` weekly. Clips especially want this: the live clips/ folder is
    # a rolling window (clips_max_gb) -- the backup is what outlives the pruning.
    # from pathlib import Path
    # cfg.backup_dest = Path(r"C:\cloud-synced-folder\backyard")
