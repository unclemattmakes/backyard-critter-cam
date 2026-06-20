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

    # Retune the species list + re-ID focus for YOUR yard (no source edit needed). The defaults are
    # a Pacific-Northwest backyard set; replace with your region's animals.
    # cfg.species_labels = ["raccoon", "red fox", "white-tailed deer", "American crow", "blue jay"]
    # cfg.reid_species = "raccoon"   # the species the dashboard's Individuals tab works on
