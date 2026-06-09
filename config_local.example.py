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
