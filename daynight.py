"""
Time-of-day -> camera-profile selection, driven by the SUN (not a fixed clock) so day/night
switching tracks the seasons automatically. Uses `astral` for civil dawn/dusk.

Default periods: "day" (between civil dawn and civil dusk -- i.e. while there's usable light)
and "night". backyard_cam applies the matching entry from config.camera_profiles and
re-applies it whenever the period changes.
"""
from __future__ import annotations

import sys
from datetime import datetime

_warned_no_astral = False


def current_period(latitude: float, longitude: float, when: datetime | None = None) -> str:
    """'day' (between civil dawn and civil dusk) or 'night', for the given location/time.
    Falls back to 'day' (= auto-exposure, the safe default) if the sun can't be resolved
    (e.g. extreme latitudes where civil twilight doesn't occur).

    If `astral` isn't installed the sun profiles can't work at ALL -- without this guard the
    old broad except swallowed the ImportError and returned 'day' forever, silently disabling
    the whole feature. So we warn once (to stderr) on a missing astral, and keep the silent
    fallback only for the genuine can't-resolve-the-sun case."""
    global _warned_no_astral
    when = when or datetime.now().astimezone()
    if when.tzinfo is None:
        when = when.astimezone()
    try:
        from astral import Observer
        from astral.sun import dawn, dusk
    except ImportError:
        if not _warned_no_astral:
            _warned_no_astral = True
            print("[daynight] WARNING: 'astral' is not installed -- sun-driven day/night camera "
                  "profiles are DISABLED (defaulting to 'day' / auto-exposure). "
                  "Fix: pip install astral", file=sys.stderr)
        return "day"
    try:
        obs = Observer(latitude=latitude, longitude=longitude)
        d, tz = when.date(), when.tzinfo
        return "day" if dawn(obs, d, tzinfo=tz) <= when <= dusk(obs, d, tzinfo=tz) else "night"
    except Exception:
        return "day"  # sun undefined at this latitude/date -- safe default


def sun_times(latitude: float, longitude: float, when: datetime | None = None) -> dict:
    """Civil dawn / sunrise / sunset / civil dusk as local-time datetimes (for logging)."""
    when = when or datetime.now().astimezone()
    from astral import Observer
    from astral.sun import dawn, dusk, sunrise, sunset
    obs = Observer(latitude=latitude, longitude=longitude)
    d, tz = when.date(), when.tzinfo
    return {name: fn(obs, d, tzinfo=tz)
            for name, fn in (("dawn", dawn), ("sunrise", sunrise),
                             ("sunset", sunset), ("dusk", dusk))}
