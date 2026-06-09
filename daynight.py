"""
Time-of-day -> camera-profile selection, driven by the SUN (not a fixed clock) so day/night
switching tracks the seasons automatically. Uses `astral` for civil dawn/dusk.

Default periods: "day" (between civil dawn and civil dusk -- i.e. while there's usable light)
and "night". backyard_cam applies the matching entry from config.camera_profiles and
re-applies it whenever the period changes.
"""
from __future__ import annotations

from datetime import datetime


def current_period(latitude: float, longitude: float, when: datetime | None = None) -> str:
    """'day' (between civil dawn and civil dusk) or 'night', for the given location/time.
    Falls back to 'day' (= auto-exposure, the safe default) if the sun can't be resolved
    (e.g. extreme latitudes where civil twilight doesn't occur)."""
    when = when or datetime.now().astimezone()
    if when.tzinfo is None:
        when = when.astimezone()
    try:
        from astral import Observer
        from astral.sun import dawn, dusk
        obs = Observer(latitude=latitude, longitude=longitude)
        d, tz = when.date(), when.tzinfo
        return "day" if dawn(obs, d, tzinfo=tz) <= when <= dusk(obs, d, tzinfo=tz) else "night"
    except Exception:
        return "day"


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
