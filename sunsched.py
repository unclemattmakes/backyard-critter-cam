"""Aim the nightly batch at the hour the sun blinds the rig, and keep it aimed as the season turns.

WHY
---
The batch used to run at a fixed 14:00 -- picked as "the activity trough". Measuring the trough
properly (2026-08-21, 903 empty-scene snapshots from refimg_store/glass_door_cam/day over 14 days)
put it somewhere better: the late-afternoon GLARE window, when the sun swings around to the
camera's bearing and the glass door washes the frame out. In that window the rig is not merely
idle, it is BLIND, so a batch that steals the GPU costs nothing at all.

    hour   glare index (mean luminance / sobel detail)   crop_quality   detections/h
    12:00  15.9                                          810            875
    15:00  22.6                                          465            454
    17:00  23.7                                          361            456
    18:00  25.7   <- worst                               310            270
    19:00  20.8                                          475            524

Sun azimuth was 253 deg at 17:00 and 265 deg at 18:00 that day, so the camera faces about W/WSW.
Raccoons -- the animals this whole re-ID pipeline is for -- are effectively absent 15:00-20:00 and
peak 21:00-05:00, so the window is an activity trough as well as an optical one.

WHY NOT JUST HARDCODE 17:30
---------------------------
Because it drifts straight out of the window. On 2026-08-21 sunset was 20:09 and the glare centre
sat at 17:25; by 10-02 sunset is 18:45, and by 10-30 the sun is below the horizon before it ever
reaches that bearing. The stable anchor is SUNSET, not the wall clock -- which is the same lesson
daynight.py already encodes for the day/night camera profiles, and the same lesson as
config.ignore_zones: a hand-measured constant fails silently when the world moves.

So the batch re-arms its own trigger every run, for the next day, from the sun.

USAGE
    python sunsched.py --show                # the next fortnight, and what it would set
    python sunsched.py --arm                 # set the task's start time for TOMORROW
    python sunsched.py --arm --date today    # ...or for today (used once, at install)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta

import config

# How far before sunset to start. The measured glare span on 2026-08-21 ran from about
# sunset-5h (15:00, index 22.6) to sunset-1.6h (18:30), worst at sunset-2.1h. 2.5h starts the
# batch just inside the leading edge and lets the ~1h run finish around the peak -- so if the box
# does bugcheck mid-batch, it does so during the deadest, blindest part of the rig's day.
SUNSET_OFFSET_H = 2.5

# Belt and braces for the far ends of the year: whatever the sun says, never schedule outside
# this. In late December sunset-2.5h is about 13:50 and in June about 18:40, so these clamps
# should never bite -- they exist so a bad latitude or a broken astral cannot park the batch at
# 03:00 on top of the raccoon peak.
EARLIEST = "11:00"
LATEST = "19:30"

TASK_NAME = "BackyardCritterCam-MotionTracks"


def target_time(when: date, tz=None) -> tuple[datetime, dict]:
    """The batch start time for `when`, plus the sun times it came from.

    `tz` defaults to THIS machine's local zone, which is the right answer in production: the rig
    is at the coordinates, and schtasks takes a local wall-clock time. It is a parameter so tests
    can pin it -- a UTC runner would otherwise put an August sunset after midnight and every
    glare-window assertion would be about a different day. (The same trap as the old `_sun` bug
    that asked the sun where the SERVER was and got dusk eight hours before dawn.)"""
    from astral import Observer
    from astral.sun import sun

    lat = getattr(config.CONFIG, "latitude", None)
    lon = getattr(config.CONFIG, "longitude", None)
    if lat is None or lon is None:
        raise SystemExit(
            "No latitude/longitude configured, so the sun cannot be resolved. Set them in "
            "config_local.py:\n    cfg.latitude = 40.7128\n    cfg.longitude = -74.0060"
        )
    tz = tz or datetime.now().astimezone().tzinfo
    s = sun(Observer(latitude=lat, longitude=lon), when, tzinfo=tz)
    t = s["sunset"] - timedelta(hours=SUNSET_OFFSET_H)
    lo = datetime.combine(when, datetime.strptime(EARLIEST, "%H:%M").time(), tzinfo=tz)
    hi = datetime.combine(when, datetime.strptime(LATEST, "%H:%M").time(), tzinfo=tz)
    return min(max(t, lo), hi), s


def arm(when: date, *, dry_run: bool = False) -> int:
    """Point the scheduled task at `when`'s computed time.

    Uses schtasks /Change /ST, which edits the existing DAILY trigger in place and needs no
    elevation for a task this user registered. Setting a time that has already passed today is
    harmless: a daily trigger simply fires at that time on the next day."""
    t, s = target_time(when)
    hhmm = t.strftime("%H:%M")
    print(f"[sunsched] {when}: sunset {s['sunset']:%H:%M} -> batch at {hhmm} "
          f"(sunset - {SUNSET_OFFSET_H}h)")
    if dry_run:
        return 0
    r = subprocess.run(["schtasks", "/Change", "/TN", TASK_NAME, "/ST", hhmm],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # Never fatal. If the trigger cannot be moved the batch still runs at yesterday's time,
        # which is at most a couple of minutes off -- the drift is about 1.5 min/day here.
        print(f"[sunsched] WARNING: could not re-arm '{TASK_NAME}' (exit {r.returncode}): "
              f"{(r.stderr or r.stdout).strip()}", file=sys.stderr)
        return 0
    print(f"[sunsched] '{TASK_NAME}' re-armed for {hhmm}.")
    return 0


def show(days: int) -> int:
    print(f"batch start = sunset - {SUNSET_OFFSET_H}h  (clamped to {EARLIEST}..{LATEST})\n")
    print("date         sunrise  sunset   batch starts")
    for i in range(days):
        d = date.today() + timedelta(days=i)
        t, s = target_time(d)
        print(f"{d}   {s['sunrise']:%H:%M}    {s['sunset']:%H:%M}    {t:%H:%M}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Aim the nightly batch at the glare window.")
    ap.add_argument("--arm", action="store_true", help="re-arm the scheduled task's start time")
    ap.add_argument("--date", choices=("today", "tomorrow"), default="tomorrow",
                    help="which day to aim at (default: tomorrow -- the batch re-arms itself for "
                         "the NEXT run at the START of each run, so a crash mid-batch still "
                         "leaves tomorrow correctly scheduled)")
    ap.add_argument("--show", type=int, nargs="?", const=14, metavar="DAYS",
                    help="print the schedule for the next DAYS days (default 14) and change nothing")
    ap.add_argument("--dry-run", action="store_true", help="say what --arm would do, do nothing")
    args = ap.parse_args()

    try:
        import astral  # noqa: F401
    except ImportError:
        print("[sunsched] 'astral' is not installed -- cannot resolve the sun, leaving the "
              "schedule alone. Fix: pip install astral", file=sys.stderr)
        return 0

    if args.show is not None:
        return show(args.show)
    if args.arm:
        d = date.today() + (timedelta(days=1) if args.date == "tomorrow" else timedelta(0))
        return arm(d, dry_run=args.dry_run)
    return show(14)


if __name__ == "__main__":
    sys.exit(main())
