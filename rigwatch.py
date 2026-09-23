"""Bring the rig back up by itself -- after a reboot, or after the app dies on its own.

WHY
---
2026-08-21 the box bugchecked at ~14:23 and came back at 14:26 with nothing watching the yard.
There was no task, no startup entry and no service that starts backyard_cam.py; every crash since
this project began has meant the yard stayed dark until Matt noticed. That is the wrong dependency
for a camera whose whole job is the hours nobody is watching -- and the crash families are
documented and NOT fixed (0x116 nvlddmkm TDR in the GPU-batch window, 0x1E after big Google Drive
uploads), so "it will not crash again" is not a plan.

setup_selfheal.bat is a different thing and does NOT cover this: it registers the elevated
USB-reset task the rig fires when the CAMERA STREAM wedges, while the app is still running.

DELIBERATE STOPS ARE RESPECTED
------------------------------
A watchdog that restarts the rig the instant you press 'q' is a bad neighbour. backyard_cam.main()
already distinguishes the two cases -- it returns 0 for a clean stop and non-zero for a crash --
so the launchers drop a .rig_pause marker on a clean exit, and this script leaves the rig alone
while that marker is newer than the last boot. Reboot, and the marker is stale by definition: a
fresh session starts the rig, because the whole point is not depending on Matt.

The marker is also cleared whenever the rig is seen RUNNING again (clear_pause_marker), so a stop
that has since been undone cannot leave the rig unguarded. That self-healing matters because not
every start goes through a launcher: start_critter_cam_lan.bat did not touch the marker at all
until 2026-08-21, and a rig started straight from python never does. Relying on each launcher to
remember is how a 18:55 marker came to be sitting on disk at 20:00 with the rig up and unwatched.

RESTART STORMS ARE CAPPED
-------------------------
If the rig is crashing on startup (bad config, wedged camera, no GPU), retrying forever would
spawn a log window and a browser tab every cycle. Three starts in a rolling hour is the ceiling;
after that it writes a loud line and waits the hour out.

USAGE
    python rigwatch.py            # one check -- what the scheduled task runs, every 5 min
    python rigwatch.py --status   # say what it sees, start nothing
    python rigwatch.py --force    # start the rig even if paused (ignores .rig_pause)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config

ROOT = config.ROOT
PAUSE_MARKER = ROOT / ".rig_pause"
STATE_FILE = ROOT / ".rigwatch_state.json"
LOG_FILE = ROOT / "logs" / "rigwatch.log"
LAUNCHER = ROOT / "start_critter_cam.bat"

MAX_STARTS_PER_HOUR = 3

# A LIVENESS CHECK IS NOT A HEALTH CHECK.
# This watchdog asked one question for its whole life -- "is there a backyard_cam.py pid?" -- and
# for eight days in 2026-08 the answer was yes while the rig named nothing at all. The naming
# helper had died on its first batch; the rig itself was perfectly alive, so nothing here fired,
# the dashboard pill still read "ready" off a status file frozen mid-sentence, and the nightly
# regression gate passed because it was comparing a frozen metric against itself. 48,275 crops
# went unnamed, and with them went visits, re-ID templates, auto-assign and the eval.
#
# So: when the rig IS up, also ask whether it is getting anything done. We never restart the rig
# over this -- backyard_cam.py supervises its own naming child now, and bouncing a healthy camera
# to fix a helper is the wrong hammer -- but a stuck namer stops being invisible.
NAMING_STALE_S = 900.0          # heartbeat is every ~5s; 15 min without one is unambiguous
NAMING_BACKLOG_ALARM = 200      # a backlog this big that is not shrinking is stuck, not busy


def log(msg: str) -> None:
    line = f"{datetime.now().astimezone():%Y-%m-%dT%H:%M:%S}  {msg}"
    try:
        # Guarded: the scheduled task runs this under pythonw.exe, which has NO stdout handle, and
        # an unguarded print there raises and kills the run -- the same trap that once silently
        # broke newsletter.py. The file log below is the one that matters anyway.
        print(line, flush=True)
    except Exception:
        pass
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def rig_pids() -> list[int] | None:
    """Pids running backyard_cam.py, or None when we genuinely cannot tell.

    Counted by command line, so the .venv redirector shows this process TWICE (shim + real, same
    command line -- the thing that once looked like 'two naming helpers'). That is fine here: we
    only ever ask whether the list is empty.

    None, not [], when psutil is missing. An empty list means "the rig is down, start it"; if a
    missing dependency could produce that, the watchdog would cheerfully restart a perfectly
    healthy rig every five minutes. Not knowing is not the same as knowing it is down."""
    try:
        import psutil
    except Exception:
        return None
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if "backyard_cam.py" in cmd and "rigwatch" not in cmd:
                out.append(p.info["pid"])
        except Exception:
            continue
    return out


def boot_time() -> float:
    try:
        import psutil
        return psutil.boot_time()
    except Exception:
        return 0.0


def paused() -> bool:
    """Did a human stop the rig on purpose, in THIS session?"""
    if not PAUSE_MARKER.exists():
        return False
    try:
        return PAUSE_MARKER.stat().st_mtime > boot_time()
    except OSError:
        return False


def clear_pause_marker() -> bool:
    """Drop the pause marker once the rig is observed RUNNING again. True if one was removed.

    The marker means "a human stopped this on purpose". A running rig means that stop is spent --
    whoever undid it, and however they started it. Clearing it HERE rather than in the launchers is
    the point: only start_critter_cam.bat ever cleared it, so a rig started from
    start_critter_cam_lan.bat, or straight from python (`--serve --host 0.0.0.0`, which is how Matt
    starts it), left the old marker sitting there. Because that marker was newer than the last
    boot, paused() read it as a deliberate stop and this watchdog quietly stopped guarding a rig
    that was up and running -- with nothing to see, until the next reboot aged the marker out.

    2026-08-21: a 18:55 marker was still on disk at 20:00 with the rig running unguarded, through
    two rigwatch restarts and two hand restarts. Making it self-healing beats asking every current
    and future launcher to remember."""
    if not PAUSE_MARKER.exists():
        return False
    try:
        PAUSE_MARKER.unlink()
    except OSError:
        return False                          # read-only / racing launcher: nothing worth failing over
    log("rig is up -- cleared a stale .rig_pause; the rig is guarded again.")
    return True


def naming_health() -> dict:
    """What the species namer's status file says, plus whether that is believable.

    Returns {present, state, age_s, backlog, named, stale, device}. `stale` is the important one:
    classify.py rewrites this file every poll, so an old timestamp means the loop is not running
    however cheerfully `state` reads. The file said "ready" for eight days after the process it
    describes had already exited."""
    out = {"present": False, "state": None, "age_s": None, "backlog": None,
           "named": None, "stale": False, "device": None}
    try:
        raw = json.loads(config.NAMING_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:                          # noqa: BLE001 -- no file yet, or a torn write
        return out
    out["present"] = True
    out["state"] = raw.get("state")
    out["named"] = raw.get("named")
    out["backlog"] = raw.get("backlog")        # absent on files written before this existed
    out["device"] = raw.get("device")
    out["detail"] = raw.get("detail")          # set by watch_loop when a poll raised
    try:
        out["age_s"] = max(0.0, time.time() - float(raw.get("ts") or 0.0))
        out["stale"] = out["age_s"] > NAMING_STALE_S
    except (TypeError, ValueError):
        out["stale"] = True
    return out


def check_naming(health: dict) -> bool:
    """Log a line when naming is not doing its job. Returns True if something was reported.

    Deliberately does NOT restart anything: the rig supervises its own naming child, and the point
    here is that a failure of BOTH becomes visible within five minutes instead of two weeks."""
    if not health["present"]:
        return False                           # naming may simply be off (--no-classify)
    if health["stale"]:
        # age_s stays None when the timestamp itself was unreadable -- the case that FORCES stale.
        # Formatting it unguarded raised TypeError and killed the whole check, under pythonw.exe
        # with no stdout: the watchdog would go silent in exactly the scenario it exists to shout
        # about. --status already guards this expression; this is the same guard.
        age = health["age_s"]
        when = f"{age / 60.0:.0f} min ago" if age is not None else "at an unreadable time"
        log(f"NAMING IS STALE: the status file still says '{health['state']}' but it was last "
            f"written {when}, so the naming loop is not running. New crops are NOT being named, "
            f"which also stops visits, re-ID templates and the nightly eval. "
            f"See logs/naming.log.")
        return True
    if health["state"] == "error":
        log(f"NAMING IS ERRORING: {health.get('detail') or 'see logs/naming.log'}.")
        return True

    backlog, prev = health.get("backlog"), _state().get("naming_backlog")
    if isinstance(backlog, int) and backlog >= NAMING_BACKLOG_ALARM:
        # Busy and shrinking is fine -- a backfill legitimately sits at a big number for a while.
        # Big and NOT shrinking between two checks five minutes apart is the stuck case.
        if isinstance(prev, int) and backlog >= prev:
            log(f"NAMING IS NOT KEEPING UP: {backlog} crops are waiting to be named and the "
                f"backlog has not fallen since the last check ({prev}). See logs/naming.log.")
            return True
    return False


def _state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"starts": []}


def _record_start() -> None:
    st = _state()
    st["starts"] = [t for t in st.get("starts", []) if time.time() - t < 3600] + [time.time()]
    try:
        STATE_FILE.write_text(json.dumps(st), encoding="utf-8")
    except OSError:
        pass


def _recent_starts() -> int:
    return len([t for t in _state().get("starts", []) if time.time() - t < 3600])


def _record_backlog(backlog) -> None:
    """Remember this check's naming backlog so the next one can tell 'busy' from 'stuck'."""
    if not isinstance(backlog, int):
        return
    st = _state()
    st["naming_backlog"] = backlog
    try:
        STATE_FILE.write_text(json.dumps(st), encoding="utf-8")
    except OSError:
        pass


def start_rig() -> int:
    if not LAUNCHER.exists():
        log(f"ERROR: launcher missing: {LAUNCHER}")
        return 1
    _record_start()
    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP -- the rig must outlive this short-lived
    # watchdog process, or Task Scheduler reaping us would take the camera down with it.
    subprocess.Popen(["cmd", "/c", "start", "", str(LAUNCHER)], cwd=str(ROOT),
                     creationflags=0x00000008 | 0x00000200, close_fds=True)
    log(f"rig was down -- started {LAUNCHER.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Restart the rig if it is not running.")
    ap.add_argument("--status", action="store_true", help="report and start nothing")
    ap.add_argument("--force", action="store_true", help="start even if paused by a clean stop")
    args = ap.parse_args()

    pids = rig_pids()
    if args.status:
        print("rig running     : " + ("UNKNOWN (psutil missing)" if pids is None
                                      else f"yes  pids={pids}" if pids else "NO"))
        print(f"paused by human : {paused()}  (marker: "
              f"{'present' if PAUSE_MARKER.exists() else 'absent'})")
        print(f"starts last hour: {_recent_starts()} / {MAX_STARTS_PER_HOUR}")
        print(f"last boot       : {datetime.fromtimestamp(boot_time()):%Y-%m-%d %H:%M:%S}")
        h = naming_health()
        if not h["present"]:
            print("species naming  : no status file (naming may be off)")
        else:
            age = f"{h['age_s'] / 60.0:.0f} min ago" if h["age_s"] is not None else "unknown"
            flag = "  <-- STALE, the loop is not running" if h["stale"] else ""
            print(f"species naming  : {h['state']} on {h['device']}, heartbeat {age}{flag}")
            print(f"  named this run: {h['named']}   backlog: {h['backlog']}")
        return 0

    if pids is None:
        # "No pids" and "cannot see pids" are the same value to a caller that only checks
        # emptiness, and acting on the second would restart a healthy rig every five minutes
        # forever. Refuse, and say why.
        log("cannot tell whether the rig is running (psutil is not installed) -- doing nothing. "
            "Fix: pip install psutil")
        return 1
    if pids:
        clear_pause_marker()                  # rig is up, so any deliberate-stop marker is spent
        # Up is not the same as working. check_naming stays quiet when naming is fine, so this
        # keeps the "say nothing, every 5 minutes, forever" contract for a genuinely healthy rig.
        health = naming_health()
        check_naming(health)
        _record_backlog(health.get("backlog"))
        return 0
    if paused() and not args.force:
        return 0                              # stopped on purpose this session -- leave it alone
    if _recent_starts() >= MAX_STARTS_PER_HOUR and not args.force:
        log(f"rig is down but it has been started {MAX_STARTS_PER_HOUR}x in the last hour -- "
            f"it is crashing on startup, not just missing. Backing off; needs a human. "
            f"See logs/backyard_cam.log.")
        return 1
    return start_rig()


if __name__ == "__main__":
    sys.exit(main())
