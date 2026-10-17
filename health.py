"""What needs a human right now, in one list, from every watcher that knows.

rigwatch.py and backup.py each notice things only a person can fix. They write what they noticed
into small JSON records, and rig_health() turns those into the list the morning email and the
dashboard show. An empty list means nothing needs anyone.

The records (both in the project root, both gitignored, both replaced atomically):

  .rigwatch_state.json   rigwatch's own bookkeeping, plus
                           "checked_at": <epoch of its last run>,
                           "alarms": {<kind>: {"severity", "message", "first_seen", "last_seen"}}
                         first_seen/last_seen are epochs; a kind disappears when it clears.
  .backup_state.json     {"last_run": <iso>, "last_ok": <iso or null>, "failures": <int>,
                          "failing_since": <iso or null>,
                          "lost_clip_days": {"<camera>/<YYYY-MM-DD>": <iso first reported>}}

Each item rig_health() returns is {"severity", "kind", "message", "since"}: severity is one of
SEVERITIES, since is a local ISO timestamp (or None). Messages are plain sentences for a person;
anything URL-shaped is masked before it leaves here, because these go out by email.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import config

SEVERITIES = ("alarm", "warn", "info")      # most urgent first; also the sort order

RIGWATCH_STALE_S = 30 * 60      # it runs every 5 min, so 30 min is six missed runs, not a blip
BACKUP_STALE_H = 36.0           # a daily job: one late run is allowed, a missed day is not
NEW_LOST_WINDOW_H = 24.0        # a lost clip day is news in the next daily email, not every one

_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)


# ---------------------------------------------------------------------------- the records

def read_json(path) -> dict:
    """The record at `path`, or {} when it is missing, torn, or not a JSON object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                          # noqa: BLE001 -- any unreadable record is "none"
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path, data: dict) -> bool:
    """Write temp + replace, so a reader never sees half a record. False if it could not land."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        return False
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.05)                   # Windows: a reader holds the target for a moment
        except OSError:
            break
    try:
        tmp.unlink()
    except OSError:
        pass
    return False


def now_iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if ts is None else ts).astimezone().isoformat(
        timespec="seconds")


# ---------------------------------------------------------------------------- the reader

def rig_health(now=None, *, rigwatch_state=None, backup_state=None) -> list[dict]:
    """Everything that currently needs a human, most urgent first. `now` is an epoch or an aware
    datetime (default: now); the paths default to the live records."""
    now_ts = _epoch(now) if now is not None else time.time()
    items = (_rigwatch_items(Path(rigwatch_state or config.RIGWATCH_STATE_FILE), now_ts)
             + _backup_items(Path(backup_state or config.BACKUP_STATE_FILE), now_ts))
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    items.sort(key=lambda it: (rank.get(it["severity"], len(rank)), it["since"] or ""))
    return items


def _rigwatch_items(path: Path, now: float) -> list[dict]:
    st = read_json(path)
    if not st:
        return []                              # never ran on this machine: nothing to judge
    checked = st.get("checked_at")
    if not isinstance(checked, (int, float)):
        try:
            checked = path.stat().st_mtime     # a record from before checked_at existed
        except OSError:
            return []
    if now - checked > RIGWATCH_STALE_S:
        # Its alarms are as old as its last run, so they say nothing about now.
        return [_item("alarm", "rigwatch_silent",
                      f"The watchdog (rigwatch.py) has not run for {_age(now - checked)}, so "
                      f"nothing is restarting the rig or checking naming and the nightly batch. "
                      f"Is its scheduled task still running?", checked)]
    out = []
    for kind, a in (st.get("alarms") or {}).items():
        if not isinstance(a, dict) or not a.get("message"):
            continue
        sev = a.get("severity") if a.get("severity") in SEVERITIES else "warn"
        first = a.get("first_seen")
        out.append(_item(sev, str(kind), str(a["message"]),
                         first if isinstance(first, (int, float)) else None))
    return out


def _backup_items(path: Path, now: float) -> list[dict]:
    st = read_json(path)
    if not st:
        return []                              # backups not set up here, or not run since upgrade
    out = []
    last_ok = _parse(st.get("last_ok"))
    failures = st.get("failures") or 0
    if failures:
        since = _parse(st.get("failing_since")) or _parse(st.get("last_run"))
        out.append(_item("alarm", "backup_failed",
                         f"The last backup run failed ({failures} failure"
                         f"{'' if failures == 1 else 's'}). backup.log in the backup folder "
                         f"says what.", since))
    if last_ok is not None and now - last_ok > BACKUP_STALE_H * 3600:
        out.append(_item("alarm", "backup_stale",
                         f"No successful backup for {_age(now - last_ok)}: new clips, crops and "
                         f"the database exist only on the rig's own disk.", last_ok))
    elif last_ok is None and not failures and _parse(st.get("last_run")) is not None:
        out.append(_item("alarm", "backup_stale", "No backup run has succeeded yet.",
                         _parse(st.get("last_run"))))
    lost = st.get("lost_clip_days") or {}
    if isinstance(lost, dict):
        new = sorted((day, t) for day, t in ((d, _parse(v)) for d, v in lost.items())
                     if t is not None and now - t <= NEW_LOST_WINDOW_H * 3600)
        if new:
            names = [d for d, _ in new]
            shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
            n = len(names)
            out.append(_item("warn", "clips_lost",
                             f"{n} day{'' if n == 1 else 's'} of clips {'was' if n == 1 else 'were'} "
                             f"pruned before the backup archived {'it' if n == 1 else 'them'} "
                             f"({shown}); that footage is gone. {len(lost)} such day"
                             f"{'' if len(lost) == 1 else 's'} in all.",
                             min(t for _, t in new)))
    return out


# ---------------------------------------------------------------------------- helpers

def _item(severity: str, kind: str, message: str, since_ts: float | None) -> dict:
    return {"severity": severity, "kind": kind,
            "message": _URL_RE.sub("<url>", message)[:400],
            "since": now_iso(since_ts) if since_ts is not None else None}


def _epoch(x) -> float:
    if isinstance(x, datetime):
        return (x if x.tzinfo else x.astimezone()).timestamp()
    return float(x)


def _parse(iso) -> float | None:
    if not iso:
        return None
    try:
        return _epoch(datetime.fromisoformat(str(iso)))
    except (TypeError, ValueError):
        return None


def _age(seconds: float) -> str:
    m = seconds / 60.0
    if m < 90:
        return f"{m:.0f} min"
    h = m / 60.0
    return f"{h:.0f} h" if h < 48 else f"{h / 24.0:.0f} days"
