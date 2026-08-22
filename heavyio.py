"""Mutual exclusion for the two jobs that hammer this box, plus a wait-for-Google-Drive-to-settle.

WHY THIS EXISTS
---------------
2026-08-21 the machine bugchecked (0x1E KMODE_EXCEPTION_NOT_HANDLED) ~23 minutes into the daily
14:00 GPU batch, while Google Drive was still uploading the ~5.5 GB of zips a hand-run backup.py
had written between 13:41 and 13:50. chkdsk then rebuilt the project directory's own $I30 index
and dumped 1,032 files into C:\\found.001. Two days earlier, 08-19, the SAME bugcheck code landed
~30 minutes after the Drive migration finished writing the backup tree -- that one corrupted
backyard.db and cost a full row-by-row salvage. Windows blamed ndis.sys on the 08-19 one and
named nothing on the 08-21 one.

Two crashes, both 0x1E, both minutes after a large Drive upload was set going, both destructive.
That is not proof -- no debugger was ever put on either dump -- but "do not run the GPU batch on
top of a multi-gigabyte cloud upload" costs nothing and removes the only condition the two share.

So: any job that is about to move a lot of bytes takes a lock here first, and backup.py does not
consider itself finished until Drive has actually drained.

WHAT THIS IS NOT
----------------
Not a general-purpose lock. It is deliberately advisory, coarse (one namespace for the whole
machine) and biased to FAIL OPEN: if anything about the lock is unclear -- unreadable file, dead
owner, absurd age -- the caller runs. A backup that silently did not happen is worse than two
jobs overlapping, because the archive is the only copy of some of this footage.

USAGE
    python heavyio.py --acquire batch --wait 3600     # blocks until clear, then holds
    python heavyio.py --release batch
    python heavyio.py --status
    python heavyio.py --drive-quiet --timeout 1800    # block until Drive stops uploading
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import config

ROOT = config.ROOT
LOCK_DIR = ROOT / ".locks"

# A holder older than this is assumed dead even if some process still owns its pid (pids get
# recycled). The batch's own Task Scheduler ExecutionTimeLimit is PT6H, so 8h can only be a
# leftover from a crash -- which, given why this module exists, is the expected failure mode.
MAX_LOCK_AGE_S = 8 * 3600

# Drive is "quiet" once its processes move less than this per sample, for CONSECUTIVE samples.
# Not zero: Drive chatters at idle (metadata polls, and the content cache it grows just from
# READS), and the upload queue oscillates -- it will report the same op ids for a while before
# actually going quiet.
DRIVE_IDLE_BYTES_PER_S = 256 * 1024
DRIVE_SAMPLE_S = 5.0
DRIVE_CONSECUTIVE_IDLE = 6          # ~30s of calm before we believe it


def _alive(pid: int) -> bool:
    """Is this pid a live process? Fail-open: if we cannot tell, say yes, so we never steal a
    lock from a job that is genuinely running."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return True


def _lock_path(name: str) -> Path:
    return LOCK_DIR / f"{name}.lock"


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def read_holders() -> list[dict]:
    """Every lock currently held, stale ones dropped. Reading is also the reaping pass."""
    out: list[dict] = []
    if not LOCK_DIR.is_dir():
        return out
    for p in sorted(LOCK_DIR.glob("*.lock")):
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            # Unreadable lock file -- exactly what a torn write leaves behind, and this whole
            # module exists because of a crash that left torn files. It tells us nothing, so it
            # gets no authority: drop it.
            _unlink(p)
            continue
        age = time.time() - float(info.get("started_epoch", 0))
        expires = info.get("expires_epoch")
        if expires is not None:
            # TTL lock. Held by a .bat, whose steps are a SEQUENCE of short-lived pythons -- the
            # process that took the lock is already gone by the time anyone reads it, so a pid
            # check would reap the lock instantly. The .bat releases explicitly when it finishes;
            # the TTL is only the backstop for the run that never finishes because the box died.
            if time.time() > float(expires):
                _unlink(p)
                continue
        elif age > MAX_LOCK_AGE_S or not _alive(int(info.get("pid", -1))):
            _unlink(p)
            continue
        info["age_s"] = age
        out.append(info)
    return out


def acquire(name: str, *, wait_s: float = 3600.0, note: str = "", owner_pid: int | None = None,
            ttl_min: float | None = None) -> int:
    """Take the lock `name`, waiting until every OTHER holder has let go.

    Returns 0 once held. Returns 0 ALSO when the wait times out -- see the fail-open note in the
    module docstring: we say so loudly and run anyway, rather than skip a backup or a night of
    tracks."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_s
    announced: set[str] = set()
    timed_out = False
    while True:
        others = [h for h in read_holders() if h.get("name") != name]
        if not others:
            break
        for h in others:
            key = f"{h.get('name')}#{h.get('pid')}"
            if key not in announced:
                announced.add(key)
                note_s = f" -- {h['note']}" if h.get("note") else ""
                print(f"[heavyio] waiting for '{h.get('name')}' (pid {h.get('pid')}, running "
                      f"{h.get('age_s', 0) / 60:.0f} min){note_s}", flush=True)
        if time.monotonic() > deadline:
            names = ", ".join(str(h.get("name")) for h in others)
            print(f"[heavyio] WARNING: waited {wait_s / 60:.0f} min for {names} and it is STILL "
                  f"held. Running anyway -- a skipped run is worse than an overlap.", flush=True)
            timed_out = True
            break
        time.sleep(5)

    payload = {
        "name": name,
        "pid": owner_pid if owner_pid else os.getpid(),
        "host": socket.gethostname(),
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "started_epoch": time.time(),
        "note": note,
    }
    if ttl_min:
        payload["expires_epoch"] = time.time() + ttl_min * 60
    _lock_path(name).write_text(json.dumps(payload), encoding="utf-8")
    if announced and not timed_out:
        print(f"[heavyio] clear -- '{name}' has the box.", flush=True)
    elif timed_out:
        print(f"[heavyio] '{name}' is proceeding WITHOUT a clear box.", flush=True)
    return 0


def release(name: str) -> int:
    _unlink(_lock_path(name))
    return 0


class held:
    """`with heavyio.held('backup', note=...):` -- for callers inside Python, i.e. backup.py."""

    def __init__(self, name: str, *, wait_s: float = 3600.0, note: str = ""):
        self.name, self.wait_s, self.note = name, wait_s, note

    def __enter__(self):
        acquire(self.name, wait_s=self.wait_s, note=self.note)
        return self

    def __exit__(self, *exc):
        release(self.name)
        return False


# --------------------------------------------------------------------------- Drive quiesce
def _drive_io_bytes() -> int | None:
    """Total bytes read+written by every Google Drive process, or None if Drive is not running.

    Counting the PROCESS io counters rather than the G: volume on purpose: G: is a virtual mount
    whose apparent traffic includes cache reads we cause ourselves just by listing it."""
    try:
        import psutil
    except Exception:
        return None
    total, found = 0, False
    for proc in psutil.process_iter(["name"]):
        try:
            nm = (proc.info.get("name") or "").lower()
            if "googledrive" not in nm and "drivefs" not in nm:
                continue
            found = True
            io = proc.io_counters()
            total += io.read_bytes + io.write_bytes
        except Exception:
            continue
    return total if found else None


def wait_drive_quiet(timeout_s: float = 1800.0) -> int:
    """Block until Google Drive stops moving bytes, or `timeout_s` elapses.

    This is the half of the fix that actually matters. backup.py logging "finished in 498 s" means
    THE ZIPS ARE ON DISK -- Drive then uploads them on its own schedule, for as long as it takes,
    and on 2026-08-21 that upload was still running when the batch started and the box died. So
    the backup job holds its lock until Drive is done, not until the last zip is written."""
    if _drive_io_bytes() is None:
        print("[heavyio] Google Drive is not running -- nothing to wait for.", flush=True)
        return 0
    deadline = time.monotonic() + timeout_s
    last, t_last, idle_streak = _drive_io_bytes(), time.monotonic(), 0
    print(f"[heavyio] waiting for Google Drive to finish uploading (idle = under "
          f"{DRIVE_IDLE_BYTES_PER_S / 1024:.0f} KB/s for "
          f"{DRIVE_CONSECUTIVE_IDLE * DRIVE_SAMPLE_S:.0f}s)...", flush=True)
    while time.monotonic() < deadline:
        time.sleep(DRIVE_SAMPLE_S)
        now, t_now = _drive_io_bytes(), time.monotonic()
        if now is None:                       # Drive exited mid-wait -- nothing left to wait on
            print("[heavyio] Drive went away; treating as quiet.", flush=True)
            return 0
        rate = (now - last) / max(1e-6, t_now - t_last)
        last, t_last = now, t_now
        if rate < DRIVE_IDLE_BYTES_PER_S:
            idle_streak += 1
            if idle_streak >= DRIVE_CONSECUTIVE_IDLE:
                print(f"[heavyio] Drive is quiet ({rate / 1024:.0f} KB/s).", flush=True)
                return 0
        else:
            if idle_streak:
                print(f"[heavyio]   ...still going ({rate / 2 ** 20:.1f} MB/s)", flush=True)
            idle_streak = 0
    print(f"[heavyio] WARNING: Drive still busy after {timeout_s / 60:.0f} min. Continuing anyway.",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Keep the heavy jobs off each other's toes.")
    ap.add_argument("--acquire", metavar="NAME", help="take the lock, waiting for other holders")
    ap.add_argument("--release", metavar="NAME", help="give the lock back")
    ap.add_argument("--note", default="", help="human note stored in the lock file")
    ap.add_argument("--wait", type=float, default=3600.0, metavar="SECONDS",
                    help="how long to wait for a clear box before running anyway (default 3600)")
    ap.add_argument("--owner-pid", type=int, default=None,
                    help="record this pid as the holder instead of our own -- for .bat callers, "
                         "where the python that TAKES the lock exits immediately and the shell "
                         "that actually does the work is the real owner")
    ap.add_argument("--ttl", type=float, default=None, metavar="MINUTES",
                    help="hold by expiry instead of by pid -- for .bat callers whose work happens "
                         "in later, separate processes. Release explicitly when done; the TTL is "
                         "only the backstop for a run the machine kills.")
    ap.add_argument("--drive-quiet", action="store_true", help="block until Drive stops uploading")
    ap.add_argument("--timeout", type=float, default=1800.0, help="--drive-quiet timeout (s)")
    ap.add_argument("--status", action="store_true", help="who holds what")
    args = ap.parse_args()

    if args.status:
        holders = read_holders()
        if not holders:
            print("heavy-io locks: none held")
        for h in holders:
            print(f"  {h['name']:8s} pid={h['pid']:<7} for {h['age_s'] / 60:5.1f} min  "
                  f"{h.get('note', '')}")
        print(f"  google drive: {'running' if _drive_io_bytes() is not None else 'not running'}")
        return 0
    if args.acquire:
        return acquire(args.acquire, wait_s=args.wait, note=args.note, owner_pid=args.owner_pid,
                       ttl_min=args.ttl)
    if args.release:
        return release(args.release)
    if args.drive_quiet:
        return wait_drive_quiet(args.timeout)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
