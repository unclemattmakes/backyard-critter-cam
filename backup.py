"""
Back up everything the rig GENERATES into a cloud-synced folder.

The one honest risk to this project is the single disk under it: the code lives on GitHub and
the model weights re-download from Zenodo, but the clips, crops and database exist nowhere
else. This script archives exactly that generated content into a destination folder -- in
practice a Google-Drive-synced folder, so the actual cloud upload is Drive's job, not ours.

Design notes (why it looks the way it does):

* Per-DAY zip archives for media, not per-month. clips/ is a ROLLING window -- the live rig
  prunes the oldest clips past clips_max_gb (~2 weeks of footage at current rates), so a clip
  can be recorded AND deleted between two monthly runs. A day folder is frozen once its date
  has passed, so each is zipped exactly once, the morning after. An existing CLIPS archive is
  never rebuilt: the source may since have been legitimately pruned, and rebuilding from a
  pruned folder would shrink the archive -- the whole point is that the archive OUTLIVES the
  pruning. (crops/ archives, by contrast, ARE rebuilt if the source folder gained files: crops
  are never pruned, but a trail-cam SD import can add photos to a past date.)
* Media zips are STORED, not compressed: mp4 and jpg are already compressed, so deflating them
  buys ~nothing and burns CPU. Zipping still pays because Drive syncs one ~75 MB file far
  faster than ~2,000 individual crop JPEGs (per-file sync overhead dominates small files).
* The database is snapshotted with SQLite's backup API over a READ-ONLY connection -- safe
  beside the live capture thread and the naming helper (WAL writers keep writing; we get a
  consistent point-in-time copy), then PRAGMA quick_check'd before it's accepted. SQLite files
  deflate well (~2-4x), so the snapshot zip IS compressed.
* Idempotent: run it as often as you like; work already archived is skipped. The weekly
  schedule (Task Scheduler, Monday 03:30) is deliberately tighter than "monthly" because of
  the pruning window above -- a month between runs would lose whatever clips_max_gb ate.

Restore: unzip every archive into the project root (the folder layout inside the zips matches
the project -- clips/2026-06-09/..., crops/...), then unzip the newest db snapshot beside the
code. The README.txt this script drops in the destination says the same thing.

Usage:
    python backup.py                 # destination from config (backup_dest in config_local.py)
    python backup.py --dest E:\\somewhere
    python backup.py --dry-run       # print what would be done, write nothing
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

from config import CONFIG, ROOT

log = logging.getLogger("backup")

# Day folders under clips//crops//frames/ are named by local calendar date.
DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Small, changing odds-and-ends bundled into one deflated "meta" zip per run: re-ID artifacts,
# tracklet thumbnails, tuning shots, logs, reports, and the machine's private config. All cheap,
# all annoying to lose. (Weights and .venv are deliberately absent: re-downloadable.)
META_ITEMS: tuple[Path, ...] = (
    ROOT / "reid",
    CONFIG.clip_crops_dir,
    ROOT / "tuning",
    ROOT / "logs",
    ROOT / "reports",
    ROOT / "config_local.py",
)


def day_dirs(src_root: Path) -> list[tuple[Path, str]]:
    """The date-named day folders under a media root, as (folder, archive-name-stem) pairs,
    oldest days first -- oldest first so we archive the days nearest the pruning axe before it
    falls. Handles BOTH clip layouts: the legacy flat clips/<date>/ AND the multi-camera
    clips/<source>/<date>/ (each camera writes under its own source name since 2026-06-26), so
    those archives are per-camera-per-day: clips-glass_door_cam-2026-06-26.zip. Anything else
    is surfaced (a future layout change should be noticed, not silently skipped)."""
    out: list[tuple[Path, str]] = []
    if not src_root.is_dir():
        return out
    for p in sorted(src_root.iterdir()):
        if p.is_dir() and DAY_DIR_RE.match(p.name):
            out.append((p, f"{src_root.name}-{p.name}"))
        elif p.is_dir():
            # A per-camera source subdir: its date folders live one level down.
            for q in sorted(p.iterdir()):
                if q.is_dir() and DAY_DIR_RE.match(q.name):
                    out.append((q, f"{src_root.name}-{p.name}-{q.name}"))
                else:
                    log.warning("ignoring unexpected entry (not a YYYY-MM-DD folder): %s", q)
        else:
            log.warning("ignoring unexpected entry (not a YYYY-MM-DD folder): %s", p)
    out.sort(key=lambda pair: pair[0].name)  # oldest DAYS first, across layouts and cameras
    return out


def zip_tree(src_dir: Path, out_zip: Path, arc_root: Path, compression: int, dry_run: bool) -> tuple[int, int]:
    """Zip src_dir (recursively) to out_zip; arcnames are relative to arc_root so extracting
    into the project root restores the original layout. Written to a .tmp then renamed, so a
    crash never leaves a plausible-looking half archive. A file that vanishes mid-zip (the
    clips pruner racing us) is logged and skipped, not fatal. Returns (files, bytes) added."""
    files = [p for p in sorted(src_dir.rglob("*")) if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    if dry_run:
        log.info("would create %s  (%d files, %.1f MB)", out_zip.name, len(files), total / 2**20)
        return len(files), total
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    n = 0
    with zipfile.ZipFile(tmp, "w", compression=compression) as zf:
        for p in files:
            try:
                zf.write(p, p.relative_to(arc_root))
                n += 1
            except FileNotFoundError:
                log.warning("vanished while zipping (pruned?): %s", p)
    os.replace(tmp, out_zip)
    log.info("created %s  (%d files, %.1f MB)", out_zip.name, n, out_zip.stat().st_size / 2**20)
    return n, total


def member_count(zip_path: Path) -> int | None:
    """How many files an existing archive holds (None = unreadable). Only the central
    directory is read, so this is cheap even on a multi-GB zip."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return sum(1 for i in zf.infolist() if not i.is_dir())
    except (OSError, zipfile.BadZipFile):
        return None


def archive_media(src_root: Path, out_dir: Path, today: date, *, rebuild_if_grown: bool, dry_run: bool) -> dict:
    """Per-day archives for one media root (clips/crops/frames). Today's folder is skipped --
    it's still being written to; the next run finalizes it. rebuild_if_grown: re-archive a past
    day whose source gained files (crops after an SD import); clips pass False because there a
    shrunken source means PRUNED, and the archive must win."""
    stats = {"created": 0, "skipped": 0, "files": 0, "bytes": 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.zip.tmp"):
        log.warning("removing leftover partial from an interrupted run: %s", stale.name)
        stale.unlink()
    for day, stem in day_dirs(src_root):
        if day.name >= today.isoformat():
            log.info("skipping %s (still today -- archived on the next run)",
                     day.relative_to(src_root.parent))
            continue
        source_files = sum(1 for p in day.rglob("*") if p.is_file())
        if source_files == 0:
            continue  # a pruned-empty leftover folder; nothing to archive
        out_zip = out_dir / f"{stem}.zip"
        if out_zip.exists():
            if rebuild_if_grown:
                have = member_count(out_zip)
                if have is not None and source_files > have:
                    log.info("%s grew since archived (%d -> %d files): rebuilding",
                             out_zip.name, have, source_files)
                else:
                    stats["skipped"] += 1
                    continue
            else:
                stats["skipped"] += 1
                continue
        n, b = zip_tree(day, out_zip, arc_root=ROOT, compression=zipfile.ZIP_STORED, dry_run=dry_run)
        stats["created"] += 1
        stats["files"] += n
        stats["bytes"] += b
    return stats


def snapshot_db(db_path: Path, out_dir: Path, today: date, dry_run: bool) -> None:
    """Consistent point-in-time copy of the live WAL database via the SQLite backup API, then
    quick_check'd -- a snapshot that doesn't open cleanly is worse than none, because it looks
    like a backup. Same-day reruns replace the day's snapshot."""
    out_zip = out_dir / f"backyard-db-{today.isoformat()}.zip"
    if dry_run:
        log.info("would snapshot %s (%.1f MB) -> %s", db_path.name, db_path.stat().st_size / 2**20, out_zip.name)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_db = out_dir / f".snapshot-{today.isoformat()}.db"
    src = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=60)
    try:
        dst = sqlite3.connect(tmp_db)
        try:
            src.backup(dst)
            verdict = dst.execute("PRAGMA quick_check").fetchone()[0]
            if verdict != "ok":
                raise RuntimeError(f"snapshot failed quick_check: {verdict}")
        finally:
            dst.close()
    finally:
        src.close()
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_db, db_path.name)
    os.replace(tmp, out_zip)
    tmp_db.unlink()
    log.info("created %s  (db %.1f MB -> %.1f MB zipped, quick_check ok)", out_zip.name,
             db_path.stat().st_size / 2**20, out_zip.stat().st_size / 2**20)


def snapshot_meta(out_dir: Path, today: date, dry_run: bool) -> None:
    """One deflated zip of all the small side content (META_ITEMS). Rewritten whole every run --
    it's tens of MB, and 'always current, always complete' beats clever here."""
    out_zip = out_dir / f"meta-{today.isoformat()}.zip"
    present = [p for p in META_ITEMS if p.exists()]
    if dry_run:
        log.info("would create %s from: %s", out_zip.name, ", ".join(p.name for p in present))
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    n = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in present:
            files = [item] if item.is_file() else [p for p in sorted(item.rglob("*")) if p.is_file()]
            for p in files:
                try:
                    zf.write(p, p.relative_to(ROOT))
                    n += 1
                except FileNotFoundError:
                    log.warning("vanished while zipping: %s", p)
    os.replace(tmp, out_zip)
    log.info("created %s  (%d files, %.1f MB)", out_zip.name, n, out_zip.stat().st_size / 2**20)


README = """Backyard Critter Cam -- content backups
=========================================

Written by backup.py in the project repo; runs weekly (Task Scheduler, Monday 03:30).

  clips/      one zip per camera per day of video clips (clips-<camera>-<date>.zip; days
              before the multi-camera layout are just clips-<date>.zip). Uncompressed
              inside -- mp4 already is.
  crops/      one zip per calendar day of detection crops (JPEGs)
  snapshots/  backyard-db-<date>.zip  = consistent SQLite snapshot, integrity-checked
              meta-<date>.zip         = re-ID data, tracklet thumbs, tuning, logs, config
  backup.log  what happened on every run

RESTORE onto a fresh machine:
  1. git clone the repo, set up .venv per the README (weights re-download themselves).
  2. Extract EVERY zip in clips/ and crops/ into the project root (each contains its own
     clips/<date>/... or crops/<date>/... paths, so they land in place).
  3. From the NEWEST snapshots/backyard-db-*.zip, extract backyard.db into the project root.
  4. From the newest meta-*.zip, extract everything into the project root.

Note: clips/ on the live machine is a rolling window (clips_max_gb prunes the oldest), so this
backup legitimately holds MORE days of video than the machine does. That's the point.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive generated content (clips, crops, db) to a synced folder.")
    ap.add_argument("--dest", type=Path, default=CONFIG.backup_dest,
                    help="destination root (default: backup_dest from config_local.py)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be done; write nothing")
    args = ap.parse_args()

    if args.dest is None:
        raise SystemExit(
            "No backup destination configured. Set it once in config_local.py, e.g.\n"
            "    from pathlib import Path\n"
            "    cfg.backup_dest = Path(r'C:\\project-content\\backyard')\n"
            "(point it at a cloud-synced folder), or pass --dest."
        )
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    if ROOT in [dest, *dest.parents]:
        raise SystemExit(f"Refusing to back up into the project itself: {dest}")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if not args.dry_run:
        handlers.append(logging.FileHandler(dest / "backup.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)

    t0 = time.monotonic()
    today = date.today()
    log.info("backup starting: %s -> %s%s", ROOT, dest, " (DRY RUN)" if args.dry_run else "")
    failures = 0

    # Media roots: clips never rebuild (pruning would shrink the archive); crops/frames rebuild
    # if a past day gained files (SD-card imports write into old dates).
    media = [(CONFIG.clips_dir, False), (CONFIG.crops_dir, True), (CONFIG.frames_dir, True)]
    for src_root, regrow in media:
        if not src_root.is_dir():
            continue
        try:
            s = archive_media(src_root, dest / src_root.name, today,
                              rebuild_if_grown=regrow, dry_run=args.dry_run)
            log.info("%s: %d day(s) archived (%d files, %.1f MB), %d already done",
                     src_root.name, s["created"], s["files"], s["bytes"] / 2**20, s["skipped"])
        except Exception:
            log.exception("archiving %s failed", src_root.name)
            failures += 1

    try:
        snapshot_db(CONFIG.db_path, dest / "snapshots", today, args.dry_run)
    except Exception:
        log.exception("database snapshot failed")
        failures += 1
    try:
        snapshot_meta(dest / "snapshots", today, args.dry_run)
    except Exception:
        log.exception("meta snapshot failed")
        failures += 1

    if not args.dry_run:
        (dest / "README.txt").write_text(README, encoding="utf-8")
    log.info("backup %s in %.1f s (%d failure(s))",
             "FAILED" if failures else "finished", time.monotonic() - t0, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
