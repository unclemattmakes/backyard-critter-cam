"""
Back up everything the rig GENERATES into a cloud-synced folder.

The one honest risk to this project is the single disk under it: the code lives on GitHub and
the model weights re-download from Zenodo, but the clips, crops and database exist nowhere
else. This script archives exactly that generated content into a destination folder -- normally
one your cloud client (Google Drive, Dropbox, OneDrive) syncs, so the upload is its job, not ours.

Design notes (why it looks the way it does):

* Per-DAY zip archives for media, not per-month. clips/ is a ROLLING window -- the live rig
  prunes the oldest clips past clips_max_gb (~2 weeks of footage at current rates), so a clip
  can be recorded AND deleted between two monthly runs. A day folder is *mostly* frozen once its
  date has passed, so each is zipped once, the morning after -- but "mostly" is not "always": a
  trail-cam SD import backfills whatever dates the card recorded, and the dump day always gets
  two separate batches, because the card goes straight back in the camera and the next dump
  brings the rest of that same day. So archives are APPEND-ONLY: an existing one is never
  rebuilt (rebuilding from a since-pruned source would shrink it, and the whole point is that
  the archive OUTLIVES the pruning), but files the source has and the archive lacks are merged
  in. The comparison is by NAME, not by count, because one day folder can lose files to the
  pruner and gain files from an import between two runs -- and then it holds FEWER files than
  its archive while still holding some that belong in it.
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
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
import zipfile
from datetime import date, datetime
from pathlib import Path

import db
from config import CONFIG, ROOT

log = logging.getLogger("backup")

# Day folders under clips//crops//frames/ are named by local calendar date.
DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Entries that sit BESIDE the day folders, are expected there, and are deliberately not archived.
# clips/reels/ is reel.py's output -- the stitched highlight reels, plus their .json manifests and
# .jpg posters. Those are a CACHE, not content: each reel is cut from the very clips this script
# does archive, its filename is a hash of that cut list, and reel.py rebuilds one on demand (and
# deletes all but the newest KEEP_REELS anyway). Archiving them would store the same footage a
# second time, in a form nothing restores from. Named one by one rather than pattern-matched, so
# that anything ELSE appearing under a media root is still surfaced as the surprise it is.
NOT_ARCHIVED = frozenset({"reels"})

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
    those archives are per-camera-per-day: clips-glass_door_cam-2026-06-26.zip. NOT_ARCHIVED
    entries are passed over quietly; anything else is surfaced (a future layout change should be
    noticed, not silently skipped)."""
    out: list[tuple[Path, str]] = []
    if not src_root.is_dir():
        return out
    for p in sorted(src_root.iterdir()):
        if p.is_dir() and DAY_DIR_RE.match(p.name):
            out.append((p, f"{src_root.name}-{p.name}"))
        elif p.name in NOT_ARCHIVED:
            # Expected, and not ours to copy. Warning about it once per FILE inside buried the run
            # (60 of ~75 lines on 2026-08-02); a warning nobody can act on trains you to ignore
            # the ones you can.
            log.debug("skipping %s (regenerable, deliberately not archived)", p)
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


def archived_names(zip_path: Path) -> set[str] | None:
    """The arcnames of the files an existing archive holds (None = unreadable). Only the central
    directory is read, so this is cheap even on a multi-GB zip. Names rather than a count because
    a count cannot tell "nine unchanged" from "six of those nine pruned, three new ones imported":
    both directions move at once in a day folder, and only the names say which files are missing."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return {i.filename for i in zf.infolist() if not i.is_dir()}
    except (OSError, zipfile.BadZipFile):
        return None


def add_to_zip(new_files: list[Path], out_zip: Path, arc_root: Path, compression: int,
               dry_run: bool) -> tuple[int, int]:
    """Add files to an EXISTING archive, keeping every member already in it -- merge, never
    rebuild. What the archive holds is often the only copy left (the clips pruner deletes from
    the source; the trail cam's card is formatted every cycle), so a file that once made it in is
    never dropped just because the source no longer has it. The new members are appended to a
    COPY that then replaces the original, because appending in place rewrites the central
    directory: an interrupted run would leave the archive it was extending unreadable. Returns
    (files, bytes) added."""
    total = sum(p.stat().st_size for p in new_files)
    if dry_run:
        log.info("would add %d file(s) (%.1f MB) to %s", len(new_files), total / 2**20, out_zip.name)
        return len(new_files), total
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    shutil.copy2(out_zip, tmp)
    n = 0
    with zipfile.ZipFile(tmp, "a", compression=compression) as zf:
        for p in new_files:
            try:
                zf.write(p, p.relative_to(arc_root))
                n += 1
            except FileNotFoundError:
                log.warning("vanished while zipping (pruned?): %s", p)
    os.replace(tmp, out_zip)
    log.info("added %d file(s) to %s  (now %.1f MB)", n, out_zip.name,
             out_zip.stat().st_size / 2**20)
    return n, total


def archive_media(src_root: Path, out_dir: Path, today: date, *, dry_run: bool) -> dict:
    """Per-day archives for one media root (clips/crops/frames). Today's folder is skipped -- it's
    still being written to; the next run finalizes it. A past day that is already archived is
    normally skipped outright, but any file its source folder holds and its archive doesn't gets
    MERGED in, because a past day is not as frozen as it looks: a trail-cam SD import backfills
    every date the card recorded, and the dump day gets a second batch a few days later when the
    rest of that day comes off the card. Meanwhile the pruner is deleting from those same folders,
    so a day can be simultaneously shorter than its archive and holding files that never reached
    it -- hence the by-name diff, and hence merging rather than rebuilding."""
    stats = {"created": 0, "merged": 0, "skipped": 0, "files": 0, "bytes": 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.zip.tmp"):
        log.warning("removing leftover partial from an interrupted run: %s", stale.name)
        stale.unlink()
    for day, stem in day_dirs(src_root):
        if day.name >= today.isoformat():
            log.info("skipping %s (still today -- archived on the next run)",
                     day.relative_to(src_root.parent))
            continue
        source = [p for p in sorted(day.rglob("*")) if p.is_file()]
        if not source:
            continue  # a pruned-empty leftover folder; nothing to archive
        out_zip = out_dir / f"{stem}.zip"
        if out_zip.exists():
            have = archived_names(out_zip)
            if have is None:
                log.warning("%s exists but cannot be read -- leaving it alone; check it by hand "
                            "(a good archive is never overwritten by this script)", out_zip.name)
                stats["skipped"] += 1
                continue
            new = [p for p in source if p.relative_to(ROOT).as_posix() not in have]
            if not new:
                stats["skipped"] += 1
                continue
            log.info("%s gained %d file(s) since it was archived (%d already in it): merging",
                     out_zip.name, len(new), len(have))
            n, b = add_to_zip(new, out_zip, arc_root=ROOT, compression=zipfile.ZIP_STORED,
                              dry_run=dry_run)
            stats["merged"] += 1
        else:
            n, b = zip_tree(day, out_zip, arc_root=ROOT, compression=zipfile.ZIP_STORED,
                            dry_run=dry_run)
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


def snapshot_weights(out_dir: Path, dry_run: bool) -> None:
    """A ONE-TIME weights archive (never rebuilt, like the clips zips): the MDv6 .pt plus the
    Hugging Face caches of MegaDescriptor / BioCLIP / the CLIP gate. backup.py's founding
    assumption -- 'weights re-download themselves' -- is true for MDv6 (Zenodo is archival, DOI'd)
    and FALSE for the three HF models: repos get pulled, gated and re-licensed routinely, and if
    hf-hub:BVRA/MegaDescriptor-L-384 vanishes, every stored 1536-d vector survives but the
    embedding space can never be extended -- the whole cross-time identity archive freezes. ~1.3 GB
    of Drive space, exactly once, closes that. Local archival of already-downloaded weights for
    continuity, not redistribution (NOTICE.md's licensing posture; MegaDescriptor is CC-BY-NC)."""
    out_zip = out_dir / "weights-archive.zip"
    if out_zip.exists():
        log.debug("weights archive already exists (never rebuilt): %s", out_zip.name)
        return
    sources: list[Path] = []
    wdir = ROOT / "weights"
    if wdir.is_dir():
        sources.append(wdir)
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hub.is_dir():
        keywords = ("megadescriptor", "bioclip", "clip-vit-b-32")
        for p in sorted(hub.iterdir()):
            if p.is_dir() and any(k in p.name.lower() for k in keywords):
                sources.append(p)
    if not sources:
        log.info("weights archive: nothing to archive yet (no weights/ or HF cache)")
        return
    total = sum(f.stat().st_size for s in sources for f in ([s] if s.is_file() else s.rglob("*"))
                if f.is_file())
    if dry_run:
        log.info("would create %s from %d source(s), %.1f GB",
                 out_zip.name, len(sources), total / 2**30)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    n = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:   # weights don't deflate
        for src in sources:
            base = src.parent
            for p in sorted(src.rglob("*")):
                if p.is_file() and not p.is_symlink():
                    try:
                        zf.write(p, Path(src.name) / p.relative_to(src))
                        n += 1
                    except (FileNotFoundError, OSError) as e:
                        log.warning("skipping %s: %s", p, e)
        zf.comment = (f"one-time weights archive, {date.today().isoformat()}; "
                      f"immutable -- restore by extracting weights/ into the project and "
                      f"models--* dirs into ~/.cache/huggingface/hub/").encode()
    os.replace(tmp, out_zip)
    log.info("created %s  (%d files, %.1f GB, one time only)", out_zip.name, n,
             out_zip.stat().st_size / 2**30)


def export_labels(out_dir: Path, today: date, dry_run: bool) -> None:
    """The human-label LEDGER: every human verdict as one dated, append-only JSONL, diffed
    against the previous export. The label set is the project's only irreplaceable asset, and
    until now its durability was welded to full-DB snapshots and its integrity had no monitor --
    a mass-mislabel event (the one-click refit hazard the identity eval warns about) would have
    been silent. Now it is a LOUD line: rows changed/removed since last week get logged, and the
    labels stay restorable and greppable without a DB restore."""
    out_jsonl = out_dir / f"labels-{today.isoformat()}.jsonl"
    conn = db.connect_readonly(CONFIG.db_path)
    if conn is None:
        log.info("label ledger: no database yet")
        return
    try:
        # Column list built from PRAGMA, not assumed: a read-only connection never migrates, so
        # this must work against a DB no new-code writer has touched yet (labeled_by landed
        # 2026-08-08 and appears only after the first read-write connect).
        have = {r[1] for r in conn.execute("PRAGMA table_info(detections)")}
        cols = [c for c in ("id", "timestamp", "species", "species_verified", "species_source",
                            "individual_id", "individual_source", "labelled_at", "labeled_by")
                if c in have]
        rows = []
        for r in conn.execute(
                f"SELECT {', '.join(cols)} FROM detections "
                "WHERE species_source = 'human' OR species_verified IS NOT NULL "
                "   OR individual_source IS NOT NULL"):
            rows.append({"kind": "detection", **{k: r[k] for k in r.keys()}})
        for table, kind in (("live_sightings", "sighting"), ("individual_status", "status"),
                            ("life_events", "event")):
            try:
                for r in conn.execute(f"SELECT * FROM {table}"):
                    rows.append({"kind": kind, **{k: r[k] for k in r.keys()}})
            except Exception:
                pass
    finally:
        conn.close()
    if dry_run:
        log.info("would write %s (%d rows)", out_jsonl.name, len(rows))
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # Diff against the previous ledger BEFORE writing this one. Keyed by (kind, id/name);
    # a changed row counts once. Removals are the alarm class.
    prev = sorted(out_dir.glob("labels-*.jsonl"))
    prev = prev[-1] if prev else None
    lines = [json.dumps(r, sort_keys=True, default=str) for r in rows]
    out_jsonl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if prev is not None and prev != out_jsonl:
        old = set(prev.read_text(encoding="utf-8").splitlines())
        new = set(lines)
        added, removed = len(new - old), len(old - new)
        if removed > 50:
            log.warning("LABEL LEDGER: %d row(s) changed or VANISHED since %s (+%d new) -- a "
                        "mass label change; if you didn't do this on purpose, the previous "
                        "ledger is the restore point", removed, prev.name, added)
        else:
            log.info("label ledger: +%d / ~%d vs %s (%d rows total)",
                     added, removed, prev.name, len(lines))
    else:
        log.info("label ledger: first export, %d rows", len(lines))
    # Keep the ledger family bounded but deep: it is small text and gzips well in the cloud.
    for old_file in sorted(out_dir.glob("labels-*.jsonl"))[:-12]:
        old_file.unlink(missing_ok=True)


def write_status(dest: Path, failures: int, dry_run: bool) -> None:
    """STATUS.txt -- the heartbeat that reaches the owner. The project has no notification
    channel of any kind; this file, rewritten by every weekly run into the Drive-synced folder,
    is the cheapest one that already exists: its CONTENTS say how the rig is, and its absence
    or staleness (mtime on the phone's Drive app) IS the alarm. Everything here is read-only
    and best-effort -- a status writer that can crash the backup would be a bad joke."""
    if dry_run:
        log.info("would write STATUS.txt")
        return
    lines = [f"Backyard Critter Cam -- weekly status, {datetime.now().astimezone().isoformat()}",
             f"backup: {'FAILED (' + str(failures) + ' failure(s))' if failures else 'ok'}"]
    try:
        log_p = ROOT / "logs" / "backyard_cam.log"
        if log_p.exists():
            age_h = (time.time() - log_p.stat().st_mtime) / 3600
            lines.append(f"rig log last wrote: {age_h:.1f} h ago"
                         + ("  <-- the rig may be DOWN" if age_h > 6 else ""))
        else:
            lines.append("rig log: none found")
    except OSError:
        pass
    conn = db.connect_readonly(CONFIG.db_path)
    if conn is not None:
        try:
            last = conn.execute("SELECT MAX(timestamp) FROM detections").fetchone()[0]
            lines.append(f"newest detection: {last or 'none'}")
            last_label = conn.execute(
                "SELECT MAX(labelled_at) FROM detections WHERE individual_source = 'human'"
            ).fetchone()[0]
            lines.append(f"newest human identity label: {last_label or 'none recorded'}"
                         + ("  <-- past the ~14-day decay horizon; the matcher is going blind"
                            if _older_than_days(last_label, 14) else ""))
            try:
                n7 = conn.execute(
                    "SELECT COUNT(*) FROM detections WHERE suppressed_at IS NOT NULL "
                    "AND suppressed_at >= datetime('now', '-7 day', 'localtime')").fetchone()[0]
                lines.append(f"refimg shadow flags, last 7 days: {n7}"
                             + ("  -- review with: python refimg.py --review" if n7 else ""))
            except Exception:
                pass
        finally:
            conn.close()
    try:
        led = sorted(ROOT.glob("*.imported-*.txt"), key=lambda p: p.stat().st_mtime)
        if led:
            age_d = (time.time() - led[-1].stat().st_mtime) / 86400
            lines.append(f"trail-cam last import: {age_d:.1f} days ago"
                         + ("  <-- the card may be filling" if age_d > 5 else ""))
    except OSError:
        pass
    try:
        rep = sorted((ROOT / "reports").glob("eval_*.json"))
        lines.append(f"newest eval artifact: {rep[-1].name if rep else 'none -- the gate has never run'}")
    except OSError:
        pass
    try:
        free = shutil.disk_usage(ROOT).free / 2**30
        lines.append(f"disk free on the rig: {free:.0f} GB")
    except OSError:
        pass
    (dest / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote STATUS.txt (%d line(s))", len(lines))


def _older_than_days(iso_ts, days: int) -> bool:
    t = db.parse_local(iso_ts) if iso_ts else None
    if t is None:
        return False
    return (datetime.now().astimezone() - t).days > days


README = """Backyard Critter Cam -- content backups
=========================================

Written by backup.py in the project repo; runs weekly (Task Scheduler, Monday 03:30).

  clips/      one zip per camera per day of video clips (clips-<camera>-<date>.zip; days
              before the multi-camera layout are just clips-<date>.zip). Uncompressed
              inside -- mp4 already is. The highlight reels (clips/reels/ on the machine)
              are NOT here: each is stitched from these clips and rebuilds itself on demand.
  crops/      one zip per calendar day of detection crops (JPEGs)
  snapshots/  backyard-db-<date>.zip  = consistent SQLite snapshot, integrity-checked
              meta-<date>.zip         = re-ID data, tracklet thumbs, tuning, logs, config
              weights-archive.zip     = ONE-TIME model-weights mirror (MDv6 + the Hugging
                                        Face checkpoints); never rebuilt -- insurance for
                                        the day a hub repo disappears
              labels-<date>.jsonl     = every human verdict as an append-only ledger,
                                        diffed weekly (a mass label change logs LOUDLY)
              export-<date>.zip       = the observation record as plain CSV + DATA.md --
                                        readable on any stack, forever, no venv required
  STATUS.txt  the weekly heartbeat: rig freshness, newest labels, shadow-review flags,
              disk headroom. If this file goes stale on your phone's Drive app, the
              backup task itself has stopped -- that staleness IS the alarm.
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
            "No backup destination configured. There is no default: point it at a folder YOUR "
            "cloud client syncs (Google Drive, Dropbox, OneDrive, a NAS mount). Set it once in "
            "config_local.py, substituting your own path:\n"
            "    from pathlib import Path\n"
            "    cfg.backup_dest = Path(r'C:\\cloud-synced-folder\\backyard')\n"
            "or pass --dest for a one-off."
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

    # Media roots, all archived the same way: a new day is zipped whole, an already-archived day
    # is topped up with whatever it has gained (SD-card imports write into past dates, in clips as
    # well as crops) and never rebuilt, because the pruner has been deleting from those same
    # folders and the archive must win.
    for src_root in (CONFIG.clips_dir, CONFIG.crops_dir, CONFIG.frames_dir):
        if not src_root.is_dir():
            continue
        try:
            s = archive_media(src_root, dest / src_root.name, today, dry_run=args.dry_run)
            log.info("%s: %d day(s) archived, %d topped up (%d files, %.1f MB), %d already done",
                     src_root.name, s["created"], s["merged"], s["files"], s["bytes"] / 2**20,
                     s["skipped"])
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
    try:
        snapshot_weights(dest / "snapshots", args.dry_run)
    except Exception:
        log.exception("weights archive failed")
        failures += 1
    try:
        export_labels(dest / "snapshots", today, args.dry_run)
    except Exception:
        log.exception("label ledger failed")
        failures += 1
    try:
        import export as _export
        _export.export_bundle(dest / "snapshots", dry_run=args.dry_run)
        # One export per run; keep the newest few (the DB snapshots hold the deep history).
        if not args.dry_run:
            for old_file in sorted((dest / "snapshots").glob("export-*.zip"))[:-4]:
                old_file.unlink(missing_ok=True)
    except Exception:
        log.exception("CSV export failed")
        failures += 1

    try:
        write_status(dest, failures, args.dry_run)
    except Exception:
        log.exception("status write failed")     # never let the heartbeat sink the backup

    if not args.dry_run:
        (dest / "README.txt").write_text(README, encoding="utf-8")
    log.info("backup %s in %.1f s (%d failure(s))",
             "FAILED" if failures else "finished", time.monotonic() - t0, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
