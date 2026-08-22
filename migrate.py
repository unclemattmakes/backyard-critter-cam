"""
Move the whole rig -- database, media, labels, references, config -- to another machine.

Two verbs, one archive format:

    python migrate.py pack D:\\rig-move        # OLD machine: write a complete, current bundle
    python migrate.py restore D:\\rig-move     # NEW machine: reassemble the rig from it

`pack` writes the SAME layout backup.py writes weekly (per-day media zips, a quick_check'd DB
snapshot, the meta zip, the one-time weights mirror). That is a design decision, not laziness:
it means `restore` needs to understand exactly one format, so the same command also rebuilds a
rig from the WEEKLY CLOUD BACKUP after a dead machine -- migration and disaster recovery are the
same operation, differing only in how fresh the source is. It also means a re-run `pack` is an
incremental top-up (the archives are append-only), not a second full copy.

What `pack` adds over a weekly run:

* TODAY's media folders are included (the weekly run leaves today for tomorrow; a move can't).
* Files written in the last minute are left for the next pass: the recorder writes each mp4 IN
  PLACE at its final name, and the archives diff by name -- a half-written file sealed today
  would block the finished one forever.
* The DB snapshot is taken NOW, over the backup API, so packing beside a live rig is safe.

Because a live rig keeps writing while pack runs, the honest move is TWO passes:

    1. pack while the rig runs        -- the hours-long bulk copy happens beside a working rig
    2. stop the rig, pack again       -- seconds: only the delta since pass 1 is added
    3. restore on the new machine

`restore` automates the README.txt steps that used to be by-hand, and adds the judgement a
hand restore forgets at 2am:

* The database is extracted to a scratch file and PRAGMA quick_check'd BEFORE it is installed;
  a snapshot that fails is skipped with a warning and the next-newest is tried (the 08-19 crash
  corrupted a live DB -- the corruption a restore exists for must not be re-installed by it).
* Clips are restored SELECTIVELY: only rows the old rig still held (clips.pruned_at IS NULL).
  The archive deliberately holds more days than any machine does; resurrecting the pruned files
  would fight the disk budget and duplicate what the dashboard already plays straight out of
  these very zips (web.py's archive_cache). Crops and frames are restored in full -- they are
  never pruned and the identity archive lives in them.
* Nothing is ever overwritten. An existing file is kept and counted, so an interrupted restore
  is simply re-run; a config_local.py you already wrote on the new machine wins over the old
  machine's copy (loudly).
* The database lands LAST, atomically. Its presence is the "restore finished" marker, which is
  also why restore REFUSES to run at all where a backyard.db already lives: this tool moves a
  rig onto a machine, it does not merge two rigs.
* After install, the restored DB is cross-checked against the restored files (do the rows'
  crop/clip paths actually exist here?) and a new-machine checklist is printed -- the .venv,
  camera index, scheduled tasks and backup destination are per-machine and no archive can
  carry them.

Weights: the bundle carries the one-time weights mirror (MegaDetector + the Hugging Face
checkpoints) and restore unpacks it into place -- weights/ into the project, models--* into
~/.cache/huggingface/hub/ -- so the new machine works offline and keeps working even if a hub
repo has since vanished. `--no-weights` on either verb skips the ~1.3 GB if you'd rather
re-download.

Not migrated, on purpose: .venv/ and environment.lock.txt's authority (per-machine builds --
run setup on the new machine), clips_web/ and archive_cache/ (rebuild on demand),
refimg_store/ (the furniture veto re-certifies fresh references within minutes of the rig
starting), and the runtime droppings (.locks/, .rig_pause, .rigwatch_state.json,
.naming_status.json) that describe a machine, not a rig.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import zipfile
from datetime import date
from pathlib import Path

import backup
import db
import heavyio
from config import CONFIG, ROOT

log = logging.getLogger("migrate")

# Where the Hugging Face models--* trees in weights-archive.zip get restored to (the mirror's
# own zip comment documents the same destination). Module-level so tests can point it at tmp.
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

# Restore writes each file to <name>.migrate-tmp in its final directory, then renames -- a crash
# leaves at most one stray tmp, swept on the next run so it can never leak into a future backup.
TMP_SUFFIX = ".migrate-tmp"

# The recorder writes mp4s in place; anything younger than this at pack time is left for the
# next pass (see the two-pass flow in the module docstring).
PACK_SETTLE_S = 60.0

# Cross-check sampling cap: enough rows to catch a systematic hole, cheap enough to run always.
CROSS_CHECK_SAMPLE = 2000


# ------------------------------------------------------------------------------------- pack
def pack(dest: Path, *, dry_run: bool = False, include_weights: bool = True) -> int:
    """Write a complete, current bundle of this rig into `dest`, in backup.py's exact format.
    Append-only like the weekly run, so re-running only adds the delta. Safe beside a live rig
    (backup-API DB snapshot; sub-minute-old media left for the next pass)."""
    if ROOT in [dest, *dest.parents]:
        raise SystemExit(f"Refusing to pack the rig into itself: {dest}")
    if not CONFIG.db_path.exists():
        raise SystemExit(f"No database at {CONFIG.db_path} -- there is no rig here to pack. "
                         "(Run this from the machine the rig lives on.)")
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    today = date.today()
    failures = 0
    t0 = time.monotonic()

    # Same crash-avoidance as backup.py: this writes even more bytes than a weekly run, so take
    # the heavy-io lock before the first one and hold it until any cloud client under `dest` has
    # actually drained (heavyio.py's module docstring has the two bugchecks that paid for this).
    if not dry_run:
        heavyio.acquire("migrate", wait_s=3600, note=f"migrate.py pack -> {dest}")
    try:
        log.info("pack starting: %s -> %s%s", ROOT, dest, " (DRY RUN)" if dry_run else "")
        for src_root in (CONFIG.clips_dir, CONFIG.crops_dir, CONFIG.frames_dir):
            if not src_root.is_dir():
                continue
            try:
                s = backup.archive_media(src_root, dest / src_root.name, today, dry_run=dry_run,
                                         include_today=True, settle_s=PACK_SETTLE_S)
                log.info("%s: %d day(s) archived, %d topped up (%d files, %.1f MB), %d already done",
                         src_root.name, s["created"], s["merged"], s["files"], s["bytes"] / 2**20,
                         s["skipped"])
            except Exception:
                log.exception("archiving %s failed", src_root.name)
                failures += 1
        for step, fn in (("database snapshot",
                          lambda: backup.snapshot_db(CONFIG.db_path, dest / "snapshots", today, dry_run)),
                         ("meta snapshot",
                          lambda: backup.snapshot_meta(dest / "snapshots", today, dry_run)),
                         ("label ledger",
                          lambda: backup.export_labels(dest / "snapshots", today, dry_run))):
            try:
                fn()
            except Exception:
                log.exception("%s failed", step)
                failures += 1
        if include_weights:
            try:
                backup.snapshot_weights(dest / "snapshots", dry_run)
            except Exception:
                log.exception("weights archive failed")
                failures += 1
        if not dry_run:
            (dest / "README.txt").write_text(backup.README, encoding="utf-8")
    finally:
        if not dry_run:
            try:
                heavyio.wait_drive_quiet(timeout_s=3600)
            except Exception:
                log.exception("waiting for the cloud upload to settle failed")
            finally:
                heavyio.release("migrate")

    log.info("pack %s in %.1f s (%d failure(s))",
             "FAILED" if failures else "finished", time.monotonic() - t0, failures)
    if not failures:
        print(f"\nPacked this rig into: {dest}"
              + ("\n(dry run -- nothing was written)" if dry_run else ""))
        print(f"""Next:
  * If the rig was RUNNING just now, the newest minutes of media postdate the database
    snapshot. Before the actual move: STOP the rig, run this same pack once more (it only
    adds the delta -- seconds, not hours), and migrate from that.
  * On the new machine: clone the repo, run setup.bat / setup.sh, then from the project root:
        python migrate.py restore {dest}
    (Point it at wherever this folder lives as seen from that machine -- a USB drive letter,
    a network share, or the cloud-synced copy.)""")
    return 1 if failures else 0


# ---------------------------------------------------------------------------------- restore
def _within(target: Path, root: Path) -> bool:
    """Is `target` inside `root` once resolved? Archive members name where they land; one that
    escapes the project root is refused, not trusted."""
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _sweep_tmp(root: Path) -> None:
    """Remove stray TMP_SUFFIX leftovers of an interrupted restore, so they can never be taken
    for content -- least of all by a future backup of the restored rig."""
    for p in root.glob(f"**/*{TMP_SUFFIX}"):
        log.warning("removing leftover partial from an interrupted restore: %s", p)
        try:
            p.unlink()
        except OSError:
            pass


def _pick_db_snapshot(snap_dir: Path, scratch: Path) -> tuple[Path, Path]:
    """Extract and verify the newest database snapshot that passes PRAGMA quick_check, newest
    first. Returns (zip used, extracted db file in `scratch`). A snapshot that fails the check
    is exactly what the 08-19-style crash leaves behind -- warn and try the one before it,
    because installing a corrupt database defeats the whole point of having snapshots."""
    candidates = sorted(snap_dir.glob("backyard-db-*.zip"), reverse=True)
    if not candidates:
        raise SystemExit(f"{snap_dir.parent} is not a backup/pack folder: no "
                         "snapshots/backyard-db-*.zip in it. Point restore at the folder "
                         "backup.py or `migrate.py pack` wrote (the one holding clips/, crops/ "
                         "and snapshots/).")
    for zp in candidates:
        out = scratch / f"candidate-{zp.stem}.db"
        try:
            with zipfile.ZipFile(zp) as zf:
                members = [i for i in zf.infolist() if not i.is_dir()]
                if len(members) != 1:
                    log.warning("%s holds %d members (expected exactly the database) -- skipping",
                                zp.name, len(members))
                    continue
                with zf.open(members[0]) as fsrc, open(out, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst, 1 << 20)
        except (OSError, zipfile.BadZipFile) as e:
            log.warning("%s is unreadable (%s) -- trying the snapshot before it", zp.name, e)
            continue
        conn = sqlite3.connect(f"file:{out.as_posix()}?mode=ro", uri=True)
        try:
            verdict = conn.execute("PRAGMA quick_check").fetchone()[0]
        except sqlite3.DatabaseError as e:
            verdict = f"not a database ({e})"
        finally:
            conn.close()
        if verdict == "ok":
            log.info("database snapshot %s: quick_check ok", zp.name)
            return zp, out
        log.warning("database snapshot %s FAILS quick_check (%s) -- trying the one before it. "
                    "Keep the bad zip: a partial salvage may still be possible by hand.",
                    zp.name, verdict)
    raise SystemExit("Every database snapshot in the folder fails its integrity check. "
                     "Nothing was restored. The newest zips may still be salvageable by hand "
                     "(sqlite3 .recover); this tool will not install a database it cannot trust.")


def _live_clip_paths(snapshot_db: Path) -> set[str] | None:
    """The clip files the old rig still HELD: clips rows without pruned_at, '/'-normalized to
    match zip arcnames. None means the snapshot cannot say (a DB from before soft-pruning) --
    the caller then restores every archived clip rather than guessing."""
    conn = sqlite3.connect(f"file:{snapshot_db.as_posix()}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(clips)")}
        if "pruned_at" not in cols:
            return None
        return {str(r[0]).replace("\\", "/")
                for r in conn.execute("SELECT clip_path FROM clips WHERE pruned_at IS NULL")}
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()


def _plan_zip(zp: Path, dest_root: Path, only: set[str] | None = None):
    """What extracting `zp` into `dest_root` would actually do: (members to extract, their
    bytes, tallies of everything passed over). Reads only the central directory, so planning a
    multi-GB archive is cheap -- and the plan IS the dry run."""
    take: list[str] = []
    take_bytes = 0
    stats = {"present": 0, "filtered": 0, "unsafe": 0}
    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if only is not None and name not in only:
                stats["filtered"] += 1
                continue
            target = dest_root / name
            if not _within(target, dest_root):
                log.warning("refusing archive member that escapes the project: %r in %s",
                            name, zp.name)
                stats["unsafe"] += 1
                continue
            if target.exists():
                stats["present"] += 1
                continue
            take.append(name)
            take_bytes += info.file_size
    return take, take_bytes, stats


def _extract(zp: Path, names: list[str], dest_root: Path) -> int:
    """Extract exactly `names` from `zp` under `dest_root`, each via a same-directory tmp then
    an atomic rename -- a crash mid-file leaves a swept-on-rerun tmp, never a plausible-looking
    half file at a real name."""
    n = 0
    with zipfile.ZipFile(zp) as zf:
        for name in names:
            target = dest_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + TMP_SUFFIX)
            with zf.open(name) as fsrc, open(tmp, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, 1 << 20)
            os.replace(tmp, target)
            n += 1
    return n


def _weights_plan(zp: Path):
    """Route the weights mirror's members to their two destinations: weights/* under the
    project, the models--* Hugging Face trees under HF_HUB -- exactly what the mirror's own zip
    comment tells a hand-restorer to do. Returns ([(member, dest_root)...], bytes, present)."""
    take: list[tuple[str, Path]] = []
    take_bytes = 0
    present = 0
    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            droot = ROOT if name.split("/", 1)[0] == "weights" else HF_HUB
            target = droot / name
            if not _within(target, droot):
                log.warning("refusing archive member that escapes its root: %r", name)
                continue
            if target.exists():
                present += 1
                continue
            take.append((name, droot))
            take_bytes += info.file_size
    return take, take_bytes, present


def _cross_check(keep: set[str] | None) -> list[str]:
    """Does the restored database agree with the restored files? Sampled for crops (hundreds of
    thousands of rows), exhaustive for live clips (thousands). A miss here is not fatal -- a
    weekly-backup restore legitimately lacks the media recorded after its last run -- but it
    must be SAID, because a dashboard of broken images should never be a surprise."""
    lines: list[str] = []
    conn = db.connect_readonly(CONFIG.db_path)
    if conn is None:
        return ["database: not readable after install (this should be impossible -- stop here)"]
    try:
        n_det = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        rows = [r[0] for r in conn.execute(
            "SELECT crop_path FROM detections ORDER BY RANDOM() LIMIT ?",
            (CROSS_CHECK_SAMPLE,))]
        missing = sum(1 for p in rows if not (ROOT / str(p).replace("\\", "/")).is_file())
        scope = "all" if n_det <= CROSS_CHECK_SAMPLE else f"{len(rows)} sampled of"
        lines.append(f"crops: {scope} {n_det} detection rows -> "
                     + ("every sampled file present" if missing == 0
                        else f"{missing} MISSING (recorded after the source's last backup?)"))
        try:
            live = [str(r[0]).replace("\\", "/") for r in conn.execute(
                "SELECT clip_path FROM clips WHERE pruned_at IS NULL")]
            gone = sum(1 for p in live if not (ROOT / p).is_file())
            lines.append(f"clips: {len(live)} live rows -> "
                         + ("every file present" if gone == 0
                            else f"{gone} MISSING (recorded after the source's last backup?)"))
            n_pruned = conn.execute(
                "SELECT COUNT(*) FROM clips WHERE pruned_at IS NOT NULL").fetchone()[0]
            if n_pruned:
                lines.append(f"clips: {n_pruned} pruned rows stay archive-only, as on the old "
                             "machine -- the dashboard plays them from the backup zips once "
                             "cfg.backup_dest points at them from here")
        except sqlite3.OperationalError:
            lines.append("clips: table predates soft-pruning; every archived clip was restored")
    finally:
        conn.close()
    return lines


CHECKLIST = """
This machine is not the old machine. Before first launch:

 1. Environment: run setup.bat (Windows) / setup.sh (Linux, macOS) if you haven't already --
    the .venv and the torch/CUDA build are per-machine and never travel in a bundle.
    (environment.lock.txt from the meta zip is the OLD machine's build record, for provenance;
    setup writes this machine's own.)
 2. config_local.py is the old machine's. Re-check its machine-specific lines:
      cfg.camera_index / cfg.cameras   which camera is index 0 HERE?  python backyard_cam.py --list-cameras
      cfg.device                       only if you pinned 'cuda' on the old rig
      cfg.backup_dest                  does that exact path exist on THIS machine? Point it at
                                       this machine's cloud-synced folder -- ideally the same
                                       archive family you just restored from, so the weekly
                                       backup continues it and pruned-clip playback works.
 3. Scheduled jobs live in the OS, not the repo. Re-register the ones you used: the weekly
    backup.py task (README "Backups"), setup_selfheal.bat, newsletter.py if you run the
    morning email.
 4. On the OLD machine: retire the rig. Stop it and DISABLE its scheduled backup task -- two
    machines appending into one archive folder interleave their zips and take turns rewriting
    STATUS.txt, and whichever you read lies about the other.
 5. Launch (python backyard_cam.py --record-clips), open http://127.0.0.1:8000, and check the
    Individuals tab still knows everybody.
""".rstrip()


def restore(src: Path, *, dry_run: bool = False, include_weights: bool = True) -> int:
    """Reassemble a rig here from a pack bundle or a weekly backup folder (same format). Media
    first, database last and atomically -- its arrival is what marks the restore finished, so
    an interrupted run is simply re-run and picks up where it stopped."""
    if not src.is_dir():
        raise SystemExit(f"Not a folder: {src}")
    if ROOT in [src, *src.parents]:
        raise SystemExit(f"The source folder is inside the project ({src}) -- restore reads a "
                         "bundle from OUTSIDE the rig it rebuilds (a USB drive, a share, the "
                         "cloud-synced backup folder).")
    if CONFIG.db_path.exists():
        raise SystemExit(
            f"There is already a rig here: {CONFIG.db_path} exists. This tool moves a rig onto "
            "a machine; it does not merge two rigs. If replacing this one is really what you "
            "want, move backyard.db (and its -wal/-shm siblings) aside yourself, then re-run.")
    for sibling in (CONFIG.db_path.with_name(CONFIG.db_path.name + "-wal"),
                    CONFIG.db_path.with_name(CONFIG.db_path.name + "-shm")):
        if sibling.exists():
            raise SystemExit(f"{sibling.name} exists without {CONFIG.db_path.name} -- a stale "
                             "WAL sidecar would silently corrupt the restored database the "
                             "moment SQLite opens it. Move it aside, then re-run.")
    if not dry_run:
        _sweep_tmp(ROOT)

    t0 = time.monotonic()
    report: list[str] = []
    with tempfile.TemporaryDirectory(prefix="critter-restore-") as td:
        scratch = Path(td)
        db_zip, tmp_db = _pick_db_snapshot(src / "snapshots", scratch)
        keep = _live_clip_paths(tmp_db)
        if keep is None:
            log.warning("the snapshot predates clip soft-pruning -- restoring every archived "
                        "clip (the disk budget will re-trim)")

        # Plan everything before writing anything: the plan is the dry run, the disk check,
        # and the work list, and reading central directories is cheap even for a season of zips.
        media_plan: list[tuple[Path, list[str]]] = []
        need_bytes = tmp_db.stat().st_size          # the extracted size, not the deflated zip's
        for root_name, only in (("clips", keep), ("crops", None), ("frames", None)):
            zips = sorted((src / root_name).glob("*.zip")) if (src / root_name).is_dir() else []
            n_take = n_bytes = n_present = n_filtered = 0
            for zp in zips:
                try:
                    take, b, st = _plan_zip(zp, ROOT, only=only)
                except (OSError, zipfile.BadZipFile) as e:
                    log.warning("%s is unreadable (%s) -- skipped; whatever it held is not "
                                "restored", zp.name, e)
                    continue
                if take:
                    media_plan.append((zp, take))
                n_take += len(take)
                n_bytes += b
                n_present += st["present"]
                n_filtered += st["filtered"]
            need_bytes += n_bytes
            if zips:
                line = f"{root_name}: {n_take} file(s) to restore ({n_bytes / 2**20:.1f} MB)"
                if n_present:
                    line += f", {n_present} already here"
                if n_filtered:
                    line += f", {n_filtered} pruned -- staying archive-only"
                report.append(line)
            else:
                report.append(f"{root_name}: nothing archived")

        meta_zips = sorted((src / "snapshots").glob("meta-*.zip"))
        meta_plan: tuple[Path, list[str]] | None = None
        if meta_zips:
            take, b, st = _plan_zip(meta_zips[-1], ROOT)
            need_bytes += b
            meta_plan = (meta_zips[-1], take)
            line = f"meta: {len(take)} file(s) from {meta_zips[-1].name}"
            # The two files a new machine may have already written itself. Existing files are
            # never overwritten anywhere, but for THESE two silence would read as "restored" --
            # say whose copy wins (the archived one stays in the meta zip for a hand diff).
            with zipfile.ZipFile(meta_zips[-1]) as zf:
                in_zip = set(zf.namelist())
            kept = [n for n in ("config_local.py", "environment.lock.txt")
                    if n in in_zip and (ROOT / n).exists()]
            if kept:
                line += f" (keeping this machine's own {', '.join(kept)})"
            report.append(line)
        else:
            report.append("meta: no meta-*.zip -- config_local.py and the ledgers are not in "
                          "this source; write config_local.py by hand")

        weights_zip = src / "snapshots" / "weights-archive.zip"
        weights_take: list[tuple[str, Path]] = []
        if include_weights and weights_zip.is_file():
            try:
                weights_take, b, present = _weights_plan(weights_zip)
                need_bytes += b
                report.append(f"weights: {len(weights_take)} file(s) ({b / 2**30:.1f} GB) into "
                              f"weights/ and {HF_HUB}"
                              + (f", {present} already here" if present else ""))
            except (OSError, zipfile.BadZipFile) as e:
                log.warning("weights-archive.zip is unreadable (%s) -- weights will re-download "
                            "on first run instead", e)
        elif not weights_zip.is_file():
            report.append("weights: no mirror in this source -- they re-download on first run")
        else:
            report.append("weights: skipped (--no-weights) -- they re-download on first run")

        report.insert(0, f"database: {db_zip.name} (quick_check ok)")

        free = shutil.disk_usage(ROOT).free
        if need_bytes > free:
            raise SystemExit(
                f"Not enough disk: this restore needs ~{need_bytes / 2**30:.1f} GB and "
                f"{ROOT.anchor or ROOT} has {free / 2**30:.1f} GB free. Nothing was written. "
                "Free space (or use --no-weights) and re-run -- already-restored files are "
                "skipped, so a re-run only does what is left.")

        header = f"{'Would restore' if dry_run else 'Restoring'} from {src} into {ROOT}:"
        print("\n" + header)
        for line in report:
            print(f"  {line}")
        if dry_run:
            print("\n(dry run -- nothing was written)")
            return 0

        for zp, names in media_plan:
            n = _extract(zp, names, ROOT)
            log.info("%s: %d file(s) restored", zp.name, n)
        if meta_plan:
            n = _extract(meta_plan[0], meta_plan[1], ROOT)
            log.info("%s: %d file(s) restored", meta_plan[0].name, n)
        if weights_take:
            by_root: dict[Path, list[str]] = {}
            for name, droot in weights_take:
                by_root.setdefault(droot, []).append(name)
            for droot, names in by_root.items():
                n = _extract(weights_zip, names, droot)
                log.info("weights-archive.zip: %d file(s) restored under %s", n, droot)

        # The database goes in LAST, atomically: everything it references is already on disk,
        # and from the next line on, this machine HAS a rig (and restore will refuse to re-run).
        staging = CONFIG.db_path.with_name(CONFIG.db_path.name + TMP_SUFFIX)
        shutil.copyfile(tmp_db, staging)
        os.replace(staging, CONFIG.db_path)
        log.info("installed %s from %s", CONFIG.db_path.name, db_zip.name)

    print(f"\nRestore finished in {time.monotonic() - t0:.1f} s. Cross-checking the database "
          "against the restored files:")
    for line in _cross_check(keep):
        print(f"  {line}")
    print(CHECKLIST)
    return 0


# ------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Move the whole rig between machines: pack it into a bundle, restore it "
                    "from one -- or from the weekly backup folder, which is the same format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The full story (two-pass pack, what restore checks, what never migrates) is at "
               "the top of this file and in the README under \"Moving the rig to a new "
               "machine\".")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_pack = sub.add_parser("pack", help="OLD machine: write a complete, current bundle "
                                         "(append-only; re-run to top up)")
    p_pack.add_argument("dest", type=Path, help="bundle folder (USB drive, network share, or a "
                                                "cloud-synced folder)")
    p_rest = sub.add_parser("restore", help="NEW machine: reassemble the rig from a bundle or "
                                            "from the weekly backup folder")
    p_rest.add_argument("src", type=Path, help="the folder pack/backup.py wrote (holds clips/, "
                                               "crops/, snapshots/)")
    for p in (p_pack, p_rest):
        p.add_argument("--dry-run", action="store_true",
                       help="say what would be done; write nothing")
        p.add_argument("--no-weights", action="store_true",
                       help="skip the ~1.3 GB model-weights mirror (they re-download instead)")
    args = ap.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if not args.dry_run:
        if args.cmd == "pack":
            # Same guard pack() itself applies, but BEFORE the log file forces the folder into
            # existence -- refusing a destination must not first create it inside the project.
            if ROOT in [args.dest, *args.dest.parents]:
                raise SystemExit(f"Refusing to pack the rig into itself: {args.dest}")
            args.dest.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(args.dest / "migrate.log", encoding="utf-8"))
        else:
            (ROOT / "logs").mkdir(exist_ok=True)
            handlers.append(logging.FileHandler(ROOT / "logs" / "migrate.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)

    if args.cmd == "pack":
        return pack(args.dest, dry_run=args.dry_run, include_weights=not args.no_weights)
    return restore(args.src, dry_run=args.dry_run, include_weights=not args.no_weights)


if __name__ == "__main__":
    sys.exit(main())
