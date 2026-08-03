"""
Tests for backup.py -- the per-day archiver that is the ONLY copy of most of this footage.

Nothing here touches the real project or the real backup destination: a fake project root is
built under tmp and backup.ROOT is pointed at it (arcnames inside the zips are relative to that
root), so every path in every assertion is synthetic. No cv2, no db, no network -- backup.py is
stdlib-only and so is this.

The rule under test is that **a media archive only ever grows**. Two forces act on one day
folder at the same time, in opposite directions:

  * the clips pruner DELETES the oldest files from it (clips_max_gb, a rolling ~2-week window),
    which is exactly why an archive must never be rebuilt from its source -- the archive is
    supposed to outlive the prune;
  * a trail-cam SD import ADDS files to it, days after the fact. The card is dumped every few
    days and goes straight back in the camera, so the dump DAY always arrives in two batches:
    2026-07-27 got clips up to 12:22 from the cycle-4 import and 16:43-onward from cycle-5.

The regression: archives for clips/ were skip-if-exists, so if a backup ran between two such
imports, that day's zip was sealed and the later clips were never archived -- and since the card
is formatted every cycle, clips/trail_cam_sd/ is the only copy, so the loss was permanent. The
fix is not "rebuild if it grew": a count-based growth check is blind when a day loses four files
to the pruner and gains three from an import (5 files vs 6 archived -- no "growth" to see), and
when it does fire it rebuilds from a pruned source and drops the already-archived files. So the
diff is by NAME and the merge is append-only; both halves are pinned below.
"""
from __future__ import annotations

import logging
import zipfile
from datetime import date
from pathlib import Path

import pytest

import backup

# Every day folder in this file is in the past relative to this; today's folder is never archived.
TODAY = date(2026, 7, 30)
DAY = "2026-07-27"          # the real two-batch dump day
ZIP = f"clips-trail_cam_sd-{DAY}.zip"

# Two batches of the same day, as the trail cam actually delivers them: the morning clips come
# off the card in one cycle, the afternoon clips (numbering restarted by the in-camera format)
# in the next.
BATCH1 = [f"{DAY}T0{i}-14-00-000_src-IMAG000{i}.mp4" for i in range(1, 7)]
BATCH2 = [f"{DAY}T1{i}-43-00-000_src-IMAG010{i}.mp4" for i in range(1, 4)]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A fake project root. backup.ROOT is a module-level constant captured from config at import
    time and used as the arcname base, so it has to be patched, not passed."""
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(backup, "ROOT", root)
    return root


@pytest.fixture
def dest(tmp_path):
    """The backup destination (Drive-synced in real life); created by archive_media."""
    return tmp_path / "dest"


def _write(day_dir: Path, names, size: int = 64) -> Path:
    """Drop `names` into a day folder as small non-empty files (content differs per file so a
    stored zip has something real to hold)."""
    day_dir.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        (day_dir / n).write_bytes(bytes([i % 251]) * size)
    return day_dir


def _clips(project: Path, names, day: str = DAY, source: str = "trail_cam_sd") -> Path:
    """Write clip files into the per-camera layout: clips/<source>/<date>/."""
    return _write(project / "clips" / source / day, names)


def _members(zip_path: Path) -> set[str]:
    """The file arcnames inside an archive (directory entries dropped)."""
    with zipfile.ZipFile(zip_path) as zf:
        return {i.filename for i in zf.infolist() if not i.is_dir()}


def _basenames(zip_path: Path) -> set[str]:
    return {m.rsplit("/", 1)[-1] for m in _members(zip_path)}


def _archive(project: Path, dest: Path, today: date = TODAY, dry_run: bool = False) -> dict:
    """archive_media over the fake project's clips/ root, as main() calls it."""
    return backup.archive_media(project / "clips", dest / "clips", today, dry_run=dry_run)


# --- the baseline: a finished day is archived once ---------------------------------------

def test_past_day_is_archived_with_project_relative_paths(project, dest):
    """Arcnames must be relative to the project root, so unzipping into it restores the layout."""
    _clips(project, BATCH1)
    stats = _archive(project, dest)

    assert stats["created"] == 1 and stats["skipped"] == 0
    assert _members(dest / "clips" / ZIP) == {
        f"clips/trail_cam_sd/{DAY}/{n}" for n in BATCH1}


def test_todays_folder_is_left_for_the_next_run(project, dest):
    """Today is still being written to; archiving it would seal a half-finished day."""
    _clips(project, BATCH1, day=TODAY.isoformat())
    stats = _archive(project, dest)

    assert stats == {"created": 0, "merged": 0, "skipped": 0, "files": 0, "bytes": 0}
    assert not (dest / "clips" / f"clips-trail_cam_sd-{TODAY.isoformat()}.zip").exists()


def test_unchanged_day_is_skipped_not_rewritten(project, dest):
    """Idempotency, and it must be free: an untouched day is not re-zipped (these are GBs)."""
    _clips(project, BATCH1)
    _archive(project, dest)
    out_zip = dest / "clips" / ZIP
    before = out_zip.stat().st_mtime_ns

    stats = _archive(project, dest)

    assert stats["created"] == 0 and stats["merged"] == 0 and stats["skipped"] == 1
    assert out_zip.stat().st_mtime_ns == before, "the archive was rewritten with no reason to be"


# --- the regression: a second import into an already-archived day ------------------------

def test_second_import_into_an_archived_day_is_merged_in(project, dest):
    """THE data-loss path. A backup runs between the two imports of one dump day; the afternoon
    clips arrive afterwards. Before the fix, that zip was sealed and those clips -- whose only
    other copy was on a card that gets formatted every cycle -- were never archived."""
    _clips(project, BATCH1)
    _archive(project, dest)
    assert _basenames(dest / "clips" / ZIP) == set(BATCH1)

    _clips(project, BATCH2)                      # the rest of 2026-07-27, imported days later
    stats = _archive(project, dest)

    assert stats["merged"] == 1 and stats["created"] == 0 and stats["files"] == len(BATCH2)
    assert _basenames(dest / "clips" / ZIP) == set(BATCH1) | set(BATCH2)


def test_merge_survives_a_prune_that_hides_the_growth(project, dest):
    """The mixed case a count-based check cannot see: four of the six archived clips have been
    pruned and three new ones imported, so the folder holds FEWER files than its archive (5 vs 6)
    while still holding three that never reached it."""
    day = _clips(project, BATCH1)
    _archive(project, dest)

    for n in BATCH1[:4]:                         # clips_max_gb eats the oldest
        (day / n).unlink()
    _clips(project, BATCH2)                      # ...and the next SD dump adds the afternoon
    assert len(list(day.iterdir())) < len(_members(dest / "clips" / ZIP))

    stats = _archive(project, dest)

    assert stats["merged"] == 1
    assert _basenames(dest / "clips" / ZIP) == set(BATCH1) | set(BATCH2)


def test_merge_never_drops_what_the_source_has_lost(project, dest):
    """The other half of the mixed case, and the reason this merges instead of rebuilding: the
    archive keeps the pruned clips even while gaining more files than it lost."""
    day = _clips(project, BATCH1)
    _archive(project, dest)

    for n in BATCH1[:2]:
        (day / n).unlink()
    _clips(project, BATCH2)                      # 7 files on disk vs 6 archived: "growth"
    _archive(project, dest)

    assert _basenames(dest / "clips" / ZIP) == set(BATCH1) | set(BATCH2)


def test_pruned_day_alone_never_shrinks_the_archive(project, dest):
    """No import, just the pruner: the archive must not be touched at all."""
    day = _clips(project, BATCH1)
    _archive(project, dest)
    out_zip = dest / "clips" / ZIP
    before = out_zip.stat().st_mtime_ns

    for n in BATCH1[:5]:
        (day / n).unlink()
    stats = _archive(project, dest)

    assert stats["skipped"] == 1 and stats["merged"] == 0
    assert out_zip.stat().st_mtime_ns == before
    assert _basenames(out_zip) == set(BATCH1)


def test_fully_pruned_day_leaves_the_archive_alone(project, dest):
    """An empty leftover day folder is not a reason to write anything -- least of all an empty zip
    over a good archive."""
    day = _clips(project, BATCH1)
    _archive(project, dest)

    for n in BATCH1:
        (day / n).unlink()
    stats = _archive(project, dest)

    assert stats == {"created": 0, "merged": 0, "skipped": 0, "files": 0, "bytes": 0}
    assert _basenames(dest / "clips" / ZIP) == set(BATCH1)


def test_merged_members_are_readable_and_intact(project, dest):
    """Appending must produce a valid archive, not just a plausible-sized file: every member
    extracts with its original bytes, old and new."""
    day = _clips(project, BATCH1)
    _archive(project, dest)
    _clips(project, BATCH2)
    _archive(project, dest)

    out_zip = dest / "clips" / ZIP
    with zipfile.ZipFile(out_zip) as zf:
        assert zf.testzip() is None
        for name in BATCH1 + BATCH2:
            arc = f"clips/trail_cam_sd/{DAY}/{name}"
            assert zf.read(arc) == (day / name).read_bytes()


def test_merge_keeps_media_stored_uncompressed(project, dest):
    """mp4/jpg are already compressed; appended members must stay STORED like the rest."""
    _clips(project, BATCH1)
    _archive(project, dest)
    _clips(project, BATCH2)
    _archive(project, dest)

    with zipfile.ZipFile(dest / "clips" / ZIP) as zf:
        assert {i.compress_type for i in zf.infolist()} == {zipfile.ZIP_STORED}


# --- the other media roots and layouts ---------------------------------------------------

def test_crops_day_backfilled_by_an_import_is_topped_up(project, dest):
    """crops/ uses the flat legacy layout (crops/<date>/) and is never pruned, but a trail-cam
    import backfills past dates there too -- same append-only path."""
    _write(project / "crops" / DAY, ["a.jpg", "b.jpg"])
    backup.archive_media(project / "crops", dest / "crops", TODAY, dry_run=False)
    _write(project / "crops" / DAY, ["c.jpg"])

    stats = backup.archive_media(project / "crops", dest / "crops", TODAY, dry_run=False)

    assert stats["merged"] == 1
    assert _basenames(dest / "crops" / f"crops-{DAY}.zip") == {"a.jpg", "b.jpg", "c.jpg"}


def test_each_camera_and_day_is_diffed_separately(project, dest):
    """Per-camera archives: a second camera's clips must not count as the first one's growth."""
    _clips(project, BATCH1)
    _clips(project, ["x.mp4"], source="glass_door_cam")
    _archive(project, dest)
    _clips(project, BATCH2)                      # only trail_cam_sd gains anything

    stats = _archive(project, dest)

    assert stats["merged"] == 1 and stats["skipped"] == 1
    assert _basenames(dest / "clips" / ZIP) == set(BATCH1) | set(BATCH2)
    assert _basenames(dest / "clips" / f"clips-glass_door_cam-{DAY}.zip") == {"x.mp4"}


# --- clips/reels/: expected company, deliberately not archived ---------------------------

# What reel.py leaves in clips/reels/: the stitched mp4, its chapter manifest, its poster frame.
REEL = "reel_2026-07-30_night_4492a4da96"
REEL_FILES = [f"{REEL}.mp4", f"{REEL}.json", f"{REEL}.jpg"]


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_reels_are_not_archived(project, dest):
    """A reel is cut from the very clips this script archives, keyed by a hash of that cut list,
    and rebuilt on demand -- so it earns no zip of its own and no room in anyone else's."""
    _clips(project, BATCH1)
    _write(project / "clips" / "reels", REEL_FILES)

    stats = _archive(project, dest)

    assert stats["created"] == 1                                   # the day, and only the day
    assert [p.name for p in sorted((dest / "clips").glob("*.zip"))] == [ZIP]
    assert _basenames(dest / "clips" / ZIP) == set(BATCH1)


def test_reels_are_passed_over_without_a_word(project, dest, caplog):
    """The noise bug (2026-08-02): reels/ is not a date folder, so day_dirs took it for a camera
    source and warned once per FILE inside -- about 60 of the run's ~75 lines, burying the INFO
    lines that say what was actually archived. Including the leftovers of a build that crashed."""
    _clips(project, BATCH1)
    _write(project / "clips" / "reels", REEL_FILES)
    _write(project / "clips" / "reels" / f".build_{REEL[-10:]}", ["seg00.mp4", "list.txt"])

    with caplog.at_level(logging.WARNING, logger="backup"):
        _archive(project, dest)

    assert _warnings(caplog) == []


def test_a_genuine_surprise_under_clips_still_warns(project, dest, caplog):
    """The warning has a job: a layout change -- or something dropped into clips/ by hand -- must
    be noticed rather than silently skipped past. Only the known cache is exempt from it."""
    _clips(project, BATCH1)
    _write(project / "clips" / "reels", REEL_FILES)
    (project / "clips" / "stray.txt").write_text("dropped here by hand")
    _write(project / "clips" / "glass_door_cam" / "scratch", ["x.mp4"])   # not a date, one down

    with caplog.at_level(logging.WARNING, logger="backup"):
        _archive(project, dest)

    warned = " | ".join(_warnings(caplog))
    assert "stray.txt" in warned and "scratch" in warned
    assert REEL not in warned


def test_day_dirs_skips_reels_without_losing_the_days(project):
    """reels/ sits beside the day folders under clips/, so pin it at the enumeration itself: it
    must not become a camera source, and the real days around it must still all be found."""
    _clips(project, ["a.mp4"], day="2026-07-24", source="glass_door_cam")
    _write(project / "clips" / "2026-07-23", ["old.mp4"])
    _write(project / "clips" / "reels", REEL_FILES)

    assert [stem for _, stem in backup.day_dirs(project / "clips")] == [
        "clips-2026-07-23", "clips-glass_door_cam-2026-07-24"]


# --- refusing to damage an archive -------------------------------------------------------

def test_unreadable_archive_is_left_untouched(project, dest):
    """If we cannot read what an archive holds we cannot know what is missing from it -- and a
    corrupt-looking file may still be recoverable, so it is never overwritten."""
    _clips(project, BATCH1)
    _archive(project, dest)
    out_zip = dest / "clips" / ZIP
    out_zip.write_bytes(b"not a zip at all")
    _clips(project, BATCH2)

    stats = _archive(project, dest)

    assert stats["skipped"] == 1 and stats["merged"] == 0 and stats["created"] == 0
    assert out_zip.read_bytes() == b"not a zip at all"


def test_leftover_tmp_from_an_interrupted_run_is_cleared(project, dest):
    """A half-written .tmp must never be mistaken for an archive; the next run sweeps it."""
    _clips(project, BATCH1)
    (dest / "clips").mkdir(parents=True)
    stale = dest / "clips" / f"{ZIP}.tmp"
    stale.write_bytes(b"half a zip")

    _archive(project, dest)

    assert not stale.exists()
    assert _basenames(dest / "clips" / ZIP) == set(BATCH1)


def test_merge_leaves_no_tmp_behind(project, dest):
    """The merge works on a copy; once it lands, the copy must be gone."""
    _clips(project, BATCH1)
    _archive(project, dest)
    _clips(project, BATCH2)
    _archive(project, dest)

    assert list((dest / "clips").glob("*.tmp")) == []


# --- dry run -----------------------------------------------------------------------------

def test_dry_run_writes_nothing_on_either_path(project, dest):
    """--dry-run must report both a new day and a top-up without touching the destination."""
    _clips(project, BATCH1)
    _archive(project, dest)
    before = (dest / "clips" / ZIP).read_bytes()
    _clips(project, BATCH2)

    stats = _archive(project, dest, dry_run=True)

    assert stats["merged"] == 1 and stats["files"] == len(BATCH2)   # reported...
    assert (dest / "clips" / ZIP).read_bytes() == before            # ...but not written
    assert list((dest / "clips").glob("*.tmp")) == []


def test_dry_run_does_not_create_a_new_archive(project, dest):
    _clips(project, BATCH1)

    stats = _archive(project, dest, dry_run=True)

    assert stats["created"] == 1
    assert not (dest / "clips" / ZIP).exists()


# --- day_dirs: both clip layouts ---------------------------------------------------------

def test_day_dirs_handles_flat_and_per_camera_layouts(project):
    """Days written before the multi-camera split live at clips/<date>/; everything since lives at
    clips/<source>/<date>/. Both are archived, with the camera in the name where there is one."""
    _write(project / "clips" / "2026-06-09", ["old.mp4"])            # legacy flat
    _clips(project, ["new.mp4"], day="2026-06-26", source="glass_door_cam")

    assert [stem for _, stem in backup.day_dirs(project / "clips")] == [
        "clips-2026-06-09", "clips-glass_door_cam-2026-06-26"]


def test_day_dirs_returns_oldest_day_first(project):
    """Oldest first, across cameras: the days nearest the pruning axe are archived before it falls."""
    _clips(project, ["b.mp4"], day="2026-07-24")
    _clips(project, ["a.mp4"], day="2026-07-22", source="glass_door_cam")
    _write(project / "clips" / "2026-07-23", ["c.mp4"])

    assert [d.name for d, _ in backup.day_dirs(project / "clips")] == [
        "2026-07-22", "2026-07-23", "2026-07-24"]
