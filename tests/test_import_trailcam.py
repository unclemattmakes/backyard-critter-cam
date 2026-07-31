"""
Tests for the trail-cam batch importer (import_trailcam).

Two regression guards live here:
  * the save_crop contract: save_crop() returns (Path, shot-quality), and the importer must
    unpack that tuple. A past bug passed the whole tuple straight to _rel(), so the importer
    crashed on the first real detection and silently saved nothing -- and no test covered it.
  * cross-cycle idempotency: a TC02 restarts numbering at IMAG0001.JPG after every in-camera
    card format, and a skip-set keyed on the bare basename silently dropped a later cycle's real
    files (hit for real 2026-07-19: 235 of 558). Skip keys are now basename|capture-second;
    pre-fix bare ledger lines are inert, with the files they recorded still covered by the
    capture stamp that has always led the crop filename.

No GPU/camera/model here: the detector is a tiny stub that returns one fixed box. The synthetic
JPEGs carry no EXIF, so image_timestamp() takes its mtime fallback -- tests pin mtimes to pin
capture times.
"""
from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
import db
import import_trailcam
from detector import Detection


class _StubDetector:
    """Stand-in for detector.Detector: returns one fixed 'animal' box, no torch/GPU."""

    def __init__(self, class_name: str = "animal", class_id: int = 0):
        self.class_name, self.class_id = class_name, class_id

    def detect(self, frame):
        h, w = frame.shape[:2]
        return [Detection(class_id=self.class_id, class_name=self.class_name, confidence=0.91,
                          bbox=(w * 0.1, h * 0.1, w * 0.6, h * 0.6))]


def _write_image(path: Path, size=(120, 160)) -> Path:
    """A small noisy BGR JPEG on disk (noisy so the crop has real shot-quality, not 0)."""
    rng = np.random.default_rng(0)
    img = (rng.random((size[0], size[1], 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _set_mtime(path: Path, when: str) -> None:
    """Pin a file's mtime ('YYYY-mm-dd HH:MM:SS', local). With no EXIF in the synthetic JPEGs,
    the mtime IS the capture time the importer keys on."""
    t = datetime.strptime(when, "%Y-%m-%d %H:%M:%S").timestamp()
    os.utime(path, (t, t))


def _ts(path: Path) -> str:
    """The file's capture second, formatted exactly as the importer keys it."""
    return import_trailcam.image_timestamp(path).strftime(import_trailcam.KEY_TS_FMT)


def _seeded_skip(conn, db_path) -> set[str]:
    """The startup skip-set exactly as main() seeds it: ledger UNION DB crop-path recovery."""
    return (import_trailcam.read_ledger(db_path, "trail_cam_sd")
            | import_trailcam.imported_keys(conn, "trail_cam_sd"))


def _import(folder: Path, conn, cfg, skip: set[str]):
    """import_folder with the boilerplate pinned down (one source, no recursion, no move)."""
    return import_trailcam.import_folder(
        folder, _StubDetector(), conn, cfg, source="trail_cam_sd", recursive=False,
        processed_dir=None, skip=skip)


def test_ingest_file_saves_crop_with_quality(conn, tmp_path):
    """One animal detection writes exactly one row, tagged source='trail_cam_sd', with a non-NULL
    crop_quality and an on-disk crop. (Pre-fix this raised AttributeError on _rel(tuple).)"""
    cfg = replace(config.CONFIG, crops_dir=tmp_path / "crops")
    img = _write_image(tmp_path / "IMG_0001.JPG")

    n_reported, n_saved = import_trailcam.ingest_file(
        img, _StubDetector(), conn, cfg, source="trail_cam_sd")

    assert (n_reported, n_saved) == (1, 1)
    row = conn.execute(
        "SELECT source, crop_path, crop_quality, detection_class FROM detections"
    ).fetchone()
    assert row["source"] == "trail_cam_sd"
    assert row["detection_class"] == "animal"
    assert row["crop_quality"] is not None        # the tuple-unpack bug dropped this on the floor

    saved = Path(row["crop_path"])
    if not saved.is_absolute():
        saved = config.ROOT / saved
    assert saved.exists()


def test_ingest_file_skips_non_saved_classes(conn, tmp_path):
    """A detection whose class isn't in save_classes (default = animals only) is reported but not
    saved -- (1 reported, 0 saved) and no DB row."""
    cfg = replace(config.CONFIG, crops_dir=tmp_path / "crops")
    img = _write_image(tmp_path / "IMG_0002.JPG")

    n_reported, n_saved = import_trailcam.ingest_file(
        img, _StubDetector(class_name="person", class_id=1), conn, cfg, source="trail_cam_sd")

    assert (n_reported, n_saved) == (1, 0)
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0


def test_ingest_file_handles_unreadable_image(conn, tmp_path):
    """A missing/corrupt file is reported as the (-1, 0) sentinel and writes nothing, so the batch
    loop can warn-and-skip rather than crash."""
    cfg = replace(config.CONFIG, crops_dir=tmp_path / "crops")
    bogus = tmp_path / "not_an_image.JPG"
    bogus.write_bytes(b"this is not a JPEG")

    n_reported, n_saved = import_trailcam.ingest_file(
        bogus, _StubDetector(), conn, cfg, source="trail_cam_sd")

    assert (n_reported, n_saved) == (-1, 0)
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0


def test_reimport_recovers_underscored_stem(conn, tmp_path):
    """DB recovery (the fallback when the ledger sidecar is gone) must rebuild the full
    stem|capture-second key for a file whose name carries an internal underscore. A past bug
    recovered only the prefix before the first underscore (IMG_0042 -> 'IMG') AND compared
    against the full filename, so the DB-recovery skip-set never matched and a re-run re-imported
    (duplicated) everything."""
    cfg = replace(config.CONFIG, crops_dir=tmp_path / "crops")
    img = _write_image(tmp_path / "IMG_0042.JPG")
    import_trailcam.ingest_file(img, _StubDetector(), conn, cfg, source="trail_cam_sd")

    recovered = import_trailcam.imported_keys(conn, "trail_cam_sd")
    # The FULL stem (not the truncated 'IMG'), paired with the capture second parsed back out of
    # the stamp that leads the crop filename. import_folder checks the stem spelling of the key,
    # so this is exactly what makes a ledger-less re-run skip the file.
    assert import_trailcam.skip_key("IMG_0042", _ts(img)) in recovered


def test_new_cycle_reusing_basenames_still_imports(conn, db_path, tmp_path):
    """THE cross-cycle collision (2026-07-19): after an in-camera card format the TC02 restarts
    numbering, so a later dump reuses earlier filenames. Same name + different capture second
    must IMPORT; a plain re-run of the same dump must still skip."""
    cfg = replace(config.CONFIG, crops_dir=tmp_path / "crops", db_path=db_path)

    cycle1 = tmp_path / "cycle1"
    cycle1.mkdir()
    f1 = _write_image(cycle1 / "IMAG0001.JPG")
    _set_mtime(f1, "2026-07-13 06:42:11")
    imported, _, skipped = _import(cycle1, conn, cfg, _seeded_skip(conn, db_path))
    assert (imported, skipped) == (1, 0)

    # Post-format cycle 2: a DIFFERENT photo wearing the same name, six days later.
    cycle2 = tmp_path / "cycle2"
    cycle2.mkdir()
    f2 = _write_image(cycle2 / "IMAG0001.JPG", size=(90, 130))
    _set_mtime(f2, "2026-07-19 20:05:33")
    imported, _, skipped = _import(cycle2, conn, cfg, _seeded_skip(conn, db_path))
    assert (imported, skipped) == (1, 0)   # keyed on the bare basename this was (0, 1): data loss

    # Idempotency is intact: re-running cycle 2 (fresh seed, like a new process) is a no-op.
    imported, _, skipped = _import(cycle2, conn, cfg, _seeded_skip(conn, db_path))
    assert (imported, skipped) == (0, 1)
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 2


def test_legacy_bare_ledger_lines_are_inert_but_db_recovery_holds(conn, db_path, tmp_path):
    """Backward compat with a pre-fix ledger. Plain-basename lines must not block anything by
    themselves (that name-only match IS the collision bug) -- but a file genuinely imported
    pre-fix still skips, because the capture stamp was always embedded in its crop name and
    imported_keys() rebuilds the stem|capture-second pair from the DB."""
    cfg = replace(config.CONFIG, crops_dir=tmp_path / "crops", db_path=db_path)
    dump = tmp_path / "dump"
    dump.mkdir()
    img = _write_image(dump / "IMAG0007.JPG")
    _set_mtime(img, "2026-07-13 07:00:00")

    # Simulate the pre-fix world: ingest wrote the DB row + crop, the ledger got the BARE name.
    import_trailcam.ingest_file(img, _StubDetector(), conn, cfg, source="trail_cam_sd")
    ledger = import_trailcam.ledger_path(db_path, "trail_cam_sd")
    ledger.write_text("IMAG0007.JPG\n", encoding="utf-8")

    assert import_trailcam.read_ledger(db_path, "trail_cam_sd") == set()   # bare line: inert
    imported, _, skipped = _import(dump, conn, cfg, _seeded_skip(conn, db_path))
    assert (imported, skipped) == (0, 1)     # same file, same capture second -> still skipped
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 1  # no duplicate row


def test_ledger_roundtrip_mixed_with_legacy_lines(db_path):
    """append_ledger writes 'name|capture-second' lines; read_ledger returns exactly those and
    ignores pre-fix bare-basename lines sharing the file."""
    key = import_trailcam.skip_key("IMAG0001.JPG", "2026-07-13T06-42-11")
    import_trailcam.append_ledger(db_path, "trail_cam_sd", key)
    with open(import_trailcam.ledger_path(db_path, "trail_cam_sd"), "a", encoding="utf-8") as f:
        f.write("IMAG0002.JPG\n")                      # a line as written before the fix
    assert import_trailcam.read_ledger(db_path, "trail_cam_sd") == {key}


# --- videos: the behaviour clips that used to be left on the card -----------------------

def _write_video(path: Path, *, seconds=2.0, fps=10.0, size=(120, 160)) -> Path:
    """A small real .mp4 on disk (mp4v so no ffmpeg is required in CI), written frame by frame so
    _probe_video's cv2 fallback reports an honest duration."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, fps, (size[1], size[0]))
    rng = np.random.default_rng(1)
    for _ in range(int(seconds * fps)):
        w.write((rng.random((size[0], size[1], 3)) * 255).astype(np.uint8))
    w.release()
    return path


def _video_cfg(tmp_path, db_path):
    """A Config pinned ENTIRELY inside tmp -- crops, clips, and db_path. db_path matters more than
    it looks: ledger_path() derives the sidecar import ledger from it, so a cfg that keeps the real
    CONFIG.db_path makes tests append to (and, for a deleted-ledger test, unlink) the project's
    live ledger. Injecting a stale key there silently retires a real file from the next import."""
    return replace(config.CONFIG, crops_dir=tmp_path / "crops", clips_dir=tmp_path / "clips",
                   db_path=db_path,
                   clips_max_gb=0)          # 0 = no cap, so tests never race the pruner


def _import_videos(folder, conn, cfg, skip, *, require_animal=True, window_s=30.0):
    return import_trailcam.import_videos(
        folder, conn, cfg, source="trail_cam_sd", recursive=False, skip=skip,
        window_s=window_s, require_animal=require_animal)


def _seed_detection(conn, when: str, conf=0.9):
    """One trail-cam detection at `when` ('YYYY-mm-dd HH:MM:SS' local) -- a video's animal gate."""
    iso = datetime.strptime(when, "%Y-%m-%d %H:%M:%S").astimezone().isoformat()
    return db.insert_detection(
        conn, timestamp=iso, source="trail_cam_sd", detection_class="animal", confidence=conf,
        bbox=(0, 0, 10, 10), frame_w=100, frame_h=100, crop_path="crops/x.jpg")


def test_video_span_ends_at_mtime(tmp_path):
    """A trail cam's mtime is when it FINISHED writing the clip, so started_at is mtime minus
    duration. Getting this backwards would file every clip ~12s late and break the pairing."""
    v = _write_video(tmp_path / "IMAG0007.MP4", seconds=2.0, fps=10.0)
    _set_mtime(v, "2026-07-22 21:29:42")
    meta = import_trailcam._probe_video(v)
    assert meta is not None
    start, end = import_trailcam.video_span(v, meta)
    assert end.strftime("%H:%M:%S") == "21:29:42"
    assert start.strftime("%H:%M:%S") == "21:29:40"      # 2s clip -> starts two seconds earlier


def test_video_with_animal_is_copied_and_rowed(conn, db_path, tmp_path):
    """The happy path: a clip whose trigger produced a detection is COPIED into clips/<source>/
    (never referenced on the card, which gets formatted) and gets a clips row carrying the
    trigger's detection count."""
    cfg = _video_cfg(tmp_path, db_path)
    v = _write_video(tmp_path / "IMAG0007.MP4", seconds=2.0, fps=10.0)
    _set_mtime(v, "2026-07-22 21:29:42")
    _seed_detection(conn, "2026-07-22 21:29:38")        # the still burst, just before the video

    stored, already, empty = _import_videos(tmp_path, conn, cfg, set())
    assert (stored, already, empty) == (1, 0, 0)

    row = conn.execute("SELECT source, clip_path, started_at, ended_at, width, height, "
                       "frame_count, detection_count FROM clips").fetchone()
    assert row["source"] == "trail_cam_sd"
    assert row["detection_count"] == 1
    assert row["width"] == 160 and row["height"] == 120
    assert row["started_at"] < row["ended_at"]
    dest = config.ROOT / row["clip_path"] if not Path(row["clip_path"]).is_absolute() else Path(row["clip_path"])
    assert dest.exists()
    assert dest.parent == cfg.clips_dir / "trail_cam_sd" / "2026-07-22"
    assert import_trailcam.SRC_TAG + "IMAG0007" in dest.name   # traceable back to the card
    assert v.exists()                                          # the card is never modified


def test_video_without_animal_is_skipped(conn, db_path, tmp_path):
    """An empty sun/wind trigger writes no clip and costs no disk -- but IS ledgered, so the next
    run doesn't re-probe it to reach the same verdict."""
    cfg = _video_cfg(tmp_path, db_path)
    v = _write_video(tmp_path / "IMAG0009.MP4")
    _set_mtime(v, "2026-07-22 13:00:00")
    _seed_detection(conn, "2026-07-22 21:29:38")        # hours away -- not this trigger

    skip = set()
    assert _import_videos(tmp_path, conn, cfg, skip) == (0, 0, 1)
    assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 0
    assert not (cfg.clips_dir / "trail_cam_sd").exists()
    assert import_trailcam.video_skip_key(v) in skip


def test_all_videos_overrides_the_animal_gate(conn, db_path, tmp_path):
    cfg = _video_cfg(tmp_path, db_path)
    v = _write_video(tmp_path / "IMAG0009.MP4")
    _set_mtime(v, "2026-07-22 13:00:00")
    assert _import_videos(tmp_path, conn, cfg, set(), require_animal=False) == (1, 0, 0)
    assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 1


def test_video_import_is_idempotent_via_ledger_and_db(conn, db_path, tmp_path):
    """Re-running over the same card must not duplicate clips -- through the sidecar ledger AND,
    if that is deleted, through the clips table (the same belt-and-braces the stills have)."""
    cfg = _video_cfg(tmp_path, db_path)
    v = _write_video(tmp_path / "IMAG0007.MP4")
    _set_mtime(v, "2026-07-22 21:29:42")
    _seed_detection(conn, "2026-07-22 21:29:38")

    skip = set()
    assert _import_videos(tmp_path, conn, cfg, skip)[0] == 1
    # Second pass, same in-memory skip-set.
    assert _import_videos(tmp_path, conn, cfg, skip) == (0, 1, 0)
    # Third pass with the ledger thrown away: DB recovery alone must still skip it.
    import_trailcam.ledger_path(cfg.db_path, "trail_cam_sd").unlink(missing_ok=True)
    recovered = import_trailcam.imported_video_keys(conn, "trail_cam_sd")
    assert _import_videos(tmp_path, conn, cfg, recovered) == (0, 1, 0)
    assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 1


def test_unreadable_video_is_retried_not_retired(conn, db_path, tmp_path):
    """A torn/locked file probes as unreadable. It must NOT be ledgered -- retiring a real clip
    on the strength of one bad read is exactly the kind of silent loss the stills path avoids."""
    cfg = _video_cfg(tmp_path, db_path)
    bad = tmp_path / "IMAG0011.MP4"
    bad.write_bytes(b"not really an mp4")
    skip = set()
    assert _import_videos(tmp_path, conn, cfg, skip) == (0, 0, 0)
    assert skip == set()
    assert import_trailcam.read_ledger(cfg.db_path, "trail_cam_sd") == set()


def test_stills_ledger_and_video_ledger_do_not_collide(conn, db_path, tmp_path):
    """IMAG0007.JPG and IMAG0007.MP4 share a stem and can share a second. Their keys must stay
    distinct, or importing one would silently retire the other."""
    cfg = _video_cfg(tmp_path, db_path)
    img = _write_image(tmp_path / "IMAG0007.JPG")
    _set_mtime(img, "2026-07-22 21:29:42")
    vid = _write_video(tmp_path / "IMAG0007.MP4")
    _set_mtime(vid, "2026-07-22 21:29:42")

    skip = _seeded_skip(conn, cfg.db_path)
    imported, saved, _ = _import(tmp_path, conn, cfg, skip)
    assert (imported, saved) == (1, 1)                  # the still went in ...
    stored, already, _ = _import_videos(tmp_path, conn, cfg, skip)
    assert (stored, already) == (1, 0)                  # ... and did NOT mask the video


# ---- --backup-first ------------------------------------------------------------------------
# run_backup_first's RETURN VALUE is the only thing standing between a failed archive and a prune
# that deletes clips the card no longer holds (it gets formatted every cycle). Every path that
# didn't demonstrably archive must read False, so main() skips the prune. False on doubt.

def _patched_backup(monkeypatch, result):
    """Run run_backup_first with subprocess.run stubbed to `result` (a CompletedProcess, or an
    Exception to raise). Returns (verdict, argv_of_the_call)."""
    import subprocess
    seen: list = []

    def fake_run(argv, **kw):
        seen.append(argv)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    verdict = import_trailcam.run_backup_first(replace(config.CONFIG, backup_dest=Path("dest")))
    return verdict, (seen[0] if seen else None)


def test_backup_first_reports_success(monkeypatch):
    """A clean backup returns True -- and shells out to backup.py with THIS interpreter, so the
    archiver runs in the same venv (a bare 'python' could miss the project's deps entirely)."""
    import subprocess
    import sys as _sys
    ok, argv = _patched_backup(monkeypatch, subprocess.CompletedProcess([], 0))
    assert ok is True
    assert argv[0] == _sys.executable and Path(argv[1]).name == "backup.py"


def test_backup_first_reports_failure_so_the_prune_is_skipped(monkeypatch):
    """A non-zero exit must read False. Treating a failed archive as success is exactly how the
    budget would get enforced against footage that exists nowhere else."""
    import subprocess
    ok, _ = _patched_backup(monkeypatch, subprocess.CompletedProcess([], 1))
    assert ok is False


def test_backup_first_survives_a_crashing_archiver(monkeypatch):
    """backup.py missing / unlaunchable raises rather than returning -- that must be False too,
    not an exception that aborts an import the card is waiting on."""
    ok, _ = _patched_backup(monkeypatch, OSError("no such file"))
    assert ok is False
