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

Plus the burst full-frame retention (2026-08-07): one kept frame per trigger, its path on that
trigger's rows, nothing kept for a trigger with no surviving detection. That last one is the
sharp edge -- staticfilter deletes rows AFTER the still pass, so the frames it orphans have to
go with them. Retention is a prerequisite for the reference-image veto, NOT the veto: nothing in
the importer judges a box against a reference, and these tests would notice if it started.

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
import staticfilter
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


def _import_videos(folder, conn, cfg, skip, *, require_animal=True, window_s=30.0,
                   trust_no_animal=True, deferred=None):
    return import_trailcam.import_videos(
        folder, conn, cfg, source="trail_cam_sd", recursive=False, skip=skip,
        window_s=window_s, require_animal=require_animal,
        trust_no_animal=trust_no_animal, deferred=deferred)


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


# --- unread stills poison the clip gate (2026-08-14) -------------------------------------
#
# The still pass feeds the video pass: "no animal in the trigger" is decided by counting the
# detections the stills wrote. A still that never loaded therefore looks exactly like a still of
# an empty yard -- and that verdict used to be LEDGERED, retiring the clip forever. One mid-run
# pip install made 121 stills unreadable and wrote off 12 videos; 11 held raccoons.

def _fail_reading(monkeypatch, *names: str, exc=None):
    """Make image_timestamp raise for `names`, exactly as a sick reader (or a broken interpreter)
    does mid-scan, while every other file reads normally."""
    real = import_trailcam.image_timestamp
    bad = set(names)

    def fake(path):
        if path.name in bad:
            raise exc or OSError(22, "Invalid argument")
        return real(path)

    monkeypatch.setattr(import_trailcam, "image_timestamp", fake)


def test_describe_read_failure_separates_gone_from_still_there(tmp_path):
    """The old message asserted 'vanished mid-scan' for both cases and threw the exception away,
    which is why the real failure could not be diagnosed afterwards. Present-but-unreadable is
    the alarming one and must not be reported as a benign race."""
    here = _write_image(tmp_path / "IMAG0001.JPG")
    gone = tmp_path / "IMAG0002.JPG"

    present = import_trailcam.describe_read_failure(here, OSError(22, "Invalid argument"))
    assert "STILL THERE" in present
    assert "Invalid argument" in present          # the evidence survives

    absent = import_trailcam.describe_read_failure(gone, FileNotFoundError(2, "nope"))
    assert "GONE" in absent


def test_unreadable_still_is_collected_and_left_unledgered(conn, db_path, tmp_path, monkeypatch):
    """A still that cannot be read is skipped (one bad file never aborts the batch) but is
    REPORTED to the caller and stays out of the ledger, so a re-run picks it up."""
    cfg = _video_cfg(tmp_path, db_path)
    good = _write_image(tmp_path / "IMAG0001.JPG")
    bad = _write_image(tmp_path / "IMAG0002.JPG")
    _set_mtime(good, "2026-08-14 06:48:36")
    _set_mtime(bad, "2026-08-14 06:49:00")
    _fail_reading(monkeypatch, "IMAG0002.JPG")

    failures: list[str] = []
    imported, saved, _ = import_trailcam.import_folder(
        tmp_path, _StubDetector(), conn, cfg, source="trail_cam_sd", recursive=False,
        processed_dir=None, skip=set(), read_failures=failures)

    assert imported == 1 and saved == 1                     # the readable one still imported
    assert len(failures) == 1 and "IMAG0002.JPG" in failures[0]
    assert "STILL THERE" in failures[0]                     # it was never actually missing
    ledger = import_trailcam.read_ledger(cfg.db_path, "trail_cam_sd")
    assert not any(k.startswith("IMAG0002") for k in ledger)


def test_unproven_no_animal_video_is_not_ledgered(conn, db_path, tmp_path):
    """THE REGRESSION. With the still pass known to be incomplete, an empty-looking trigger is
    still skipped for this run but must NOT be settled: no ledger line, no skip-key, so the next
    run re-probes it once the stills are actually in."""
    cfg = _video_cfg(tmp_path, db_path)
    v = _write_video(tmp_path / "IMAG0009.MP4")
    _set_mtime(v, "2026-08-14 06:50:22")                    # no detection seeded for this trigger

    skip: set[str] = set()
    deferred: list[str] = []
    assert _import_videos(tmp_path, conn, cfg, skip,
                          trust_no_animal=False, deferred=deferred) == (0, 0, 1)
    assert deferred == ["IMAG0009.MP4"]
    assert skip == set()                                    # nothing retired
    assert import_trailcam.read_ledger(cfg.db_path, "trail_cam_sd") == set()
    assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 0

    # ...and the re-run, once the stills are in, actually stores it.
    _seed_detection(conn, "2026-08-14 06:50:20")
    assert _import_videos(tmp_path, conn, cfg, skip)[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 1


def test_unproven_run_still_ledgers_videos_that_DID_find_an_animal(conn, db_path, tmp_path):
    """Only the negative verdicts are held back. A clip whose trigger produced a crop rests on
    evidence that exists, so it stores and ledgers normally even on a run with read failures --
    otherwise every incomplete run would re-copy gigabytes it already has."""
    cfg = _video_cfg(tmp_path, db_path)
    v = _write_video(tmp_path / "IMAG0007.MP4", seconds=2.0, fps=10.0)
    _set_mtime(v, "2026-08-14 06:51:42")
    _seed_detection(conn, "2026-08-14 06:51:38")

    skip: set[str] = set()
    deferred: list[str] = []
    assert _import_videos(tmp_path, conn, cfg, skip,
                          trust_no_animal=False, deferred=deferred) == (1, 0, 0)
    assert deferred == []
    assert import_trailcam.video_skip_key(v) in skip


def test_read_failure_flows_through_to_the_clip_gate(conn, db_path, tmp_path, monkeypatch):
    """The two halves wired the way main() wires them (trust_no_animal=not read_failures) -- the
    seam the original bug lived in. An unreadable still must make that trigger's video unproven
    rather than empty."""
    cfg = _video_cfg(tmp_path, db_path)
    still = _write_image(tmp_path / "IMAG3119.JPG")
    _set_mtime(still, "2026-08-14 06:50:20")
    video = _write_video(tmp_path / "IMAG3131.MP4")
    _set_mtime(video, "2026-08-14 06:50:22")
    _fail_reading(monkeypatch, "IMAG3119.JPG")

    failures: list[str] = []
    skip: set[str] = set()
    import_trailcam.import_folder(
        tmp_path, _StubDetector(), conn, cfg, source="trail_cam_sd", recursive=False,
        processed_dir=None, skip=skip, read_failures=failures)
    assert failures                                          # the still never made it in

    deferred: list[str] = []
    _import_videos(tmp_path, conn, cfg, skip,
                   trust_no_animal=not failures, deferred=deferred)

    assert deferred == ["IMAG3131.MP4"]                      # held open, not written off
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


# ---- one full frame per still burst (BurstFrames) --------------------------------------------
# The prerequisite from docs/refimg-design-2026-08-07.md section 6: a still detection on this
# camera has never had a full frame, so a reference-image veto has nothing of the same sensor path
# to compare against. Retention ONLY -- no veto runs at import time, and these tests should fail
# loudly if one ever starts to.

def _burst_cfg(tmp_path, db_path, **over):
    """A Config pinned entirely inside tmp -- crops, FRAMES, clips and db_path (the last because
    ledger_path() derives the import ledger from it; see _video_cfg)."""
    return replace(config.CONFIG, crops_dir=tmp_path / "crops", frames_dir=tmp_path / "frames",
                   clips_dir=tmp_path / "clips", db_path=db_path, **over)


def _kept_frames(cfg) -> list[Path]:
    """Every retained full frame on disk, sorted -- from the per-source subdir the importer uses."""
    root = cfg.frames_dir / "trail_cam_sd"
    return sorted(p for p in root.rglob("*.jpg")) if root.is_dir() else []


def _burst_dump(folder: Path, stamps: list[str], *, first=1, size=(120, 160)) -> list[Path]:
    """One JPEG per capture stamp, named so filename order matches capture order (which is what a
    real card does, and what import_folder walks in)."""
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i, when in enumerate(stamps, start=first):
        p = _write_image(folder / f"IMAG{i:04d}.JPG", size=size)
        _set_mtime(p, when)
        out.append(p)
    return out


def _import_bursts(folder, conn, cfg, bursts, skip=None):
    return import_trailcam.import_folder(
        folder, _StubDetector(), conn, cfg, source="trail_cam_sd", recursive=False,
        processed_dir=None, skip=set() if skip is None else skip, bursts=bursts)


def test_one_frame_per_burst_and_its_path_on_every_row_of_that_burst(conn, db_path, tmp_path):
    """The whole feature in one test: three triggers of three stills each keep exactly THREE
    frames (not nine), each frame is the burst's first still, and every detection of a burst
    carries that burst's frame_path. Bursts are 1 s apart within a trigger and 5 minutes apart
    between them -- the real card's gaps are 0-5 s and 28 s+, so this is not a marginal split."""
    cfg = _burst_cfg(tmp_path, db_path)
    dump = tmp_path / "dump"
    _burst_dump(dump, ["2026-08-06 21:00:00", "2026-08-06 21:00:01", "2026-08-06 21:00:02",
                       "2026-08-06 21:05:00", "2026-08-06 21:05:01", "2026-08-06 21:05:02",
                       "2026-08-06 21:10:00", "2026-08-06 21:10:01", "2026-08-06 21:10:02"])

    bursts = import_trailcam.BurstFrames(cfg, "trail_cam_sd")
    imported, saved, _ = _import_bursts(dump, conn, cfg, bursts)
    assert (imported, saved) == (9, 9)

    frames = _kept_frames(cfg)
    assert len(frames) == 3 and bursts.bursts == 3        # one per BURST, not one per still
    # ... and it is the first still of each trigger, written from the array already decoded.
    assert [f.name.split(import_trailcam.SRC_TAG)[1] for f in frames] == \
        ["IMAG0001.jpg", "IMAG0004.jpg", "IMAG0007.jpg"]
    # frames/<source>/<date>/ -- the clips layout, which backup.py's day_dirs() already walks.
    assert {f.parent for f in frames} == {cfg.frames_dir / "trail_cam_sd" / "2026-08-06"}

    rows = conn.execute("SELECT frame_path FROM detections ORDER BY id").fetchall()
    assert len(rows) == 9
    paths = [r["frame_path"] for r in rows]
    assert all(p for p in paths)                          # no row left without its frame
    assert [paths[0]] * 3 == paths[0:3]                   # each burst's three rows share one frame
    assert [paths[3]] * 3 == paths[3:6]
    assert [paths[6]] * 3 == paths[6:9]
    assert len(set(paths)) == 3
    assert {config.ROOT / p for p in set(paths)} == set(frames)


def test_burst_frames_off_writes_no_frame_and_no_path(conn, db_path, tmp_path):
    """The knob (config.trailcam_keep_burst_frame / --no-burst-frames). Off means the importer
    behaves exactly as it always has: frame_path NULL, and frames/ never even created."""
    cfg = _burst_cfg(tmp_path, db_path, trailcam_keep_burst_frame=False)
    dump = tmp_path / "dump"
    _burst_dump(dump, ["2026-08-06 21:00:00", "2026-08-06 21:00:01"])

    bursts = import_trailcam.BurstFrames(cfg, "trail_cam_sd")
    assert bursts.enabled is False
    assert _import_bursts(dump, conn, cfg, bursts)[:2] == (2, 2)

    assert not cfg.frames_dir.exists()
    assert bursts.written == [] and bursts.bytes_written == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM detections WHERE frame_path IS NOT NULL").fetchone()[0] == 0


def test_burst_with_no_saveable_detection_keeps_no_frame(conn, db_path, tmp_path):
    """A trigger the detector found nothing savable in writes no row, so there is nothing for a
    frame to serve -- and none is written. The save is lazy for exactly this reason: a sun-storm
    card is mostly empty triggers, and banking a picture of the yard for each is not insurance."""
    cfg = _burst_cfg(tmp_path, db_path)
    dump = tmp_path / "dump"
    _burst_dump(dump, ["2026-08-06 13:00:00", "2026-08-06 13:00:01"])

    bursts = import_trailcam.BurstFrames(cfg, "trail_cam_sd")
    imported, saved, _ = import_trailcam.import_folder(
        dump, _StubDetector(class_name="person", class_id=1), conn, cfg, source="trail_cam_sd",
        recursive=False, processed_dir=None, skip=set(), bursts=bursts)

    assert (imported, saved) == (2, 0)
    assert _kept_frames(cfg) == []
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0


def test_burst_retention_leaves_reimport_idempotency_alone(conn, db_path, tmp_path):
    """Retention must not touch the skip keys. The ledger still holds exactly 'name|capture-
    second' lines and nothing else, a re-run is still a no-op, and it writes no SECOND copy of
    the burst's frame (which would double the cost of every re-scan of a card)."""
    cfg = _burst_cfg(tmp_path, db_path)
    dump = tmp_path / "dump"
    files = _burst_dump(dump, ["2026-08-06 21:00:00", "2026-08-06 21:00:01"])
    expected_keys = {import_trailcam.skip_key(f.name, _ts(f)) for f in files}

    bursts = import_trailcam.BurstFrames(cfg, "trail_cam_sd")
    assert _import_bursts(dump, conn, cfg, bursts, _seeded_skip(conn, db_path))[:2] == (2, 2)
    assert import_trailcam.read_ledger(db_path, "trail_cam_sd") == expected_keys
    first = _kept_frames(cfg)
    assert len(first) == 1

    # A fresh process: new BurstFrames, skip-set re-seeded from the ledger UNION the DB crops.
    again = import_trailcam.BurstFrames(cfg, "trail_cam_sd")
    imported, saved, skipped = _import_bursts(dump, conn, cfg, again,
                                              _seeded_skip(conn, db_path))
    assert (imported, saved, skipped) == (0, 0, 2)
    assert again.written == [] and again.bursts == 0      # skipped files never open a burst
    assert _kept_frames(cfg) == first                     # no duplicate frame on disk
    assert import_trailcam.read_ledger(db_path, "trail_cam_sd") == expected_keys
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 2


def test_static_filtered_burst_keeps_zero_frames(conn, db_path, tmp_path):
    """No animal, no point. staticfilter runs AFTER the still pass and DELETES the rows it judges
    furniture, so a trigger whose only 'animal' was the grill is left holding a frame nothing
    points at. Here 16 stills hold one identical box across 75 minutes (furniture by the filter's
    own rule) while a separate 3-still burst carries a different box -- the furniture frames go,
    the real one stays, and the day folder emptied by the sweep is tidied up too."""
    cfg = _burst_cfg(tmp_path, db_path)
    dump = tmp_path / "dump"
    # The static spot: one still every 5 minutes -> 16 separate bursts, 16 retained frames.
    _burst_dump(dump, [f"2026-08-05 {12 + (i * 5) // 60:02d}:{(i * 5) % 60:02d}:00"
                       for i in range(16)])
    # A real trigger the next evening. A different image SIZE means a different box, so the
    # static filter clusters it apart (IoU ~0.51, under its 0.75 bar) and its count of 3 is
    # nowhere near the 15 it needs anyway.
    _burst_dump(dump, ["2026-08-06 20:00:00", "2026-08-06 20:00:01", "2026-08-06 20:00:02"],
                first=100, size=(90, 130))

    bursts = import_trailcam.BurstFrames(cfg, "trail_cam_sd")
    imported, saved, _ = _import_bursts(dump, conn, cfg, bursts)
    assert (imported, saved) == (19, 19)
    assert len(_kept_frames(cfg)) == 17            # 16 furniture bursts + 1 real one

    dropped = staticfilter.sweep_batch(conn, cfg, "trail_cam_sd", min_id=0)
    assert dropped == 16                           # the 16 identical boxes, and only those
    assert bursts.drop_orphans(conn) == 16

    frames = _kept_frames(cfg)
    assert len(frames) == 1
    assert import_trailcam.SRC_TAG + "IMAG0100" in frames[0].name
    assert bursts.written == frames                # the survivor is still counted, the rest aren't
    assert not (cfg.frames_dir / "trail_cam_sd" / "2026-08-05").exists()   # emptied day tidied up
    # Every surviving detection still points at a frame that exists.
    for (fp,) in conn.execute("SELECT DISTINCT frame_path FROM detections"):
        assert (config.ROOT / fp).exists()


def test_orphan_drop_is_a_no_op_when_nothing_was_filtered(conn, db_path, tmp_path):
    """The ordinary case: nothing was deleted, so nothing is reclaimed. Worth pinning because
    drop_orphans() unlinks files -- an off-by-one here would delete a live burst's only frame."""
    cfg = _burst_cfg(tmp_path, db_path)
    dump = tmp_path / "dump"
    _burst_dump(dump, ["2026-08-06 21:00:00", "2026-08-06 21:05:00"])

    bursts = import_trailcam.BurstFrames(cfg, "trail_cam_sd")
    _import_bursts(dump, conn, cfg, bursts)
    before = _kept_frames(cfg)
    assert len(before) == 2

    assert bursts.drop_orphans(conn) == 0
    assert _kept_frames(cfg) == before


def test_identical_mtimes_cannot_collapse_a_cycle_into_one_burst(conn, db_path, tmp_path):
    """The EXIF-less card. image_timestamp() then falls back to the file mtime, and a bulk copy
    stamps every file with the same second -- which by the gap rule alone is ONE burst, hanging a
    single frame on hundreds of unrelated detections. BURST_MAX_STILLS caps that; the largest
    real burst on this corpus is 10 stills, so the cap can never split a genuine trigger."""
    cfg = _burst_cfg(tmp_path, db_path)
    dump = tmp_path / "dump"
    n = import_trailcam.BURST_MAX_STILLS + 6
    _burst_dump(dump, ["2026-08-06 21:00:00"] * n)

    bursts = import_trailcam.BurstFrames(cfg, "trail_cam_sd")
    imported, saved, _ = _import_bursts(dump, conn, cfg, bursts)
    assert (imported, saved) == (n, n)
    assert bursts.bursts == 2 and len(_kept_frames(cfg)) == 2
    counts = dict(conn.execute(
        "SELECT frame_path, COUNT(*) FROM detections GROUP BY frame_path"))
    assert sorted(counts.values()) == [6, import_trailcam.BURST_MAX_STILLS]
