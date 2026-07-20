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
