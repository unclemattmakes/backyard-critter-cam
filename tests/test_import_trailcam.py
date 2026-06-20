"""
Smoke test for the trail-cam batch importer (import_trailcam.ingest_file).

Regression guard for the save_crop contract: save_crop() returns (Path, shot-quality), and the
importer must unpack that tuple. A past bug passed the whole tuple straight to _rel(), so the
importer crashed on the first real detection and silently saved nothing -- and no test covered it.
No GPU/camera/model here: the detector is a tiny stub that returns one fixed box.
"""
from __future__ import annotations

from dataclasses import replace
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
