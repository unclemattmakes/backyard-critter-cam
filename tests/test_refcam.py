"""
Tests for certified identity reference batches (refcam).

What is actually load-bearing here, in the order it would hurt:

  * REFERENCES NEVER LAND IN `detections`. A reference photo is a different imaging domain from
    every camera in the yard (max measured cross-source similarity 0.363), so if one ever became a
    re-ID template, a single accepted match would let refit() propose that name across the corpus
    -- the documented mass-mislabel. Two tests assert the containment, one over the DB and one
    over the module's own guard.
  * NAME FROM THE DIRECTORY, and from nothing else. The folder name is the certification.
  * HASH-KEYED IDEMPOTENCY. Matt will run it twice; content hashes (not filenames -- phones rename
    on export) make the second run a no-op, and the same bytes under two names is a LOUD conflict
    rather than a silent choice.
  * DRY RUN IS THE DEFAULT and writes nothing at all -- not even the tables.
  * A corrupt file is skipped, not fatal.
  * VIDEO FRAMES ARE WEAKER EVIDENCE THAN PHOTOGRAPHS, and the rows say so. A folder holding only
    videos is a real batch (it used to import as silence); a frame the detector picked is marked
    media_kind='video_frame' so no query can mistake it for a picture a human framed; and a video
    holding SEVERAL animals is recorded as a GROUP claim -- individual_id names the set that was
    present, not the animal in any one box. Getting that last one wrong would inject exactly the
    blended labels the rest of the project spends its time undoing.

No GPU/model here: the --crop path and the whole video path are exercised with tiny stub detectors,
like the trail-cam importer's tests. Test videos are SYNTHESIZED with cv2.VideoWriter rather than
shipped as fixtures -- each frame is a flat grey ramping from dark to light, so a stub can score a
frame off its own pixels and "which frames did it keep" becomes a deterministic assertion instead
of a guess about a model. Everything runs against pytest's tmp_path and the throwaway DB from
conftest.py; the real backyard.db is never opened.
"""
from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
import db
import refcam
from detector import Detection


class _StubDetector:
    """Stand-in for detector.Detector: one fixed 'animal' box, no torch/GPU."""

    def __init__(self, class_name: str = "animal", class_id: int = 0, boxes: int = 1):
        self.class_name, self.class_id, self.boxes = class_name, class_id, boxes

    def detect(self, frame):
        h, w = frame.shape[:2]
        return [Detection(class_id=self.class_id, class_name=self.class_name, confidence=0.93,
                          bbox=(w * 0.1, h * 0.1, w * 0.6, h * 0.6)) for _ in range(self.boxes)]


class _VideoStubDetector:
    """Stand-in for detector.Detector on the VIDEO path.

    Its confidence is the frame's own mean brightness, so a synthesized ramp video has genuinely
    better and worse frames and `select_frames` can be asserted against rather than guessed at. It
    returns TWO boxes once a frame is brighter than `multi_above`, which is how a multi-animal
    video is simulated without a model -- and how the group-vs-individual scope call is tested."""

    def __init__(self, multi_above: float | None = None, class_name: str = "animal"):
        self.multi_above, self.class_name = multi_above, class_name

    def detect(self, frame):
        h, w = frame.shape[:2]
        mean = float(frame.mean())
        conf = min(0.99, max(0.05, mean / 255.0))
        n = 2 if (self.multi_above is not None and mean >= self.multi_above) else 1
        return [Detection(class_id=0, class_name=self.class_name, confidence=conf,
                          bbox=(w * 0.1, h * 0.1, w * 0.5, h * 0.5)) for _ in range(n)]


def _video(path: Path, *, n_frames: int = 60, fps: float = 10.0, size=(160, 120)) -> Path:
    """A tiny mp4 whose frames are flat greys ramping dark -> light, written with cv2.VideoWriter.

    Synthesized rather than shipped: a binary fixture in the repo would be un-reviewable, and the
    ramp is what makes frame SELECTION testable -- the later a frame, the brighter it is, so with
    _VideoStubDetector the best-scoring frames are the last ones and the min-gap spread is
    visible in the stored video_time_s values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "OpenCV could not open an mp4v writer for the test video"
    for i in range(n_frames):
        grey = int(20 + (i / max(1, n_frames - 1)) * 200)
        writer.write(np.full((h, w, 3), grey, np.uint8))
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


def _photo(path: Path, seed: int = 0, size=(120, 160)) -> Path:
    """A small noisy BGR JPEG on disk. The seed makes two files differ in CONTENT, which is what
    the dedupe key is computed over -- two same-seed files are byte-identical on purpose."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), (rng.random((size[0], size[1], 3)) * 255).astype(np.uint8))
    return path


def _cfg(tmp_path, db_path):
    """A Config pinned entirely inside tmp -- db and reference crops both, so nothing a test does
    can reach the project's real database or write into its crops tree."""
    return replace(config.CONFIG, db_path=db_path,
                   reference_dir=tmp_path / "reference",
                   reference_crops_dir=tmp_path / "reference_crops")


def _drop_folder(tmp_path, names_to_counts: dict) -> Path:
    """A reference drop folder: reference/<name>/photo_<i>.jpg, each photo unique."""
    root = tmp_path / "reference"
    seed = 0
    for name, count in names_to_counts.items():
        for i in range(count):
            seed += 1
            _photo(root / name / f"photo_{i}.jpg", seed=seed)
    return root


def _run(conn, cfg, root, *, apply=False, detector=None, individual=None, video=None):
    """find_batches -> plan_batches -> import_plans, the way main() strings them together."""
    batches, loose = refcam.find_batches(root, individual=individual)
    plans = refcam.plan_batches(conn, batches)
    tally = refcam.import_plans(conn, cfg, plans, source=refcam.reference_source("iphone"),
                                apply=apply, detector=detector, video=video)
    return tally, loose


def _rows(conn, *cols, order="id"):
    """Every identity_references row as a list of tuples, for terse assertions."""
    return conn.execute(
        f"SELECT {', '.join(cols)} FROM identity_references ORDER BY {order}").fetchall()


# ---- name inference ---------------------------------------------------------------------

def test_name_comes_from_the_directory(tmp_path):
    """The folder name IS the individual, spelling preserved, and each folder's photos stay with
    it. Nothing infers a name from the image or the filename."""
    root = _drop_folder(tmp_path, {"Alpha": 2, "Beta One": 1})
    batches, loose = refcam.find_batches(root)

    assert [name for name, _ in batches] == ["Alpha", "Beta One"]   # exact spelling, incl. space
    assert [len(photos) for _, photos in batches] == [2, 1]
    assert loose == []


def test_loose_photos_are_never_certified_under_a_guessed_name(conn, db_path, tmp_path):
    """A photo dropped at the top level has no folder, so it has no name. It must be reported and
    SKIPPED -- inventing a name is the one thing a certification store may never do."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    _photo(root / "stray.jpg", seed=99)

    tally, loose = _run(conn, cfg, root, apply=True)
    assert [p.name for p in loose] == ["stray.jpg"]
    names = [r[0] for r in conn.execute("SELECT individual_id FROM identity_references")]
    assert names == ["Alpha"]


def test_individual_override_handles_a_flat_folder(tmp_path):
    """A fresh AirDrop dump is flat. --individual names the whole folder in one go."""
    flat = tmp_path / "airdrop"
    _photo(flat / "IMG_0001.JPG", seed=1)
    _photo(flat / "IMG_0002.JPG", seed=2)
    batches, loose = refcam.find_batches(flat, individual="Alpha")
    assert len(batches) == 1 and batches[0][0] == "Alpha" and len(batches[0][1]) == 2


def test_hidden_and_underscore_folders_are_not_individuals(tmp_path):
    """'_rejects' and '.thumbnails' sit next to real batches all the time; neither is an animal."""
    root = _drop_folder(tmp_path, {"Alpha": 1})
    _photo(root / "_rejects" / "blurry.jpg", seed=50)
    _photo(root / ".cache" / "thumb.jpg", seed=51)
    batches, _ = refcam.find_batches(root)
    assert [name for name, _ in batches] == ["Alpha"]


# ---- containment: the whole point --------------------------------------------------------

def test_references_never_land_in_the_detections_table(conn, db_path, tmp_path):
    """THE guard. An import writes identity_references and NOTHING into detections -- not one row,
    ever. A reference in detections becomes a re-ID template in a domain where the best measured
    cross-source similarity is 0.363, and one accepted match would spread the name."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 2, "Bravo": 1})

    tally, _ = _run(conn, cfg, root, apply=True)

    assert tally[refcam.NEW] == 3
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 0
    assert refcam.check_not_in_detections(conn) == 0


def test_reference_source_is_its_own_domain_string(conn, db_path, tmp_path):
    """The stored source must be a 'reference_*' string -- distinct from every capture source, so
    individuals.py's plain source-equality guard covers references for free."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    _run(conn, cfg, root, apply=True)

    src = conn.execute("SELECT source FROM identity_references").fetchone()[0]
    assert src == "reference_iphone"
    assert refcam.is_reference_source(src)
    assert src not in (db.SOURCE_GLASS_DOOR_CAM, db.SOURCE_TRAIL_CAM_SD)
    assert not refcam.is_reference_source(db.SOURCE_GLASS_DOOR_CAM)


def test_guard_would_notice_a_reference_in_detections(conn):
    """The guard must actually fire -- a check that can only ever return 0 checks nothing. Here a
    row is planted by hand (refcam itself has no code path that could write it)."""
    db.insert_detection(conn, timestamp=db.now_local_iso(), source="reference_iphone",
                        detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10),
                        frame_w=100, frame_h=100, crop_path="crops/x.jpg")
    assert refcam.check_not_in_detections(conn) == 1


# ---- idempotency ------------------------------------------------------------------------

def test_second_run_is_a_no_op_via_content_hash(conn, db_path, tmp_path):
    """Matt WILL run it twice. The second pass certifies nothing new and adds no rows."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 2})

    first, _ = _run(conn, cfg, root, apply=True)
    second, _ = _run(conn, cfg, root, apply=True)

    assert first[refcam.NEW] == 2
    assert (second[refcam.NEW], second[refcam.DUPLICATE]) == (0, 2)
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 2


def test_renamed_copy_is_recognised_as_the_same_photo(conn, db_path, tmp_path):
    """Keyed on CONTENT, not filename: phones rename on export, and a re-export of the same shot
    must not be certified twice."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    _run(conn, cfg, root, apply=True)

    original = next((root / "Alpha").glob("*.jpg"))
    (root / "Alpha" / "IMG_9999.JPG").write_bytes(original.read_bytes())

    tally, _ = _run(conn, cfg, root, apply=True)
    assert (tally[refcam.NEW], tally[refcam.DUPLICATE]) == (0, 2)
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 1


def test_case_typo_in_the_folder_name_is_a_duplicate_not_a_conflict(conn, db_path, tmp_path):
    """A folder renamed 'alpha' -> 'Alpha' is a typo, not two animals. CONFLICT is the loudest
    thing this module says and it must not be spent on that."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    _run(conn, cfg, root, apply=True)
    original = next((root / "Alpha").glob("*.jpg"))

    # The batch list is built by hand rather than by renaming the folder: Windows filesystems are
    # case-insensitive, so 'reference/alpha' and 'reference/Alpha' are one directory there and a
    # folder-level test would silently exercise nothing.
    plans = refcam.plan_batches(conn, [("alpha", [original])])
    assert [p.status for p in plans] == [refcam.DUPLICATE]

    conflicting = refcam.plan_batches(conn, [("Bravo", [original])])
    assert [p.status for p in conflicting] == [refcam.CONFLICT]   # a real different name still is


def test_duplicates_within_one_run_do_not_crash_the_unique_key(conn, db_path, tmp_path):
    """The same bytes twice inside a single drop folder is a duplicate, not an IntegrityError."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    original = next((root / "Alpha").glob("*.jpg"))
    (root / "Alpha" / "again.jpg").write_bytes(original.read_bytes())

    tally, _ = _run(conn, cfg, root, apply=True)
    assert (tally[refcam.NEW], tally[refcam.DUPLICATE]) == (1, 1)


def test_same_photo_under_two_names_is_a_loud_conflict(conn, db_path, tmp_path):
    """A certification is the one label nothing else can check, so identical bytes filed under two
    names is reported and REFUSED -- never resolved by picking one."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    original = next((root / "Alpha").glob("*.jpg"))
    _photo(root / "Bravo" / "placeholder.jpg", seed=77)     # so Bravo is a real batch
    (root / "Bravo" / "same_animal.jpg").write_bytes(original.read_bytes())

    tally, _ = _run(conn, cfg, root, apply=True)

    assert tally[refcam.CONFLICT] == 1
    rows = dict(conn.execute(
        "SELECT individual_id, COUNT(*) FROM identity_references GROUP BY individual_id"))
    assert rows == {"Alpha": 1, "Bravo": 1}                 # the conflicting file wrote nothing


# ---- dry run ----------------------------------------------------------------------------

def test_dry_run_writes_nothing_at_all(db_path, tmp_path):
    """The default. Not one row, and not even the tables -- a dry run must be able to run against
    a database it has no business modifying."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 3})
    conn = db.connect(db_path)                 # a real DB with the pipeline schema, no refcam tables
    try:
        assert not refcam.has_schema(conn)
        tally, _ = _run(conn, cfg, root, apply=False)
        assert tally[refcam.NEW] == 3          # it still REPORTS what it would do
        assert not refcam.has_schema(conn)     # ... and created nothing
    finally:
        conn.close()


def test_dry_run_is_the_default_on_the_command_line():
    """--apply is opt-in, and --dry-run wins if both are given (fail-safe)."""
    assert refcam.parse_args(["folder"]).apply is False
    both = refcam.parse_args(["folder", "--apply", "--dry-run"])
    assert both.apply is True and both.dry_run is True      # main() resolves this to dry run


def test_main_dry_run_over_a_folder_writes_nothing(db_path, tmp_path, capsys):
    """End to end through main(): the default run opens the DB read-only, so 'writes nothing' is
    enforced by the connection rather than by branching."""
    root = _drop_folder(tmp_path, {"Alpha": 2})
    db.connect(db_path).close()                             # the DB exists, with no refcam tables
    rc = refcam.main([str(root), "--db", str(db_path)])
    assert rc == 0
    assert "Would import 2" in capsys.readouterr().out

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='identity_references'"
                            ).fetchone() is None
    finally:
        conn.close()


# ---- robustness ---------------------------------------------------------------------------

def test_corrupt_image_is_skipped_not_fatal(conn, db_path, tmp_path):
    """One bad file never aborts a batch (the trail-cam importer's posture). The corrupt photo is
    reported as unreadable and its healthy neighbours still import."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 2})
    (root / "Alpha" / "broken.jpg").write_bytes(b"this is not a JPEG")

    tally, _ = _run(conn, cfg, root, apply=True)

    assert (tally[refcam.NEW], tally[refcam.UNREADABLE]) == (2, 1)
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 2


def test_heic_is_reported_not_half_imported(conn, db_path, tmp_path, monkeypatch):
    """Without pillow-heif a HEIC file is counted and SKIPPED so the CLI can say 'export as JPEG'.
    It must never be recorded with unknown dimensions and no readable pixels."""
    cfg = _cfg(tmp_path, db_path)
    monkeypatch.setattr(refcam, "_heic_ready", lambda: False)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    (root / "Alpha" / "IMG_0100.HEIC").write_bytes(b"\x00\x00\x00\x18ftypheic")

    tally, _ = _run(conn, cfg, root, apply=True)

    assert (tally[refcam.NEW], tally[refcam.HEIC_UNSUPPORTED]) == (1, 1)
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 1


def test_capture_time_is_exif_or_null_never_the_mtime(conn, db_path, tmp_path):
    """A reference photo's mtime is when it was copied off the phone. With no EXIF the column
    stays NULL -- an honest gap beats a confidently wrong capture date."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})          # cv2-written JPEGs carry no EXIF
    _run(conn, cfg, root, apply=True)
    assert conn.execute("SELECT captured_at FROM identity_references").fetchone()[0] is None


def test_provenance_is_recorded(conn, db_path, tmp_path):
    """Per the requirement: name, source, file path, content hash, dimensions, import time and a
    free-text note (here from the folder's note.txt) all land on the row."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    (root / "Alpha" / refcam.NOTE_FILENAME).write_text("back fence, daylight", encoding="utf-8")

    batches, _ = refcam.find_batches(root)
    plans = refcam.plan_batches(conn, batches)
    notes = {name: refcam.folder_note(photos[0].parent) for name, photos in batches}
    refcam.import_plans(conn, cfg, plans, source=refcam.reference_source("iphone"),
                        batch_notes=notes, apply=True)

    row = conn.execute("SELECT individual_id, source, file_path, content_sha256, width, height, "
                       "imported_at, note, batch FROM identity_references").fetchone()
    assert row["individual_id"] == "Alpha"
    assert row["source"] == "reference_iphone"
    assert Path(row["file_path"]).name.endswith(".jpg")
    assert len(row["content_sha256"]) == 64
    assert (row["width"], row["height"]) == (160, 120)
    assert row["imported_at"] and row["note"] == "back fence, daylight"
    assert row["batch"] == "Alpha"


# ---- the optional crop step ------------------------------------------------------------

def test_crop_step_is_optional_and_separable(conn, db_path, tmp_path):
    """Without a detector, references import with no crops at all -- the crop pass is a second,
    re-runnable step over the same rows, not part of certifying them."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    tally, _ = _run(conn, cfg, root, apply=True, detector=None)
    assert tally["crops"] == 0
    assert conn.execute("SELECT COUNT(*) FROM identity_reference_crops").fetchone()[0] == 0


def test_crop_writes_outside_the_pipeline_crops_tree(conn, db_path, tmp_path):
    """With a detector, each photo gets a crop under reference_crops_dir/<name>/ -- cut by the
    project's own save_crop (so padding, JPEG quality and the quality score match every other crop)
    and STILL not a detection row."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})

    tally, _ = _run(conn, cfg, root, apply=True, detector=_StubDetector())

    assert tally["crops"] == 1
    row = conn.execute("SELECT reference_id, crop_path, detection_class, confidence, crop_quality "
                       "FROM identity_reference_crops").fetchone()
    assert row["detection_class"] == "animal"
    assert row["crop_quality"] is not None
    path = Path(row["crop_path"])
    if not path.is_absolute():
        path = config.ROOT / path
    assert path.exists()
    assert path.parent == cfg.reference_crops_dir / "alpha"
    # The crop name leads with the reference's content hash, so two phones' IMG_0007.JPG can't
    # overwrite each other and any crop is traceable back to its row by name alone.
    sha = conn.execute("SELECT content_sha256 FROM identity_references").fetchone()[0]
    assert path.name.startswith(sha[:8] + "_" + refcam.SRC_TAG)
    assert cfg.crops_dir not in path.parents          # never the pipeline's crops tree
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0


def test_crop_of_an_empty_photo_keeps_the_reference(conn, db_path, tmp_path):
    """A photo the detector finds nothing in still counts as certified -- the human's word is the
    data; the crop is a convenience."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    tally, _ = _run(conn, cfg, root, apply=True,
                    detector=_StubDetector(class_name="person", class_id=1))
    assert (tally[refcam.NEW], tally["crops"]) == (1, 0)
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 1


def test_crops_cascade_when_a_reference_goes(conn, db_path, tmp_path):
    """The crop table is derived data: deleting a reference takes its crops with it (FK CASCADE,
    with db.connect's PRAGMA foreign_keys=ON)."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    _run(conn, cfg, root, apply=True, detector=_StubDetector())
    conn.execute("DELETE FROM identity_references")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM identity_reference_crops").fetchone()[0] == 0


# ---- schema hygiene ----------------------------------------------------------------------

def test_ensure_schema_is_idempotent(conn):
    """Additive and re-runnable, per the project's schema rule."""
    refcam.ensure_schema(conn)
    refcam.ensure_schema(conn)
    assert refcam.has_schema(conn)


def test_summary_is_empty_before_anything_is_certified(conn):
    assert refcam.summary_rows(conn) == []


def test_summary_rolls_up_per_individual(conn, db_path, tmp_path):
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 3, "Bravo": 1})
    _run(conn, cfg, root, apply=True, detector=_StubDetector())
    rows = {r["name"]: r for r in refcam.summary_rows(conn)}
    assert rows["Alpha"]["n"] == 3 and rows["Bravo"]["n"] == 1
    assert rows["Alpha"]["n_crops"] == 3
    assert rows["Alpha"]["sources"] == "reference_iphone"


@pytest.mark.parametrize("camera,expected", [
    ("iphone", "reference_iphone"),
    ("Pixel 8 Pro", "reference_pixel_8_pro"),
    ("", "reference_handheld"),
])
def test_source_string_is_slugged_per_camera(camera, expected):
    """One string per physical camera: a year from now this column is the only way to tell two
    phones apart, and it has to be stable and filesystem-safe."""
    assert refcam.reference_source(camera) == expected


def test_migration_adds_the_video_columns_to_a_legacy_reference_table(conn):
    """Additive and idempotent, over a table written BEFORE videos existed. The columns appear, the
    existing row is untouched, and its NULLs are left alone -- backfilling media_kind='photo' would
    claim this module had inspected a row it never saw (it reads as 'photo' via COALESCE instead)."""
    conn.executescript("""
        CREATE TABLE identity_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT, individual_id TEXT NOT NULL,
            source TEXT NOT NULL, captured_at TEXT, imported_at TEXT NOT NULL,
            file_path TEXT NOT NULL, content_sha256 TEXT NOT NULL UNIQUE,
            width INTEGER, height INTEGER, batch TEXT, note TEXT);
    """)
    conn.execute("INSERT INTO identity_references (individual_id, source, imported_at, file_path, "
                 "content_sha256) VALUES ('Alpha', 'reference_iphone', 'then', 'a.jpg', 'deadbeef')")
    conn.commit()

    refcam.ensure_schema(conn)
    refcam.ensure_schema(conn)                      # idempotent

    cols = refcam.reference_columns(conn)
    assert set(refcam.REFERENCE_COLUMNS_ADDED) <= cols
    row = conn.execute("SELECT individual_id, media_kind, identity_scope, source_sha256 "
                       "FROM identity_references").fetchone()
    assert row[0] == "Alpha" and row[1] is None and row[2] is None and row[3] is None
    # A legacy row counts as a photograph, because that is what it is.
    assert refcam.summary_rows(conn)[0]["n_frames"] == 0


# ---- videos: a folder of them is a real batch ---------------------------------------------

def test_a_video_only_folder_is_no_longer_silently_skipped(tmp_path):
    """THE bug. 'Stans Kids' held two videos and nothing else, so it imported as silence -- and the
    kits are the least-documented animals in the project precisely because they never hold still
    for a photo. A folder of videos is a batch."""
    root = tmp_path / "reference"
    _video(root / "Stans Kids" / "clip.mp4", n_frames=20)
    _photo(root / "Alpha" / "photo.jpg", seed=1)

    batches, loose = refcam.find_batches(root)

    assert [name for name, _ in batches] == ["Alpha", "Stans Kids"]   # spelling AND space kept
    assert [p.name for _, files in batches for p in files] == ["photo.jpg", "clip.mp4"]
    assert loose == []


def test_video_extensions_are_matched_case_insensitively(tmp_path):
    """Phones shout their extensions: the real folder held IMG_5175.MOV."""
    root = tmp_path / "reference"
    _video(root / "Alpha" / "IMG_0001.MOV", n_frames=12)
    batches, _ = refcam.find_batches(root)
    assert [p.name for _, files in batches for p in files] == ["IMG_0001.MOV"]
    assert refcam.is_video(Path("x.MOV")) and refcam.is_video(Path("x.mp4"))
    assert not refcam.is_video(Path("x.jpg"))


# ---- videos: what gets written ------------------------------------------------------------

def test_video_frames_are_certified_with_full_provenance(conn, db_path, tmp_path):
    """Each kept frame gets its own row, and the row can be traced back to the exact frame it came
    from: source video, frame index, offset in seconds, animal count -- plus media_kind, which is
    what stops it being read as a photograph a human framed."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    video = _video(root / "Alpha" / "clip.mp4", n_frames=60, fps=10.0)

    tally, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
                    video=refcam.VideoOptions(max_frames=4, min_gap_s=0.0))

    assert tally["videos"] == 1 and tally["video_frames"] == 4
    rows = _rows(conn, "media_kind", "source_video", "source_sha256", "video_frame_index",
                 "video_time_s", "n_animals", "width", "height", "individual_id")
    assert len(rows) == 4
    video_sha = refcam.file_sha256(video)
    for r in rows:
        assert r["media_kind"] == refcam.MEDIA_VIDEO_FRAME
        assert Path(r["source_video"]).name == "clip.mp4"
        assert r["source_sha256"] == video_sha
        assert r["video_frame_index"] is not None
        # The offset is the frame index over the video's own frame rate, to the millisecond.
        assert r["video_time_s"] == pytest.approx(r["video_frame_index"] / 10.0, abs=1e-6)
        assert r["n_animals"] == 1
        assert (r["width"], r["height"]) == (160, 120)
        assert r["individual_id"] == "Alpha"


def test_extracted_frames_are_distinguishable_from_photo_references(conn, db_path, tmp_path):
    """One folder, one name, two kinds of evidence -- and every query can tell them apart. A photo
    is a deliberate act; a frame is a detector's pick out of footage nobody curated."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _photo(root / "Alpha" / "portrait.jpg", seed=3)
    _video(root / "Alpha" / "clip.mp4", n_frames=40)

    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
         video=refcam.VideoOptions(max_frames=3, min_gap_s=0.0))

    kinds = dict(conn.execute(
        "SELECT COALESCE(media_kind, 'photo'), COUNT(*) FROM identity_references GROUP BY 1"))
    assert kinds == {refcam.MEDIA_PHOTO: 1, refcam.MEDIA_VIDEO_FRAME: 3}
    photo = conn.execute("SELECT source_video, video_frame_index, n_animals FROM "
                         "identity_references WHERE media_kind = 'photo'").fetchone()
    assert tuple(photo) == (None, None, None)      # a photograph carries no video provenance
    summary = {r["name"]: r for r in refcam.summary_rows(conn)}
    assert summary["Alpha"]["n"] == 4 and summary["Alpha"]["n_frames"] == 3


def test_extracted_frames_land_outside_the_pipeline_crops_tree(conn, db_path, tmp_path):
    """Frames are a regenerable cache under <reference crops>/_frames/<name>/ -- never crops/,
    which backup.py zips by capture date, and never the drop folder (the '_' prefix also keeps
    find_batches from ever certifying the cache as an animal)."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "clip.mp4", n_frames=30)

    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
         video=refcam.VideoOptions(max_frames=2, min_gap_s=0.0))

    frames_dir = refcam.reference_frames_dir(cfg)
    assert frames_dir == cfg.reference_crops_dir / "_frames"
    for (fp,) in conn.execute("SELECT file_path FROM identity_references"):
        path = Path(fp)
        if not path.is_absolute():
            path = config.ROOT / path
        assert path.exists() and path.suffix == ".jpg"
        assert path.parent == frames_dir / "alpha"
        assert cfg.crops_dir not in path.parents
    # The cache sits beside the crops, and can never be certified as an animal itself: '_frames'
    # is '_'-prefixed, so find_batches skips it the way it skips '_rejects'.
    cache_batches, _cache_loose = refcam.find_batches(frames_dir.parent)
    assert "_frames" not in [n for n, _ in cache_batches]


def test_video_frames_get_crops_because_the_boxes_are_why_they_were_chosen(conn, db_path, tmp_path):
    """A photo's crop is optional (--crop); a video frame's is not, because the detector was
    already loaded and the box is the entire reason this frame beat the others."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "clip.mp4", n_frames=30)

    tally, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
                    video=refcam.VideoOptions(max_frames=3, min_gap_s=0.0))

    assert tally["crops"] == 3
    n = conn.execute("SELECT COUNT(*) FROM identity_reference_crops c JOIN identity_references r "
                     "ON r.id = c.reference_id WHERE r.media_kind = 'video_frame'").fetchone()[0]
    assert n == 3


# ---- videos: choosing a SPREAD, not a burst -----------------------------------------------

def test_frame_cap_is_respected(conn, db_path, tmp_path):
    """A video must not flood the reference set. The cap is a hard ceiling."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "clip.mp4", n_frames=120, fps=10.0)

    tally, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
                    video=refcam.VideoOptions(max_frames=3, min_gap_s=0.0))

    assert tally["video_frames"] == 3
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 3


def test_minimum_gap_between_kept_frames_is_respected(conn, db_path, tmp_path):
    """Consecutive frames are one observation wearing many hats. With a 6-second video sampled
    twice a second and a 1-second floor, no two kept frames may sit closer than that -- which caps
    this clip at 6 frames even though the frame cap allows 12."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "clip.mp4", n_frames=60, fps=10.0)      # 6.0 s

    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
         video=refcam.VideoOptions(max_frames=12, min_gap_s=1.0, scan_fps=2.0))

    times = sorted(r[0] for r in conn.execute("SELECT video_time_s FROM identity_references"))
    assert len(times) == 6
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(times, times[1:]))


def test_the_best_frames_win_not_the_first_ones(conn, db_path, tmp_path):
    """Selection is by score (confidence + how much of the frame the animal spans), so on a video
    that gets steadily better the LAST frames are kept, not whatever came first."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "clip.mp4", n_frames=60, fps=10.0)      # brightness ramps up

    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
         video=refcam.VideoOptions(max_frames=3, min_gap_s=0.0, scan_fps=2.0))

    kept = sorted(r[0] for r in conn.execute("SELECT video_frame_index FROM identity_references"))
    assert kept == [45, 50, 55]        # the three brightest sampled frames


def test_a_long_video_is_sampled_more_sparsely_not_more_expensively(tmp_path):
    """max_scans bounds the detector work per video: a long clip widens the stride instead of
    running thousands of passes."""
    meta = {"frame_count": 100_000, "fps": 30.0, "duration_s": 100_000 / 30.0,
            "width": 1920, "height": 1080}
    indices, stride = refcam.scan_plan(meta, refcam.VideoOptions(scan_fps=2.0, max_scans=300))
    assert len(indices) <= 300 and stride > 15
    short, short_stride = refcam.scan_plan(
        {"frame_count": 60, "fps": 10.0, "duration_s": 6.0, "width": 160, "height": 120},
        refcam.VideoOptions(scan_fps=2.0, max_scans=300))
    assert short_stride == 5 and short == list(range(0, 60, 5))


# ---- videos: THE provenance decision ------------------------------------------------------

def test_a_multi_animal_video_is_a_group_claim_not_individual_ground_truth(conn, db_path, tmp_path):
    """The load-bearing test. 'Stans Kids' means "Stan and her kits are in here", NOT "every box is
    Stan" -- and nothing in a video of four raccoons says which one any box holds. Every frame kept
    from such a video is scoped 'group': the name is preserved verbatim, but no consumer may read
    one of these rows as single-animal ground truth."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Stans Kids" / "clip.mp4", n_frames=60, fps=10.0)

    # Brighter than 128 -> two boxes, i.e. a second animal is visible in the later frames.
    tally, _ = _run(conn, cfg, root, apply=True,
                    detector=_VideoStubDetector(multi_above=128.0),
                    video=refcam.VideoOptions(max_frames=6, min_gap_s=0.0, scan_fps=2.0))

    rows = _rows(conn, "individual_id", "identity_scope", "n_animals", "media_kind")
    assert rows, "the multi-animal video certified nothing"
    assert {r["individual_id"] for r in rows} == {"Stans Kids"}      # the human's exact words
    assert {r["identity_scope"] for r in rows} == {refcam.SCOPE_GROUP}
    assert tally["video_group"] == len(rows)
    # ... and the raw count that justified the call is on the row, not merely implied.
    assert max(r["n_animals"] for r in rows) >= 2


def test_group_scope_covers_even_the_solo_frames_of_a_multi_animal_video(conn, db_path, tmp_path):
    """Judged per VIDEO, not per frame. A frame showing one animal, lifted out of footage that
    elsewhere shows two, is still ONE OF TWO -- and certifying it as an individual would be exactly
    the blended label this project keeps having to undo."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Stans Kids" / "clip.mp4", n_frames=60, fps=10.0)

    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(multi_above=128.0),
         video=refcam.VideoOptions(max_frames=12, min_gap_s=0.0, scan_fps=2.0))

    rows = _rows(conn, "identity_scope", "n_animals")
    solo = [r for r in rows if r["n_animals"] == 1]
    assert solo, "expected some kept frames to hold a single animal"
    assert {r["identity_scope"] for r in solo} == {refcam.SCOPE_GROUP}


def test_a_single_animal_video_stays_individual_scope_but_never_becomes_a_photo(conn, db_path,
                                                                                tmp_path):
    """Where the detector never once saw two animals, the human's "this video is <name>"
    distributes over the frames exactly as it does over a folder of photos -- so the scope is
    'individual'. media_kind still says video_frame: the evidence is weaker in KIND either way."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "clip.mp4", n_frames=40, fps=10.0)

    tally, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(multi_above=None),
                    video=refcam.VideoOptions(max_frames=3, min_gap_s=0.0))

    rows = _rows(conn, "identity_scope", "media_kind", "n_animals")
    assert {r["identity_scope"] for r in rows} == {refcam.SCOPE_INDIVIDUAL}
    assert {r["media_kind"] for r in rows} == {refcam.MEDIA_VIDEO_FRAME}
    assert tally["video_group"] == 0


def test_scope_is_decided_over_every_scanned_frame_not_only_the_kept_ones():
    """A video demonstrably holds a second animal even when the frame that proved it scored too
    low to keep. video_identity_scope therefore reads the whole scan."""
    solo = [refcam.ScannedFrame(index=i, time_s=i / 10, score=0.9, confidence=0.9, n_animals=1)
            for i in range(5)]
    assert refcam.video_identity_scope(solo) == refcam.SCOPE_INDIVIDUAL
    with_pair = solo + [refcam.ScannedFrame(index=99, time_s=9.9, score=0.1,
                                            confidence=0.1, n_animals=2)]
    assert refcam.video_identity_scope(with_pair) == refcam.SCOPE_GROUP


def test_a_photo_is_still_certified_as_an_individual(conn, db_path, tmp_path):
    """The strong case stays strong: nothing about video support weakens a photograph."""
    cfg = _cfg(tmp_path, db_path)
    root = _drop_folder(tmp_path, {"Alpha": 1})
    _run(conn, cfg, root, apply=True)
    row = conn.execute("SELECT media_kind, identity_scope FROM identity_references").fetchone()
    assert (row[0], row[1]) == (refcam.MEDIA_PHOTO, refcam.SCOPE_INDIVIDUAL)


# ---- videos: idempotency ------------------------------------------------------------------

def test_second_run_over_the_same_video_imports_nothing_new(conn, db_path, tmp_path):
    """Matt WILL run it twice, and a 578 MB phone video must not be re-scanned to find that out.
    The video's own content hash is on every frame row, so the second pass skips it without
    decoding anything."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "clip.mp4", n_frames=40)
    opts = refcam.VideoOptions(max_frames=3, min_gap_s=0.0)

    first, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(), video=opts)
    second, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(), video=opts)

    assert first["video_frames"] == 3
    assert (second["video_frames"], second[refcam.DUPLICATE]) == (0, 1)
    assert second["video_scanned"] == 0            # nothing was decoded on the second pass
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 3


def test_a_renamed_copy_of_a_video_is_recognised_as_the_same_video(conn, db_path, tmp_path):
    """Keyed on CONTENT, like the photos -- phones rename on export and Matt re-copies folders."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    video = _video(root / "Alpha" / "clip.mp4", n_frames=40)
    opts = refcam.VideoOptions(max_frames=2, min_gap_s=0.0)
    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(), video=opts)

    (root / "Alpha" / "IMG_9999.MOV").write_bytes(video.read_bytes())
    tally, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(), video=opts)

    assert (tally["video_frames"], tally[refcam.DUPLICATE]) == (0, 2)
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 2


def test_the_same_video_under_two_names_is_a_loud_conflict(conn, db_path, tmp_path):
    """Same refusal a duplicated photograph gets: a certification is the one label nothing else can
    check, so identical footage filed under two names is reported, not resolved by picking one."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    video = _video(root / "Alpha" / "clip.mp4", n_frames=30)
    opts = refcam.VideoOptions(max_frames=2, min_gap_s=0.0)
    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(), video=opts)

    (root / "Bravo" / "same.mp4").parent.mkdir(parents=True, exist_ok=True)
    (root / "Bravo" / "same.mp4").write_bytes(video.read_bytes())
    tally, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(), video=opts)

    assert tally[refcam.CONFLICT] == 1
    assert {r[0] for r in conn.execute("SELECT individual_id FROM identity_references")} == {"Alpha"}


def test_the_frame_key_is_the_video_hash_plus_the_index_not_the_jpeg_bytes(conn, db_path, tmp_path):
    """Re-encoding is not byte-stable, so hashing the extracted JPEG would certify the same frame
    again on every run. The key is sha256('<video sha>:frame:<n>'), which survives a re-run, a
    re-encode and a rename."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    video = _video(root / "Alpha" / "clip.mp4", n_frames=30)

    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
         video=refcam.VideoOptions(max_frames=2, min_gap_s=0.0))

    video_sha = refcam.file_sha256(video)
    for r in _rows(conn, "content_sha256", "video_frame_index", "file_path"):
        assert r["content_sha256"] == refcam.frame_content_key(video_sha, r["video_frame_index"])
        # ... and emphatically NOT the hash of the file the row points at.
        path = Path(r["file_path"])
        if not path.is_absolute():
            path = config.ROOT / path
        assert r["content_sha256"] != refcam.file_sha256(path)


# ---- videos: the dry run still writes nothing ----------------------------------------------

def test_dry_run_over_a_video_writes_nothing_and_decodes_nothing(db_path, tmp_path, capsys):
    """The default, unchanged by video support: not one row, not the tables, not a frame on disk.
    A video is LISTED in a dry run, never scanned -- scanning needs a detector, and a dry run
    loads none."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Stans Kids" / "clip.mp4", n_frames=30)
    conn = db.connect(db_path)
    try:
        assert not refcam.has_schema(conn)
        tally, _ = _run(conn, cfg, root, apply=False, detector=None)
        assert (tally[refcam.NEW], tally["videos"]) == (1, 1)      # it still REPORTS the video
        assert tally["video_frames"] == 0
        assert not refcam.has_schema(conn)
    finally:
        conn.close()
    assert not refcam.reference_frames_dir(cfg).exists()


def test_main_dry_run_over_a_video_folder_opens_the_db_read_only(db_path, tmp_path, capsys):
    """End to end through main(): the read-only connection is what ENFORCES 'writes nothing', and
    a video batch must not weaken that into a branch."""
    root = tmp_path / "reference"
    _video(root / "Stans Kids" / "clip.mp4", n_frames=30)
    db.connect(db_path).close()

    rc = refcam.main([str(root), "--db", str(db_path),
                      "--crops-dir", str(tmp_path / "reference_crops")])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Stans Kids" in out and "[video]" in out
    assert "Would import 1" in out
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='identity_references'"
                            ).fetchone() is None
    finally:
        conn.close()


# ---- videos: robustness ---------------------------------------------------------------------

def test_corrupt_video_is_skipped_not_fatal(conn, db_path, tmp_path):
    """One bad file never aborts a batch -- the trail-cam importer's posture, extended to videos.
    The unreadable clip is reported and its healthy neighbours still import."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "good.mp4", n_frames=30)
    (root / "Alpha" / "broken.mp4").write_bytes(b"this is not a video, not even a little")
    _photo(root / "Alpha" / "photo.jpg", seed=7)

    tally, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
                    video=refcam.VideoOptions(max_frames=2, min_gap_s=0.0))

    assert tally[refcam.UNREADABLE] == 1
    assert tally["video_frames"] == 2
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 3


def test_a_video_with_no_animal_certifies_nothing_and_says_so(conn, db_path, tmp_path, capsys):
    """A detector that finds nothing means there is no reference here. The video is counted as
    empty rather than half-imported, and nothing is written."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "clip.mp4", n_frames=30)

    tally, _ = _run(conn, cfg, root, apply=True,
                    detector=_VideoStubDetector(class_name="person"),
                    video=refcam.VideoOptions(max_frames=3, min_gap_s=0.0))

    assert (tally["videos"], tally["video_frames"], tally["video_empty"]) == (1, 0, 1)
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 0
    assert "no animal" in capsys.readouterr().out


def test_a_video_without_a_detector_is_reported_never_silently_dropped(conn, db_path, tmp_path,
                                                                       capsys):
    """A photo is certified by the human's word alone; a video FRAME does not exist until a
    detector has chosen it. With no detector the video is skipped LOUDLY -- silence here is the
    entire bug being fixed."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Stans Kids" / "clip.mp4", n_frames=30)

    tally, _ = _run(conn, cfg, root, apply=True, detector=None)

    assert tally["video_skipped"] == 1 and tally["video_frames"] == 0
    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] == 0
    assert "clip.mp4" in capsys.readouterr().out


def test_no_video_frames_disables_the_whole_path(conn, db_path, tmp_path):
    """--no-video-frames: photos still import, videos are left alone and SAID so."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "clip.mp4", n_frames=30)
    _photo(root / "Alpha" / "photo.jpg", seed=11)

    tally, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
                    video=refcam.VideoOptions(enabled=False))

    assert (tally["video_skipped"], tally["video_frames"]) == (1, 0)
    kinds = [r[0] for r in conn.execute("SELECT media_kind FROM identity_references")]
    assert kinds == [refcam.MEDIA_PHOTO]


def test_the_opencv_fallback_works_when_ffmpeg_is_missing(conn, db_path, tmp_path, monkeypatch):
    """ffmpeg is preferred (and hardware-accelerated where the box allows), but it is NOT a
    dependency: with no ffmpeg on PATH, cv2.VideoCapture decodes the same frames."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "clip.mp4", n_frames=40, fps=10.0)
    monkeypatch.setattr(refcam, "_FFMPEG", None)
    monkeypatch.setattr(refcam, "_FFPROBE", None)

    tally, _ = _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
                    video=refcam.VideoOptions(max_frames=3, min_gap_s=0.0, scan_fps=2.0))

    assert tally["video_frames"] == 3
    kept = sorted(r[0] for r in conn.execute("SELECT video_frame_index FROM identity_references"))
    assert kept == [25, 30, 35]        # same frames the ffmpeg path picks from this ramp


def test_both_decoders_return_the_same_source_frames(tmp_path, monkeypatch):
    """The fallback has to be CORRECT, not merely present: an off-by-one there would attach the
    wrong frame index to a stored reference and the provenance would be a lie."""
    video = _video(tmp_path / "clip.mp4", n_frames=40, fps=10.0)
    meta = refcam.video_meta(video)
    indices, stride = refcam.scan_plan(meta, refcam.VideoOptions(scan_fps=2.0))
    native = (meta["width"], meta["height"])

    with_ffmpeg = [(i, round(float(f.mean()))) for i, f in
                   refcam.iter_video_frames(video, indices, native_size=native, stride=stride)]
    monkeypatch.setattr(refcam, "_FFMPEG", None)
    with_cv2 = [(i, round(float(f.mean()))) for i, f in
                refcam.iter_video_frames(video, indices, native_size=native, stride=stride)]

    assert [i for i, _ in with_ffmpeg] == indices == [i for i, _ in with_cv2]
    for (_, a), (_, b) in zip(with_ffmpeg, with_cv2):
        assert abs(a - b) <= 3          # same frame; colour conversion differs by a hair


# ---- videos: capture time -------------------------------------------------------------------

def test_video_capture_time_comes_from_an_iso_filename_never_the_mtime(conn, db_path, tmp_path):
    """EXIF is meaningless for a container. A clip named the way THIS rig names its own
    ('2026-08-06T01-55-50-431.mp4' is clips.py's start wall-clock) keeps its real capture time, and
    each kept frame's own captured_at is that start plus its offset into the video."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Stans Kids" / "2026-08-06T01-55-50-431.mp4", n_frames=60, fps=10.0)

    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
         video=refcam.VideoOptions(max_frames=3, min_gap_s=0.0))

    rows = _rows(conn, "captured_at", "video_time_s")
    assert rows
    for r in rows:
        when = db.parse_local(r["captured_at"])
        assert when is not None
        assert when.strftime("%Y-%m-%d %H:%M") == "2026-08-06 01:55"
        # start (01:55:50.431) + the offset into the clip, to the millisecond.
        offset = when - db.parse_local("2026-08-06T01:55:50.431").astimezone()
        assert offset.total_seconds() == pytest.approx(r["video_time_s"], abs=1e-3)


def test_a_video_with_no_readable_date_gets_an_honest_null(conn, db_path, tmp_path):
    """No container tag, no timestamp in the name -> NULL. NEVER the mtime: a reference file's
    mtime is when it was copied off the phone, and a confidently wrong date is worse than a gap."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Alpha" / "IMG_0042.mp4", n_frames=30)

    assert refcam.video_captured_at(root / "Alpha" / "IMG_0042.mp4") is None
    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(),
         video=refcam.VideoOptions(max_frames=2, min_gap_s=0.0))
    assert all(r[0] is None for r in conn.execute(
        "SELECT captured_at FROM identity_references"))


@pytest.mark.parametrize("raw,expect", [
    ("2026-07-24T02:17:04.000000Z", "2026-07-24T02:17:04+00:00"),   # ffprobe creation_time (UTC)
    ("2026-07-23T19:17:04-0700", "2026-07-23T19:17:04-07:00"),      # Apple's, colon-less offset
    ("nonsense", None),
])
def test_container_timestamps_parse_including_the_spellings_fromisoformat_rejects(raw, expect):
    got = refcam._parse_iso_loose(raw)
    assert (None if got is None else got.isoformat()) == expect


# ---- videos: containment, which is still the whole point ------------------------------------

def test_video_frames_never_land_in_the_detections_table(conn, db_path, tmp_path):
    """THE guard, over the new path. A video frame is run through the same detector the rigs use,
    which is exactly why it would be so easy to write it as a detection -- and exactly why it must
    not be. Nothing here touches detections, visits or clips."""
    cfg = _cfg(tmp_path, db_path)
    root = tmp_path / "reference"
    _video(root / "Stans Kids" / "clip.mp4", n_frames=40)

    _run(conn, cfg, root, apply=True, detector=_VideoStubDetector(multi_above=128.0),
         video=refcam.VideoOptions(max_frames=4, min_gap_s=0.0))

    assert conn.execute("SELECT COUNT(*) FROM identity_references").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 0
    assert refcam.check_not_in_detections(conn) == 0
    src = conn.execute("SELECT DISTINCT source FROM identity_references").fetchone()[0]
    assert refcam.is_reference_source(src)
