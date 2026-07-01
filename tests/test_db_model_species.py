"""
Tests for the model_species preservation columns (db.py).

The bug they guard: a human CORRECTION used to overwrite detections.species and force
confidence=1.0, destroying the classifier's own prediction -- so a corrected crop could never
grade the model (eval.py's biggest data gap). The fix snapshots the model's call into
model_species/_confidence/_source at classify time, and both correction paths PRESERVE it. These
tests pin every branch: the snapshot, the two preserve paths, the double-correction case, and the
migration backfill (including that a pre-fix correction is honestly left unrecoverable).

All on a throwaway tempfile DB via the shared `conn` fixture; the real backyard.db is never touched.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

import db

BASE = datetime(2026, 6, 10, 21, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)


def _ts(minutes=0.0):
    return (BASE + timedelta(minutes=minutes)).isoformat()


def _insert(conn, minutes=0.0):
    return db.insert_detection(
        conn, timestamp=_ts(minutes), source=db.SOURCE_GLASS_DOOR_CAM, detection_class="animal",
        confidence=0.9, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
        crop_path=f"crops/{minutes}.jpg")


def _row(conn, det_id):
    return conn.execute(
        "SELECT species, species_confidence, species_source, species_verified, "
        "model_species, model_species_confidence, model_species_source "
        "FROM detections WHERE id = ?", (det_id,)).fetchone()


def test_set_species_snapshots_the_model_prediction(conn):
    did = _insert(conn)
    db.set_species(conn, did, "raccoon", 0.87, "bioclip")
    r = _row(conn, did)
    assert r["species"] == "raccoon" and r["species_confidence"] == 0.87
    assert r["model_species"] == "raccoon"
    assert r["model_species_confidence"] == 0.87
    assert r["model_species_source"] == "bioclip"


def test_correct_species_preserves_model_prediction(conn):
    did = _insert(conn)
    db.set_species(conn, did, "American crow", 0.42, "bioclip")   # model's (wrong) call
    db.correct_species(conn, did, "raccoon")                      # human overrides
    r = _row(conn, did)
    # Live label is now the human's answer...
    assert r["species"] == "raccoon" and r["species_confidence"] == 1.0
    assert r["species_verified"] == 1 and r["species_source"] == "human"
    # ...but the model's original prediction survives for grading.
    assert r["model_species"] == "American crow"
    assert r["model_species_confidence"] == 0.42
    assert r["model_species_source"] == "bioclip"


def test_correct_species_snapshots_current_species_when_no_prior_model(conn):
    # A legacy-shaped row that got a species without going through set_species (model_species NULL):
    # correct_species must COALESCE-snapshot the current species before overwriting it.
    did = _insert(conn)
    conn.execute("UPDATE detections SET species = ?, species_confidence = ?, species_source = ? "
                 "WHERE id = ?", ("gray squirrel", 0.7, "bioclip", did))
    conn.commit()
    db.correct_species(conn, did, "raccoon")
    r = _row(conn, did)
    assert r["species"] == "raccoon"
    assert r["model_species"] == "gray squirrel"
    assert r["model_species_confidence"] == 0.7


def test_double_correction_keeps_the_original_model_prediction(conn):
    did = _insert(conn)
    db.set_species(conn, did, "American crow", 0.42, "bioclip")
    db.correct_species(conn, did, "Virginia opossum")   # first human guess
    db.correct_species(conn, did, "raccoon")            # corrected again
    r = _row(conn, did)
    assert r["species"] == "raccoon"
    # COALESCE keeps the FIRST snapshot -- the model's call, not the intermediate human label.
    assert r["model_species"] == "American crow"
    assert r["model_species_confidence"] == 0.42


def test_apply_visit_label_species_preserves_model_prediction(conn):
    # Two crops the model called 'American crow'; a bulk visit correction to 'raccoon' by time span
    # must preserve each crop's model prediction (the whole-visit twin of correct_species).
    ids = [_insert(conn, minutes=m) for m in (0, 0.5)]
    for d in ids:
        db.set_species(conn, d, "American crow", 0.4, "bioclip")
    conn.commit()
    res = db.apply_visit_label(conn, source=db.SOURCE_GLASS_DOOR_CAM,
                               start=_ts(-1), end=_ts(1), species="raccoon")
    assert res["detections"] == 2
    for d in ids:
        r = _row(conn, d)
        assert r["species"] == "raccoon" and r["species_source"] == "human"
        assert r["model_species"] == "American crow"
        assert r["model_species_confidence"] == 0.4


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35, 0),
                    reason="ALTER TABLE DROP COLUMN needs SQLite >= 3.35")
def test_migrate_backfills_model_species_for_uncorrected_rows_only(conn):
    # Populate a full DB, then recreate the pre-fix shape by DROPPING the model_species columns and
    # re-running _migrate -- exactly the upgrade path a real backyard.db takes on first launch.
    good = _insert(conn)
    db.set_species(conn, good, "raccoon", 0.9, "bioclip")            # still the model's label
    corrected = _insert(conn, minutes=1)
    db.set_species(conn, corrected, "American crow", 0.3, "bioclip")
    db.correct_species(conn, corrected, "raccoon")                  # human-corrected before the fix
    blank = _insert(conn, minutes=2)                                # never classified
    conn.commit()

    for col in ("model_species", "model_species_confidence", "model_species_source"):
        conn.execute(f"ALTER TABLE detections DROP COLUMN {col}")
    conn.commit()
    db._migrate(conn)          # re-adds the columns and backfills
    conn.commit()

    rows = {r["id"]: r for r in conn.execute(
        "SELECT id, model_species, model_species_confidence FROM detections")}
    # Uncorrected auto-labelled row: snapshot taken from its (still the model's) species.
    assert rows[good]["model_species"] == "raccoon"
    assert rows[good]["model_species_confidence"] == 0.9
    # Human-corrected before the fix (species_source='human'): unrecoverable -> left NULL.
    assert rows[corrected]["model_species"] is None
    # Never classified: nothing to snapshot.
    assert rows[blank]["model_species"] is None
