"""
Tests for eval.py -- the read-only evaluation harness's DB-facing logic.

Two things matter most here and both get pinned:
  * the LEAVE-ONE-VISIT-OUT contamination guard (individual_centroids): when you score a visit,
    its own prototype must never be inside its individual's template, or the match is measuring the
    visit against itself and every "cross-session" number is a lie;
  * the harness is genuinely READ-ONLY on backyard.db -- opened via the mode=ro URI, a stray write
    must raise rather than contend for the live rig's WAL lock.

Everything else is exercised on a synthetic DB built by the shared `conn` fixture (throwaway
tempfile; the real backyard.db is never touched), with hand-built unit-vector embeddings so
who-matches-whom is exact by construction -- the same convention as test_individuals.py.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pytest

import config
import db
import eval as evalmod
import individuals

BASE = datetime(2026, 6, 10, 21, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)


def _ts(minutes=0.0):
    return (BASE + timedelta(minutes=minutes)).isoformat()


def _u(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# The leave-one-visit-out contamination guard (pure, no DB).
# ---------------------------------------------------------------------------

def test_individual_centroids_leaves_the_named_visit_out():
    # Stan has two visits pointing slightly differently; Notch one. Excluding Stan's visit 2 must
    # build his centroid from visit 1 ALONE -- so it equals visit 1's vector, not the mean of both.
    protos = {1: _u(1, 0, 0), 2: _u(0, 1, 0), 3: _u(0, 0, 1)}
    labels = {1: "Stan", 2: "Stan", 3: "Notch"}

    full = evalmod.individual_centroids(protos, labels)          # no exclusion
    assert full["Stan"] @ _u(1, 1, 0) == pytest.approx(1.0)      # mean of visit 1 and 2

    held = evalmod.individual_centroids(protos, labels, exclude_visit=2)
    assert held["Stan"] @ _u(1, 0, 0) == pytest.approx(1.0)      # visit 2 dropped -> just visit 1
    assert "Notch" in held                                       # untouched individuals remain


def test_individual_centroids_drops_individual_with_no_visits_left():
    # If the excluded visit is an individual's ONLY visit, that individual can't be a template.
    protos = {1: _u(1, 0), 2: _u(0, 1)}
    labels = {1: "Solo", 2: "Other"}
    held = evalmod.individual_centroids(protos, labels, exclude_visit=1)
    assert "Solo" not in held and "Other" in held


# ---------------------------------------------------------------------------
# Read-only guarantee.
# ---------------------------------------------------------------------------

def test_open_readonly_refuses_writes(conn, db_path):
    # conn (read-write) has already created the DB + schema. Re-open it through the harness path.
    conn.execute("INSERT INTO detections (timestamp, source, detection_class, confidence, "
                 "bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h, crop_path) "
                 "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (_ts(), db.SOURCE_GLASS_DOOR_CAM, "animal", 0.9, 0, 0, 1, 1, 10, 10, "c.jpg"))
    conn.commit()
    ro = evalmod.open_readonly(db_path)
    try:
        assert ro is not None
        assert ro.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("UPDATE detections SET confidence = 0.1")
            ro.commit()
    finally:
        ro.close()


def test_open_readonly_missing_db_returns_none(tmp_path):
    assert evalmod.open_readonly(tmp_path / "nope.db") is None


# ---------------------------------------------------------------------------
# Species eval: the prediction-overwrite handling.
# ---------------------------------------------------------------------------

def _det(conn, *, species, conf, verified, source_stage, minutes=0.0):
    """Insert a reviewed detection and set its species/verified/source columns directly (the
    harness reads exactly these)."""
    did = db.insert_detection(
        conn, timestamp=_ts(minutes), source=db.SOURCE_GLASS_DOOR_CAM, detection_class="animal",
        confidence=0.9, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
        crop_path=f"crops/{minutes}.jpg")
    conn.execute("UPDATE detections SET species=?, species_confidence=?, species_verified=?, "
                 "species_source=? WHERE id=?", (species, conf, verified, source_stage, did))
    conn.commit()
    return did


def test_eval_species_separates_intact_from_overwritten(conn):
    cfg = config.Config()
    # Two prediction-intact rows (a confirmed-correct raccoon, a rejected crow) + one human
    # correction whose original prediction is gone.
    _det(conn, species="raccoon", conf=0.99, verified=1, source_stage="bioclip", minutes=0)
    _det(conn, species="American crow", conf=0.30, verified=0, source_stage="bioclip", minutes=1)
    _det(conn, species="raccoon", conf=1.0, verified=1, source_stage="human", minutes=2)

    s = evalmod.eval_species(conn, cfg)
    gt = s["ground_truth"]
    assert gt["verified_rows_total"] == 3
    assert gt["prediction_intact"] == 2
    assert gt["prediction_overwritten_by_correction"] == 1     # the human row is NOT graded
    assert gt["confirmed_correct"] == 1 and gt["rejected_wrong"] == 1
    assert s["graded_rows"] == 2
    # The confirmed raccoon is a true positive; the rejected crow is a false positive for 'crow'.
    per = s["precision_recall_f1"]["per_label"]
    assert per["raccoon"]["precision"] == pytest.approx(1.0)
    # Trust-rule buckets: the 0.99 confirm sits in >=0.8 (accuracy 1.0); the 0.30 reject in <0.5 (0).
    tr = s["trust_rule_check"]
    assert tr["conf_ge_0.8"]["accuracy"] == pytest.approx(1.0)
    assert tr["conf_lt_0.5"]["accuracy"] == pytest.approx(0.0)


def test_eval_species_grades_recovered_corrections(conn):
    # A confirmed raccoon (intact) + a correction where the model said 'American crow' but a human
    # relabelled it 'raccoon'. The model_species fix preserves the crow prediction, so it must now
    # grade as a crow->raccoon MISS (a real confusion row), not the ungradable lost sentinel.
    cfg = config.Config()
    _det(conn, species="raccoon", conf=0.95, verified=1, source_stage="bioclip", minutes=0)
    did = db.insert_detection(
        conn, timestamp=_ts(1), source=db.SOURCE_GLASS_DOOR_CAM, detection_class="animal",
        confidence=0.9, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100, crop_path="crops/rec.jpg")
    db.set_species(conn, did, "American crow", 0.4, "bioclip")   # the model's call, snapshotted
    conn.commit()
    db.correct_species(conn, did, "raccoon")                     # human overrides -> prediction kept

    s = evalmod.eval_species(conn, cfg)
    gt = s["ground_truth"]
    assert gt["correction_recovered"] == 1
    assert gt["prediction_overwritten_by_correction"] == 0       # nothing lost -- it was preserved
    assert s["graded_rows"] == 2                                 # confirmed raccoon + recovered crow
    per = s["precision_recall_f1"]["per_label"]
    # crow PREDICTION with raccoon TRUTH: a false positive for crow, a false negative for raccoon
    # recall -- both only measurable because the correction preserved the model's prediction.
    assert per["American crow"]["fp"] == 1
    assert per["raccoon"]["fn"] == 1
    assert per["raccoon"]["support"] == 2


def test_eval_species_empty_is_safe(conn):
    s = evalmod.eval_species(conn, config.Config())
    assert s["graded_rows"] == 0 and s["ground_truth"]["verified_rows_total"] == 0


# ---------------------------------------------------------------------------
# Re-ID eval: end-to-end separation on synthetic vectors.
# ---------------------------------------------------------------------------

def _embed(conn, det_id, vec):
    v = _u(*vec)
    db.insert_embedding(conn, det_id, individuals.EMBED_MODEL, len(v), v.tobytes())
    conn.commit()


def _solo_visit(conn, vec, *, start_min):
    """A three-crop solo visit whose crops all point along `vec` (one animal lingering)."""
    ids = []
    for k in range(3):
        d = db.insert_detection(
            conn, timestamp=_ts(start_min + k * 0.2), source=db.SOURCE_GLASS_DOOR_CAM,
            detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10), frame_w=100,
            frame_h=100, crop_path=f"crops/{start_min}_{k}.jpg", species="raccoon",
            crop_quality=10.0)
        base = np.asarray(vec, dtype=np.float32)
        _embed(conn, d, base + 0.01 * np.array([k, -k, k], dtype=np.float32)[:len(base)])
        ids.append(d)
    vid = db.insert_visit(conn, source=db.SOURCE_GLASS_DOOR_CAM, species="raccoon",
                          individual_id=None, started_at=_ts(start_min),
                          ended_at=_ts(start_min + 1), detection_count=3, max_confidence=0.9,
                          representative_detection_id=ids[0])
    db.assign_visit(conn, ids, vid)
    conn.commit()
    return vid


def test_eval_reid_measures_separation_and_excludes_multi(conn):
    cfg = config.Config()
    cfg.reid_co_presence_min = 1
    # Two individuals, two solo visits each: same-individual pairs must out-score cross pairs.
    a1 = _solo_visit(conn, [1, 0, 0], start_min=0)
    a2 = _solo_visit(conn, [1, 0.05, 0], start_min=60)
    b1 = _solo_visit(conn, [0, 1, 0], start_min=120)
    b2 = _solo_visit(conn, [0, 1, 0.05], start_min=180)
    db.label_visit(conn, a1, "Stan")
    db.label_visit(conn, a2, "Stan")
    db.label_visit(conn, b1, "Notch")
    db.label_visit(conn, b2, "Notch")

    # A confirmed PAIR visit (two separated boxes every frame) must be excluded from templates.
    ids = []
    for k in range(3):
        pa = db.insert_detection(conn, timestamp=_ts(240 + k * 0.2), source=db.SOURCE_GLASS_DOOR_CAM,
                                 detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10),
                                 frame_w=100, frame_h=100, crop_path=f"crops/p{k}a.jpg",
                                 species="raccoon", crop_quality=10.0)
        pb = db.insert_detection(conn, timestamp=_ts(240 + k * 0.2), source=db.SOURCE_GLASS_DOOR_CAM,
                                 detection_class="animal", confidence=0.9, bbox=(50, 50, 60, 60),
                                 frame_w=100, frame_h=100, crop_path=f"crops/p{k}b.jpg",
                                 species="raccoon", crop_quality=10.0)
        _embed(conn, pa, [1, 0, 0])
        _embed(conn, pb, [0, 1, 0])
        ids += [pa, pb]
    pair = db.insert_visit(conn, source=db.SOURCE_GLASS_DOOR_CAM, species="raccoon",
                           individual_id=None, started_at=_ts(240), ended_at=_ts(241),
                           detection_count=6, max_confidence=0.9, representative_detection_id=ids[0])
    db.assign_visit(conn, ids, pair)
    conn.commit()
    db.label_visit(conn, pair, "Stan")

    r = evalmod.eval_reid(conn, cfg, "raccoon")
    cv = r["confirmed_visits"]
    assert cv["solo_with_prototype"] == 4
    assert cv["excluded_multi_animal"] == 1               # the pair visit, correctly left out
    sep = r["separation"]
    assert sep["same_individual_pairs"]["median"] > sep["different_individual_pairs"]["median"]
    assert sep["auc"] == pytest.approx(1.0)               # perfectly separable by construction
    # LOO identification: with the self excluded, each visit's nearest individual is still correct.
    assert r["identification_loo"]["top1_accuracy"] == pytest.approx(1.0)
    assert r["identification_loo"]["coverage"]["scorable_visits"] == 4


def test_eval_reid_thin_data_degrades_honestly(conn):
    cfg = config.Config()
    v = _solo_visit(conn, [1, 0, 0], start_min=0)
    db.label_visit(conn, v, "Stan")
    r = evalmod.eval_reid(conn, cfg, "raccoon")
    assert r["confirmed_visits"]["solo_with_prototype"] == 1
    assert "note" in r["separation"]                      # one solo visit -> nothing to separate
