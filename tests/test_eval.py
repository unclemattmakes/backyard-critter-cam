"""
Tests for eval.py -- the read-only evaluation harness's DB-facing logic.

Four things matter most here and all four get pinned:
  * the LEAVE-ONE-VISIT-OUT contamination guard (individual_centroids): when you score a visit,
    its own prototype must never be inside its individual's template, or the match is measuring the
    visit against itself and every "cross-session" number is a lie;
  * SESSION BLOCKING (2026-08-05): the same guard one level up. A visit must not match a template
    of the same individual from the SAME NIGHT either -- one night is one session, one set of
    lighting and one auto-white-balance decision, and matching a raccoon to its own 40-minutes-ago
    self measures nothing the product sells. The test below builds a corpus where blocking flips
    the answer from right to wrong, because a guard that never changes an outcome is not a guard;
  * HELD-OUT operating-point selection: the sweep must pick its (threshold, margin) on the
    training folds and never see the fold it is scored on. Both a direct spy on what the selector
    is handed, and a behavioural case where an in-sample sweep reports zero errors while the
    held-out procedure demonstrably makes one;
  * the harness is genuinely READ-ONLY on backyard.db -- opened via the mode=ro URI, a stray write
    must raise rather than contend for the live rig's WAL lock.

Everything else is exercised on a synthetic DB built by the shared `conn` fixture (throwaway
tempfile; the real backyard.db is never touched), with hand-built unit-vector embeddings so
who-matches-whom is exact by construction -- the same convention as test_individuals.py.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pytest

import config
import db
import eval as evalmod
import evalmetrics as em
import individuals

BASE = datetime(2026, 6, 10, 21, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)
DAY = 24 * 60  # minutes; visits a day apart are on different NIGHTS, which is what blocking keys on


def _ts(minutes=0.0):
    return (BASE + timedelta(minutes=minutes)).isoformat()


def _u(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _probes(protos, labels, nights=None):
    """Run the protocol over hand-built prototypes and hand back its probe records -- the input
    the operating-point sweep takes. Default: every visit on its OWN night, so blocking is inert
    and these fixtures test the sweep rather than the blocking."""
    units = [em.LooUnit(key=v, label=labels[v], night=(nights or {}).get(v, f"night-{v}"))
             for v in sorted(protos)]
    return em.leave_one_visit_out(units, evalmod.appearance_similarity(protos))["probes"]


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


def test_individual_centroids_excludes_a_whole_set():
    # exclude_visits is how session blocking is applied to the centroid variant: drop the probe
    # AND every same-night visit of its individual in one call.
    protos = {1: _u(1, 0, 0), 2: _u(1, 0.1, 0), 3: _u(0, 0, 1), 4: _u(0, 1, 0)}
    labels = {1: "Stan", 2: "Stan", 3: "Stan", 4: "Notch"}
    held = evalmod.individual_centroids(protos, labels, exclude_visits={1, 2})
    assert held["Stan"] @ _u(0, 0, 1) == pytest.approx(1.0)   # only visit 3 survived
    # The single-visit form still works, and the two are unioned rather than one winning.
    both = evalmod.individual_centroids(protos, labels, exclude_visit=3, exclude_visits={1})
    assert both["Stan"] @ _u(1, 0.1, 0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SESSION BLOCKING -- the phase A3 fix. A probe may not match a template of the
# SAME individual from the SAME night (night key = timestamp shifted -12h).
# ---------------------------------------------------------------------------

# One corpus, built so blocking flips a verdict. Stan is seen twice on one night (S1, S2 -- nearly
# identical, as two looks at one session always are) and once much later wearing a different look
# (S3). Notch's single visit (T1) happens to resemble S1 fairly closely. Probe S1:
#   unblocked -> its best template is S2, its own 40-minutes-later self  -> "Stan", correct;
#   blocked   -> S2 is gone, Stan is represented only by the stale S3    -> "Notch", WRONG.
# That flip IS the session leak, reproduced in four visits.
_S1, _S2, _S3, _T1 = 1, 2, 3, 4
_BLOCK_PROTOS = {_S1: _u(1, 0, 0), _S2: _u(1, 0.02, 0), _S3: _u(0, 0, 1), _T1: _u(0.9, 0.44, 0)}
_BLOCK_LABELS = {_S1: "Stan", _S2: "Stan", _S3: "Stan", _T1: "Notch"}
_BLOCK_NIGHTS = {_S1: "2026-06-10", _S2: "2026-06-10",       # <- the same night
                 _S3: "2026-06-20", _T1: "2026-06-25"}


def _block_units():
    return [em.LooUnit(key=k, label=_BLOCK_LABELS[k], night=_BLOCK_NIGHTS[k])
            for k in sorted(_BLOCK_PROTOS)]


def test_session_blocking_excludes_the_same_night_twin_and_changes_the_answer():
    sim = evalmod.appearance_similarity(_BLOCK_PROTOS)
    units = _block_units()
    unblocked = em.leave_one_visit_out(units, sim, session_blocked=False)
    blocked = em.leave_one_visit_out(units, sim, session_blocked=True)

    p_un = {p["key"]: p for p in unblocked["probes"]}[_S1]
    p_bl = {p["key"]: p for p in blocked["probes"]}[_S1]

    # Unblocked, S1's winning template is its own same-night twin -- and it is "correct".
    assert p_un["top"] == "Stan" and p_un["correct"] and p_un["via"] == _S2
    # Blocked, the twin is unavailable, so the answer changes to the wrong animal.
    assert p_bl["via"] != _S2
    assert p_bl["top"] == "Notch" and not p_bl["correct"]
    # ...and that shows up in the headline number, which is the whole point of measuring it.
    assert blocked["top1_accuracy"] < unblocked["top1_accuracy"]


def test_session_blocking_keeps_an_identical_probe_set():
    # The blocked and unblocked runs must be comparable: same probes, same denominator. A probe
    # left with no template for its own identity is scored WRONG, never quietly dropped -- dropping
    # it would shrink the denominator under blocking and understate the leak.
    sim = evalmod.appearance_similarity(_BLOCK_PROTOS)
    units = _block_units()
    unblocked = em.leave_one_visit_out(units, sim, session_blocked=False)
    blocked = em.leave_one_visit_out(units, sim, session_blocked=True)
    assert blocked["n_probes"] == unblocked["n_probes"] == len(_BLOCK_PROTOS)
    assert ([p["key"] for p in blocked["probes"]] == [p["key"] for p in unblocked["probes"]])
    # Notch has exactly one visit, so it is a NOVEL probe under either rule.
    assert blocked["n_novel_probes"] >= 1
    # Chance is reported beside the accuracy, over the same probes: Stan is 3 of 4.
    assert blocked["chance"]["rate"] == pytest.approx(0.75)
    assert blocked["chance"]["label"] == "Stan"


def test_session_blocking_only_blocks_the_SAME_individual():
    # Notch's visit shares a night with Stan's. Blocking must NOT remove it: a different animal
    # seen the same night is a legitimate template (and a legitimately hard negative). If blocking
    # over-reached to all same-night visits, Notch would vanish from S1's candidates entirely.
    nights = {**_BLOCK_NIGHTS, _T1: "2026-06-10"}
    units = [em.LooUnit(key=k, label=_BLOCK_LABELS[k], night=nights[k])
             for k in sorted(_BLOCK_PROTOS)]
    blocked = em.leave_one_visit_out(units, evalmod.appearance_similarity(_BLOCK_PROTOS),
                                     session_blocked=True)
    p = {x["key"]: x for x in blocked["probes"]}[_S1]
    assert p["top"] == "Notch" and p["via"] == _T1


def test_night_key_puts_after_midnight_on_the_previous_evening():
    # 23:40 and 01:10 are ONE night. A plain calendar date splits them; the -12h shift doesn't.
    assert em.night_key("2026-06-10T23:40:00-07:00") == "2026-06-10"
    assert em.night_key("2026-06-11T01:10:00-07:00") == "2026-06-10"
    assert em.night_key("2026-06-11T13:00:00-07:00") == "2026-06-11"   # daytime -> its own date
    assert em.night_key("not a timestamp") is None                     # never raises
    assert em.night_key(None) is None


def test_pair_separation_drops_same_night_positive_pairs_only():
    protos, labels, nights = _BLOCK_PROTOS, _BLOCK_LABELS, _BLOCK_NIGHTS
    leaked = evalmod._pair_separation(protos, labels, nights, session_blocked=False)
    blocked = evalmod._pair_separation(protos, labels, nights, session_blocked=True)
    # Stan pairs: (S1,S2) same night, (S1,S3), (S2,S3). Only the first is dropped.
    assert leaked["same_individual_pairs"]["n"] == 3
    assert blocked["same_individual_pairs"]["n"] == 2
    assert blocked["same_night_positive_pairs_dropped"] == 1
    # Negatives are untouched -- dropping same-night cross-individual pairs would re-inflate AUC.
    assert (blocked["different_individual_pairs"]["n"]
            == leaked["different_individual_pairs"]["n"] == 3)


def test_eval_reid_reports_blocked_and_unblocked_side_by_side(conn):
    # End to end through the DB: two Stan visits on ONE night plus one much later, and a Notch.
    cfg = config.Config()
    cfg.reid_co_presence_min = 1
    s1 = _solo_visit(conn, [1, 0, 0], start_min=0)             # 21:00, night A
    s2 = _solo_visit(conn, [1, 0.02, 0], start_min=200)        # 00:20 next day -- still night A
    s3 = _solo_visit(conn, [0, 0, 1], start_min=10 * DAY)
    t1 = _solo_visit(conn, [0.9, 0.44, 0], start_min=15 * DAY)
    for v, name in ((s1, "Stan"), (s2, "Stan"), (s3, "Stan"), (t1, "Notch")):
        db.label_visit(conn, v, name)

    r = evalmod.eval_reid(conn, cfg, "raccoon", kfold_repeats=1)
    loo = r["identification_loo"]
    assert loo["blocked"]["protocol"]["session_blocked"] is True
    assert loo["unblocked"]["protocol"]["session_blocked"] is False
    # The -12h shift is what makes 00:20 the same night as 21:00: three distinct nights, not four.
    assert r["confirmed_visits"]["distinct_nights"] == 3
    assert loo["blocked"]["top1_accuracy"] < loo["unblocked"]["top1_accuracy"]
    assert loo["session_leak_gap"] > 0
    # Chance rides along with every accuracy.
    assert loo["blocked"]["chance"]["rate"] == pytest.approx(0.75)
    # Both separations are reported, and the blocked one dropped the same-night Stan pair.
    assert r["separation"]["same_night_positive_pairs_dropped"] == 1
    assert r["separation_unblocked"]["same_night_positive_pairs_dropped"] == 0
    # The embargo curve is session-blocked throughout and carries its own chance rate per row.
    assert [row["embargo_days"] for row in loo["embargo_curve"]][:2] == [None, 1]
    assert all(row["chance"] is not None for row in loo["embargo_curve"])


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
    # Two individuals, two solo visits each, each visit on its OWN night -- otherwise session
    # blocking (correctly) throws the same-night pairs away and there is no separation left to
    # measure. A one-night corpus is not a cross-session corpus, which is the lesson of phase A3.
    a1 = _solo_visit(conn, [1, 0, 0], start_min=0)
    a2 = _solo_visit(conn, [1, 0.05, 0], start_min=1 * DAY)
    b1 = _solo_visit(conn, [0, 1, 0], start_min=2 * DAY)
    b2 = _solo_visit(conn, [0, 1, 0.05], start_min=3 * DAY)
    db.label_visit(conn, a1, "Stan")
    db.label_visit(conn, a2, "Stan")
    db.label_visit(conn, b1, "Notch")
    db.label_visit(conn, b2, "Notch")

    # A confirmed PAIR visit (two separated boxes every frame) must be excluded from templates.
    ids = []
    for k in range(3):
        pa = db.insert_detection(conn, timestamp=_ts(4 * DAY + k * 0.2), source=db.SOURCE_GLASS_DOOR_CAM,
                                 detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10),
                                 frame_w=100, frame_h=100, crop_path=f"crops/p{k}a.jpg",
                                 species="raccoon", crop_quality=10.0)
        pb = db.insert_detection(conn, timestamp=_ts(4 * DAY + k * 0.2), source=db.SOURCE_GLASS_DOOR_CAM,
                                 detection_class="animal", confidence=0.9, bbox=(50, 50, 60, 60),
                                 frame_w=100, frame_h=100, crop_path=f"crops/p{k}b.jpg",
                                 species="raccoon", crop_quality=10.0)
        _embed(conn, pa, [1, 0, 0])
        _embed(conn, pb, [0, 1, 0])
        ids += [pa, pb]
    pair = db.insert_visit(conn, source=db.SOURCE_GLASS_DOOR_CAM, species="raccoon",
                           individual_id=None, started_at=_ts(4 * DAY), ended_at=_ts(4 * DAY + 1),
                           detection_count=6, max_confidence=0.9, representative_detection_id=ids[0])
    db.assign_visit(conn, ids, pair)
    conn.commit()
    db.label_visit(conn, pair, "Stan")

    r = evalmod.eval_reid(conn, cfg, "raccoon", kfold_repeats=1)
    cv = r["confirmed_visits"]
    assert cv["solo_with_prototype"] == 4
    assert cv["excluded_multi_animal"] == 1               # the pair visit, correctly left out
    sep = r["separation"]
    assert sep["same_individual_pairs"]["median"] > sep["different_individual_pairs"]["median"]
    assert sep["auc"] == pytest.approx(1.0)               # perfectly separable by construction
    assert sep["same_night_positive_pairs_dropped"] == 0  # every visit is on its own night
    # LOO identification: with the self excluded, each visit's nearest individual is still correct
    # -- and blocking costs nothing here, because nothing was leaning on a same-night twin.
    loo = r["identification_loo"]
    assert loo["blocked"]["top1_accuracy"] == pytest.approx(1.0)
    assert loo["unblocked"]["top1_accuracy"] == pytest.approx(1.0)
    assert loo["session_leak_gap"] == pytest.approx(0.0)
    assert loo["blocked"]["n_probes"] == 4 and loo["blocked"]["n_scorable"] == 4
    assert r["identification_loo_centroid"]["blocked"]["top1_accuracy"] == pytest.approx(1.0)
    assert r["identification_loo_centroid"]["blocked"]["coverage"]["scorable_visits"] == 4


def test_eval_reid_and_its_summary_survive_a_one_night_corpus(conn, capsys):
    # A corpus where every same-individual pair is same-night has NO cross-session evidence: after
    # blocking, the positive class is empty. That is a finding to print, not a crash to hit -- the
    # console summary used to KeyError straight through it.
    cfg = config.Config()
    cfg.reid_co_presence_min = 1
    a = _solo_visit(conn, [1, 0, 0], start_min=0)
    b = _solo_visit(conn, [1, 0.05, 0], start_min=120)      # same night as `a`
    c = _solo_visit(conn, [0, 1, 0], start_min=5 * DAY)
    for v, name in ((a, "Stan"), (b, "Stan"), (c, "Notch")):
        db.label_visit(conn, v, name)

    r = evalmod.eval_reid(conn, cfg, "raccoon", kfold_repeats=1)
    assert r["separation"]["same_individual_pairs"]["n"] == 0
    assert r["separation"]["same_night_positive_pairs_dropped"] == 1
    assert r["separation"]["auc"] is None
    assert r["separation_unblocked"]["same_individual_pairs"]["n"] == 1
    evalmod._print_reid(r)                                   # must not raise
    out = capsys.readouterr().out
    assert "no cross-session evidence" in out
    # The identification section still prints -- it is the part that still has something to say.
    assert "LEAVE-ONE-VISIT-OUT IDENTIFICATION" in out


def test_eval_reid_thin_data_degrades_honestly(conn):
    cfg = config.Config()
    v = _solo_visit(conn, [1, 0, 0], start_min=0)
    db.label_visit(conn, v, "Stan")
    r = evalmod.eval_reid(conn, cfg, "raccoon")
    assert r["confirmed_visits"]["solo_with_prototype"] == 1
    assert "note" in r["separation"]                      # one solo visit -> nothing to separate


# ---------------------------------------------------------------------------
# Auto-assign sweep: the zero-error operating-point recommendation.
# ---------------------------------------------------------------------------

def test_auto_assign_sweep_recommends_full_coverage_when_clean():
    # Two well-separated individuals with two visits each, plus "Ghost" -- one visit only, so its
    # identity is absent from the LOO templates: the stand-in for a genuinely NEW animal.
    protos = {1: _u(1, 0, 0), 2: _u(1, 0.05, 0),
              3: _u(0, 1, 0), 4: _u(0.05, 1, 0),
              5: _u(0, 0, 1)}
    labels = {1: "Stan", 2: "Stan", 3: "Notch", 4: "Notch", 5: "Ghost"}
    sw = evalmod._auto_assign_sweep(_probes(protos, labels), repeats=1)
    assert sw["n_probes"] == 4 and sw["n_novel_probes"] == 1
    rec = sw["in_sample"]["recommended"]
    # Same-pairs sit ~1.0 with a huge lead; the Ghost probe's best match is ~0 (never accepted):
    # some point must cover all four known probes with zero errors of either kind.
    assert rec and rec["auto_named"] == 4 and rec["coverage"] == 1.0
    assert sw["recommended"] == rec                       # back-compat alias for older readers
    # The recommendation is also scored as a FIXED point, with a rule-of-three bound on its
    # zero observed errors -- "0 wrong out of 4" is not a 0% error rate.
    at = [p for p in sw["at_points"] if p["why"] == "in-sample recommendation"][0]
    assert at["overall"]["errors"] == 0
    assert at["overall"]["error_rate_upper_bound_95"] == pytest.approx(3.0 / 4)


def test_auto_assign_sweep_confident_wrong_match_blocks_recommendation():
    # A mislabelled/confusable corpus: "Stan"'s second visit actually points along Notch's axis,
    # as strongly as every correct match. Any grid point that accepts the correct matches also
    # accepts this wrong one -> there is NO zero-error operating point, and the sweep must say so
    # rather than recommend bars that would mis-name visits nightly.
    protos = {1: _u(1, 0, 0), 2: _u(0, 1, 0.02),
              3: _u(0, 1, 0), 4: _u(0, 1, 0.04)}
    labels = {1: "Stan", 2: "Stan", 3: "Notch", 4: "Notch"}
    sw = evalmod._auto_assign_sweep(_probes(protos, labels), repeats=1)
    assert sw["recommended"] is None
    assert sw["in_sample"]["recommended"] is None
    assert sw["in_sample"]["zero_error_points"] == []


def test_auto_assign_sweep_at_a_fixed_point_is_scored_directly():
    protos = {1: _u(1, 0, 0), 2: _u(1, 0.05, 0), 3: _u(0, 1, 0), 4: _u(0.05, 1, 0)}
    labels = {1: "Stan", 2: "Stan", 3: "Notch", 4: "Notch"}
    sw = evalmod._auto_assign_sweep(_probes(protos, labels), repeats=1,
                                    at_points=[(0.9999, 0.0, "impossible"), (0.5, 0.0, "loose")])
    by_why = {p["why"]: p["overall"] for p in sw["at_points"]}
    assert by_why["impossible"]["assigned"] == 0          # nothing clears a 0.9999 bar here
    assert by_why["loose"]["assigned"] == 4               # everything clears a loose one
    assert by_why["loose"]["wrong"] == 0


# ---------------------------------------------------------------------------
# HELD-OUT operating-point selection: the sweep must never see the fold it scores.
# ---------------------------------------------------------------------------

def _synthetic_probes(n_good=20):
    """`n_good` easy correct matches with spread-out margins, plus ONE confident WRONG match.

    The spread matters: it makes the max-coverage rule pick a LOOSE margin when the trap is not in
    the training data, which is exactly the mistake a held-out score is supposed to expose."""
    probes = [{"key": i, "truth": "Stan", "top": "Stan", "s1": 0.95,
               "lead": 0.05 + 0.25 * i / max(1, n_good - 1), "novel_probe": False}
              for i in range(n_good)]
    probes.append({"key": 999, "truth": "Stan", "top": "Notch", "s1": 0.90, "lead": 0.10,
                   "novel_probe": False})
    return probes


def test_kfold_selection_never_sees_the_test_fold(monkeypatch):
    # Direct proof, not inference: spy on every probe set the selector is handed. With k = n and
    # one repeat, each call must be handed exactly all-but-one probe, and across the n calls each
    # probe must be the held-out one exactly once.
    probes = [{"key": i, "truth": "Stan", "top": "Stan", "s1": 0.9, "lead": 0.2,
               "novel_probe": False} for i in range(6)]
    all_keys = {p["key"] for p in probes}
    seen_train = []
    real = em.select_operating_point

    def spy(train, *a, **kw):
        seen_train.append({p["key"] for p in train})
        return real(train, *a, **kw)

    monkeypatch.setattr(em, "select_operating_point", spy)
    em.kfold_operating_point(probes, k=6, repeats=1, seed=0)

    assert len(seen_train) == 6
    held_out = [all_keys - t for t in seen_train]
    assert all(len(h) == 1 for h in held_out)             # exactly one probe hidden per fold
    assert sorted(k for h in held_out for k in h) == sorted(all_keys)   # each hidden exactly once


def test_kfold_reports_the_error_the_in_sample_sweep_hides():
    probes = _synthetic_probes()
    # In sample, a zero-error point exists: raise the margin above the trap's 0.10 and it is
    # excluded. The sweep therefore reports a clean recommendation...
    rec = em.select_operating_point(probes)
    assert rec is not None
    assert em.evaluate_point(probes, rec["threshold"], rec["margin"])["errors"] == 0
    assert rec["margin"] > 0.10                           # the only way to dodge the trap

    # ...but when the trap is in the HELD-OUT fold, the training folds contain only clean probes,
    # the selector reaches for max coverage (a loose margin), and the trap gets mis-named. A
    # non-zero pooled error is only possible if selection genuinely never saw the test fold.
    ho = em.kfold_operating_point(probes, k=5, repeats=1, seed=0)
    assert ho["pooled"]["wrong"] >= 1
    assert ho["pooled"]["error_rate"] > 0
    assert "SELECTION PROCEDURE" in ho["note"]


def test_procedure_error_and_fixed_point_error_are_different_numbers():
    # The correction this whole phase exists to make explicit: the k-fold number grades the SWEEP,
    # a fixed point grades ITSELF, and they must not be reported as the same quantity.
    probes = _synthetic_probes()
    procedure = em.kfold_operating_point(probes, k=5, repeats=1, seed=0)["pooled"]["error_rate"]
    fixed = em.evaluate_point(probes, 0.90, 0.20)          # dodges the trap's 0.10 lead
    assert fixed["errors"] == 0 and fixed["error_rate"] == 0.0
    assert procedure > fixed["error_rate"]
    # A fixed point's fold spread pools back to exactly the whole-corpus number (no selection).
    spread = em.point_fold_spread(probes, 0.90, 0.20, k=5, repeats=1, seed=0)
    assert spread["overall"]["assigned"] == fixed["assigned"]


# ---------------------------------------------------------------------------
# --baseline: the regression gate.
# ---------------------------------------------------------------------------

def _artifact(**overrides) -> dict:
    a = {"meta": {"run_at": "2026-08-01T00:00:00-07:00", "git_commit": "deadbeef"},
         "reid": {"separation": {"auc": 0.80},
                  "identification_loo": {"blocked": {"top1_accuracy": 0.74,
                                                     "chance": {"rate": 0.35}},
                                         "unblocked": {"top1_accuracy": 0.81}},
                  "auto_assign_sweep": {
                      "in_sample": {"recommended": {"threshold": 0.88, "margin": 0.02,
                                                    "coverage": 0.12}},
                      "held_out_procedure": {"pooled": {"coverage": 0.10, "error_rate": 0.05}}}},
         "species": {"chance": {"rate": 0.5},
                     "calibration": {"ece": 0.10},
                     "precision_recall_f1": {"accuracy": 0.9,
                                             "macro_present_classes": {"f1": 0.8}}}}
    for dotted, val in overrides.items():
        cur = a
        parts = dotted.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = val
    return a


def test_compare_artifacts_flags_a_drop_past_tolerance():
    base = _artifact()
    worse = _artifact(**{"reid.identification_loo.blocked.top1_accuracy": 0.60})
    diff = evalmod.compare_artifacts(base, worse, tolerance=0.05)
    assert not diff["ok"]
    assert [r["metric"] for r in diff["regressions"]] == ["LOO top-1 (session-blocked)"]


def test_compare_artifacts_ignores_noise_inside_tolerance():
    base = _artifact()
    jittered = _artifact(**{"reid.identification_loo.blocked.top1_accuracy": 0.71,
                            "reid.separation.auc": 0.78})
    assert evalmod.compare_artifacts(base, jittered, tolerance=0.05)["ok"]


def test_compare_artifacts_direction_is_per_metric():
    # An error rate RISING is the regression; an accuracy rising is not.
    base = _artifact()
    worse = _artifact(**{"reid.auto_assign_sweep.held_out_procedure.pooled.error_rate": 0.30})
    assert not evalmod.compare_artifacts(base, worse, tolerance=0.05)["ok"]
    better = _artifact(**{"reid.identification_loo.blocked.top1_accuracy": 0.99})
    assert evalmod.compare_artifacts(base, better, tolerance=0.05)["ok"]


def test_compare_artifacts_moved_operating_point_is_news_not_failure():
    # The recommended bars are 'info': a corpus that legitimately changed must not fail the gate.
    moved = _artifact(**{"reid.auto_assign_sweep.in_sample.recommended.threshold": 0.60,
                         "reid.auto_assign_sweep.in_sample.recommended.margin": 0.30})
    diff = evalmod.compare_artifacts(_artifact(), moved, tolerance=0.001)
    assert diff["ok"]
    row = [r for r in diff["rows"] if r["metric"] == "recommended threshold"][0]
    assert row["delta"] == pytest.approx(-0.28) and not row["regressed"]


def test_compare_artifacts_missing_metric_is_not_a_regression():
    thin = {"meta": {}, "reid": {}}
    diff = evalmod.compare_artifacts(_artifact(), thin, tolerance=0.05)
    assert diff["ok"]
    assert all(r["note"] for r in diff["rows"] if r["current"] is None)


def test_compare_artifacts_falls_back_to_a_legacy_artifact_path():
    # A pre-A3 artifact has no blocked/unblocked split; the centroid-scorable metric is the
    # like-for-like fallback, and the row says a fallback was used.
    legacy = {"meta": {}, "reid": {"identification_loo": {"top1_accuracy": 0.812}}}
    current = _artifact(**{"reid.identification_loo_centroid":
                           {"unblocked": {"top1_accuracy_scorable": 0.80}}})
    diff = evalmod.compare_artifacts(legacy, current, tolerance=0.05)
    row = [r for r in diff["rows"] if r["metric"].startswith("centroid LOO")][0]
    assert row["baseline"] == pytest.approx(0.812) and row["current"] == pytest.approx(0.80)
    assert "fallback" in row["note"]


def test_baseline_flag_exits_non_zero_on_a_regression(conn, db_path, tmp_path, monkeypatch, capsys):
    # Full CLI path: build a corpus that scores badly on purpose, hand main() a baseline artifact
    # claiming perfection, and require a non-zero exit. --no-save keeps reports/ untouched.
    s1 = _solo_visit(conn, [1, 0, 0], start_min=0)
    s2 = _solo_visit(conn, [1, 0.02, 0], start_min=200)     # same night as s1
    s3 = _solo_visit(conn, [0, 0, 1], start_min=10 * DAY)
    t1 = _solo_visit(conn, [0.9, 0.44, 0], start_min=15 * DAY)
    for v, name in ((s1, "Stan"), (s2, "Stan"), (s3, "Stan"), (t1, "Notch")):
        db.label_visit(conn, v, name)
    conn.commit()

    good = tmp_path / "baseline_good.json"
    good.write_text(json.dumps(_artifact(**{
        "reid.identification_loo.blocked.top1_accuracy": 1.0, "reid.separation.auc": 1.0})))
    lenient = tmp_path / "baseline_lenient.json"
    lenient.write_text(json.dumps(_artifact(**{
        "reid.identification_loo.blocked.top1_accuracy": 0.0, "reid.separation.auc": 0.0,
        "reid.identification_loo.unblocked.top1_accuracy": 0.0,
        "reid.auto_assign_sweep.in_sample.recommended.coverage": 0.0,
        "reid.auto_assign_sweep.held_out_procedure.pooled.coverage": 0.0,
        "reid.auto_assign_sweep.held_out_procedure.pooled.error_rate": 1.0})))

    original_db_path = config.CONFIG.db_path
    try:
        def _run(baseline):
            monkeypatch.setattr("sys.argv", ["eval.py", "--reid", "--no-save",
                                             "--db", str(db_path), "--baseline", str(baseline),
                                             "--kfold-repeats", "1"])
            return evalmod.main()

        assert _run(good) == 1                 # regressed against an impossible baseline
        out = capsys.readouterr().out
        assert "REGRESSION" in out
        assert _run(lenient) == 0              # nothing regressed -> clean exit
    finally:
        config.CONFIG.db_path = original_db_path


def test_latest_artifact_picks_the_newest_stamp(tmp_path):
    for name in ("eval_20260101T000000Z.json", "eval_20260718T030508Z.json",
                 "eval_20260305T000000Z.json"):
        (tmp_path / name).write_text("{}")
    assert evalmod.latest_artifact(tmp_path).name == "eval_20260718T030508Z.json"
    assert evalmod.latest_artifact(tmp_path / "empty") is None


def test_load_artifact_missing_file_is_a_clean_exit(tmp_path):
    with pytest.raises(SystemExit):
        evalmod.load_artifact(tmp_path / "nope.json")
