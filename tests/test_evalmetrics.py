"""
Tests for evalmetrics.py -- the pure-logic metric helpers the eval harness grades everything with.

The whole reason these helpers exist (instead of sklearn) is so they can be checked by hand on
synthetic inputs with KNOWN answers -- a metric you can't verify is a metric you can't trust to
grade the classifier or the re-ID separation. So every test here has a worked-out expected value:
a 2x2 confusion the F1 is derived from, a perfectly-separable set whose AUC must be exactly 1.0,
a tied-score case that pins the average-rank tie rule, a calibration curve with a bin whose
accuracy is hand-counted. Pure numpy/stdlib; no DB, GPU, or model.

The second half covers the identification PROTOCOL that moved in here in 2026-08 (phase A3):
majority_baseline, night_key, leave_one_visit_out, and the operating-point sweep. Same standard --
every case is small enough to work out on paper, and each one pins a decision that is easy to get
subtly wrong and impossible to notice afterwards: what "same night" means at 01:00, whether the
scorer can see the labels (it cannot), whether a held-out fold leaks into selection (it must not),
and whether zero observed errors is allowed to print as a zero error rate (it is not).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

import evalmetrics as em


# ---------------------------------------------------------------------------
# precision / recall / F1 + confusion matrix.
# ---------------------------------------------------------------------------

def test_precision_recall_f1_binary_worked_example():
    # 2 classes. Lay out TP/FP/FN by hand:
    #   raccoon: predicted for 3 rows, 2 truly raccoon (TP=2, FP=1); 1 raccoon called crow (FN=1)
    #   crow:    predicted for 2 rows, 1 truly crow (TP=1, FP=1); 1 crow called raccoon (FN=1)
    y_true = ["raccoon", "raccoon", "raccoon", "crow", "crow"]
    y_pred = ["raccoon", "raccoon", "crow", "raccoon", "crow"]
    r = em.precision_recall_f1(y_true, y_pred)
    rac = r["per_label"]["raccoon"]
    assert (rac["tp"], rac["fp"], rac["fn"]) == (2, 1, 1)
    assert rac["precision"] == pytest.approx(2 / 3)
    assert rac["recall"] == pytest.approx(2 / 3)
    assert rac["f1"] == pytest.approx(2 / 3)
    crow = r["per_label"]["crow"]
    assert crow["precision"] == pytest.approx(1 / 2)
    assert crow["recall"] == pytest.approx(1 / 2)
    assert r["accuracy"] == pytest.approx(3 / 5)      # 3 of 5 rows on the diagonal


def test_precision_recall_f1_perfect_and_empty_class():
    y = ["a", "a", "b"]
    r = em.precision_recall_f1(y, y, labels=["a", "b", "c"])  # 'c' never appears
    assert r["per_label"]["a"]["f1"] == 1.0 and r["per_label"]["b"]["f1"] == 1.0
    # An absent class scores 0 (not NaN) and carries zero support -- no crash on a thin slice.
    assert r["per_label"]["c"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                                   "support": 0, "tp": 0, "fp": 0, "fn": 0}
    assert r["macro"]["f1"] == pytest.approx(2 / 3)   # (1 + 1 + 0) / 3


def test_precision_recall_f1_length_mismatch_raises():
    with pytest.raises(ValueError):
        em.precision_recall_f1(["a"], ["a", "b"])


def test_confusion_matrix_counts_and_off_diagonal():
    y_true = ["raccoon", "raccoon", "crow"]
    y_pred = ["raccoon", "crow", "crow"]
    cm = em.confusion_matrix(y_true, y_pred, labels=["raccoon", "crow"])
    assert cm["matrix"]["raccoon"]["raccoon"] == 1
    assert cm["matrix"]["raccoon"]["crow"] == 1     # one raccoon mislabelled crow
    assert cm["matrix"]["crow"]["crow"] == 1
    assert cm["n"] == 3


def test_confusion_matrix_skips_labels_outside_the_requested_set():
    # A row whose true or pred label isn't in `labels` is dropped, not crashed on.
    cm = em.confusion_matrix(["a", "z"], ["a", "a"], labels=["a"])
    assert cm["matrix"]["a"]["a"] == 1 and cm["n"] == 1


# ---------------------------------------------------------------------------
# ROC-AUC + best threshold (the re-ID separation grader).
# ---------------------------------------------------------------------------

def test_roc_auc_perfect_separation_is_one():
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    labels = [1, 1, 1, 0, 0]
    assert em.roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_reversed_separation_is_zero():
    scores = [0.1, 0.2, 0.9, 0.8]
    labels = [1, 1, 0, 0]                 # positives score LOWER than negatives
    assert em.roc_auc(scores, labels) == pytest.approx(0.0)


def test_roc_auc_ties_use_average_rank_half():
    # One positive and one negative with the IDENTICAL score -> AUC is exactly 0.5 (a coin flip),
    # which only comes out right if ties share their average rank.
    assert em.roc_auc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)


def test_roc_auc_known_fraction():
    # positives {3,1}, negatives {2,0}. Pairs where pos > neg: (3>2),(3>0),(1>0) = 3 of 4 -> 0.75.
    assert em.roc_auc([3, 1, 2, 0], [1, 1, 0, 0]) == pytest.approx(0.75)


def test_roc_auc_single_class_returns_half():
    assert em.roc_auc([0.9, 0.8], [1, 1]) == 0.5   # nothing to separate


def test_best_threshold_finds_the_clean_cut():
    # Same >= 0.7, different <= 0.4: any cut in (0.4, 0.7] separates perfectly. Youden's J picks the
    # highest-scoring positive's value (0.7) as the >= cut that still catches every positive.
    scores = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]
    labels = [1, 1, 1, 0, 0, 0]
    bt = em.best_threshold(scores, labels)
    assert bt["threshold"] == pytest.approx(0.7)
    assert bt["tpr"] == pytest.approx(1.0) and bt["fpr"] == pytest.approx(0.0)
    assert bt["accuracy"] == pytest.approx(1.0)


def test_best_threshold_empty_is_null():
    bt = em.best_threshold([], [])
    assert bt["threshold"] is None


# ---------------------------------------------------------------------------
# Calibration / reliability curve (the 0.8-trust folklore grader).
# ---------------------------------------------------------------------------

def test_reliability_curve_perfectly_calibrated_has_zero_ece():
    # In each bin, accuracy == mean confidence, so the calibration error is 0.
    # bin [0.8,0.9): two preds at conf 0.8, exactly one correct -> accuracy 0.5... not calibrated.
    # Build a genuinely-calibrated set instead: at conf 1.0 always right, at conf 0.0 always wrong.
    conf = [1.0, 1.0, 0.0, 0.0]
    correct = [1, 1, 0, 0]
    rc = em.reliability_curve(conf, correct, n_bins=10)
    assert rc["ece"] == pytest.approx(0.0)


def test_reliability_curve_miscalibrated_ece_is_the_gap():
    # All predictions at confidence 0.9, but only half are right -> the model is OVER-confident by
    # 0.4, and with every prediction in one bin the ECE is exactly that gap.
    conf = [0.9, 0.9, 0.9, 0.9]
    correct = [1, 0, 1, 0]
    rc = em.reliability_curve(conf, correct, n_bins=10)
    bin_9 = [b for b in rc["bins"] if b["lo"] == pytest.approx(0.9)][0]
    assert bin_9["n"] == 4 and bin_9["accuracy"] == pytest.approx(0.5)
    assert bin_9["mean_confidence"] == pytest.approx(0.9)
    assert rc["ece"] == pytest.approx(0.4)


def test_reliability_curve_confidence_one_lands_in_last_bin():
    # The top bin is closed on the right so a confidence of exactly 1.0 isn't dropped.
    rc = em.reliability_curve([1.0], [1], n_bins=10)
    assert rc["bins"][-1]["n"] == 1


def test_reliability_curve_empty_input():
    rc = em.reliability_curve([], [], n_bins=5)
    assert rc["ece"] is None and all(b["n"] == 0 for b in rc["bins"])


def test_average_ranks_handles_ties():
    # values 10,10,20 -> the two 10s share rank (1+2)/2 = 1.5; the 20 gets rank 3.
    r = em._average_ranks(np.array([10.0, 10.0, 20.0]))
    assert list(r) == [1.5, 1.5, 3.0]


# ---------------------------------------------------------------------------
# Chance rates -- the number that has to sit beside every accuracy.
# ---------------------------------------------------------------------------

def test_majority_baseline_worked_example():
    b = em.majority_baseline(["Stan"] * 48 + ["Notch"] * 46 + ["Pedro"] * 36 + ["Elliot"] * 8)
    assert b["n"] == 138 and b["n_classes"] == 4
    assert b["label"] == "Stan"
    assert b["rate"] == pytest.approx(48 / 138)      # the 0.348 this project quotes
    assert b["uniform_rate"] == pytest.approx(0.25)


def test_majority_baseline_empty_and_ties():
    assert em.majority_baseline([])["rate"] is None
    # A tie must resolve deterministically (by label order), or two runs disagree on the baseline.
    assert em.majority_baseline(["b", "a"])["label"] == em.majority_baseline(["a", "b"])["label"]


# ---------------------------------------------------------------------------
# night_key: the -12h shift that makes one evening one session.
# ---------------------------------------------------------------------------

def test_night_key_groups_an_evening_across_midnight():
    evening = em.night_key("2026-07-31T22:15:00-07:00")
    after_midnight = em.night_key("2026-08-01T02:40:00-07:00")
    next_evening = em.night_key("2026-08-01T22:15:00-07:00")
    assert evening == after_midnight == "2026-07-31"
    assert next_evening == "2026-08-01"


def test_night_key_boundary_is_noon():
    # 11:59 still belongs to the night that started the previous evening; 12:00 starts a new one.
    assert em.night_key("2026-07-31T11:59:00+00:00") == "2026-07-30"
    assert em.night_key("2026-07-31T12:00:00+00:00") == "2026-07-31"


def test_night_key_accepts_a_datetime_and_a_custom_shift():
    assert em.night_key(datetime(2026, 7, 31, 3, 0, 0)) == "2026-07-30"
    assert em.night_key("2026-07-31T03:00:00+00:00", shift_hours=0.0) == "2026-07-31"


# ---------------------------------------------------------------------------
# The protocol itself: leave_one_visit_out.
# ---------------------------------------------------------------------------

def _sim_table(table):
    """A similarity callable backed by a hand-written {(a, b): score} table (symmetric)."""
    def sim(a, b):
        return float(table.get((a, b), table.get((b, a), 0.0)))
    return sim


def test_leave_one_visit_out_never_scores_a_unit_against_itself():
    seen = []

    def sim(a, b):
        seen.append((a, b))
        return 0.5
    units = [em.LooUnit("a", "X", "n1"), em.LooUnit("b", "X", "n2"), em.LooUnit("c", "Y", "n3")]
    em.leave_one_visit_out(units, sim)
    assert seen and all(a != b for a, b in seen)


def test_leave_one_visit_out_similarity_is_the_only_thing_it_ranks_on():
    # The scorer is handed KEYS, never labels -- that is what makes a trait scorer droppable in
    # without it being able to cheat off the ground truth.
    units = [em.LooUnit(1, "Stan", "n1"), em.LooUnit(2, "Stan", "n2"), em.LooUnit(3, "Notch", "n3")]
    keys = {u.key for u in units}

    def sim(a, b):
        assert a in keys and b in keys               # keys, not names
        return {frozenset((1, 2)): 0.9, frozenset((1, 3)): 0.4,
                frozenset((2, 3)): 0.4}[frozenset((a, b))]

    r = em.leave_one_visit_out(units, sim)
    assert r["top1_accuracy"] == pytest.approx(2 / 3)    # 1 and 2 right, 3 is novel -> wrong
    assert r["n_novel_probes"] == 1


def test_leave_one_visit_out_max_vs_mean_aggregation():
    # Stan has one great template and one terrible one; Notch has a mediocre pair. Nearest-template
    # (max) picks Stan; averaging the pairwise scores picks Notch. Both are defensible, so the
    # protocol makes the choice explicit rather than burying it.
    units = [em.LooUnit("p", "Stan", "n0"), em.LooUnit("s1", "Stan", "n1"),
             em.LooUnit("s2", "Stan", "n2"), em.LooUnit("t1", "Notch", "n3"),
             em.LooUnit("t2", "Notch", "n4")]
    table = {("p", "s1"): 0.9, ("p", "s2"): 0.0, ("p", "t1"): 0.6, ("p", "t2"): 0.6}
    by_max = em.leave_one_visit_out(units, _sim_table(table), aggregate="max")
    by_mean = em.leave_one_visit_out(units, _sim_table(table), aggregate="mean")
    assert {p["key"]: p["top"] for p in by_max["probes"]}["p"] == "Stan"
    assert {p["key"]: p["top"] for p in by_mean["probes"]}["p"] == "Notch"


def test_leave_one_visit_out_embargo_narrows_the_allowed_templates():
    t0 = datetime(2026, 6, 1, 21, 0)
    units = [em.LooUnit("p", "Stan", "n0", t0),
             em.LooUnit("near", "Stan", "n1", t0 + timedelta(days=1)),
             em.LooUnit("far", "Notch", "n2", t0 + timedelta(days=30))]
    table = {("p", "near"): 0.9, ("p", "far"): 0.3}
    none = em.leave_one_visit_out(units, _sim_table(table), embargo_days=None)
    wide = em.leave_one_visit_out(units, _sim_table(table), embargo_days=7)
    assert {p["key"]: p["top"] for p in none["probes"]}["p"] == "Stan"
    # With a 7-day embargo the only allowed template is 30 days away -- and it is the wrong animal.
    assert {p["key"]: p["top"] for p in wide["probes"]}["p"] == "Notch"


def test_leave_one_visit_out_allow_hook_can_only_remove():
    units = [em.LooUnit(1, "Stan", "n1"), em.LooUnit(2, "Stan", "n2"), em.LooUnit(3, "Notch", "n3")]
    table = {(1, 2): 0.9, (1, 3): 0.4, (2, 3): 0.4}
    r = em.leave_one_visit_out(units, _sim_table(table), allow=lambda p, t: t.key != 2)
    p1 = {p["key"]: p for p in r["probes"]}[1]
    assert p1["top"] == "Notch" and p1["novel_probe"] is True   # Stan's only template was removed


def test_leave_one_visit_out_rejects_duplicate_keys():
    units = [em.LooUnit(1, "Stan"), em.LooUnit(1, "Notch")]
    with pytest.raises(ValueError):
        em.leave_one_visit_out(units, lambda a, b: 0.0)


def test_leave_one_visit_out_accepts_tuples_and_dicts():
    tup = em.leave_one_visit_out([(1, "Stan", "n1"), (2, "Stan", "n2")], lambda a, b: 0.9)
    dct = em.leave_one_visit_out([{"key": 1, "label": "Stan", "night": "n1"},
                                  {"key": 2, "label": "Stan", "night": "n2"}], lambda a, b: 0.9)
    assert tup["top1_accuracy"] == dct["top1_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Operating points.
# ---------------------------------------------------------------------------

def _p(key, truth, top, s1, lead, novel=False):
    return {"key": key, "truth": truth, "top": top, "s1": s1, "lead": lead, "novel_probe": novel}


def test_evaluate_point_counts_both_error_channels():
    probes = [_p(1, "Stan", "Stan", 0.9, 0.3),          # named, right
              _p(2, "Stan", "Notch", 0.9, 0.3),         # named, WRONG
              _p(3, "Stan", "Stan", 0.5, 0.3),          # below threshold -> not named
              _p(4, "Ghost", "Stan", 0.9, 0.3, True)]   # novel false accept
    ev = em.evaluate_point(probes, 0.8, 0.1)
    assert ev["assigned"] == 2 and ev["wrong"] == 1 and ev["novel_false_accepts"] == 1
    assert ev["coverage"] == pytest.approx(2 / 3)       # over the 3 KNOWN probes
    assert ev["wrong_name_rate"] == pytest.approx(0.5)
    assert ev["names_written"] == 3 and ev["errors"] == 2
    assert ev["error_rate"] == pytest.approx(2 / 3)
    assert ev["error_rate_upper_bound_95"] is None      # the bound only applies at zero errors
    assert ev["wrong_visit_keys"] == [2]


def test_evaluate_point_margin_gate_is_independent_of_the_threshold():
    probes = [_p(1, "Stan", "Stan", 0.95, 0.01)]
    assert em.evaluate_point(probes, 0.9, 0.0)["assigned"] == 1
    assert em.evaluate_point(probes, 0.9, 0.02)["assigned"] == 0   # clears similarity, not margin


def test_evaluate_point_rule_of_three_bound_at_zero_errors():
    probes = [_p(i, "Stan", "Stan", 0.95, 0.3) for i in range(20)]
    ev = em.evaluate_point(probes, 0.9, 0.1)
    assert ev["errors"] == 0
    assert ev["error_rate_upper_bound_95"] == pytest.approx(3 / 20)   # ~15%, not 0%


def test_sweep_prefers_coverage_then_the_safer_point():
    # Many points name all 3; the sweep must return the highest threshold (then margin) among
    # them -- identical measured behaviour, more headroom against the next raccoon.
    probes = [_p(i, "Stan", "Stan", 0.86, 0.30) for i in range(3)]
    pts = em.sweep_operating_points(probes)
    assert pts[0]["auto_named"] == 3
    assert pts[0]["threshold"] == pytest.approx(0.86)
    assert pts[0]["margin"] == pytest.approx(0.30)


def test_select_operating_point_returns_none_when_no_point_is_safe():
    # A confidently wrong match at the top of the range: nothing on the grid excludes it.
    probes = [_p(1, "Stan", "Stan", 0.95, 0.4), _p(2, "Stan", "Notch", 0.95, 0.4)]
    assert em.select_operating_point(probes) is None
    assert em.sweep_operating_points(probes) == []


def test_kfold_is_deterministic_for_a_seed():
    probes = [_p(i, "Stan", "Stan", 0.9, 0.2) for i in range(10)]
    a = em.kfold_operating_point(probes, k=5, repeats=2, seed=7)
    b = em.kfold_operating_point(probes, k=5, repeats=2, seed=7)
    c = em.kfold_operating_point(probes, k=5, repeats=2, seed=8)
    assert a["pooled"] == b["pooled"]
    assert a["seed"] == 7 and c["seed"] == 8


def test_kfold_counts_a_fold_with_no_safe_point_as_zero_coverage():
    # Training folds that support no zero-error point must name nothing -- and their held-out
    # probes must still count in the denominator, or coverage flatters the procedure.
    probes = [_p(1, "Stan", "Stan", 0.95, 0.4), _p(2, "Stan", "Notch", 0.95, 0.4),
              _p(3, "Stan", "Notch", 0.95, 0.4), _p(4, "Stan", "Notch", 0.95, 0.4)]
    ho = em.kfold_operating_point(probes, k=2, repeats=1, seed=0)
    assert ho["folds_with_no_safe_point"] >= 1
    assert ho["pooled"]["n_known_probes"] == 4          # every probe was held out exactly once


def test_kfold_too_few_probes_degrades_honestly():
    assert "note" in em.kfold_operating_point([_p(1, "Stan", "Stan", 0.9, 0.2)])


def test_point_fold_spread_pools_back_to_the_whole_corpus_number():
    probes = [_p(i, "Stan", "Stan", 0.9, 0.2) for i in range(10)]
    spread = em.point_fold_spread(probes, 0.85, 0.1, k=5, repeats=1, seed=3)
    assert spread["overall"]["assigned"] == 10
    assert spread["folds"]["coverage_min"] == pytest.approx(1.0)
    assert spread["folds"]["errors_max_in_a_fold"] == 0
