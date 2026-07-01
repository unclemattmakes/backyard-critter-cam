"""
Tests for evalmetrics.py -- the pure-logic metric helpers the eval harness grades everything with.

The whole reason these helpers exist (instead of sklearn) is so they can be checked by hand on
synthetic inputs with KNOWN answers -- a metric you can't verify is a metric you can't trust to
grade the classifier or the re-ID separation. So every test here has a worked-out expected value:
a 2x2 confusion the F1 is derived from, a perfectly-separable set whose AUC must be exactly 1.0,
a tied-score case that pins the average-rank tie rule, a calibration curve with a bin whose
accuracy is hand-counted. Pure numpy/stdlib; no DB, GPU, or model.
"""
from __future__ import annotations

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
