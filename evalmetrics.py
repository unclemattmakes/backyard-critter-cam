"""
Pure-logic evaluation metrics for the eval harness (eval.py) -- and NOTHING else.

Why a separate module? The whole point of the eval harness is a DEFENSIBLE, OFFLINE
measurement: no cloud, no LLM, no sklearn. sklearn is not installed here (and dragging it in
for six textbook formulas would violate the "boring and robust, no new heavy deps" rule the
rest of the project holds to), so the classifier / re-ID scores it needs -- precision, recall,
F1, a confusion matrix, ROC-AUC, and a calibration (reliability) curve -- are implemented here
from scratch on numpy + stdlib. Keeping them here, apart from any DB or model code, is what
makes them trivially UNIT-TESTABLE on synthetic inputs with known answers (tests/test_evalmetrics.py):
a metric you can't check by hand is a metric you can't trust to grade the rest of the system.

Everything is a pure function of its arguments -- no I/O, no globals, no config. Labels are
plain hashable values (usually species strings); scores are floats; "correct" flags are 0/1.
The tie handling in roc_auc and the empty-input guards everywhere are deliberate: real yard
data has confidence ties and thin per-species slices, and a metric that raises on a degenerate
slice is useless exactly when you most want a number.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Classification: confusion matrix + per-label precision / recall / F1.
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: Sequence[Hashable], y_pred: Sequence[Hashable],
                     labels: Sequence[Hashable] | None = None) -> dict:
    """Confusion counts as a nested dict counts[true][pred] -> n, plus the ordered label list.

    Returns {"labels": [...], "matrix": {true: {pred: n, ...}, ...}, "n": total}. `labels` fixes
    the row/column order (and can include labels absent from the data, e.g. a species nobody has
    confirmed yet); when omitted it's the sorted union of everything seen. A dict-of-dicts (not a
    dense numpy array) because the yard's label set is small, sparse, and string-keyed -- readable
    straight out of the JSON artifact without a separate legend."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=_sort_key)
    labels = list(labels)
    idx = set(labels)
    matrix = {t: {p: 0 for p in labels} for t in labels}
    n = 0
    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            # A label outside the requested set is silently skipped so a caller can scope the
            # matrix to a subset of classes without pre-filtering the rows.
            continue
        matrix[t][p] += 1
        n += 1
    return {"labels": labels, "matrix": matrix, "n": n}


def precision_recall_f1(y_true: Sequence[Hashable], y_pred: Sequence[Hashable],
                        labels: Sequence[Hashable] | None = None) -> dict:
    """Per-label precision / recall / F1 + support, with macro and micro averages.

    Standard one-vs-rest definitions, computed per label:
        precision = TP / (TP + FP)   -- of the crops the model CALLED this species, how many were?
        recall    = TP / (TP + FN)   -- of the crops that ARE this species, how many did it catch?
        f1        = harmonic mean of the two.
    A zero denominator yields 0.0 (not NaN) so a species the model never predicted, or that never
    appears in the truth, still returns a clean row -- the support count tells you it's empty.

    Returns {"per_label": {label: {precision, recall, f1, support, tp, fp, fn}}, "macro": {...},
    "micro": {...}, "accuracy": ..., "n": ...}. `macro` is the unweighted mean over labels (every
    species counts equally -- the honest read when a rare crow matters as much as a common raccoon);
    `micro` pools all TP/FP/FN first (dominated by the common classes). Both are reported because
    they answer different questions and disagree exactly when the class balance is skewed, which on
    a raccoon-heavy yard it always is."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=_sort_key)
    labels = list(labels)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    correct = 0
    n = 0
    label_set = set(labels)
    for t, p in zip(y_true, y_pred):
        n += 1
        if t == p:
            correct += 1
        if t in label_set:
            support[t] += 1
            if t == p:
                tp[t] += 1
            else:
                fn[t] += 1
        if p in label_set and p != t:
            fp[p] += 1

    per_label = {}
    for lab in labels:
        prec = _safe_div(tp[lab], tp[lab] + fp[lab])
        rec = _safe_div(tp[lab], tp[lab] + fn[lab])
        f1 = _safe_div(2 * prec * rec, prec + rec)
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1,
                          "support": support[lab], "tp": tp[lab], "fp": fp[lab], "fn": fn[lab]}

    macro = {k: float(np.mean([per_label[l][k] for l in labels])) if labels else 0.0
             for k in ("precision", "recall", "f1")}
    tot_tp = sum(tp.values())
    tot_fp = sum(fp.values())
    tot_fn = sum(fn.values())
    mprec = _safe_div(tot_tp, tot_tp + tot_fp)
    mrec = _safe_div(tot_tp, tot_tp + tot_fn)
    micro = {"precision": mprec, "recall": mrec, "f1": _safe_div(2 * mprec * mrec, mprec + mrec)}
    return {"per_label": per_label, "macro": macro, "micro": micro,
            "accuracy": _safe_div(correct, n), "n": n}


# ---------------------------------------------------------------------------
# Calibration: does a predicted confidence of 0.8 actually mean ~80% right?
# This is the direct test of the "0.8 trust / 0.5 look closer" folklore.
# ---------------------------------------------------------------------------

def reliability_curve(confidences: Sequence[float], correct: Sequence[int],
                      n_bins: int = 10) -> dict:
    """Bin predictions by confidence and compare mean confidence vs actual accuracy per bin.

    The reliability (calibration) curve is the honest test of "trust >= 0.8, look closer < 0.5":
    for the model to be trustworthy at a threshold, crops it labels with ~0.8 confidence must
    actually be right ~80% of the time. Each bin reports [lo, hi), how many predictions fell in
    it (`n` -- watch this: a bin with n=1 says nothing), the mean predicted confidence, and the
    empirical accuracy. A perfectly-calibrated model sits on the diagonal (accuracy == confidence).

    Also returns ECE -- Expected Calibration Error -- the support-weighted mean gap between
    confidence and accuracy across non-empty bins: one number for "how far off is the confidence,
    overall" (0 = perfectly calibrated). `correct` is 0/1 per prediction (was the model right?)."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n_total = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin is closed on the right so a confidence of exactly 1.0 lands somewhere.
        in_bin = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        n = int(in_bin.sum())
        if n:
            mean_conf = float(conf[in_bin].mean())
            acc = float(corr[in_bin].mean())
            ece += (n / n_total) * abs(acc - mean_conf)
        else:
            mean_conf = acc = None
        bins.append({"lo": round(lo, 3), "hi": round(hi, 3), "n": n,
                     "mean_confidence": mean_conf, "accuracy": acc})
    return {"bins": bins, "ece": ece if n_total else None, "n": n_total}


# ---------------------------------------------------------------------------
# Ranking / separation: ROC-AUC and the best operating threshold.
# Used by the re-ID eval to score same-individual vs different-individual
# visit-prototype cosine similarity.
# ---------------------------------------------------------------------------

def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve, via the Mann-Whitney U statistic (average ranks for ties).

    AUC = P(a random positive scores higher than a random negative) -- exactly the question the
    re-ID eval asks: does a same-individual visit pair reliably out-score a different-individual
    pair? 0.5 = no separation (a coin flip), 1.0 = perfect. Computed as the rank-sum statistic
    rather than by integrating the curve, so it's exact and O(n log n); ties get AVERAGE ranks so
    a cluster of equal cosines can't inflate or deflate the score. Returns 0.5 when either class
    is empty (nothing to separate)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(scores)
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(scores: Sequence[float], labels: Sequence[int]) -> dict:
    """ROC points (threshold, tpr, fpr) swept over every distinct score, plus the AUC.

    Threshold semantics: a pair is called "same" when its score >= threshold. Sweeping the unique
    scores from high to low traces the curve from (fpr=0, tpr=0) up to (1, 1). Returned as parallel
    lists so it lands cleanly in JSON for later plotting, without pulling in a plotting dep here."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    thresholds = np.unique(scores)[::-1]  # high -> low
    tpr, fpr = [], []
    for t in thresholds:
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        tpr.append(_safe_div(tp, n_pos))
        fpr.append(_safe_div(fp, n_neg))
    return {"thresholds": [float(t) for t in thresholds], "tpr": tpr, "fpr": fpr,
            "auc": roc_auc(scores, labels)}


def best_threshold(scores: Sequence[float], labels: Sequence[int]) -> dict:
    """The score threshold that best separates the two classes, by Youden's J (tpr - fpr).

    Youden's J picks the operating point furthest above the chance diagonal -- the single cut that
    maximises (correctly-called-same) minus (wrongly-called-same). This is the number the eval
    holds up against the live config: "current reid_novel_threshold X; the data's best cut is Y."
    Returns {threshold, tpr, fpr, youden_j, accuracy} at that point (accuracy = fraction of all
    pairs called correctly there). Empty / single-class input returns a null threshold."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0 or len(scores) == 0:
        return {"threshold": None, "tpr": None, "fpr": None, "youden_j": None, "accuracy": None}
    best = {"youden_j": -1.0, "threshold": None, "tpr": None, "fpr": None, "accuracy": None}
    for t in np.unique(scores):
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        tn = n_neg - fp
        tpr = tp / n_pos
        fpr = fp / n_neg
        j = tpr - fpr
        if j > best["youden_j"]:
            best = {"youden_j": float(j), "threshold": float(t), "tpr": float(tpr),
                    "fpr": float(fpr), "accuracy": float((tp + tn) / (n_pos + n_neg))}
    return best


# ---------------------------------------------------------------------------
# small internals
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    """a / b, but 0.0 when b == 0 -- an empty class scores 0, never NaN or a ZeroDivisionError."""
    return float(a) / float(b) if b else 0.0


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks of `x` with ties sharing their average rank (the tie rule ROC-AUC needs)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average of the tied block
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _sort_key(v: Hashable):
    """Stable ordering for a mixed label set: sort by (type-name, value) so strings and a sentinel
    object never trip the '<' between str and, say, None during sorted()."""
    return (type(v).__name__, str(v))
