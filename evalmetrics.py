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

It also holds the project's identification PROTOCOL (added 2026-08-05, phase A3): the
session-blocked leave-one-visit-out evaluation, plus the operating-point sweep and the
held-out k-fold wrapper around it. That lives here, next to the metrics and away from the DB,
for one reason: every future identity idea -- a trait scorer, a whitened metric, a re-ranked
prototype -- must be measured on the SAME protocol as the appearance embedding it hopes to
beat, and the only way to guarantee that is to make the protocol a pure function you inject a
similarity callable into. `leave_one_visit_out()` is that function; `eval.py` is just one
caller of it. See its docstring for the contract.

Everything is a pure function of its arguments -- no I/O, no globals, no config. Labels are
plain hashable values (usually species strings); scores are floats; "correct" flags are 0/1.
The tie handling in roc_auc and the empty-input guards everywhere are deliberate: real yard
data has confidence ties and thin per-species slices, and a metric that raises on a degenerate
slice is useless exactly when you most want a number.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Callable, Hashable, NamedTuple, Sequence

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
# Chance rates. An accuracy without its baseline is not a result.
# ---------------------------------------------------------------------------

def majority_baseline(labels: Sequence[Hashable], unpredictable: Sequence[Hashable] = ()) -> dict:
    """The score a constant "always answer the commonest class" predictor gets on `labels`.

    This is the number that has to sit next to every accuracy this project prints. 0.42 top-1
    over three animals sounds like something; against a 0.348 majority baseline it is worth
    about seven percentage points, and against a 0.45 one it is worth less than nothing. The
    two rates answer different questions and both are cheap, so both are returned:
      * `rate`    -- majority class share (the baseline a lazy predictor actually achieves);
      * `uniform_rate` -- what a uniform-random guess over the answerable classes achieves.

    `unpredictable` names truth labels no predictor can ever EMIT -- here, the eval harness's
    MISPREDICTED sentinel, which means "the model was wrong and the real class was never
    recorded". Those rows stay in the denominator, because a constant predictor really does get
    them wrong, but they must not be eligible as the majority ANSWER. Without this a slice that
    is mostly rejected predictions reports a 91% "baseline" that nothing could achieve, and every
    real accuracy beside it looks like a failure.

    Empty input returns Nones rather than raising -- a degenerate slice must still print."""
    labs = list(labels)
    n = len(labs)
    if not n:
        return {"rate": None, "label": None, "n": 0, "n_classes": 0, "uniform_rate": None}
    counts = Counter(labs)
    skip = set(unpredictable)
    answerable = {k: v for k, v in counts.items() if k not in skip}
    if not answerable:
        # Every row's truth is unanswerable: no predictor can score above zero here.
        return {"rate": 0.0, "label": None, "n": n, "n_classes": 0, "uniform_rate": 0.0,
                "n_unanswerable": n}
    label, top = max(answerable.items(), key=lambda kv: (kv[1], _sort_key(kv[0])))
    answerable_rows = sum(answerable.values())
    return {"rate": top / n, "label": label, "n": n, "n_classes": len(answerable),
            # P(correct) for a uniform guess = (1/|C|) * P(the truth is answerable at all).
            "uniform_rate": (1.0 / len(answerable)) * (answerable_rows / n),
            "n_unanswerable": n - answerable_rows}


# ---------------------------------------------------------------------------
# THE PROTOCOL: session-blocked leave-one-visit-out identification.
#
# Read this before adding any new identity signal to the project. Every phase of
# docs/identity-eval-2026-08-05.md is judged on the bundle `leave_one_visit_out`
# returns, computed with `session_blocked=True`. A number produced any other way
# is not comparable to the ones in that document and must not be quoted beside them.
# ---------------------------------------------------------------------------

NIGHT_SHIFT_HOURS = 12.0

# The operating-point grid the auto-assign sweep walks. Cosine similarities on this corpus live
# in roughly 0.2-0.95, so 0.30-0.90 in 0.02 steps covers the usable band without pretending to a
# precision 138 probes cannot support.
DEFAULT_THRESHOLDS = tuple(round(0.30 + 0.02 * i, 2) for i in range(31))   # 0.30 .. 0.90
DEFAULT_MARGINS = tuple(round(0.02 * i, 2) for i in range(21))             # 0.00 .. 0.40


class LooUnit(NamedTuple):
    """One scorable unit of the protocol -- in this project, one confirmed solo VISIT.

      key    -- unique, hashable, and what gets handed to the similarity callable (a visit id).
      label  -- the ground-truth identity ("Stan").
      night  -- the session this unit belongs to; see night_key(). None disables blocking for
                this unit (it can never be same-night with anything).
      when   -- an optional datetime, only needed for the `embargo_days` sweep.
    """
    key: Hashable
    label: Hashable
    night: Hashable = None
    when: object = None


def night_key(when, shift_hours: float = NIGHT_SHIFT_HOURS):
    """Which NIGHT a timestamp belongs to: the calendar date of (timestamp - 12 hours).

    A raccoon that arrives at 23:40 and again at 01:10 was here ONCE, on one night, under one
    set of weather / IR / white-balance conditions -- but a plain calendar date splits it across
    two days and lets the 01:10 visit match the 23:40 one as though they were independent
    sessions. That single confound is the difference between this project's headline 0.812 and
    its honest 0.739. Shifting by -12h puts the whole evening under the date it started on.

    Accepts a datetime or an ISO-8601 string (the form the rig stores). Returns an ISO date
    string, or None if the input is unparseable -- a unit with night None is simply never
    blocked, which is the safe (accuracy-optimistic, never accuracy-inventing) failure mode."""
    dt = when
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except (ValueError, TypeError):
            return None
    if not isinstance(dt, datetime):
        return None
    return (dt - timedelta(hours=shift_hours)).date().isoformat()


def _as_unit(u) -> LooUnit:
    """Coerce a LooUnit / dict / 2-4 tuple into a LooUnit, so callers can pass whatever they have."""
    if isinstance(u, LooUnit):
        return u
    if isinstance(u, dict):
        return LooUnit(u["key"], u["label"], u.get("night"), u.get("when"))
    seq = tuple(u)
    if not 2 <= len(seq) <= 4:
        raise ValueError("a unit must be (key, label[, night[, when]])")
    return LooUnit(*(seq + (None,) * (4 - len(seq))))


def leave_one_visit_out(units: Sequence, similarity: Callable[[Hashable, Hashable], float], *,
                        session_blocked: bool = True, embargo_days: float | None = None,
                        allow: Callable[[LooUnit, LooUnit], bool] | None = None,
                        aggregate: str = "max") -> dict:
    """Session-blocked leave-one-visit-out identification. THE protocol; inject your own scorer.

    For every unit in turn, this hides that unit, ranks the remaining identities by how well the
    hidden unit matches them, and asks whether the top-ranked identity is the true one. Three
    exclusions define what "allowed template" means:

      1. a unit is never its own template (the plain leave-one-out guard);
      2. SESSION BLOCKING (`session_blocked`, on by default) -- a unit may not match a template of
         the SAME INDIVIDUAL from the SAME NIGHT. Without this the harness measures "can you
         recognise this animal an hour from now under identical light", which the system already
         does well and nobody needs. With it, the harness measures the thing the product claims:
         recognising the animal on a LATER night. On this corpus the gap is 0.812 -> 0.739;
      3. `embargo_days` -- optionally also require a minimum probe/template time gap (needs
         `when` on the units). This traces the decay curve rather than a single point.

    `allow(probe, template) -> bool` is a further, caller-supplied predicate for exclusions this
    module should not know about (a per-source guard, say). It runs last and only ever removes.

    `similarity(probe_key, template_key) -> float`: the injected scorer, higher = more alike. It
    is called once per allowed (probe, template) pair and nothing else is assumed about it -- so
    a facial-landmark geometry score, a tail-pattern distance, or a whitened cosine all drop in
    here and produce a bundle directly comparable to the appearance baseline. It must NOT read
    the labels; it gets keys only, deliberately.

    `aggregate` folds a label's several templates into one score: "max" (default) is
    nearest-template, which is what production ranking does (individuals.rank_templates) and what
    a bimodal look-mode distribution needs; "mean" averages the pairwise scores.

    Returns a metric bundle:
      protocol              -- how this run was configured (record it beside any number quoted).
      n_probes              -- EVERY unit is a probe. A unit whose identity has no allowed
                               template left is scored and counted WRONG, not dropped: dropping
                               it would silently change the probe set between the blocked and
                               unblocked runs and make the two incomparable, which is precisely
                               the mistake this whole exercise exists to stop.
      top1_accuracy         -- headline, over all n_probes.
      top1_accuracy_scorable-- over the subset whose identity was actually available to be found.
      chance                -- majority_baseline over the probe truths. Always print it.
      per_individual        -- per-identity n / correct / accuracy.
      probes                -- one record per probe: truth, top, s1, runner-up, lead, correct,
                               novel_probe. This is the input the operating-point functions below
                               take, so a sweep never re-derives the ranking.
    """
    if aggregate not in ("max", "mean"):
        raise ValueError("aggregate must be 'max' or 'mean'")
    us = [_as_unit(u) for u in units]
    us.sort(key=lambda u: _sort_key(u.key))          # deterministic probe order -> stable folds
    keys = [u.key for u in us]
    if len(set(keys)) != len(keys):
        raise ValueError("unit keys must be unique")
    embargo_s = None if embargo_days is None else float(embargo_days) * 86400.0

    probes = []
    per_individual: dict = defaultdict(lambda: {"n": 0, "top1_correct": 0})
    for p in us:
        scores: dict = defaultdict(list)
        for t in us:
            if t.key == p.key:
                continue
            if (session_blocked and t.label == p.label
                    and p.night is not None and t.night is not None and t.night == p.night):
                continue
            if embargo_s is not None:
                if p.when is None or t.when is None:
                    continue          # can't prove the gap -> can't allow it
                if abs((p.when - t.when).total_seconds()) < embargo_s:
                    continue
            if allow is not None and not allow(p, t):
                continue
            scores[t.label].append((float(similarity(p.key, t.key)), t.key))
        ranked = []
        for name, pairs in scores.items():
            if aggregate == "max":
                s, via = max(pairs, key=lambda sv: sv[0])
            else:
                s, via = float(np.mean([sv[0] for sv in pairs])), None
            ranked.append((name, s, via))
        ranked.sort(key=lambda r: (-r[1], _sort_key(r[0])))

        truth_available = any(name == p.label for name, _, _ in ranked)
        top_name = ranked[0][0] if ranked else None
        s1 = ranked[0][1] if ranked else 0.0
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        correct = bool(ranked) and top_name == p.label
        per_individual[p.label]["n"] += 1
        per_individual[p.label]["top1_correct"] += int(correct)
        probes.append({
            "key": p.key, "truth": p.label, "night": p.night,
            "top": top_name, "s1": s1, "s2": s2, "lead": s1 - s2,
            "via": ranked[0][2] if ranked else None,
            "correct": correct, "truth_available": truth_available,
            "novel_probe": not truth_available,
            "n_candidate_labels": len(ranked),
        })

    n = len(probes)
    scorable = [p for p in probes if p["truth_available"]]
    n_correct = sum(1 for p in probes if p["correct"])
    for v in per_individual.values():
        v["accuracy"] = (v["top1_correct"] / v["n"]) if v["n"] else None
    return {
        "protocol": {
            "session_blocked": session_blocked,
            "night_shift_hours": NIGHT_SHIFT_HOURS if session_blocked else None,
            "embargo_days": embargo_days,
            "aggregate": aggregate,
            "n_units": len(us),
            "note": ("a probe may not match a template of the same individual from the same night "
                     "(night = timestamp shifted -12h)") if session_blocked else
                    "NO session blocking -- same-night same-individual templates allowed (leaked)",
        },
        "n_probes": n,
        "n_scorable": len(scorable),
        "n_novel_probes": n - len(scorable),
        "top1_correct": n_correct,
        "wrong": n - n_correct,
        "top1_accuracy": (n_correct / n) if n else None,
        "top1_accuracy_scorable": (sum(1 for p in scorable if p["correct"]) / len(scorable))
                                  if scorable else None,
        "chance": majority_baseline([p["truth"] for p in probes]),
        "per_individual": {k: dict(v) for k, v in sorted(per_individual.items(),
                                                         key=lambda kv: _sort_key(kv[0]))},
        "probes": probes,
    }


# ---------------------------------------------------------------------------
# Operating points over a protocol bundle's probes: evaluate, sweep, select,
# and -- the part that was missing -- score a selection rule OUT of sample.
# ---------------------------------------------------------------------------

def evaluate_point(probes: Sequence[dict], threshold: float, margin: float) -> dict:
    """What the auto-assign tier would do at (threshold, margin), and how much of it is wrong.

    Production semantics (individuals.VisitMatcher.auto_assign): write a name iff the top
    identity's similarity >= threshold AND its lead over the runner-up identity >= margin.
    Two error channels, kept separate because they fail differently:
      * `wrong`               -- a known animal named as a different known animal;
      * `novel_false_accepts` -- a probe whose true identity is NOT in the template set at all
                                 (the stand-in for a stranger walking in) still cleared the bars.
    `wrong_name_rate` is wrong/assigned; `error_rate` counts BOTH channels over every name that
    would have been written. When zero errors are observed, `error_rate_upper_bound_95` gives the
    rule-of-three bound (3/n) -- because "0 wrong out of 20" is not evidence of a 0% error rate."""
    known = [p for p in probes if not p["novel_probe"]]
    novel = [p for p in probes if p["novel_probe"]]

    def _passes(p):
        return p["top"] is not None and p["s1"] >= threshold and p["lead"] >= margin

    named = [p for p in known if _passes(p)]
    wrong = [p for p in named if p["top"] != p["truth"]]
    nfa = [p for p in novel if _passes(p)]
    written = len(named) + len(nfa)
    errors = len(wrong) + len(nfa)
    return {
        "threshold": threshold, "margin": margin,
        "n_known_probes": len(known), "n_novel_probes": len(novel),
        "assigned": len(named),
        "coverage": (len(named) / len(known)) if known else None,
        "wrong": len(wrong),
        "wrong_name_rate": (len(wrong) / len(named)) if named else None,
        "novel_false_accepts": len(nfa),
        "names_written": written,
        "errors": errors,
        "error_rate": (errors / written) if written else None,
        "error_rate_upper_bound_95": (3.0 / written) if (written and errors == 0) else None,
        "wrong_visit_keys": [p["key"] for p in wrong],
    }


def sweep_operating_points(probes: Sequence[dict], thresholds=None, margins=None) -> list:
    """Every (threshold, margin) on the grid that makes ZERO errors of either kind while still
    naming something, best coverage first. Ties break toward the SAFER point (higher threshold,
    then higher margin): identical measured behaviour, more headroom against the next raccoon."""
    ths = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    mgs = DEFAULT_MARGINS if margins is None else margins
    out = []
    for t in ths:
        for m in mgs:
            ev = evaluate_point(probes, t, m)
            if ev["errors"] == 0 and ev["assigned"]:
                out.append({"threshold": t, "margin": m, "auto_named": ev["assigned"],
                            "coverage": round(ev["coverage"], 3)})
    out.sort(key=lambda r: (-r["auto_named"], -r["threshold"], -r["margin"]))
    return out


def select_operating_point(probes: Sequence[dict], thresholds=None, margins=None):
    """The selection RULE, isolated so it can be applied to a training fold and nothing else.
    Max-coverage zero-error point, or None when the corpus supports no safe point at all."""
    pts = sweep_operating_points(probes, thresholds, margins)
    return pts[0] if pts else None


def _folds(n: int, k: int, repeats: int, seed: int):
    """Deterministic repeated k-fold index splits: yields (repeat, test_index_set)."""
    rng = np.random.RandomState(seed)
    for rep in range(repeats):
        order = rng.permutation(n)
        for part in np.array_split(order, min(k, n) if n else 1):
            yield rep, {int(i) for i in part}


def kfold_operating_point(probes: Sequence[dict], *, k: int = 5, repeats: int = 5,
                          seed: int = 0, thresholds=None, margins=None) -> dict:
    """Score the SELECTION PROCEDURE out of sample: pick the point on the training folds, then
    apply it to the held-out fold and count what it got wrong there.

    This answers a strictly different question from `evaluate_point` at a named point, and the two
    have been conflated in this project's own write-ups before. Read the two numbers like this:

      * THIS function's `pooled.error_rate` is the error of *the rule* "sweep this corpus and take
        the max-coverage zero-error point". It averages over whatever each fold happens to select,
        most of which are looser than the point a full-corpus sweep lands on. It is the honest
        estimate of how much you should distrust a freshly-swept operating point in general.
      * `evaluate_point(probes, 0.88, 0.02)` is the performance of ONE FIXED point, chosen ahead
        of time. That is the number that describes what shipping 0.88/0.02 actually does.

    Neither is a substitute for the other, and the procedure number must never be quoted as if it
    were the recommended point's error rate."""
    probes = list(probes)
    n = len(probes)
    if n < 2:
        return {"note": "too few probes for k-fold", "n_probes": n}
    pooled = {"n_known_probes": 0, "assigned": 0, "wrong": 0, "novel_false_accepts": 0}
    fold_rows, selected = [], Counter()
    n_no_point = 0
    for rep, test_idx in _folds(n, k, repeats, seed):
        train = [p for i, p in enumerate(probes) if i not in test_idx]
        test = [probes[i] for i in sorted(test_idx)]
        pt = select_operating_point(train, thresholds, margins)
        if pt is None:
            n_no_point += 1
            ev = evaluate_point(test, float("inf"), float("inf"))   # names nothing
        else:
            selected[(pt["threshold"], pt["margin"])] += 1
            ev = evaluate_point(test, pt["threshold"], pt["margin"])
        for key in pooled:
            pooled[key] += ev[key]
        fold_rows.append({"repeat": rep, "selected": None if pt is None
                          else [pt["threshold"], pt["margin"]],
                          "assigned": ev["assigned"], "wrong": ev["wrong"],
                          "novel_false_accepts": ev["novel_false_accepts"],
                          "n_known_probes": ev["n_known_probes"]})
    written = pooled["assigned"] + pooled["novel_false_accepts"]
    errors = pooled["wrong"] + pooled["novel_false_accepts"]
    return {
        "k": k, "repeats": repeats, "seed": seed, "n_probes": n,
        "folds_scored": len(fold_rows), "folds_with_no_safe_point": n_no_point,
        "pooled": {
            **pooled,
            "coverage": (pooled["assigned"] / pooled["n_known_probes"])
                        if pooled["n_known_probes"] else None,
            "wrong_name_rate": (pooled["wrong"] / pooled["assigned"])
                               if pooled["assigned"] else None,
            "names_written": written,
            "errors": errors,
            "error_rate": (errors / written) if written else None,
        },
        "selected_points": [{"threshold": t, "margin": m, "n_folds": c}
                            for (t, m), c in sorted(selected.items(), key=lambda kv: -kv[1])],
        "note": "error of the SELECTION PROCEDURE averaged over folds -- NOT the error of any one "
                "recommended point (see held_out_at_point for that)",
    }


def point_fold_spread(probes: Sequence[dict], threshold: float, margin: float, *,
                      k: int = 5, repeats: int = 5, seed: int = 0) -> dict:
    """Evaluate ONE FIXED point on the same held-out folds, to show its stability.

    A point fixed ahead of time involves no selection, so pooling the folds reproduces the
    whole-corpus number exactly -- the value here is the SPREAD: how far coverage and wrong-name
    count swing between folds. A point whose coverage ranges 8%-32% across folds is one lucky
    fold away from a very different recommendation, and that is worth seeing."""
    probes = list(probes)
    n = len(probes)
    overall = evaluate_point(probes, threshold, margin)
    if n < 2:
        return {"overall": overall, "note": "too few probes to fold"}
    covs, wrongs = [], []
    for _rep, test_idx in _folds(n, k, repeats, seed):
        ev = evaluate_point([probes[i] for i in sorted(test_idx)], threshold, margin)
        if ev["coverage"] is not None:
            covs.append(ev["coverage"])
        wrongs.append(ev["wrong"] + ev["novel_false_accepts"])
    return {
        "overall": overall,
        "folds": {"k": k, "repeats": repeats, "n_folds": len(wrongs),
                  "coverage_min": min(covs) if covs else None,
                  "coverage_mean": float(np.mean(covs)) if covs else None,
                  "coverage_max": max(covs) if covs else None,
                  "errors_max_in_a_fold": max(wrongs) if wrongs else None},
        "note": "the point is FIXED (not selected from this data), so pooled folds reproduce "
                "`overall` exactly; the fold range is a stability read, not a second estimate",
    }


# ---------------------------------------------------------------------------
# Forward-chronological replay: "does identity PROPAGATE forward?"
# ---------------------------------------------------------------------------

def forward_chain_replay(units: Sequence, similarity: Callable[[Hashable, Hashable], float], *,
                         threshold: float, margin: float, horizon_days: float | None = None,
                         recency_days: float | None = None, session_blocked: bool = True,
                         anchor_labels: Sequence | None = None,
                         unlabelled: Sequence | None = None,
                         allow: Callable[[LooUnit, LooUnit], bool] | None = None,
                         aggregate: str = "max") -> dict:
    """Walk the corpus FORWARD IN TIME and ask whether a name, once written, carries to the next
    visit. The shape leave_one_visit_out cannot have.

    LOO is an embargo protocol: it hides one unit and lets every other unit -- past AND future --
    be its template. That is the right question for "how separable are these animals" and the
    WRONG one for "does identity propagate", because propagation is causal. Here a probe sees only
    what existed before it, and an assignment the replay makes can itself become a template. That
    is the only way to measure the thing the chain-of-eras design claims and the only way to
    measure what it risks: a wrong name teaching the matcher its own mistake.

    THE ARMS, which is why `horizon_days` doubles as the switch:
      horizon_days=None  DIRECT. Templates are the human labels of earlier visits, age-blind.
      horizon_days=d     CHAINED. The same age-blind human pool, PLUS anchors -- visits this
                         replay named itself -- within `d` days of the probe. ADDITIVE, never a
                         filter: a pure recency gate was already measured and REJECTED (wrong-name
                         rate 0.119 -> 0.137 while coverage FELL 18.0% -> 16.1%,
                         docs/identity-eval-2026-08-05.md part 5), and `recency_days` exists only
                         so that rejected arm can be re-run as a baseline beside this one.

    `unlabelled` IS WHAT MAKES THE COMPARISON MEAN ANYTHING, and it is not optional dressing. On a
    corpus where every unit carries a human label, an anchor can NEVER reach somewhere a human
    template does not already reach -- the anchor's own visit is a labelled unit sitting in the
    same past, with the same similarity. Chaining is then redundant by construction and can only
    add a WRONG candidate where the human label was right. Pass the keys of visits whose human
    label is to be withheld (real unnamed visits, or a held-out slice of named ones) and the chain
    finally has somewhere to go: those units are still probes, still scored against their true
    label, and may still become anchors.

    AN ANCHOR IS A MACHINE TEMPLATE, and this is where the design's own compensating control gets
    measured rather than asserted: every anchor carries the DEPTH of the chain holding it up and a
    TAINT flag set when any assignment in that chain was wrong. `descendant_contamination` counts
    the assignments standing on tainted ground -- names that look confident and are downstream of
    a mistake. `anchor_labels` scopes which identities may anchor at all (the design says Stan /
    Notch / Pedro; the kits, with zero labels, must never be propagated onto).

    Causality is enforced on the HUMAN pool too: a visit's human label becomes available only from
    its own timestamp onward. Letting a label confirmed next month vouch for a probe today would
    be the same leak session blocking exists to stop, wearing a different hat.

    Returns the leave_one_visit_out bundle shape -- `probes` records carry key/truth/top/s1/s2/
    lead/correct/novel_probe, so evaluate_point and the sweep functions read them unchanged -- plus
    the chain-specific counters. NOTE that unlike LOO, the ranking here DEPENDS on (threshold,
    margin): what gets assigned decides what becomes a template. A sweep must therefore re-run the
    replay per point rather than re-scoring one set of probes.
    """
    if aggregate not in ("max", "mean"):
        raise ValueError("aggregate must be 'max' or 'mean'")
    us = [_as_unit(u) for u in units if _as_unit(u).when is not None]
    us.sort(key=lambda u: (u.when, _sort_key(u.key)))
    if len({u.key for u in us}) != len(us):
        raise ValueError("unit keys must be unique")
    anchor_ok = None if anchor_labels is None else set(anchor_labels)
    hidden = set() if unlabelled is None else set(unlabelled)
    horizon_s = None if horizon_days is None else float(horizon_days) * 86400.0
    recency_s = None if recency_days is None else float(recency_days) * 86400.0

    seen: list = []          # human templates: (unit, ) of visits already in the past
    anchors: list = []       # machine templates: dicts with key/label/when/night/depth/tainted
    probes, chain_rows = [], []
    per_individual: dict = defaultdict(lambda: {"n": 0, "top1_correct": 0})

    for p in us:
        cands: list = []     # (label, score, via_key, depth, tainted)
        for t in seen:
            if t.key in hidden:
                continue                 # nobody named this visit: it is not a human template
            if session_blocked and t.label == p.label and p.night is not None and \
                    t.night is not None and t.night == p.night:
                continue
            if recency_s is not None and (p.when - t.when).total_seconds() > recency_s:
                continue
            if allow is not None and not allow(p, t):
                continue
            cands.append((t.label, float(similarity(p.key, t.key)), t.key, 0, False))
        if horizon_s is not None:
            for a in anchors:
                if (p.when - a["when"]).total_seconds() > horizon_s:
                    continue
                if session_blocked and a["label"] == p.label and p.night is not None and \
                        a["night"] is not None and a["night"] == p.night:
                    continue
                cands.append((a["label"], float(similarity(p.key, a["key"])), a["key"],
                              a["depth"], a["tainted"]))

        by_label: dict = defaultdict(list)
        for lab, s, via, depth, tainted in cands:
            by_label[lab].append((s, via, depth, tainted))
        ranked = []
        for lab, rows in by_label.items():
            if aggregate == "max":
                s, via, depth, tainted = max(rows, key=lambda r: r[0])
            else:
                s = float(np.mean([r[0] for r in rows]))
                via, depth, tainted = rows[0][1], min(r[2] for r in rows), all(r[3] for r in rows)
            ranked.append((lab, s, via, depth, tainted))
        ranked.sort(key=lambda r: (-r[1], _sort_key(r[0])))

        top = ranked[0] if ranked else None
        s1 = top[1] if top else 0.0
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        truth_available = any(lab == p.label for lab, *_ in ranked)
        correct = bool(top) and top[0] == p.label
        assigned = bool(top) and s1 >= threshold and (s1 - s2) >= margin
        per_individual[p.label]["n"] += 1
        per_individual[p.label]["top1_correct"] += int(correct)
        probes.append({
            "key": p.key, "truth": p.label, "night": p.night,
            "top": top[0] if top else None, "s1": s1, "s2": s2, "lead": s1 - s2,
            "via": top[2] if top else None,
            "correct": correct, "truth_available": truth_available,
            "novel_probe": not truth_available,
            "n_candidate_labels": len(ranked),
        })
        chain_rows.append({
            "key": p.key, "assigned": assigned,
            "assigned_name": top[0] if (assigned and top) else None,
            "assigned_wrong": bool(assigned and top and top[0] != p.label),
            "anchor_depth": top[3] if (assigned and top) else None,
            "on_tainted_chain": bool(assigned and top and top[4]),
        })

        # The probe joins the past. Its HUMAN label is a template from now on; if the replay named
        # it and that name is allowed to anchor, it also becomes a machine template -- carrying the
        # depth and the taint of whatever vouched for it.
        seen.append(p)
        if assigned and horizon_s is not None and (anchor_ok is None or top[0] in anchor_ok):
            anchors.append({"key": p.key, "label": top[0], "when": p.when, "night": p.night,
                            "depth": top[3] + 1, "tainted": bool(top[4] or top[0] != p.label)})

    n = len(probes)
    named = [c for c in chain_rows if c["assigned"]]
    wrong = [c for c in named if c["assigned_wrong"]]
    tainted = [c for c in named if c["on_tainted_chain"]]
    depths = [c["anchor_depth"] for c in named if c["anchor_depth"] is not None]
    scorable = [p for p in probes if p["truth_available"]]
    n_correct = sum(1 for p in probes if p["correct"])
    for v in per_individual.values():
        v["accuracy"] = (v["top1_correct"] / v["n"]) if v["n"] else None
    return {
        "protocol": {
            "arm": "direct" if horizon_s is None else f"chained@{horizon_days:g}d",
            "threshold": threshold, "margin": margin,
            "horizon_days": horizon_days, "recency_days": recency_days,
            "session_blocked": session_blocked, "aggregate": aggregate,
            "anchor_labels": None if anchor_ok is None else sorted(anchor_ok, key=_sort_key),
            "n_unlabelled": len(hidden & {u.key for u in us}),
            "n_units": len(us),
            "note": "forward-chronological: a probe sees only what existed before it",
        },
        "n_probes": n,
        "assigned": len(named),
        "wrong": len(wrong),
        "coverage": (len(named) / n) if n else None,
        "wrong_name_rate": (len(wrong) / len(named)) if named else None,
        "n_anchors": len([c for c in named if c["assigned_name"] is not None])
                     if horizon_s is not None else 0,
        "anchor_depth_max": max(depths) if depths else 0,
        "anchor_depth_mean": (sum(depths) / len(depths)) if depths else 0.0,
        "descendant_contamination": len(tainted),
        "top1_correct": n_correct,
        "top1_accuracy": (n_correct / n) if n else None,
        "top1_accuracy_scorable": (sum(1 for p in scorable if p["correct"]) / len(scorable))
                                  if scorable else None,
        "chance": majority_baseline([p["truth"] for p in probes]),
        "per_individual": {k: dict(v) for k, v in sorted(per_individual.items(),
                                                         key=lambda kv: _sort_key(kv[0]))},
        "probes": probes,
        "chain": chain_rows,
    }


def adjacency_propagation(units: Sequence, *, horizon_days: float = 7.0) -> dict:
    """The OTHER documented rejection, as a baseline: name each visit after the most recent visit
    within the horizon, with no appearance test at all. "The same animal usually comes back
    tomorrow" is a real prior in this yard, and any chained arm that cannot beat it is measuring
    the calendar rather than the animal."""
    us = [_as_unit(u) for u in units if _as_unit(u).when is not None]
    us.sort(key=lambda u: (u.when, _sort_key(u.key)))
    horizon_s = float(horizon_days) * 86400.0
    named = wrong = 0
    prev = None
    for p in us:
        if prev is not None and (p.when - prev.when).total_seconds() <= horizon_s:
            named += 1
            if prev.label != p.label:
                wrong += 1
        prev = p
    return {"arm": f"adjacency@{horizon_days:g}d", "n_probes": len(us), "assigned": named,
            "wrong": wrong, "coverage": (named / len(us)) if us else None,
            "wrong_name_rate": (wrong / named) if named else None}


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
