"""
eval.py -- the local, READ-ONLY machine-learning evaluation harness.

The rig's trust rules have always been folklore: "species_confidence >= 0.8 you can trust,
< 0.5 look closer", "raccoons are the best case, crows the hardest", "single crops don't match
across sessions but VISIT PROTOTYPES do (same animal 0.83-0.93 cosine, different animals
0.36-0.42)", "IoU < 0.45 means two separate bodies". Every one of those was eyeballed on a
handful of early crops and never measured again. This harness turns them into TRACKED,
REPRODUCIBLE numbers computed from the real database -- so a claim in the README can be checked,
and a future change can be regression-tested against a saved baseline.

It runs two evaluations, both pure offline numpy (no cloud, no LLM, no GPU, no model load):

  1. SPECIES CLASSIFIER eval -- against the human verdicts (detections.species_verified), how
     good is BioCLIP's label + confidence? Per-species precision/recall/F1, a confusion matrix,
     and a calibration curve that directly tests the 0.8/0.5 confidence folklore. Stratified by
     predicted species, by day/night, and by source (glass door vs trail cam).

     CORRECTION HANDLING (a limitation this harness surfaced, since fixed at the source): a human
     CORRECTION overwrites detections.species and forces confidence=1.0. db.correct_species /
     apply_visit_label now PRESERVE the model's prediction into detections.model_species first, so a
     corrected crop stays gradable -- the model's call and the human's true label are both known
     (category 'corrected_recovered'), which is what finally lets recall for a corrected-away class
     like crow be measured. The classifier eval scores confirmed + rejected + recovered rows; only
     LEGACY corrections made before the preservation fix have an unrecoverable prediction (counted as
     prediction_overwritten_by_correction). That original gap -- and how much of the review corpus it
     stranded -- is itself a finding the report still surfaces.

  2. RE-ID eval -- reproduce the cross-session individual-ID result as a MEASURED number. Using
     the exact visit-prototype logic the live loop uses (individuals.VisitMatcher / prototype /
     confirmed_visit_labels), compute the cosine separation between same-individual visit pairs
     and different-individual visit pairs, then ROC + AUC + the best operating threshold, held up
     against the live config's reid_novel_threshold. Correctness guarantees the analysis flagged:
       * SOLO templates only -- multi-animal visits (VisitMatcher.is_multi) blend two animals and
         are excluded.
       * No leakage -- the pairwise separation compares two DISTINCT visits (neither prototype
         contains the other's crops), and the leave-one-visit-out identification test builds each
         individual's template from that individual's OTHER visits, never the one being scored.
     Plus a compact check of the IoU < 0.45 co-presence cut against the independent clip signal.

     SESSION BLOCKING (2026-08-05, phase A3 of docs/identity-eval-2026-08-05.md). The guarantee
     above -- "never the visit being scored" -- turned out not to be enough. A raccoon's 23:40
     visit and its 01:10 visit are the SAME night, the same light, the same auto-white-balance
     decision and the same wet fur; letting one be the other's template measures "can you
     recognise this animal an hour from now", not "can you recognise it next week", which is the
     only question the product actually asks. Every re-ID number here is therefore computed BOTH
     ways and printed side by side: session-BLOCKED (a probe may not match a template of the same
     individual from the same night; night key = timestamp shifted -12h, so 01:00 belongs to the
     previous evening) and UNBLOCKED (the old, leaked number). The gap between them is the
     cleanest measure this project has of how much of its "identity" signal is really session
     signal -- treat the blocked column as the real one and the gap as the diagnostic.

     Likewise the auto-assign operating point is no longer selected and scored on the same
     probes. Repeated k-fold picks the point on the TRAINING folds and scores it on the HELD-OUT
     fold, and the harness prints three distinct numbers that have been conflated before: the
     in-sample recommendation, the error of the SELECTION PROCEDURE across folds, and the direct
     performance of individual FIXED points (the configured one, the recommended one).

     Every accuracy line carries its chance / majority-class baseline. A 0.42 that beats 0.35
     chance and a 0.42 that loses to 0.45 chance are not the same result and must not print the
     same way.

  READ-ONLY, always. The live capture rig may be running right now and holds the WAL write lock;
  opening the DB read-write has crashed the capture thread before ("database is locked"). This
  harness opens it with sqlite3's file:...?mode=ro URI (db.connect_readonly) AND sets
  PRAGMA query_only, so it can never contend for the write lock. It writes its results ONLY to
  reports/ (a timestamped JSON artifact + a console summary), never back into backyard.db.

  It MEASURES and RECOMMENDS -- it never edits config.py. Threshold lines read
  "current X; data suggests Y"; applying that is a human decision.

    python eval.py                       # both evals on the live DB, console summary + JSON artifact
    python eval.py --species             # species classifier eval only
    python eval.py --reid                 # re-ID eval only
    python eval.py --reid-species raccoon # which species the re-ID eval runs on (default raccoon)
    python eval.py --json                 # print the full machine-readable result to stdout too
    python eval.py --no-save              # don't write the reports/ artifact (console only)
    python eval.py --baseline latest      # diff against the newest reports/ artifact; EXIT 1 on a
                                          #   regression past --tolerance (the regression gate)
    python eval.py --baseline reports/eval_20260718T030508Z.json --tolerance 0.02
    python eval.py --at-point 0.88,0.02   # also score this fixed auto-assign point (repeatable)
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
import db
import evalmetrics as em
# NOTE: production's ranker (individuals.rank_templates) is NOT imported. The protocol's "max"
# aggregation reproduces its semantics exactly (nearest template per individual, best first) but
# takes a similarity CALLABLE instead of vectors -- that seam is what lets a future trait scorer be
# measured on the identical protocol. If rank_templates ever changes, change em's aggregation too.
from individuals import EMBED_MODEL, VisitMatcher, iou

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"

# Public re-exports of the identification PROTOCOL (it lives in evalmetrics so it stays pure and
# DB-free). Re-exported here because `from eval import leave_one_visit_out` is the import a future
# trait scorer will reach for first, and there must be exactly one protocol either way.
LooUnit = em.LooUnit
night_key = em.night_key
leave_one_visit_out = em.leave_one_visit_out

# Sentinel "true label" for a model prediction the human REJECTED (species_verified = 0): the
# model was wrong, but the corrected true species was overwritten, so the real class is unknown.
# It rides in the confusion matrix as a true-label row so those rows still count against precision.
MISPREDICTED = "(mispredicted - true label not recorded)"

# Truth labels no classifier can ever OUTPUT, so they can never be a chance-rate baseline's
# answer. The sentinel above is the only one: a row whose truth is "the model was wrong and the
# real class was lost" is unwinnable for every predictor, constant or otherwise. It still counts
# in the denominator (a lazy predictor really does get it wrong) -- see em.majority_baseline.
_UNPREDICTABLE = (MISPREDICTED,)

# Non-critter / decoy labels that are model *outputs* but not real species -- the same denylist the
# dashboard and digest hide (kept in sync with stats._NON_CRITTER). Reported separately so a "not an
# animal" prediction doesn't masquerade as a species row, but still counted for calibration.
try:
    from stats import _NON_CRITTER as _NON_CRITTER
except Exception:      # stats imports are heavier; fall back to the essentials if it can't load
    _NON_CRITTER = {"not an animal", "person", "vehicle", "blur", "food", "cat food"}


# ---------------------------------------------------------------------------
# Read-only DB access (WAL-safe: never contends for the live rig's write lock).
# ---------------------------------------------------------------------------

def open_readonly(db_path):
    """Open backyard.db strictly read-only. db.connect_readonly uses the file:...?mode=ro URI;
    we additionally set PRAGMA query_only as belt-and-suspenders so even a stray UPDATE in this
    process would raise rather than touch the live database. Returns a Row-factory connection, or
    None if the DB doesn't exist."""
    conn = db.connect_readonly(db_path)
    if conn is None:
        return None
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = ON;")
    except sqlite3.OperationalError:
        pass  # mode=ro already blocks writes; this is only extra insurance
    return conn


def _table_columns(conn, table: str) -> set:
    """Column names of `table` -- lets the harness stay robust to an un-migrated DB (one opened
    read-only before the model_species columns were added), degrading to legacy behaviour instead
    of raising 'no such column'."""
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def git_commit() -> str | None:
    """Best-effort short git hash of the working tree, so each artifact is pinned to the code that
    produced it (a run is only comparable to another if you know what changed between them)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Day / night: sun-driven when the yard's lat/lon are configured, else a fixed
# hour cut. Time-of-day is a required stratification for the species eval.
# ---------------------------------------------------------------------------

def _daynight_fn(cfg):
    """Return (fn(datetime) -> 'day'|'night', method_str). Prefer the project's astral-based
    civil-dawn/dusk logic when lat/lon are set AND astral is importable; otherwise fall back to a
    plain hour cut (night = 20:00-06:00, the crepuscular-mammal convention this yard sees), which
    the task explicitly sanctions. The method string is recorded in the artifact so a reader knows
    which rule produced the split."""
    lat, lon = getattr(cfg, "latitude", None), getattr(cfg, "longitude", None)
    if lat is not None and lon is not None:
        try:
            import astral  # noqa: F401 -- presence check only
            from daynight import current_period

            def fn(dt):
                return current_period(lat, lon, dt)
            return fn, f"astral civil dawn/dusk @ ({lat:.3f},{lon:.3f})"
        except Exception:
            pass

    def fn(dt):
        return "night" if (dt.hour < 6 or dt.hour >= 20) else "day"
    return fn, "hour cut (night = 20:00-06:00 local)"


# ===========================================================================
# EVAL 1 -- Species classifier vs the human verdicts.
# ===========================================================================

def _fetch_verified_rows(conn):
    """Every human-reviewed detection (species_verified IS NOT NULL), each resolved to what the
    MODEL predicted vs the true label so even corrected rows can be graded. One of four categories:

      * 'confirmed'  -- species_source != 'human', verified=1: the model's label survived and was
                        right (true label = that species).
      * 'rejected'   -- species_source != 'human', verified=0: the model's label survived and was
                        wrong; the real class is unknown, so the true label is the MISPREDICTED
                        sentinel.
      * 'corrected_recovered' -- species_source == 'human' but model_species was preserved (a
                        post-fix correction): the model's original prediction AND the human's true
                        label are both known -> a real confusion-matrix row (this is what makes a
                        corrected-away class like crow measurable).
      * 'corrected_lost' -- species_source == 'human' with no preserved model_species (a legacy
                        correction made before the fix): ungradable, counted only.

    Each dict carries resolved pred / pred_conf / true / correct / gradable / category, plus the raw
    source & timestamp the stratification needs. Robust to an un-migrated DB (no model_species
    columns): those rows fall back to the intact-only behaviour."""
    has_model = "model_species" in _table_columns(conn, "detections")
    extra = ", model_species, model_species_confidence" if has_model else ""
    rows = conn.execute(
        f"""SELECT id, source, timestamp, species, species_confidence, species_verified,
                   species_source{extra}
            FROM detections WHERE species_verified IS NOT NULL"""
    ).fetchall()
    out = []
    for r in rows:
        stage = r["species_source"]
        mpred = r["model_species"] if has_model else None
        mconf = r["model_species_confidence"] if has_model else None
        if stage != "human":
            # Prediction intact: the live species IS the model's own call.
            pred, pred_conf = r["species"], r["species_confidence"]
            true = r["species"] if r["species_verified"] == 1 else MISPREDICTED
            correct = 1 if r["species_verified"] == 1 else 0
            gradable = pred is not None and pred_conf is not None
            category = "confirmed" if r["species_verified"] == 1 else "rejected"
        elif mpred is not None:
            # Human correction with the model's prediction preserved -> fully gradable, and the true
            # label is the human's answer (a REAL class, not the sentinel).
            pred, pred_conf = mpred, mconf
            true = r["species"]
            correct = 1 if mpred == r["species"] else 0
            gradable = pred_conf is not None
            category = "corrected_recovered"
        else:
            # Legacy correction: the prediction was overwritten in place and is unrecoverable.
            pred = pred_conf = true = correct = None
            gradable = False
            category = "corrected_lost"
        out.append({
            "id": r["id"], "source": r["source"], "timestamp": r["timestamp"], "stage": stage,
            "pred": pred, "pred_conf": pred_conf, "true": true, "correct": correct,
            "gradable": gradable, "category": category, "verified": r["species_verified"],
        })
    return out


def eval_species(conn, cfg) -> dict:
    """Score the species classifier against the human verdicts. See module docstring for the
    prediction-overwrite limitation this has to work around."""
    dn_fn, dn_method = _daynight_fn(cfg)
    all_rows = _fetch_verified_rows(conn)

    graded = [r for r in all_rows if r["gradable"]]
    by_cat = Counter(r["category"] for r in all_rows)

    # Each gradable row already carries its resolved (pred, true, correct): confirmed -> pred==true;
    # rejected -> MISPREDICTED sentinel truth (model wrong, real class unknown); corrected_recovered
    # -> the model's prediction vs the human's real corrected class.
    y_pred, y_true, conf, correct = [], [], [], []
    for r in graded:
        if r["pred"] is None or r["pred_conf"] is None:
            continue
        y_pred.append(r["pred"])
        y_true.append(r["true"])
        conf.append(float(r["pred_conf"]))
        correct.append(int(r["correct"]))

    # Labels span both what the model PREDICTED and the real classes recovered from corrections -- a
    # crow the model called that a human relabelled 'raccoon' contributes a raccoon TRUE row the
    # model never predicted, and only a union label set makes that missed class's recall measurable.
    pred_species = sorted(set(y_pred), key=em._sort_key)
    true_species = sorted({t for t in y_true if t != MISPREDICTED}, key=em._sort_key)
    class_labels = sorted(set(pred_species) | set(true_species), key=em._sort_key)
    labels = class_labels + [MISPREDICTED]
    prf = em.precision_recall_f1(y_true, y_pred, labels=class_labels)
    cm = em.confusion_matrix(y_true, y_pred, labels=labels)
    calib = em.reliability_curve(conf, correct, n_bins=10)

    # A macro average over EVERY predicted label is misleading here: a species the human always
    # REJECTED (e.g. a couple of wrong 'European starling' calls) has real ground truth of zero, so
    # it enters the mean as a 0.0 that swamps the classes that actually matter (raccoon, opossum).
    # Report a second macro restricted to labels the truth actually contains -- the honest per-class
    # read for a corpus this skewed. `prf['macro']` (all labels) stays in the artifact for parity.
    present = [l for l in class_labels if prf["per_label"][l]["support"] > 0]
    prf["macro_present_classes"] = {
        k: (float(np.mean([prf["per_label"][l][k] for l in present])) if present else 0.0)
        for k in ("precision", "recall", "f1")}
    prf["present_classes"] = present

    # ---- Stratification (over the gradable rows: intact + recovered corrections) ----
    # Every slice carries its OWN chance rate. A slice's majority baseline moves with its class
    # mix -- night is nearly all raccoon, day is not -- so one global baseline would flatter the
    # easy slice and libel the hard one. `chance.rate` is what "always guess the commonest true
    # label" scores on exactly these rows; compare each accuracy to the number beside it.
    def _reliability_for(subset):
        c = [x[0] for x in subset]
        k = [x[1] for x in subset]
        rc = em.reliability_curve(c, k, n_bins=10) if subset else None
        acc = float(np.mean(k)) if subset else None
        return {"n": len(subset), "accuracy": acc, "chance": em.majority_baseline(
                    [x[2]["true"] for x in subset], _UNPREDICTABLE),
                "ece": rc["ece"] if rc else None, "reliability": rc}

    triples = [(float(r["pred_conf"]), int(r["correct"]), r)
               for r in graded if r["pred_conf"] is not None]

    by_species = {}
    for sp in pred_species:
        sub = [t for t in triples if t[2]["pred"] == sp]
        by_species[sp] = {"n": len(sub),
                          "accuracy": float(np.mean([k for _, k, _ in sub])) if sub else None,
                          "chance": em.majority_baseline([t[2]["true"] for t in sub], _UNPREDICTABLE),
                          "non_critter": sp.lower() in _NON_CRITTER}

    by_daynight = {}
    for period in ("day", "night"):
        sub = [t for t in triples
               if (p := db.parse_local(t[2]["timestamp"])) is not None and dn_fn(p) == period]
        by_daynight[period] = _reliability_for(sub)

    by_source = {}
    for src in sorted({r["source"] for r in graded}):
        sub = [t for t in triples if t[2]["source"] == src]
        by_source[src] = _reliability_for(sub)

    # ---- Folklore verdicts, in plain language ----
    # "0.8 trust / 0.5 look closer": accuracy of predictions AT/ABOVE 0.8 vs BELOW 0.5.
    hi = [t for t in triples if t[0] >= 0.8]
    lo = [t for t in triples if t[0] < 0.5]
    mid = [t for t in triples if 0.5 <= t[0] < 0.8]

    def _bucket(rows):
        return {"n": len(rows),
                "accuracy": float(np.mean([k for _, k, _ in rows])) if rows else None,
                "chance": em.majority_baseline([t[2]["true"] for t in rows], _UNPREDICTABLE)}

    trust_rule = {"conf_ge_0.8": _bucket(hi), "conf_0.5_to_0.8": _bucket(mid),
                  "conf_lt_0.5": _bucket(lo)}

    return {
        # The one baseline every headline species number is read against: what "always answer the
        # commonest true label" scores over all graded rows.
        "chance": em.majority_baseline(y_true, _UNPREDICTABLE),
        "ground_truth": {
            "verified_rows_total": len(all_rows),
            "prediction_intact": by_cat["confirmed"] + by_cat["rejected"],  # model's own label survived
            "correction_recovered": by_cat["corrected_recovered"],  # gradable again (the model_species fix)
            "prediction_overwritten_by_correction": by_cat["corrected_lost"],  # legacy, unrecoverable
            "confirmed_correct": by_cat["confirmed"],
            "rejected_wrong": by_cat["rejected"],
            "note": ("A human correction overwrites detections.species and forces confidence=1.0. "
                     "Since the model_species fix, correct_species/apply_visit_label PRESERVE the "
                     "model's prediction first, so corrected crops grade again (correction_recovered) "
                     "against the human's real label. Only legacy corrections predating the fix "
                     "(prediction_overwritten_by_correction) stay unrecoverable."),
        },
        "daynight_method": dn_method,
        "graded_rows": len(y_pred),
        "precision_recall_f1": prf,
        "confusion_matrix": cm,
        "calibration": calib,
        "trust_rule_check": trust_rule,
        "by_predicted_species": by_species,
        "by_daynight": by_daynight,
        "by_source": by_source,
    }


# ===========================================================================
# EVAL 2 -- Re-ID: same- vs different-individual visit-prototype separation.
# ===========================================================================

def individual_centroids(protos: dict, labels: dict, exclude_visit=None,
                         exclude_visits=None) -> dict:
    """{name: L2-normalized mean of that individual's visit prototypes}, optionally LEAVING OUT one
    visit. This is the leave-one-visit-out contamination guard made explicit and testable: when you
    score whether visit X belongs to individual Y, X's own prototype must never be inside Y's
    template, or the match is measuring X against itself. Pass exclude_visit=X to drop it.

    `exclude_visits` drops a whole SET, which is how session blocking is applied here: the probe
    plus every same-night visit of the probe's own individual (see em.night_key). Both arguments
    are unioned, so the single-visit form keeps working unchanged.

    `protos` = {visit_id: prototype_vector}; `labels` = {visit_id: individual_name}. Only visits
    present in both are used. A name left with no visits after the exclusion simply doesn't appear
    in the result (it can't be a template for this comparison)."""
    drop = set() if exclude_visits is None else set(exclude_visits)
    if exclude_visit is not None:
        drop.add(exclude_visit)
    groups: dict = defaultdict(list)
    for vid, name in labels.items():
        if vid in drop or vid not in protos:
            continue
        groups[name].append(protos[vid])
    out = {}
    for name, vecs in groups.items():
        c = np.mean(np.stack(vecs), axis=0)
        n = np.linalg.norm(c)
        out[name] = c / n if n else c
    return out


def _pair_separation(protos: dict, labels: dict, nights: dict | None = None,
                     session_blocked: bool = False) -> dict:
    """Pairwise same- vs different-individual visit-prototype cosine separation + ROC/AUC/threshold.
    Every pair is two DISTINCT visits, so neither prototype contains the other's crops -- the
    separation is leakage-free in the LOO sense by construction.

    It is NOT leakage-free in the session sense, which is the leak that matters here: a
    same-individual pair drawn from one night is two looks at one session, and counting it as a
    positive inflates AUC with signal the product can never use. With `session_blocked=True` (and
    a {visit_id: night_key} map) those pairs are dropped from the POSITIVE class. Same-night pairs
    of DIFFERENT individuals are kept deliberately -- they are legitimate, and unusually hard,
    negatives; dropping them too would hand back the inflation from the other side."""
    vids = sorted(protos)
    same, diff = [], []
    dropped_same_night = 0
    for a, b in itertools.combinations(vids, 2):
        s = float(protos[a] @ protos[b])
        if labels[a] == labels[b]:
            if (session_blocked and nights
                    and nights.get(a) is not None and nights.get(a) == nights.get(b)):
                dropped_same_night += 1
                continue
            same.append(s)
        else:
            diff.append(s)
    scores = np.array(same + diff, dtype=float)
    y = np.array([1] * len(same) + [0] * len(diff))

    def _dist(arr):
        # Always the same key set, even when empty -- session blocking CAN empty the positive
        # class (a corpus where every same-individual pair is same-night has no cross-session
        # evidence at all, which is a finding, not a crash), and a reader that KeyErrors on that
        # is a reader that hides it.
        if not arr:
            return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None,
                    "min": None, "max": None}
        a = np.asarray(arr)
        return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
                "min": float(a.min()), "max": float(a.max())}

    roc = em.roc_curve(scores, y) if len(same) and len(diff) else {"auc": None}
    return {
        "session_blocked": bool(session_blocked and nights),
        "same_night_positive_pairs_dropped": dropped_same_night,
        "chance_auc": 0.5,
        "same_individual_pairs": _dist(same),
        "different_individual_pairs": _dist(diff),
        "auc": roc.get("auc"),
        "best_threshold": em.best_threshold(scores, y),
        # Down-sample the ROC curve into the artifact (full sweep can be thousands of points).
        "roc": {"auc": roc.get("auc"),
                "tpr": roc.get("tpr", [])[::max(1, len(roc.get("tpr", [1])) // 40)],
                "fpr": roc.get("fpr", [])[::max(1, len(roc.get("fpr", [1])) // 40)]},
    }


def loo_units(protos: dict, labels: dict, started: dict | None = None) -> list:
    """Build the protocol's unit list from this project's shapes: {visit_id: prototype},
    {visit_id: name}, {visit_id: iso timestamp}. Each unit carries its NIGHT key (em.night_key,
    -12h shift) so session blocking has something to block on, and its parsed datetime so the
    embargo curve has something to measure. Only visits present in both protos and labels are
    units -- a labelled visit with no prototype cannot be scored and must not silently become a
    zero row."""
    out = []
    for vid, name in labels.items():
        if vid not in protos:
            continue
        ts = (started or {}).get(vid)
        when = db.parse_local(ts) if ts else None
        out.append(em.LooUnit(key=vid, label=name, night=em.night_key(when), when=when))
    return out


def appearance_similarity(protos: dict):
    """The similarity callable the protocol runs on TODAY: cosine between two visit prototypes.

    This is deliberately a one-liner behind a named factory, because it is the seam. A trait
    scorer (facial-landmark geometry, tail ring pattern, within-crop fur-luminance ratios) is
    dropped in by writing another function of the same shape -- (visit_id, visit_id) -> float --
    and handing it to em.leave_one_visit_out with everything else held fixed. That is the only
    way its number is comparable to the 0.739 appearance baseline in
    docs/identity-eval-2026-08-05.md rather than a new number measured a new way."""
    def sim(a, b):
        return float(protos[a] @ protos[b])
    return sim


def _identify_loo(protos: dict, labels: dict, threshold: float, nights: dict | None = None,
                  session_blocked: bool = False) -> dict:
    """Leave-one-visit-out cross-session IDENTIFICATION against CENTROID templates: for each solo
    confirmed visit, rebuild the cast's centroid templates from every OTHER solo visit
    (individual_centroids), then see whether the visit's nearest template is its TRUE individual.
    This is the direct exercise of the live novelty threshold: at `threshold`, is a correct top
    match ACCEPTED (>= threshold) or wrongly flagged 'possibly someone new'?

    With `session_blocked=True` the excluded set grows from {this visit} to {this visit} + {every
    visit of the SAME individual on the SAME night}: the same-night twin is the same session, and
    letting it into the template is the leak this harness exists to stop.

    EVERY visit is a probe, including one whose individual has no template left after the
    exclusion -- it is counted WRONG rather than dropped, so the blocked and unblocked runs score
    the IDENTICAL probe set and their difference is the blocking effect and nothing else.
    `top1_accuracy_scorable` reports the other view (over probes whose identity was findable at
    all) for readers who want it.

    Note this is the CENTROID variant (mean prototype per individual). Production ranks by
    NEAREST VISIT (individuals.rank_templates); that variant is measured by em.leave_one_visit_out
    in eval_reid. Both are reported because they disagree by a probe or two and neither is
    obviously right on a corpus with bimodal look-modes."""
    n_probes = 0
    scorable = 0
    top1_correct = 0
    scorable_correct = 0
    accepted_correct = 0
    flagged_novel_correct = 0  # correct individual, but similarity fell below threshold
    wrong = 0
    per_individual = defaultdict(lambda: {"n": 0, "top1_correct": 0})
    for vid, truth in labels.items():
        if vid not in protos:
            continue
        drop = {vid}
        if session_blocked and nights and nights.get(vid) is not None:
            drop |= {v for v, nm in labels.items()
                     if nm == truth and nights.get(v) == nights.get(vid)}
        cents = individual_centroids(protos, labels, exclude_visits=drop)
        n_probes += 1
        per_individual[truth]["n"] += 1
        available = truth in cents
        scorable += int(available)
        ranked = sorted(((n, float(protos[vid] @ c)) for n, c in cents.items()),
                        key=lambda r: -r[1])
        top_name, top_sim = ranked[0] if ranked else (None, 0.0)
        if top_name == truth:
            top1_correct += 1
            scorable_correct += 1
            per_individual[truth]["top1_correct"] += 1
            if top_sim >= threshold:
                accepted_correct += 1
            else:
                flagged_novel_correct += 1
        else:
            wrong += 1
    truths = [labels[v] for v in labels if v in protos]
    return {
        "session_blocked": bool(session_blocked and nights),
        "n_probes": n_probes,
        "coverage": {"scorable_visits": scorable,
                     "note": "a visit is scorable only if its individual still has an ALLOWED "
                             "visit (not itself, and under blocking not a same-night twin) to "
                             "build a leave-one-out template from; unscorable probes are still "
                             "scored, and counted wrong"},
        "top1_accuracy": (top1_correct / n_probes) if n_probes else None,
        "top1_accuracy_scorable": (scorable_correct / scorable) if scorable else None,
        "chance": em.majority_baseline(truths),
        "at_threshold": {
            "threshold": threshold,
            "accepted_correct": accepted_correct,
            "correct_but_flagged_novel": flagged_novel_correct,
            "wrong_individual": wrong,
        },
        "per_individual": {k: dict(v) for k, v in sorted(per_individual.items())},
    }


def _blocked_unblocked_loo(units, similarity, **kw) -> dict:
    """Run the protocol both ways over the identical probe set and report the GAP.

    The gap is the headline diagnostic of phase A3: it is how much of the measured "identity"
    was really "same night". Everything the project steers by should be read off the `blocked`
    side; `unblocked` is kept only so the size of the old lie stays visible."""
    blocked = em.leave_one_visit_out(units, similarity, session_blocked=True, **kw)
    unblocked = em.leave_one_visit_out(units, similarity, session_blocked=False, **kw)
    b, u = blocked["top1_accuracy"], unblocked["top1_accuracy"]
    return {
        "blocked": {k: v for k, v in blocked.items() if k != "probes"},
        "unblocked": {k: v for k, v in unblocked.items() if k != "probes"},
        "session_leak_gap": (u - b) if (b is not None and u is not None) else None,
        "note": "blocked = a probe may not match a same-night template of the same individual "
                "(night = timestamp shifted -12h). The gap IS the session leak.",
        "_probes": blocked["probes"],          # in-memory only; stripped from the artifact
        "_probes_unblocked": unblocked["probes"],
    }


def _embargo_curve(units, similarity, days=(None, 1, 3, 7, 14, 21)) -> list:
    """Top-1 as a function of a minimum enforced probe/template time gap -- the decay curve that
    says how long an appearance template stays worth anything. Session-blocked throughout (an
    embargo of 0 days is not the same thing as blocking the session, and conflating them is how
    the 0.818 headline survived so long). Each row carries its own chance rate, because the probe
    set narrows as the embargo grows and a shrinking probe set moves the baseline too."""
    out = []
    for d in days:
        r = em.leave_one_visit_out(units, similarity, session_blocked=True, embargo_days=d)
        out.append({"embargo_days": d, "n_probes": r["n_probes"],
                    "n_scorable": r["n_scorable"], "top1_accuracy": r["top1_accuracy"],
                    "chance": r["chance"]["rate"]})
    return out


def _auto_assign_sweep(probes: list, *, k: int = 5, repeats: int = 5, seed: int = 0,
                       at_points: list | None = None) -> dict:
    """Recommend an operating point for the nightly AUTO-ASSIGN pass -- and, unlike the version
    this replaces, say honestly how much that recommendation is worth.

    Production semantics (individuals.VisitMatcher.auto_assign): a visit is auto-named iff top
    similarity >= threshold AND (top - runner-up individual) >= margin. Two error channels, both
    required to be zero at the recommended point:
      - wrong:            a known individual's visit auto-named as someone else;
      - novel_false_accept: a probe whose TRUE individual has no allowed template (the stand-in
        for a genuinely NEW animal) still passing the bars -- "a stranger walks in and gets Stan's
        name", measurable only this way.

    THREE NUMBERS, deliberately kept apart because this project has already conflated two of them
    once in print:
      1. `in_sample.recommended` -- the point a sweep over ALL probes picks. It is selected and
         scored on the same data, so its zero-error property is a property of the fit, not a
         prediction. This is what you copy into config_local.py, knowing that.
      2. `held_out_procedure` -- repeated k-fold: select on the training folds, score the held-out
         fold. This is the error of THE SWEEP ITSELF, averaged over whatever each fold picks
         (mostly looser points than the full-corpus winner). It is NOT the recommended point's
         error rate. An earlier draft of docs/identity-eval-2026-08-05.md attached this number to
         the recommendation and had to be corrected.
      3. `at_points` -- direct evaluation of specific FIXED points (the configured one, the
         recommendation, anything passed with --at-point). No selection is involved, so this is
         what shipping that point actually does, with a rule-of-three upper bound when it makes
         no errors -- 0 wrong out of 20 is not a 0% error rate, it is "<=15%, probably".

    CAVEAT all three inherit: the confirmed corpus leans toward visits a human already agreed
    matched, so coverage is optimistic for the wild population."""
    known = [p for p in probes if not p["novel_probe"]]
    if not known:
        return {"note": "no scorable probes (need >= 2 confirmed solo visits)", "n_probes": 0}
    zero_error = em.sweep_operating_points(probes)
    rec = zero_error[0] if zero_error else None

    points = []
    seen = set()
    for (t, m, why) in ([(rec["threshold"], rec["margin"], "in-sample recommendation")] if rec else []) \
            + [(t, m, why) for t, m, why in (at_points or [])]:
        if (t, m) in seen:
            continue
        seen.add((t, m))
        row = em.point_fold_spread(probes, t, m, k=k, repeats=repeats, seed=seed)
        row["why"] = why
        points.append(row)

    return {
        "n_probes": len(known),
        "n_novel_probes": len(probes) - len(known),
        "in_sample": {
            "recommended": rec,
            "zero_error_points": zero_error[:40],
            "note": "max coverage with 0 wrong names AND 0 novel false-accepts, selected on ALL "
                    "probes -- optimistic by construction; the held-out numbers bound it",
        },
        # Back-compat: several readers (and the older artifacts --baseline diffs against) look for
        # `recommended` right here. Keep the alias rather than break them.
        "recommended": rec,
        "held_out_procedure": em.kfold_operating_point(probes, k=k, repeats=repeats, seed=seed),
        "at_points": points,
        "note": "recommended = max coverage with 0 wrong names AND 0 novel false-accepts; set "
                "config reid_auto_threshold/reid_auto_margin from it (re-run as the cast grows). "
                "Read held_out_procedure.pooled.error_rate as the error of the SWEEP, and "
                "at_points[].overall as the error of a FIXED point. They are different numbers.",
    }


def _iou_co_presence_check(conn, species: str, matcher: VisitMatcher, iou_cut: float = 0.45) -> dict:
    """Measure the IoU < 0.45 co-presence cut. Folklore: two SEPARATE animals in one frame sit at
    low IoU, while the detector double-boxing ONE animal sits at IoU 0.5-1.0 -- so 0.45 should fall
    in a low-density valley between the modes. We histogram every same-timestamp same-species box
    PAIR's IoU, then cross-check the still-frame rule against the INDEPENDENT clip signal
    (VisitMatcher.is_multi via sustained tracklets, which doesn't use IoU at all): do the visits the
    clips call 'pair' also carry more sub-cut simultaneous boxes than the clip-solo visits?"""
    rows = conn.execute(
        """SELECT visit_id, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2
           FROM detections WHERE species = ? AND visit_id IS NOT NULL""", (species,)).fetchall()
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[(r["visit_id"], r["timestamp"])].append(
            (r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]))

    ious = []
    for boxes in by_frame.values():
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ious.append(iou(boxes[i], boxes[j]))
    hist = {}
    if ious:
        a = np.asarray(ious)
        edges = np.linspace(0, 1, 11)
        counts, _ = np.histogram(a, bins=edges)
        hist = {f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(counts[i]) for i in range(10)}
        below = float((a < iou_cut).mean())
    else:
        below = None

    # Independent cross-check: clip-multi vs clip-solo visits, mean sub-cut simultaneous-box frames.
    clip_multi = {vid: n for vid, n in matcher.clip_co_presence.items()
                  if n >= matcher.cfg.reid_clip_co_presence_min_clips}
    multi_frames, solo_frames = [], []
    for vid in matcher.protos:
        f = matcher.co_presence(vid)  # count of frames with IoU<0.45 separated boxes, in this visit
        (multi_frames if vid in clip_multi else solo_frames).append(f)
    return {
        "iou_cut": iou_cut,
        "simultaneous_box_pairs": len(ious),
        "fraction_below_cut": below,
        "histogram": hist,
        "clip_crosscheck": {
            "note": "clip-multi flag is IoU-independent (sustained tracklets); if 0.45 is a good "
                    "cut, clip-pair visits should show more sub-cut simultaneous-box frames.",
            "clip_pair_visits": {"n": len(multi_frames),
                                 "mean_subcut_frames": float(np.mean(multi_frames)) if multi_frames else None},
            "clip_solo_visits": {"n": len(solo_frames),
                                 "mean_subcut_frames": float(np.mean(solo_frames)) if solo_frames else None},
        },
    }


def eval_reid(conn, cfg, species: str, *, kfold_k: int = 5, kfold_repeats: int = 5,
              kfold_seed: int = 0, at_points: list | None = None) -> dict:
    """Re-ID separation eval for one species (default raccoon). Reuses VisitMatcher so the
    prototypes, confirmed labels, and multi-animal exclusion are byte-for-byte what the live loop
    uses -- this measures the SHIPPING logic, not a re-implementation of it."""
    matcher = VisitMatcher(conn, species, cfg)
    confirmed = matcher.confirmed                      # {visit_id: individual_name}

    solo, excluded_multi, excluded_thin = {}, {}, {}
    for vid, name in confirmed.items():
        if vid not in matcher.protos:
            excluded_thin[vid] = name                  # too few embedded crops for a prototype
        elif matcher.is_multi(vid):
            excluded_multi[vid] = name                 # blended two-animal prototype -> exclude
        else:
            solo[vid] = name

    # Same/different pairwise separation needs >= 2 individuals with >= 1 solo visit each, and at
    # least one individual with >= 2 (to form a same-pair). Report honestly if the data is too thin.
    counts = Counter(solo.values())
    protos = {vid: matcher.protos[vid] for vid in solo}

    # The protocol's units: identity + NIGHT key (-12h shift) + timestamp, one per solo visit.
    units = loo_units(protos, solo, matcher.visit_started)
    nights = {u.key: u.night for u in units}
    sim = appearance_similarity(protos)

    overall = (_pair_separation(protos, solo, nights, session_blocked=True)
               if len(solo) >= 2 else {"note": "too few solo visits"})
    overall_unblocked = (_pair_separation(protos, solo, nights, session_blocked=False)
                         if len(solo) >= 2 else {"note": "too few solo visits"})

    # Stratify by source (a same-individual pair is compared within one source's visits).
    src_of = {r[0]: r[1] for r in conn.execute(
        "SELECT id, source FROM visits WHERE species = ?", (species,)).fetchall()}
    by_source = {}
    for src in sorted({src_of.get(v) for v in solo if src_of.get(v)}):
        sub = {v: protos[v] for v in solo if src_of.get(v) == src}
        sub_lab = {v: solo[v] for v in sub}
        if len({p for p in sub_lab.values()}) >= 2:
            by_source[src] = _pair_separation(sub, sub_lab, nights, session_blocked=True)
        else:
            by_source[src] = {"n_solo_visits": len(sub), "note": "single individual in this source"}

    # Identification, both ways, on the identical probe set. `loo` is the production-faithful
    # nearest-template ranking through the injectable protocol; `loo_centroid` is the mean-template
    # variant that also exercises the live reid_novel_threshold.
    loo = _blocked_unblocked_loo(units, sim)
    probes = loo.pop("_probes")
    loo.pop("_probes_unblocked", None)
    loo["embargo_curve"] = _embargo_curve(units, sim)
    loo_centroid = {
        "blocked": _identify_loo(protos, solo, cfg.reid_novel_threshold, nights,
                                 session_blocked=True),
        "unblocked": _identify_loo(protos, solo, cfg.reid_novel_threshold, nights,
                                   session_blocked=False),
        "note": "centroid templates (mean prototype per individual); production ranks by nearest "
                "template -- see identification_loo for that",
    }

    iou_check = _iou_co_presence_check(conn, species, matcher)

    # Fixed points worth scoring directly: whatever the operator asked for, plus the point the
    # live config is running right now (so the report always answers "what is my config doing?").
    fixed = list(at_points or [])
    if getattr(cfg, "reid_auto_threshold", 0.0):
        fixed.append((round(float(cfg.reid_auto_threshold), 4),
                      round(float(cfg.reid_auto_margin), 4), "current config"))
    auto_sweep = _auto_assign_sweep(probes, k=kfold_k, repeats=kfold_repeats, seed=kfold_seed,
                                    at_points=fixed)

    return {
        "species": species,
        "embed_model": EMBED_MODEL,
        "confirmed_visits": {
            "total": len(confirmed),
            "solo_with_prototype": len(solo),
            "excluded_multi_animal": len(excluded_multi),
            "excluded_too_thin": len(excluded_thin),
            "solo_by_individual": dict(counts),
            "distinct_nights": len({n for n in nights.values() if n is not None}),
        },
        "separation": overall,
        "separation_unblocked": overall_unblocked,
        "by_source": by_source,
        "identification_loo": loo,
        "identification_loo_centroid": loo_centroid,
        "iou_co_presence": iou_check,
        "auto_assign_sweep": auto_sweep,
        "config_threshold": {
            "reid_novel_threshold": cfg.reid_novel_threshold,
            "note": "the live 'possibly someone new' cut; compare to separation.best_threshold",
        },
        "config_auto_assign": {
            "reid_auto_threshold": cfg.reid_auto_threshold,
            "reid_auto_margin": cfg.reid_auto_margin,
            "note": "the nightly auto-name bars; compare to auto_assign_sweep.recommended "
                    "(0.0 = the pass is disabled)",
        },
    }


# ===========================================================================
# Reporting: JSON artifact + console summary.
# ===========================================================================

def _save_artifact(result: dict) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"eval_{stamp}.json"
    path.write_text(json.dumps(result, indent=2, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# --baseline: the regression gate the docstring has promised since day one.
#
# _save_artifact has been writing reports/eval_<stamp>.json for months and NOTHING has ever read
# one back. That is how a documented AUC slide of 0.81 -> 0.635 -> 0.617 accumulated across
# releases without anybody noticing which change caused which step. This closes the loop: diff
# today's run against a saved one, print the deltas, and exit non-zero when a headline metric has
# moved the wrong way by more than a stated tolerance.
# ---------------------------------------------------------------------------

# (label, candidate dotted paths in preference order, direction).
#   "higher" -- a drop past the tolerance is a regression (accuracy, AUC, coverage).
#   "lower"  -- a rise past the tolerance is a regression (error rate, calibration error).
#   "info"   -- printed and never gated (chance rates, the recommended bars themselves: a moved
#               operating point is news, not a failure, and gating on it would make the harness
#               refuse to notice a corpus that legitimately changed).
# The extra paths are fallbacks so a pre-2026-08-05 artifact still diffs against a current run --
# each fallback is the closest LIKE-FOR-LIKE quantity (the old flat identification_loo.top1_accuracy
# was centroid templates over scorable probes only, so it is matched against exactly that), and the
# printed row says when a fallback was used.
BASELINE_METRICS = (
    ("pair AUC (session-blocked)", ("reid.separation.auc",), "higher"),
    ("pair AUC (unblocked)", ("reid.separation_unblocked.auc", "reid.separation.auc"), "higher"),
    ("LOO top-1 (session-blocked)", ("reid.identification_loo.blocked.top1_accuracy",), "higher"),
    ("LOO top-1 (unblocked)", ("reid.identification_loo.unblocked.top1_accuracy",), "higher"),
    ("LOO chance (majority class)", ("reid.identification_loo.blocked.chance.rate",), "info"),
    ("centroid LOO top-1, scorable (unblocked)",
     ("reid.identification_loo_centroid.unblocked.top1_accuracy_scorable",
      "reid.identification_loo.top1_accuracy"), "higher"),
    ("recommended threshold", ("reid.auto_assign_sweep.in_sample.recommended.threshold",
                               "reid.auto_assign_sweep.recommended.threshold"), "info"),
    ("recommended margin", ("reid.auto_assign_sweep.in_sample.recommended.margin",
                            "reid.auto_assign_sweep.recommended.margin"), "info"),
    ("recommended coverage (in-sample)", ("reid.auto_assign_sweep.in_sample.recommended.coverage",
                                          "reid.auto_assign_sweep.recommended.coverage"), "higher"),
    ("held-out coverage (procedure)",
     ("reid.auto_assign_sweep.held_out_procedure.pooled.coverage",), "higher"),
    ("held-out error rate (procedure)",
     ("reid.auto_assign_sweep.held_out_procedure.pooled.error_rate",), "lower"),
    ("species macro-F1 (present classes)",
     ("species.precision_recall_f1.macro_present_classes.f1",), "higher"),
    ("species accuracy", ("species.precision_recall_f1.accuracy",), "higher"),
    ("species chance (majority class)", ("species.chance.rate",), "info"),
    ("species calibration ECE", ("species.calibration.ece",), "lower"),
)


def _dig(d, dotted: str):
    """Walk a dotted path into nested dicts; None if any step is missing or not a dict."""
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _first_present(d, paths):
    """First (value, path) whose dotted path resolves to a number; (None, None) if none do."""
    for p in paths:
        v = _dig(d, p)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v), p
    return None, None


def load_artifact(path) -> dict:
    """Read a saved reports/eval_*.json back into a dict. Raises SystemExit with a readable
    message rather than a traceback -- this runs from a .bat, and a stack trace there is noise."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--baseline: no such artifact: {p}")
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise SystemExit(f"--baseline: cannot read {p}: {e}")


def latest_artifact(reports_dir=None):
    """Newest reports/eval_*.json by filename (the stamp sorts chronologically), or None."""
    d = Path(reports_dir) if reports_dir else REPORTS_DIR
    files = sorted(d.glob("eval_*.json")) if d.exists() else []
    return files[-1] if files else None


def compare_artifacts(baseline: dict, current: dict, tolerance: float = 0.05) -> dict:
    """Diff two eval artifacts on the headline metrics. Pure -- takes two dicts, returns a report.

    A metric regresses when it moves the wrong way by MORE than `tolerance` (absolute, in the
    metric's own units). The tolerance exists because this corpus is 138 visits over three
    effective classes: single-visit noise moves top-1 by 0.007 and moves nothing real, and a gate
    that fires on that gets muted within a week, which is worse than no gate. A metric missing
    from either side is reported as such and never counted as a regression -- "the species eval
    wasn't run today" is not a regression."""
    rows, regressions = [], []
    for label, paths, direction in BASELINE_METRICS:
        b, bpath = _first_present(baseline, paths)
        c, cpath = _first_present(current, paths)
        row = {"metric": label, "direction": direction, "baseline": b, "current": c,
               "delta": (c - b) if (b is not None and c is not None) else None,
               "regressed": False, "note": None}
        if b is None or c is None:
            row["note"] = ("not in the baseline artifact" if b is None
                           else "not measured in this run")
        else:
            if bpath != cpath:
                row["note"] = f"compared via fallback path ({bpath} -> {cpath})"
            if direction == "higher" and c < b - tolerance:
                row["regressed"] = True
            elif direction == "lower" and c > b + tolerance:
                row["regressed"] = True
        if row["regressed"]:
            regressions.append(row)
        rows.append(row)
    return {"tolerance": tolerance, "rows": rows, "regressions": regressions,
            "ok": not regressions,
            "baseline_run_at": _dig(baseline, "meta.run_at"),
            "baseline_commit": _dig(baseline, "meta.git_commit"),
            "current_commit": _dig(current, "meta.git_commit")}


def _print_baseline_diff(diff: dict, baseline_path) -> None:
    print("\n" + "=" * 74)
    print("BASELINE DIFF  (regression gate)")
    print("=" * 74)
    print(f"  baseline:  {baseline_path}")
    print(f"    run at:  {diff['baseline_run_at'] or '(unknown)'}   "
          f"commit {(diff['baseline_commit'] or '?')[:8]}")
    print(f"    vs this run's commit {(diff['current_commit'] or '?')[:8]}   "
          f"tolerance +/- {diff['tolerance']:.3f}")
    print(f"\n    {'metric':42} {'baseline':>9} {'current':>9} {'delta':>9}")
    for r in diff["rows"]:
        b = "     n/a" if r["baseline"] is None else f"{r['baseline']:9.3f}"
        c = "     n/a" if r["current"] is None else f"{r['current']:9.3f}"
        d = "      n/a" if r["delta"] is None else f"{r['delta']:+9.3f}"
        flag = "  <- REGRESSION" if r["regressed"] else ("  (info)" if r["direction"] == "info" else "")
        print(f"    {r['metric'][:42]:42} {b} {c} {d}{flag}")
        if r["note"]:
            print(f"      {'':40}   note: {r['note']}")
    if diff["ok"]:
        print("\n  -> no regression past tolerance. (exit 0)")
    else:
        print(f"\n  -> {len(diff['regressions'])} REGRESSION(S) past tolerance: "
              f"{', '.join(r['metric'] for r in diff['regressions'])}  (exit 1)")


def _fmt_pct(x):
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def _fmt_acc(x, chance):
    """An accuracy is never printed alone. `chance` is the majority-class rate for the SAME rows."""
    if x is None:
        return "  n/a"
    if chance is None:
        return f"{x * 100:5.1f}%  (chance n/a)"
    return f"{x * 100:5.1f}%  (chance {chance * 100:4.1f}%, {x - chance:+.3f})"


def _print_species(s: dict) -> None:
    gt = s["ground_truth"]
    print("\n" + "=" * 74)
    print("SPECIES CLASSIFIER EVAL  (BioCLIP label + confidence vs human verdicts)")
    print("=" * 74)
    print(f"  human-reviewed rows:        {gt['verified_rows_total']:>6}")
    print(f"  prediction intact (gradable):{gt['prediction_intact']:>5}   "
          f"(confirmed {gt['confirmed_correct']}, rejected {gt['rejected_wrong']})")
    print(f"  correction recovered:        {gt['correction_recovered']:>5}   "
          f"<- model prediction preserved, graded vs the human's true label")
    print(f"  overwritten by a correction: {gt['prediction_overwritten_by_correction']:>5}   "
          f"<- legacy (pre-fix): model's own prediction lost, NOT gradable")
    if not s["graded_rows"]:
        print("  -> nothing gradable. (Confirm some auto-labels in the dashboard to build truth.)")
        return

    prf = s["precision_recall_f1"]
    print(f"\n  Per graded species (recovered corrections now contribute real true labels, so recall")
    print(f"  is measurable; legacy pre-fix corrections still cap how complete it is):")
    print(f"    {'species':22} {'prec':>6} {'rec':>6} {'F1':>6} {'support':>8}")
    per = prf["per_label"]
    for sp in sorted(per, key=lambda k: -per[k]["support"]):
        m = per[sp]
        tag = "  (non-critter)" if sp.lower() in _NON_CRITTER else ""
        note = "  <- rejected-only (no true instances)" if m["support"] == 0 else ""
        print(f"    {sp[:22]:22} {m['precision']:6.2f} {m['recall']:6.2f} {m['f1']:6.2f} "
              f"{m['support']:8}{tag}{note}")
    mp = prf["macro_present_classes"]
    print(f"    {'macro (present classes)':22} {mp['precision']:6.2f} {mp['recall']:6.2f} "
          f"{mp['f1']:6.2f}   over {len(prf['present_classes'])} species with ground truth")

    print(f"\n  Calibration (does the confidence mean what it says?)  ECE = "
          f"{s['calibration']['ece']:.3f}" if s['calibration']['ece'] is not None else
          "\n  Calibration: no data")
    print(f"    {'confidence bin':16} {'mean conf':>10} {'accuracy':>9} {'n':>6}")
    for b in s["calibration"]["bins"]:
        if b["n"]:
            print(f"    [{b['lo']:.1f}, {b['hi']:.1f})     "
                  f"{b['mean_confidence']:10.2f} {_fmt_pct(b['accuracy'])} {b['n']:6}")

    ch = s.get("chance") or {}
    print(f"\n  CHANCE RATE over the graded rows: always answering '{ch.get('label')}' would score "
          f"{_fmt_pct(ch.get('rate'))}")
    print(f"    (a uniform guess over {ch.get('n_classes')} classes: "
          f"{_fmt_pct(ch.get('uniform_rate'))}; {ch.get('n_unanswerable', 0)} rows are unwinnable "
          f"for any predictor -- rejected calls whose true class was never recorded)")
    print(f"    Every accuracy below carries its OWN slice's baseline, not this one.")

    tr = s["trust_rule_check"]
    print(f"\n  FOLKLORE: 'confidence >= 0.8 trust, < 0.5 look closer'")
    for key, lab in (("conf_ge_0.8", "conf >= 0.8"), ("conf_0.5_to_0.8", "0.5 - 0.8"),
                     ("conf_lt_0.5", "conf < 0.5 ")):
        b = tr[key]
        print(f"    {lab} : accuracy {_fmt_acc(b['accuracy'], (b.get('chance') or {}).get('rate'))}"
              f"  (n={b['n']})")

    print(f"\n  By time of day ({s['daynight_method']}):")
    for period, d in s["by_daynight"].items():
        print(f"    {period:6}: accuracy {_fmt_acc(d['accuracy'], (d.get('chance') or {}).get('rate'))}"
              f"  (n={d['n']})")
    print(f"  By source:")
    for src, d in s["by_source"].items():
        print(f"    {src:16}: accuracy {_fmt_acc(d['accuracy'], (d.get('chance') or {}).get('rate'))}"
              f"  (n={d['n']})")


def _print_reid(r: dict) -> None:
    print("\n" + "=" * 74)
    print(f"RE-ID EVAL  (visit-prototype cross-session separation, species = {r['species']})")
    print("=" * 74)
    cv = r["confirmed_visits"]
    print(f"  confirmed visits:            {cv['total']:>4}")
    print(f"  solo visits w/ prototype:    {cv['solo_with_prototype']:>4}   "
          f"{dict(cv['solo_by_individual'])}")
    print(f"  excluded (multi-animal):     {cv['excluded_multi_animal']:>4}   "
          f"(blended prototype -- correctly left out)")
    print(f"  excluded (too few crops):    {cv['excluded_too_thin']:>4}")

    if r["confirmed_visits"].get("distinct_nights") is not None:
        print(f"  distinct nights covered:     {r['confirmed_visits']['distinct_nights']:>4}   "
              f"(night = timestamp shifted -12h)")

    sep = r["separation"]
    if "same_individual_pairs" not in sep:
        print(f"  -> {sep.get('note', 'not enough data to measure separation')}")
    else:
        sa, di = sep["same_individual_pairs"], sep["different_individual_pairs"]
        if sa["n"] and di["n"]:
            print(f"\n  SAME-individual visit pairs  (n={sa['n']:>5}): "
                  f"median {sa['median']:.3f}  mean {sa['mean']:.3f}  "
                  f"[p10 {sa['p10']:.3f} .. p90 {sa['p90']:.3f}]")
            print(f"  DIFF-individual visit pairs  (n={di['n']:>5}): "
                  f"median {di['median']:.3f}  mean {di['mean']:.3f}  "
                  f"[p10 {di['p10']:.3f} .. p90 {di['p90']:.3f}]")
            print(f"  FOLKLORE claim: same 0.83-0.93, different 0.36-0.42.")
        else:
            print(f"\n  SAME-individual pairs n={sa['n']}, DIFF-individual pairs n={di['n']} -- "
                  f"no separation is measurable.")
        print(f"  ({sep['same_night_positive_pairs_dropped']} same-night same-individual pairs "
              f"dropped from the positive class -- one night is one session, not two observations)")
        if sa["n"] == 0 and sep["same_night_positive_pairs_dropped"]:
            print(f"  -> EVERY same-individual pair was same-night. This corpus contains no "
                  f"cross-session evidence at all; the unblocked AUC below is measuring sessions.")

        unb = r.get("separation_unblocked") or {}
        if sep.get("auc") is not None:
            print(f"\n  Separation ROC-AUC:          {sep['auc']:.3f}   session-BLOCKED   "
                  f"(0.5 = chance)")
            if unb.get("auc") is not None:
                print(f"                               {unb['auc']:.3f}   unblocked (leaked)  "
                      f"-> session leak {unb['auc'] - sep['auc']:+.3f}")
        bt = sep["best_threshold"]
        cur_novel = r['config_threshold']['reid_novel_threshold']
        if bt.get("threshold") is not None:
            print(f"  Best operating threshold:    {bt['threshold']:.3f}   "
                  f"(tpr {bt['tpr']:.2f}, fpr {bt['fpr']:.2f}, acc {bt['accuracy']:.2f})")
            print(f"  Current config value:        reid_novel_threshold = {cur_novel}")
            diff = bt['threshold'] - cur_novel
            print(f"    -> data suggests {bt['threshold']:.2f} "
                  f"({'higher' if diff > 0 else 'lower'} than current by {abs(diff):.2f}); "
                  f"MEASURE-and-recommend only, config is not changed.")
        else:
            print(f"  Current config value:        reid_novel_threshold = {cur_novel} "
                  f"(no data to recommend against)")

    # ---- Identification, blocked vs unblocked. The single most important block here. ----
    loo = r.get("identification_loo") or {}
    b, u = loo.get("blocked"), loo.get("unblocked")
    if b and b.get("top1_accuracy") is not None:
        chance = (b.get("chance") or {}).get("rate")
        print(f"\n  LEAVE-ONE-VISIT-OUT IDENTIFICATION  (nearest template per individual, the "
              f"production ranking)")
        print(f"    probes:                    {b['n_probes']}  "
              f"(every solo visit; one whose identity has no allowed template is scored WRONG, "
              f"so both rows below share an identical probe set)")
        print(f"    top-1, SESSION-BLOCKED:    {_fmt_acc(b['top1_accuracy'], chance)}   <- the real number")
        print(f"    top-1, unblocked (leaked): {_fmt_acc(u['top1_accuracy'], chance)}")
        gap = loo.get("session_leak_gap")
        if gap is not None:
            print(f"    SESSION LEAK:              {gap:+.3f} top-1 -- that much of 'identity' was "
                  f"'same night'")
        print(f"    blocked, over probes whose identity WAS available: "
              f"{_fmt_pct(b.get('top1_accuracy_scorable'))} (n={b['n_scorable']})")
        worst = sorted(b["per_individual"].items(), key=lambda kv: (kv[1]["accuracy"] or 0.0))
        print(f"    per individual (blocked):  " + ",  ".join(
            f"{n} {v['top1_correct']}/{v['n']}" for n, v in worst))

    ec = loo.get("embargo_curve") or []
    if ec:
        print(f"\n    Decay with an enforced probe->template time gap (session-blocked throughout):")
        print(f"      {'embargo':>9}  {'top-1':>7}  {'chance':>7}  {'probes':>6}")
        for row in ec:
            lab = "none" if row["embargo_days"] is None else f"{row['embargo_days']}d"
            print(f"      {lab:>9}  {_fmt_pct(row['top1_accuracy'])}  "
                  f"{_fmt_pct(row['chance'])}  {row['n_probes']:>6}")

    cen = r.get("identification_loo_centroid") or {}
    cb = cen.get("blocked")
    if cb and cb.get("top1_accuracy") is not None:
        at = cb["at_threshold"]
        cu = cen.get("unblocked") or {}
        print(f"\n    Centroid-template variant (mean prototype per individual):")
        print(f"      top-1 blocked {_fmt_pct(cb['top1_accuracy'])} / unblocked "
              f"{_fmt_pct(cu.get('top1_accuracy'))}   chance "
              f"{_fmt_pct((cb.get('chance') or {}).get('rate'))}")
        print(f"      at reid_novel_threshold {at['threshold']:.2f} (blocked): "
              f"{at['accepted_correct']} accepted-correct, "
              f"{at['correct_but_flagged_novel']} correct-but-flagged-novel, "
              f"{at['wrong_individual']} wrong")

    # ---- Auto-assign: three numbers that are NOT the same number. ----
    sw = r.get("auto_assign_sweep") or {}
    if sw.get("n_probes"):
        print(f"\n  AUTO-ASSIGN OPERATING POINT  (session-blocked LOO, {sw['n_probes']} known "
              f"probes + {sw['n_novel_probes']} novel probes)")
        rec = (sw.get("in_sample") or {}).get("recommended") or sw.get("recommended")
        ca = r.get("config_auto_assign", {})
        cur_t, cur_m = ca.get("reid_auto_threshold", 0.0), ca.get("reid_auto_margin", 0.0)
        if rec:
            print(f"    1. IN-SAMPLE recommendation (selected and scored on the same probes -- "
                  f"optimistic):")
            print(f"       similarity >= {rec['threshold']:.2f}  AND  runner-up margin >= "
                  f"{rec['margin']:.2f}")
            print(f"       -> auto-names {rec['auto_named']}/{sw['n_probes']} "
                  f"({_fmt_pct(rec['coverage'])}) with 0 wrong + 0 novel false-accepts")
        else:
            print(f"    1. IN-SAMPLE recommendation: none -- no zero-error point exists on this "
                  f"corpus. Leave auto-assign disabled.")

        ho = sw.get("held_out_procedure") or {}
        pooled = ho.get("pooled")
        if pooled:
            print(f"    2. HELD-OUT, the SWEEP PROCEDURE ({ho['repeats']}x{ho['k']}-fold: point "
                  f"picked on the training folds, scored on the held-out fold):")
            print(f"       coverage {_fmt_pct(pooled['coverage'])} of "
                  f"{pooled['n_known_probes']} held-out known probes; "
                  f"{pooled['wrong']} wrong + {pooled['novel_false_accepts']} novel false-accepts "
                  f"of {pooled['names_written']} names written")
            print(f"       error rate of THE PROCEDURE: {_fmt_pct(pooled['error_rate'])}   "
                  f"<- this is NOT the error rate of any single point below")
            if ho.get("folds_with_no_safe_point"):
                print(f"       ({ho['folds_with_no_safe_point']} of {ho['folds_scored']} folds "
                      f"found no safe point and named nothing)")
            if ho.get("selected_points"):
                picks = ", ".join(f"{p['threshold']:.2f}/{p['margin']:.2f} x{p['n_folds']}"
                                  for p in ho["selected_points"][:4])
                print(f"       points the folds picked: {picks}")

        pts = sw.get("at_points") or []
        if pts:
            print(f"    3. FIXED points, evaluated directly (no selection involved -- this is what "
                  f"shipping that point does):")
            for row in pts:
                o, f = row["overall"], row.get("folds") or {}
                bound = (f"   (0 errors -> <= {o['error_rate_upper_bound_95'] * 100:.0f}% by "
                         f"rule-of-three)" if o["error_rate_upper_bound_95"] is not None else "")
                print(f"       {o['threshold']:.2f} / {o['margin']:.2f}  [{row.get('why', '')}]: "
                      f"{o['assigned']}/{o['n_known_probes']} assigned "
                      f"({_fmt_pct(o['coverage'])}), {o['wrong']} wrong, "
                      f"{o['novel_false_accepts']} novel false-accept{bound}")
                if f.get("coverage_mean") is not None:
                    print(f"           fold stability: coverage "
                          f"{_fmt_pct(f['coverage_min'])} .. {_fmt_pct(f['coverage_max'])} "
                          f"across {f['n_folds']} held-out folds, worst fold "
                          f"{f['errors_max_in_a_fold']} error(s)")
        state = "DISABLED" if not cur_t else f"{cur_t:.2f} / {cur_m:.2f}"
        print(f"    current config: reid_auto_threshold/margin = {state}; "
              f"MEASURE-and-recommend only, config is not changed.")

    io = r["iou_co_presence"]
    if io["fraction_below_cut"] is not None:
        cc = io["clip_crosscheck"]
        print(f"\n  IoU {io['iou_cut']} co-presence cut  ({io['simultaneous_box_pairs']} simultaneous "
              f"box pairs, {_fmt_pct(io['fraction_below_cut'])} below cut):")
        mp = cc["clip_pair_visits"]["mean_subcut_frames"]
        ms = cc["clip_solo_visits"]["mean_subcut_frames"]
        print(f"    clip-pair visits (n={cc['clip_pair_visits']['n']}): "
              f"mean {mp:.1f} sub-cut frames/visit" if mp is not None else
              "    clip-pair visits: none")
        print(f"    clip-solo visits (n={cc['clip_solo_visits']['n']}): "
              f"mean {ms:.1f} sub-cut frames/visit" if ms is not None else
              "    clip-solo visits: none")


def _print_header(meta: dict) -> None:
    print("=" * 74)
    print("BACKYARD CRITTER-CAM  ::  ML EVALUATION HARNESS  (read-only, offline)")
    print("=" * 74)
    print(f"  run:     {meta['run_at']}")
    print(f"  commit:  {meta['git_commit'] or '(not a git checkout)'}")
    print(f"  db:      {meta['db_path']}")


# ===========================================================================
# CLI
# ===========================================================================

def run(cfg, *, do_species: bool, do_reid: bool, reid_species: str,
        kfold_k: int = 5, kfold_repeats: int = 5, kfold_seed: int = 0,
        at_points: list | None = None) -> dict:
    """Run the requested evals against the live DB (read-only) and return the full result dict."""
    conn = open_readonly(cfg.db_path)
    if conn is None:
        raise SystemExit(f"No database at {cfg.db_path} -- run the rig first (or pass --db).")
    meta = {
        "run_at": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
        "db_path": str(cfg.db_path),
        # 2 = session-blocked protocol + held-out operating-point selection (2026-08-05). A v1
        # artifact's re-ID numbers are session-leaked and in-sample; --baseline says so when it
        # has to fall back to them.
        "harness_version": 2,
    }
    result = {"meta": meta}
    try:
        if do_species:
            result["species"] = eval_species(conn, cfg)
        if do_reid:
            result["reid"] = eval_reid(conn, cfg, reid_species, kfold_k=kfold_k,
                                       kfold_repeats=kfold_repeats, kfold_seed=kfold_seed,
                                       at_points=at_points)
    finally:
        conn.close()
    return result


def _parse_at_point(s: str):
    """'--at-point 0.88,0.02' -> (0.88, 0.02, '--at-point'). Fails loudly: a typo'd operating
    point silently scored at the wrong bars is worse than a crash."""
    try:
        t, m = s.split(",")
        return (round(float(t), 4), round(float(m), 4), "--at-point")
    except ValueError:
        raise SystemExit(f"--at-point expects THRESHOLD,MARGIN (e.g. 0.88,0.02); got {s!r}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Local, read-only ML evaluation harness for the backyard critter rig.")
    p.add_argument("--species", action="store_true",
                   help="Run ONLY the species-classifier eval (default: run both).")
    p.add_argument("--reid", action="store_true",
                   help="Run ONLY the re-ID eval (default: run both).")
    p.add_argument("--reid-species", default=None,
                   help="Species the re-ID eval runs on (default: config.reid_species, 'raccoon').")
    p.add_argument("--db", default=None, help="Path to the DB (default: config.db_path). Read-only.")
    p.add_argument("--json", action="store_true",
                   help="Also print the full machine-readable result JSON to stdout.")
    p.add_argument("--no-save", action="store_true",
                   help="Don't write the reports/ JSON artifact (console summary only).")
    p.add_argument("--baseline", default=None, metavar="PATH|latest",
                   help="Diff this run against a saved reports/eval_*.json ('latest' picks the "
                        "newest). EXITS NON-ZERO when a headline metric regresses past "
                        "--tolerance -- this is the regression gate for a .bat or a hook.")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="How far a metric may move the wrong way before --baseline calls it a "
                        "regression (absolute, metric's own units; default 0.05).")
    p.add_argument("--at-point", action="append", default=None, metavar="THRESH,MARGIN",
                   help="Also score this FIXED auto-assign operating point directly "
                        "(e.g. 0.88,0.02). Repeatable. The live config's point is always scored.")
    p.add_argument("--kfold", type=int, default=5, help="Folds for the held-out sweep (default 5).")
    p.add_argument("--kfold-repeats", type=int, default=5,
                   help="Repeats of the k-fold split (default 5).")
    p.add_argument("--kfold-seed", type=int, default=0,
                   help="Seed for the k-fold shuffle -- fixed so a re-run reproduces (default 0).")
    args = p.parse_args()

    cfg = config.CONFIG
    if args.db:
        cfg.db_path = Path(args.db)
    reid_species = args.reid_species or getattr(cfg, "reid_species", "raccoon")

    # Resolve the baseline BEFORE this run writes its own artifact, or 'latest' would diff the
    # run against itself and always pass.
    baseline_path = baseline = None
    if args.baseline:
        baseline_path = latest_artifact() if args.baseline == "latest" else Path(args.baseline)
        if baseline_path is None:
            raise SystemExit(f"--baseline latest: no artifacts in {REPORTS_DIR}")
        baseline = load_artifact(baseline_path)

    # No flag, or neither of the two selectors -> run both.
    do_species = args.species or not (args.species or args.reid)
    do_reid = args.reid or not (args.species or args.reid)

    result = run(cfg, do_species=do_species, do_reid=do_reid, reid_species=reid_species,
                 kfold_k=args.kfold, kfold_repeats=args.kfold_repeats, kfold_seed=args.kfold_seed,
                 at_points=[_parse_at_point(s) for s in (args.at_point or [])])

    _print_header(result["meta"])
    if "species" in result:
        _print_species(result["species"])
    if "reid" in result:
        _print_reid(result["reid"])

    exit_code = 0
    if baseline is not None:
        diff = compare_artifacts(baseline, result, args.tolerance)
        result["baseline_diff"] = diff
        _print_baseline_diff(diff, baseline_path)
        exit_code = 0 if diff["ok"] else 1

    if not args.no_save:
        path = _save_artifact(result)
        print(f"\nSaved artifact: {path}")
    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(result, indent=2, default=_json_default))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
