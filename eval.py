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
from individuals import EMBED_MODEL, VisitMatcher, iou, rank_templates

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"

# Sentinel "true label" for a model prediction the human REJECTED (species_verified = 0): the
# model was wrong, but the corrected true species was overwritten, so the real class is unknown.
# It rides in the confusion matrix as a true-label row so those rows still count against precision.
MISPREDICTED = "(mispredicted - true label not recorded)"

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
    def _reliability_for(subset):
        c = [x[0] for x in subset]
        k = [x[1] for x in subset]
        rc = em.reliability_curve(c, k, n_bins=10) if subset else None
        acc = float(np.mean(k)) if subset else None
        return {"n": len(subset), "accuracy": acc,
                "ece": rc["ece"] if rc else None, "reliability": rc}

    triples = [(float(r["pred_conf"]), int(r["correct"]), r)
               for r in graded if r["pred_conf"] is not None]

    by_species = {}
    for sp in pred_species:
        sub = [(c, k) for c, k, r in triples if r["pred"] == sp]
        by_species[sp] = {"n": len(sub), "accuracy": float(np.mean([k for _, k in sub]))
                          if sub else None, "non_critter": sp.lower() in _NON_CRITTER}

    by_daynight = {}
    for period in ("day", "night"):
        sub = [(c, k) for c, k, r in triples
               if (p := db.parse_local(r["timestamp"])) is not None and dn_fn(p) == period]
        by_daynight[period] = _reliability_for(sub)

    by_source = {}
    for src in sorted({r["source"] for r in graded}):
        sub = [(c, k) for c, k, r in triples if r["source"] == src]
        by_source[src] = _reliability_for(sub)

    # ---- Folklore verdicts, in plain language ----
    # "0.8 trust / 0.5 look closer": accuracy of predictions AT/ABOVE 0.8 vs BELOW 0.5.
    hi = [k for c, k, _ in triples if c >= 0.8]
    lo = [k for c, k, _ in triples if c < 0.5]
    mid = [k for c, k, _ in triples if 0.5 <= c < 0.8]
    trust_rule = {
        "conf_ge_0.8": {"n": len(hi), "accuracy": float(np.mean(hi)) if hi else None},
        "conf_0.5_to_0.8": {"n": len(mid), "accuracy": float(np.mean(mid)) if mid else None},
        "conf_lt_0.5": {"n": len(lo), "accuracy": float(np.mean(lo)) if lo else None},
    }

    return {
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

def individual_centroids(protos: dict, labels: dict, exclude_visit=None) -> dict:
    """{name: L2-normalized mean of that individual's visit prototypes}, optionally LEAVING OUT one
    visit. This is the leave-one-visit-out contamination guard made explicit and testable: when you
    score whether visit X belongs to individual Y, X's own prototype must never be inside Y's
    template, or the match is measuring X against itself. Pass exclude_visit=X to drop it.

    `protos` = {visit_id: prototype_vector}; `labels` = {visit_id: individual_name}. Only visits
    present in both are used. A name left with no visits after the exclusion simply doesn't appear
    in the result (it can't be a template for this comparison)."""
    groups: dict = defaultdict(list)
    for vid, name in labels.items():
        if vid == exclude_visit or vid not in protos:
            continue
        groups[name].append(protos[vid])
    out = {}
    for name, vecs in groups.items():
        c = np.mean(np.stack(vecs), axis=0)
        n = np.linalg.norm(c)
        out[name] = c / n if n else c
    return out


def _pair_separation(protos: dict, labels: dict) -> dict:
    """Pairwise same- vs different-individual visit-prototype cosine separation + ROC/AUC/threshold.
    Every pair is two DISTINCT visits, so neither prototype contains the other's crops -- the
    separation is leakage-free by construction (no LOO needed for the pairwise view)."""
    vids = sorted(protos)
    same, diff = [], []
    for a, b in itertools.combinations(vids, 2):
        s = float(protos[a] @ protos[b])
        (same if labels[a] == labels[b] else diff).append(s)
    scores = np.array(same + diff, dtype=float)
    y = np.array([1] * len(same) + [0] * len(diff))

    def _dist(arr):
        if not arr:
            return {"n": 0}
        a = np.asarray(arr)
        return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
                "min": float(a.min()), "max": float(a.max())}

    roc = em.roc_curve(scores, y) if len(same) and len(diff) else {"auc": None}
    return {
        "same_individual_pairs": _dist(same),
        "different_individual_pairs": _dist(diff),
        "auc": roc.get("auc"),
        "best_threshold": em.best_threshold(scores, y),
        # Down-sample the ROC curve into the artifact (full sweep can be thousands of points).
        "roc": {"auc": roc.get("auc"),
                "tpr": roc.get("tpr", [])[::max(1, len(roc.get("tpr", [1])) // 40)],
                "fpr": roc.get("fpr", [])[::max(1, len(roc.get("fpr", [1])) // 40)]},
    }


def _identify_loo(protos: dict, labels: dict, threshold: float) -> dict:
    """Leave-one-visit-out cross-session IDENTIFICATION: for each solo confirmed visit, rebuild the
    cast's centroid templates from every OTHER solo visit (individual_centroids with exclude_visit),
    then see whether the visit's nearest template is its TRUE individual. This is the strict "could
    the system actually name this animal across sessions?" test -- and the direct exercise of the
    operating threshold: at `threshold`, is a correct top match ACCEPTED (>= threshold) or wrongly
    flagged 'possibly someone new'?

    Only visits whose individual has >= 2 solo visits are scorable (else nothing is left to build a
    template from after leaving this one out); `coverage` reports that. Returns overall top-1
    accuracy and the accept/flag breakdown at the threshold."""
    scorable = 0
    top1_correct = 0
    accepted_correct = 0
    flagged_novel_correct = 0  # correct individual, but similarity fell below threshold
    wrong = 0
    per_individual = defaultdict(lambda: {"n": 0, "top1_correct": 0})
    for vid, truth in labels.items():
        if vid not in protos:
            continue
        cents = individual_centroids(protos, labels, exclude_visit=vid)
        if truth not in cents:
            continue  # its only solo visit is this one -> no held-out template exists
        scorable += 1
        per_individual[truth]["n"] += 1
        ranked = sorted(((n, float(protos[vid] @ c)) for n, c in cents.items()),
                        key=lambda r: -r[1])
        top_name, top_sim = ranked[0]
        if top_name == truth:
            top1_correct += 1
            per_individual[truth]["top1_correct"] += 1
            if top_sim >= threshold:
                accepted_correct += 1
            else:
                flagged_novel_correct += 1
        else:
            wrong += 1
    return {
        "coverage": {"scorable_visits": scorable,
                     "note": "a visit is scorable only if its individual has another solo visit "
                             "to build a leave-one-out template from"},
        "top1_accuracy": (top1_correct / scorable) if scorable else None,
        "at_threshold": {
            "threshold": threshold,
            "accepted_correct": accepted_correct,
            "correct_but_flagged_novel": flagged_novel_correct,
            "wrong_individual": wrong,
        },
        "per_individual": {k: v for k, v in sorted(per_individual.items())},
    }


def _auto_assign_sweep(protos: dict, labels: dict) -> dict:
    """Sweep the (similarity threshold, runner-up margin) grid for the nightly AUTO-ASSIGN pass
    and recommend the max-coverage operating point with ZERO mistakes on the confirmed corpus.

    Mirrors production semantics exactly (individuals.VisitMatcher.auto_assign): each confirmed
    solo visit is matched leave-one-out against every OTHER confirmed visit, nearest-VISIT per
    individual (rank_templates, not centroids), and would be auto-named iff top similarity >=
    threshold AND (top - runner-up individual) >= margin. Two error channels, both required to be
    zero at the recommended point:
      - wrong:            a known individual's visit auto-named as someone else;
      - novel_false_accept: a probe whose TRUE individual has no other solo visit (so its identity
        is absent from the templates -- the stand-in for a genuinely NEW animal) still passing the
        bars. This is the "a stranger walks in and gets Stan's name" risk, measurable only this way.

    CAVEAT the numbers inherit: the confirmed corpus leans toward visits a human already agreed
    matched, so measured coverage is optimistic for the wild population; the zero-error property
    is what the recommendation optimizes, not the coverage estimate."""
    probes = []
    for vid, truth in labels.items():
        if vid not in protos:
            continue
        temps = [(labels[v], v, protos[v]) for v in protos if v != vid and v in labels]
        if not temps:
            continue
        ranked = rank_templates(protos[vid], temps)
        top_name, s1, _via = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        probes.append({"truth": truth, "top": top_name, "s1": s1, "lead": s1 - s2,
                       "novel_probe": not any(n == truth for n, _, _ in ranked)})
    known = [p for p in probes if not p["novel_probe"]]
    novel = [p for p in probes if p["novel_probe"]]
    if not known:
        return {"note": "no scorable probes (need >= 2 confirmed solo visits)", "n_probes": 0}

    thresholds = [round(t, 2) for t in np.arange(0.30, 0.92, 0.02)]
    margins = [round(m, 2) for m in np.arange(0.00, 0.42, 0.02)]
    zero_error = []
    for t in thresholds:
        for m in margins:
            named = [p for p in known if p["s1"] >= t and p["lead"] >= m]
            wrong = sum(1 for p in named if p["top"] != p["truth"])
            nfa = sum(1 for p in novel if p["s1"] >= t and p["lead"] >= m)
            if wrong == 0 and nfa == 0 and named:
                zero_error.append({"threshold": t, "margin": m, "auto_named": len(named),
                                   "coverage": round(len(named) / len(known), 3)})
    # Max coverage first; among ties prefer the SAFER point (higher threshold, then margin).
    zero_error.sort(key=lambda r: (-r["auto_named"], -r["threshold"], -r["margin"]))
    return {
        "n_probes": len(known),
        "n_novel_probes": len(novel),
        "recommended": zero_error[0] if zero_error else None,
        "zero_error_points": zero_error[:40],
        "note": "recommended = max coverage with 0 wrong names AND 0 novel false-accepts; "
                "set config reid_auto_threshold/reid_auto_margin from it (re-run as the cast grows)",
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


def eval_reid(conn, cfg, species: str) -> dict:
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

    overall = _pair_separation(protos, solo) if len(solo) >= 2 else {"note": "too few solo visits"}

    # Stratify by source (a same-individual pair is compared within one source's visits).
    src_of = {r[0]: r[1] for r in conn.execute(
        "SELECT id, source FROM visits WHERE species = ?", (species,)).fetchall()}
    by_source = {}
    for src in sorted({src_of.get(v) for v in solo if src_of.get(v)}):
        sub = {v: protos[v] for v in solo if src_of.get(v) == src}
        sub_lab = {v: solo[v] for v in sub}
        if len({p for p in sub_lab.values()}) >= 2:
            by_source[src] = _pair_separation(sub, sub_lab)
        else:
            by_source[src] = {"n_solo_visits": len(sub), "note": "single individual in this source"}

    loo = _identify_loo(protos, solo, cfg.reid_novel_threshold)
    iou_check = _iou_co_presence_check(conn, species, matcher)
    auto_sweep = _auto_assign_sweep(protos, solo)

    return {
        "species": species,
        "embed_model": EMBED_MODEL,
        "confirmed_visits": {
            "total": len(confirmed),
            "solo_with_prototype": len(solo),
            "excluded_multi_animal": len(excluded_multi),
            "excluded_too_thin": len(excluded_thin),
            "solo_by_individual": dict(counts),
        },
        "separation": overall,
        "by_source": by_source,
        "identification_loo": loo,
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


def _fmt_pct(x):
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


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

    tr = s["trust_rule_check"]
    print(f"\n  FOLKLORE: 'confidence >= 0.8 trust, < 0.5 look closer'")
    print(f"    conf >= 0.8 : accuracy {_fmt_pct(tr['conf_ge_0.8']['accuracy'])}  "
          f"(n={tr['conf_ge_0.8']['n']})")
    print(f"    0.5 - 0.8   : accuracy {_fmt_pct(tr['conf_0.5_to_0.8']['accuracy'])}  "
          f"(n={tr['conf_0.5_to_0.8']['n']})")
    print(f"    conf < 0.5  : accuracy {_fmt_pct(tr['conf_lt_0.5']['accuracy'])}  "
          f"(n={tr['conf_lt_0.5']['n']})")

    print(f"\n  By time of day ({s['daynight_method']}):")
    for period, d in s["by_daynight"].items():
        print(f"    {period:6}: accuracy {_fmt_pct(d['accuracy'])}  (n={d['n']})")
    print(f"  By source:")
    for src, d in s["by_source"].items():
        print(f"    {src:16}: accuracy {_fmt_pct(d['accuracy'])}  (n={d['n']})")


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

    sep = r["separation"]
    if "same_individual_pairs" not in sep:
        print(f"  -> {sep.get('note', 'not enough data to measure separation')}")
        return
    sa, di = sep["same_individual_pairs"], sep["different_individual_pairs"]
    print(f"\n  SAME-individual visit pairs  (n={sa['n']:>5}): "
          f"median {sa['median']:.3f}  mean {sa['mean']:.3f}  [p10 {sa['p10']:.3f} .. p90 {sa['p90']:.3f}]")
    print(f"  DIFF-individual visit pairs  (n={di['n']:>5}): "
          f"median {di['median']:.3f}  mean {di['mean']:.3f}  [p10 {di['p10']:.3f} .. p90 {di['p90']:.3f}]")
    print(f"  FOLKLORE claim: same 0.83-0.93, different 0.36-0.42.")

    bt = sep["best_threshold"]
    print(f"\n  Separation ROC-AUC:          {sep['auc']:.3f}   (0.5 = chance, 1.0 = perfect)")
    print(f"  Best operating threshold:    {bt['threshold']:.3f}   "
          f"(tpr {bt['tpr']:.2f}, fpr {bt['fpr']:.2f}, acc {bt['accuracy']:.2f})")
    print(f"  Current config value:        reid_novel_threshold = "
          f"{r['config_threshold']['reid_novel_threshold']}")
    diff = bt['threshold'] - r['config_threshold']['reid_novel_threshold']
    print(f"    -> data suggests {bt['threshold']:.2f} "
          f"({'higher' if diff > 0 else 'lower'} than current by {abs(diff):.2f}); "
          f"MEASURE-and-recommend only, config is not changed.")

    loo = r["identification_loo"]
    if loo["top1_accuracy"] is not None:
        at = loo["at_threshold"]
        print(f"\n  Leave-one-visit-out identification (nearest-individual centroid, self excluded):")
        print(f"    scorable visits:           {loo['coverage']['scorable_visits']}")
        print(f"    top-1 correct individual:  {_fmt_pct(loo['top1_accuracy'])}")
        print(f"    at threshold {at['threshold']:.2f}: {at['accepted_correct']} accepted-correct, "
              f"{at['correct_but_flagged_novel']} correct-but-flagged-novel, "
              f"{at['wrong_individual']} wrong")

    sw = r.get("auto_assign_sweep") or {}
    if sw.get("n_probes"):
        print(f"\n  Auto-assign sweep (nearest-visit LOO, {sw['n_probes']} known probes + "
              f"{sw['n_novel_probes']} novel probes):")
        rec = sw.get("recommended")
        if rec:
            ca = r.get("config_auto_assign", {})
            cur_t, cur_m = ca.get("reid_auto_threshold", 0.0), ca.get("reid_auto_margin", 0.0)
            print(f"    recommended bars:          similarity >= {rec['threshold']:.2f}  AND  "
                  f"runner-up margin >= {rec['margin']:.2f}")
            print(f"    -> would auto-name {rec['auto_named']}/{sw['n_probes']} "
                  f"({_fmt_pct(rec['coverage'])}) with 0 wrong + 0 novel false-accepts")
            state = "DISABLED" if not cur_t else f"{cur_t:.2f} / {cur_m:.2f}"
            print(f"    current config:            reid_auto_threshold/margin = {state}; "
                  f"MEASURE-and-recommend only, config is not changed.")
        else:
            print(f"    no zero-error operating point found -- leave auto-assign disabled.")

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

def run(cfg, *, do_species: bool, do_reid: bool, reid_species: str) -> dict:
    """Run the requested evals against the live DB (read-only) and return the full result dict."""
    conn = open_readonly(cfg.db_path)
    if conn is None:
        raise SystemExit(f"No database at {cfg.db_path} -- run the rig first (or pass --db).")
    meta = {
        "run_at": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
        "db_path": str(cfg.db_path),
        "harness_version": 1,
    }
    result = {"meta": meta}
    try:
        if do_species:
            result["species"] = eval_species(conn, cfg)
        if do_reid:
            result["reid"] = eval_reid(conn, cfg, reid_species)
    finally:
        conn.close()
    return result


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
    args = p.parse_args()

    cfg = config.CONFIG
    if args.db:
        cfg.db_path = Path(args.db)
    reid_species = args.reid_species or getattr(cfg, "reid_species", "raccoon")

    # No flag, or neither of the two selectors -> run both.
    do_species = args.species or not (args.species or args.reid)
    do_reid = args.reid or not (args.species or args.reid)

    result = run(cfg, do_species=do_species, do_reid=do_reid, reid_species=reid_species)

    _print_header(result["meta"])
    if "species" in result:
        _print_species(result["species"])
    if "reid" in result:
        _print_reid(result["reid"])

    if not args.no_save:
        path = _save_artifact(result)
        print(f"\nSaved artifact: {path}")
    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(result, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
