"""
Phase 3 (part 3) -- the SUGGEST-CONFIRM loop: "looks like Stan -- confirm or correct."

reid.py proved single crops can't be matched across sessions (cosine ~0.5 ceiling, clusters bind
to pose+lighting). What DOES work -- validated on real visits 2026-06-11 -- is matching at the
VISIT level: average a visit's best crops into one appearance PROTOTYPE and pose+burst noise
washes out. Same-animal visits on different nights then match at 0.83-0.93, while two DIFFERENT
raccoons photographed in the same frame (same light, same glass -- the perfect controlled test,
courtesy of the Notch-and-friend pair) score only ~0.36-0.42. The signal was always there; crops
were just the wrong unit.

So the loop is:
  1. every unconfirmed raccoon visit gets a suggestion -- the nearest HUMAN-CONFIRMED visit's
     individual ("Stan 0.84, via visit #1014"), or a "possibly someone new" novelty flag;
  2. you confirm or correct it in the dashboard (one click labels the whole visit's crops --
     a solo visit is one animal, so the visit is the honest labelling unit);
  3. the newly-confirmed visit becomes another template, so the next suggestion is sharper.
     That's the whole learning mechanism: no training, just accumulating verified prototypes
     (nearest-visit beats per-individual centroids because the same animal looks different
     night-to-night -- similarity is bimodal, and a centroid would blur the high mode away).

Multi-animal visits (the pair): frames holding two separated raccoon boxes mark the visit
"2+ raccoons". Its blended prototype never becomes a template, and the badge itself is signal --
co-arrival is a behaviour fingerprint (PLAN.md: surface appearance and behaviour separately).

Two guards protect the human label set -- the only irreplaceable thing in this project (2026-08-05
evaluation, docs/identity-eval-2026-08-05.md, phases A2 and C1):
  * SOURCE GUARD -- a visit is only ever ranked, fitted or auto-named against templates confirmed
    on the SAME camera. Cross-camera appearance is not a weak signal, it is no signal (best
    similarity across the entire 397x93 matrix: 0.363), while 83.7% of trail-cam visit PAIRS
    already clear the novelty cut -- so one cross-camera name that sticks would let refit() offer
    ~83 more visits under it. Plain source equality, because Matt moves cameras on purpose and a
    hand-measured zone would fail silently the day one moves.
  * TEMPLATE FLOOR -- auto_assign refuses to WRITE a name backed by fewer than
    cfg.reid_auto_min_templates confirmed solo visits. The cast is Stan 48 / Notch 46 / Pedro 36
    then Elliot 4 / CutiePie 3 / The Dude 1; the tier has already machine-named visits off a
    single template, and no threshold swept on one visit is a measurement.
Both are refusals, never re-rankings: what the matcher believes is unchanged, it just declines to
spend the label set on it.

Cold start, before anything is confirmed: --bootstrap clusters the unconfirmed visit prototypes
into a handful of visit-groups (5-8 on real data, vs the 319 useless crop-level clusters) so the
first round of naming is "name these groups", not "name 1653 crops".

  python individuals.py --queue              # recent unconfirmed visits + their suggestions
  python individuals.py --visit 1062         # one visit's full suggestion read-out
  python individuals.py --bootstrap          # cold-start visit-groups for first naming
  python individuals.py --confirm 1014 Stan  # CLI confirm (the dashboard is the usual way)
  python individuals.py --auto-assign        # name the unambiguous visits (the nightly batch's
                                             # "review by exception" pass; --dry-run to preview)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

import config
import db
import reidutil

EMBED_MODEL = "megadescriptor-l-384"


# ---------------------------------------------------------------------------
# Pure geometry/vector helpers (unit-tested; no DB, no model).
# ---------------------------------------------------------------------------

def iou(a, b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes. Two simultaneous raccoon boxes
    with LOW IoU are two bodies; high IoU is the detector double-boxing one animal."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def prototype(vectors: np.ndarray, qualities, top_k: int) -> np.ndarray:
    """A visit's appearance template: mean of its `top_k` highest-quality crops' (L2-normalized)
    vectors, re-normalized. Averaging over the best crops is what lifts the cross-session signal
    out of pose/burst noise -- the core finding this module is built on."""
    order = np.argsort([-(q if q is not None else 0.0) for q in qualities])[:top_k]
    p = vectors[order].mean(axis=0)
    n = np.linalg.norm(p)
    return p / n if n else p


def rank_templates(proto: np.ndarray, templates) -> list:
    """Rank confirmed individuals by their NEAREST confirmed visit's similarity to `proto`.
    `templates` = [(name, visit_id, vector), ...]. Returns [(name, sim, via_visit_id), ...] best
    first, one row per individual. Nearest-visit, not centroid: the same animal's visits split
    into look-modes (similarity is bimodal), and max() keys on the matching mode."""
    best: dict = {}
    for name, vid, vec in templates:
        s = float(proto @ vec)
        if name not in best or s > best[name][0]:
            best[name] = (s, vid)
    return sorted(((n, s, v) for n, (s, v) in best.items()), key=lambda r: -r[1])


def clip_match(vec: np.ndarray, templates: dict, threshold: float) -> list:
    """Rank CLIP-space templates {name: (centroid, n_tracklets)} by cosine to `vec`, keeping only
    matches >= `threshold`. Returns [(name, sim, n_tracklets), ...] best first. Used both ways:
    vec = an un-blend cluster centroid (clip<->clip: "this separated animal looks like Elliot"),
    and vec = a visit's still prototype (cross-space: flag a clip-templated individual in a new
    still visit). Its own threshold because clip vectors sit lower than the still prototypes."""
    ranked = sorted(((n, float(vec @ c), k) for n, (c, k) in templates.items()), key=lambda r: -r[1])
    return [(n, s, k) for n, s, k in ranked if s >= threshold]


def co_present_frames(rows, iou_max: float = 0.45) -> int:
    """How many timestamps in `rows` ([(timestamp, (x1,y1,x2,y2)), ...], one species) hold >= 2
    plausibly-separate boxes (pairwise IoU < iou_max). This is the "2+ raccoons at once" badge.
    The cut is empirical (2026-06-11, the Notch-pair corpus): the detector's double-boxes of ONE
    animal sit at IoU 0.5-1.0, while two huddled-but-distinct raccoons mostly land below 0.45.
    It is a HINT for the human eye, not an oracle -- a tightly-stacked pair can still hide above
    the cut (visits 1058/1063 did), and a stray double-box can sneak under it."""
    by_ts = defaultdict(list)
    for ts, box in rows:
        by_ts[ts].append(box)
    n = 0
    for boxes in by_ts.values():
        if len(boxes) < 2:
            continue
        if any(iou(boxes[i], boxes[j]) < iou_max
               for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
            n += 1
    return n


def _parse_ts(ts):
    """ISO timestamp -> datetime, or None for anything unparseable (a malformed import row must
    not abort a whole visit's tracklet build)."""
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _norm_centre_dist(a, b) -> float:
    """Centre-to-centre distance between two boxes in units of their mean diagonal -- a
    scale-free "how many body-lengths did it move" gate for chaining stills. A close-up animal
    (huge box) may travel hundreds of pixels between saved crops and still be continuous; a
    distant one (small box) moving the same pixels has crossed the yard."""
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    da = ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5
    db_ = ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5
    diag = (da + db_) / 2.0
    if diag <= 0:
        return float("inf")
    return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / diag


def still_tracklets(rows, *, link_gap_s: float = 20.0, iou_max: float = 0.45):
    """Chain one species' still detections into per-animal TRACKLETS -- clipmotion's job, done on
    the sparse saved-crop stream, so a visit can be separated even when it has no usable clips
    (trail-cam photo cycles, pruned footage, the recorder off). `rows` =
    [(det_id, iso_timestamp, (x1, y1, x2, y2)), ...]. Returns (tracks, cannot): `tracks` =
    [[det_id, ...], ...] in first-appearance order, and `cannot` = {(track_i, track_j), ...} --
    pairs of tracks observed in the SAME frame, which are therefore DIFFERENT animals. That
    constraint is free and absolute (one body cannot be in two places), and it's what keeps
    lookalike littermate kits apart downstream where appearance alone can't.

    Three steps, each deliberately conservative -- a fragmented track re-merges by appearance in
    the clustering step, but a chain that runs across two animals poisons its prototype and
    nothing downstream can split it again:
      1. same-frame boxes at pairwise IoU >= `iou_max` merge into ONE observation (the detector
         double-boxing one animal -- co_present_frames' empirical 0.45 cut, used in reverse);
      2. an observation joins the best-scoring active track whose last box it continues (IoU, or
         centre distance within about one body length) across a gap of at most `link_gap_s`
         seconds (validated 2026-07-31: the position jump that separated Stan's ground track
         from a kit's wall-frames fails the distance gate at ~1.5 body lengths);
      3. each track takes at most ONE observation per frame (structural, not scored) -- which is
         exactly what makes the cannot-link pairs trustworthy."""
    frames = defaultdict(list)
    for det_id, ts, box in rows:
        frames[ts].append((det_id, box))
    order = sorted(((t, members) for ts, members in frames.items()
                    if (t := _parse_ts(ts)) is not None), key=lambda x: x[0])

    tracks = []          # each: {"ids": [...], "box": last box, "t": last-seen datetime}
    cannot = set()
    for t, members in order:
        # 1. Merge double-boxes into observations (greedy connected grouping on IoU >= iou_max).
        obs = []
        for det_id, box in members:
            for o in obs:
                if iou(o["box"], box) >= iou_max:
                    o["ids"].append(det_id)
                    o["box"] = (min(o["box"][0], box[0]), min(o["box"][1], box[1]),
                                max(o["box"][2], box[2]), max(o["box"][3], box[3]))
                    break
            else:
                obs.append({"ids": [det_id], "box": tuple(box)})
        # 2. Score every (active track, observation) pair that passes the continuity gate.
        cands = []
        for ti, tr in enumerate(tracks):
            dt = (t - tr["t"]).total_seconds()
            if dt < 0 or dt > link_gap_s:
                continue
            for oi, o in enumerate(obs):
                i = iou(tr["box"], o["box"])
                nd = _norm_centre_dist(tr["box"], o["box"])
                if i >= 0.05 or nd <= 1.0:
                    # Overlap-continuations outrank pure distance links.
                    cands.append((i + max(0.0, 1.0 - nd), ti, oi))
        # 3. Greedy assignment, each track and each observation used at most once this frame.
        cands.sort(key=lambda c: -c[0])
        used_t, used_o = set(), set()
        placed = []                      # track index of every observation in this frame
        for _, ti, oi in cands:
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            tracks[ti]["ids"].extend(obs[oi]["ids"])
            tracks[ti]["box"] = obs[oi]["box"]
            tracks[ti]["t"] = t
            placed.append(ti)
        for oi, o in enumerate(obs):
            if oi not in used_o:
                placed.append(len(tracks))
                tracks.append({"ids": list(o["ids"]), "box": o["box"], "t": t})
        # Co-present observations pin their tracks apart for good.
        placed.sort()
        for a in range(len(placed)):
            for b in range(a + 1, len(placed)):
                cannot.add((placed[a], placed[b]))
    return [tr["ids"] for tr in tracks], cannot


def pose_groups(conn, individual_id: str, *, distance: float = 0.35, max_crops: int = 400,
                min_group: int = 3, cfg=None) -> list:
    """Cluster ONE confirmed individual's crops by appearance embedding. With identity held fixed,
    the embedding's remaining variation is POSE / viewpoint / lighting -- so the clusters are the
    animal's characteristic poses (the dish-hunch, the up-on-hind-legs alert, the tail-on exit).
    This is the very pose+background binding that DEFEATED cross-individual re-ID (see reid.py),
    used in reverse: a within-individual signal instead of a cross-individual confound. Returns
    [{n, rep_crops, first, last, cohesion}, ...] biggest pose-group first. Clustering is O(n^2),
    so only the `max_crops` most readable crops are used (poses read clearest on sharp crops)."""
    cfg = cfg or config.CONFIG
    rows = conn.execute(
        """SELECT d.crop_path, d.crop_quality, d.timestamp, e.embedding
           FROM detections d JOIN detection_embeddings e
             ON e.detection_id = d.id AND e.model = ?
           WHERE d.individual_id = ? AND d.crop_path IS NOT NULL""",
        (EMBED_MODEL, individual_id)).fetchall()
    if len(rows) < max(min_group, 2):
        return []
    rows = sorted(rows, key=lambda r: -(r["crop_quality"] or 0))[:max_crops]
    X = np.stack([reidutil.decode_vector(r["embedding"]) for r in rows])
    labels = reidutil.cluster_cosine(X, threshold=distance)
    groups = defaultdict(list)
    for i, l in enumerate(labels):
        groups[l].append(i)
    out = []
    for idxs in groups.values():
        if len(idxs) < min_group:
            continue
        coh = reidutil.mean_pairwise_cosine(X[idxs])
        idxs.sort(key=lambda i: -(rows[i]["crop_quality"] or 0))
        ts = sorted(rows[i]["timestamp"] for i in idxs)
        out.append({"n": len(idxs), "cohesion": round(coh, 2),
                    "rep_crops": [rows[i]["crop_path"].replace("\\", "/") for i in idxs[:8]],
                    "first": ts[0], "last": ts[-1]})
    out.sort(key=lambda g: -g["n"])
    return out


def _iso_overlap(a0, a1, b0, b1) -> bool:
    """True if the ISO-timestamp spans [a0,a1] and [b0,b1] overlap at all."""
    try:
        return (datetime.fromisoformat(a0) <= datetime.fromisoformat(b1)
                and datetime.fromisoformat(b0) <= datetime.fromisoformat(a1))
    except (ValueError, TypeError):
        return False


def _clip_template_vectors(conn, solo_visits: dict) -> dict:
    """{(clip_source, name): [vector, ...]} -- the attribution pass behind clip_templates(). Kept
    KEYED BY SOURCE so a caller can honour the source guard without re-reading every embedding
    (VisitMatcher folds this once into an all-source view plus one view per camera)."""
    rows = db.load_clip_track_embeddings(conn, EMBED_MODEL)
    if not rows:
        return {}
    vmeta = {v["id"]: v for v in conn.execute(
        "SELECT id, source, started_at, ended_at FROM visits")}
    groups: dict = defaultdict(list)
    for r in rows:
        name = r["individual_id"]
        if not name:                                   # not explicitly labelled -> try solo overlap
            for vid, nm in solo_visits.items():
                v = vmeta.get(vid)
                if v and v["source"] == r["clip_source"] and _iso_overlap(
                        r["clip_started_at"], r["clip_ended_at"] or r["clip_started_at"],
                        v["started_at"], v["ended_at"]):
                    name = nm
                    break
        if name:
            groups[(r["clip_source"], name)].append(reidutil.decode_vector(r["embedding"]))
    return dict(groups)


def _fold_clip_templates(groups: dict, source=None) -> dict:
    """{name: (centroid, n_tracklets)} from _clip_template_vectors' output. `source` restricts to
    tracklets recorded on ONE camera -- the SOURCE GUARD (see VisitMatcher.templates)."""
    per_name: dict = defaultdict(list)
    for (src, name), vecs in groups.items():
        if source is not None and src != source:
            continue
        per_name[name].extend(vecs)
    out = {}
    for name, vecs in per_name.items():
        c = np.stack(vecs).mean(axis=0)
        nrm = np.linalg.norm(c)
        out[name] = (c / nrm if nrm else c, len(vecs))
    return out


def clip_templates(conn, solo_visits: dict, *, source=None, cfg=None) -> dict:
    """{name: (centroid, n_tracklets)} -- a per-individual CLIP-space appearance template = the
    mean (re-normalized) of that individual's labelled tracklet prototypes. A tracklet is
    attributed to an individual if EITHER it carries an explicit un-blend label
    (clip_tracks.individual_id), OR its clip overlaps a confirmed SOLO visit of that individual
    (`solo_visits` = {visit_id: name}). So the lone resident (Stan) gets a clip template for free,
    while the never-solo pair member (Elliot) gets one the moment its cluster is un-blend-labelled
    -- which is the whole point: it's how Elliot becomes findable. Clip space is its own regime,
    matched with cfg.reid_clip_match_threshold, never the still novelty cut.

    `source` (optional) keeps only tracklets recorded on that camera -- the source guard, for
    callers ranking against a probe from a known source."""
    return _fold_clip_templates(_clip_template_vectors(conn, solo_visits), source)


def unblend_visit(conn, visit_id: int, *, distance: float = 0.45, templates: dict = None,
                  elim_templates: dict = None, cfg=None) -> dict:
    """Separate a multi-animal visit into its individuals using the CLIP TRACKLETS (which track
    each animal independently). Clusters the visit's tracklet appearance prototypes and returns
    the groups biggest-first -- in a true pair visit the two dominant clusters are the two animals
    (validated: the peak Notch+Elliot visit split 36/29). Each group carries representative
    frame-crops (clipembed rep_crop) so the eye can tell which is which, and any individual label
    already on its tracklets. The caller labels each cluster (db.set_clip_track_individual), which
    is how the pair member who is never solo finally gets a clean appearance template.

    CO-PRESENCE SEED: if a human logged who was visiting (live_sightings overlapping this visit),
    the two biggest clusters carry that logged pair as `co_names` (one-click assign, no typing), and
    when the appearance signal resolves ONE side, the OTHER is named by ELIMINATION (`co_elim`) --
    e.g. a cluster matches Notch's template, you logged "Notch + Elliot", so the other cluster is
    Elliot. That's how a never-solo pair member gets named from co-presence alone, before he has any
    template of his own. `templates` drives the displayed `suggestion` (explicit clip-labels only,
    by design); `elim_templates` (richer: solo-attributed + explicit) is used ONLY to break the tie
    between the two logged names -- safe because the choice is constrained to that human-attested pair.

    Tracklets are matched to the visit by clip<->visit time overlap on the same source. Returns
    {visit_id, n_tracklets, groups:[{track_ids, n, cohesion, rep_crops, label, suggestion, co_names,
    co_elim}], co_present:{names, observed_at}, note}."""
    cfg = cfg or config.CONFIG
    thr = cfg.reid_clip_match_threshold
    v = conn.execute("SELECT source, started_at, ended_at FROM visits WHERE id = ?",
                     (int(visit_id),)).fetchone()
    if v is None:
        return {"visit_id": visit_id, "groups": [], "n_tracklets": 0, "note": "no such visit"}
    src, vs, ve = v["source"], v["started_at"], v["ended_at"]
    rows = [r for r in db.load_clip_track_embeddings(conn, EMBED_MODEL)
            if r["clip_source"] == src and _iso_overlap(
                r["clip_started_at"], r["clip_ended_at"] or r["clip_started_at"], vs, ve)]
    out = {"visit_id": visit_id, "n_tracklets": len(rows), "groups": [], "note": None,
           "co_present": db.co_present_sighting_names(conn, src, vs, ve)}
    if len(rows) < 2:
        out["note"] = ("no embedded tracklets for this visit yet -- run clipmotion.py then "
                       "clipembed.py" if not rows else "only one tracklet -- nothing to separate")
        return out
    X = np.stack([reidutil.decode_vector(r["embedding"]) for r in rows])
    S = X @ X.T
    D = np.clip(1.0 - S, 0.0, None)
    np.fill_diagonal(D, 0.0)
    labels = reidutil.cluster_cosine(dist=D, threshold=distance)
    groups = defaultdict(list)
    for i, l in enumerate(labels):
        groups[l].append(i)
    built = []          # (group_dict, normalized_centroid) -- centroid kept for elimination below.
    for idxs in groups.values():
        sub = X[idxs]
        coh = reidutil.mean_pairwise_cosine(sub) if len(idxs) > 1 else 1.0
        # Who does this separated animal look like? Match the cluster's centroid against the known
        # clip-space templates (clip<->clip) so each group comes pre-suggested -- confirm in a click.
        cc = sub.mean(axis=0)
        nrm = np.linalg.norm(cc)
        ccn = cc / nrm if nrm else cc
        suggestion = clip_match(ccn, templates, thr) if templates else []
        idxs.sort(key=lambda i: -rows[i]["n_hits"])           # sturdiest tracklets first
        labelled = {rows[i]["individual_id"] for i in idxs if rows[i]["individual_id"]}
        built.append(({
            "suggestion": [{"name": n, "similarity": round(s, 3)} for n, s, _ in suggestion[:3]],
            "track_ids": [rows[i]["track_id"] for i in idxs],
            "n": len(idxs), "cohesion": round(coh, 2),
            "rep_crops": [rows[i]["rep_crop"] for i in idxs if rows[i]["rep_crop"]][:8],
            "label": sorted(labelled)[0] if len(labelled) == 1 else None,
        }, ccn))
    built.sort(key=lambda gc: -gc[0]["n"])
    out["groups"] = [gd for gd, _ in built]
    _seed_unblend_from_co_presence(out["co_present"]["names"], built,
                                   elim_templates or templates or {}, thr)
    return out


def _seed_unblend_from_co_presence(co_names, built, elim_templates, thr) -> None:
    """Fold a human co-presence log into the two biggest un-blend clusters (mutates `built`'s group
    dicts). Attaches the logged pair as `co_names` (quick-pick), then tries ELIMINATION: match each
    of the two clusters against `elim_templates` RESTRICTED to the logged names; if exactly one side
    resolves, the other gets the leftover name (`co_elim`). Restricting to the logged pair is what
    makes using the coarser solo-attributed templates safe here -- it's a 2-way choice the human
    already vouched for, not an open guess across the whole cast. No-op unless >=2 names and >=2
    clusters; ambiguous matches (both sides land on the same name) fall back to the quick-pick."""
    if len(co_names) < 2 or len(built) < 2:
        return
    top = built[:2]
    for gd, _ in top:
        gd["co_names"] = list(co_names)
    restricted = {n: elim_templates[n] for n in co_names if n in elim_templates}
    if not restricted:
        return                                       # cold start: no template yet -> quick-pick only
    picks = []
    for _, ccn in top:
        m = clip_match(ccn, restricted, thr)
        picks.append(m[0][0] if m else None)         # best logged-name match for this cluster, or None
    m0, m1 = picks
    assign = {}
    if m0 and m1 and m0 != m1:                        # both sides resolved by appearance
        assign = {0: m0, 1: m1}
    elif m0 and not m1:                               # one side resolved -> the other by elimination
        other = [n for n in co_names if n != m0]
        if len(other) == 1:
            assign = {0: m0, 1: other[0]}
    elif m1 and not m0:
        other = [n for n in co_names if n != m1]
        if len(other) == 1:
            assign = {0: other[0], 1: m1}
    for gi, nm in assign.items():
        top[gi][0]["co_elim"] = nm


def unblend_visit_stills(conn, visit_id: int, *, distance: float = 0.45, templates=None,
                         cfg=None) -> dict:
    """The STILLS basis for un-blend: separate a multi-animal visit using tracklets chained from
    its saved crops (still_tracklets), for visits the clip basis can't serve -- trail-cam photo
    cycles, pruned footage, the recorder off, or simply too few clip tracklets. Same group shape
    as unblend_visit, with `detection_ids` in place of `track_ids`: labelling a stills group
    stamps individual_id onto the crops directly (db.set_individual_bulk, source='human'), which
    survives visit renumbering the same way every detection-level label does.

    Validated at corpus scale 2026-07-31 (n=492 same-frame raccoon pairs vs n=2319 adjacent-frame
    pairs): two co-present raccoons score median 0.19 cosine at crop level while one animal
    frame-to-frame scores 0.72 -- so tracklet mini-prototypes separate cleanly, and the
    same-frame CANNOT-LINK pairs hold the split even where appearance is ambiguous (lookalike
    littermates). Crops are loaded with NO confidence gate: the second animal is nearly always
    the low-confidence box (2,423 of 2,500 unembedded co-present pair sides sat under the 0.5
    embed gate), so vectorless crops still take part structurally -- they chain, they constrain,
    they get stamped -- they just don't vote on appearance.

    `templates` = VisitMatcher.templates() rows: the human-confirmed SOLO visits. That's the
    still-space explicit set, so ONE template tier suffices where the clip basis needs two
    (there's no coarse solo-attribution layer to keep out of open suggestions). Suggestions gate
    at cfg.reid_track_match_threshold and are EXCLUSIVE across groups -- a visit's two animals
    can't both be Stan. Returns {visit_id, basis: 'stills', n_tracklets, groups:
    [{detection_ids, n, n_crops, cohesion, rep_crops, label, suggestion, co_names, co_elim}],
    co_present, note}."""
    cfg = cfg or config.CONFIG
    thr = cfg.reid_track_match_threshold
    v = conn.execute("SELECT source, species, started_at, ended_at FROM visits WHERE id = ?",
                     (int(visit_id),)).fetchone()
    out = {"visit_id": visit_id, "basis": "stills", "n_tracklets": 0, "groups": [], "note": None,
           "co_present": {"names": [], "observed_at": None, "n": 0}}
    if v is None:
        out["note"] = "no such visit"
        return out
    out["co_present"] = db.co_present_sighting_names(conn, v["source"], v["started_at"],
                                                     v["ended_at"])
    # SOURCE GUARD: a template confirmed on ANOTHER camera may not rank here (or seed a
    # co-presence elimination). Same rule, same reason as VisitMatcher.templates -- and a
    # template whose visit can't be resolved to a source is dropped too, failing closed.
    if templates:
        tsrc = dict(conn.execute("SELECT id, source FROM visits"))
        templates = [t for t in templates if tsrc.get(t[1]) == v["source"]]
    sp = v["species"]
    if sp is None:       # species-less visit: scope to its dominant species (apply_visit_label's rule)
        r = conn.execute(
            "SELECT species FROM detections WHERE visit_id = ? AND species IS NOT NULL "
            "GROUP BY species ORDER BY COUNT(*) DESC LIMIT 1", (int(visit_id),)).fetchone()
        sp = r[0] if r else None
    where = "d.visit_id = ?" + ("" if sp is None else " AND d.species = ?")
    params = [int(visit_id)] + ([] if sp is None else [sp])
    rows = conn.execute(
        f"""SELECT d.id, d.timestamp, d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2,
                   d.crop_quality, d.crop_path, d.individual_id, e.embedding
            FROM detections d LEFT JOIN detection_embeddings e
              ON e.detection_id = d.id AND e.model = ?
            WHERE {where} ORDER BY d.timestamp""", [EMBED_MODEL] + params).fetchall()
    if not rows:
        out["note"] = "no detections on this visit"
        return out
    byid = {r["id"]: r for r in rows}
    tracks, cannot = still_tracklets(
        [(r["id"], r["timestamp"], (r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]))
         for r in rows], link_gap_s=cfg.reid_track_link_gap_s)
    out["n_tracklets"] = len(tracks)
    if len(tracks) < 2:
        out["note"] = "only one animal-track in the stills -- nothing to separate"
        return out
    protos = []
    for ids in tracks:
        vecs = [reidutil.decode_vector(byid[i]["embedding"]) for i in ids
                if byid[i]["embedding"] is not None]
        quals = [byid[i]["crop_quality"] for i in ids if byid[i]["embedding"] is not None]
        protos.append(prototype(np.stack(vecs), quals, cfg.reid_proto_top_k) if vecs else None)
    # Cluster the tracklets that can vote on appearance, under the same-frame cannot-links. A
    # vectorless tracklet (every crop below the embed gate -- run embed.py --co-present) stays
    # its own group: honest, and the eye can still name it from the rep crops.
    vt = [i for i, p in enumerate(protos) if p is not None]
    pos = {t: k for k, t in enumerate(vt)}
    clusters = []
    if len(vt) >= 2:
        P = np.stack([protos[i] for i in vt])
        S = P @ P.T
        D = np.clip(1.0 - S, 0.0, None)
        np.fill_diagonal(D, 0.0)
        cl = {(pos[a], pos[b]) for a, b in cannot if a in pos and b in pos}
        labels = reidutil.cluster_cosine_constrained(D, distance, cl)
        grouped = defaultdict(list)
        for k, l in zip(vt, labels):
            grouped[l].append(k)
        clusters = list(grouped.values())
    elif vt:
        clusters = [[vt[0]]]
    clusters += [[i] for i, p in enumerate(protos) if p is None]

    built = []           # (group_dict, normalized centroid or None), like unblend_visit's shape
    for tidx in clusters:
        det_ids = [i for t in tidx for i in tracks[t]]
        members = sorted((byid[i] for i in det_ids), key=lambda m: -(m["crop_quality"] or 0))
        pvecs = [protos[t] for t in tidx if protos[t] is not None]
        coh = reidutil.mean_pairwise_cosine(np.stack(pvecs)) if len(pvecs) > 1 else 1.0
        cen = None
        if pvecs:
            c = np.stack(pvecs).mean(axis=0)
            nrm = np.linalg.norm(c)
            cen = c / nrm if nrm else c
        labelled = {m["individual_id"] for m in members if m["individual_id"]}
        built.append(({
            "detection_ids": [int(i) for i in det_ids],
            "n": len(tidx), "n_crops": len(det_ids), "cohesion": round(coh, 2),
            "rep_crops": [m["crop_path"].replace("\\", "/") for m in members
                          if m["crop_path"]][:8],
            "label": sorted(labelled)[0] if len(labelled) == 1 else None,
            "suggestion": [],
        }, cen))
    built.sort(key=lambda gc: -gc[0]["n_crops"])

    # Suggestions vs the confirmed-visit still templates, EXCLUSIVE across groups: rank each
    # cluster, then let the strongest claim win each name -- the other groups fall through to
    # their next candidate. Two animals in one visit cannot share an identity.
    if templates:
        ranked = [([(n, s) for n, s, _v in rank_templates(cen, templates) if s >= thr][:5]
                   if cen is not None else []) for _gd, cen in built]
        taken: dict = {}
        for s, gi, n in sorted(((s, gi, n) for gi, lst in enumerate(ranked) for n, s in lst),
                               key=lambda x: -x[0]):
            if gi in taken or n in taken.values():
                continue
            taken[gi] = n
        for gi, (gd, _cen) in enumerate(built):
            gd["suggestion"] = [{"name": n, "similarity": round(s, 3)} for n, s in ranked[gi]
                                if taken.get(gi) == n or n not in taken.values()][:3]
    out["groups"] = [gd for gd, _ in built]
    _seed_stills_from_co_presence(out["co_present"]["names"], built, templates or [], thr)
    return out


def _seed_stills_from_co_presence(co_names, built, templates, thr) -> None:
    """The N-way generalization of _seed_unblend_from_co_presence, for the stills basis (a mother
    with three kits is four logged names, not a pair). Names get CLAIMED in three rounds, each
    with uniqueness (two co-present clusters can never share a name):
      1. a cluster already LABELLED with a logged name has claimed it -- the 2026-07-31 kit visit
         opened with Stan's live-stamped ground track as a labelled group, and offering "+ Stan"
         on the kit groups would invite exactly the mislabel this module exists to prevent;
      2. appearance resolves what it can, RESTRICTED to the logged names (a choice the human
         already vouched for), greedily by similarity;
      3. when exactly ONE name and ONE cluster remain, ELIMINATION closes it -- which, thanks to
         round 1, works even with no templates at all (label + log, the cold-start case).
    Each cluster's `co_names` quick-picks then exclude names claimed elsewhere, and a resolved
    unlabelled cluster carries its name as `co_elim` (the dashboard's starred from-your-log
    pick). With "Stan + 3 kits" logged and only Stan resolved: the three kit groups keep three
    kit quick-picks -- until two are named, when the last closes by elimination. Mutates
    `built`'s group dicts; no-op without 2+ names."""
    if len(co_names) < 2 or not built:
        return
    top = built[:min(len(co_names), len(built))]
    logged = list(co_names)
    # Round 1: existing labels claim their logged name.
    assign: dict = {}
    for gi, (gd, _cen) in enumerate(top):
        if gd["label"] in set(logged) and gd["label"] not in assign.values():
            assign[gi] = gd["label"]
    # Round 2: appearance, restricted to the logged names, greedy-unique by similarity.
    restricted = [t for t in (templates or []) if t[0] in set(logged)]
    if restricted:
        picks = []
        for gi, (_gd, cen) in enumerate(top):
            if gi in assign or cen is None:
                continue
            m = rank_templates(cen, restricted)
            if m and m[0][1] >= thr:
                picks.append((m[0][1], gi, m[0][0]))
        for _s, gi, n in sorted(picks, key=lambda x: -x[0]):
            if gi in assign or n in assign.values():
                continue
            assign[gi] = n
    # Round 3: one name, one cluster left -> elimination.
    left_names = [n for n in logged if n not in assign.values()]
    left_groups = [gi for gi in range(len(top)) if gi not in assign]
    if len(left_names) == 1 and len(left_groups) == 1:
        assign[left_groups[0]] = left_names[0]
    for gi, (gd, _cen) in enumerate(top):
        claimed_elsewhere = {n for g2, n in assign.items() if g2 != gi}
        gd["co_names"] = [n for n in logged if n not in claimed_elsewhere]
        if gi in assign and not gd["label"]:
            gd["co_elim"] = assign[gi]


def clips_for_individual(conn, individual_id: str, species: str = "raccoon", limit: int = 24,
                         cfg=None) -> list:
    """Behaviour clips attributable to one confirmed individual: clips that overlap (in time, same
    source) a visit labelled `individual_id`. Newest first. Each clip carries `multi=True` when its
    visit holds 2+ animals -- then the footage shows the individual ALONGSIDE another, so it's
    watch-with-care, not clean solo footage. (A clip has no own identity; it inherits the visit's.)
    This is the 'watch Stan move' view -- and the substrate for later clip-appearance clustering."""
    vrows = conn.execute(
        "SELECT id, source, started_at, ended_at FROM visits WHERE species = ? AND individual_id = ?",
        (species, individual_id)).fetchall()
    if not vrows:
        return []
    multi = clip_co_presence_by_visit(conn, species)
    multi_min = cfg.reid_clip_co_presence_min_clips if cfg is not None else 2
    # Playback surface: soft-pruned clips (video gone, row kept for its derived data) are
    # excluded here, while clip_co_presence_by_visit above deliberately keeps them -- the
    # co-presence SIGNAL outlives the footage.
    clips = conn.execute(
        "SELECT id, source, clip_path, started_at, ended_at, frame_count, fps FROM clips "
        "WHERE pruned_at IS NULL").fetchall()
    out = []
    for c in clips:
        for v in vrows:
            if v["source"] == c["source"] and _iso_overlap(
                    c["started_at"], c["ended_at"] or c["started_at"],
                    v["started_at"], v["ended_at"]):
                dur = None
                try:
                    dur = (datetime.fromisoformat(c["ended_at"])
                           - datetime.fromisoformat(c["started_at"])).total_seconds()
                except (ValueError, TypeError):
                    if c["frame_count"] and c["fps"]:
                        dur = c["frame_count"] / c["fps"]
                out.append({"clip_id": c["id"],
                            "clip_path": (c["clip_path"] or "").replace("\\", "/"),
                            "started_at": c["started_at"],
                            "duration_s": round(dur, 1) if dur else None,
                            "visit_id": v["id"], "multi": multi.get(v["id"], 0) >= multi_min})
                break
    out.sort(key=lambda r: r["started_at"], reverse=True)
    return out[:limit]


def clip_co_presence_by_visit(conn, species: str) -> dict:
    """{visit_id: n_clips} -- how many of each visit's clips show >= 2 SUSTAINED motion tracklets
    (two animals moving at once). This is the clip-based multi-animal signal: the full-frame-rate
    clips catch a second raccoon in moments the sparse live stills miss. A clip is attributed to
    the `species` visit it overlaps in time on the same source. "Sustained" (clipmotion's
    SUSTAINED_HITS) excludes the detection-dropout fragments that would otherwise masquerade as a
    second animal. Returns {} when no tracks exist yet (lazy clipmotion import avoids a cycle)."""
    from clipmotion import SUSTAINED_HITS
    clip_rows = conn.execute(
        """SELECT c.id, c.source, c.started_at, c.ended_at,
                  SUM(CASE WHEN t.n_hits >= ? THEN 1 ELSE 0 END) AS n_sustained
           FROM clips c JOIN clip_tracks t ON t.clip_id = c.id
           GROUP BY c.id HAVING n_sustained >= 2""", (SUSTAINED_HITS,)).fetchall()
    if not clip_rows:
        return {}
    visits = conn.execute(
        "SELECT id, source, started_at, ended_at FROM visits WHERE species = ?",
        (species,)).fetchall()
    out: dict = defaultdict(int)
    for c in clip_rows:
        for v in visits:
            if v["source"] == c["source"] and _iso_overlap(
                    c["started_at"], c["ended_at"] or c["started_at"],
                    v["started_at"], v["ended_at"]):
                out[v["id"]] += 1
                break
    return dict(out)


def multi_name_sighting_spans(conn) -> list:
    """(source, span_start, span_end) for every live sighting whose names testify to multiple
    animals -- direct human testimony that a span held several bodies, the third multi-visit
    signal beside the stills badge and clip co-presence. A kit convoy can enter the sparse stills
    one animal at a time (zero same-instant frames) and still be four animals; the 2026-07-31
    "Stan + 3 kits" span did exactly that. Two forms count: 2+ names logged together, and ONE
    name that is itself a group label ("Stan + Kits", db.is_group_label) -- the family-stamp
    convention, which before 2026-08-08 read as solo here and let a whole family night masquerade
    as one animal. A sighting with no span falls back to its observed_at instant (the
    db.co_present_sighting_names convention). Empty on a DB no writer has migrated yet."""
    try:
        rows = conn.execute(
            "SELECT source, observed_at, span_start, span_end, names FROM live_sightings"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        src, observed, s0, s1, names_json = tuple(r)
        try:
            names = json.loads(names_json) if names_json else []
        except ValueError:
            names = []
        if len(names) >= 2 or any(db.is_group_label(n) for n in names):
            out.append((src, s0 or observed, s1 or observed))
    return out


def co_present_visit_ids(conn, *, iou_max: float = 0.45, min_frames: int = 1,
                         include_sightings: bool = True) -> set:
    """Visit ids that plausibly hold 2+ animals -- the target list for embed.py's --co-present
    widening pass. GENEROUS on purpose (min_frames=1, vs the badge's reid_co_presence_min=3):
    a false positive costs a few extra low-confidence embeddings, a miss leaves a kit with no
    vectors and the splitter blind. Counts same-species same-frame box pairs at IoU < iou_max
    per visit; with `include_sightings`, adds visits overlapped by a multi-name live sighting
    (the human SAW two+, whatever the stills caught)."""
    per_visit: dict = defaultdict(lambda: defaultdict(list))   # visit -> (ts, species) -> boxes
    for r in conn.execute(
            """SELECT visit_id, timestamp, species, bbox_x1, bbox_y1, bbox_x2, bbox_y2
               FROM detections WHERE visit_id IS NOT NULL AND species IS NOT NULL"""):
        per_visit[r["visit_id"]][(r["timestamp"], r["species"])].append(
            (r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]))
    out = set()
    for vid, frames in per_visit.items():
        n = 0
        for boxes in frames.values():
            if len(boxes) < 2:
                continue
            if any(iou(boxes[i], boxes[j]) < iou_max
                   for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
                n += 1
                if n >= min_frames:
                    out.add(vid)
                    break
    if include_sightings:
        spans = multi_name_sighting_spans(conn)
        if spans:
            for v in conn.execute("SELECT id, source, started_at, ended_at FROM visits"):
                if v["id"] in out:
                    continue
                if any(src == v["source"] and _iso_overlap(s0, s1, v["started_at"], v["ended_at"])
                       for src, s0, s1 in spans):
                    out.add(v["id"])
    return out


# ---------------------------------------------------------------------------
# The matcher: visit prototypes + confirmed templates, loaded once per request.
# ---------------------------------------------------------------------------

# Sentinel for VisitMatcher.templates(): "every source", which is NOT the same as source=None
# ("visits whose source is unknown"). Anything that ranks must pass a real source.
_ALL_SOURCES = object()


class VisitMatcher:
    """Loads one species' embedded crops grouped by visit, builds prototypes, and answers
    "who does visit N look like?" against the human-confirmed visits. Read-only over the DB."""

    def __init__(self, conn, species: str, cfg=None):
        self.cfg = cfg or config.CONFIG
        self.species = species
        rows = conn.execute(
            """SELECT d.id, d.visit_id, d.timestamp, d.crop_quality, d.confidence,
                      d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2, e.embedding
               FROM detections d JOIN detection_embeddings e
                 ON e.detection_id = d.id AND e.model = ?
               WHERE d.species = ? AND d.visit_id IS NOT NULL AND d.confidence >= ?""",
            (EMBED_MODEL, species, self.cfg.reid_suggest_min_conf)).fetchall()

        self._by_visit: dict = defaultdict(list)
        vecs: dict = defaultdict(list)
        for r in rows:
            self._by_visit[r["visit_id"]].append(r)
            vecs[r["visit_id"]].append(reidutil.decode_vector(r["embedding"]))

        # Co-presence reads ALL of the species' boxes, with NO confidence gate -- the second
        # animal of a huddled pair is usually a low-confidence, occluded box that never gets an
        # embedding, and missing it is exactly how a pair visit would masquerade as solo. The
        # same-timestamp + IoU<0.3 requirement keeps spurious low-conf boxes from counting.
        self._co_rows: dict = defaultdict(list)
        for r in conn.execute(
                """SELECT visit_id, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2
                   FROM detections WHERE species = ? AND visit_id IS NOT NULL""", (species,)):
            self._co_rows[r["visit_id"]].append(
                (r["timestamp"], (r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"])))

        self.protos: dict = {}
        self.visit_started: dict = {}
        for vid, members in self._by_visit.items():
            if len(members) < self.cfg.reid_proto_min_crops:
                continue
            X = np.stack(vecs[vid])
            qs = [m["crop_quality"] for m in members]
            self.protos[vid] = prototype(X, qs, self.cfg.reid_proto_top_k)
            self.visit_started[vid] = min(m["timestamp"] for m in members)

        self.confirmed = db.confirmed_visit_labels(conn, species)   # {visit_id: name}
        self.auto = db.visit_labels_by_source(conn, "auto", species)  # nightly auto-assigned names
        self.rejected = db.rejected_visit_ids(conn, species)  # human said "leave unnamed" -- hands off
        # THE ROSTER: {casefolded name: last day resident (or None)} for individuals the human has
        # marked departed. Loaded here so auto_assign can consult it; deliberately NOT consulted by
        # templates(), suggest(), refit() or any ranking -- see is_departed().
        self.departed = db.departed_individuals(conn)
        self._co_cache: dict = {}
        self.clip_co_presence = clip_co_presence_by_visit(conn, species)  # {visit_id: n_clips}
        # THIRD multi-animal signal: the human's own live log. A kit convoy can enter the sparse
        # stills one animal at a time (zero same-instant frames) and still be four animals -- the
        # 2026-07-31 "Stan + 3 kits" span did exactly that. A sighting logging 2+ names over a
        # visit is direct testimony, stronger than either detector-side badge.
        self._visit_meta = {v["id"]: (v["source"], v["started_at"], v["ended_at"])
                            for v in conn.execute(
                                "SELECT id, source, started_at, ended_at FROM visits")}
        # SOURCE GUARD (2026-08-05 eval, phase C1): which camera each visit came from. Every
        # ranking below is scoped to ONE source -- see templates().
        self.visit_source = {vid: m[0] for vid, m in self._visit_meta.items()}
        self._multi_sightings = multi_name_sighting_spans(conn)
        self._sighting_cache: dict = {}
        # CLIP-space templates: an individual's labelled tracklets, the only way a never-solo pair
        # member (Elliot) gets matched. Built from confirmed SOLO visits + explicit un-blend labels.
        # Group labels ("Stan + Kits") are refused here for the same reason templates() refuses
        # them: the tracklets of a family visit belong to several animals under one name.
        solo = {vid: nm for vid, nm in self.confirmed.items()
                if not self.is_multi(vid) and not db.is_group_label(nm)}
        _clip_groups = _clip_template_vectors(conn, solo)
        # The all-source fold stays as the public attribute (web.py's un-blend elimination pool);
        # matching a still prototype uses the per-source fold via clip_templates_for().
        self.clip_templates = _fold_clip_templates(_clip_groups)
        self._clip_templates_src = {s: _fold_clip_templates(_clip_groups, s)
                                    for s in set(self.visit_source.values())}

    def co_presence(self, visit_id: int) -> int:
        """Frames in this visit with two separated boxes of this species (cached). Counted over
        ALL detections, not just embedded ones -- see the loader comment."""
        if visit_id not in self._co_cache:
            self._co_cache[visit_id] = co_present_frames(self._co_rows.get(visit_id, ()))
        return self._co_cache[visit_id]

    def sighting_multi(self, visit_id: int) -> bool:
        """True when a human live-logged 2+ names over a span overlapping this visit (cached)."""
        if visit_id not in self._sighting_cache:
            meta = self._visit_meta.get(visit_id)
            hit = False
            if meta and self._multi_sightings:
                src, s0, s1 = meta
                hit = any(src == m[0] and _iso_overlap(m[1], m[2], s0, s1)
                          for m in self._multi_sightings)
            self._sighting_cache[visit_id] = hit
        return self._sighting_cache[visit_id]

    def is_multi(self, visit_id: int) -> bool:
        """A visit holds 2+ animals if ANY signal fires: the detection badge (simultaneous
        separated boxes in the sparse live stills), clip co-presence (>= min_clips clips with
        two sustained tracklets -- the full-rate clips catch pairs the stills miss), or a
        multi-name LIVE SIGHTING overlapping the visit (the human watched 2+ animals arrive,
        even if they never shared a saved frame)."""
        return (self.co_presence(visit_id) >= self.cfg.reid_co_presence_min
                or self.clip_co_presence.get(visit_id, 0)
                >= self.cfg.reid_clip_co_presence_min_clips
                or self.sighting_multi(visit_id))

    def is_departed(self, name, started) -> bool:
        """Has the human recorded that `name` had already left the yard by the time a visit
        starting at `started` happened? A DATE comparison, not a blanket exclusion.

        Why this exists: one of this cast's raccoons stopped appearing on 2026-06-30 and never
        came back, but its 46 templates did not stop existing -- so at the recommended operating
        point the auto tier lined up to write that name onto two visits on 2026-07-03, three days
        after the animal was last here. That error is invisible to every metric the project has,
        because leave-one-visit-out scores against labels and a departed animal's labels simply
        stop: the probe set can never contain the case "named an animal that no longer lives
        here". So the fact comes from the human, who knows it, via db.set_individual_status.

        Deliberately narrow, in both directions:
          * a visit that started ON OR BEFORE the effective date is NOT blocked -- it happened
            while the animal was resident, and those are exactly the visits still worth naming
            (the unreviewed backlog reaches back well before any departure);
          * only auto_assign asks. Ranking, suggestions, templates and every cast surface treat a
            departed individual exactly as before, because a human may look at an old visit and
            recognise it. This guard governs what the MACHINE writes, nothing else.

        Fails closed on the two cases where residency cannot be established: no effective date
        recorded, or a visit with no start timestamp."""
        if not self.departed:
            return False
        key = str(name or "").strip().casefold()
        if key not in self.departed:
            return False
        last_day, back_on = self.departed[key]
        if not last_day or not started:
            return True                       # departed, but nothing to compare -> never auto-write
        day = str(started)[:10]
        if day <= str(last_day)[:10]:
            return False                      # happened while the animal was still resident
        # ...and if it came BACK, the absence is an interval, not everything-after. Notch was away
        # 2026-06-24 -> 2026-08-06; without this his real August visits would be refused forever
        # because of a June departure, which is the opposite of what the guard is for.
        if back_on and day >= str(back_on)[:10]:
            return False
        return True

    def templates(self, source=_ALL_SOURCES) -> list:
        """(name, visit_id, prototype) for every confirmed SOLO visit with a usable prototype.
        Multi-animal visits are excluded -- their prototype blends two animals. GROUP-labelled
        visits ("Stan + Kits", db.is_group_label) are excluded BY NAME as well as by evidence:
        a family span whose stills happened to catch one body at a time carries no co-presence
        badge, but the human's own label says several animals -- and a blended family prototype
        competing as a pseudo-individual is exactly the contamination the 5-template auto floor
        would then be counting toward.

        THE SOURCE GUARD. Pass a `source` and only that camera's confirmed visits come back; the
        default returns every source, which is the corpus-wide view ("does ANY template exist?",
        the cast surfaces) and must never be handed to a ranker. Measured 2026-08-05: across the
        whole 397x93 cross-source matrix the best similarity is 0.363, so scoping costs nothing
        today -- but 83.7% of trail-cam visit PAIRS already clear the 0.31 novelty cut, so the
        first cross-camera name that sticks would let refit() propose ~83 of the other 92
        trail-cam visits under it. A source-equality test is the one form of this guard that
        survives Matt moving a camera: no geometry, no per-source thresholds, no camera list.
        `source=None` means "visits whose source is unknown" (fail closed), not "any"."""
        rows = [(name, vid, self.protos[vid]) for vid, name in self.confirmed.items()
                if vid in self.protos and not self.is_multi(vid)
                and not db.is_group_label(name)]
        if source is _ALL_SOURCES:
            return rows
        return [t for t in rows if self.visit_source.get(t[1]) == source]

    def templates_for(self, visit_id: int) -> list:
        """The ONLY template pool a probe from `visit_id` may be ranked against: same source."""
        return self.templates(self.visit_source.get(visit_id))

    def clip_templates_for(self, visit_id: int) -> dict:
        """Same guard for the CLIP-space templates: tracklets recorded on this visit's camera."""
        return self._clip_templates_src.get(self.visit_source.get(visit_id), {})

    def suggest(self, visit_id: int) -> dict:
        """The suggestion read-out for one visit: ranked candidates, novelty flag, co-presence.
        Degrades honestly -- explains WHY there's no suggestion when there isn't one."""
        out = {"visit_id": visit_id,
               "source": self.visit_source.get(visit_id),
               "n_embedded": len(self._by_visit.get(visit_id, ())),
               "co_present_frames": self.co_presence(visit_id),
               "co_present_clips": self.clip_co_presence.get(visit_id, 0),
               "co_present_sighting": self.sighting_multi(visit_id),
               "multi": self.is_multi(visit_id),
               "confirmed_as": self.confirmed.get(visit_id),
               "auto_as": self.auto.get(visit_id),
               "candidates": [], "clip_candidates": [], "novel": False, "note": None}
        if visit_id not in self.protos:
            out["note"] = ("no embedded crops yet -- run: python embed.py --min-confidence 0.5"
                           if not out["n_embedded"] else
                           f"only {out['n_embedded']} embedded crop(s) -- too thin to match")
            return out
        # Cross-space clip match (still prototype vs each individual's clip-template centroid): an
        # ADDITIONAL signal that can name a clip-templated individual -- including one with no still
        # template at all (Elliot). Surfaced separately from the still-based candidates.
        cm = clip_match(self.protos[visit_id], self.clip_templates_for(visit_id),
                        self.cfg.reid_clip_match_threshold)
        out["clip_candidates"] = [{"name": n, "similarity": round(s, 3), "n_tracklets": k}
                                  for n, s, k in cm[:3]]
        # SOURCE GUARD: rank only against templates confirmed on THIS visit's camera.
        temps = self.templates_for(visit_id)
        if not temps:
            if self.templates():           # templates exist, just not on this camera
                out["note"] = (
                    f"no confirmed {out['source']} visit yet -- identity is never matched across "
                    f"cameras (the best cross-camera similarity ever measured on this corpus is "
                    f"0.363, which is a fact about the cameras, not the animal). Confirm a "
                    f"{out['source']} visit to start this camera's own template set.")
                return out
            out["note"] = ("clip-template match only (no still templates yet)"
                           if out["clip_candidates"] else
                           "no confirmed individuals yet -- name a visit (or bootstrap groups) first")
            return out
        ranked = rank_templates(self.protos[visit_id], temps)
        out["candidates"] = [
            {"name": n, "similarity": round(s, 3), "via_visit": v,
             "via_started": self.visit_started.get(v)} for n, s, v in ranked[:3]]
        out["novel"] = ranked[0][1] < self.cfg.reid_novel_threshold
        if out["multi"]:
            ev = []
            if out["co_present_frames"]:
                ev.append(f"{out['co_present_frames']} still frame(s)")
            if out["co_present_clips"]:
                ev.append(f"{out['co_present_clips']} clip(s)")
            if out["co_present_sighting"]:
                ev.append("your live log")
            out["note"] = (f"2+ raccoons here ({' + '.join(ev)} show two at once) -- the "
                           f"suggestion is a blend; name the visit by its main animal, or skip")
        return out

    def _cluster_visits(self, vids: list, distance: float) -> list:
        """Agglomerative-cluster the given visits by prototype cosine distance into groups
        [{visits, started, cohesion, multi}, ...] biggest first. Shared by bootstrap_groups
        (cluster everything unconfirmed) and refit (cluster only the novel residual)."""
        vids = sorted(vids)
        if not vids:
            return []
        if len(vids) == 1:
            return [{"visits": vids, "cohesion": 1.0,
                     "started": [self.visit_started[vids[0]]],
                     "multi": [self.is_multi(vids[0])]}]
        P = np.stack([self.protos[v] for v in vids])
        S = P @ P.T
        D = np.clip(1.0 - S, 0.0, None)
        np.fill_diagonal(D, 0.0)
        labels = reidutil.cluster_cosine(dist=D, threshold=distance)
        groups = defaultdict(list)
        for v, l in zip(vids, labels):
            groups[l].append(v)
        out = []
        for vs in groups.values():
            ii = [vids.index(v) for v in vs]
            coh = reidutil.mean_pairwise_cosine(P[ii]) if len(vs) > 1 else 1.0
            out.append({"visits": sorted(vs), "cohesion": round(coh, 2),
                        "started": [self.visit_started[v] for v in sorted(vs)],
                        "multi": [self.is_multi(v) for v in sorted(vs)]})
        out.sort(key=lambda g: -len(g["visits"]))
        return out

    def bootstrap_groups(self, distance: float = 0.45) -> list:
        """Cold start: agglomerative-cluster the UNCONFIRMED visit prototypes so the first naming
        pass works on a handful of visit-groups (biggest first). Multi-animal visits are grouped
        too but flagged (name them last)."""
        return self._cluster_visits([v for v in self.protos if v not in self.confirmed], distance)

    def refit(self, distance: float = 0.45) -> dict:
        """Re-fit the corpus to the confirmed cast: assign every UNCONFIRMED solo visit to its
        nearest confirmed individual when the match clears the novelty threshold, and cluster the
        residual (everything that looks like nobody on file) into candidate NEW individuals. This
        is "now that I've named a few, sort the rest" -- it compounds: each confirmation grows a
        template, so the next refit pulls more visits out of 'novel'.

        Also reports `untemplated`: confirmed individuals with NO usable solo template because
        they were only named on multi-animal visits (their prototype blends two animals). Those
        individuals CANNOT be matched until a solo visit is confirmed for them -- surfacing that
        is the difference between a silent miss and an actionable nudge."""
        templated = {n for n, _, _ in self.templates()}
        untemplated = sorted(set(self.confirmed.values()) - templated)
        fits: dict = {}
        novel = []
        by_source: dict = {}
        for v in self.protos:
            # Auto-assigned visits are already handled (their card shows the auto chip; undo
            # returns them here) -- listing them again as "fits" would double-surface them.
            if v in self.confirmed or v in self.auto or self.is_multi(v):
                continue
            # SOURCE GUARD: a visit is only ever fitted to its OWN camera's templates. This is the
            # phase-C1 hazard in one line -- without it, one confirmed cross-camera name lets this
            # loop propose the whole of the other camera's corpus under it.
            src = self.visit_source.get(v)
            if src not in by_source:
                by_source[src] = self.templates(src)
            temps = by_source[src]
            ranked = rank_templates(self.protos[v], temps) if temps else []
            if ranked and ranked[0][1] >= self.cfg.reid_novel_threshold:
                name, sim, via = ranked[0]
                fits.setdefault(name, []).append(
                    {"visit_id": v, "similarity": round(sim, 3),
                     "started": self.visit_started.get(v), "via_visit": via})
            else:
                novel.append(v)
        for lst in fits.values():
            lst.sort(key=lambda r: -r["similarity"])
        # The residual clusters WITHIN a source too: a candidate-new-individual group spanning two
        # cameras invites one name onto both, which is the same mislabel by another door.
        novel_groups = []
        for src in sorted({self.visit_source.get(v) for v in novel}, key=lambda s: (s is None, s)):
            novel_groups += self._cluster_visits(
                [v for v in novel if self.visit_source.get(v) == src], distance)
        novel_groups.sort(key=lambda g: -len(g["visits"]))
        return {"fits": fits, "novel_groups": novel_groups,
                "untemplated": untemplated,
                "n_fit": sum(len(l) for l in fits.values()), "n_novel": len(novel)}

    def auto_assign(self, conn, *, threshold: float = None, margin: float = None,
                    min_templates: int = None, dry_run: bool = False) -> dict:
        """The "review by exception" tier: NAME the unambiguous solo visits automatically, so the
        human reviews what the machine did instead of confirming everything by hand. A visit is
        auto-named only when its best match clears BOTH bars: nearest-confirmed-visit similarity
        >= `threshold` AND lead over the runner-up INDIVIDUAL >= `margin` (both from eval.py's
        auto-assign sweep -- the measured zero-wrong-assignment operating point, cfg defaults).

        Guardrails, each load-bearing:
          - stamps individual_source='auto' (db.label_visit): visible on tracking surfaces, but
            NEVER a suggestion template (confirmed_visit_labels is human-only) -- a wrong auto
            name can't teach the matcher;
          - solo visits only (a pair visit's blended prototype names two animals at once);
          - skips human-confirmed, already-auto-named, and human-REJECTED visits (the reject
            tombstone is what makes an undo stick across nightly runs);
          - one-individual casts still need the full margin over an empty runner-up (0.0), so a
            lone template can't vacuum up everything above the threshold;
          - SOURCE GUARD (2026-08-05): a visit is only ranked against its OWN camera's templates,
            and a camera with no confirmed visit of its own gets no auto names at all
            (`no_same_source_template`). See templates() for the measurement;
          - TEMPLATE FLOOR (2026-08-05): a name backed by fewer than `min_templates` confirmed
            SOLO visits ON THIS SOURCE is never written (`thin_templates`). Measured: the shipped
            corpus is Stan 48 / Notch 46 / Pedro 36 / Elliot 4 / CutiePie 3 / The Dude 1, and this
            tier has already auto-named a CutiePie visit and a The Dude visit off ONE template
            each. Nothing swept on a 1-4 visit individual is a measurement, so the tier must not
            spend the human label set on it. The thin individual still COMPETES in the ranking --
            it can still block an assignment as the runner-up -- it just can't be written;
          - THE ROSTER (2026-08-05): a name a human has marked DEPARTED is never written onto a
            visit that started after the departure date (`departed`). Templates outlive the
            animal: this cast's Notch stopped appearing on 2026-06-30 and the tier still lined up
            to name two 2026-07-03 visits after him. Note what this is NOT -- it is not a recency
            gate on template age, which was implemented, measured and rejected (wrong-name rate
            0.119 -> 0.137 for LESS coverage). It is a fact the human supplies and the machine
            cannot infer, and it is a DATE test: visits from before the departure are still
            auto-nameable, and the departed individual is still ranked, suggested and templated
            everywhere else. See is_departed().

        Returns {enabled, assigned: [{visit_id, name, similarity, margin, started}], skipped}."""
        cfg = self.cfg
        threshold = cfg.reid_auto_threshold if threshold is None else threshold
        margin = cfg.reid_auto_margin if margin is None else margin
        min_templates = (cfg.reid_auto_min_templates if min_templates is None else min_templates)
        if not threshold or threshold <= 0:
            return {"enabled": False, "assigned": [], "skipped": {},
                    "note": "disabled -- set reid_auto_threshold from eval.py --reid's sweep"}
        out = {"enabled": True, "threshold": threshold, "margin": margin,
               "min_templates": min_templates,
               "assigned": [], "skipped": defaultdict(int)}
        if not self.templates():
            out["skipped"]["no_templates"] = len(self.protos)
            out["skipped"] = dict(out["skipped"])
            return out
        by_source: dict = {}
        for vid in sorted(self.protos):
            if vid in self.confirmed:
                out["skipped"]["confirmed"] += 1
                continue
            if vid in self.auto:
                out["skipped"]["already_auto"] += 1
                continue
            if vid in self.rejected:
                out["skipped"]["human_rejected"] += 1
                continue
            if self.is_multi(vid):
                out["skipped"]["multi_animal"] += 1
                continue
            src = self.visit_source.get(vid)
            if src not in by_source:
                by_source[src] = self.templates(src)
            temps = by_source[src]
            if not temps:
                out["skipped"]["no_same_source_template"] += 1
                continue
            ranked = rank_templates(self.protos[vid], temps)
            name, sim, _via = ranked[0]
            lead = sim - (ranked[1][1] if len(ranked) > 1 else 0.0)
            if sim < threshold:
                out["skipped"]["below_threshold"] += 1
                continue
            if lead < margin:
                out["skipped"]["ambiguous"] += 1          # near-tie between two individuals
                continue
            # The floor is checked LAST on purpose: `thin_templates` then counts exactly the
            # visits this guard took off the assignment list, not every visit whose top match
            # happened to be a thin individual -- so a dry run reads as "the floor cost you N".
            if sum(1 for n, _v, _p in temps if n == name) < min_templates:
                out["skipped"]["thin_templates"] += 1
                continue
            # THE ROSTER, checked last for the same reason: `departed` counts exactly the names
            # this guard refused to write. The visit is SKIPPED, not re-awarded to the runner-up --
            # the margin was measured against the departed individual, so handing the name down
            # the ranking would write a name no bar was ever cleared for.
            if self.is_departed(name, self.visit_started.get(vid)):
                out["skipped"]["departed"] += 1
                continue
            if not dry_run:
                db.label_visit(conn, vid, name, source="auto")
            out["assigned"].append({"visit_id": vid, "name": name,
                                    "similarity": round(sim, 3), "margin": round(lead, 3),
                                    "started": self.visit_started.get(vid)})
        out["skipped"] = dict(out["skipped"])
        return out


# ---------------------------------------------------------------------------
# CLI (the dashboard drives the same calls through web.py).
# ---------------------------------------------------------------------------

def _fmt_started(ts):
    try:
        return datetime.fromisoformat(ts).strftime("%m-%d %H:%M")
    except (ValueError, TypeError):
        return "?"


def _print_suggestion(s, matcher):
    started = _fmt_started(matcher.visit_started.get(s["visit_id"]))
    tag = "2+" if s["multi"] else ("  " if not s["novel"] else "??")
    head = f"{tag} visit #{s['visit_id']:<5} {started}  ({s['n_embedded']} crops)"
    if s["confirmed_as"]:
        print(f"{head}  = {s['confirmed_as']} (confirmed)")
        return
    if s["note"] and not s["candidates"]:
        print(f"{head}  {s['note']}")
        return
    cands = "  ".join(f"{c['name']} {c['similarity']:.2f}" for c in s["candidates"])
    novel = "  <- below threshold: possibly someone NEW" if s["novel"] else ""
    print(f"{head}  looks like: {cands}{novel}")
    if s["note"]:
        print(f"      note: {s['note']}")


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 3: suggest-confirm loop for individual ID.")
    p.add_argument("--species", default="raccoon", help="Species to suggest for (default raccoon).")
    p.add_argument("--queue", action="store_true", help="Recent unconfirmed visits + suggestions.")
    p.add_argument("--visit", type=int, default=None, help="Suggestion read-out for one visit id.")
    p.add_argument("--bootstrap", action="store_true",
                   help="Cold-start: cluster unconfirmed visits into nameable groups.")
    p.add_argument("--refit", action="store_true",
                   help="Re-fit unconfirmed visits to the confirmed cast + cluster the novel residual.")
    p.add_argument("--auto-assign", action="store_true",
                   help="Auto-name the unambiguous solo visits (individual_source='auto'; bars from "
                        "config reid_auto_threshold/margin, set via eval.py --reid's sweep). The "
                        "nightly batch runs this; auto names never feed the suggestion templates.")
    p.add_argument("--dry-run", action="store_true",
                   help="With --auto-assign: report what would be named, write nothing.")
    p.add_argument("--confirm", nargs=2, metavar=("VISIT_ID", "NAME"), default=None,
                   help="Confirm a visit's individual, e.g. --confirm 1014 Stan.")
    p.add_argument("--limit", type=int, default=25, help="Queue length (default 25).")
    args = p.parse_args()

    conn = db.connect(config.CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    try:
        if args.confirm is not None:
            vid, name = int(args.confirm[0]), args.confirm[1].strip()
            n = db.label_visit(conn, vid, name or None)
            print(f"Visit #{vid}: stamped individual_id='{name}' on {n} crop(s). "
                  f"It is now a suggestion template.")
            return 0

        matcher = VisitMatcher(conn, args.species)
        if args.visit is not None:
            _print_suggestion(matcher.suggest(args.visit), matcher)
            return 0

        if args.auto_assign:
            r = matcher.auto_assign(conn, dry_run=args.dry_run)
            if not r["enabled"]:
                print(f"Auto-assign is DISABLED ({r['note']}).")
                return 0
            verb = "would name" if args.dry_run else "named"
            print(f"Auto-assign ({args.species}, similarity >= {r['threshold']:.2f}, "
                  f"margin >= {r['margin']:.2f}, >= {r['min_templates']} template(s) per name, "
                  f"same camera only): {verb} {len(r['assigned'])} visit(s).")
            for a in r["assigned"]:
                print(f"  visit #{a['visit_id']:<6} {_fmt_started(a['started'])}  "
                      f"{a['name']}  sim {a['similarity']:.2f}  lead {a['margin']:.2f}")
            if r["skipped"]:
                parts = ", ".join(f"{k} {v}" for k, v in sorted(r["skipped"].items()))
                print(f"  (skipped: {parts})")
            if not args.dry_run and r["assigned"]:
                print("  Review them in the dashboard queue -- [keep] promotes to a confirmed "
                      "template, [not them] clears and pins the visit against re-naming.")
            return 0

        if args.refit:
            r = matcher.refit()
            if not matcher.templates():
                print("No usable templates yet -- confirm a SOLO visit first "
                      "(python individuals.py --bootstrap).")
                return 0
            print(f"Re-fit of {r['n_fit'] + r['n_novel']} unconfirmed {args.species} visit(s) "
                  f"against the confirmed cast:\n")
            for name, lst in sorted(r["fits"].items(), key=lambda kv: -len(kv[1])):
                preview = ", ".join(f"#{x['visit_id']} {x['similarity']:.2f}" for x in lst[:6])
                print(f"  fits {name}: {len(lst)} visit(s)  "
                      f"[{preview}{' ...' if len(lst) > 6 else ''}]")
            print(f"\n  NOVEL (look like nobody on file): {r['n_novel']} visit(s) in "
                  f"{len(r['novel_groups'])} candidate-new-individual group(s):")
            for i, g in enumerate(r["novel_groups"], 1):
                span = f"{_fmt_started(g['started'][0])} .. {_fmt_started(g['started'][-1])}"
                print(f"    group {i}: {len(g['visits'])} visit(s)  cohesion {g['cohesion']:.2f}  {span}")
            if r["untemplated"]:
                print(f"\n  /!\\ confirmed but NOT matchable yet (named only on multi-animal "
                      f"visits -- blended template): {', '.join(r['untemplated'])}")
                print(f"      -> confirm a SOLO visit for each to let the system find them.")
            return 0

        if args.bootstrap:
            groups = matcher.bootstrap_groups()
            if not groups:
                print("Nothing to bootstrap -- no unconfirmed visits with embedded crops.")
                return 0
            print(f"{len(groups)} look-alike visit-group(s) among unconfirmed {args.species} "
                  f"visits (each is PROBABLY one individual -- your eye decides):")
            for i, g in enumerate(groups, 1):
                span = f"{_fmt_started(g['started'][0])} .. {_fmt_started(g['started'][-1])}"
                multi = sum(g["multi"])
                extra = f"  ({multi} multi-animal)" if multi else ""
                print(f"  group {i}: {len(g['visits'])} visit(s)  cohesion {g['cohesion']:.2f}  "
                      f"{span}{extra}")
                print(f"           visits: {' '.join('#' + str(v) for v in g['visits'])}")
            print("\nName one with: python individuals.py --confirm <visit_id> <Name> "
                  "(or use the dashboard's Individuals tab).")
            return 0

        # Default / --queue: recent visits of this species, unconfirmed first.
        rows = conn.execute(
            "SELECT id FROM visits WHERE species = ? ORDER BY started_at DESC LIMIT ?",
            (args.species, args.limit)).fetchall()
        if not rows:
            print(f"No {args.species} visits yet (run: python visits.py).")
            return 0
        print(f"Suggestions for the last {len(rows)} {args.species} visit(s)  "
              f"(?? = possibly new, 2+ = multiple animals):\n")
        for r in rows:
            _print_suggestion(matcher.suggest(r["id"]), matcher)
        n_t = len(matcher.templates())
        print(f"\n{len(matcher.confirmed)} confirmed visit(s) -> {n_t} usable template(s). "
              f"Every confirmation sharpens the next suggestion.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
