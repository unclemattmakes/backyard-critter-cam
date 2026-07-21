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

Cold start, before anything is confirmed: --bootstrap clusters the unconfirmed visit prototypes
into a handful of visit-groups (5-8 on real data, vs the 319 useless crop-level clusters) so the
first round of naming is "name these groups", not "name 1653 crops".

  python individuals.py --queue              # recent unconfirmed visits + their suggestions
  python individuals.py --visit 1062         # one visit's full suggestion read-out
  python individuals.py --bootstrap          # cold-start visit-groups for first naming
  python individuals.py --confirm 1014 Stan  # CLI confirm (the dashboard is the usual way)
"""
from __future__ import annotations

import argparse
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


def clip_templates(conn, solo_visits: dict, *, cfg=None) -> dict:
    """{name: (centroid, n_tracklets)} -- a per-individual CLIP-space appearance template = the
    mean (re-normalized) of that individual's labelled tracklet prototypes. A tracklet is
    attributed to an individual if EITHER it carries an explicit un-blend label
    (clip_tracks.individual_id), OR its clip overlaps a confirmed SOLO visit of that individual
    (`solo_visits` = {visit_id: name}). So the lone resident (Stan) gets a clip template for free,
    while the never-solo pair member (Elliot) gets one the moment its cluster is un-blend-labelled
    -- which is the whole point: it's how Elliot becomes findable. Clip space is its own regime,
    matched with cfg.reid_clip_match_threshold, never the still novelty cut."""
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
            groups[name].append(reidutil.decode_vector(r["embedding"]))
    out = {}
    for name, vecs in groups.items():
        c = np.stack(vecs).mean(axis=0)
        nrm = np.linalg.norm(c)
        out[name] = (c / nrm if nrm else c, len(vecs))
    return out


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


# ---------------------------------------------------------------------------
# The matcher: visit prototypes + confirmed templates, loaded once per request.
# ---------------------------------------------------------------------------

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
        self._co_cache: dict = {}
        self.clip_co_presence = clip_co_presence_by_visit(conn, species)  # {visit_id: n_clips}
        # CLIP-space templates: an individual's labelled tracklets, the only way a never-solo pair
        # member (Elliot) gets matched. Built from confirmed SOLO visits + explicit un-blend labels.
        solo = {vid: nm for vid, nm in self.confirmed.items() if not self.is_multi(vid)}
        self.clip_templates = clip_templates(conn, solo, cfg=self.cfg)

    def co_presence(self, visit_id: int) -> int:
        """Frames in this visit with two separated boxes of this species (cached). Counted over
        ALL detections, not just embedded ones -- see the loader comment."""
        if visit_id not in self._co_cache:
            self._co_cache[visit_id] = co_present_frames(self._co_rows.get(visit_id, ()))
        return self._co_cache[visit_id]

    def is_multi(self, visit_id: int) -> bool:
        """A visit holds 2+ animals if EITHER signal fires: the detection badge (simultaneous
        separated boxes in the sparse live stills) OR clip co-presence (>= min_clips clips with
        two sustained tracklets -- the full-rate clips catch pairs the stills miss)."""
        return (self.co_presence(visit_id) >= self.cfg.reid_co_presence_min
                or self.clip_co_presence.get(visit_id, 0)
                >= self.cfg.reid_clip_co_presence_min_clips)

    def templates(self) -> list:
        """(name, visit_id, prototype) for every confirmed SOLO visit with a usable prototype.
        Multi-animal visits are excluded -- their prototype blends two animals."""
        return [(name, vid, self.protos[vid]) for vid, name in self.confirmed.items()
                if vid in self.protos and not self.is_multi(vid)]

    def suggest(self, visit_id: int) -> dict:
        """The suggestion read-out for one visit: ranked candidates, novelty flag, co-presence.
        Degrades honestly -- explains WHY there's no suggestion when there isn't one."""
        out = {"visit_id": visit_id,
               "n_embedded": len(self._by_visit.get(visit_id, ())),
               "co_present_frames": self.co_presence(visit_id),
               "co_present_clips": self.clip_co_presence.get(visit_id, 0),
               "multi": self.is_multi(visit_id),
               "confirmed_as": self.confirmed.get(visit_id),
               "candidates": [], "clip_candidates": [], "novel": False, "note": None}
        if visit_id not in self.protos:
            out["note"] = ("no embedded crops yet -- run: python embed.py --min-confidence 0.5"
                           if not out["n_embedded"] else
                           f"only {out['n_embedded']} embedded crop(s) -- too thin to match")
            return out
        # Cross-space clip match (still prototype vs each individual's clip-template centroid): an
        # ADDITIONAL signal that can name a clip-templated individual -- including one with no still
        # template at all (Elliot). Surfaced separately from the still-based candidates.
        cm = clip_match(self.protos[visit_id], self.clip_templates, self.cfg.reid_clip_match_threshold)
        out["clip_candidates"] = [{"name": n, "similarity": round(s, 3), "n_tracklets": k}
                                  for n, s, k in cm[:3]]
        temps = self.templates()
        if not temps:
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
        temps = self.templates()
        templated = {n for n, _, _ in temps}
        untemplated = sorted(set(self.confirmed.values()) - templated)
        fits: dict = {}
        novel = []
        for v in self.protos:
            if v in self.confirmed or self.is_multi(v):
                continue
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
        return {"fits": fits, "novel_groups": self._cluster_visits(novel, distance),
                "untemplated": untemplated,
                "n_fit": sum(len(l) for l in fits.values()), "n_novel": len(novel)}


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
