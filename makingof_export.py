"""
Freeze a curated slice of the live yard into a STATIC, public, camera-free dataset for the
making-of page (making-of/). The live dashboard is localhost-only because it shows your camera;
this is the inverse -- a frozen, hand-curated, people-free export that can be hosted publicly with
no GPU, no server, and no live feed.

It reads backyard.db (READ-ONLY) + the saved crops/clips and writes:
    making-of/data/*.json     one file per demo, plus meta.json
    making-of/media/...        the handful of crops/frames each demo shows (downscaled)

Everything here is computed from STORED values (species, confidences, embeddings, visits,
behaviour profiles, suggestions, the un-blend split) -- no model is required to build the page.
The one optional enrichment is the non-animal gate's per-crop probability (--score), which loads
the same open_clip gate the live rig uses; without it the decoy demo still stands on the stored
BioCLIP verdicts.

PRIVACY: only detection_class='animal' crops are ever exported, and a name denylist drops any
person label (e.g. 'homeowner') and the non-critter labels. No full frames except wide frames
pulled from behaviour clips, which are yard-only by construction.

    python makingof_export.py            # build the whole dataset from the DB
    python makingof_export.py --score     # also score the non-animal gate (loads open_clip)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
import db
import reidutil
import behavior
import individuals
from individuals import EMBED_MODEL, VisitMatcher, prototype, rank_templates
from stats import _NON_CRITTER

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "making-of"
DATA = OUT / "data"
MEDIA = OUT / "media"

# Labels that must never reach the public page: the non-critter denylist + any person label.
PRIVACY_DENY = set(_NON_CRITTER) | {"homeowner", "person", "people", "human"}


# --------------------------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------------------------

def _conn():
    c = db.connect_readonly(config.CONFIG.db_path)
    if c is None:
        sys.exit(f"No database at {config.CONFIG.db_path} -- run the rig first.")
    return c


def _ensure_dirs():
    for d in (DATA, MEDIA, MEDIA / "crops", MEDIA / "frames", MEDIA / "thumbs"):
        d.mkdir(parents=True, exist_ok=True)


def _write_json(name: str, obj) -> None:
    path = DATA / name
    path.write_text(json.dumps(obj, separators=(",", ":")))
    print(f"  wrote data/{name}  ({path.stat().st_size/1024:.0f} KB)")


def copy_crop(rel_path, *, kind="crops", prefix="crop", key=None, size=None) -> str | None:
    """Copy one stored crop/frame into making-of/media, optionally downscaling (PIL). Returns the
    web path (relative to the page) or None if the source is missing. `key` makes a stable,
    timestamp-free filename so the public files don't leak capture times."""
    if not rel_path:
        return None
    src = db.crop_abspath(rel_path)
    if not src.exists():
        return None
    stem = f"{prefix}_{key if key is not None else Path(rel_path).stem}"
    dst = MEDIA / kind / f"{stem}.jpg"
    try:
        if size:
            from PIL import Image
            im = Image.open(src).convert("RGB")
            im.thumbnail((size, size))
            im.save(dst, "JPEG", quality=82)
        else:
            shutil.copyfile(src, dst)
    except Exception as e:
        print(f"    ! copy failed for {rel_path}: {e}")
        return None
    return f"media/{kind}/{stem}.jpg"


def _crop_path(conn, det_id):
    """crop_path for a detection id (the re-ID matcher rows don't carry it)."""
    r = conn.execute("SELECT crop_path FROM detections WHERE id=?", (det_id,)).fetchone()
    return r["crop_path"] if r else None


def _norm01(xy: np.ndarray) -> np.ndarray:
    """Scale an (n,2) array into the unit square, preserving aspect (pad the shorter axis)."""
    lo = xy.min(0)
    span = (xy.max(0) - lo)
    span[span == 0] = 1.0
    s = span.max()
    return (xy - lo) / s


def pca2(X: np.ndarray, mean=None, basis=None):
    """2-D PCA. Returns (coords, mean, basis) so prototypes can be projected through the SAME
    basis the crops were fit on."""
    if mean is None or basis is None:
        mean = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
        basis = Vt[:2]
    return (X - mean) @ basis.T, mean, basis


# --------------------------------------------------------------------------------------------
# meta.json -- the real headline numbers for chapter 06
# --------------------------------------------------------------------------------------------

def export_meta(conn) -> None:
    g = lambda s: conn.execute(s).fetchone()[0]
    days = conn.execute(
        "SELECT COUNT(DISTINCT substr(timestamp,1,10)) FROM detections").fetchone()[0]
    species_n = conn.execute(
        "SELECT COUNT(DISTINCT species) FROM detections WHERE species IS NOT NULL "
        "AND lower(species) NOT IN (%s)" % ",".join("?" * len(PRIVACY_DENY)),
        list(PRIVACY_DENY)).fetchone()[0]
    cast = conn.execute(
        "SELECT individual_id, COUNT(*) n FROM detections "
        "WHERE individual_source='human' AND individual_id IS NOT NULL "
        "GROUP BY individual_id ORDER BY n DESC").fetchall()
    meta = {
        "crops": g("SELECT COUNT(*) FROM detections"),
        "clips": g("SELECT COUNT(*) FROM clips"),
        "visits": g("SELECT COUNT(*) FROM visits"),
        "days": days,
        "species_named": species_n,
        "embeddings": g("SELECT COUNT(*) FROM detection_embeddings"),
        "first": conn.execute("SELECT MIN(timestamp) FROM detections").fetchone()[0],
        "last": conn.execute("SELECT MAX(timestamp) FROM detections").fetchone()[0],
        "cast": [{"name": r["individual_id"], "crops": r["n"]} for r in cast
                 if r["individual_id"].lower() not in PRIVACY_DENY],
    }
    _write_json("meta.json", meta)


# --------------------------------------------------------------------------------------------
# demo 1 -- pipeline scrubber: one real capture through every stage
# --------------------------------------------------------------------------------------------

def _extract_frames(clip_rel, bbox_norm):
    """Pull a wide 'arriving' frame + a MOG2 motion mask + the 'detected' frame out of a clip,
    exactly the gate the live rig runs. Returns dict of web paths, or None if cv2/the clip fail."""
    try:
        import cv2
    except Exception:
        return None
    src = db.crop_abspath(clip_rel)
    if not src.exists():
        return None
    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n < 4:
        cap.release()
        return None
    mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    empty_frame = None
    det_frame = None
    mask = None
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        m = mog.apply(frame)
        if i == 1:
            empty_frame = frame.copy()
        if i == int(n * 0.7):           # well into the clip: the animal is present
            det_frame = frame.copy()
            mask = m.copy()
            break
    cap.release()
    if det_frame is None:
        return None
    key = Path(clip_rel).stem
    out = {}
    fp = MEDIA / "frames"
    if empty_frame is not None:
        cv2.imwrite(str(fp / f"empty_{key}.jpg"), empty_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out["empty"] = f"media/frames/empty_{key}.jpg"
    cv2.imwrite(str(fp / f"frame_{key}.jpg"), det_frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
    out["frame"] = f"media/frames/frame_{key}.jpg"
    if mask is not None:
        cv2.imwrite(str(fp / f"mask_{key}.png"), mask)
        out["mask"] = f"media/frames/mask_{key}.png"
    out["w"], out["h"] = det_frame.shape[1], det_frame.shape[0]
    return out


def export_pipeline(conn) -> None:
    # Solo raccoon visits with a high-confidence, well-scored representative crop + an overlapping clip.
    rows = conn.execute(
        """SELECT v.id vid, v.individual_id vid_individual, v.started_at,
                  v.representative_detection_id rep, v.source
           FROM visits v
           WHERE v.species='raccoon' AND v.individual_id IN ('Stan','Notch')
             AND v.representative_detection_id IS NOT NULL
           ORDER BY v.detection_count DESC LIMIT 30""").fetchall()
    samples = []
    for v in rows:
        d = conn.execute(
            "SELECT id, timestamp, source, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
            "frame_w, frame_h, crop_path, crop_quality, species, species_confidence, "
            "species_source, individual_id, visit_id FROM detections WHERE id=?",
            (v["rep"],)).fetchone()
        if not d or not d["crop_path"]:
            continue
        bbox_norm = [d["bbox_x1"] / d["frame_w"], d["bbox_y1"] / d["frame_h"],
                     d["bbox_x2"] / d["frame_w"], d["bbox_y2"] / d["frame_h"]]
        # find an overlapping clip for the wide frame + motion mask
        clip = conn.execute(
            "SELECT clip_path, started_at, ended_at FROM clips WHERE source=? "
            "AND started_at <= ? AND COALESCE(ended_at, started_at) >= ? "
            "AND pruned_at IS NULL LIMIT 1",
            (d["source"], v["started_at"], v["started_at"])).fetchone()
        frames = _extract_frames(clip["clip_path"], bbox_norm) if clip else None
        if not frames:                       # the scrubber needs the wide frame + motion mask
            continue
        crop_web = copy_crop(d["crop_path"], kind="crops", prefix="pipe", key=d["id"], size=320)
        samples.append({
            "id": d["id"], "individual": v["vid_individual"], "species": d["species"],
            "species_confidence": round(d["species_confidence"] or 0, 3),
            "species_source": d["species_source"], "confidence": round(d["confidence"], 3),
            "crop_quality": round(d["crop_quality"] or 0, 3),
            "bbox": [round(x, 4) for x in bbox_norm],
            "timestamp": d["timestamp"], "crop": crop_web, "frames": frames,
        })
        if len([s for s in samples if s["frames"]]) >= 3:
            break
    _write_json("pipeline.json", {"samples": samples})


# --------------------------------------------------------------------------------------------
# demo 2 -- decoy bench: the species model can't say "not an animal"; the gate can
# --------------------------------------------------------------------------------------------

DECOY_PICKS = [
    ("raccoon", "animal", 3), ("American crow", "animal", 2),
    ("eastern gray squirrel", "animal", 1), ("Virginia opossum", "animal", 1),
    # 'brown rat' is BioCLIP's low-confidence catch-all, not true food: the gate rates these as
    # animals, so the demo treats them as "uncertain", separate from the genuine non-animals.
    ("brown rat", "uncertain", 3), ("not an animal", "nonanimal", 2),
]


def export_decoy(conn, score=False) -> None:
    tray, srcs = [], []
    for species, truth, k in DECOY_PICKS:
        rows = conn.execute(
            "SELECT id, crop_path, species, species_confidence, species_source "
            "FROM detections WHERE detection_class='animal' AND species=? "
            "AND crop_path IS NOT NULL AND COALESCE(species_verified,0)!=1 "
            "ORDER BY confidence DESC LIMIT ?", (species, k * 4)).fetchall()
        added = 0
        for r in rows:
            web = copy_crop(r["crop_path"], kind="crops", prefix="decoy", key=r["id"], size=300)
            if not web:
                continue
            tray.append({
                "id": r["id"], "truth": truth, "crop": web,
                "bioclip_species": r["species"],
                "bioclip_confidence": round(r["species_confidence"] or 0, 3),
                "bioclip_source": r["species_source"],
                "gate_p_nonanimal": None,   # filled by --score
            })
            srcs.append(r["crop_path"])
            added += 1
            if added >= k:
                break

    dist = None
    if score:
        dist = _score_gate(conn, tray, srcs)

    _write_json("decoy.json", {
        "tray": tray,
        "threshold": config.CONFIG.nonanimal_threshold,
        "gate_model": f"{config.CONFIG.nonanimal_model}/{config.CONFIG.nonanimal_pretrained}",
        "dist": dist,
        "scored": bool(score),
    })


def _score_gate(conn, tray, srcs):
    """OPTIONAL: run the real open_clip non-animal gate on the tray crops + a food/animal cohort to
    produce per-crop p_nonanimal and a threshold sweep. Loads a model -- guarded so a build without
    a GPU/weights still produces the rest of the dataset."""
    try:
        from clipfilter import AnimalFilter
        cfg = config.CONFIG
        af = AnimalFilter(cfg.nonanimal_model, cfg.nonanimal_pretrained, cfg.device,
                          cfg.nonanimal_threshold)
    except Exception as e:
        print(f"  ! gate scoring unavailable ({e}); leaving decoy gate scores empty.")
        return None
    paths = [str(db.crop_abspath(s)) for s in srcs]
    for t, (is_animal, p_non) in zip(tray, af.judge(paths)):
        t["gate_p_nonanimal"] = round(float(p_non), 3)
    # Two real p_nonanimal distributions, so the demo's threshold slider shows the actual
    # separation: confident real animals (should be ~0) vs the gate-rejected non-animals
    # (should be high). The 0.60 cut lives in the empty gap between the two clouds.
    def cohort(where, limit=160):
        rs = conn.execute(
            f"SELECT crop_path FROM detections WHERE {where} AND crop_path IS NOT NULL "
            "ORDER BY id LIMIT ?", (limit,)).fetchall()
        ps = [str(db.crop_abspath(r["crop_path"])) for r in rs
              if db.crop_abspath(r["crop_path"]).exists()]
        return [round(float(pn), 3) for _, pn in af.judge(ps)] if ps else []
    animals = cohort("detection_class='animal' AND confidence>=0.6 AND species IN "
                     "('raccoon','American crow','Virginia opossum','eastern gray squirrel')")
    nonanimals = cohort("species='not an animal'")
    return {"animal_pnon": animals, "nonanimal_pnon": nonanimals}


# --------------------------------------------------------------------------------------------
# demo 3 -- appearance map: 2-D projection of real MegaDescriptor embeddings
# --------------------------------------------------------------------------------------------

CAST = ["Stan", "Notch", "Elliot", "Miss B.", "Pepsi"]
MAP_VISITS_PER_IND = 7
MAP_CROPS_PER_VISIT = 6
MIN_PROTO_CROPS = 10        # thin visits make noisy prototypes; show their crops, skip their proto


def export_appearance(conn) -> None:
    # embedded, confirmed raccoon crops grouped by (individual, visit)
    rows = conn.execute(
        """SELECT d.id, d.individual_id, d.visit_id, d.crop_path, d.crop_quality, e.embedding
           FROM detections d JOIN detection_embeddings e
             ON e.detection_id=d.id AND e.model=?
           WHERE d.species='raccoon' AND d.individual_source='human'
             AND d.individual_id IS NOT NULL AND d.visit_id IS NOT NULL""",
        (EMBED_MODEL,)).fetchall()
    by_vis = defaultdict(list)
    for r in rows:
        if r["individual_id"] in CAST:
            by_vis[(r["individual_id"], r["visit_id"])].append(r)

    # pick a spread of visits per individual (largest first), capped crops per visit
    chosen = defaultdict(list)
    for (ind, vid), members in by_vis.items():
        chosen[ind].append((vid, members))
    points, vecs = [], []
    proto_pts, proto_vecs = [], []
    for ind in CAST:
        visits = sorted(chosen.get(ind, []), key=lambda kv: -len(kv[1]))[:MAP_VISITS_PER_IND]
        for vid, members in visits:
            members = sorted(members, key=lambda r: -(r["crop_quality"] or 0))
            V = np.stack([reidutil.decode_vector(m["embedding"]) for m in members])
            if len(members) >= MIN_PROTO_CROPS:    # only solid visits become prototypes
                proto = prototype(V, [m["crop_quality"] for m in members],
                                  config.CONFIG.reid_proto_top_k)
                proto_vecs.append(proto)
                proto_pts.append({"individual": ind, "visit": vid, "n": len(members)})
            for m in members[:MAP_CROPS_PER_VISIT]:
                web = copy_crop(m["crop_path"], kind="thumbs", prefix="emb", key=m["id"], size=72)
                if not web:
                    continue
                vecs.append(reidutil.decode_vector(m["embedding"]))
                points.append({"id": m["id"], "individual": ind, "visit": vid, "thumb": web})

    if not vecs:
        _write_json("appearance.json", {"points": [], "prototypes": []})
        return
    X = np.stack(vecs)
    coords, mean, basis = pca2(X)
    pcoords, _, _ = pca2(np.stack(proto_vecs), mean, basis)
    allc = _norm01(np.vstack([coords, pcoords]))
    cc, pc = allc[:len(coords)], allc[len(coords):]
    for p, xy in zip(points, cc):
        p["x"], p["y"] = round(float(xy[0]), 4), round(float(xy[1]), 4)
    for p, xy in zip(proto_pts, pc):
        p["x"], p["y"] = round(float(xy[0]), 4), round(float(xy[1]), 4)

    # quantized full cosine matrix among the displayed crops (exact click-two-dots similarity)
    S = (X @ X.T)
    n = len(points)
    tri = []
    for i in range(n):
        for j in range(i + 1, n):
            tri.append(int(round(float(S[i, j]) * 100)))

    P = np.stack(proto_vecs)
    featured = _featured_pairs(points, S, proto_pts, P)

    _write_json("appearance.json", {
        "points": points, "prototypes": proto_pts,
        "cos_tri": tri, "n_points": n, "featured": featured,
        "proto_cos": _proto_matrix(proto_pts, P),
    })


def _featured_pairs(points, S, proto_pts, P):
    """Three honest story pairs: a same-visit near-duplicate burst (~0.97), the strongest
    cross-session same-individual PROTOTYPE pair (the re-ID that actually works, ~0.85+), and a
    different-individual prototype pair (~0.2). Crop-space for the burst; prototype-space for the
    other two, because that's the unit the system matches on."""
    out = {}
    idx_by = defaultdict(list)
    for i, p in enumerate(points):
        idx_by[(p["individual"], p["visit"])].append(i)
    # same-visit: the MAX-cosine crop pair within any one visit (the near-duplicate burst)
    best = (-1, None, None)
    for idxs in idx_by.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                c = float(S[idxs[a], idxs[b]])
                if c > best[0]:
                    best = (c, idxs[a], idxs[b])
    if best[1] is not None:
        out["same_visit"] = {"a": best[1], "b": best[2], "cos": round(best[0], 2)}

    Sp = P @ P.T
    # cross-session same individual: strongest prototype pair from different visits, same individual
    cs = (-1, None, None)
    for i in range(len(proto_pts)):
        for j in range(i + 1, len(proto_pts)):
            if proto_pts[i]["individual"] == proto_pts[j]["individual"]:
                if float(Sp[i, j]) > cs[0]:
                    cs = (float(Sp[i, j]), i, j)
    if cs[1] is not None:
        out["cross_session"] = {"a": cs[1], "b": cs[2], "cos": round(cs[0], 2),
                                "individual": proto_pts[cs[1]]["individual"]}
    # different individuals: a representative cross-individual prototype pair (median-ish)
    diffs = [(float(Sp[i, j]), i, j) for i in range(len(proto_pts))
             for j in range(i + 1, len(proto_pts))
             if proto_pts[i]["individual"] != proto_pts[j]["individual"]]
    if diffs:
        diffs.sort()
        d = diffs[len(diffs) // 2]
        out["different"] = {"a": d[1], "b": d[2], "cos": round(d[0], 2),
                            "individuals": [proto_pts[d[1]]["individual"],
                                            proto_pts[d[2]]["individual"]]}
    return out


def _proto_matrix(proto_pts, P):
    """Pairwise cosine among the visit prototypes (to show the post-collapse separation)."""
    S = P @ P.T
    n = len(proto_pts)
    return [[int(round(float(S[i, j]) * 100)) for j in range(n)] for i in range(n)]


# --------------------------------------------------------------------------------------------
# demo 4 -- suggest-confirm game + the un-blend finale
# --------------------------------------------------------------------------------------------

def export_reid_game(conn) -> None:
    matcher = VisitMatcher(conn, "raccoon")
    templates = matcher.templates()                 # (name, vid, proto) confirmed solo visits
    rounds = []
    for vid, name in matcher.confirmed.items():
        if vid not in matcher.protos or matcher.is_multi(vid):
            continue
        # leave-this-visit-out so the model isn't just reading its own confirmation
        others = [t for t in templates if t[1] != vid]
        ranked = rank_templates(matcher.protos[vid], others)
        if not ranked:
            continue
        members = sorted(matcher._by_visit.get(vid, []),
                         key=lambda r: -(r["crop_quality"] or 0))[:5]
        crops = [c for c in (copy_crop(_crop_path(conn, m["id"]), kind="crops", prefix="game",
                                       key=m["id"], size=240) for m in members) if c]
        rounds.append({
            "visit": vid, "truth": name, "started": matcher.visit_started.get(vid),
            "n_crops": len(matcher._by_visit.get(vid, [])), "crops": crops,
            "candidates": [{"name": n, "sim": round(s, 2)} for n, s, _ in ranked[:3]],
            "novel": ranked[0][1] < config.CONFIG.reid_novel_threshold,
        })
    # variety: a few correct, and any where the top guess is wrong (the interesting ones)
    rounds.sort(key=lambda r: (r["candidates"][0]["name"] == r["truth"], -r["n_crops"]))
    rounds = [r for r in rounds if r["crops"]][:6]

    finale = _unblend_finale(conn, matcher)
    _write_json("reid_game.json", {"rounds": rounds, "finale": finale,
                                   "novel_threshold": config.CONFIG.reid_novel_threshold})


def _unblend_finale(conn, matcher):
    """Find a real multi-animal raccoon visit with >=2 embedded tracklets, run the live un-blend,
    and bake the actual cluster split. Prefers a visit where 2+ confirmed individuals are present
    (real ground truth) and where the split is clean. Each cluster keeps its clip-template
    suggestion (who the separated animal looks like) when one exists."""
    cand = [vid for vid in matcher.protos if matcher.is_multi(vid)]
    best = None
    for vid in cand:
        try:
            res = individuals.unblend_visit(conn, vid)
        except Exception:
            continue
        groups = res.get("groups") or []
        if len(groups) < 2 or groups[0]["n"] < 3 or groups[1]["n"] < 3:
            continue
        present = [r[0] for r in conn.execute(
            "SELECT DISTINCT individual_id FROM detections WHERE visit_id=? "
            "AND individual_source='human' AND individual_id IS NOT NULL", (vid,)).fetchall()
            if r[0] in CAST]
        # score: prefer 2+ known individuals present, then a balanced, sizeable split
        score = (1000 if len(present) >= 2 else 0) + min(groups[0]["n"], groups[1]["n"])
        if best is None or score > best["_score"]:
            glist = []
            for gi, g in enumerate(groups[:2]):
                reps = [c for c in (copy_crop(rc, kind="crops", prefix="ub",
                                              key=f"{vid}_{gi}_{i}", size=200)
                                    for i, rc in enumerate(g.get("rep_crops") or [])) if c]
                sugg = (g.get("suggestion") or [])
                glist.append({"n": g["n"], "cohesion": g["cohesion"], "rep_crops": reps[:4],
                              "label": g.get("label"),
                              "suggestion": sugg[0] if sugg else None})
            best = {"_score": score, "visit": vid, "groups": glist,
                    "present": present, "n_tracklets": res.get("n_tracklets")}
    if best:
        best.pop("_score", None)
    return best


# --------------------------------------------------------------------------------------------
# demo 5 -- two-axis dial: appearance vs behaviour, and the disagreement
# --------------------------------------------------------------------------------------------

def export_twoaxis(conn) -> None:
    racc = behavior.species_profile(conn, "raccoon")
    notch = behavior.individual_profile(conn, "Notch")
    matcher = VisitMatcher(conn, "raccoon")
    templates = matcher.templates()

    # a handful of real raccoon visits placed on the appearance x behaviour plane
    visits = conn.execute(
        "SELECT id, species, individual_id, started_at, ended_at, detection_count, "
        "representative_detection_id FROM visits WHERE species='raccoon' "
        "ORDER BY started_at DESC LIMIT 60").fetchall()
    examples = []
    for v in visits:
        if v["id"] not in matcher.protos:
            continue
        ranked = rank_templates(matcher.protos[v["id"]],
                                [t for t in templates if t[1] != v["id"]])
        if not ranked:
            continue
        dt = db.parse_local(v["started_at"])
        examples.append({
            "visit": v["id"], "hour": dt.hour if dt else None,
            "looks_like": ranked[0][0], "appearance": round(ranked[0][1], 2),
            "truth": v["individual_id"],
        })
        if len(examples) >= 16:
            break

    def prof_json(p):
        if not p:
            return None
        return {"label": p["label"], "n_visits": p["n_visits"],
                "typical_window": p["typical_window"], "arrival_hours": p["arrival_hours"],
                "dwell_median_s": p["dwell_median_s"], "peak_hour": p["peak_hour"]}

    _write_json("twoaxis.json", {
        "raccoon": prof_json(racc), "notch": prof_json(notch),
        "examples": examples, "novel_threshold": config.CONFIG.reid_novel_threshold,
    })


# --------------------------------------------------------------------------------------------
# bonus demos: through-the-glass image quality · co-presence IoU · gait cadence
# --------------------------------------------------------------------------------------------

def export_glass(conn) -> None:
    """A spread of real raccoon crops across the crop_quality range, tagged day/night, so the demo
    can scrub from the softest glass-flared night shots to the crispest daytime ones."""
    import statistics
    rows = conn.execute(
        """SELECT id, crop_path, crop_quality, timestamp FROM detections
           WHERE species='raccoon' AND crop_quality IS NOT NULL AND crop_path IS NOT NULL
           ORDER BY crop_quality""").fetchall()
    if not rows:
        _write_json("glass.json", {"crops": []}); return
    n = len(rows)
    allq = [r["crop_quality"] for r in rows]
    picks = []
    seen = set()
    for p in (0.0, 0.04, 0.10, 0.20, 0.33, 0.48, 0.62, 0.75, 0.86, 0.95):
        i = int(round(p * (n - 1)))
        if i in seen:
            continue
        seen.add(i)
        r = rows[i]
        hr = int(r["timestamp"][11:13])
        web = copy_crop(r["crop_path"], kind="crops", prefix="glass", key=r["id"], size=300)
        if web:
            picks.append({"crop": web, "quality": round(r["crop_quality"]), "hour": hr,
                          "night": hr >= 20 or hr < 6})
    _write_json("glass.json", {
        "crops": picks, "min": round(allq[0]), "median": round(statistics.median(allq)),
        "max": round(allq[-1]), "n": n})


def export_copresence(conn) -> None:
    """A real night-yard wide frame (from a busy raccoon visit's clip) as a backdrop for the
    draggable-box IoU toy. The 0.45 cut is the real co-presence threshold (individuals.iou)."""
    v = conn.execute(
        """SELECT id, started_at, source FROM visits WHERE species='raccoon'
           AND representative_detection_id IS NOT NULL ORDER BY detection_count DESC LIMIT 1""").fetchone()
    fr = None
    if v:
        clip = conn.execute(
            "SELECT clip_path FROM clips WHERE source=? AND started_at <= ? "
            "AND COALESCE(ended_at, started_at) >= ? AND pruned_at IS NULL LIMIT 1",
            (v["source"], v["started_at"], v["started_at"])).fetchone()
        if clip:
            fr = _extract_frames(clip["clip_path"], None)
    _write_json("copresence.json", {
        "frame": fr.get("frame") if fr else None,
        "w": fr["w"] if fr else 1280, "h": fr["h"] if fr else 720,
        "iou_threshold": 0.45})


def export_gait(conn) -> None:
    """Real per-tracklet gait estimates (clipmotion.py): stride cadence in Hz, how trustworthy it
    is, and seconds of walking — with a frame-crop of the animal that walked."""
    rows = conn.execute(
        """SELECT t.id, t.stride_hz, t.stride_strength, t.walk_s, t.n_hits, t.avg_speed,
                  t.straightness, e.rep_crop
           FROM clip_tracks t JOIN clip_track_embeddings e ON e.track_id = t.id
           WHERE t.stride_hz IS NOT NULL AND t.stride_strength >= 0.3 AND t.walk_s >= 2
           ORDER BY t.stride_hz""").fetchall()
    tracks = []
    for r in rows:
        web = copy_crop(r["rep_crop"], kind="crops", prefix="gait", key=r["id"], size=200)
        tracks.append({
            "id": r["id"], "hz": round(r["stride_hz"], 2), "strength": round(r["stride_strength"], 2),
            "walk_s": round(r["walk_s"], 1), "hits": r["n_hits"],
            "speed": round(r["avg_speed"], 3) if r["avg_speed"] is not None else None,
            "straightness": round(r["straightness"], 2) if r["straightness"] is not None else None,
            "crop": web})
    _write_json("gait.json", {"tracks": tracks})


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze a public making-of dataset from backyard.db.")
    ap.add_argument("--score", action="store_true",
                    help="Also run the non-animal gate (open_clip) for the decoy demo's gate scores.")
    args = ap.parse_args()

    _ensure_dirs()
    conn = _conn()
    conn.row_factory = __import__("sqlite3").Row
    print(f"Exporting making-of dataset from {config.CONFIG.db_path} -> {OUT}/")
    for name, fn in [("meta", export_meta), ("pipeline", export_pipeline),
                     ("decoy", lambda c: export_decoy(c, score=args.score)),
                     ("appearance", export_appearance), ("reid_game", export_reid_game),
                     ("twoaxis", export_twoaxis), ("glass", export_glass),
                     ("copresence", export_copresence), ("gait", export_gait)]:
        try:
            print(f"- {name}")
            fn(conn)
        except Exception as e:
            import traceback
            print(f"  !! {name} export failed: {e}")
            traceback.print_exc()
    conn.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
