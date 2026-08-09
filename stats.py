"""
Stats over the detections DB -- pure data, shared by the CLI (`--stats`) and the web
dashboard. Depends only on `db` + stdlib (no cv2/torch), so the web server imports it cheaply.

The key idea (see PLAN.md): report VISITS, not just crops. One lingering critter fires many
detections; a "visit" collapses consecutive detections on one source that are < N minutes
apart. This is a V1 estimate -- it can't yet separate two animals present at once (that needs
phase-3 individual IDs).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, time, timedelta, timezone
import math
import time as _time

import db

LATEST_LIMIT = 24          # how many recent crops to surface (gallery)
RECENT_VISITS_LIMIT = 20   # how many recent visit events to surface
REEL_LIMIT = 24            # max clips in a dispatch "highlight reel" (busiest kept, then time-ordered)


# One ISO-timestamp parser for the whole project: db.parse_local (tz-normalizing, so a legacy naive
# row never trips offset-naive/aware subtraction). Aliased here to keep the local call sites terse.
_parse = db.parse_local


def _ind_of(r):
    """A row's phase-3 individual_id, or None -- tolerant of callers whose SELECT doesn't
    include the column (sqlite3.Row raises IndexError for a missing key, dicts KeyError)."""
    try:
        return r["individual_id"]
    except (IndexError, KeyError):
        return None


def compute_visits(rows, gap_minutes: float, rep_key=None):
    """Collapse time-ordered detection rows into visit events, per source.

    `rep_key(row) -> float` scores each crop for the visit's REPRESENTATIVE (thumbnail) pick; the
    highest-scoring crop wins. Defaults to detector confidence (the most readable crop). visits_page
    passes a portrait-aware score (confidence x how much of the frame the animal fills) so a visit's
    thumbnail leans toward a close, usable shot -- the nearest thing to "a photo of its face"
    without a face detector -- rather than whichever frame merely scored highest.

    Each visit also tallies `individuals` (Counter of the rows' phase-3 individual_ids) when the
    caller's SELECT carries that column -- how "the 1am visit was Stan" reaches the dashboard."""
    rep_key = rep_key or (lambda r: r["confidence"] or 0.0)
    gap = timedelta(minutes=gap_minutes)
    by_source = defaultdict(list)
    for r in rows:
        dt = _parse(r["timestamp"])
        if dt is not None:
            by_source[r["source"]].append((dt, r))

    visits = []
    for source, items in by_source.items():
        items.sort(key=lambda x: x[0])
        cur = None
        for dt, r in items:
            if cur is None or (dt - cur["end"]) > gap:
                if cur is not None:
                    visits.append(cur)
                cur = {"source": source, "start": dt, "end": dt, "count": 0, "max_conf": 0.0,
                       "rep_score": None, "classes": Counter(), "rep_crop": None,
                       "individuals": Counter()}
            conf = r["confidence"] or 0.0
            score = rep_key(r)
            cur["end"] = dt
            cur["count"] += 1
            if cur["rep_score"] is None or score > cur["rep_score"]:   # keep the best thumbnail crop
                cur["rep_crop"] = r["crop_path"]
                cur["rep_score"] = score
            cur["max_conf"] = max(cur["max_conf"], conf)
            cur["classes"][r["species"] or r["detection_class"]] += 1
            iid = _ind_of(r)
            if iid:
                cur["individuals"][iid] += 1
        if cur is not None:
            visits.append(cur)
    return visits


def _named_of(visit, min_crops=2):
    """A visit's NAMED individuals (placeholder clusters like raccoon_c07 excluded), most-seen
    first. >= min_crops so a single stray stamp doesn't headline a visit with the wrong name."""
    named = []
    for iid, n in (visit.get("individuals") or Counter()).most_common():
        if "_c" in iid and iid.rsplit("_c", 1)[-1].isdigit():
            continue
        if n >= min_crops or visit.get("count", 0) <= 2:
            named.append(iid)
    return named


def _shot_score(r) -> float:
    """Rank a crop as a THUMBNAIL -- "the best/cutest shot of this visit". Leads with the stored
    image quality (sharpness x night eyeshine; quality.py), lifted by how much of the frame the
    animal fills and how centered it is (both cheap from the bbox already in the row). Falls back
    to confidence x size for crops captured before quality scoring existed (crop_quality NULL).
    Works on a sqlite3.Row or a plain dict, so visits_page and the digest can share it. cv2-free."""
    try:
        fw, fh = (r["frame_w"] or 0), (r["frame_h"] or 0)
        size = ((r["bbox_x2"] - r["bbox_x1"]) * (r["bbox_y2"] - r["bbox_y1"]) / float(fw * fh)
                if fw and fh else 0.0)
        cx = (r["bbox_x1"] + r["bbox_x2"]) / 2.0 / (fw or 1)
        cy = (r["bbox_y1"] + r["bbox_y2"]) / 2.0 / (fh or 1)
        center = 1.0 - min(1.0, abs(cx - 0.5) + abs(cy - 0.5))   # 1 = dead-center, ->0 at the edges
    except Exception:
        size, center = 0.0, 0.5
    try:
        q = r["crop_quality"]
    except (KeyError, IndexError):
        q = None
    if q is not None:
        return float(q) * (0.6 + min(size, 0.4)) * (0.7 + 0.3 * center)
    conf = (r["species_confidence"] or r["confidence"] or 0.0)    # transitional pre-scoring fallback
    return conf * (1.0 + min(size, 0.25))


# ---------------------------------------------------------------------------
# Behaviour CLIPS <-> visits / periods / crops. A clip (clips table) spans [started_at, ended_at]
# on one source; every detection written during that window falls inside it -- no FK, matched by
# time (verified against the live DB). So a VISIT, a digest PERIOD, or a single crop can each be
# linked to the video that was rolling at that moment, which is what lets a thumbnail play its clip.
# Pure time math -- no cv2/torch, like the rest of stats.py.
# ---------------------------------------------------------------------------

def load_clips(conn, include_pruned: bool = False) -> list:
    """Every PLAYABLE clip, parsed once for in-memory time-overlap matching. Each entry carries
    the parsed span (sdt/edt) plus the URL-friendly fields the dashboard needs. Soft-pruned clips
    (video deleted for the disk budget, row kept for its derived tracks/embeddings) are excluded
    -- this feeds watch/play surfaces, and a play button on a missing file is a broken promise.
    `include_pruned=True` keeps them, flagged `archived` (backup.py zipped the video before any
    prune could touch it), so the individual profile can offer the archived copy instead of
    pretending the visit was never filmed. [] if there are no clips."""
    try:
        rows = conn.execute(
            "SELECT id, source, clip_path, started_at, ended_at, fps, width, height, "
            "frame_count, detection_count, max_confidence, pruned_at FROM clips "
            + ("" if include_pruned else "WHERE pruned_at IS NULL ")
            + "ORDER BY started_at"
        ).fetchall()
    except Exception:
        return []          # an old DB without the clips table -> just no clips to link
    out = []
    for r in rows:
        sdt = _parse(r["started_at"])
        if sdt is None:
            continue
        edt = _parse(r["ended_at"]) if r["ended_at"] else sdt
        if r["fps"] and r["frame_count"]:
            seconds = round(r["frame_count"] / r["fps"], 1)   # true playback length
        else:
            seconds = round((edt - sdt).total_seconds(), 1) if edt else None
        out.append({
            "id": r["id"], "source": r["source"], "sdt": sdt, "edt": edt or sdt,
            "clip_path": _web(r["clip_path"]), "start": r["started_at"], "seconds": seconds,
            "dets": r["detection_count"] or 0,
            "conf": round(r["max_confidence"], 3) if r["max_confidence"] is not None else None,
            "archived": bool(r["pruned_at"]),
        })
    return out


def _clip_out(c) -> dict | None:
    """JSON-able view of a clip (drops the parsed datetimes used only for matching). `archived`
    (and the id the archive route needs) is only emitted when set, so the everyday payloads --
    which never contain pruned clips -- don't grow by two fields per clip."""
    if not c:
        return None
    out = {"clip_path": c["clip_path"], "start": c["start"], "seconds": c["seconds"],
           "dets": c["dets"], "conf": c["conf"]}
    if c.get("archived"):
        out["archived"] = True
        out["id"] = c.get("id")
    return out


def clips_overlapping(clips, source, start, end) -> list:
    """Clips on `source` whose [sdt,edt] overlaps the window [start,end] (datetimes), busiest
    first (most detections, then longest) -- the order to offer a visit's clips for playback."""
    if start is None or end is None:
        return []
    hits = [c for c in clips if c["source"] == source and c["sdt"] <= end and c["edt"] >= start]
    hits.sort(key=lambda c: (c["dets"], c["seconds"] or 0), reverse=True)
    return hits


def clip_at(clips, source, dt):
    """The clip whose window contains instant `dt` on `source` (busiest wins ties), or None --
    links one specific crop/detection to the video that was rolling then."""
    if dt is None:
        return None
    cover = [c for c in clips if c["source"] == source and c["sdt"] <= dt <= c["edt"]]
    cover.sort(key=lambda c: c["dets"], reverse=True)
    return cover[0] if cover else None


def compute_stats(cfg) -> dict | None:
    """Return a JSON-able summary of the detections DB, or None if the DB doesn't exist yet.

    Shape is stable and shared by the CLI formatter (backyard_cam.print_stats) and the web
    dashboard (/api/stats). All paths are returned with forward slashes for URL friendliness.
    """
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return None
    try:
        rows = conn.execute(
            "SELECT id, source, timestamp, detection_class, species, confidence, "
            "species_confidence, crop_path "
            "FROM detections ORDER BY timestamp"
        ).fetchall()
        clips = load_clips(conn)

        gap = cfg.visit_gap_minutes
        visits = compute_visits(rows, gap)

        def day(ts): return ts[:10]
        def web(p):  return p.replace("\\", "/") if p else None

        src_crops = Counter(r["source"] for r in rows)
        src_visits = Counter(v["source"] for v in visits)
        cls_crops = Counter((r["species"] or r["detection_class"]) for r in rows)
        visits_by_day = Counter(v["start"].strftime("%Y-%m-%d") for v in visits)
        hour_counts = Counter(v["start"].hour for v in visits)

        # Per-day tallies in ONE pass, not the old O(rows*days) re-scan of `rows` inside the day
        # loop: accumulate a crop Counter and a class Counter keyed by day, then just read them out.
        day_crops = Counter()
        day_classes = defaultdict(Counter)
        for r in rows:
            d = day(r["timestamp"])
            day_crops[d] += 1
            day_classes[d][r["species"] or r["detection_class"]] += 1
        by_day = [{"day": d, "crops": day_crops[d], "visits": visits_by_day.get(d, 0),
                   "classes": dict(day_classes[d].most_common())}
                  for d in sorted(day_crops)]

        latest = [{
            "timestamp": r["timestamp"],
            "source": r["source"],
            "detection_class": r["detection_class"],
            "species": r["species"],
            "confidence": (round(r["confidence"], 3) if r["confidence"] is not None else None),
            "species_confidence": (round(r["species_confidence"], 3)
                                   if r["species_confidence"] is not None else None),
            "crop_path": web(r["crop_path"]),
            # The clip rolling when this crop was caught (so the live card can play it), or None.
            "clip": _clip_out(clip_at(clips, r["source"], _parse(r["timestamp"]))),
        } for r in list(reversed(rows))[:LATEST_LIMIT]]

        recent_visits = [{
            "source": v["source"],
            "start": v["start"].isoformat(),
            "end": v["end"].isoformat(),
            "count": v["count"],
            "minutes": round((v["end"] - v["start"]).total_seconds() / 60.0, 1),
            "max_conf": round(v["max_conf"], 3),
            "classes": dict(v["classes"].most_common()),
            "clips": [_clip_out(c) for c in clips_overlapping(clips, v["source"], v["start"], v["end"])],
        } for v in sorted(visits, key=lambda v: v["start"], reverse=True)[:RECENT_VISITS_LIMIT]]

        return {
            "db_path": str(cfg.db_path),
            "gap_minutes": gap,
            "total_crops": len(rows),
            "total_visits": len(visits),
            "span": ({"start": rows[0]["timestamp"], "end": rows[-1]["timestamp"]} if rows else None),
            "by_source": [{"source": s, "crops": src_crops[s], "visits": src_visits.get(s, 0)}
                          for s in sorted(src_crops)],
            "by_class": [{"name": c, "crops": n} for c, n in cls_crops.most_common()],
            "by_day": by_day,
            "by_hour": [{"hour": h, "visits": hour_counts[h]} for h in range(24) if hour_counts.get(h)],
            "latest": latest,
            "recent_visits": recent_visits,
        }
    finally:
        conn.close()


def current_live_visit(cfg, source: str | None = None, lookback: int = 600) -> dict:
    """The visit happening (or most recently) on a live source -- the span the Live tab's
    "who's here now?" control names. Walks back from the newest detection on `source`,
    accumulating while consecutive gaps stay under `visit_gap_minutes`; the first larger gap
    ends the span. Returns {source, count, start, end, minutes, latest, latest_age_s, active,
    species}; `active` = the newest detection is still within the gap (the animal hasn't left).
    Just {source, count: 0} when nothing is on that source yet."""
    source = source or cfg.source
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"source": source, "count": 0}
    try:
        rows = conn.execute(
            "SELECT timestamp, species FROM detections WHERE source = ? "
            "ORDER BY id DESC LIMIT ?", (source, int(lookback))).fetchall()
        # Walk back over the recent window by INSTANT, not insertion id: the live rig writes in
        # time order so the two agree, but sorting by parsed time keeps the gap detection correct
        # even if a backfill or clock hiccup left an out-of-order row in the window.
        cand = sorted(((t, r["species"]) for r in rows
                       if (t := _parse(r["timestamp"])) is not None),
                      key=lambda x: x[0], reverse=True)
        gap = timedelta(minutes=cfg.visit_gap_minutes)
        span, prev = [], None      # newest-first; stop at the first gap >= visit_gap.
        for t, sp in cand:
            if prev is not None and (prev - t) >= gap:
                break
            span.append((t, sp))
            prev = t
        if not span:
            return {"source": source, "count": 0}
        times = [t for t, _ in span]
        start, end = min(times), max(times)
        now = datetime.now().astimezone()
        species = Counter(s for _, s in span if s)
        return {
            "source": source,
            "count": len(span),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "minutes": round((end - start).total_seconds() / 60.0, 1),
            "latest": end.isoformat(),
            "latest_age_s": round((now - end).total_seconds(), 1),
            "active": (now - end) < gap,
            "species": dict(species.most_common()),
        }
    finally:
        conn.close()


def species_overview(cfg) -> dict | None:
    """Per-species rollup for the most/least-frequent display: count, avg confidence, review
    tallies, and a representative (verified-or-most-confident) crop. None if no DB yet."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return None
    try:
        rows = conn.execute(
            "SELECT species, COUNT(*) n, ROUND(AVG(species_confidence), 3) avg_conf, "
            "SUM(CASE WHEN species_verified = 1 THEN 1 ELSE 0 END) verified, "
            "SUM(CASE WHEN species_verified = 0 THEN 1 ELSE 0 END) rejected "
            "FROM detections WHERE species IS NOT NULL GROUP BY species ORDER BY n DESC"
        ).fetchall()
        # Drop non-critter human-correction labels (chair, bricks, "not an animal", person, ...)
        # so the catalogue and the "Rarely Seen" cards stay about real animals -- same filter the
        # period digest applies (see _NON_CRITTER). Real rarities (Douglas squirrel) are NOT here.
        rows = [r for r in rows if (r["species"] or "").lower() not in _NON_CRITTER]
        species = []
        for r in rows:
            s = conn.execute(
                "SELECT crop_path FROM detections WHERE species = ? "
                "ORDER BY (species_verified = 1) DESC, species_confidence DESC LIMIT 1",
                (r["species"],)).fetchone()
            species.append({
                "species": r["species"], "count": r["n"], "avg_conf": r["avg_conf"],
                "verified": r["verified"], "rejected": r["rejected"],
                "sample": (s["crop_path"].replace("\\", "/") if s and s["crop_path"] else None),
            })
        return {"species": species, "total": sum(s["count"] for s in species)}
    finally:
        conn.close()


def individuals_overview(cfg, thumbs: int = 6) -> dict:
    """Phase-3 labelling: every individual_id group (placeholder clusters like 'raccoon_c01' and
    hand-named individuals like 'Notch') with crop count, time span, dominant species, and a strip
    of its most-readable crops. Powers the dashboard's Individuals tab, where naming a group --
    or naming two groups the same -- is the cheap human step that turns clusters into a cast."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"groups": [], "total_crops": 0}
    try:
        rows = conn.execute(
            "SELECT individual_id iid, COUNT(*) n, MIN(timestamp) first_seen, MAX(timestamp) last_seen "
            "FROM detections WHERE individual_id IS NOT NULL GROUP BY individual_id"
        ).fetchall()
        groups = []
        for r in rows:
            sp = conn.execute(
                "SELECT species, COUNT(*) c FROM detections WHERE individual_id = ? AND species IS "
                "NOT NULL GROUP BY species ORDER BY c DESC LIMIT 1", (r["iid"],)).fetchone()
            crops = conn.execute(
                "SELECT crop_path FROM detections WHERE individual_id = ? "
                "ORDER BY confidence DESC LIMIT ?", (r["iid"], thumbs)).fetchall()
            # Placeholder = reid.py --write-clusters output ('<species>_cNN'); named = anything else.
            is_placeholder = "_c" in r["iid"] and r["iid"].rsplit("_c", 1)[-1].isdigit()
            groups.append({
                "id": r["iid"], "n_crops": r["n"], "placeholder": is_placeholder,
                "species": sp["species"] if sp else None,
                "first_seen": r["first_seen"], "last_seen": r["last_seen"],
                "crops": [c["crop_path"].replace("\\", "/") for c in crops if c["crop_path"]],
            })
        # Named individuals first (the cast), then placeholders by size (biggest worth naming first).
        groups.sort(key=lambda g: (g["placeholder"], -g["n_crops"]))
        return {"groups": groups, "total_crops": sum(g["n_crops"] for g in groups),
                "named": sum(1 for g in groups if not g["placeholder"])}
    finally:
        conn.close()


def seasons_overview(cfg) -> dict:
    """The longitudinal view the Calendar's day cells can't give: per-species WEEKLY visit
    counts (the sparkline grid), first-ever and last-seen dates, and the yard's
    species-accumulation curve. This is where "the crows collapsed after mid-July" and "raccoon
    traffic doubled when the kits emerged" become visible instead of remembered -- and where
    next year's "same week last year" comparison will live, so building it now starts the clock
    on the yard's first annual cycle. Visits, not crops (one lingering critter fires hundreds of
    detections); denylist applied; read-only."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"species": [], "weeks": [], "accumulation": []}
    try:
        rows = conn.execute(
            "SELECT started_at, species FROM visits WHERE species IS NOT NULL "
            "ORDER BY started_at").fetchall()
    finally:
        conn.close()
    per = defaultdict(lambda: {"weeks": Counter(), "first": None, "last": None, "n": 0})
    weeks_seen = set()
    accumulation = []
    seen_species = set()
    for r in rows:
        sp = r["species"]
        if (sp or "").lower() in _NON_CRITTER:
            continue
        dt = _parse(r["started_at"])
        if dt is None:
            continue
        wk = dt.strftime("%G-W%V")
        weeks_seen.add(wk)
        p = per[sp]
        p["weeks"][wk] += 1
        p["n"] += 1
        if p["first"] is None:
            p["first"] = dt.date().isoformat()
        p["last"] = dt.date().isoformat()
        if sp not in seen_species:
            seen_species.add(sp)
            accumulation.append({"date": dt.date().isoformat(), "n_species": len(seen_species),
                                 "species": sp})
    weeks = sorted(weeks_seen)
    species = [{"species": sp, "n_visits": p["n"], "first": p["first"], "last": p["last"],
                "weekly": [p["weeks"].get(w, 0) for w in weeks]}
               for sp, p in sorted(per.items(), key=lambda kv: -kv[1]["n"])]
    return {"weeks": weeks, "species": species, "accumulation": accumulation}


def cast_rollcall(cfg, now=None) -> dict:
    """The named cast with last-seen + an 'overdue' flag -- the daily "who's back / who hasn't
    shown" roll for the Dispatch and Individuals tab. Placeholder clusters (raccoon_c01) are
    excluded; this is the hand-named cast only. 'Overdue' = a REGULAR (seen on >=3 distinct days)
    now gone notably longer than its own typical gap between appearances -- a one-off visitor is
    never flagged overdue. Sorted overdue-first, then by most-recently-seen."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"cast": []}
    try:
        now = now or _local_now()
        rows = conn.execute(
            "SELECT individual_id iid, COUNT(*) n, MAX(timestamp) last_seen "
            "FROM detections WHERE individual_id IS NOT NULL GROUP BY individual_id").fetchall()
        cast = []
        for r in rows:
            iid = r["iid"]
            if "_c" in iid and iid.rsplit("_c", 1)[-1].isdigit():
                continue                                  # placeholder cluster -> not the named cast
            days = [d["d"] for d in conn.execute(
                "SELECT DISTINCT substr(timestamp, 1, 10) d FROM detections WHERE individual_id = ? "
                "ORDER BY d", (iid,)).fetchall()]
            last = db.parse_local(r["last_seen"])
            days_since = (now.date() - last.date()).days if last else None
            gaps = sorted(
                (datetime.fromisoformat(days[i]).date() - datetime.fromisoformat(days[i - 1]).date()).days
                for i in range(1, len(days)))
            med_gap = gaps[len(gaps) // 2] if gaps else None
            regular = len(days) >= 3
            overdue = bool(regular and med_gap and days_since is not None
                           and days_since > max(2, 2 * med_gap))
            sp = conn.execute(
                "SELECT species FROM detections WHERE individual_id = ? AND species IS NOT NULL "
                "GROUP BY species ORDER BY COUNT(*) DESC LIMIT 1", (iid,)).fetchone()
            crop = conn.execute(
                "SELECT crop_path FROM detections WHERE individual_id = ? "
                "ORDER BY (species_verified = 1) DESC, crop_quality DESC LIMIT 1", (iid,)).fetchone()
            cast.append({
                "id": iid, "species": sp["species"] if sp else None,
                "n_crops": r["n"], "nights": len(days),
                "last_seen": r["last_seen"], "days_since": days_since,
                "typical_gap_days": med_gap, "regular": regular, "overdue": overdue,
                "crop": (crop["crop_path"].replace("\\", "/") if crop and crop["crop_path"] else None),
            })
        # FAMILY LINK (2026-08-08): "Stan + Kits" is evidence of STAN. Before this, the solo
        # identity read "overdue -- 8 days" while her family group was stamped YESTERDAY: the
        # roll was calling a present animal missing, because the group string is a separate
        # individual_id to the DB. A group label's recency now updates its BASE name's
        # last_seen/days_since/overdue (never its counts -- the group's crops stay the
        # group's), and the card says how it knows (`via_group`).
        solos = {c["id"].casefold(): c for c in cast if not db.is_group_label(c["id"])}
        for c in cast:
            if not db.is_group_label(c["id"]):
                continue
            base = solos.get(c["id"].split(" + ", 1)[0].strip().casefold())
            if base is None or c["days_since"] is None:
                continue
            if base["days_since"] is None or c["days_since"] < base["days_since"]:
                base["days_since"] = c["days_since"]
                base["last_seen"] = c["last_seen"]
                base["via_group"] = c["id"]
                base["overdue"] = bool(base["regular"] and base["typical_gap_days"]
                                       and base["days_since"] > max(2, 2 * base["typical_gap_days"]))
        cast.sort(key=lambda c: (not c["overdue"],
                                 c["days_since"] if c["days_since"] is not None else 1e9))
        return {"cast": cast, "as_of": now.isoformat()}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# MOTION surfacing -- phase-4 clip_tracks (clipmotion.py) read out for the dashboard. A tracklet is
# one animal's box-centre trajectory through a clip: how fast, how straight, how hesitant, and --
# via area_trend (end/start box-area ratio) -- whether it approached the camera or backed away.
# These surface that behaviour signal per VISIT and per named INDIVIDUAL; both are read-only.
# CAVEAT: speed/area are in normalized (fraction-of-frame) units, so they're bbox-SIZE-confounded --
# a nearer or larger animal reads "faster" for the same real motion. Treat as descriptive, not
# authoritative (matches the documented re-ID caveat -- motion is unvalidated as an ID/metric).
# ---------------------------------------------------------------------------

def _behaviour_tag(minutes, moving_frac, straightness):
    """A coarse what-they-DID word from the stored motion features -- 'fed here' / 'passed
    through' / 'lingered'. The corpus median moving_frac is 0.185 (animals at this dish are
    mostly stationary, i.e. eating), yet no surface ever said so. This is the cheapest possible
    behaviour classification: three features that survived measurement (gait is dead at this
    frame rate -- 27 of 25,041 tracks resolve a stride), thresholds set by eyeball against real
    clips, worded as a reading, never a verdict."""
    if moving_frac is None:
        return None
    if minutes >= 3 and moving_frac <= 0.30:
        return "fed here"
    if minutes < 1.5 and (moving_frac >= 0.5 or (straightness or 0) >= 0.7):
        return "passed through"
    return "lingered"


def _approach_label(mean_area_trend) -> str:
    """area_trend is end/start box-area: >1.15 the animal grew in frame (approached), <0.85 it
    shrank (retreated), else steady. None (no area data) reads as 'steady'."""
    if mean_area_trend is None:
        return "steady"
    if mean_area_trend > 1.15:
        return "approach"
    if mean_area_trend < 0.85:
        return "retreat"
    return "steady"


def visit_motion(cfg, visit_id) -> dict:
    """Motion summary for one visit: the clip_tracks whose parent clip overlaps the visit's
    [started_at, ended_at] window on the SAME source (no FK -- matched by time, like the rest of
    stats.py). Returns a small JSON-able dict -- {tracks, avg_speed, peak_speed, straightness,
    moving_frac, area_trend, approach} -- or {"tracks": 0, ...} when the visit has no overlapping
    tracks (or the visit / clips don't exist). Speeds are bbox-size-confounded (see module note)."""
    empty = {"tracks": 0, "avg_speed": None, "peak_speed": None, "straightness": None,
             "moving_frac": None, "area_trend": None, "approach": "steady"}
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return empty
    try:
        v = conn.execute(
            "SELECT source, started_at, ended_at FROM visits WHERE id = ?", (visit_id,)).fetchone()
        if v is None:
            return empty
        start, end = _parse(v["started_at"]), _parse(v["ended_at"])
        if start is None or end is None:
            return empty
        # Join tracks to their clip, keep those on this source whose clip window overlaps the visit.
        rows = conn.execute(
            "SELECT t.avg_speed, t.peak_speed, t.straightness, t.moving_frac, t.area_trend, "
            "c.started_at, c.ended_at FROM clip_tracks t JOIN clips c ON c.id = t.clip_id "
            "WHERE c.source = ?", (v["source"],)).fetchall()
        tracks = []
        for r in rows:
            csdt = _parse(r["started_at"])
            cedt = _parse(r["ended_at"]) if r["ended_at"] else csdt
            if csdt is not None and csdt <= end and (cedt or csdt) >= start:
                tracks.append(r)
        if not tracks:
            return empty

        def _mean(col):
            vals = [r[col] for r in tracks if r[col] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        peaks = [r["peak_speed"] for r in tracks if r["peak_speed"] is not None]
        area = _mean("area_trend")
        mins = (end - start).total_seconds() / 60.0
        mfm, stm = _mean("moving_frac"), _mean("straightness")
        return {
            "tracks": len(tracks),
            "avg_speed": _mean("avg_speed"),
            "peak_speed": round(max(peaks), 4) if peaks else None,
            "straightness": stm,
            "moving_frac": mfm,
            "area_trend": area,
            "approach": _approach_label(area),
            "tag": _behaviour_tag(mins, mfm, stm),
        }
    finally:
        conn.close()


def individual_motion(cfg, individual_id) -> dict:
    """Aggregate motion fingerprint for one named individual (e.g. 'Notch', 'Stan') from every
    clip_track linked to it (clip_tracks.individual_id = ?). Returns a JSON-able dict: track count,
    mean/median avg_speed, mean straightness, mean moving_frac, and approach/retreat/steady counts.
    NOTE: per-individual speed is bbox-SIZE-confounded and UNVALIDATED (a nearer animal reads faster);
    this is a descriptive fingerprint, not an authoritative metric -- do not present it as ground
    truth. {"tracks": 0, ...} when the individual has no linked tracks."""
    empty = {"tracks": 0, "avg_speed_mean": None, "avg_speed_median": None, "straightness": None,
             "moving_frac": None, "approach": 0, "retreat": 0, "steady": 0}
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return empty
    try:
        rows = conn.execute(
            "SELECT avg_speed, straightness, moving_frac, area_trend FROM clip_tracks "
            "WHERE individual_id = ?", (individual_id,)).fetchall()
        if not rows:
            return empty

        def _mean(col):
            vals = [r[col] for r in rows if r[col] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        speeds = sorted(r["avg_speed"] for r in rows if r["avg_speed"] is not None)
        median = round(speeds[len(speeds) // 2], 4) if speeds else None   # simple mid-element median
        approach = retreat = steady = 0
        for r in rows:
            lab = _approach_label(r["area_trend"])   # per-track, so counts sum to tracks-with-area
            if r["area_trend"] is None:
                continue                             # no area data -> not counted as any direction
            approach += lab == "approach"
            retreat += lab == "retreat"
            steady += lab == "steady"
        return {
            "tracks": len(rows),
            "avg_speed_mean": _mean("avg_speed"),
            "avg_speed_median": median,
            "straightness": _mean("straightness"),
            "moving_frac": _mean("moving_frac"),
            "approach": approach, "retreat": retreat, "steady": steady,
        }
    finally:
        conn.close()


_REVIEW_SUSPECT = {"brown rat", "domestic dog"}   # real-ish but error-prone labels (per the analysis)
# Clearly day-active species: one detected at deep night is almost certainly a nocturnal mammal
# mislabeled. Used ONLY to prioritize the review queue -- never to relabel anything.
_DIURNAL = {
    "american crow", "dark-eyed junco", "spotted towhee", "european starling", "house finch",
    "house sparrow", "northern flicker", "american robin", "band-tailed pigeon", "song sparrow",
    "california scrub-jay", "varied thrush", "chestnut-backed chickadee", "american goldfinch",
    "golden-crowned sparrow", "white-crowned sparrow", "bushtit", "black-capped chickadee",
    "bewick's wren", "steller's jay", "eastern gray squirrel", "douglas squirrel",
    "townsend's chipmunk",
}


def review_queue(cfg, limit: int = 150) -> dict:
    """Prioritized 'most likely mislabeled' crops for a human verify pass -- the suspect crops the
    analysis flagged. UNVERIFIED only (species_verified IS NULL); each crop carries a `reason` and is
    ordered worst-first so a few minutes of ✓/✗ clears the shakiest labels. Reuses the dashboard's
    existing per-crop review controls. Priorities (high->low): a flagged suspect species (brown rat,
    domestic dog); the clip-filter's 'not an animal' / other non-critter labels; a clearly day-active
    species detected at deep night (22:00-05:00); and very low detector confidence. This only RANKS
    what to look at -- it never changes a label."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"crops": [], "total": 0, "shown": 0}
    try:
        rows = conn.execute(
            "SELECT id, crop_path, species, ROUND(confidence, 3) det_conf, "
            "ROUND(species_confidence, 3) sp_conf, species_verified, timestamp, source "
            "FROM detections WHERE species_verified IS NULL AND species IS NOT NULL").fetchall()
        out = []
        for r in rows:
            sp = (r["species"] or "").lower()
            ts = r["timestamp"] or ""
            hour = int(ts[11:13]) if len(ts) >= 13 and ts[11:13].isdigit() else 12
            night = hour >= 22 or hour < 5
            if sp in _REVIEW_SUSPECT:
                reason, prio = f"suspect label · {r['species']}", 4
            elif sp in _NON_CRITTER:
                reason, prio = "flagged non-animal", 3
            elif sp in _DIURNAL and night:
                reason, prio = "day species, seen at night", 3
            elif ((r["det_conf"] if r["det_conf"] is not None else 1.0) < 0.45
                  and r["sp_conf"] is not None and r["sp_conf"] < 0.6):
                reason, prio = "both models unsure", 2   # detector barely saw it AND BioCLIP hesitated
            else:
                continue
            out.append({"id": r["id"], "crop_path": (r["crop_path"] or "").replace("\\", "/"),
                        "species": r["species"], "confidence": r["det_conf"],
                        "species_confidence": r["sp_conf"], "verified": r["species_verified"],
                        "timestamp": r["timestamp"], "reason": reason, "_p": prio})
        out.sort(key=lambda c: (-c["_p"], c["timestamp"]))   # worst first, then oldest first
        total = len(out)
        for c in out:
            c.pop("_p", None)
        return {"crops": out[:max(1, int(limit))], "total": total, "shown": min(int(limit), total)}
    finally:
        conn.close()


def species_crops(cfg, species: str, limit: int = 160) -> list:
    """Recent crops for one species (newest first) for the by-species browser."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT id, crop_path, ROUND(species_confidence, 3) conf, species_verified, "
            "timestamp, source FROM detections WHERE species = ? ORDER BY id DESC LIMIT ?",
            (species, int(limit))).fetchall()
        return [{"id": r["id"], "crop_path": (r["crop_path"] or "").replace("\\", "/"),
                 "confidence": r["conf"], "verified": r["species_verified"],
                 "timestamp": r["timestamp"], "source": r["source"]} for r in rows]
    finally:
        conn.close()


def crops_page(cfg, day=None, species=None, start=None, end=None, offset=0, limit=60,
               individual=None) -> dict:
    """A filtered, paginated slice of detection crops (newest first) for the explorer drill-down.
    Filters (all optional, AND-combined): day='YYYY-MM-DD', species exact, start/end ISO timestamp
    bounds, individual exact (the phase-3 name stamped on the crop). Returns {crops, total,
    offset, limit} so the UI can show "loaded of total"."""
    limit = max(1, min(int(limit), 200))      # clamp so a negative/huge ?limit can't dump the table
    offset = max(0, int(offset))
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"crops": [], "total": 0, "offset": 0, "limit": limit}
    try:
        where, args = [], []
        if day:        where.append("timestamp LIKE ?");   args.append(str(day) + "%")
        if species:    where.append("species = ?");        args.append(species)
        if start:      where.append("timestamp >= ?");     args.append(start)
        if end:        where.append("timestamp <= ?");     args.append(end)
        if individual: where.append("individual_id = ?");  args.append(individual)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        total = conn.execute(f"SELECT COUNT(*) FROM detections {clause}", args).fetchone()[0]
        rows = conn.execute(
            f"SELECT id, timestamp, species, confidence, species_confidence, species_verified, "
            f"crop_path FROM detections {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
            args + [int(limit), int(offset)]).fetchall()
        crops = [{"id": r["id"], "timestamp": r["timestamp"], "species": r["species"],
                  "confidence": r["confidence"], "species_confidence": r["species_confidence"],
                  "verified": r["species_verified"],
                  "crop_path": (r["crop_path"] or "").replace("\\", "/")} for r in rows]
        return {"crops": crops, "total": total, "offset": int(offset), "limit": int(limit)}
    finally:
        conn.close()


_VISITS_SCAN_ROWS = 15000   # newest detections scanned for the no-filter visits page (see below)


def visits_page(cfg, day=None, limit=200) -> dict:
    """Visit events (newest first), each with a representative crop + any named individuals, for
    the explorer. Optionally restricted to a single day ('YYYY-MM-DD').

    Perf: without a day filter this used to SELECT the ENTIRE detections table and cluster it in
    Python on every request (~5s / half an MB at 80k rows, on the page people open daily). Nobody
    scrolls hundreds of cards, so instead scan only the newest _VISITS_SCAN_ROWS detections --
    enough for a few hundred visits -- and say so via `window: true` (`total` is then "the visits
    in this window", not all-time). A day filter still scans exactly that day."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"visits": [], "total": 0}
    try:
        cols = ("id, source, timestamp, detection_class, species, confidence, species_confidence, "
                "crop_path, crop_quality, individual_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
                "frame_w, frame_h")
        windowed = False
        if day:
            rows = conn.execute(
                f"SELECT {cols} FROM detections WHERE timestamp LIKE ? ORDER BY timestamp",
                [str(day) + "%"]).fetchall()
        else:
            rows = conn.execute(   # newest first via the timestamp index, then back to time order
                f"SELECT {cols} FROM detections ORDER BY timestamp DESC LIMIT ?",
                [_VISITS_SCAN_ROWS]).fetchall()
            windowed = len(rows) >= _VISITS_SCAN_ROWS
            rows = rows[::-1]
        clips = load_clips(conn)

        visits = compute_visits(rows, cfg.visit_gap_minutes, rep_key=_shot_score)
        if windowed and visits:
            # The window may open mid-visit, leaving each source's OLDEST visit a fragment
            # (wrong start/count). Drop those rather than show a truncated visit as real.
            oldest = {}
            for v in visits:
                if v["source"] not in oldest or v["start"] < oldest[v["source"]]["start"]:
                    oldest[v["source"]] = v
            visits = [v for v in visits if oldest[v["source"]] is not v]
        visits.sort(key=lambda v: v["start"], reverse=True)
        out = []
        for v in visits[:limit]:
            title = v["classes"].most_common(1)[0][0] if v["classes"] else "animal"
            out.append({
                "start": v["start"].isoformat(), "end": v["end"].isoformat(), "source": v["source"],
                "count": v["count"], "minutes": round((v["end"] - v["start"]).total_seconds() / 60.0, 1),
                "max_conf": round(v["max_conf"], 3), "title": title,
                "classes": dict(v["classes"].most_common()),
                "individuals": _named_of(v),
                "rep_crop": ((v.get("rep_crop") or "").replace("\\", "/") or None),
                # The clips that rolled during this visit (busiest first) -> click the card, watch the
                # video. Empty for visits before clip recording was on (e.g. daytime pre-06-09).
                "clips": [_clip_out(c) for c in clips_overlapping(clips, v["source"], v["start"], v["end"])],
            })
        return {"visits": out, "total": len(visits), "window": windowed}
    finally:
        conn.close()


def individual_profile(cfg, name, visit_limit: int = 200) -> dict:
    """Everything the log holds about ONE named individual: every visit their name is stamped
    into (via detections.visit_id -> the visits ledger), their photo count for the paged crop
    grid, the clips that rolled during those visits -- INCLUDING soft-pruned ones, flagged
    `archived` so the dashboard can offer the backup copy -- plus reference photos (refcam),
    who they showed up with, and how the stamps got there (human / auto tier / clustering).

    Visit dicts are shaped exactly like visits_page's so the dashboard renders both with the
    same card. The visit list is capped at `visit_limit` (newest first); `n_visits` is the true
    total. Crops stamped with the name but not yet filed into a visit (the ledger refreshes on
    label passes and shutdown, so the newest visit can lag) are counted as `unfiled`."""
    name = (name or "").strip()
    conn = db.connect_readonly(cfg.db_path) if name else None
    if conn is None:
        return {"found": False, "name": name}
    try:
        head = conn.execute(
            "SELECT COUNT(*) n, MIN(timestamp) first_seen, MAX(timestamp) last_seen "
            "FROM detections WHERE individual_id = ?", (name,)).fetchone()
        if not head or not head["n"]:
            return {"found": False, "name": name}

        species_mix = [{"species": r["sp"], "n": r["n"]} for r in conn.execute(
            "SELECT COALESCE(species, detection_class) sp, COUNT(*) n FROM detections "
            "WHERE individual_id = ? GROUP BY sp ORDER BY n DESC", (name,))]
        stamp_mix = {(r["s"] or "human"): r["n"] for r in conn.execute(
            "SELECT individual_source s, COUNT(*) n FROM detections "
            "WHERE individual_id = ? GROUP BY s", (name,))}
        by_source = [{"source": r["source"], "n": r["n"]} for r in conn.execute(
            "SELECT source, COUNT(*) n FROM detections WHERE individual_id = ? "
            "GROUP BY source ORDER BY n DESC", (name,))]
        unfiled = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE individual_id = ? AND visit_id IS NULL",
            (name,)).fetchone()[0]

        # Every crop of every visit the name appears in -- the co-present animals' crops too, so
        # each visit's classes/companions/span reflect the whole visit, not just this animal.
        rows = conn.execute(
            "SELECT id, source, timestamp, detection_class, species, confidence, "
            "species_confidence, crop_path, crop_quality, individual_id, visit_id, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h "
            "FROM detections WHERE visit_id IN "
            "(SELECT DISTINCT visit_id FROM detections WHERE individual_id = ? "
            " AND visit_id IS NOT NULL) ORDER BY timestamp", (name,)).fetchall()
        clips = load_clips(conn, include_pruned=True)

        grouped: dict = {}
        for r in rows:
            g = grouped.setdefault(r["visit_id"], {
                "source": r["source"], "start": None, "end": None, "count": 0, "max_conf": 0.0,
                "classes": Counter(), "individuals": Counter(),
                "rep_crop": None, "rep_score": None,        # best shot of THIS individual
                "any_crop": None, "any_score": None,        # fallback: best shot of the visit
                "n_mine": 0,
            })
            dt = _parse(r["timestamp"])
            if dt is not None:
                g["start"] = dt if g["start"] is None else min(g["start"], dt)
                g["end"] = dt if g["end"] is None else max(g["end"], dt)
            g["count"] += 1
            g["max_conf"] = max(g["max_conf"], r["confidence"] or 0.0)
            g["classes"][r["species"] or r["detection_class"]] += 1
            iid = r["individual_id"]
            if iid:
                g["individuals"][iid] += 1
            score = _shot_score(r)
            if g["any_score"] is None or score > g["any_score"]:
                g["any_crop"], g["any_score"] = r["crop_path"], score
            if iid == name:
                g["n_mine"] += 1
                if g["rep_score"] is None or score > g["rep_score"]:
                    g["rep_crop"], g["rep_score"] = r["crop_path"], score

        companions = Counter()
        visits = []
        for vid, g in grouped.items():
            if g["start"] is None:
                continue
            title = g["classes"].most_common(1)[0][0] if g["classes"] else "animal"
            for other in _named_of(g):
                if other != name:
                    companions[other] += 1
            visits.append({
                "visit_id": vid, "start": g["start"].isoformat(), "end": g["end"].isoformat(),
                "source": g["source"], "count": g["count"], "n_mine": g["n_mine"],
                "minutes": round((g["end"] - g["start"]).total_seconds() / 60.0, 1),
                "max_conf": round(g["max_conf"], 3), "title": title,
                "classes": dict(g["classes"].most_common()),
                "individuals": _named_of(g),
                "rep_crop": _web(g["rep_crop"] or g["any_crop"]),
                "clips": [_clip_out(c) for c in
                          clips_overlapping(clips, g["source"], g["start"], g["end"])],
            })
        visits.sort(key=lambda v: v["start"], reverse=True)
        n_visits = len(visits)
        visits = visits[:max(1, int(visit_limit))]

        # Reference photos (refcam): the detector crops cut from Matt's phone shots -- served
        # from reference_crops/; the raw originals live outside the project and are never served.
        try:
            refs = [{"crop_path": _web(r["crop_path"]), "captured_at": r["captured_at"],
                     "kind": r["media_kind"] or "photo", "note": r["note"]}
                    for r in conn.execute(
                        "SELECT c.crop_path, r.captured_at, r.media_kind, r.note "
                        "FROM identity_reference_crops c "
                        "JOIN identity_references r ON r.id = c.reference_id "
                        "WHERE r.individual_id = ? "
                        "ORDER BY (r.captured_at IS NULL), r.captured_at, c.id", (name,))]
        except Exception:
            refs = []                       # an older DB without the reference tables

        try:                        # statuses are keyed by the raw typed name; match nocase
            status = next((s for n, s in db.individual_statuses(conn).items()
                           if str(n).strip().casefold() == name.casefold()), None)
        except Exception:
            status = None

        return {
            "found": True, "name": name,
            "species": species_mix[0]["species"] if species_mix else None,
            "species_mix": species_mix, "stamp_mix": stamp_mix, "by_source": by_source,
            "n_crops": head["n"], "first_seen": head["first_seen"], "last_seen": head["last_seen"],
            "n_visits": n_visits, "unfiled": unfiled,
            "companions": [{"name": k, "n_visits": v} for k, v in companions.most_common()],
            "visits": visits,
            "references": refs,
            "status": status,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Period digest -- "The Dispatch": a summary of the most-recently-COMPLETED
# sun-defined period. Shown in the MORNING it summarizes the NIGHT just past;
# shown in the EVENING it summarizes the DAY just past. One artifact, two
# editions. Pure data over the same detections table; reuses compute_visits so
# it counts VISITS, not crops (see PLAN.md). Sun boundaries come from
# daynight.sun_times (astral); everything degrades to fixed clock hours if
# lat/lon or astral are missing, so the digest always renders.
# ---------------------------------------------------------------------------

# Moon: astral.moon.phase() returns 0..27.99 (0/28 = new, 7 = first quarter,
# 14 = full, 21 = last quarter). Upper edge -> (name, emoji glyph).
_MOON_PHASES = [
    (1.75, "New Moon", "\U0001F311"), (5.25, "Waxing Crescent", "\U0001F312"),
    (8.75, "First Quarter", "\U0001F313"), (12.25, "Waxing Gibbous", "\U0001F314"),
    (15.75, "Full Moon", "\U0001F315"), (19.25, "Waning Gibbous", "\U0001F316"),
    (22.75, "Last Quarter", "\U0001F317"), (26.25, "Waning Crescent", "\U0001F318"),
    (28.01, "New Moon", "\U0001F311"),
]

_SUN_CACHE: dict = {}   # (date.toordinal(), lat, lon) -> (dawn, dusk); stable per date, so memoize.

# Non-critter labels: things a human correction can name that aren't actual visitors -- false
# triggers (patio bricks, a blurry smear), the feeding station, the household itself. Excluded from
# the digest's roll / plate / novelty so "who visited" stays about animals. Genuine rare species
# (Douglas squirrel, Anna's hummingbird) are NOT here -- they're real and SHOULD surface. The
# unclassified coarse label 'animal' is kept in the roll (a real critter, just unnamed) but excluded
# from novelty/quiet headlines. Operators: add the individual names you use for your own household
# (yourself, housemates, your own pets if you don't want them counted) to this set.
_NON_CRITTER = {
    # human corrections of false triggers and of the feeder
    "bricks", "brick", "blur", "blurry", "cat food", "catfood", "food",
    "door", "porch", "broom", "chair",
    # likely future static-object corrections + generic non-visitors
    "fence", "wall", "table", "plant", "pot", "hose", "shadow", "reflection", "leaf", "leaves",
    "rock", "stick", "sticks", "ground", "tree", "bush", "person", "people", "human", "vehicle",
    "car", "unknown", "unidentified", "nothing", "empty", "none", "background", "n/a", "na", "",
    # clipfilter.py's general-CLIP gate writes this when a crop isn't an animal (food, bare deck).
    "not an animal",
}


def _web(p):
    return p.replace("\\", "/") if p else None


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _sun(cfg, d):
    """(dawn, dusk) tz-aware datetimes for calendar date `d` AT THE CAMERA, from
    daynight.sun_times when lat/lon (and astral) are available, else a 06:00/18:00 fallback.

    The reference frame is the YARD's, derived from longitude -- never the server's clock. That
    distinction is not pedantry: astral returns "the dawn/dusk falling on this calendar date IN
    THIS TIMEZONE", so asking in the machine's zone splits the pair across two local days as soon
    as the machine disagrees with the camera. Measured at this yard (lat 47.5, lon -122.2) from a
    UTC machine: dawn 2026-08-07T12:19Z but dusk 2026-08-07T04:10Z -- dusk BEFORE dawn, a negative
    day length, and every period boundary, moon bucket and sun-anchored arrival built on top of it
    quietly wrong. It never showed here because this rig's clock happens to match its own yard.
    It stops being hypothetical the moment an archive is read somewhere else, which --serve-only
    now invites, and it is the same principle as the rest of the project: derive from the camera,
    not from incidental machine state.

    round(lon / 15) is the yard's solar offset to the nearest hour. It is deliberately NOT the
    civil timezone (no DST, no political boundaries) because nothing here needs one: it exists
    only to name the local DAY the sun times belong to, and at this latitude it returns instants
    identical to the correct local-zone answer."""
    lat, lon = getattr(cfg, "latitude", None), getattr(cfg, "longitude", None)
    key = (d.toordinal(), lat, lon)
    if key in _SUN_CACHE:
        return _SUN_CACHE[key]
    tz = (timezone(timedelta(hours=round(lon / 15))) if lon is not None
          else datetime.now().astimezone().tzinfo)   # no location -> the clock is all there is
    base = datetime.combine(d, time(12, 0), tzinfo=tz)
    res = None
    if lat is not None and lon is not None:
        try:
            import daynight
            st = daynight.sun_times(lat, lon, base)
            res = (st["dawn"], st["dusk"])
            if res[1] <= res[0]:        # never hand back a negative day (polar dates, odd tz)
                res = None
        except Exception:
            res = None
    if res is None:
        res = (datetime.combine(d, time(6, 0), tzinfo=tz),
               datetime.combine(d, time(18, 0), tzinfo=tz))
    _SUN_CACHE[key] = res
    return res


def _iter_completed(cfg, now, max_back=40):
    """All completed sun-periods, newest-first: each is (start, end, label, anchor_date).
    A 'day' anchored on date d is [dawn(d), dusk(d)]; a 'night' anchored on d is the night
    that ENDS on the morning of d, i.e. [dusk(d-1), dawn(d)]."""
    res, today = [], now.date()
    for back in range(0, max_back + 1):
        d = today - timedelta(days=back)
        dawn_d, dusk_d = _sun(cfg, d)
        _, dusk_p = _sun(cfg, d - timedelta(days=1))
        for c in ((dawn_d, dusk_d, "day", d), (dusk_p, dawn_d, "night", d)):
            if c[1] <= now:                      # only COMPLETED periods
                res.append(c)
    res.sort(key=lambda c: c[1], reverse=True)
    seen, out = set(), []
    for c in res:
        k = (c[0].isoformat(), c[1].isoformat())
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _prior_windows(cfg, label, anchor, k=12):
    """The k periods of the SAME type immediately before `anchor` (newest-first),
    each (start, end, date) -- the baseline for streaks and quiet-regular flags."""
    out, d = [], anchor
    while len(out) < k:
        d = d - timedelta(days=1)
        dawn_d, dusk_d = _sun(cfg, d)
        if label == "day":
            out.append((dawn_d, dusk_d, d))
        else:
            _, dusk_p = _sun(cfg, d - timedelta(days=1))
            out.append((dusk_p, dawn_d, d))
    return out


def _fmt_hour(h: int) -> str:
    return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"


def _typical_window(hist, frac=0.7):
    """Shortest CIRCULAR run of hours (handles the around-midnight case) whose detections
    sum to >= `frac` of the total -> 'When does this species usually show?' e.g. '9pm-1am'."""
    total = sum(hist)
    if total <= 0:
        return None
    target = total * frac
    best = None
    for s in range(24):
        csum = 0
        for length in range(1, 25):
            csum += hist[(s + length - 1) % 24]
            if csum >= target:
                if best is None or length < best[0]:
                    best = (length, s, (s + length - 1) % 24)
                break
    if not best:
        return None
    if best[1] == best[2]:                      # a single dominant hour
        return f"~{_fmt_hour(best[1])}"
    return f"{_fmt_hour(best[1])}–{_fmt_hour(best[2])}"


def _moon(d):
    """Moon name/glyph/illumination for date `d`, or None if astral is unavailable."""
    try:
        from astral import moon
        p = float(moon.phase(d))
    except Exception:
        return None
    name, glyph = _MOON_PHASES[-1][1], _MOON_PHASES[-1][2]
    for edge, nm, gl in _MOON_PHASES:
        if p < edge:
            name, glyph = nm, gl
            break
    illum = round(50.0 * (1.0 - math.cos(2.0 * math.pi * p / 28.0)))
    return {"phase": round(p, 1), "name": name, "glyph": glyph, "illum_pct": illum}


def _title_for(label, anchor, today) -> str:
    delta = (today - anchor).days
    if label == "night":
        if delta <= 0:
            return "Last Night"
        if delta == 1:
            return "The Night Before"
        return "Night of " + anchor.strftime("%a, %b ") + str(anchor.day)
    if delta <= 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    return anchor.strftime("%a, %b ") + str(anchor.day)


def resolve_period(cfg, edition="auto", now=None, date=None):
    """The completed sun-period a digest/reel request refers to: (start, end, label, anchor)
    or None. `date`='YYYY-MM-DD' pins the anchor day (the Dispatch's back-in-time nav);
    without it, the most recently completed period (of `edition`, or either for 'auto')."""
    now = now or _local_now()
    completed = _iter_completed(cfg, now)
    if date:
        try:
            want = datetime.fromisoformat(str(date)).date()
        except ValueError:
            return None
        return next((c for c in completed
                     if c[3] == want and (edition == "auto" or c[2] == edition)), None)
    return next((c for c in completed if edition == "auto" or c[2] == edition), None)


# The digest re-reads the whole detections table per call; a completed period barely changes
# (only label corrections touch it), so a short-lived cache makes tab flips and the Dispatch's
# day-to-day nav instant. Keyed by (edition, date); skipped entirely when a caller pins `now`
# (tests need determinism). POST label/name actions clear it via clear_digest_cache().
_DIGEST_CACHE: dict = {}
_DIGEST_TTL_S = 90


def clear_digest_cache() -> None:
    _DIGEST_CACHE.clear()


def period_digest(cfg, edition="auto", now=None, date=None, *, regular_frac=0.4, novelty_days=3) -> dict:
    """Summarize one completed period. `edition`: 'auto' (whichever ended most recently ->
    night in the morning, day in the evening), or force 'day'/'night'; `date` pins the anchor
    day for back-in-time navigation.

    Returns a JSON-able dict: edition, title, start/end, prev/next nav anchors, totals, a
    chronological `visit_log` (the who-came-and-when timeline: species mix, named individuals,
    clips, motion), a per-species roll (visits, rep crop, first/last, novelty, streak, typical
    hours + 24-bucket histogram), headline `novel` species, `quiet` regulars that didn't show,
    `plate` of the period, busiest hour, and `moon` (night editions). `empty:true` when nobody
    showed."""
    cache_key = (str(edition), str(date) if date else None, regular_frac, novelty_days)
    cacheable = now is None
    if cacheable:
        hit = _DIGEST_CACHE.get(cache_key)
        if hit and hit[0] > _time.time():
            return hit[1]

    def done(payload):
        if cacheable:
            if len(_DIGEST_CACHE) >= 48:                  # bound it over a long-lived server
                _DIGEST_CACHE.pop(next(iter(_DIGEST_CACHE)), None)
            _DIGEST_CACHE[cache_key] = (_time.time() + _DIGEST_TTL_S, payload)
        return payload

    now = now or _local_now()
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"empty": True, "edition": edition, "reason": "no database yet"}
    raw = conn.execute(
        "SELECT id, timestamp, source, detection_class, species, confidence, species_confidence, "
        "species_verified, crop_path, crop_quality, individual_id, "
        "bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h "
        "FROM detections ORDER BY timestamp").fetchall()
    clips = load_clips(conn)

    rows = []
    for r in raw:
        dt = _parse(r["timestamp"])
        if dt is None:
            continue
        rows.append({
            "id": r["id"], "dt": dt, "timestamp": r["timestamp"], "source": r["source"],
            "detection_class": r["detection_class"], "species": r["species"],
            "confidence": r["confidence"] or 0.0, "species_confidence": r["species_confidence"],
            "verified": r["species_verified"],
            "crop_path": r["crop_path"], "label": r["species"] or r["detection_class"],
            "crop_quality": r["crop_quality"], "individual_id": r["individual_id"],
            "bbox_x1": r["bbox_x1"], "bbox_y1": r["bbox_y1"], "bbox_x2": r["bbox_x2"],
            "bbox_y2": r["bbox_y2"], "frame_w": r["frame_w"], "frame_h": r["frame_h"],
        })
    # Drop false triggers / non-visitor human labels up front so every downstream tally
    # (roll, plate, novelty, streaks) is about actual animals.
    rows = [r for r in rows if (r["label"] or "").lower() not in _NON_CRITTER]

    def in_window(s, e):
        return [r for r in rows if s <= r["dt"] < e]

    completed = _iter_completed(cfg, now)
    chosen = resolve_period(cfg, edition, now, date)
    if chosen is None:
        conn.close()
        reason = "no completed period yet" if not date else f"no {edition} period on {date}"
        return done({"empty": True, "edition": edition, "reason": reason})

    start, end, label, anchor = chosen
    pr = in_window(start, end)
    backed_off = False
    # Nothing in the most recent period and the caller didn't force an edition or a date? Fall
    # back to the most recent period (either type) that has detections, so the page is never blank.
    if not pr and edition == "auto" and not date:
        for c in completed:
            cand = in_window(c[0], c[1])
            if cand:
                start, end, label, anchor, pr = c[0], c[1], c[2], c[3], cand
                backed_off = True
                break

    # Back-in-time nav: the anchors one day either side of this one, bounded by the first day
    # of data and by "is that period completed yet". The dashboard's ‹ › walk on these.
    first_day = rows[0]["dt"].date() if rows else None
    prev_anchor = anchor - timedelta(days=1)
    next_anchor = anchor + timedelta(days=1)
    prev_date = prev_anchor.isoformat() if (first_day and prev_anchor >= first_day) else None
    next_date = (next_anchor.isoformat()
                 if any(c[3] == next_anchor and c[2] == label for c in completed) else None)

    today = now.date()
    out = {
        "edition": label, "title": _title_for(label, anchor, today),
        "start": start.isoformat(), "end": end.isoformat(), "backed_off": backed_off,
        "anchor": anchor.isoformat(), "prev_date": prev_date, "next_date": next_date,
        "latest": not date,
        "moon": _moon(anchor - timedelta(days=1)) if label == "night" else None,
    }
    if not pr:
        conn.close()
        out.update({"empty": True, "visits": 0, "crops": 0, "species": [],
                    "novel": [], "quiet": [], "visit_log": []})
        return done(out)

    # Clip-motion tracks over the window, for the visit log's one-line motion strips. Fetched
    # while the read-only conn is still open; bucketed per visit further down.
    try:
        period_tracks = conn.execute(
            "SELECT c.source src, c.started_at sa, t.moving_frac mf, t.straightness st, "
            "t.area_trend tr FROM clip_tracks t JOIN clips c ON c.id = t.clip_id "
            "WHERE c.started_at >= ? AND c.started_at <= ?",
            ((start - timedelta(minutes=5)).isoformat(), end.isoformat())).fetchall()
    except Exception:
        period_tracks = []                       # an older DB without clip_tracks
    # EFFORT: was the primary camera even watching this period? (coverage_events ledger --
    # written by the rig at open/read-fail/reconnect/stop since 2026-08-08.) None = unknown
    # (the ledger predates this period), and unknown stays silent: absence of evidence must
    # never render as evidence of coverage. With a known dark stretch, the Dispatch says so
    # beside its absence claims -- a dark camera is not an empty yard.
    dark_s = db.coverage_dark_seconds(conn, cfg.source, start, end)
    conn.close()

    # --- visits over the period, attributed to every species they contain ---
    visits = compute_visits(pr, cfg.visit_gap_minutes, rep_key=_shot_score)
    visits.sort(key=lambda v: v["start"])
    per_sp_visits = Counter()
    for v in visits:
        for sp in v["classes"]:
            per_sp_visits[sp] += 1

    # --- the visit log: the period's visits in order -- the "who came, and when" timeline the
    # Dispatch leads with. Each entry carries its species mix, any NAMED individuals (from the
    # crops' phase-3 ids; placeholder clusters excluded), its clips, and a motion one-liner
    # aggregated from the clip-tracks that rolled during it. ---
    tracks_parsed = []
    for t in period_tracks:
        sdt = _parse(t["sa"])
        if sdt is not None:
            tracks_parsed.append((t["src"], sdt, t["mf"], t["st"], t["tr"]))

    visit_log = []
    for v in visits:
        individuals = _named_of(v)
        vtracks = [t for t in tracks_parsed
                   if t[0] == v["source"] and v["start"] - timedelta(minutes=5) <= t[1] <= v["end"]]
        motion = None
        if vtracks:
            mf = [t[2] for t in vtracks if t[2] is not None]
            st = [t[3] for t in vtracks if t[3] is not None]
            tr = [t[4] for t in vtracks if t[4] is not None]
            mins = (v["end"] - v["start"]).total_seconds() / 60.0
            mfm = round(sum(mf) / len(mf), 2) if mf else None
            stm = round(sum(st) / len(st), 2) if st else None
            motion = {
                "tracks": len(vtracks),
                "approach": _approach_label(sum(tr) / len(tr) if tr else None),
                "straightness": stm,
                "moving_frac": mfm,
                "tag": _behaviour_tag(mins, mfm, stm),
            }
        visit_log.append({
            "start": v["start"].isoformat(), "end": v["end"].isoformat(),
            "minutes": round((v["end"] - v["start"]).total_seconds() / 60.0, 1),
            "count": v["count"], "source": v["source"],
            "species": [sp for sp, _n in v["classes"].most_common()],
            "individuals": individuals,
            "rep_crop": _web(v.get("rep_crop")),
            "clips": [_clip_out(c) for c in clips_overlapping(clips, v["source"], v["start"], v["end"])],
            "motion": motion,
        })

    # --- all-time per-species history (for novelty + activity clock) ---
    sp_all = defaultdict(list)
    for r in rows:                 # rows are globally timestamp-sorted
        sp_all[r["label"]].append(r)

    def hours_hist(rs):
        h = [0] * 24
        for r in rs:
            h[r["dt"].hour] += 1
        return h

    # --- presence across current + prior same-type windows (streak & quiet) ---
    priors = _prior_windows(cfg, label, anchor)
    window_species = [set(r["label"] for r in pr)]
    window_species += [set(r["label"] for r in in_window(s, e)) for s, e, _d in priors]

    def streak_of(sp):
        n = 0
        for ws in window_species:
            if sp in ws:
                n += 1
            else:
                break
        return n

    crops_by_sp = Counter(r["label"] for r in pr)
    species_roll = []
    for sp, crops in crops_by_sp.most_common():
        srs = [r for r in pr if r["label"] == sp]
        rep = max(srs, key=_shot_score)        # lead the roll row with this species' cutest frame
        prev = [r for r in sp_all[sp] if r["dt"] < start]
        prev_dt = prev[-1]["dt"] if prev else None
        hist = hours_hist(sp_all[sp])
        species_roll.append({
            "species": sp, "visits": per_sp_visits.get(sp, 0), "crops": crops,
            "rep_crop": _web(rep["crop_path"]),
            "rep_conf": round(rep["species_confidence"] or rep["confidence"] or 0.0, 3),
            "first": min(r["dt"] for r in srs).isoformat(),
            "last": max(r["dt"] for r in srs).isoformat(),
            "novelty": {"first_ever": prev_dt is None,
                        "days_since": ((start.date() - prev_dt.date()).days if prev_dt else None)},
            "streak": streak_of(sp), "typical": _typical_window(hist), "hours": hist,
            "active_hours": sorted({r["dt"].hour for r in srs}),   # hours active THIS period
            # The clip behind this species' best frame this period (so the roll row plays it).
            "clip": _clip_out(clip_at(clips, rep["source"], rep["dt"])),
        })

    # --- SURPRISING SIGHTINGS: the temporal prior turned back onto the label (2026-08-08) ---
    # A night roll listing goldfinches at 2 AM is almost never a goldfinch -- it is a kit-melee
    # crop the classifier forced onto the nearest species, and counting it made one family
    # night read "27 species". The Off-Pattern panel already KNEW ("usually 6am-10am" printed
    # beside a midnight sighting) but the knowledge was display-only; this feeds it back into
    # what the digest asserts. Test: the share of a species' REFERENCE activity that falls in
    # this edition's hours -- the reference being its human-VERIFIED crops when at least a
    # dozen verdicts exist, else all its crops. Verified-first is what breaks the circularity:
    # the all-time histogram of a polluted label is polluted too (this DB's "Anna's hummingbird,
    # usually 16-01h"), but the verified subset is the human's own eyes. Under 15% => flagged
    # `surprising`: still listed (its crops are exactly the ones worth a ✎), sorted last,
    # excluded from the headline species count and from novelty -- listed as a question, not
    # asserted as fauna. A verified crop IN this period clears the flag outright.
    period_hours = set()
    _cur = start
    while _cur < end:
        period_hours.add(_cur.hour)
        _cur += timedelta(hours=1)
    n_surprising = 0
    for s in species_roll:
        sp = s["species"]
        if sp == "animal":
            continue
        srs = [r for r in pr if r["label"] == sp]
        if any(r.get("verified") == 1 for r in srs):
            continue                                   # the human has seen one this period: real
        ref = [r for r in sp_all[sp] if r.get("verified") == 1]
        if len(ref) < 12:
            ref = sp_all[sp]
        if not ref:
            continue
        in_band = sum(1 for r in ref if r["dt"].hour in period_hours)
        frac = in_band / len(ref)
        if frac < 0.15:
            s["surprising"] = True
            s["surprise_note"] = (f"only {round(frac * 100)}% of this species' "
                                  f"{'verified ' if len(ref) != len(sp_all[sp]) else ''}record falls "
                                  f"in {label} hours -- likely a mislabeled crop; worth a ✎")
            n_surprising += 1
    species_roll.sort(key=lambda s: bool(s.get("surprising")))   # stable: questions sink last

    # --- headline novelty (first-ever, then rarest first) + quiet regulars ---
    alltime = {sp: len(rs) for sp, rs in sp_all.items()}
    novel_cands = [s for s in species_roll if s["species"] != "animal"
                   and not s.get("surprising")
                   and (s["novelty"]["first_ever"] or (s["novelty"]["days_since"] or 0) >= novelty_days)]
    novel_cands.sort(key=lambda s: (not s["novelty"]["first_ever"], alltime.get(s["species"], 0)))
    novel = [s["species"] for s in novel_cands[:6]]
    present_now, prior_ws = window_species[0], window_species[1:]
    quiet = []
    if prior_ws:
        seen_counts = Counter(sp for ws in prior_ws for sp in ws if sp != "animal")
        for sp, c in seen_counts.items():
            frac = c / len(prior_ws)
            if frac >= regular_frac and sp not in present_now:
                since = 1
                for ws in prior_ws:
                    if sp in ws:
                        break
                    since += 1
                quiet.append({"species": sp, "frac": round(frac, 2), "periods_since": since})
        quiet.sort(key=lambda q: -q["frac"])

    # --- plate of the period (the single best/cutest shot), first/last visitor, busiest hour ---
    plate = max(pr, key=_shot_score)
    busy = Counter(v["start"].hour for v in visits)
    bh = busy.most_common(1)[0] if busy else None

    # --- highlight reel: each in-window clip fronted by its CUTEST frame (best _shot_score, not
    # just the most-confident). Drop clips whose best shot is weak relative to the night -- the
    # all-blurry / tiny / distant ones -- keep the cutest REEL_LIMIT, then replay in time order so
    # the reel relives the night. Only clips that actually caught a real visitor are considered. ---
    cand = []
    for c in clips:
        if c["edt"] < start or c["sdt"] > end:
            continue
        inside = [r for r in pr if c["sdt"] <= r["dt"] <= c["edt"]]
        if not inside:
            continue
        best = max(inside, key=_shot_score)
        cand.append({**_clip_out(c), "thumb": _web(best["crop_path"]),
                     "species": Counter(r["label"] for r in inside).most_common(1)[0][0],
                     "_score": _shot_score(best)})
    reel = []
    if cand:
        median = sorted(x["_score"] for x in cand)[len(cand) // 2]
        floor = 0.45 * median                          # drop shots well below the night's typical
        kept = [x for x in cand if x["_score"] >= floor] or cand
        kept.sort(key=lambda x: x["_score"], reverse=True)              # cutest first ...
        reel = sorted(kept[:REEL_LIMIT], key=lambda x: x["start"])      # ... then replay in order
        for x in reel:
            x.pop("_score", None)

    period_s = max(1.0, (end - start).total_seconds())
    out.update({
        "empty": False, "visits": len(visits), "crops": len(pr), "species": species_roll,
        "n_surprising": n_surprising,
        "coverage": (None if dark_s is None else
                     {"source": cfg.source, "dark_minutes": round(dark_s / 60),
                      "frac_dark": round(dark_s / period_s, 2)}),
        "novel": novel, "quiet": quiet[:4], "reel": reel, "visit_log": visit_log,
        "plate": {"crop_path": _web(plate["crop_path"]), "species": plate["label"],
                  "conf": round(plate["species_confidence"] or plate["confidence"] or 0.0, 3),
                  "time": plate["dt"].isoformat(),
                  "clip": _clip_out(clip_at(clips, plate["source"], plate["dt"]))},
        "first_visitor": {"species": pr[0]["label"], "time": pr[0]["dt"].isoformat()},
        "last_visitor": {"species": pr[-1]["label"], "time": pr[-1]["dt"].isoformat()},
        "busiest_hour": ({"hour": bh[0], "visits": bh[1]} if bh else None),
    })
    return done(out)
