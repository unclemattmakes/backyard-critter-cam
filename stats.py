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
from datetime import datetime, time, timedelta
import math

import db

LATEST_LIMIT = 24          # how many recent crops to surface (gallery)
RECENT_VISITS_LIMIT = 20   # how many recent visit events to surface
REEL_LIMIT = 24            # max clips in a dispatch "highlight reel" (busiest kept, then time-ordered)


def _parse(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def compute_visits(rows, gap_minutes: float, rep_key=None):
    """Collapse time-ordered detection rows into visit events, per source.

    `rep_key(row) -> float` scores each crop for the visit's REPRESENTATIVE (thumbnail) pick; the
    highest-scoring crop wins. Defaults to detector confidence (the most readable crop). visits_page
    passes a portrait-aware score (confidence x how much of the frame the animal fills) so a visit's
    thumbnail leans toward a close, usable shot -- the nearest thing to "a photo of its face"
    without a face detector -- rather than whichever frame merely scored highest."""
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
                       "rep_score": None, "classes": Counter(), "rep_crop": None}
            conf = r["confidence"] or 0.0
            score = rep_key(r)
            cur["end"] = dt
            cur["count"] += 1
            if cur["rep_score"] is None or score > cur["rep_score"]:   # keep the best thumbnail crop
                cur["rep_crop"] = r["crop_path"]
                cur["rep_score"] = score
            cur["max_conf"] = max(cur["max_conf"], conf)
            cur["classes"][r["species"] or r["detection_class"]] += 1
        if cur is not None:
            visits.append(cur)
    return visits


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

def load_clips(conn) -> list:
    """Every clip, parsed once for in-memory time-overlap matching. Each entry carries the parsed
    span (sdt/edt) plus the URL-friendly fields the dashboard needs. [] if there are no clips."""
    try:
        rows = conn.execute(
            "SELECT source, clip_path, started_at, ended_at, fps, width, height, "
            "frame_count, detection_count, max_confidence FROM clips ORDER BY started_at"
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
            "source": r["source"], "sdt": sdt, "edt": edt or sdt,
            "clip_path": _web(r["clip_path"]), "start": r["started_at"], "seconds": seconds,
            "dets": r["detection_count"] or 0,
            "conf": round(r["max_confidence"], 3) if r["max_confidence"] is not None else None,
        })
    return out


def _clip_out(c) -> dict | None:
    """JSON-able view of a clip (drops the parsed datetimes used only for matching)."""
    if not c:
        return None
    return {"clip_path": c["clip_path"], "start": c["start"], "seconds": c["seconds"],
            "dets": c["dets"], "conf": c["conf"]}


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
    rows = conn.execute(
        "SELECT id, source, timestamp, detection_class, species, confidence, "
        "species_confidence, crop_path "
        "FROM detections ORDER BY timestamp"
    ).fetchall()
    clips = load_clips(conn)
    conn.close()

    gap = cfg.visit_gap_minutes
    visits = compute_visits(rows, gap)

    def day(ts): return ts[:10]
    def web(p):  return p.replace("\\", "/") if p else None

    days_set = sorted({day(r["timestamp"]) for r in rows}) if rows else []
    src_crops = Counter(r["source"] for r in rows)
    src_visits = Counter(v["source"] for v in visits)
    cls_crops = Counter((r["species"] or r["detection_class"]) for r in rows)
    visits_by_day = Counter(v["start"].strftime("%Y-%m-%d") for v in visits)
    hour_counts = Counter(v["start"].hour for v in visits)

    by_day = []
    for d in days_set:
        drows = [r for r in rows if day(r["timestamp"]) == d]
        cls = Counter((r["species"] or r["detection_class"]) for r in drows)
        by_day.append({"day": d, "crops": len(drows),
                       "visits": visits_by_day.get(d, 0), "classes": dict(cls.most_common())})

    latest = [{
        "timestamp": r["timestamp"],
        "source": r["source"],
        "detection_class": r["detection_class"],
        "species": r["species"],
        "confidence": round(r["confidence"], 3),
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


def species_overview(cfg) -> dict | None:
    """Per-species rollup for the most/least-frequent display: count, avg confidence, review
    tallies, and a representative (verified-or-most-confident) crop. None if no DB yet."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return None
    rows = conn.execute(
        "SELECT species, COUNT(*) n, ROUND(AVG(species_confidence), 3) avg_conf, "
        "SUM(CASE WHEN species_verified = 1 THEN 1 ELSE 0 END) verified, "
        "SUM(CASE WHEN species_verified = 0 THEN 1 ELSE 0 END) rejected "
        "FROM detections WHERE species IS NOT NULL GROUP BY species ORDER BY n DESC"
    ).fetchall()
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
    conn.close()
    return {"species": species, "total": sum(s["count"] for s in species)}


def individuals_overview(cfg, thumbs: int = 6) -> dict:
    """Phase-3 labelling: every individual_id group (placeholder clusters like 'raccoon_c01' and
    hand-named individuals like 'Notch') with crop count, time span, dominant species, and a strip
    of its most-readable crops. Powers the dashboard's Individuals tab, where naming a group --
    or naming two groups the same -- is the cheap human step that turns clusters into a cast."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"groups": [], "total_crops": 0}
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
    conn.close()
    # Named individuals first (the cast), then placeholders by size (biggest worth naming first).
    groups.sort(key=lambda g: (g["placeholder"], -g["n_crops"]))
    return {"groups": groups, "total_crops": sum(g["n_crops"] for g in groups),
            "named": sum(1 for g in groups if not g["placeholder"])}


def species_crops(cfg, species: str, limit: int = 160) -> list:
    """Recent crops for one species (newest first) for the by-species browser."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return []
    rows = conn.execute(
        "SELECT id, crop_path, ROUND(species_confidence, 3) conf, species_verified, "
        "timestamp, source FROM detections WHERE species = ? ORDER BY id DESC LIMIT ?",
        (species, int(limit))).fetchall()
    conn.close()
    return [{"id": r["id"], "crop_path": (r["crop_path"] or "").replace("\\", "/"),
             "confidence": r["conf"], "verified": r["species_verified"],
             "timestamp": r["timestamp"], "source": r["source"]} for r in rows]


def crops_page(cfg, day=None, species=None, start=None, end=None, offset=0, limit=60) -> dict:
    """A filtered, paginated slice of detection crops (newest first) for the explorer drill-down.
    Filters (all optional, AND-combined): day='YYYY-MM-DD', species exact, start/end ISO timestamp
    bounds. Returns {crops, total, offset, limit} so the UI can show "loaded of total"."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"crops": [], "total": 0, "offset": 0, "limit": limit}
    where, args = [], []
    if day:     where.append("timestamp LIKE ?"); args.append(str(day) + "%")
    if species: where.append("species = ?");      args.append(species)
    if start:   where.append("timestamp >= ?");   args.append(start)
    if end:     where.append("timestamp <= ?");   args.append(end)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    total = conn.execute(f"SELECT COUNT(*) FROM detections {clause}", args).fetchone()[0]
    rows = conn.execute(
        f"SELECT id, timestamp, species, confidence, species_confidence, species_verified, "
        f"crop_path FROM detections {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
        args + [int(limit), int(offset)]).fetchall()
    conn.close()
    crops = [{"id": r["id"], "timestamp": r["timestamp"], "species": r["species"],
              "confidence": r["confidence"], "species_confidence": r["species_confidence"],
              "verified": r["species_verified"],
              "crop_path": (r["crop_path"] or "").replace("\\", "/")} for r in rows]
    return {"crops": crops, "total": total, "offset": int(offset), "limit": int(limit)}


def visits_page(cfg, day=None, limit=300) -> dict:
    """All visit events (newest first), each with a representative crop, for the explorer.
    Optionally restricted to a single day ('YYYY-MM-DD')."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"visits": [], "total": 0}
    clause, args = ("WHERE timestamp LIKE ?", [str(day) + "%"]) if day else ("", [])
    rows = conn.execute(
        f"SELECT id, source, timestamp, detection_class, species, confidence, species_confidence, "
        f"crop_path, crop_quality, bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h "
        f"FROM detections {clause} ORDER BY timestamp", args).fetchall()
    clips = load_clips(conn)
    conn.close()

    visits = compute_visits(rows, cfg.visit_gap_minutes, rep_key=_shot_score)
    visits.sort(key=lambda v: v["start"], reverse=True)
    out = []
    for v in visits[:limit]:
        title = v["classes"].most_common(1)[0][0] if v["classes"] else "animal"
        out.append({
            "start": v["start"].isoformat(), "end": v["end"].isoformat(), "source": v["source"],
            "count": v["count"], "minutes": round((v["end"] - v["start"]).total_seconds() / 60.0, 1),
            "max_conf": round(v["max_conf"], 3), "title": title,
            "classes": dict(v["classes"].most_common()),
            "rep_crop": ((v.get("rep_crop") or "").replace("\\", "/") or None),
            # The clips that rolled during this visit (busiest first) -> click the card, watch the
            # video. Empty for visits before clip recording was on (e.g. daytime pre-06-09).
            "clips": [_clip_out(c) for c in clips_overlapping(clips, v["source"], v["start"], v["end"])],
        })
    return {"visits": out, "total": len(visits)}


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

# Human-correction labels that aren't actual visitors -- false triggers (patio bricks, a blurry
# smear), the feeding station, or Matt himself. Excluded from the digest's roll / plate / novelty
# so "who visited" stays about animals. Genuine rare species (Douglas squirrel, Anna's hummingbird)
# are NOT here -- they're real and SHOULD surface. The unclassified coarse label 'animal' is kept
# in the roll (a real critter, just unnamed) but excluded from novelty/quiet headlines.
_NON_CRITTER = {
    # seen in this DB (human corrections of false triggers / the feeder / Matt himself)
    "bricks", "brick", "blur", "blurry", "cat food", "catfood", "food", "homeowner",
    "door", "porch", "broom", "chair",
    # likely future static-object corrections + generic non-visitors
    "fence", "wall", "table", "plant", "pot", "hose", "shadow", "reflection", "leaf", "leaves",
    "rock", "stick", "sticks", "ground", "tree", "bush", "person", "people", "human", "vehicle",
    "car", "unknown", "unidentified", "nothing", "empty", "none", "background", "n/a", "na", "",
}


def _web(p):
    return p.replace("\\", "/") if p else None


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _sun(cfg, d):
    """(dawn, dusk) tz-aware datetimes for calendar date `d`. Uses daynight.sun_times when
    lat/lon (and astral) are available, else a fixed 06:00/18:00 local fallback."""
    lat, lon = getattr(cfg, "latitude", None), getattr(cfg, "longitude", None)
    key = (d.toordinal(), lat, lon)
    if key in _SUN_CACHE:
        return _SUN_CACHE[key]
    base = datetime.combine(d, time(12, 0)).astimezone()
    res = None
    if lat is not None and lon is not None:
        try:
            import daynight
            st = daynight.sun_times(lat, lon, base)
            res = (st["dawn"], st["dusk"])
        except Exception:
            res = None
    if res is None:
        tz = base.tzinfo
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


def period_digest(cfg, edition="auto", now=None, *, regular_frac=0.4, novelty_days=3) -> dict:
    """Summarize the most-recently-completed period. `edition`: 'auto' (whichever ended most
    recently -> night in the morning, day in the evening), or force 'day'/'night'.

    Returns a JSON-able dict: edition, title, start/end, totals, a per-species roll (visits,
    rep crop, first/last, novelty, streak, typical hours + 24-bucket histogram), headline
    `novel` species, `quiet` regulars that didn't show, `plate` of the period, first/last
    visitor, busiest hour, and `moon` (night editions). `empty:true` when nobody showed."""
    now = now or _local_now()
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"empty": True, "edition": edition, "reason": "no database yet"}
    raw = conn.execute(
        "SELECT id, timestamp, source, detection_class, species, confidence, species_confidence, "
        "crop_path, crop_quality, bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h "
        "FROM detections ORDER BY timestamp").fetchall()
    clips = load_clips(conn)
    conn.close()

    rows = []
    for r in raw:
        dt = _parse(r["timestamp"])
        if dt is None:
            continue
        rows.append({
            "id": r["id"], "dt": dt, "timestamp": r["timestamp"], "source": r["source"],
            "detection_class": r["detection_class"], "species": r["species"],
            "confidence": r["confidence"] or 0.0, "species_confidence": r["species_confidence"],
            "crop_path": r["crop_path"], "label": r["species"] or r["detection_class"],
            "crop_quality": r["crop_quality"],
            "bbox_x1": r["bbox_x1"], "bbox_y1": r["bbox_y1"], "bbox_x2": r["bbox_x2"],
            "bbox_y2": r["bbox_y2"], "frame_w": r["frame_w"], "frame_h": r["frame_h"],
        })
    # Drop false triggers / non-visitor human labels up front so every downstream tally
    # (roll, plate, novelty, streaks) is about actual animals.
    rows = [r for r in rows if (r["label"] or "").lower() not in _NON_CRITTER]

    def in_window(s, e):
        return [r for r in rows if s <= r["dt"] < e]

    completed = _iter_completed(cfg, now)
    chosen = next((c for c in completed if edition == "auto" or c[2] == edition), None)
    if chosen is None:
        return {"empty": True, "edition": edition, "reason": "no completed period yet"}

    start, end, label, anchor = chosen
    pr = in_window(start, end)
    backed_off = False
    # Nothing in the most recent period and the caller didn't force an edition? Fall back to
    # the most recent period (either type) that has detections, so the page is never blank.
    if not pr and edition == "auto":
        for c in completed:
            cand = in_window(c[0], c[1])
            if cand:
                start, end, label, anchor, pr = c[0], c[1], c[2], c[3], cand
                backed_off = True
                break

    today = now.date()
    out = {
        "edition": label, "title": _title_for(label, anchor, today),
        "start": start.isoformat(), "end": end.isoformat(), "backed_off": backed_off,
        "moon": _moon(anchor - timedelta(days=1)) if label == "night" else None,
    }
    if not pr:
        out.update({"empty": True, "visits": 0, "crops": 0, "species": [],
                    "novel": [], "quiet": []})
        return out

    # --- visits over the period, attributed to every species they contain ---
    visits = compute_visits(pr, cfg.visit_gap_minutes)
    visits.sort(key=lambda v: v["start"])
    per_sp_visits = Counter()
    for v in visits:
        for sp in v["classes"]:
            per_sp_visits[sp] += 1

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

    # --- headline novelty (first-ever, then rarest first) + quiet regulars ---
    alltime = {sp: len(rs) for sp, rs in sp_all.items()}
    novel_cands = [s for s in species_roll if s["species"] != "animal"
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

    out.update({
        "empty": False, "visits": len(visits), "crops": len(pr), "species": species_roll,
        "novel": novel, "quiet": quiet[:4], "reel": reel,
        "plate": {"crop_path": _web(plate["crop_path"]), "species": plate["label"],
                  "conf": round(plate["species_confidence"] or plate["confidence"] or 0.0, 3),
                  "time": plate["dt"].isoformat(),
                  "clip": _clip_out(clip_at(clips, plate["source"], plate["dt"]))},
        "first_visitor": {"species": pr[0]["label"], "time": pr[0]["dt"].isoformat()},
        "last_visitor": {"species": pr[-1]["label"], "time": pr[-1]["dt"].isoformat()},
        "busiest_hour": ({"hour": bh[0], "visits": bh[1]} if bh else None),
    })
    return out
