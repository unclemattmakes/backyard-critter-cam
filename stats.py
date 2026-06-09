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
from datetime import datetime, timedelta

import db

LATEST_LIMIT = 24          # how many recent crops to surface (gallery)
RECENT_VISITS_LIMIT = 20   # how many recent visit events to surface


def _parse(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def compute_visits(rows, gap_minutes: float):
    """Collapse time-ordered detection rows into visit events, per source."""
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
                cur = {"source": source, "start": dt, "end": dt, "count": 0,
                       "max_conf": 0.0, "classes": Counter(), "rep_crop": None}
            conf = r["confidence"] or 0.0
            cur["end"] = dt
            cur["count"] += 1
            if cur["rep_crop"] is None or conf > cur["max_conf"]:   # keep the most readable crop
                cur["rep_crop"] = r["crop_path"]
            cur["max_conf"] = max(cur["max_conf"], conf)
            cur["classes"][r["species"] or r["detection_class"]] += 1
        if cur is not None:
            visits.append(cur)
    return visits


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
    } for r in list(reversed(rows))[:LATEST_LIMIT]]

    recent_visits = [{
        "source": v["source"],
        "start": v["start"].isoformat(),
        "end": v["end"].isoformat(),
        "count": v["count"],
        "minutes": round((v["end"] - v["start"]).total_seconds() / 60.0, 1),
        "max_conf": round(v["max_conf"], 3),
        "classes": dict(v["classes"].most_common()),
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
        f"SELECT id, source, timestamp, detection_class, species, confidence, crop_path "
        f"FROM detections {clause} ORDER BY timestamp", args).fetchall()
    conn.close()
    visits = compute_visits(rows, cfg.visit_gap_minutes)
    visits.sort(key=lambda v: v["start"], reverse=True)
    out = []
    for v in visits[:limit]:
        title = v["classes"].most_common(1)[0][0] if v["classes"] else "animal"
        out.append({
            "start": v["start"].isoformat(), "end": v["end"].isoformat(),
            "count": v["count"], "minutes": round((v["end"] - v["start"]).total_seconds() / 60.0, 1),
            "max_conf": round(v["max_conf"], 3), "title": title,
            "classes": dict(v["classes"].most_common()),
            "rep_crop": ((v.get("rep_crop") or "").replace("\\", "/") or None),
        })
    return {"visits": out, "total": len(visits)}
