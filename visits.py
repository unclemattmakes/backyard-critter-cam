"""
Phase 4 -- VISIT-EVENT COLLAPSING (the honest activity unit).

One lingering critter fires many detections (a single dusk raccoon banked ~45 crops), so raw
crop counts over-count "visits" ~10x. This pass collapses consecutive detections on the same
`source` that are < gap minutes apart into one VISIT row (db `visits` table) and stamps each
detection with its `visit_id`. Every frequency / behaviour statistic downstream (behavior.py)
counts visits, not crops.

A visit records its DOMINANT species and individual_id (the most common label among its crops),
its span (started_at..ended_at -> dwell time), crop count, best confidence, and the most readable
crop as a representative. It can't yet split two animals present at once -- that needs reliable
per-individual re-ID -- which is the documented V1/phase-3 limitation.

Boring and robust: a full from-scratch rebuild each run (the detection set is small; simpler and
safer than incremental upkeep). Re-run it after new capture, after classify.py names crops, or
after reid.py assigns individuals.

  python visits.py                 # rebuild visits at the configured gap (config.visit_gap_minutes)
  python visits.py --gap 10        # ... with a 10-minute gap instead
  python visits.py --stats         # rebuild, then print a summary (visits, dwell, by species)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import config
import db


_parse = db.parse_local   # canonical ISO parser (tz-aware; normalises any legacy naive string)


def _dominant(counter: Counter):
    """Most common key in a Counter, or None if empty (used for a visit's species/individual)."""
    return counter.most_common(1)[0][0] if counter else None


def build_visits(conn, gap_minutes: float, *, verbose: bool = True) -> dict:
    """Rebuild the visits table from scratch and stamp detections.visit_id. Returns a summary."""
    gap = timedelta(minutes=gap_minutes)
    rows = conn.execute(
        """SELECT id, source, timestamp, species, individual_id, detection_class,
                  confidence, crop_path
           FROM detections ORDER BY source, timestamp"""
    ).fetchall()

    by_source = defaultdict(list)
    for r in rows:
        dt = _parse(r["timestamp"])
        if dt is not None:
            by_source[r["source"]].append((dt, r))

    db.clear_visits(conn)
    n_visits = 0
    n_stamped = 0

    def _flush(source, members):
        """Write one visit from its member (dt, row) tuples and stamp them."""
        nonlocal n_visits, n_stamped
        species = Counter(m[1]["species"] for m in members if m[1]["species"])
        indiv = Counter(m[1]["individual_id"] for m in members if m[1]["individual_id"])
        confs = [(m[1]["confidence"] or 0.0, m[1]["id"]) for m in members]
        best_conf, rep_id = max(confs, key=lambda c: c[0])
        vid = db.insert_visit(
            conn, source=source, species=_dominant(species), individual_id=_dominant(indiv),
            started_at=members[0][1]["timestamp"], ended_at=members[-1][1]["timestamp"],
            detection_count=len(members), max_confidence=best_conf,
            representative_detection_id=rep_id,
        )
        db.assign_visit(conn, [m[1]["id"] for m in members], vid)
        n_visits += 1
        n_stamped += len(members)

    for source, items in by_source.items():
        items.sort(key=lambda x: x[0])
        cur: list = []
        last_dt = None
        for dt, r in items:
            if cur and (dt - last_dt) > gap:
                _flush(source, cur)
                cur = []
            cur.append((dt, r))
            last_dt = dt
        if cur:
            _flush(source, cur)

    conn.commit()
    if verbose:
        print(f"Rebuilt {n_visits} visit(s) from {n_stamped} detection(s) "
              f"(gap {gap_minutes:g} min).")
    return {"visits": n_visits, "detections": n_stamped}


def print_stats(conn) -> None:
    """Quick read-back: visits + dwell by species, so you can eyeball the collapse."""
    total = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
    if not total:
        print("No visits yet -- capture some detections first.")
        return
    print(f"\n{total} visit(s). By species (dominant label):")
    rows = conn.execute(
        """SELECT COALESCE(species, '(unclassified)') sp, COUNT(*) v,
                  SUM(detection_count) crops, AVG(detection_count) avg_crops
           FROM visits GROUP BY sp ORDER BY v DESC"""
    ).fetchall()
    print(f"  {'species':22} {'visits':>7} {'crops':>7} {'crops/visit':>12}")
    for r in rows:
        print(f"  {r['sp']:22} {r['v']:>7} {r['crops']:>7} {r['avg_crops']:>12.1f}")

    # Dwell: how long visits last (ended-started), a behaviour signal in its own right.
    durs = []
    for r in conn.execute("SELECT started_at, ended_at FROM visits"):
        a, b = _parse(r["started_at"]), _parse(r["ended_at"])
        if a and b:
            durs.append((b - a).total_seconds())
    if durs:
        durs.sort()
        med = durs[len(durs) // 2]
        print(f"\nDwell (visit duration): median {med:.0f}s  max {max(durs):.0f}s  "
              f"({sum(1 for d in durs if d < 1):d} single-frame fly-bys)")


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4: collapse detections into visit events.")
    p.add_argument("--gap", type=float, default=None,
                   help="Minutes between detections that separates one visit from the next "
                        "(default: config.visit_gap_minutes).")
    p.add_argument("--stats", action="store_true", help="Print a visit summary after rebuilding.")
    args = p.parse_args()

    gap = args.gap if args.gap is not None else config.CONFIG.visit_gap_minutes
    conn = db.connect(config.CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    build_visits(conn, gap)
    if args.stats:
        print_stats(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
