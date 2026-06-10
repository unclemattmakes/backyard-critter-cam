"""
Phase 4 -- BEHAVIOUR profiles (the second axis).

PLAN.md's whole point is to keep APPEARANCE and BEHAVIOUR on separate axes and surface both,
so the system can say "looks like Notch, but isn't acting like Notch." reid.py is the appearance
axis; this is the behaviour axis. It reads the collapsed `visits` (run visits.py first) and, per
SPECIES and per INDIVIDUAL (placeholder clusters today, hand-labelled names later), profiles:

  * arrival pattern -- when they show up (hour-of-day histogram + a "typical window", computed
    on the 24-h circle so a crepuscular animal that spans midnight, e.g. raccoons at 20-23h AND
    01-05h, gets one sensible window instead of two);
  * dwell -- how long a visit lasts (median / max seconds);
  * frequency -- visits per active day;
  * co-occurrence -- which species turn up together (two species in one visit's crops). Crows run
    in family groups, so who-arrives-with-whom is a real individuating signal (PLAN.md).

These profiles are what the two-axis read-out (twoaxis.py) scores a new visit against. They work
at the SPECIES level on the data we already have; per-individual sharpens as re-ID labels improve.

  python behavior.py                       # overview: every critter species' profile + co-occurrence
  python behavior.py --species raccoon
  python behavior.py --individual raccoon_c01
  python behavior.py --co-occurrence
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from datetime import datetime

import config
import db
from stats import _NON_CRITTER   # shared denylist: false-trigger / non-visitor labels (door, blur, ...)


def _parse(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def typical_window(hour_counts: dict, coverage: float = 0.8):
    """Tightest circular [start_hour..end_hour] covering >= `coverage` of visits. Returns a dict
    or None. Circular so a span across midnight collapses to one window, not two."""
    total = sum(hour_counts.values())
    if total == 0:
        return None
    target = coverage * total
    best = None
    for start in range(24):
        acc = 0
        for w in range(24):
            acc += hour_counts.get((start + w) % 24, 0)
            if acc >= target:
                if best is None or (w + 1) < best[0]:
                    best = (w + 1, start, (start + w) % 24)
                break
    if best is None:
        return None
    return {"start_hour": best[1], "end_hour": best[2], "width_hours": best[0]}


def _dwells(visit_rows):
    out = []
    for r in visit_rows:
        a, b = _parse(r["started_at"]), _parse(r["ended_at"])
        if a and b:
            out.append((b - a).total_seconds())
    return out


def _profile(conn, where_sql, params, label):
    """Shared profiler for a species or an individual: the WHERE selects its visits."""
    rows = conn.execute(
        f"SELECT started_at, ended_at, detection_count, max_confidence "
        f"FROM visits WHERE {where_sql} ORDER BY started_at", params
    ).fetchall()
    if not rows:
        return None
    starts = [_parse(r["started_at"]) for r in rows]
    starts = [s for s in starts if s]
    hour_counts = Counter(s.hour for s in starts)
    days = {s.strftime("%Y-%m-%d") for s in starts}
    dwells = _dwells(rows)
    dwells.sort()
    return {
        "label": label,
        "n_visits": len(rows),
        "active_days": len(days),
        "visits_per_day": round(len(rows) / max(len(days), 1), 1),
        "first_seen": min(starts).isoformat() if starts else None,
        "last_seen": max(starts).isoformat() if starts else None,
        "arrival_hours": dict(hour_counts),
        "peak_hour": hour_counts.most_common(1)[0][0] if hour_counts else None,
        "typical_window": typical_window(hour_counts),
        "dwell_median_s": dwells[len(dwells) // 2] if dwells else 0,
        "dwell_max_s": max(dwells) if dwells else 0,
        "crops_per_visit": round(sum(r["detection_count"] for r in rows) / len(rows), 1),
    }


def species_profile(conn, species: str):
    return _profile(conn, "species = ?", [species], species)


def individual_profile(conn, individual_id: str):
    return _profile(conn, "individual_id = ?", [individual_id], individual_id)


def co_occurrence(conn):
    """Species pairs that share a visit (both appear among one visit's crops). Returns a Counter
    keyed by a sorted ('crow','raccoon') tuple -> number of visits they co-occurred in."""
    rows = conn.execute(
        "SELECT visit_id, species FROM detections "
        "WHERE visit_id IS NOT NULL AND species IS NOT NULL GROUP BY visit_id, species"
    ).fetchall()
    per_visit = {}
    for r in rows:
        sp = (r["species"] or "").lower()
        if sp in _NON_CRITTER:
            continue
        per_visit.setdefault(r["visit_id"], set()).add(r["species"])
    pairs = Counter()
    for species_set in per_visit.values():
        sl = sorted(species_set)
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                pairs[(sl[i], sl[j])] += 1
    return pairs


def _critter_species(conn):
    """Distinct dominant-species labels in visits, minus the false-trigger denylist."""
    rows = conn.execute("SELECT DISTINCT species FROM visits WHERE species IS NOT NULL").fetchall()
    return sorted(s["species"] for s in rows if (s["species"] or "").lower() not in _NON_CRITTER)


def _print_profile(p):
    if p is None:
        print("  (no visits)")
        return
    tw = p["typical_window"]
    win = (f"{tw['start_hour']:02d}h-{tw['end_hour']:02d}h ({tw['width_hours']}h span)"
           if tw else "n/a")
    spark = _sparkline(p["arrival_hours"])
    print(f"  {p['label']:22} {p['n_visits']:>3} visits  {p['visits_per_day']:>4}/day  "
          f"dwell~{p['dwell_median_s']:>4.0f}s  peak {p['peak_hour']:02d}h  window {win}")
    print(f"     hours |{spark}| (0..23h)")


def _sparkline(hour_counts: dict) -> str:
    bars = " .:-=+*#%@"
    peak = max(hour_counts.values()) if hour_counts else 0
    out = ""
    for h in range(24):
        n = hour_counts.get(h, 0)
        out += bars[0] if n == 0 else bars[min(len(bars) - 1, 1 + int((len(bars) - 2) * n / peak))]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4: behaviour profiles from visits.")
    p.add_argument("--species", help="Profile one species (e.g. raccoon).")
    p.add_argument("--individual", help="Profile one individual / placeholder cluster (e.g. raccoon_c01).")
    p.add_argument("--co-occurrence", action="store_true", help="Only show the co-occurrence table.")
    args = p.parse_args()

    conn = db.connect(config.CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    if conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 0:
        print("No visits yet. Run:  python visits.py")
        conn.close()
        return 0

    if args.species:
        _print_profile(species_profile(conn, args.species))
    elif args.individual:
        _print_profile(individual_profile(conn, args.individual))
    elif args.co_occurrence:
        pass  # fall through to co-occurrence print below
    else:
        print("Behaviour profiles by species (visits, frequency, dwell, arrival window):")
        for sp in _critter_species(conn):
            _print_profile(species_profile(conn, sp))

    if not args.species and not args.individual:
        co = co_occurrence(conn)
        if co:
            print("\nCo-occurrence (species sharing a visit):")
            for (a, b), n in co.most_common(12):
                print(f"  {a} + {b}: {n} visit(s)")
        else:
            print("\nNo multi-species visits recorded yet.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
