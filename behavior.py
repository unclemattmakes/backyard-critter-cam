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


_parse = db.parse_local   # canonical ISO parser (tz-aware; normalises any legacy naive string)


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


def sun_anchor(starts, cfg):
    """Arrival times re-expressed as MINUTES AFTER the sun event that actually paces the animal
    -- the season-proof version of the clock-hour histogram. Clock hours smear as sunset walks
    (~45 min across this corpus already, accelerating into autumn): a raccoon that faithfully
    tracks dusk looks like it "drifts". Anchored, it just reads "~40 min after dusk" all year.

    Guild first: arrivals are NOCTURNAL when most of them fall outside [dawn..dusk] of their own
    day. The nocturnal guild anchors to DUSK -- and a post-midnight arrival to the PREVIOUS
    evening's dusk (the digest's own night convention: the night of day d runs dusk(d) ->
    dawn(d+1); subtracting same-date dusk from a 4 AM opossum yields nonsense negatives). The
    diurnal guild anchors to DAWN of its own day; minutes-after-sunset is meaningless for a
    junco. Events are stats._sun's CIVIL dawn/dusk (that's what the cache holds), so every label
    says "dusk", not "sunset" -- they differ by ~25-35 min at this latitude and the label must
    not lie. Returns None without lat/lon (stats._sun falls back to fixed 06:00/18:00, which
    would silently fake precision here).

    `weekly` carries the per-ISO-week median offset for weeks with >= 5 arrivals -- the drift
    line. Expect it to become legible over autumn; the headline median is useful today."""
    if getattr(cfg, "latitude", None) is None or getattr(cfg, "longitude", None) is None:
        return None
    from stats import _sun   # lazy: stats imports only db+stdlib, so no cycle; reuse its cache
    from datetime import timedelta
    if not starts:
        return None
    offs = []                                   # (start_dt, offset_minutes)
    nocturnal_votes = 0
    for s in starts:
        dawn, dusk = _sun(cfg, s.date())
        if s < dawn or s >= dusk:
            nocturnal_votes += 1
    nocturnal = nocturnal_votes > len(starts) / 2
    for s in starts:
        dawn, dusk = _sun(cfg, s.date())
        if nocturnal:
            anchor = _sun(cfg, s.date() - timedelta(days=1))[1] if s < dawn else dusk
        else:
            anchor = dawn
        offs.append((s, round((s - anchor).total_seconds() / 60.0)))
    offs.sort(key=lambda x: x[1])
    med = offs[len(offs) // 2][1]
    weekly = Counter()
    by_week = {}
    for s, o in offs:
        wk = s.strftime("%G-W%V")
        by_week.setdefault(wk, []).append(o)
    weekly = [{"week": wk, "median_offset_min": sorted(v)[len(v) // 2], "n": len(v)}
              for wk, v in sorted(by_week.items()) if len(v) >= 5]
    return {"anchor": "dusk" if nocturnal else "dawn", "median_offset_min": med,
            "n": len(offs), "weekly": weekly}


def species_profile(conn, species: str, cfg=None):
    p = _profile(conn, "species = ?", [species], species)
    if p and cfg is not None:
        starts = [s for s in (_parse(r["started_at"]) for r in conn.execute(
            "SELECT started_at FROM visits WHERE species = ?", [species])) if s]
        p["sun_anchor"] = sun_anchor(starts, cfg)
    return p


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


def yard_politics(conn, *, min_events: int = 8, horizon_h: float = 24.0):
    """Directional interaction between species -- the question 'seen together' can't answer,
    from data already in the visits table. Two measurements, both with sample floors so a thin
    pair never headlines:

      SUPPRESSION: after an A visit ends, how long until the next B arrival on the same camera,
      vs B's own baseline gap between visits? A factor well above 1 reads 'B stays away longer
      after A has been' (does the cat suppress the opossum?). Medians, same-source only, capped
      at `horizon_h` so an overnight absence doesn't drown the signal.

      YIELDING: B is mid-visit when A arrives -- does B's visit END within three minutes of A's
      arrival? The rate over all such encounters is who-gives-up-the-yard-to-whom.

    Both are DESCRIPTIVE (observation, not experiment: a shared cause -- dawn, rain, the dish
    refilled -- can move both species). The payload says so."""
    rows = conn.execute(
        "SELECT source, species, started_at, ended_at FROM visits "
        "WHERE species IS NOT NULL ORDER BY started_at").fetchall()
    by_src = {}
    for r in rows:
        sp = (r["species"] or "").lower()
        if sp in _NON_CRITTER:
            continue
        a, b = _parse(r["started_at"]), _parse(r["ended_at"])
        if a is None or b is None:
            continue
        by_src.setdefault(r["source"], []).append((a, b, r["species"]))

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    # GUILD GATE. Without it the top 'suppressions' were all nocturnal-vs-diurnal pairs at 30x+
    # -- a sparrow does not avoid a raccoon, it avoids the night; the day/night cycle IS the
    # shared cause the caveat warns about, so filter it out structurally: only compare species
    # whose active hours genuinely overlap (>= 2 shared hours, each holding >= 5% of that
    # species' arrivals). Cat-vs-opossum survives; raccoon-vs-goldfinch never fires.
    active_hours = {}
    for vs in by_src.values():
        for a, _b, sp in vs:
            active_hours.setdefault(sp, Counter())[a.hour] += 1
    def hours_of(sp):
        c = active_hours.get(sp, Counter())
        total = sum(c.values()) or 1
        return {h for h, n in c.items() if n / total >= 0.05}
    def share_hours(A, B):
        return len(hours_of(A) & hours_of(B)) >= 2

    # Baseline inter-arrival gap per species (minutes, capped at the horizon), per source pooled.
    base_gaps = Counter()
    gaps_by_sp = {}
    for src, vs in by_src.items():
        per_sp = {}
        for a, b, sp in vs:
            per_sp.setdefault(sp, []).append(a)
        for sp, starts in per_sp.items():
            for i in range(1, len(starts)):
                g = (starts[i] - starts[i - 1]).total_seconds() / 60.0
                if g <= horizon_h * 60:
                    gaps_by_sp.setdefault(sp, []).append(g)

    suppression, yields = [], []
    species = sorted({sp for vs in by_src.values() for _a, _b, sp in vs})
    for A in species:
        for B in species:
            if A == B or not share_hours(A, B):
                continue
            after, n_enc, n_yield = [], 0, 0
            for src, vs in by_src.items():
                b_starts = sorted(a for a, _b, sp in vs if sp == B)
                for a0, a1, sp in vs:
                    if sp != A:
                        continue
                    # SUPPRESSION: next B arrival after this A visit ends.
                    nxt = next((s for s in b_starts if s > a1), None)
                    if nxt is not None:
                        g = (nxt - a1).total_seconds() / 60.0
                        if g <= horizon_h * 60:
                            after.append(g)
                    # YIELDING: B mid-visit when A arrives; does B leave within 3 minutes?
                    for b0, b1, spB in vs:
                        if spB != B or not (b0 <= a0 <= b1):
                            continue
                        n_enc += 1
                        if (b1 - a0).total_seconds() <= 180:
                            n_yield += 1
            base = median(gaps_by_sp.get(B, []))
            aft = median(after)
            if base and aft and len(after) >= min_events:
                factor = aft / base
                if factor >= 1.5:
                    suppression.append({"a": A, "b": B, "n": len(after),
                                        "baseline_min": round(base), "after_min": round(aft),
                                        "factor": round(factor, 1)})
            if n_enc >= min_events and n_yield / n_enc >= 0.5:
                yields.append({"a": A, "b": B, "n_encounters": n_enc, "n_yield": n_yield,
                               "rate": round(n_yield / n_enc, 2)})
    suppression.sort(key=lambda x: -x["factor"])
    yields.sort(key=lambda x: -x["rate"])
    return {"suppression": suppression[:8], "yields": yields[:8],
            "note": "observational, not causal -- dawn, rain, or a refilled dish moves two "
                    "species at once; read as 'the pattern holds', never 'A causes B'"}


def moon_activity(conn, cfg, nights: int = 90):
    """Per-night nocturnal visit counts beside the moon's illumination -- the chart that turns
    the digest's decorative moon glyph into a question with an answer either way ('urban
    raccoons ignore the moon in a lit yard' is itself a finding). A night is the digest's own
    convention: dusk of day d to dawn of d+1 (stats._sun; fixed 18:00/06:00 without lat/lon).
    The payload carries its caveat -- yard lighting may dominate any lunar signal."""
    from stats import _sun, _moon
    from datetime import timedelta
    rows = conn.execute(
        "SELECT started_at, species FROM visits WHERE species IS NOT NULL").fetchall()
    per_night = Counter()
    for r in rows:
        if (r["species"] or "").lower() in _NON_CRITTER:
            continue
        s = _parse(r["started_at"])
        if s is None:
            continue
        dawn, dusk = _sun(cfg, s.date())
        if s >= dusk:
            per_night[s.date()] += 1
        elif s < dawn:
            per_night[s.date() - timedelta(days=1)] += 1
    if not per_night:
        return None
    recent = sorted(per_night)[-nights:]
    out = []
    for d in recent:
        m = _moon(d)
        out.append({"night": d.isoformat(), "n_visits": per_night[d],
                    "illum_pct": (m or {}).get("illum_pct")})
    return {"nights": out,
            "note": "nocturnal visits per night vs moon illumination; this yard has artificial "
                    "light, which may dominate any lunar signal"}


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


def overview(cfg, recent: int = 40) -> dict:
    """JSON-able behaviour summary for the dashboard (/api/behavior): per-species profiles, the
    co-occurrence table, and recent visits scored by species-fit (the two-axis disagreements).
    Read-only and WAL-safe. Returns {"need_rebuild": True} if the visits table is empty."""
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"visits": 0, "species": [], "co_occurrence": [], "flags": []}
    try:
        if conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 0:
            return {"visits": 0, "species": [], "co_occurrence": [], "flags": [],
                    "need_rebuild": True}
        species = []
        for sp in _critter_species(conn):
            p = species_profile(conn, sp, cfg)
            if not p:
                continue
            species.append({
                "species": sp, "n_visits": p["n_visits"], "visits_per_day": p["visits_per_day"],
                "dwell_median_s": p["dwell_median_s"], "peak_hour": p["peak_hour"],
                "typical_window": p["typical_window"], "arrival_hours": p["arrival_hours"],
                "crops_per_visit": p["crops_per_visit"],
                "sun_anchor": p.get("sun_anchor"),
            })
        species.sort(key=lambda s: -s["n_visits"])
        co = [{"a": a, "b": b, "n": n} for (a, b), n in co_occurrence(conn).most_common(12)]

        import twoaxis  # lazy: twoaxis imports behavior; this avoids a circular import at load
        flags = []
        rows = conn.execute("SELECT * FROM visits ORDER BY started_at DESC LIMIT ?",
                            [recent]).fetchall()
        for v in rows:
            sp = (v["species"] or "").lower()
            if sp in _NON_CRITTER:
                continue
            prof = species_profile(conn, v["species"]) if v["species"] else None
            verdict, notes = twoaxis.species_fit(v, prof)
            a, b = _parse(v["started_at"]), _parse(v["ended_at"])
            flags.append({
                "visit_id": v["id"], "species": v["species"],
                "started_at": v["started_at"],
                "dwell_s": int((b - a).total_seconds()) if a and b else 0,
                "rep_crop": _rep_crop(conn, v["representative_detection_id"]),
                "verdict": verdict, "notes": notes,
            })
        total = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        return {"visits": total, "species": species, "co_occurrence": co, "flags": flags,
                "moon": moon_activity(conn, cfg), "politics": yard_politics(conn)}
    finally:
        conn.close()


def _rep_crop(conn, detection_id):
    if detection_id is None:
        return None
    r = conn.execute("SELECT crop_path FROM detections WHERE id = ?", [detection_id]).fetchone()
    return r["crop_path"].replace("\\", "/") if r and r["crop_path"] else None


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
