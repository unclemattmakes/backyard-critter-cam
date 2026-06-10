"""
Phase 4 -- THE TWO-AXIS READ-OUT (the payoff).

PLAN.md's core idea: never collapse to a single "who is this?" answer. Put the APPEARANCE match
(reid.py) NEXT TO the BEHAVIOUR fit (behavior.py) and surface BOTH -- the information is exactly
when they DISAGREE: "looks like Notch, but isn't acting like Notch" = either a new look-alike or a
known individual in an unusual state.

Two layers, because individual labels are still thin placeholder clusters but species behaviour is
already rich:

  * SPECIES fit (works now): does a visit's arrival time + dwell fit how that species usually
    behaves? A "raccoon" at 11am, or a "crow" at 3am, DISAGREES with the species' behaviour --
    which flags either a genuinely unusual event or a mis-classification. Immediately useful.
  * INDIVIDUAL match (the aspirational target): the labelled individual the visit's most-readable
    crop looks most like, with that individual's behaviour fit beside it. Sharpens as you
    hand-label individuals (reid.py --name) and they accumulate visits.

Appearance is deliberately only ONE axis -- it's weak on raccoons through glass (see reid.py) --
which is the whole reason behaviour rides alongside it.

  python twoaxis.py                 # scan recent visits, flag any appearance/behaviour disagreement
  python twoaxis.py --species raccoon --limit 20
  python twoaxis.py --visit 57      # full two-axis read-out for one visit
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime

import config
import db
import behavior
from stats import _NON_CRITTER


def _parse(ts):
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _in_window(hour, win):
    """Is `hour` inside a circular typical window dict (from behavior.typical_window)?"""
    if not win:
        return True
    s, e = win["start_hour"], win["end_hour"]
    return s <= hour <= e if s <= e else (hour >= s or hour <= e)


def species_fit(visit, prof):
    """Score a visit against its species' behaviour profile. Returns (verdict, notes)."""
    if prof is None or prof["n_visits"] < 3:
        return "thin", ["species profile too thin to judge"]
    dt = _parse(visit["started_at"])
    hour = dt.hour if dt else None
    dwell = (_parse(visit["ended_at"]) - dt).total_seconds() if dt else 0
    notes = []
    arrival_ok = _in_window(hour, prof["typical_window"])
    win = prof["typical_window"]
    wtxt = f"{win['start_hour']:02d}-{win['end_hour']:02d}h" if win else "n/a"
    notes.append(f"arrived {hour:02d}h "
                 f"[{prof['label']} usually {wtxt}] {'OK' if arrival_ok else 'UNUSUAL'}")
    med = prof["dwell_median_s"] or 1
    ratio = dwell / med
    dwell_ok = 0.15 <= ratio <= 8 or dwell < 5
    notes.append(f"dwell {dwell:.0f}s [{prof['label']} median {med:.0f}s] "
                 f"{'OK' if dwell_ok else f'{ratio:.1f}x'}")
    verdict = "FITS" if arrival_ok else "DISAGREES"   # timing is the strong species signal
    return verdict, notes


def two_axis(conn, store, visit):
    """Build the full read-out dict for one visit: behaviour fit + (if embedded) appearance match."""
    species = visit["species"]
    prof = behavior.species_profile(conn, species) if species else None
    verdict, notes = species_fit(visit, prof)

    appearance = None
    rep = visit["representative_detection_id"]
    if store is not None and rep is not None:
        try:
            appearance = store.match(rep, top=3)
        except (KeyError, ValueError):
            appearance = None
    return {"verdict": verdict, "notes": notes, "appearance": appearance}


def _print_visit(visit, readout, full=False):
    dt = _parse(visit["started_at"])
    dwell = (_parse(visit["ended_at"]) - dt).total_seconds() if dt else 0
    tag = {"FITS": "  ", "DISAGREES": "!!", "thin": "..", "skip": "  "}.get(readout["verdict"], "  ")
    print(f"{tag} visit #{visit['id']:<4} {visit['species'] or '(unknown)':18} "
          f"{dt.strftime('%m-%d %H:%M') if dt else '?'}  dwell {dwell:>4.0f}s  "
          f"-> {readout['verdict']}")
    if full or readout["verdict"] == "DISAGREES":
        for n in readout["notes"]:
            print(f"       behaviour: {n}")
        if readout["appearance"]:
            looks = " | ".join(f"{ind} ({sim:.2f})" for ind, sim, _ in readout["appearance"])
            print(f"       appearance: looks like {looks}")


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4: the two-axis appearance/behaviour read-out.")
    p.add_argument("--species", default="raccoon",
                   help="Embed-able species to load appearance vectors for (default raccoon). "
                        "Behaviour fit is computed for every species regardless.")
    p.add_argument("--visit", type=int, default=None, help="Full read-out for one visit id.")
    p.add_argument("--limit", type=int, default=25, help="How many recent visits to scan (default 25).")
    args = p.parse_args()

    conn = db.connect(config.CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    if conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 0:
        print("No visits yet. Run:  python visits.py")
        conn.close()
        return 0

    # Appearance vectors (optional -- only the embedded species get an appearance match).
    store = None
    try:
        from reid import EmbeddingStore
        from embed import model_tag
        store = EmbeddingStore(conn, args.species, 0.0, model_tag(False))
        if len(store) == 0:
            store = None
    except Exception:
        store = None

    if args.visit is not None:
        v = conn.execute("SELECT * FROM visits WHERE id = ?", [args.visit]).fetchone()
        if v is None:
            print(f"No visit #{args.visit}.")
            conn.close()
            return 1
        print(f"Two-axis read-out for visit #{args.visit}:")
        _print_visit(v, two_axis(conn, store, v), full=True)
        conn.close()
        return 0

    visits = conn.execute(
        "SELECT * FROM visits ORDER BY started_at DESC LIMIT ?", [args.limit]
    ).fetchall()
    print(f"Two-axis scan of the last {len(visits)} visit(s)  "
          f"(!! = appearance/behaviour disagreement):\n")
    flagged = 0
    for v in visits:
        sp = (v["species"] or "").lower()
        if sp in _NON_CRITTER:
            continue
        readout = two_axis(conn, store, v)
        _print_visit(v, readout)
        if readout["verdict"] == "DISAGREES":
            flagged += 1
    print(f"\n{flagged} visit(s) where behaviour disagreed with the species' usual pattern.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
