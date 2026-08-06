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
  python visits.py --species-vote-report   # DRY RUN: which visits a confidence-weighted species
                                           # vote would relabel. Opens the DB READ-ONLY and writes
                                           # NOTHING -- see species_vote_report().
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
    """Most common key in a Counter, or None if empty (used for a visit's individual)."""
    return counter.most_common(1)[0][0] if counter else None


# ---------------------------------------------------------------------------
# The SPECIES VOTE.
#
# A visit's species is decided by its crops. The original vote was an unweighted crop-count mode,
# which lets VOLUME beat CERTAINTY: a static artifact firing 800 crops at 0.3 outvotes a real
# animal's 100 crops at 0.9, and the result is committed silently. Measured over the live DB
# (docs/identity-eval-2026-08-05.md C5), a >=0.8-confidence-only vote moves 108 of 2,416 visits,
# 12 of them INTO raccoon.
#
# That is not a free improvement: species GATES the re-ID gallery, so a raccoon relabelled opossum
# can never again be matched to Stan. So the weighted vote ships DISABLED (config.py) and there is
# a read-only report (species_vote_report) that says exactly which visits would move BEFORE
# anything moves. The margin is recorded either way -- a near-tie is a thing to surface, not bury.
# ---------------------------------------------------------------------------

def _species_vote(members, *, weighted: bool, min_confidence: float,
                  weight_by_confidence: bool = True):
    """Decide a visit's species from its member (dt, row) tuples.

    Returns (species, margin). `margin` is (winner - runner-up) / total vote weight, 0..1: 1.0 =
    every crop agreed, ~0 = a coin flip. Both are None when no crop carries a species.

    weighted=False reproduces the historical unweighted crop-count mode EXACTLY (Counter order,
    including its first-seen tie-break) so enabling the margin never moves a label by itself.

    weighted=True sums species_confidence per label over the crops that clear `min_confidence`. If
    NO crop clears the bar the whole visit falls back to the ungated weighted vote rather than
    going NULL -- a visit the classifier was unsure about is still a visit, and silently dropping
    its species would be a bigger change than the one being made. A crop with a species but NO
    species_confidence (a legacy or hand-set label) weighs 1.0: treating an unscored human label as
    low-confidence is the dangerous direction.

    weight_by_confidence=False keeps the GATE but counts the survivors instead of summing their
    confidences -- the narrower reading of "a >=0.8-confidence-only vote", and the one
    docs/identity-eval-2026-08-05.md measured at 108 moved visits. Measured 2026-08-05 on the live
    DB, the two readings are NOT the same: the gate alone moves 108 visits, gate-plus-weight moves
    158, and the extra 50 are near-ties the weights tipped (46 of the 158 land with margin < 0.10).
    Both are reported side by side by species_vote_report so the choice is made with the numbers
    in view."""
    pairs = [(m[1]["species"],
              1.0 if m[1]["species_confidence"] is None else float(m[1]["species_confidence"]))
             for m in members if m[1]["species"]]
    if not pairs:
        return None, None

    if not weighted:
        counts = Counter(s for s, _ in pairs)
        ranked = counts.most_common()
        total = float(sum(counts.values()))
    else:
        eligible = [p for p in pairs if p[1] >= min_confidence] or pairs
        totals: dict = {}
        for s, w in eligible:
            totals[s] = totals.get(s, 0.0) + (w if weight_by_confidence else 1.0)
        ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)  # stable: first-seen tie-break
        total = float(sum(totals.values()))

    winner, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = (top - runner_up) / total if total > 0 else None
    return winner, margin


def _group_detections(rows, gap_minutes: float):
    """Yield (source, members) visit groups from detection rows -- the ONE grouping definition,
    shared by the rebuild and by the read-only report so the report can never describe a
    different partition than the one that would actually be written. `members` are (dt, row)
    tuples in time order; rows with an unparseable timestamp are skipped."""
    gap = timedelta(minutes=gap_minutes)
    by_source = defaultdict(list)
    for r in rows:
        dt = _parse(r["timestamp"])
        if dt is not None:
            by_source[r["source"]].append((dt, r))
    for source, items in by_source.items():
        items.sort(key=lambda x: x[0])
        cur: list = []
        last_dt = None
        for dt, r in items:
            if cur and (dt - last_dt) > gap:
                yield source, cur
                cur = []
            cur.append((dt, r))
            last_dt = dt
        if cur:
            yield source, cur


_DETECTION_COLUMNS = """SELECT id, source, timestamp, species, species_confidence, individual_id,
                               detection_class, confidence, crop_path
                        FROM detections ORDER BY source, timestamp"""


def build_visits(conn, gap_minutes: float, *, verbose: bool = True,
                 weighted_species: bool | None = None,
                 species_min_confidence: float | None = None) -> dict:
    """Rebuild the visits table from scratch and stamp detections.visit_id. Returns a summary.

    `weighted_species` / `species_min_confidence` default to config (shipped disabled = the
    historical unweighted crop-count vote). THIS WRITES LABELS: run species_vote_report() first to
    see what a change of vote would move."""
    cfg = config.CONFIG
    if weighted_species is None:
        weighted_species = getattr(cfg, "species_vote_confidence_weighted", False)
    if species_min_confidence is None:
        species_min_confidence = getattr(cfg, "species_vote_min_confidence", 0.8)
    rows = conn.execute(_DETECTION_COLUMNS).fetchall()
    # Spans a human live-logged CONFLICTING names over -- flagged onto the visits they overlap so a
    # contested span is never mistaken for clean single-animal ground truth (db.record_live_sighting).
    # Parsed ONCE into instants, per source: the members already carry parsed datetimes, so the
    # per-visit test is a comparison and not another four fromisoformat calls.
    conflict_by_source: dict = defaultdict(list)
    for src, s0, s1 in db.conflicting_sighting_spans(conn):
        lo, hi = _parse(s0), _parse(s1)
        if lo is not None and hi is not None:
            conflict_by_source[src].append((lo, hi))

    db.clear_visits(conn)
    n_visits = 0
    n_stamped = 0
    n_conflict = 0

    def _flush(source, members):
        """Write one visit from its member (dt, row) tuples and stamp them."""
        nonlocal n_visits, n_stamped, n_conflict
        sp, margin = _species_vote(members, weighted=weighted_species,
                                   min_confidence=species_min_confidence)
        indiv = Counter(m[1]["individual_id"] for m in members if m[1]["individual_id"])
        confs = [(m[1]["confidence"] or 0.0, m[1]["id"]) for m in members]
        best_conf, rep_id = max(confs, key=lambda c: c[0])
        started, ended = members[0][1]["timestamp"], members[-1][1]["timestamp"]
        v0, v1 = members[0][0], members[-1][0]      # already-parsed instants
        conflict = any(lo <= v1 and v0 <= hi for lo, hi in conflict_by_source.get(source, ()))
        vid = db.insert_visit(
            conn, source=source, species=sp, individual_id=_dominant(indiv),
            started_at=started, ended_at=ended,
            detection_count=len(members), max_confidence=best_conf,
            representative_detection_id=rep_id,
            species_margin=margin, sighting_conflict=conflict,
        )
        db.assign_visit(conn, [m[1]["id"] for m in members], vid)
        n_visits += 1
        n_stamped += len(members)
        n_conflict += int(conflict)

    for source, members in _group_detections(rows, gap_minutes):
        _flush(source, members)

    conn.commit()
    if verbose:
        print(f"Rebuilt {n_visits} visit(s) from {n_stamped} detection(s) "
              f"(gap {gap_minutes:g} min"
              f"{', confidence-weighted species vote' if weighted_species else ''}).")
        if n_conflict:
            print(f"  {n_conflict} visit(s) flagged: a human live-logged conflicting names "
                  f"over the span.")
    return {"visits": n_visits, "detections": n_stamped, "sighting_conflicts": n_conflict}


def refresh(conn, gap_minutes: float) -> bool:
    """Best-effort rebuild for pipeline steps that change what visits would say -- new capture
    (rig shutdown, trail-cam import) and fresh species labels (classify.py). One shared helper so
    every step prints the same messages and honours the same contract: NEVER raise -- a failed
    refresh must not fail the capture/import/labeling that already succeeded (the fallback is the
    manual `python visits.py`). Subsecond at this scale. Returns True on success.

    build_visits reads columns by name, so row_factory is switched to sqlite3.Row for the rebuild
    and restored after -- callers like classify.py unpack rows positionally and keep their
    connection for further work."""
    prior_factory = conn.row_factory
    try:
        conn.row_factory = sqlite3.Row
        build_visits(conn, gap_minutes, verbose=False)
        print("  visit ledger refreshed.")
        return True
    except Exception as e:
        print(f"  [visits] could not refresh visit events (run `python visits.py`): {e}")
        return False
    finally:
        try:
            conn.row_factory = prior_factory
        except Exception:
            pass   # a dead connection can't take the restore -- nothing left to protect


def species_vote_report(conn, gap_minutes: float, *,
                        species_min_confidence: float | None = None) -> dict:
    """DRY RUN. Which visits would change species if the vote became confidence-weighted?

    Writes NOTHING -- it never calls clear_visits, insert_visit or assign_visit, and the CLI hands
    it a READ-ONLY connection so the guarantee is enforced by SQLite and not just by review. It
    groups detections with the same _group_detections the rebuild uses, runs BOTH votes over each
    group, and returns the disagreements.

    This exists because a species relabel is not reversible in practice: species gates the re-ID
    gallery, so a raccoon silently relabelled 'Virginia opossum' drops out of every template and
    every suggestion, and nothing in the UI would say why. Report first, decide, then rebuild.

    `flips_gate_only` is the count under the NARROWER reading -- keep the confidence gate but count
    the survivors rather than summing their confidences. On the live DB that reading moves 108
    visits and the shipped one moves 158; the gap is 50 near-ties the weights tipped, and it is
    reported rather than hidden so the choice is made with both numbers in view.

    Returns {visits, labelled, flips: [...], flips_gate_only, near_ties, into: {species: n},
             out_of: {species: n}, min_confidence}. Each flip: {source, started_at, ended_at,
             detections, from, to, margin_from, margin_to, n_above, n_below}."""
    if species_min_confidence is None:
        species_min_confidence = getattr(config.CONFIG, "species_vote_min_confidence", 0.8)
    rows = conn.execute(_DETECTION_COLUMNS).fetchall()
    flips, n_visits, n_labelled, n_gate_only, n_near_ties = [], 0, 0, 0, 0
    into: Counter = Counter()
    out_of: Counter = Counter()
    for source, members in _group_detections(rows, gap_minutes):
        n_visits += 1
        old, m_old = _species_vote(members, weighted=False, min_confidence=0.0)
        new, m_new = _species_vote(members, weighted=True,
                                   min_confidence=species_min_confidence)
        gate_only, _ = _species_vote(members, weighted=True,
                                     min_confidence=species_min_confidence,
                                     weight_by_confidence=False)
        if old is not None:
            n_labelled += 1
        if old != gate_only:
            n_gate_only += 1
        if old == new:
            continue
        if m_new is not None and m_new < 0.10:
            n_near_ties += 1
        confs = [(1.0 if m[1]["species_confidence"] is None else float(m[1]["species_confidence"]))
                 for m in members if m[1]["species"]]
        flips.append({
            "source": source,
            "started_at": members[0][1]["timestamp"], "ended_at": members[-1][1]["timestamp"],
            "detections": len(members), "from": old, "to": new,
            "margin_from": m_old, "margin_to": m_new,
            "n_above": sum(1 for c in confs if c >= species_min_confidence),
            "n_below": sum(1 for c in confs if c < species_min_confidence),
        })
        into[new] += 1
        out_of[old] += 1
    return {"visits": n_visits, "labelled": n_labelled, "flips": flips,
            "flips_gate_only": n_gate_only, "near_ties": n_near_ties,
            "into": dict(into), "out_of": dict(out_of),
            "min_confidence": float(species_min_confidence)}


def print_species_vote_report(report: dict, *, limit: int = 25) -> None:
    """Human-readable form of species_vote_report -- the thing to read before enabling the vote."""
    flips = report["flips"]
    print(f"\nSpecies vote, DRY RUN -- nothing was written. Comparing the shipped unweighted "
          f"crop-count\nvote against a confidence-weighted one (bar {report['min_confidence']:g}).")
    pct = (100.0 * len(flips) / report["visits"]) if report["visits"] else 0.0
    print(f"  {report['visits']} visit(s), {report['labelled']} with a species; "
          f"{len(flips)} would change label ({pct:.1f}%).")
    print(f"  {report['near_ties']} of those land with a margin under 0.10 -- coin flips, not "
          f"corrections.")
    print(f"  ({report['flips_gate_only']} move under the gate ALONE, i.e. counting the crops that "
          f"clear the bar\n   instead of summing their confidence. The gap is near-ties the "
          f"weights tipped.)")
    if not flips:
        print("  No visit changes species. Enabling the weighted vote is a no-op on this data.")
        return
    print("\n  Into:   " + ", ".join(f"{k or '(none)'} +{v}" for k, v in
                                     sorted(report["into"].items(), key=lambda kv: -kv[1])))
    print("  Out of: " + ", ".join(f"{k or '(none)'} -{v}" for k, v in
                                   sorted(report["out_of"].items(), key=lambda kv: -kv[1])))
    print(f"\n  {'started':26} {'src':16} {'crops':>6} {'from -> to':38} {'margin':>14}")
    for f in flips[:limit]:
        arrow = f"{f['from'] or '(none)'} -> {f['to'] or '(none)'}"
        m0 = "-" if f["margin_from"] is None else f"{f['margin_from']:.2f}"
        m1 = "-" if f["margin_to"] is None else f"{f['margin_to']:.2f}"
        print(f"  {f['started_at'][:26]:26} {f['source'][:16]:16} {f['detections']:>6} "
              f"{arrow[:38]:38} {m0:>6} ->{m1:>6}")
    if len(flips) > limit:
        print(f"  ... and {len(flips) - limit} more.")


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
    p.add_argument("--species-vote-report", action="store_true",
                   help="DRY RUN: report which visits a confidence-weighted species vote would "
                        "relabel, then exit. Opens the DB read-only and rebuilds NOTHING.")
    p.add_argument("--species-min-confidence", type=float, default=None,
                   help="Confidence bar for the weighted vote (default: "
                        "config.species_vote_min_confidence).")
    p.add_argument("--weighted-species", action="store_true",
                   help="Rebuild with the confidence-weighted species vote even if config ships it "
                        "disabled. CHANGES STORED LABELS -- run --species-vote-report first.")
    args = p.parse_args()

    gap = args.gap if args.gap is not None else config.CONFIG.visit_gap_minutes

    if args.species_vote_report:
        # Read-only on purpose: a dry run that cannot write is a dry run you can trust on the live DB.
        ro = db.connect_readonly(config.CONFIG.db_path)
        if ro is None:
            print(f"No database at {config.CONFIG.db_path} yet.")
            return 1
        try:
            print_species_vote_report(
                species_vote_report(ro, gap,
                                    species_min_confidence=args.species_min_confidence))
        finally:
            ro.close()
        return 0

    conn = db.connect(config.CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    build_visits(conn, gap,
                 weighted_species=True if args.weighted_species else None,
                 species_min_confidence=args.species_min_confidence)
    if args.stats:
        print_stats(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
