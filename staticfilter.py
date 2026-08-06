"""
STATIC FALSE-FIRE SUPPRESSION for batch imports -- self-calibrating, no fixed coordinates.

A detector pointed at a yard fires on things that are not animals. On the live rig that was a
dark opening in the retaining wall, fixed with config.ignore_zones: a hand-measured box, dropped
by IoU. That fix does not transfer to a trail cam. The TC02 gets repositioned between cycles on
purpose, and a hand-measured zone would go stale the first time it moved -- silently, which is
the worst way for a filter to fail (a zone that no longer covers the grill drops nothing and
says nothing).

So this module measures the zones itself, per batch, from the detections alone:

    cluster every box by IoU, then call a cluster STATIC when it repeats in the same place
    for longer than any animal would hold still.

That is the whole idea, and it is self-calibrating: move the camera and the next import derives
new clusters with no config to update. What it keys on is not "where" but "for how long, without
moving" -- 2026-08-04's covered Weber grill sat in an identical 235x486 box for five and a half
hours across 178 detections, while the busiest real raccoon session in the same cycle lasted 22
minutes and never held one box (a moving animal's box changes size and shape with its pose, so
consecutive frames of it do not even reach the IoU threshold, let alone the span).

WHY IoU RATHER THAN A GRID. Quantizing box corners to a grid splits an object that straddles a
cell boundary into two half-sized clusters, either of which can fall under min_count. IoU
compares whole boxes and has no boundary to straddle.

WHAT IT DELIBERATELY WILL NOT CATCH. An object that only fires a handful of times, or one whose
appearances are spread thin, stays in -- the thresholds are set so a real animal can never be
mistaken for furniture, and the cost of that choice is some junk surviving. False negatives are
recoverable (run the sweep again with a lower --min-count once you have looked); a false
positive deletes an animal, which is not.

NOTHING IS DELETED SILENTLY. Every applied sweep appends the dropped rows to a sidecar manifest
next to the DB (backyard.db.static-dropped-<source>.txt), the same convention as the import
ledger, and the crop JPEGs are LEFT ON DISK -- so a bad sweep is inspectable after the fact and
the images survive it. Only DB rows go, and the embeddings cascade with them.

  python staticfilter.py --source trail_cam_sd                     # dry run: report, change nothing
  python staticfilter.py --source trail_cam_sd --apply             # ... actually delete the rows
  python staticfilter.py --since 2026-08-03 --until 2026-08-05     # limit to one cycle's window
  python staticfilter.py --min-count 40 --min-span-minutes 120     # stricter (fewer clusters)
  python staticfilter.py --explain                                 # show every cluster + verdict
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

import config
import db

# --- Thresholds -----------------------------------------------------------------------------
# Measured against the 2026-08-04 trail-cam cycle, where the three real static objects (a covered
# grill, its chimney starter, and a dark gap that read as eyeshine) and the real animals separate
# by more than an order of magnitude on BOTH axes -- so these sit in a wide empty gap, not on a
# knife edge. Raising either one only ever keeps more junk; it can never eat an animal.
DEFAULT_IOU = 0.75            # how identical two boxes must be to count as "the same spot"
DEFAULT_MIN_COUNT = 15        # detections in one spot before it can be furniture
DEFAULT_MIN_SPAN_MINUTES = 60.0   # ... spread over at least this long

MANIFEST_SUFFIX = "static-dropped"


def manifest_path(db_path: Path | str, source: str) -> Path:
    """Sidecar file recording what a sweep deleted -- mirrors the import ledger's naming so the
    two live side by side (backyard.db.imported-X.txt / backyard.db.static-dropped-X.txt)."""
    p = Path(db_path)
    return p.with_name(f"{p.name}.{MANIFEST_SUFFIX}-{source}.txt")


def _iou(a, b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes. 0.0 when they don't overlap."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    union = ar + br - inter
    return inter / union if union > 0 else 0.0


class Cluster:
    """One spot in the frame that at least one detection landed on, plus every detection that
    landed on it. `anchor` is the first box seen -- fixed, never re-averaged, so membership can't
    drift across the frame one near-miss at a time (a running centroid lets a slow-moving animal
    walk a cluster along with it, which is exactly the failure this filter must not have)."""

    __slots__ = ("anchor", "rows")

    def __init__(self, anchor, row):
        self.anchor = anchor
        self.rows = [row]

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def span(self) -> timedelta:
        ts = [r["_dt"] for r in self.rows]
        return max(ts) - min(ts)

    @property
    def span_minutes(self) -> float:
        return self.span.total_seconds() / 60.0

    def species_tally(self) -> dict:
        out: dict = {}
        for r in self.rows:
            k = r["species"] or "(unlabeled)"
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def is_static(self, min_count: int, min_span_minutes: float) -> bool:
        """Furniture, not an animal: it appeared often enough AND held one box long enough.
        Both conditions are required -- a burst of six photos of a sitting animal clears the
        count in two seconds and is rejected on span, which is the point of having both."""
        return self.count >= min_count and self.span_minutes >= min_span_minutes


def cluster_detections(rows, iou_thresh: float = DEFAULT_IOU) -> list[Cluster]:
    """Greedily group rows whose boxes sit on top of each other (IoU >= iou_thresh).

    Rows must already carry a parsed '_dt'. O(n*k) in detections x clusters, which at a few
    thousand detections per card is milliseconds; if a cycle ever gets big enough to notice,
    bucket the clusters by box centre before comparing.
    """
    clusters: list[Cluster] = []
    for r in rows:
        box = (r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"])
        for c in clusters:
            if _iou(box, c.anchor) >= iou_thresh:
                c.rows.append(r)
                break
        else:
            clusters.append(Cluster(box, r))
    return clusters


def load_rows(conn, source: str, *, since: str | None = None, until: str | None = None,
              min_id: int | None = None) -> list[dict]:
    """Detections for `source`, optionally limited to a time window or to ids above `min_id`
    (which is how an import limits the sweep to the batch it just wrote). Rows with an
    unparseable timestamp are dropped: without a time we cannot judge span, and a cluster that
    cannot be judged must never be deleted."""
    where = ["source = ?"]
    params: list = [source]
    if since:
        where.append("timestamp >= ?")
        params.append(since)
    if until:
        where.append("timestamp < ?")
        params.append(until)
    if min_id is not None:
        where.append("id > ?")
        params.append(int(min_id))
    prior = conn.row_factory
    try:
        conn.row_factory = sqlite3.Row
        raw = conn.execute(
            "SELECT id, timestamp, species, crop_path, confidence, "
            "       bbox_x1, bbox_y1, bbox_x2, bbox_y2 "
            f"FROM detections WHERE {' AND '.join(where)} ORDER BY timestamp, id",
            params,
        ).fetchall()
    finally:
        conn.row_factory = prior
    out = []
    for r in raw:
        dt = db.parse_local(r["timestamp"])
        if dt is None:
            continue
        d = dict(r)
        d["_dt"] = dt
        out.append(d)
    return out


def find_static(rows, *, iou: float = DEFAULT_IOU, min_count: int = DEFAULT_MIN_COUNT,
                min_span_minutes: float = DEFAULT_MIN_SPAN_MINUTES
                ) -> tuple[list[Cluster], list[Cluster]]:
    """(static_clusters, all_clusters) for a batch of detection rows."""
    clusters = cluster_detections(rows, iou)
    static = [c for c in clusters if c.is_static(min_count, min_span_minutes)]
    return static, clusters


def describe(static: list[Cluster], total_rows: int, *, prefix: str = "  ") -> list[str]:
    """Human-readable lines for a report -- shared by the CLI and the importer so both say the
    same thing in the same shape."""
    lines = []
    dropped = sum(c.count for c in static)
    if not static:
        lines.append(f"{prefix}no static clusters found in {total_rows} detection(s).")
        return lines
    pct = (100.0 * dropped / total_rows) if total_rows else 0.0
    lines.append(f"{prefix}{len(static)} static spot(s) -> {dropped} of {total_rows} "
                 f"detection(s) ({pct:.0f}%):")
    for c in sorted(static, key=lambda c: -c.count):
        x1, y1, x2, y2 = (int(v) for v in c.anchor)
        tally = ", ".join(f"{k}:{v}" for k, v in c.species_tally().items())
        lines.append(f"{prefix}  box ({x1},{y1})-({x2},{y2}) {x2-x1}x{y2-y1}  "
                     f"n={c.count}  span={c.span_minutes:.0f}min  [{tally}]")
        lines.append(f"{prefix}    e.g. {c.rows[0]['crop_path']}")
    return lines


def append_manifest(db_path: Path | str, source: str, static: list[Cluster]) -> None:
    """Record every deleted row before it goes. Best-effort, like the import ledger -- but note
    the ORDER in apply(): the manifest is written FIRST, so a crash mid-delete leaves a record
    of more than was deleted rather than less."""
    try:
        with open(manifest_path(db_path, source), "a", encoding="utf-8") as f:
            for c in static:
                x1, y1, x2, y2 = (int(v) for v in c.anchor)
                f.write(f"# static spot ({x1},{y1})-({x2},{y2}) n={c.count} "
                        f"span={c.span_minutes:.0f}min\n")
                for r in c.rows:
                    f.write(f"{r['id']}|{r['timestamp']}|{r['species'] or ''}|{r['crop_path']}\n")
    except Exception as e:
        print(f"  [warn] could not write the static-drop manifest: {e}")


def apply(conn, db_path: Path | str, source: str, static: list[Cluster]) -> int:
    """Delete the clustered rows. detection_embeddings cascade (FK ON DELETE CASCADE, and
    db.connect sets PRAGMA foreign_keys=ON). Visits are NOT touched here: visits.build_visits
    rebuilds from scratch, so a refresh after this call is what makes the phantom visits go --
    deleting them here as well would just be undone by the next rebuild. The crop JPEGs stay on
    disk on purpose (see the module docstring). Returns rows deleted."""
    ids = [r["id"] for c in static for r in c.rows]
    if not ids:
        return 0
    append_manifest(db_path, source, static)
    for i in range(0, len(ids), 500):          # chunked: SQLite caps host parameters per statement
        chunk = ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        # visits.representative_detection_id points HERE with no ON DELETE CASCADE (unlike
        # detection_embeddings), so deleting a row that some visit picked as its best crop trips
        # the FK. The column is nullable and the visit is about to be rebuilt from scratch
        # anyway, so releasing the pointer first is both legal and lossless.
        conn.execute(f"UPDATE visits SET representative_detection_id = NULL "
                     f"WHERE representative_detection_id IN ({marks})", chunk)
        conn.execute(f"DELETE FROM detections WHERE id IN ({marks})", chunk)
    conn.commit()
    return len(ids)


def sweep_batch(conn, cfg, source: str, *, min_id: int, iou: float = DEFAULT_IOU,
                min_count: int = DEFAULT_MIN_COUNT,
                min_span_minutes: float = DEFAULT_MIN_SPAN_MINUTES) -> int:
    """The importer's entry point: judge ONLY the rows this run just wrote (id > min_id) and
    delete the static ones. Deliberately scoped to the batch -- a cycle is one camera placement,
    which is the unit over which "the same spot" means anything at all. Returns rows deleted.

    Never raises: a filter that cannot run must not fail an import that already succeeded. The
    unfiltered rows are still there and `python staticfilter.py` can sweep them afterwards.
    """
    try:
        rows = load_rows(conn, source, min_id=min_id)
        if not rows:
            return 0
        static, _ = find_static(rows, iou=iou, min_count=min_count,
                                min_span_minutes=min_span_minutes)
        if not static:
            print(f"  [static] none of this batch's {len(rows)} detection(s) held one spot for "
                  f"{min_span_minutes:.0f}+ min -- nothing dropped.")
            return 0
        print("\n[static] self-calibrating false-fire filter:")
        for line in describe(static, len(rows)):
            print(line)
        n = apply(conn, cfg.db_path, source, static)
        print(f"  [static] dropped {n} detection(s); crops left on disk, listed in "
              f"{manifest_path(cfg.db_path, source).name}")
        return n
    except Exception as e:
        print(f"  [static] filter skipped ({e}) -- rows kept; run `python staticfilter.py` to "
              "sweep them by hand.")
        return 0


def main() -> int:
    c = config.CONFIG
    p = argparse.ArgumentParser(
        description="Find and remove static false-fire detections (a grill, a post, a dark gap) "
                    "by spotting boxes that repeat in one place for longer than an animal would.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default=db.SOURCE_TRAIL_CAM_SD, help="detections.source to sweep.")
    p.add_argument("--db", default=str(c.db_path), help="SQLite database.")
    p.add_argument("--since", default=None, help="Only detections at/after this ISO date.")
    p.add_argument("--until", default=None, help="Only detections BEFORE this ISO date.")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU,
                   help="How identical two boxes must be to count as the same spot.")
    p.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                   help="Detections in one spot before it can be called furniture.")
    p.add_argument("--min-span-minutes", type=float, default=DEFAULT_MIN_SPAN_MINUTES,
                   help="How long that spot must keep firing before it counts as static.")
    p.add_argument("--explain", action="store_true",
                   help="Print EVERY cluster with its verdict, not just the static ones -- how "
                        "you check the thresholds sit in a gap and not on a knife edge.")
    p.add_argument("--apply", action="store_true",
                   help="Actually delete the rows (default is a dry run that changes nothing).")
    args = p.parse_args()

    conn = db.connect(Path(args.db))
    try:
        rows = load_rows(conn, args.source, since=args.since, until=args.until)
        if not rows:
            print(f"No detections for source='{args.source}' in that window.")
            return 0
        static, clusters = find_static(rows, iou=args.iou, min_count=args.min_count,
                                       min_span_minutes=args.min_span_minutes)
        print(f"{len(rows)} detection(s) for source='{args.source}' -> {len(clusters)} spot(s) "
              f"at IoU {args.iou:g}.")
        if args.explain:
            print(f"\n  {'n':>5} {'span(min)':>10}  {'verdict':<8} box")
            for cl in sorted(clusters, key=lambda c: -c.count):
                x1, y1, x2, y2 = (int(v) for v in cl.anchor)
                verdict = "STATIC" if cl.is_static(args.min_count, args.min_span_minutes) else "keep"
                print(f"  {cl.count:5d} {cl.span_minutes:10.1f}  {verdict:<8} "
                      f"({x1},{y1})-({x2},{y2})  [{', '.join(f'{k}:{v}' for k, v in cl.species_tally().items())}]")
        print()
        for line in describe(static, len(rows), prefix=""):
            print(line)
        if not static:
            return 0
        if not args.apply:
            print("\nDry run -- nothing deleted. Re-run with --apply to remove these rows.")
            return 0
        n = apply(conn, Path(args.db), args.source, static)
        print(f"\nDeleted {n} detection(s). Crops left on disk; manifest: "
              f"{manifest_path(args.db, args.source).name}")
        import visits
        visits.refresh(conn, c.visit_gap_minutes)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
