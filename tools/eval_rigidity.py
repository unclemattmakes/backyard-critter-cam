"""Sweep the rigidity threshold in staticfilter.py against this yard's own history.

WHY THIS EXISTS. staticfilter's rigidity test deletes rows, and a false positive there deletes
an animal, which no manifest can bring back. So the threshold is not a guess -- it is the widest
gap between two measured populations, and this script is how that gap gets re-measured whenever
the camera, its IR illuminator or the crop pipeline changes.

THE TWO POPULATIONS.
  FURNITURE  the sidecar manifest (backyard.db.static-dropped-<source>.txt), MINUS any spot the
             rigidity test itself condemned. That subtraction is the whole reason this eval can
             be trusted: sweep_batch runs rigidity by default, so without it the manifest would
             fill with the test's own verdicts and the script would be grading its own homework.
             Spots are skipped on the `via=rigid` tag that append_manifest writes.
             What survives the subtraction is furniture on independent evidence -- but be precise
             about WHICH evidence. Most of it held one box for hours and was condemned by the
             SPAN test. A minority (10 spots / 669 rows as of 2026-08-12, spans of 4-36 min) came
             from a hand-lowered sweep after a human looked at the crops -- the 2026-08-11 battery
             cycle. That provenance is weaker, and two of the spots rigidity "recovers" are among
             them, so treat the recall figure as partly reflecting the judgement that motivated
             the feature. It is the ANIMAL floor, not recall, that the threshold rests on.
             The crop JPEGs survive deletion by design, which is the only reason any of this
             set is still measurable.
  ANIMALS    surviving clusters where one MAMMAL species takes >=85% of the detections. This is
             a proxy, not a hand-label: there are only a handful of species_verified rows in the
             DB, far too few to sweep against. It is a conservative proxy in the right direction
             -- furniture tends to draw a SPREAD of confident-sounding labels (one 74x153 paver
             edge collected twelve species, eight of them birds), so demanding one dominant mammal
             filters toward things that really were animals.

READING THE RESULT. The number that matters is the LOWEST animal, because the threshold has to
sit clear underneath it. Recall is the secondary number and it is expected to be poor: rigidity
only ever catches furniture that is genuinely rigid, and wind-blown vegetation is furniture that
moves. Do not raise the threshold to chase recall -- the span test and a human are the answer
for the rest.

  python tools/eval_rigidity.py                       # sweep the shipped default
  python tools/eval_rigidity.py --source trail_cam_sd --min-count 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                                    # noqa: E402
import db                                                        # noqa: E402
import staticfilter                                              # noqa: E402

# One dominant species from this set is the proxy for "really an animal". Birds are excluded on
# purpose: they are small, they perch on the furniture, and a static post reads as a different
# LBJ every time it fires -- so a bird-dominant cluster is not evidence of anything.
#
# BROWN RAT IS EXCLUDED, and that exclusion is load-bearing rather than cosmetic. On this rig
# "brown rat" is the label a dark upright object collects: 272 of them on 2026-08-11 were a bin
# (one object, split across two boxes by the IoU cut), and 306 in the 2026-08-05 cycle were a
# grill and a chimney starter. Left in, the THREE lowest "animals" this script found were all
# unswept furniture from cycles that predate the filter -- 0.134, 0.147, 0.154, all brown rat --
# which dragged the apparent animal floor from 0.225 down to 0.134 and condemned the shipped
# threshold for hitting "animals" that were a bin. A proxy that counts furniture as an animal
# does not measure a safety margin, it destroys one.
#
# Quantified exposure, since excluding a label is not the same as the risk going away: at the
# shipped 0.15 there are two surviving brown-rat clusters in the DB (0.134 n=52 on 07-31, 0.147
# n=104 on 07-21, 156 rows) that a sweep would delete. Both look like the bin. Neither is proven.
#
# The honest cost: this eval therefore says NOTHING about a real brown rat. If one ever visits
# and holds still, rigidity could flag it, and no number here would have warned you.
MAMMALS = {"raccoon", "Virginia opossum", "eastern gray squirrel", "eastern cottontail",
           "domestic cat", "Townsend's chipmunk"}
DOMINANCE = 0.85


class _Fake:
    """Manifest rows carry a crop_path and nothing else; rigidity() only needs that."""

    def __init__(self, paths):
        self.rows = [{"crop_path": p} for p in paths]

    def sample_rows(self, k=staticfilter.RIGID_SAMPLE):
        step = max(1, len(self.rows) // k)
        return self.rows[::step][:k]


def furniture_spots(db_path, source, min_count):
    """Furniture ground truth from the manifest, EXCLUDING the rigidity test's own verdicts.

    Returns (paths, span_minutes) per spot. The span comes back because it is what separates the
    two provenances -- a spot over min_span_minutes was condemned by the span rule, one under it
    by a human -- and because it is the only way to report what rigidity ADDS rather than what it
    merely agrees with.
    """
    path = staticfilter.manifest_path(db_path, source)
    if not path.exists():
        return []
    spots, cur, keep = [], None, True
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            # "# static spot (x1,y1)-(x2,y2) n=N span=Mmin via=rigid rigidity=0.123"
            keep = "via=rigid" not in line
            span = 0.0
            for tok in line.split():
                if tok.startswith("span="):
                    try:
                        span = float(tok[5:].rstrip("min"))
                    except ValueError:
                        span = 0.0
            cur = [[], span]
            if keep:
                spots.append(cur)
            continue
        parts = line.split("|")
        if len(parts) >= 4 and cur is not None and keep:
            cur[0].append(parts[3])
    return [(p, s) for p, s in spots if len(p) >= min_count]


def animal_clusters(conn, source, min_count, iou):
    """Surviving clusters dominated by one mammal -- clustered per DAY, because a cycle is one
    camera placement and 'the same spot' means nothing across a move."""
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(timestamp,1,10) FROM detections WHERE source=? ORDER BY 1",
        (source,))]
    out = []
    for d in days:
        rows = staticfilter.load_rows(conn, source, since=d, until=d + "T23:59:59")
        for c in staticfilter.cluster_detections(rows, iou):
            if c.count < min_count:
                continue
            tally = c.species_tally()
            top, n = next(iter(tally.items()))
            if top in MAMMALS and n / c.count >= DOMINANCE:
                out.append((c, d, top, n))
    return out


def main() -> int:
    cfg = config.CONFIG
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=db.SOURCE_TRAIL_CAM_SD)
    p.add_argument("--db", default=str(cfg.db_path))
    p.add_argument("--iou", type=float, default=staticfilter.DEFAULT_IOU)
    p.add_argument("--min-count", type=int, default=staticfilter.DEFAULT_MIN_COUNT)
    args = p.parse_args()

    conn = db.connect(Path(args.db))
    try:
        fur = []
        for paths, span in furniture_spots(Path(args.db), args.source, args.min_count):
            r = staticfilter.rigidity(_Fake(paths), config.ROOT)
            if r is not None:
                # span-missed spots are the only ones rigidity can ADD; the rest it merely agrees
                # with, and counting those as a win overstates the test.
                fur.append((r, len(paths), span < staticfilter.DEFAULT_MIN_SPAN_MINUTES))
        ani = []
        for c, d, top, n in animal_clusters(conn, args.source, args.min_count, args.iou):
            r = staticfilter.rigidity(c, config.ROOT)
            if r is not None:
                ani.append((r, c.count, d, f"{top}:{n}"))
    finally:
        conn.close()

    if not fur or not ani:
        print(f"not enough data: {len(fur)} furniture spot(s), {len(ani)} animal cluster(s). "
              "The manifest only fills once --apply has run at least once.")
        return 1

    fur.sort()
    ani.sort()
    print(f"source={args.source}  furniture spots={len(fur)}  animal clusters={len(ani)}\n")
    print("LOWEST-SCORING ANIMALS (the ceiling the threshold must stay under)")
    for r, n, d, sp in ani[:8]:
        print(f"  {r:.3f}  n={n:<4} {d}  {sp}")
    floor = ani[0][0]
    missed = sum(1 for f in fur if f[2])
    print(f"\nlowest animal      : {floor:.3f}")
    print(f"furniture range    : {fur[0][0]:.3f} .. {fur[-1][0]:.3f}")
    print(f"shipped threshold  : {staticfilter.DEFAULT_RIGID_MAXPAIR:.3f}")
    print(f"span-missed spots  : {missed} of {len(fur)}  (the only ones rigidity can ADD)\n")
    print(f"{'thresh':>7} {'headroom':>9} {'animals hit':>12} {'spots':>7} {'rows':>7} "
          f"{'NEW spots':>10} {'NEW rows':>9}")
    for t in (0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25):
        hit = sum(1 for r, *_ in ani if r < t)
        spots = [f for f in fur if f[0] < t]
        new = [f for f in spots if f[2]]
        # headroom: how much of the way from 0 to the lowest animal is left unused. Deliberately
        # NOT "% below the animal" -- 0.15 is 33% below 0.225, and conflating the two flatters
        # the threshold.
        headroom = (floor - t) / t * 100
        flag = "  <-- SHIPPED" if abs(t - staticfilter.DEFAULT_RIGID_MAXPAIR) < 1e-9 else ""
        bad = "  *** HITS AN ANIMAL" if hit else ""
        print(f"{t:7.2f} {headroom:8.0f}% {hit:12d} {len(spots):7d} {sum(s[1] for s in spots):7d} "
              f"{len(new):10d} {sum(s[1] for s in new):9d}{flag}{bad}")
    print("\nA threshold is only safe while 'animals hit' is 0 AND the headroom is wide enough "
          "that the next unseen animal cannot land under it.")
    print("'NEW' counts only span-missed furniture -- what rigidity ADDS. The plain 'spots'/'rows'"
          "\ncolumns include furniture the span test already deletes, and overstate the test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
