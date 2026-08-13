"""
COUNTING THE ANIMALS IN ONE FRAME -- honestly, and without inventing any.

Co-presence is the richest identity evidence this project has: when two animals share a frame,
depth is controlled for free, so their relative size means something where absolute size is noise.
It is also the only way to say "kit" -- a kit is an animal smaller than the adult standing next to
it. So the same-frame animal count is worth getting right.

This module exists because that count is wrong in BOTH directions, and only one of the two errors
can be fixed in code.

--------------------------------------------------------------------------------------------
WHAT DOES NOT WORK: recovering the animals the detector never proposes (tiling, sliced
inference, two-stage zoom). Measured here 2026-08-06, and NOT shipped. Do not re-implement it.
--------------------------------------------------------------------------------------------
MegaDetector v6's yolov10-c head is `end2end` (one-to-one assignment, no NMS -- Ultralytics'
non_max_suppression short-circuits for it and the `iou` argument is inert). On a frame holding
several animals it concedes the frame to the most salient one: on a 4K reference frame with FOUR
raccoons at one bowl it returns ONE box, 1978 px wide, at 0.899. The others ARE proposed at a
0.004 floor, well localised, at 0.008-0.048.

That last fact makes a proposal-anchored zoom look obvious -- crop each sub-gate proposal into
its own window and re-score it. It was built, with a fill guard, an interior-edge guard, an
anchor-overlap guard, three context scales and cross-window NMS, and measured over the 60
audited clip frames:

    66 base boxes  ->  29 boxes ADDED   (+44%, which looks like a win in any metric)
    of those 29, opened and counted by eye:  5 animals, 24 furniture.

The 24 were garden lanterns, the covered barbecue, dark panels of the retaining wall, and shrubs
-- the SAME objects on frame after frame of the same clip. Wiring that in would not have added
animals; it would have parked a phantom companion beside every real raccoon all night, which is
worse than the merging bug it was meant to fix. Confidence gives no way to separate them: the
top-scoring additions were 0.94 (empty ground), 0.88 (empty ground), 0.83 and 0.82 (bare wall).

The seductive number to watch for, if anyone tries this again: a window cropped tightly around a
proposal is returned as ONE box covering the whole window at 0.93-0.98, whatever is in it. Its
IoU against the anchor then reads 0.42-0.71, which looks like a recovered animal. It is not; the
tell is that the box FILLS the window. A window at 2.2x context containing two large, sharp,
plainly visible raccoons returns ZERO boxes at the shipped gate. The score is not recoverable by
re-framing, so no cropping scheme reaches these animals.

Therefore this module reports co-presence as a LOWER BOUND (`FrameCount.lower_bound` is always
True). The deficit is real and measured elsewhere: on glass-door night frames that hold 2-4
raccoons, detector recall is about 0.39.

--------------------------------------------------------------------------------------------
WHAT DOES WORK, and is what this module ships: not counting the same animal twice.
--------------------------------------------------------------------------------------------
Because the head runs with no NMS at all, nothing downstream removes duplicate boxes. Measured
over 33,403 glass-door frames since 2026-07-21 (4,643 same-frame box pairs), the geometry is
sharply BIMODAL and therefore easy to cut safely:

    63% of pairs are disjoint (IoU 0.000)  -- genuinely two objects
    34% of pairs are near-coincident (containment >= 0.9, IoU mostly >= 0.55) -- one object twice
    ~3% lie anywhere in between

Opened and confirmed by eye on raccoon frames: a single raccoon on the wall wearing two boxes at
IoU 0.588, stored as two "raccoon" rows; another at IoU 0.702; a single bird wearing two boxes at
IoU 0.925 and stored as one "American crow" plus one "house finch".

Two collapse rules, both pure ratios so they survive the camera moving:

  DUPLICATE   two boxes at IoU >= `iou_max` are one animal. 0.55 sits in the empty middle of the
              bimodal gap; moving it between 0.45 and 0.70 changes the share of collapsed pairs
              by three percentage points, so it is not a knife edge.
  UMBRELLA    a box that is well explained by the UNION of two or more boxes it contains is the
              detector bracketing a group, not a third animal. Requiring TWO OR MORE children is
              deliberate: a kit standing inside its mother's box is exactly one child, and must
              survive -- that pair is the signal the whole project wants.

A bare "drop the contained box" rule was rejected for that reason. It would buy only ~2% more
pairs (containment >= 0.9 is 34% of pairs, IoU >= 0.55 is 32%) and it would spend them on the
mother/kit case.

--------------------------------------------------------------------------------------------
FURNITURE (batch only): the other half of the inflation.
--------------------------------------------------------------------------------------------
Dedupe does not touch the second inflation mechanism, because a shrub is genuinely disjoint from
the raccoon beside it. Frames stored as "four animals, three of them raccoons" turned out by eye
to be one raccoon plus a shrub, a tipped bucket and a wall.

staticfilter.py already suppresses furniture on batch imports. Its primary test keys on box
persistence -- "nothing holds one box for an hour" -- which is false for a fixed-framing camera
where a raccoon feeds at a fixed dish, so running THAT test on the live rig would delete
thousands of real raccoon rows. (Since 2026-08-12 it has a second, pixel-based test as well --
see its "rigidity" -- which is the same argument this module makes; the two were arrived at
independently and agree. The difference is that staticfilter applies rigidity on its own,
"however briefly it fired", where this module insists on BOTH conditions, because it runs live
against a raccoon that may sit still for one burst.) So this module adds the criterion an
animal cannot satisfy: a static box's PIXELS
do not change either.

    motion_ratio = mean |frame_t - frame_t-1| INSIDE the box
                 / mean |frame_t - frame_t-1| over the whole frame

Normalised against the frame's own change, so it is free of exposure, gain, IR state and scene,
and needs no calibration when the camera moves. Measured over 70 random glass-door clips, 36
box-clusters, every one opened and judged by eye:

    furniture (shrub, bucket, lantern, bare wall, whole-frame box)   motion_ratio 0.98 - 2.00
    animals (raccoon, crow, cat, opossum, squirrel)                  motion_ratio 2.56 - 25.6

The shipped cut of 1.35 sits inside that gap with margin on both sides, and BOTH conditions are
required (persistent box AND dead pixels), so the still-crow at 2.56 and the dish-feeding raccoon
are never at risk. The asymmetry is deliberate and matches staticfilter's: leaving furniture in
overstates co-presence a little and is recoverable; removing an animal is not. A flagged box is
only ever dropped from a COUNT here -- nothing in this module writes to the database.

--------------------------------------------------------------------------------------------
COST, and where each half can run
--------------------------------------------------------------------------------------------
  distinct()/count_frame()   pure geometry, no model, no I/O. ~20 us for a 4-box frame; under
                             0.1% of the live loop's per-frame budget. SAFE TO RUN LIVE.
  motion_ratio()             one grayscale abs-diff per frame pair, ~1.5 ms at 1280x720.
                             Cheap, but it needs the PREVIOUS frame and a window of frames to
                             judge persistence, so it is a BATCH tool over stored clips.
  scan_clip()                dominated by the detector, ~30 ms per sampled frame on the RTX 5050
                             (81 ms under contention with the live rig). At the default stride
                             this is roughly 1 s per 10 s clip. BATCH ONLY -- do not put it in
                             the capture path.

CLI (read-only, dry run, never writes to the database or to disk):

    python tiledetect.py --clips clips/glass_door_cam/2026-08-06        # scan a night of clips
    python tiledetect.py --clips ... --limit 40 --explain               # per-cluster verdicts
    python tiledetect.py --frame path/to/frame.jpg                      # one still
    python tiledetect.py --audit-db --since 2026-07-21                  # read-only DB: how much
                                                                        # stored co-presence is
                                                                        # one animal boxed twice
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# --- Thresholds --------------------------------------------------------------------------------
# Every one of these is a RATIO. Nothing here is a pixel constant, a coordinate or a zone, because
# the cameras in this yard get repositioned on purpose and a hand-measured constant fails silently
# when they do. Defaults are the measured values from the module docstring; config.py carries the
# same names so they can be tuned per-yard in config_local.py without editing source.
DEFAULT_IOU_MAX = 0.55          # two boxes this alike are one animal (bimodal gap: 0.45-0.70)
DEFAULT_UNION_MIN = 0.75        # an umbrella box this well explained by its children is not an animal
DEFAULT_CHILD_CONTAIN = 0.80    # ... a "child" is a box at least this fraction inside the umbrella
DEFAULT_MIN_CHILDREN = 2        # ... and it takes TWO of them; one child is the kit/mother case
DEFAULT_MOTION_RATIO = 1.35     # below this, the pixels in the box are not moving (furniture)
DEFAULT_STATIC_IOU = 0.60       # box-cluster tracking across frames
DEFAULT_STATIC_MIN_FRAMES = 8   # ... seen in at least this many sampled frames
# Share of the window the cluster must occupy. OFF by default, and that is a measured choice, not
# laziness: furniture on this camera false-fires SPORADICALLY, not continuously. Over one night's
# 60 clips (1,139 sampled frames) the shrub-and-bucket cluster appeared 29 times -- 2.5% of the
# window -- so any share test tuned for a single 10 s clip silently switches the furniture arm off
# at the scale where it actually works. Persistence is carried by `min_frames`; what protects
# animals is the motion ratio, not the share.
DEFAULT_STATIC_MIN_SHARE = 0.0
DEFAULT_MAX_AREA_FRAC = 0.80    # a box covering this much of the frame is the whole-frame artifact
DEFAULT_STRIDE = 6              # sample every Nth video frame
DEFAULT_MAX_FRAMES = 40


# --- Geometry (pure, no dependencies beyond the stdlib) ----------------------------------------
Box = tuple[float, float, float, float]


def area(b: Box) -> float:
    """Box area; 0.0 for a degenerate or inverted box."""
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    """Area of overlap between two boxes (0.0 if they do not touch)."""
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def iou(a: Box, b: Box) -> float:
    """Intersection over union. Scale-free by construction, which is why every threshold in this
    module is expressed against it rather than against pixels."""
    inter = intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0.0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Fraction of `inner` that lies inside `outer`. Asymmetric on purpose: the umbrella rule asks
    "is this small box inside that big one", which IoU cannot express (a tiny box perfectly inside
    a huge one has a near-zero IoU)."""
    a_in = area(inner)
    return intersection(inner, outer) / a_in if a_in > 0.0 else 0.0


def union_box(boxes: Sequence[Box]) -> Box:
    """Smallest axis-aligned box containing all of `boxes`."""
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


# --- The count ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameCount:
    """How many distinct animals one frame shows.

    `lower_bound` is always True and is not a placeholder. The detector concedes multi-animal
    frames: on glass-door night frames holding 2-4 raccoons its recall is about 0.39, and the
    animals it drops are not recoverable by any cropping scheme (see the module docstring). Any
    surface that reports co-presence should say "at least N", never "N".
    """
    n_distinct: int
    n_raw: int
    kept: tuple[int, ...]              # indices into the input, in input order
    dropped_duplicate: tuple[int, ...]
    dropped_umbrella: tuple[int, ...]
    lower_bound: bool = True

    @property
    def n_collapsed(self) -> int:
        """How many input boxes this frame's count shed. The inflation, per frame."""
        return len(self.dropped_duplicate) + len(self.dropped_umbrella)


def _as_boxes(detections: Iterable) -> tuple[list[Box], list[float]]:
    """Accept either detector.Detection objects or plain (x1, y1, x2, y2) tuples, optionally
    paired with a score. Keeping this module happy with bare tuples is what lets the tests --
    and the boundary cases that matter -- be written without loading a 100 MB model."""
    boxes: list[Box] = []
    scores: list[float] = []
    for d in detections:
        bbox = getattr(d, "bbox", None)
        if bbox is not None:
            boxes.append(tuple(float(v) for v in bbox))          # type: ignore[arg-type]
            scores.append(float(getattr(d, "confidence", 0.0)))
            continue
        if isinstance(d, (tuple, list)) and len(d) == 2 and isinstance(d[0], (tuple, list)):
            boxes.append(tuple(float(v) for v in d[0]))          # type: ignore[arg-type]
            scores.append(float(d[1]))
            continue
        boxes.append(tuple(float(v) for v in d))                 # type: ignore[arg-type]
        scores.append(0.0)
    return boxes, scores


def distinct(detections: Iterable, *,
             iou_max: float = DEFAULT_IOU_MAX,
             union_min: float = DEFAULT_UNION_MIN,
             child_contain: float = DEFAULT_CHILD_CONTAIN,
             min_children: int = DEFAULT_MIN_CHILDREN) -> FrameCount:
    """Collapse one frame's boxes to one box per animal.

    Pure, deterministic and order-independent: boxes are considered strongest-first, ties broken
    on coordinates, so shuffling the input cannot change the answer. Nothing stochastic, nothing
    learned, no I/O, no model.

    Two rules, both argued in the module docstring:
      DUPLICATE  IoU >= iou_max against an already-kept box.
      UMBRELLA   >= min_children kept boxes each sit >= child_contain inside this box, AND their
                 union box matches it at IoU >= union_min. Requiring two children is what keeps a
                 kit inside its mother's box.

    This can only ever REMOVE boxes. It never proposes one, which is the whole point: an invented
    box manufactures co-presence that never happened, and that is worse than the merge it fixes.
    """
    boxes, scores = _as_boxes(detections)
    n = len(boxes)
    if n == 0:
        return FrameCount(0, 0, (), (), ())

    order = sorted(range(n), key=lambda i: (-scores[i], boxes[i]))

    kept: list[int] = []
    dup: list[int] = []
    umb: list[int] = []
    for i in order:
        b = boxes[i]
        if area(b) <= 0.0:
            dup.append(i)                       # a degenerate box is not an animal
            continue
        if any(iou(b, boxes[k]) >= iou_max for k in kept):
            dup.append(i)
            continue
        children = [k for k in kept if containment(boxes[k], b) >= child_contain]
        if len(children) >= min_children:
            # Is this box essentially just the bracket around those children? If the box is much
            # bigger than their union it is a big animal that happens to overlap small ones, and
            # it stays.
            if iou(b, union_box([boxes[k] for k in children])) >= union_min:
                umb.append(i)
                continue
        kept.append(i)

    # An umbrella can be accepted before its children are (it usually scores higher), so sweep
    # once more over the kept set now that everything is known. Iterating to a fixed point is
    # unnecessary -- removing an umbrella never creates another one, since children only lose a
    # potential parent.
    final: list[int] = []
    for i in kept:
        b = boxes[i]
        children = [k for k in kept if k != i and containment(boxes[k], b) >= child_contain]
        if len(children) >= min_children and iou(b, union_box([boxes[k] for k in children])) >= union_min:
            umb.append(i)
            continue
        final.append(i)

    return FrameCount(
        n_distinct=len(final),
        n_raw=n,
        kept=tuple(sorted(final)),
        dropped_duplicate=tuple(sorted(dup)),
        dropped_umbrella=tuple(sorted(umb)),
    )


def count_frame(detections: Iterable, **kw) -> int:
    """Convenience: the distinct-animal count alone. Remember it is a LOWER bound."""
    return distinct(detections, **kw).n_distinct


# --- Furniture (batch only: needs a window of frames) -------------------------------------------
def motion_ratio(prev_gray, gray, box: Box) -> float:
    """How much the pixels inside `box` changed, relative to how much the whole frame changed.

    ~1.0 means "exactly as much as the noise floor", i.e. nothing moved there. Dividing by the
    frame's own change is what makes this survive a camera move, a gain change, an IR cut-in and
    a different yard: there is no absolute threshold anywhere in it.

    `prev_gray` / `gray` are float arrays (numpy) of the same shape. Returns 1.0 if the box is
    empty or degenerate, which is the "cannot tell" answer and errs towards keeping the box.
    """
    import numpy as np

    diff = np.abs(gray - prev_gray)
    whole = float(diff.mean())
    if whole <= 1e-6:
        return 1.0
    h, w = diff.shape[:2]
    x1 = max(0, int(round(box[0]))); y1 = max(0, int(round(box[1])))
    x2 = min(w, int(round(box[2]))); y2 = min(h, int(round(box[3])))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return 1.0
    return float(diff[y1:y2, x1:x2].mean()) / whole


@dataclass
class Cluster:
    """One box seen repeatedly in the same place across a window of frames."""
    box: Box
    n_frames: int = 0
    ratios: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def median_ratio(self) -> float:
        if not self.ratios:
            return 1.0
        s = sorted(self.ratios)
        m = len(s) // 2
        return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])

    def is_furniture(self, n_sampled: int, *,
                     motion_max: float = DEFAULT_MOTION_RATIO,
                     min_frames: int = DEFAULT_STATIC_MIN_FRAMES,
                     min_share: float = DEFAULT_STATIC_MIN_SHARE) -> bool:
        """BOTH conditions, never either. Persistence alone is what makes staticfilter.py's SPAN
        test unsafe on the live rig (a raccoon feeding at a fixed dish holds a near-identical
        box); dead pixels alone would catch an animal that happened to freeze for a moment.
        staticfilter ships its pixel test unconditionally because a trail-cam batch is judged
        after the fact, where a wrong call is inspectable; live, it would not be."""
        if self.n_frames < min_frames:
            return False
        if n_sampled > 0 and self.n_frames / n_sampled < min_share:
            return False
        return self.median_ratio <= motion_max


def cluster_boxes(per_frame, *, static_iou: float = DEFAULT_STATIC_IOU) -> list[Cluster]:
    """Group boxes that keep reappearing in the same place.

    `per_frame` is a sequence of per-frame lists of ``(box, score, motion_ratio)``. Pure, and
    deterministic: frames are walked in order and each box joins the FIRST cluster it matches,
    so the same input always yields the same clusters.
    """
    clusters: list[Cluster] = []
    for boxes in per_frame:
        for box, score, ratio in boxes:
            hit = next((c for c in clusters if iou(c.box, box) >= static_iou), None)
            if hit is None:
                clusters.append(Cluster(box=tuple(box), n_frames=1, ratios=[ratio], scores=[score]))
            else:
                hit.n_frames += 1
                hit.ratios.append(ratio)
                hit.scores.append(score)
                # Drift the cluster box slowly so a slightly-wandering static object stays one
                # cluster, without letting a real animal drag it across the frame.
                hit.box = tuple(0.7 * o + 0.3 * b for o, b in zip(hit.box, box))
    return clusters


@dataclass
class ClipScan:
    """The result of scanning one clip. Nothing here is written anywhere."""
    path: Path
    n_sampled: int
    counts_raw: list[int]
    counts_distinct: list[int]
    counts_clean: list[int]          # distinct, minus boxes flagged as furniture
    clusters: list[Cluster]
    furniture: list[Cluster]
    seconds: float = 0.0
    # Per-sampled-frame [(box, score, motion_ratio), ...] for the boxes that survived distinct().
    # Carried so scan_clips() can re-cluster over a whole night, which is the window where the
    # furniture arm actually separates (see scan_clips).
    per_frame: list = field(default_factory=list)

    @property
    def max_raw(self) -> int:
        return max(self.counts_raw, default=0)

    @property
    def max_distinct(self) -> int:
        return max(self.counts_distinct, default=0)

    @property
    def max_clean(self) -> int:
        return max(self.counts_clean, default=0)


def scan_clip(path, detector, *, stride: int = DEFAULT_STRIDE,
              max_frames: int = DEFAULT_MAX_FRAMES,
              max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
              **kw) -> ClipScan | None:
    """Count the animals in a stored clip, three ways: raw boxes, distinct animals, and distinct
    animals with static furniture removed. BATCH ONLY -- it decodes video and runs the detector
    over a window of frames. Returns None if the clip will not decode.

    `detector` is a detector.Detector (or anything with a compatible .detect(frame_bgr)). This
    module never builds one itself and never changes the shipped gate: what the rig sees is what
    is counted.
    """
    import time

    import cv2
    import numpy as np

    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(f)
        i += 1
    cap.release()
    if len(frames) < 2:
        return None

    grays = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
             for f in frames]
    fh, fw = grays[0].shape[:2]
    frame_area = float(fh * fw)

    per_frame = []
    counts_raw, counts_distinct = [], []
    for k in range(1, len(frames)):
        dets = list(detector.detect(frames[k]))
        boxes, scores = _as_boxes(dets)
        # A box swallowing the whole frame is the detector's "vehicle 0.37 over everything"
        # artifact, not an animal. Dropping it is a ratio test, so it survives a camera move.
        live = [(b, s) for b, s in zip(boxes, scores) if area(b) < max_area_frac * frame_area]
        fc = distinct([(b, s) for b, s in live], **kw)
        counts_raw.append(len(live))
        counts_distinct.append(fc.n_distinct)
        per_frame.append([(live[j][0], live[j][1], motion_ratio(grays[k - 1], grays[k], live[j][0]))
                          for j in fc.kept])

    clusters = cluster_boxes(per_frame)
    n_sampled = len(frames) - 1
    furniture = [c for c in clusters if c.is_furniture(n_sampled)]
    counts_clean = []
    for boxes in per_frame:
        n = sum(1 for box, _, _ in boxes
                if not any(iou(box, c.box) >= DEFAULT_STATIC_IOU for c in furniture))
        counts_clean.append(n)

    return ClipScan(path=Path(path), n_sampled=n_sampled, counts_raw=counts_raw,
                    counts_distinct=counts_distinct, counts_clean=counts_clean,
                    clusters=clusters, furniture=furniture,
                    seconds=time.perf_counter() - t0, per_frame=per_frame)


@dataclass
class NightScan:
    """Many clips scanned together. This is the useful unit for the furniture arm."""
    scans: list[ClipScan]
    clusters: list[Cluster]
    furniture: list[Cluster]

    @property
    def n_sampled(self) -> int:
        return sum(s.n_sampled for s in self.scans)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.scans)

    def clean_counts(self, scan: ClipScan) -> list[int]:
        """One clip's per-frame counts with the NIGHT's furniture removed."""
        out = []
        for boxes in scan.per_frame:
            out.append(sum(1 for box, _, _ in boxes
                           if not any(iou(box, c.box) >= DEFAULT_STATIC_IOU for c in self.furniture)))
        return out


def scan_clips(paths, detector, **kw) -> NightScan:
    """Scan several clips and judge furniture over ALL of them at once.

    Measured, and the reason this function exists: within one 10-second clip the shrub that reads
    as a raccoon fires on 1 sampled frame in 20, so no per-clip persistence test can see it. Over
    one night's 60 clips (1,139 sampled frames) the same object forms a single 29-sighting cluster
    at motion_ratio 1.07, while every real-animal cluster in the same sweep sits between 7.6 and
    16.4. The separation is total at this scale and invisible at the other.

    BATCH ONLY. Detector-dominated, ~50 ms per sampled frame on the RTX 5050 alongside the live rig.
    """
    scans = [s for s in (scan_clip(p, detector, **kw) for p in paths) if s is not None]
    per_frame = [f for s in scans for f in s.per_frame]
    clusters = cluster_boxes(per_frame)
    n = len(per_frame)
    return NightScan(scans=scans, clusters=clusters,
                     furniture=[c for c in clusters if c.is_furniture(n)])


# --- CLI (read-only, dry run; there is no --apply because nothing here writes) -------------------
def _audit_db(db_path: str, source: str, since: str | None, limit: int) -> int:
    """Read-only: how much of the STORED co-presence record is one animal boxed twice?

    Opens the database in SQLite read-only URI mode. The live rig is capturing while this runs.
    """
    import sqlite3
    from collections import defaultdict

    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    sql = ("SELECT timestamp, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2 FROM detections "
           "WHERE source=? AND detection_class='animal'")
    args: list = [source]
    if since:
        sql += " AND substr(timestamp,1,10) >= ?"
        args.append(since)
    frames: dict = defaultdict(list)
    for ts, conf, x1, y1, x2, y2 in conn.execute(sql, args):
        frames[ts].append(((x1, y1, x2, y2), conf))
    conn.close()

    multi_raw = multi_distinct = 0
    collapsed = 0
    hist_raw: dict = defaultdict(int)
    hist_dist: dict = defaultdict(int)
    for boxes in frames.values():
        fc = distinct(boxes)
        hist_raw[fc.n_raw] += 1
        hist_dist[fc.n_distinct] += 1
        collapsed += fc.n_collapsed
        multi_raw += fc.n_raw >= 2
        multi_distinct += fc.n_distinct >= 2

    print(f"source={source}  frames={len(frames)}  detections={sum(hist_raw[k]*k for k in hist_raw)}")
    print(f"  frames the record calls multi-animal : {multi_raw:6d}  "
          f"({100*multi_raw/max(1,len(frames)):.2f}%)")
    print(f"  frames that survive de-duplication   : {multi_distinct:6d}  "
          f"({100*multi_distinct/max(1,len(frames)):.2f}%)")
    if multi_raw:
        print(f"  inflation                            : {100*(multi_raw-multi_distinct)/multi_raw:.1f}% "
              f"of 'multi-animal' frames are one animal boxed twice")
    print(f"  boxes collapsed                      : {collapsed}")
    print("  count histogram (raw -> distinct):")
    for k in sorted(set(hist_raw) | set(hist_dist)):
        if k:
            print(f"     {k}: {hist_raw.get(k,0):7d}  ->  {hist_dist.get(k,0):7d}")
    print("\n  NOTE: every count above is a LOWER bound. De-duplication fixes over-counting only;")
    print("  the detector's ~0.39 recall on multi-animal night frames is not fixable in code.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import config

    # The module constants above are the shipped defaults and stay independent of any machine, so
    # the tests are hermetic; the CLI is where a per-yard override in config_local.py takes effect.
    cfg_iou = getattr(config.CONFIG, "copresence_iou_max", DEFAULT_IOU_MAX)
    cfg_union = getattr(config.CONFIG, "copresence_union_min", DEFAULT_UNION_MIN)

    p = argparse.ArgumentParser(
        description="Count the animals in a frame, honestly: collapse duplicate and umbrella "
                    "boxes, flag static furniture, and report the result as a lower bound. "
                    "Read-only and dry-run: this tool never writes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--clips", help="Directory of .mp4 clips to scan (batch; runs the detector).")
    src.add_argument("--frame", help="A single still image to count.")
    src.add_argument("--audit-db", action="store_true",
                     help="Read-only DB audit of how inflated the stored co-presence record is.")
    p.add_argument("--db", default=None, help="SQLite database (default: config.CONFIG.db_path).")
    p.add_argument("--source", default="glass_door_cam", help="detections.source for --audit-db.")
    p.add_argument("--since", default=None, help="Only detections on/after this ISO date.")
    p.add_argument("--limit", type=int, default=25, help="Max clips to scan.")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Sample every Nth video frame.")
    p.add_argument("--iou-max", type=float, default=cfg_iou,
                   help="Two boxes at least this alike are one animal.")
    p.add_argument("--union-min", type=float, default=cfg_union,
                   help="An umbrella box this well explained by its children is not an animal.")
    p.add_argument("--explain", action="store_true", help="Print every box-cluster and its verdict.")
    a = p.parse_args(argv)

    db_path = a.db or str(config.CONFIG.db_path)

    if a.audit_db:
        return _audit_db(db_path, a.source, a.since, a.limit)

    import cv2
    from detector import Detector
    det = Detector(config.CONFIG.model_version, device="auto",
                   min_confidence=config.CONFIG.min_confidence, classes=("animal",))

    if a.frame:
        frame = cv2.imread(a.frame)
        if frame is None:
            print(f"could not read {a.frame}", file=sys.stderr)
            return 2
        dets = det.detect(frame)
        fc = distinct(dets, iou_max=a.iou_max, union_min=a.union_min)
        print(f"{a.frame}: raw {fc.n_raw} boxes -> at least {fc.n_distinct} animal(s)")
        if fc.dropped_duplicate:
            print(f"  duplicates collapsed: {list(fc.dropped_duplicate)}")
        if fc.dropped_umbrella:
            print(f"  umbrella boxes dropped: {list(fc.dropped_umbrella)}")
        return 0

    clips = sorted(Path(a.clips).rglob("*.mp4"))[:a.limit]
    if not clips:
        print(f"no clips under {a.clips}", file=sys.stderr)
        return 2
    # Furniture is judged over ALL the clips at once, never per clip: on this camera a false-firing
    # shrub appears in ~2.5% of a night's frames, which no single 10 s window can see.
    night = scan_clips(clips, det, stride=a.stride, iou_max=a.iou_max, union_min=a.union_min)
    tot_raw = tot_dist = tot_clean = 0
    for scan in night.scans:
        clean = night.clean_counts(scan)
        mx_clean = max(clean, default=0)
        tot_raw += scan.max_raw
        tot_dist += scan.max_distinct
        tot_clean += mx_clean
        flag = "  <-- collapsed" if mx_clean < scan.max_raw else ""
        print(f"{scan.path.name}  frames={scan.n_sampled:3d}  max raw {scan.max_raw} "
              f"-> distinct {scan.max_distinct} -> minus furniture {mx_clean}{flag}")
    if a.explain:
        print("\nbox-clusters over the whole sweep:")
        for c in sorted(night.clusters, key=lambda z: -z.n_frames):
            verdict = "FURNITURE" if c in night.furniture else "animal"
            print(f"   {[round(v) for v in c.box]}  n={c.n_frames:3d}  "
                  f"motion_ratio={c.median_ratio:6.2f}  -> {verdict}")
    n_frames, secs = night.n_sampled, night.seconds
    print(f"\n{len(night.scans)} clips, {n_frames} sampled frames in {secs:.1f}s "
          f"({1000*secs/max(1,n_frames):.0f} ms/frame -- BATCH tool, not the live loop)")
    print(f"summed per-clip maxima: raw {tot_raw} -> distinct {tot_dist} -> clean {tot_clean}")
    print(f"furniture clusters flagged over the sweep: {len(night.furniture)}")
    print("Counts are LOWER bounds: the detector concedes multi-animal frames (recall ~0.39).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
