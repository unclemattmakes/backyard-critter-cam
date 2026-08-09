"""
REFERENCE-IMAGE VETO -- "that box is furniture, and here is the empty yard that proves it".

The detector fires on things that are not animals: a tipped watering can, a covered barbecue, a
dark gap in a retaining wall. config.ignore_zones fixes that with a hand-measured rectangle, and
staticfilter.py fixes it after the fact for a batch import -- but neither works on the live rig,
because Matt repositions the cameras on purpose (a stale zone fails SILENTLY) and because
position alone cannot separate furniture from a raccoon that feeds at a fixed dish every night
for 27 days. This module asks the question directly instead: *does this box look exactly like the
empty yard did, at a spot that keeps firing?*

Everything here is a port of the raced prototype behind docs/refimg-design-2026-08-07.md, at the
same 320x180 working resolution the thresholds were measured at. Its measured operating point on
the glass door, over 2026-08-06 00:00-03:00:

    animals suppressed     0 of 4,649   (including all 623 near-miss evaluations where a real
                                         raccoon and the watering can shared a frame)
    furniture suppressed   384 of 652   (58.9%; every suppressed crop was opened -- all can)

-------------------------------------------------------------------------------------------------
SHADOW MODE. Nothing in this module deletes, hides or alters a detection.
-------------------------------------------------------------------------------------------------
A decision is METADATA. `mark_suppressed()` writes the four additive columns from the design doc
(detections.suppressed_at / suppressed_by / suppress_ref_id / suppress_detail) and NOTHING reads
them: not the dashboard, not individuals.still_tracklets, not the co-presence badge, not stats.
The point is a week of flagged rows on one contact sheet (`python refimg.py --review`) before any
consumer honours the flag. An erased animal writes no row at all and is a silent permanent loss;
a surviving grill costs one bogus co-presence edge. That asymmetry decides every tie below, and
it is why the veto ABSTAINS on every ambiguity rather than guessing.

-------------------------------------------------------------------------------------------------
THE THREE PARTS, AND WHY EACH IS LOAD-BEARING
-------------------------------------------------------------------------------------------------
1. THE REFERENCE (policy E, `ReferenceManager`). A frame is eligible to become the reference only
   when THE DETECTOR ACTUALLY RAN ON IT and returned zero boxes, motion stayed quiet, and that
   held for `refimg_certify_hold_s`. The tempting shortcut -- "no detection row near this frame,
   so the frame must be empty" -- was raced and it stores a reference WITH THE RACCOON IN IT
   (design §3.1: a 1-minute-old "empty" reference that scores dssim 0.008 against the sleeping
   animal it contains). A resident animal cannot enter a certified reference because while it is
   there the detector is firing on it, which resets the clock.

   A certified reference is still capped by the detector's RECALL, and the race caught exactly
   that: a properly certified glass-door reference with an undetected raccoon walking the wall in
   it (design §4.3; the project has this blind spot on record -- 2026-07-20 dusk, MDV6 missed the
   raccoon in the dark bokeh above the wall, 0.89 when lit). So policy E adds the channel the
   detector does not have: every pixel that motion-blobbed in the last `refimg_no_update_s` is
   marked NOT COVERED, and the veto abstains there. A missed animal then costs an abstention
   instead of an erasure. The price is measured and accepted: 58.9% of furniture instead of 90.2%.

   REFERENCES ARE SWITCHED, NEVER BLENDED, and a decaying background model is FORBIDDEN here. A
   MOG2-style model absorbs a still object in ~history frames -- 500 frames at 21.6 fps is 23
   seconds, three orders of magnitude below the 823 s of eye-verified stationary residency in
   this corpus. Anything that learns continuously learns the sleeping animal.

2. THE VETO (`ShadowVeto`), a CONJUNCTION in which every gate is load-bearing, measured:
     * the pixel test ALONE erases raccoons -- with the gates off it fired on 131 of 372 confident
       animal boxes in one Stan-and-kits visit, at scores as low as 0.007x the threshold;
     * the recurrence test ALONE flags the FOOD BOWL, which is 27 days of real raccoons;
     * 422 of 4,649 animal boxes (9.1%) passed the full three-metric pixel test and were saved
       ONLY by recurrence. Neither gate may be dropped, in either order.

3. THE AUDIT (`--review`). Read-only contact sheets of everything the veto suppressed, grouped by
   recurrence cluster, with every score and threshold printed next to the crop. This is the loop
   that produced the design doc's verification, and it is the entire reason shadow mode exists.

-------------------------------------------------------------------------------------------------
WHAT THIS MODULE DELIBERATELY DOES NOT DO
-------------------------------------------------------------------------------------------------
* It does not run on the trail cam. `import_trailcam.py` runs the detector on the SD card's STILL
  photos and never on video, so a trail-cam clip frame carries NO detector evidence and cannot
  certify anything; and the stills save crops only, so there is no still full frame to compare
  against. Design §5 has the two prerequisites. Worse, the two cameras DISAGREE ABOUT WHICH
  METRIC WORKS -- in IR, `lum` and `sobel` separate cleanly and `dssim` does not, the opposite of
  the glass door, and the published IR dssim threshold kills 7 of 60 real raccoons. Copying
  thresholds between cameras is the specific mistake to avoid.
* It never suppresses on recurrence alone, on pixels alone, on a reference older than
  `refimg_max_age_s`, across a view epoch, or over pixels the reference does not know.
* It holds no opinion about species, individuals, or anything downstream.

    python refimg.py --review               # last 7 days of suppressions -> contact sheet PNGs
    python refimg.py --review --days 1      # just last night
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

import config

# --- Working resolution ---------------------------------------------------------------------
# 320x180 is not a performance choice, it is the resolution EVERY threshold below was measured
# at. Changing it invalidates them all. (Cost at this size, measured: ~7.6 ms to build the
# frame-vs-reference comparison once per frame, then 0.056 ms per box. The capture loop runs at
# ~21.6 fps with the detector capped at one call per second, so the 7.6 ms lands only on frames
# that already carry detections.)
W, H = 320, 180
COLOUR_W, COLOUR_H = 80, 45      # coarse colour, only for the illumination test

# --- Illumination, derived FROM THE FRAME (never from a clock) --------------------------------
# Keyed on the frame because a clock cannot know that a floodlight came on or that the trail cam
# flipped to IR, and the day<->night flip is the single largest photometric event in this corpus
# (lum 94.3 / DSSIM 0.87 -- larger than a camera reposition).
CHROMA_IR_MAX = 6.0              # no chroma at all => IR illuminator
NIGHT_MEDIAN_MAX = 90.0          # dark but coloured => night (flood / ambient)

# --- View epochs ------------------------------------------------------------------------------
VIEW_CORR_MIN = 0.55             # edge-fingerprint correlation below this = a different view
# The design doc's headline correction: a FRAME-COUNT debounce is the wrong shape. Measured over
# 2026-08-06, no debounce fired 30 times and a 3-frame debounce still fired 9 times -- all nine on
# the drift agent's three named transient haze/flare regimes -- because frame arrival is bursty
# and "3 frames" can be 3 seconds. Only disagreement SUSTAINED over wall-clock time is a
# reposition.
VIEW_PERSIST_S = 300.0
# ... and silence is not evidence either: if day frames stop arriving mid-disagreement the yard
# was not being watched, so the pending disagreement is dropped rather than bridged.
VIEW_MAX_GAP_S = 60.0
# The same rule applied to the TEMPLATE, which is what the shadow week showed it also needs.
# Measured 2026-08-09 on this glass door: with the camera provably stationary, day-frame
# fingerprints taken at different times of day correlate 0.075-0.68 -- far below VIEW_CORR_MIN.
# Nothing about that is a reposition; it is the sun. The template survives it only because it
# BLENDS toward every agreeing frame, i.e. it tracks the light as long as frames keep arriving.
# Across the night no day frame arrives at all, so the template freezes on an evening frame, and
# the first five continuous minutes of dawn read as five minutes of sustained disagreement: the
# rig bumped the epoch at 2026-08-09T05:56:52 at corr 0.261, while the two references four
# minutes either side of the bump correlate 0.987 -- identical framing, same wall, same pots.
# A template from before an interval nobody watched cannot testify that the camera moved, so it
# is re-seeded instead. THE COST IS REAL AND DELIBERATE: a reposition performed during a lull
# longer than this is never detected. That is the safe direction -- a missed epoch bump leaves a
# stale reference that the pixel test and refimg_max_age_s still have to get past, while a FALSE
# bump silently destroys the recurrence ledger every single morning.
VIEW_TEMPLATE_MAX_GAP_S = VIEW_PERSIST_S

# --- Certification ----------------------------------------------------------------------------
CERTIFY_QUIET_AREA = 200.0       # largest motion blob, 320x180 px^2 (~0.35% of frame)
CERTIFY_MIN_FRAMES = 4           # ... over at least this many detector verdicts
CERTIFY_MAX_GAP_S = 8.0          # a certification run may not bridge a hole in observation
MOTION_BLOB_MIN_PX = 25.0        # 320x180 px^2 below which a blob is sensor noise
MOTION_DILATE_PX = 4             # grow each remembered blob before subtracting it from cover
# 0.9 STAYS. It was re-examined against the live shadow week on 2026-08-09 and the bar is not what
# makes the veto thin -- the accumulation underneath it is. Replaying all 5,402 glass-door
# detections of 2026-08-07 21:21 -> 08-09 11:39 against the banked references (every box scored on
# its own crop, so nothing depends on clip frame timing) gives, at bars 0.9 / 0.7 / 0.5 / off:
# 26 / 34 / 68 / 117 suppressions -- and EVERY ONE of those 117, opened and looked at, is the same
# tipped watering can. Lowering the bar therefore buys more copies of an object already caught,
# while spending the one gate that answers the design's own photographed failure (a certified
# reference with an undetected raccoon walking the wall in it, design section 4.3). Measured cover
# fractions at real boxes over that window: day median 0.000 / max 0.825 -- NO day box has ever
# reached this bar, so on this camera the veto is a night instrument; night median 0.000 but
# 7.2% at or above 0.9, which is where all 18 live flags came from.
COVER_MIN_FRACTION = 0.9         # this share of a box's pixels must be KNOWN or the veto abstains

# --- Decisions --------------------------------------------------------------------------------
KEEP, SUPPRESS, ABSTAIN = "KEEP", "SUPPRESS", "ABSTAIN"
METRICS = ("lum", "dssim", "sobel")
SUPPRESSED_BY = "refimg_veto"                 # detections.suppressed_by
PROVENANCE_CERTIFIED = "certified"
PROVENANCE_MOTION_MASKED = "certified+motion_masked"
# Provenances carrying per-frame detector proof of emptiness. A motion-masked certified snapshot
# is the same frame with strictly MORE evidence, so it belongs here.
CERTIFIED_PROVENANCES = (PROVENANCE_CERTIFIED, PROVENANCE_MOTION_MASKED)

EPS = 1e-6


# =================================================================================================
# Illumination
# =================================================================================================

def illumination(frame_bgr) -> str:
    """'day' | 'night' | 'ir' for one BGR frame, at any resolution.

    chroma < 6 => 'ir'; else median luminance < 90 => 'night'; else 'day'. The same rule the drift
    agent's era clustering used. Switched, never blended: a reference is keyed on this value, so a
    frame that is halfway between two states must land in exactly one of them.
    """
    frame = np.asarray(frame_bgr)
    if frame.ndim == 2:                      # already grey: no chroma to measure, so never 'ir'
        gray = _to_working_gray(frame)
        return "night" if float(np.median(gray)) < NIGHT_MEDIAN_MAX else "day"
    return _illumination_of(_to_working_gray(frame), _to_working_colour(frame))


def _illumination_of(gray, colour) -> str:
    c = colour.astype(np.float32)
    chroma = float(np.mean(np.max(c, 2) - np.min(c, 2)))
    if chroma < CHROMA_IR_MAX:
        return "ir"
    return "night" if float(np.median(gray)) < NIGHT_MEDIAN_MAX else "day"


def _to_working_gray(frame):
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if g.shape[:2] == (H, W):
        return g
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)


def _to_working_colour(frame):
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.resize(frame, (COLOUR_W, COLOUR_H), interpolation=cv2.INTER_AREA)


# =================================================================================================
# Edge fingerprint + view epochs
# =================================================================================================

class EdgeFingerprint:
    """A zero-mean, unit-norm map of where the edges are -- the "is this the same camera view?"
    signature, stored as reference_images.edge_fp.

    Correlation is a plain dot product, so two fingerprints of the same view score near 1 and a
    cross-view pair falls away fast. VIEW_CORR_MIN = 0.55 is the drift agent's view-cluster rule.
    It is normalised precisely so it survives the things that are NOT a reposition: an exposure
    change, the auto-white-balance re-latch, the IR flash.
    """
    __slots__ = ("vec",)

    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    @classmethod
    def of(cls, gray) -> "EdgeFingerprint":
        f = cv2.GaussianBlur(np.asarray(gray).astype(np.float32), (3, 3), 0)
        s = np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3),
                     cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))
        s = cv2.GaussianBlur(s, (5, 5), 0)
        s -= s.mean()
        return cls(s / (float(np.linalg.norm(s)) + EPS))

    def correlate(self, other) -> float:
        o = other.vec if isinstance(other, EdgeFingerprint) else np.asarray(other, np.float32)
        return float((self.vec * o).sum())

    def blend(self, other, weight=0.1) -> "EdgeFingerprint":
        """Drift the template slowly toward a frame that still AGREES with it, so slow scene
        change (a plant growing, furniture moved by an inch) does not accumulate into a false
        reposition. Re-normalised, so the correlation scale is preserved."""
        o = other.vec if isinstance(other, EdgeFingerprint) else np.asarray(other, np.float32)
        v = (1.0 - weight) * self.vec + weight * o
        return EdgeFingerprint(v / (float(np.linalg.norm(v)) + EPS))

    def tobytes(self) -> bytes:
        return self.vec.astype(np.float32).tobytes()

    @classmethod
    def frombytes(cls, blob, shape=(H, W)) -> "EdgeFingerprint":
        return cls(np.frombuffer(blob, dtype=np.float32).reshape(shape))


class ViewWatcher:
    """Detects that the camera MOVED, and bumps a view epoch when it did.

    Two rules, both from measurements:

    DAY FRAMES ONLY. The drift agent found every IR frame across 12 days -- days that span
    confirmed repositions -- collapsing into a single view cluster. IR cannot see that the camera
    moved, so a night reference can never learn from itself that it has been invalidated. A day
    frame's verdict bumps the epoch for EVERY illumination, which is how the night reference finds
    out.

    WALL-CLOCK PERSISTENCE, NOT A FRAME COUNT. See VIEW_PERSIST_S: a 3-frame debounce still
    produced 9 false epochs in one day on the glass door, every one of them inside a known
    transient haze or flare regime, because 3 frames can be 3 seconds. Disagreement must be
    sustained across `persist_s` of wall clock, uninterrupted by a gap in observation, before the
    epoch moves.

    AND THE TEMPLATE IS RE-SEEDED ACROSS A GAP, not only the pending disagreement. See
    VIEW_TEMPLATE_MAX_GAP_S: this fingerprint does not survive a change of daylight (0.075-0.68
    across one stationary day), it only TRACKS it, and it cannot track across a night. Without
    this the rig calls sunrise a reposition every morning -- measured once on 2026-08-09, and the
    mechanism guarantees it repeats.

    An epoch bump is the loudest event in this module: it retires every reference and flushes the
    recurrence ledger, because both are expressed in image coordinates that no longer mean
    anything. Suppression then stops until a fresh reference certifies and a spot re-earns its
    recurrence -- at least two calendar days. That is the intended cost.
    """

    def __init__(self, corr_min=VIEW_CORR_MIN, persist_s=VIEW_PERSIST_S,
                 max_gap_s=VIEW_MAX_GAP_S, epoch=0, template_max_gap_s=VIEW_TEMPLATE_MAX_GAP_S):
        self.corr_min = float(corr_min)
        self.persist_s = float(persist_s)
        self.max_gap_s = float(max_gap_s)
        self.template_max_gap_s = float(template_max_gap_s)
        self.epoch = int(epoch)
        self.template: EdgeFingerprint | None = None
        self.changes: list[tuple[float, float]] = []   # (ts, correlation) of each bump
        self.reseeds = 0            # templates dropped across an unwatched gap (dawn, mostly)
        self._since = None          # when the current disagreement started
        self._worst = None          # lowest correlation seen during it
        self._last_day_ts = None

    def observe(self, ts, illum, fingerprint) -> int:
        """Feed one frame; returns the current epoch. Non-day frames are ignored (but still get
        the current epoch back, which is what keys their reference)."""
        if illum != "day":
            return self.epoch
        ts = float(ts)
        gap = None if self._last_day_ts is None else ts - self._last_day_ts
        self._last_day_ts = ts
        if self.template is None:
            self.template = fingerprint
            return self.epoch
        if gap is not None and gap > self.max_gap_s:
            # We stopped watching. Whatever was pending is not evidence of anything.
            self._since = self._worst = None
            if gap > self.template_max_gap_s:
                # ... and neither is the template. Disagreement is only evidence when it is
                # SUSTAINED across an interval we actually watched, so a template from before an
                # interval we did not watch cannot start that clock. This is the line that stops
                # sunrise reading as a camera move; see VIEW_TEMPLATE_MAX_GAP_S.
                self.template = fingerprint
                self.reseeds += 1
                return self.epoch
        corr = self.template.correlate(fingerprint)
        if corr >= self.corr_min:
            self._since = self._worst = None
            self.template = self.template.blend(fingerprint)
            return self.epoch
        if self._since is None:
            self._since = ts
            self._worst = corr
            return self.epoch
        self._worst = min(self._worst, corr)
        if ts - self._since >= self.persist_s:
            self.epoch += 1
            self.changes.append((ts, round(self._worst, 3)))
            self.template = fingerprint
            self._since = self._worst = None
        return self.epoch


# =================================================================================================
# Pixel metrics
# =================================================================================================

def photometric_align(b, a):
    """Map image `b` onto `a`'s luminance with a robust gain+offset fitted on percentiles.

    This is what lets a comparison survive the glass door's auto-white-balance re-latching and the
    trail cam's IR flash: both are (approximately) GLOBAL photometric transforms, and a veto that
    fired on them would be reporting the light, not the yard.
    """
    pb = np.percentile(b, [10, 50, 90])
    pa = np.percentile(a, [10, 50, 90])
    g = (pa[2] - pa[0]) / max(1e-3, pb[2] - pb[0])
    g = float(np.clip(g, 0.2, 5.0))
    return b * g + (pa[1] - g * pb[1])


def sobel_field(g):
    f = cv2.GaussianBlur(np.asarray(g).astype(np.float32), (3, 3), 0)
    return np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))


def ssim_map(a, b, C1=6.5025, C2=58.5225):
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)
    k, s = (11, 11), 1.5
    mu1 = cv2.GaussianBlur(a, k, s)
    mu2 = cv2.GaussianBlur(b, k, s)
    m11, m22, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s11 = cv2.GaussianBlur(a * a, k, s) - m11
    s22 = cv2.GaussianBlur(b * b, k, s) - m22
    s12 = cv2.GaussianBlur(a * b, k, s) - m12
    return ((2 * m12 + C1) * (2 * s12 + C2)) / ((m11 + m22 + C1) * (s11 + s22 + C2) + EPS)


def census_field(g, k=5):
    """Per-pixel "am I brighter than my neighbourhood?" -- an illumination-invariant texture sign.

    Emitted but NOT part of the shipped conjunction: measured in IR it is INVERTED (furniture
    scores HIGHER than animals), so it must never be used in the dark. It stays in the scores dict
    because the whole point of suppress_detail is that a bad suppression is diagnosable months
    later without re-running anything.
    """
    g = np.asarray(g).astype(np.float32)
    return g > cv2.blur(g, (k, k))


class FramePair:
    """One frame against one reference, prepared once and then sliced per box.

    Bigger is more different, on every metric. The arithmetic is identical to the drift agent's
    metrics.Pair, so its measured thresholds transfer unchanged -- do not "improve" it.
    """

    def __init__(self, frame_gray, ref_gray):
        self.A = np.asarray(frame_gray).astype(np.float32)
        self.B = np.asarray(ref_gray).astype(np.float32)
        self.Bn = photometric_align(self.B, self.A)
        sA, sB = sobel_field(self.A), sobel_field(self.B)
        self.sAn = sA / (sA.mean() + EPS)
        self.sBn = sB / (sB.mean() + EPS)
        self.ss = ssim_map(self.A, self.Bn)
        self.cx = census_field(self.A) ^ census_field(self.B)

    def scores(self, box=None) -> dict:
        sl = _box_slice(box)
        return dict(
            lum=float(np.abs(self.A[sl] - self.Bn[sl]).mean()),
            sobel=float(np.abs(self.sAn[sl] - self.sBn[sl]).mean()),
            dssim=float(1.0 - self.ss[sl].mean()),
            census=float(self.cx[sl].mean()),
        )


def _box_slice(box):
    if box is None:
        return (slice(None), slice(None))
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1 = max(0, min(W - 2, x1))
    y1 = max(0, min(H - 2, y1))
    x2 = max(x1 + 2, min(W, x2))
    y2 = max(y1 + 2, min(H, y2))
    return (slice(y1, y2), slice(x1, x2))


def scale_box(box, frame_w, frame_h):
    """A detector box in full-frame pixels -> the 320x180 working frame. Every box that crosses
    this module's boundary is scaled here, so nothing downstream ever holds a capture-resolution
    coordinate (the rig has already changed capture resolution once, 1280x720 -> 1920x1080)."""
    x1, y1, x2, y2 = box
    sx, sy = W / float(frame_w), H / float(frame_h)
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# =================================================================================================
# The reference
# =================================================================================================

@dataclass
class Reference:
    """One certified-empty view of the yard, plus the mask of what it actually KNOWS.

    `cover` is load-bearing and is the difference between policy A and policy E. An unknown pixel
    is not evidence of emptiness, so the veto abstains wherever cover is False rather than
    comparing against a fabrication.
    """
    image: np.ndarray                       # (H, W) uint8 grey at the working resolution
    captured_at: float                      # epoch seconds
    illumination: str
    view_epoch: int
    provenance: str = PROVENANCE_MOTION_MASKED
    fingerprint: EdgeFingerprint | None = None
    cover: np.ndarray | None = None         # (H, W) bool; None == fully covered
    source: str | None = None
    id: int | None = None                   # reference_images.id once persisted
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.cover is None:
            self.cover = np.ones((H, W), bool)
        if self.fingerprint is None:
            self.fingerprint = EdgeFingerprint.of(self.image)

    def covered(self, box, need=COVER_MIN_FRACTION) -> bool:
        return float(self.cover[_box_slice(box)].mean()) >= need

    def cover_fraction(self, box=None) -> float:
        return float(self.cover[_box_slice(box)].mean())

    def age_s(self, now) -> float:
        return float(now) - float(self.captured_at)


class ReferenceManager:
    """POLICY E: a detector-certified empty frame, with every recently-moving pixel disowned.

    Certification, per illumination state:

        for `hold_s` of continuous wall clock and at least `min_frames` verdicts, the detector
        RAN on every frame, returned zero boxes, and the largest motion blob stayed under
        `quiet_area` -- with no gap in observation longer than `max_gap_s`.

    `detections is None` means the detector did not run on this frame and is NOT the same as
    `detections == []`. That distinction is the whole ballgame (design §3.1): treating "no
    detection nearby" as certification stores a reference with the sleeping raccoon in it.

    Coverage: every motion blob seen in the last `no_update_s` (3600 s, from the stationarity
    floor -- 823 s of eye-verified residency, 12,657 s merged worst case) is dilated and subtracted
    from the reference's cover mask. So an animal the detector MISSED still costs only an
    abstention, while the watering can -- which fires the detector constantly but never moves --
    stays vetoable.

    That memory is ONE ARRAY OF TIMESTAMPS, not a list of rectangles, and the difference is two
    measured bugs rather than a refactor. The first shipped version remembered
    `cv2.boundingRect()` of each blob, which disowns pixels that never moved: measured over the
    13,438 motion-positive frames of 2026-08-09 00:00-05:30, a frame's bounding boxes claim 1.37x
    the area its blobs actually occupy (median 1.33, p90 1.53). The second is cost -- it kept every
    rectangle of the last hour and re-drew all of them on EVERY frame, on the capture thread:
    measured 4.9 ms at 2,500 remembered rectangles and 16.7 ms at 10,000, against the 7.6 ms the
    whole per-frame veto was budgeted at, and the busiest measured hour on this rig ran 1,546
    detector frames. A per-pixel "last blobbed at" map is 460 KB flat, O(1) per frame, and disowns
    exactly the pixels something moved over.

    Be clear about what that fix is NOT: it does not unblock the veto. Re-measured on the same
    night, the honest cover at real detection boxes goes from median 0.000 to 0.009 and 0.0% of
    boxes reach COVER_MIN_FRACTION either way. A detector box is, for anything that moves, exactly
    the pixels that just moved -- so coverage abstaining on animals is the gate WORKING, and the
    thin part is that furniture's own pixels get disowned too whenever something passes near it.

    The counters (`n_detector_runs`, `n_detector_empty`, `n_certified`, `longest_empty_run_s`) are
    design §8 item 1: true certification availability is unmeasurable offline, because the DB only
    records frames where something WAS found. The shadow week is what measures it.
    """

    def __init__(self, source=None, hold_s=None, no_update_s=None, quiet_area=CERTIFY_QUIET_AREA,
                 min_frames=CERTIFY_MIN_FRAMES, max_gap_s=CERTIFY_MAX_GAP_S,
                 blob_min_px=MOTION_BLOB_MIN_PX, dilate_px=MOTION_DILATE_PX, view=None, cfg=None):
        cfg = cfg or config.CONFIG
        self.source = source or getattr(cfg, "source", "glass_door_cam")
        self.hold_s = float(hold_s if hold_s is not None else cfg.refimg_certify_hold_s)
        self.no_update_s = float(no_update_s if no_update_s is not None else cfg.refimg_no_update_s)
        self.quiet_area = float(quiet_area)
        self.min_frames = int(min_frames)
        self.max_gap_s = float(max_gap_s)
        self.blob_min_px = float(blob_min_px)
        self.dilate_px = int(dilate_px)
        self.view = view if view is not None else ViewWatcher()

        self._refs: dict[str, Reference] = {}
        # Per-pixel "when did something last blob over you", at the working resolution. -inf means
        # "never", so an untouched map covers the whole frame without a special case.
        self._motion_at = np.full((H, W), -np.inf, np.float64)
        self._run_start = None
        self._run_illum = None
        self._run_n = 0
        self._last_ts = None
        self._epoch = self.view.epoch
        # Shadow-week instrumentation (design §8.1).
        self.n_detector_runs = 0
        self.n_detector_empty = 0
        self.n_certified = 0
        self.longest_empty_run_s = 0.0
        self.epoch_changes = 0

    # -- the state machine -----------------------------------------------------------------
    def observe(self, frame, detections=None, motion_mask=None, now=None) -> "Observation":
        """Feed one captured frame; returns the prepared `Observation` to hand to the veto.

        frame        a BGR (or grey) ndarray at any resolution, or an already-prepared Observation
        detections   the detector's boxes for THIS frame in full-frame pixels, [] for "it ran and
                     found nothing", or None for "the detector did not run" (never certifies)
        motion_mask  MotionGate's foreground mask, any resolution (the rig already computes it and
                     throws it away). None = no motion channel, which only ever means MORE cover
                     and therefore fewer abstentions -- so pass it whenever you have it.
        now          epoch seconds; defaults to wall clock
        """
        obs = frame if isinstance(frame, Observation) else prepare(frame, now=now)
        if now is not None:
            obs = replace(obs, ts=float(now))
        obs.detector_ran = detections is not None
        obs.boxes = tuple(obs.to_working(b) for b in (detections or ()))

        area, blobs, footprint = _blobs(motion_mask, self.blob_min_px, self.dilate_px)
        obs.motion_area = area
        obs.motion_boxes = tuple(blobs)

        epoch = self.view.observe(obs.ts, obs.illumination, obs.fingerprint)
        obs.view_epoch = epoch
        if epoch != self._epoch:
            self._epoch = epoch
            self.epoch_changes += 1
            self.flush()

        if footprint is not None:
            self._motion_at[footprint] = obs.ts

        self._advance(obs)
        return obs

    def _advance(self, obs):
        gap = None if self._last_ts is None else obs.ts - self._last_ts
        self._last_ts = obs.ts
        if obs.detector_ran:
            self.n_detector_runs += 1
            if not obs.boxes:
                self.n_detector_empty += 1

        quiet = obs.detector_ran and not obs.boxes and obs.motion_area <= self.quiet_area
        broke = (gap is not None and gap > self.max_gap_s) or obs.illumination != self._run_illum
        if not quiet or broke:
            # A detection, a motion blob, a hole in observation or an illumination flip all reset
            # the clock. The animal's own detections are what protect it from being learned.
            self._run_start = obs.ts if quiet else None
            self._run_illum = obs.illumination if quiet else None
            self._run_n = 1 if quiet else 0
            return
        if self._run_start is None:
            self._run_start = obs.ts
            self._run_illum = obs.illumination
            self._run_n = 1
            return
        self._run_n += 1
        held = obs.ts - self._run_start
        self.longest_empty_run_s = max(self.longest_empty_run_s, held)
        if held >= self.hold_s and self._run_n >= self.min_frames:
            self._refs[obs.illumination] = Reference(
                image=obs.gray.copy(), captured_at=obs.ts, illumination=obs.illumination,
                view_epoch=obs.view_epoch, provenance=PROVENANCE_MOTION_MASKED,
                fingerprint=obs.fingerprint, source=self.source,
                detail=dict(held_s=round(held, 2), n_verdicts=self._run_n))
            self.n_certified += 1

    def flush(self):
        """Drop everything. Called on a view-epoch change: a reference for a scene that no longer
        exists is exactly the silent failure config.py warns about with hand-measured zones."""
        self._refs.clear()
        self._motion_at.fill(-np.inf)
        self._run_start = self._run_illum = None
        self._run_n = 0

    # -- reading it back -------------------------------------------------------------------
    def get(self, illumination, now=None) -> Reference | None:
        """The current reference for an illumination state, with its cover mask applied.

        Age is NOT checked here -- the veto owns that gate, so a stale reference produces an
        explicit ABSTAIN with an age in the trace instead of silently vanishing.

        `now` defaults to the last frame observed, which is what the capture loop wants: coverage
        is read immediately after observe(), and reading it against a LATER clock would quietly
        expire motion the reference has not yet been re-certified against.
        """
        ref = self._refs.get(illumination)
        if ref is None:
            return None
        if now is None:
            now = self._last_ts if self._last_ts is not None else 0.0
        blocked = (float(now) - self._motion_at) <= self.no_update_s
        return Reference(
            image=ref.image, captured_at=ref.captured_at, illumination=ref.illumination,
            view_epoch=ref.view_epoch, provenance=PROVENANCE_MOTION_MASKED,
            fingerprint=ref.fingerprint, cover=~blocked, source=ref.source, id=ref.id,
            detail=dict(ref.detail, masked_fraction=round(float(blocked.mean()), 4)))

    @property
    def view_epoch(self) -> int:
        return self._epoch


@dataclass
class Observation:
    """One prepared frame: everything the veto needs, computed once per frame rather than per box.

    Held separately from the raw frame because the expensive parts (the working-resolution grey,
    the edge fingerprint) are shared by every box in the frame and by the reference manager.
    """
    ts: float
    gray: np.ndarray
    illumination: str
    fingerprint: EdgeFingerprint
    frame_w: int
    frame_h: int
    detector_ran: bool = False
    boxes: tuple = ()
    motion_area: float = 0.0
    motion_boxes: tuple = ()
    view_epoch: int = 0

    def to_working(self, box):
        return scale_box(box, self.frame_w, self.frame_h)


def prepare(frame_bgr, now=None) -> Observation:
    """Downscale one captured frame to the working resolution and derive its scene state."""
    frame = np.asarray(frame_bgr)
    h, w = frame.shape[:2]
    gray = _to_working_gray(frame)
    illum = _illumination_of(gray, _to_working_colour(frame))
    return Observation(ts=float(now if now is not None else datetime.now().timestamp()),
                       gray=gray, illumination=illum, fingerprint=EdgeFingerprint.of(gray),
                       frame_w=w, frame_h=h)


def _blobs(motion_mask, blob_min_px, dilate_px=MOTION_DILATE_PX):
    """(largest blob area, [bounding boxes], footprint mask) from a MotionGate foreground mask.

    Everything is at the 320x180 working resolution. The FOOTPRINT is what coverage consumes: each
    kept blob's own filled outline, dilated by `dilate_px`. The boxes are informational only
    (`Observation.motion_boxes`).

    Filled outline, NOT the bounding rectangle -- the rectangle was what the first version
    remembered, and it disowns pixels nothing ever moved over. Measured over the 13,438
    motion-positive frames of 2026-08-09 00:00-05:30 on this camera: a frame's bounding boxes claim
    1.37x the area of its blobs (median 1.33, p90 1.53). The outline is filled rather than used
    raw because MOG2's foreground on a night animal is ragged -- an unfilled silhouette would leave
    holes inside the very body the coverage gate exists to protect.

    The prototype kept only the LARGEST blob's box, because that is all the rig's MotionGate
    exposes. Keeping every blob above the noise floor is strictly more conservative: it can only
    remove more pixels from the cover mask, i.e. abstain more and suppress less.
    """
    if motion_mask is None:
        return 0.0, [], None
    m = np.asarray(motion_mask)
    if m.dtype != np.uint8:
        m = (m.astype(bool) * 255).astype(np.uint8)
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest, boxes, keep = 0.0, [], []
    for c in contours:
        a = float(cv2.contourArea(c))
        largest = max(largest, a)
        if a >= blob_min_px:
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append((float(x), float(y), float(x + bw), float(y + bh)))
            keep.append(c)
    if not keep:
        return largest, boxes, None
    footprint = np.zeros((H, W), np.uint8)
    cv2.drawContours(footprint, keep, -1, 1, cv2.FILLED)
    d = int(dilate_px)
    if d > 0:
        footprint = cv2.dilate(
            footprint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    return largest, boxes, footprint.astype(bool)


# =================================================================================================
# The recurrence ledger
# =================================================================================================

@dataclass
class RecurrenceCluster:
    box: tuple
    n: int = 1
    events: int = 1
    first: float = 0.0
    last: float = 0.0
    days: set = field(default_factory=set)

    def stats(self) -> dict:
        return dict(events=self.events, days=len(self.days), n=self.n)


class Recurrence:
    """"Has something kept firing in exactly this spot, on more than one day?"

    An online, restart-surviving version of staticfilter.py's clustering rule: group boxes by IoU,
    count INDEPENDENT firings (two hits less than `event_gap_s` apart are one occasion) and
    distinct calendar days.

    ON ITS OWN THIS GATE IS WORTHLESS AND MUST NEVER BE USED ALONE. The food bowl in this yard
    recurs on 27 separate days at confidence 0.95, and every one of those is a real raccoon; the
    glass-door label audit found a cluster of 34 detections spanning 1,646 minutes over two days
    that is a raccoon on the wall on two consecutive nights. Location recurrence does not separate
    furniture from animals here. It earns its place only ANDed with the pixel test, where it saved
    422 of 4,649 animal boxes (9.1%) that had passed all three metrics.

    PERSISTENCE: a small JSON sidecar in refimg_store_dir, not the DB. Deliberate -- this ledger
    updates on every detection in the capture thread, and this project has already killed the rig
    once by holding the SQLite write lock from a background worker ("database is locked", the
    naming helper's BioCLIP backlog). A sidecar rewritten atomically at most every `save_every_s`
    costs the capture loop nothing and loses at most that much history on a hard kill. Boxes are
    stored in 320x180 WORKING coordinates, so a capture-resolution change cannot corrupt it.
    """

    VERSION = 1

    def __init__(self, iou=None, event_gap_s=None, min_events=None, min_days=None,
                 path=None, retain_days=30, save_every_s=300.0, cfg=None):
        cfg = cfg or config.CONFIG
        self.iou = float(iou if iou is not None else cfg.refimg_recurrence_iou)
        self.event_gap_s = float(event_gap_s if event_gap_s is not None
                                 else cfg.refimg_recurrence_event_gap_s)
        self.min_events = int(min_events if min_events is not None
                              else cfg.refimg_recurrence_min_events)
        self.min_days = int(min_days if min_days is not None else cfg.refimg_recurrence_min_days)
        self.path = Path(path) if path else None
        self.retain_days = float(retain_days)
        self.save_every_s = float(save_every_s)
        self.clusters: list[RecurrenceCluster] = []
        self.epoch = 0
        self._last_save = 0.0

    # -- online updates --------------------------------------------------------------------
    def observe(self, box, when, epoch=None) -> RecurrenceCluster:
        """Record one detection at `box` (working coordinates) at time `when`."""
        if epoch is not None and epoch != self.epoch:
            # The camera moved: every stored box points at a spot that is no longer there.
            self.clusters.clear()
            self.epoch = int(epoch)
        ts, day = _ts_and_day(when)
        cl, best = self._match(box)
        if cl is None:
            cl = RecurrenceCluster(box=tuple(float(v) for v in box), first=ts, last=ts, days={day})
            self.clusters.append(cl)
            return cl
        cl.n += 1
        if ts - cl.last >= self.event_gap_s:
            cl.events += 1
        cl.last = max(cl.last, ts)
        cl.days.add(day)
        # Drift the anchor slowly toward the new box: a detector's box on one static object
        # wobbles by a pixel or two, and a hard anchor would slowly shed its own cluster.
        cl.box = tuple(float(0.9 * a + 0.1 * b) for a, b in zip(cl.box, box))
        return cl

    def stats(self, box) -> dict:
        cl, _ = self._match(box)
        return cl.stats() if cl is not None else dict(events=0, days=0, n=0)

    def satisfied(self, box) -> bool:
        s = self.stats(box)
        return s["events"] >= self.min_events and s["days"] >= self.min_days

    def _match(self, box):
        best, score = None, 0.0
        for cl in self.clusters:
            v = box_iou(box, cl.box)
            if v > score:
                best, score = cl, v
        return (best, score) if score >= self.iou else (None, score)

    # -- persistence -----------------------------------------------------------------------
    def prune(self, now=None):
        """Forget clusters that stopped firing. A spot silent for `retain_days` is either gone or
        the camera moved; either way its evidence is stale and must be re-earned."""
        now = float(now if now is not None else datetime.now().timestamp())
        cutoff = now - self.retain_days * 86400.0
        self.clusters = [c for c in self.clusters if c.last >= cutoff]

    def to_dict(self) -> dict:
        return dict(version=self.VERSION, epoch=self.epoch, iou=self.iou,
                    event_gap_s=self.event_gap_s, working_size=[W, H],
                    clusters=[dict(box=list(c.box), n=c.n, events=c.events, first=c.first,
                                   last=c.last, days=sorted(c.days)) for c in self.clusters])

    def load_dict(self, data):
        if not data or int(data.get("version", 0)) != self.VERSION:
            return self
        if list(data.get("working_size", [W, H])) != [W, H]:
            return self       # boxes from another working resolution mean nothing here
        self.epoch = int(data.get("epoch", 0))
        self.clusters = [RecurrenceCluster(box=tuple(d["box"]), n=int(d["n"]),
                                           events=int(d["events"]), first=float(d["first"]),
                                           last=float(d["last"]), days=set(d.get("days", [])))
                         for d in data.get("clusters", [])]
        return self

    def load(self):
        if self.path and self.path.exists():
            try:
                self.load_dict(json.loads(self.path.read_text(encoding="utf-8")))
            except (ValueError, OSError, KeyError):
                pass          # a corrupt ledger costs recurrence evidence, never an erasure
        return self

    def save(self, now=None, force=False) -> bool:
        """Atomic rewrite, throttled. Returns True when it actually wrote."""
        if self.path is None:
            return False
        now = float(now if now is not None else datetime.now().timestamp())
        if not force and now - self._last_save < self.save_every_s:
            return False
        self.prune(now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        os.replace(tmp, self.path)
        self._last_save = now
        return True


def _local(ts) -> datetime:
    """Epoch seconds -> local time, tz-aware. Routed through UTC on purpose: a bare
    datetime.fromtimestamp() raises OSError on Windows for timestamps near or below the epoch,
    which turns a synthetic or clock-skewed value into a crash instead of a date."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone()


def _ts_and_day(when):
    if isinstance(when, datetime):
        return when.timestamp(), when.date().isoformat()
    ts = float(when)
    return ts, _local(ts).date().isoformat()


# =================================================================================================
# The veto
# =================================================================================================

@dataclass(frozen=True)
class Decision:
    """The whole decision, serialisable to detections.suppress_detail.

    Written in full even when the answer is KEEP or ABSTAIN, so a bad suppression is diagnosable
    months later without re-running anything -- which is the only way the shadow week produces a
    number rather than an impression.
    """
    decision: str
    reason: str
    provenance: str | None = None
    view_corr: float | None = None
    view_epoch: int | None = None
    age_s: float | None = None
    cover: float | None = None
    scores: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    recurrence: dict = field(default_factory=dict)
    ref_id: int | None = None

    @property
    def suppressed(self) -> bool:
        return self.decision == SUPPRESS

    def to_detail(self) -> dict:
        """The suppress_detail payload, exactly as specified in the design doc §7."""
        d = dict(decision=self.decision, reason=self.reason)
        for key in ("provenance", "view_corr", "view_epoch", "age_s", "cover", "ref_id"):
            v = getattr(self, key)
            if v is not None:
                d[key] = v
        for key in ("scores", "thresholds", "recurrence"):
            v = getattr(self, key)
            if v:
                d[key] = v
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_detail(), sort_keys=True)

    @classmethod
    def from_detail(cls, detail) -> "Decision":
        d = dict(detail or {})
        return cls(decision=d.get("decision", ABSTAIN), reason=d.get("reason", "unknown"),
                   provenance=d.get("provenance"), view_corr=d.get("view_corr"),
                   view_epoch=d.get("view_epoch"), age_s=d.get("age_s"), cover=d.get("cover"),
                   scores=d.get("scores", {}), thresholds=d.get("thresholds", {}),
                   recurrence=d.get("recurrence", {}), ref_id=d.get("ref_id"))

    @classmethod
    def from_json(cls, text) -> "Decision":
        try:
            return cls.from_detail(json.loads(text))
        except (ValueError, TypeError):
            return cls(decision=ABSTAIN, reason="unreadable_detail")


class ShadowVeto:
    """The conjunction. Deterministic and pure over its inputs; it writes nothing.

    Gate order, and what each one costs when it fails (always ABSTAIN or KEEP, never a suppression
    by default):

      0  a reference exists ....................... ABSTAIN no_reference
      1  same view epoch .......................... ABSTAIN view_epoch_changed
      2  edge fingerprint still correlates ........ ABSTAIN camera_moved
      3  reference younger than max_age_s ......... ABSTAIN reference_stale
      4  the reference KNOWS these pixels ......... ABSTAIN reference_has_no_pixels_here
      5  lum AND dssim AND sobel below threshold .. KEEP    pixels_differ_from_empty
      6  recurrence: >= min_events over >= min_days  KEEP    not_recurrent_enough
      -> SUPPRESS matches_empty_reference_at_a_recurring_spot

    Gates 1-4 exist because a reference that is stale, cross-view or ignorant of a region vetoes
    essentially at random -- a cross-view empty pair exceeded the safe metric threshold 100% of the
    time in the drift study. Gates 5 and 6 are each necessary and each insufficient; see the module
    docstring for the two measurements that pin that down.

    Thresholds come from `config.refimg_metrics[illumination]` and are PER CAMERA. The glass-door
    defaults must not be pointed at another camera without re-measuring: in IR the trail cam's
    `lum` and `sobel` separate cleanly while `dssim` kills 7 of 60 real raccoons.
    """

    def __init__(self, thresholds=None, max_age_s=None, view_corr_min=VIEW_CORR_MIN,
                 cover_min=COVER_MIN_FRACTION, cfg=None):
        cfg = cfg or config.CONFIG
        # Copied a level deep: a veto that mutated its thresholds in place would silently retune
        # every other veto in the process, config.CONFIG included.
        src = thresholds if thresholds is not None else cfg.refimg_metrics
        self.thresholds = {k: dict(v) for k, v in src.items()}
        self.max_age_s = float(max_age_s if max_age_s is not None else cfg.refimg_max_age_s)
        self.view_corr_min = float(view_corr_min)
        self.cover_min = float(cover_min)
        self._pair_key = None
        self._pair = None

    def evaluate(self, box, frame, ref, recurrence) -> Decision:
        """Judge ONE detector box.

        box          the box in 320x180 working coordinates (see scale_box)
        frame        the Observation for the frame the box came from
        ref          a Reference, or None
        recurrence   a Recurrence ledger (already updated with this box)
        """
        if ref is None:
            return Decision(decision=ABSTAIN, reason="no_reference")

        base = dict(provenance=ref.provenance, ref_id=ref.id, view_epoch=ref.view_epoch)

        if ref.view_epoch != frame.view_epoch:
            return Decision(decision=ABSTAIN, reason="view_epoch_changed", **base)

        corr = round(ref.fingerprint.correlate(frame.fingerprint), 3)
        base["view_corr"] = corr
        if corr < self.view_corr_min:
            return Decision(decision=ABSTAIN, reason="camera_moved", **base)

        age = round(ref.age_s(frame.ts), 1)
        base["age_s"] = age
        if age > self.max_age_s or age < 0:
            return Decision(decision=ABSTAIN, reason="reference_stale", **base)

        cover = round(ref.cover_fraction(box), 4)
        base["cover"] = cover
        if cover < self.cover_min:
            # Policy E's whole purpose: something moved here within the hour, so the reference
            # cannot vouch for these pixels. An undetected animal costs an abstention, not a row.
            return Decision(decision=ABSTAIN, reason="reference_has_no_pixels_here", **base)

        thresholds = self.thresholds.get(frame.illumination)
        if not thresholds:
            return Decision(decision=ABSTAIN, reason="no_thresholds_for_illumination", **base)

        scores = self._pair_for(frame, ref).scores(box)
        base["scores"] = {k: round(v, 4) for k, v in scores.items()}
        base["thresholds"] = dict(thresholds)
        if not all(scores[m] < thresholds[m] for m in METRICS if m in thresholds):
            return Decision(decision=KEEP, reason="pixels_differ_from_empty", **base)

        stats = recurrence.stats(box)
        base["recurrence"] = stats
        if stats["events"] < recurrence.min_events or stats["days"] < recurrence.min_days:
            return Decision(decision=KEEP, reason="not_recurrent_enough", **base)

        return Decision(decision=SUPPRESS,
                        reason="matches_empty_reference_at_a_recurring_spot", **base)

    def _pair_for(self, frame, ref) -> FramePair:
        """One FramePair per (frame, reference), reused across the frame's boxes -- the 7.6 ms
        construction is amortised over the frame's detections, leaving 0.056 ms per box."""
        key = (id(frame), frame.ts, id(ref), ref.captured_at)
        if key != self._pair_key:
            self._pair = FramePair(frame.gray, ref.image)
            self._pair_key = key
        return self._pair


# =================================================================================================
# Persistence: reference_images, view_epochs, and the suppression columns
# =================================================================================================
# db.py owns the real schema. These helpers exist so refimg can be exercised standalone and in
# tests, and so the integration agent has one obvious place to wire each write. Every statement is
# idempotent, so running them alongside db.py's own migration is harmless.

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS reference_images (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT    NOT NULL,
    illumination  TEXT    NOT NULL,
    view_epoch    INTEGER NOT NULL,
    captured_at   TEXT    NOT NULL,
    provenance    TEXT    NOT NULL,
    image_path    TEXT    NOT NULL,
    cover_path    TEXT,
    edge_fp       BLOB,
    n_frames      INTEGER,
    span_s        REAL,
    retired_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_refimg_lookup
    ON reference_images(source, illumination, view_epoch, captured_at);

CREATE TABLE IF NOT EXISTS view_epochs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT    NOT NULL,
    epoch        INTEGER NOT NULL,
    started_at   TEXT    NOT NULL,
    detected_by  TEXT    NOT NULL,
    corr         REAL,
    UNIQUE(source, epoch)
);
"""

SUPPRESSION_COLUMNS = {
    "suppressed_at": "TEXT",        # local ISO 8601 w/ offset; NULL == a live row
    "suppressed_by": "TEXT",        # 'refimg_veto' | 'staticfilter' | ...
    "suppress_ref_id": "INTEGER",   # -> reference_images.id
    "suppress_detail": "TEXT",      # the JSON gate trace (Decision.to_json)
}


def ensure_tables(conn):
    """Create the additive tables/columns if they are not already there. Never drops anything."""
    conn.executescript(SCHEMA_SQL)
    if _table_exists(conn, "detections"):
        have = {r[1] for r in conn.execute("PRAGMA table_info(detections)")}
        for name, decl in SUPPRESSION_COLUMNS.items():
            if name not in have:
                conn.execute(f"ALTER TABLE detections ADD COLUMN {name} {decl}")
    conn.commit()


def _table_exists(conn, name) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def _iso(ts) -> str:
    """Local time WITH offset, matching the project's one timestamp convention (db.now_local_iso)."""
    return _local(ts).isoformat()


def store_dir(cfg=None) -> Path:
    return Path((cfg or config.CONFIG).refimg_store_dir)


def save_reference(conn, ref: Reference, root=None, cfg=None) -> int:
    """Write the reference PNGs and insert its reference_images row; returns the new id.

    Full frames of the yard, so they live under refimg_store_dir (gitignored, like crops/) and
    never under crops/, which backup.py zips per capture day.
    """
    cfg = cfg or config.CONFIG
    base = Path(root) if root else store_dir(cfg)
    source = ref.source or getattr(cfg, "source", "glass_door_cam")
    folder = base / source / ref.illumination
    folder.mkdir(parents=True, exist_ok=True)
    stamp = _local(ref.captured_at).strftime("%Y%m%d-%H%M%S")
    stem = f"ref-e{ref.view_epoch}-{stamp}"
    img_path = folder / f"{stem}.png"
    cv2.imwrite(str(img_path), ref.image)
    cover_path = None
    if not bool(ref.cover.all()):
        cover_path = folder / f"{stem}-cover.png"
        cv2.imwrite(str(cover_path), (ref.cover.astype(np.uint8) * 255))
    cur = conn.execute(
        "INSERT INTO reference_images (source, illumination, view_epoch, captured_at, provenance,"
        " image_path, cover_path, edge_fp, n_frames, span_s) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (source, ref.illumination, ref.view_epoch, _iso(ref.captured_at), ref.provenance,
         str(img_path), str(cover_path) if cover_path else None,
         ref.fingerprint.tobytes(), int(ref.detail.get("n_verdicts", 1) or 1),
         float(ref.detail.get("held_s", 0.0) or 0.0)))
    conn.commit()
    ref.id = int(cur.lastrowid)
    return ref.id


def load_reference(conn, source, illumination, view_epoch) -> Reference | None:
    """The newest un-retired reference for a (source, illumination, view_epoch), or None."""
    row = conn.execute(
        "SELECT id, captured_at, provenance, image_path, cover_path, edge_fp FROM reference_images"
        " WHERE source=? AND illumination=? AND view_epoch=? AND retired_at IS NULL"
        " ORDER BY captured_at DESC LIMIT 1", (source, illumination, view_epoch)).fetchone()
    if row is None:
        return None
    rid, captured_at, provenance, image_path, cover_path, edge_fp = row
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    cover = None
    if cover_path:
        c = cv2.imread(str(cover_path), cv2.IMREAD_GRAYSCALE)
        cover = None if c is None else (c > 127)
    fp = EdgeFingerprint.frombytes(edge_fp) if edge_fp else None
    return Reference(image=image, captured_at=datetime.fromisoformat(captured_at).timestamp(),
                     illumination=illumination, view_epoch=view_epoch, provenance=provenance,
                     fingerprint=fp, cover=cover, source=source, id=rid)


def retire_references(conn, source, view_epoch, when=None):
    """Stamp retired_at on a superseded epoch's references. NOTHING is deleted -- a suppression
    must stay replayable against the exact image that produced it."""
    conn.execute("UPDATE reference_images SET retired_at=? WHERE source=? AND view_epoch=?"
                 " AND retired_at IS NULL",
                 (_iso(when if when is not None else datetime.now().timestamp()),
                  source, view_epoch))
    conn.commit()


def record_view_epoch(conn, source, epoch, started_at=None, corr=None, detected_by="edge_fp_corr"):
    """Record that the camera was seen to move, so it is a first-class event rather than something
    inferred after the fact -- the failure mode config.py warns about (a stale zone fails
    SILENTLY)."""
    conn.execute("INSERT OR IGNORE INTO view_epochs (source, epoch, started_at, detected_by, corr)"
                 " VALUES (?,?,?,?,?)",
                 (source, int(epoch),
                  _iso(started_at if started_at is not None else datetime.now().timestamp()),
                  detected_by, corr))
    conn.commit()


def mark_suppressed(conn, detection_id, decision: Decision, ref_id=None, when=None) -> bool:
    """SHADOW MODE WRITE: flag a detection row and record the full gate trace. Returns False for a
    non-SUPPRESS decision, so a caller can hand it every decision without branching.

    This writes ONLY the four additive columns. The row itself, its crop, its embeddings and every
    existing query are untouched -- NULL suppressed_at means "a live row", so nothing changes until
    a consumer opts in. That opt-in is deliberately not part of this change.

    Does not commit: this is meant to run inside the transaction that just INSERTed the detection,
    and the capture thread must not be handed an extra commit (this project has already lost the
    rig once to a background worker holding the WAL write lock).
    """
    if not decision.suppressed:
        return False
    conn.execute("UPDATE detections SET suppressed_at=?, suppressed_by=?, suppress_ref_id=?,"
                 " suppress_detail=? WHERE id=?",
                 (_iso(when if when is not None else datetime.now().timestamp()), SUPPRESSED_BY,
                  ref_id if ref_id is not None else decision.ref_id,
                  decision.to_json(), int(detection_id)))
    return True


# =================================================================================================
# --review : the audit loop the shadow week exists for
# =================================================================================================

# Tile width is set by the WIDEST caption line ("lum 0.00/3.87  ds 0.10/0.23  so 0.09/0.49"), not
# by the crop: a score you cannot read is not evidence, and the whole sheet exists to be read.
TILE_W, TILE_H = 208, 168
SHEET_COLS = 5
SHEET_MAX_TILES = 60
CAPTION_CHARS = 40


def read_only(db_path) -> sqlite3.Connection:
    """A read-only connection. The live rig owns backyard.db; --review never writes to it."""
    p = Path(db_path).resolve().as_posix()
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def load_suppressed(conn, days=7, source=None, now=None) -> list[dict]:
    """Every detection the veto flagged in the last `days`, newest first."""
    if not _has_suppression_columns(conn):
        return []
    now = now or datetime.now().astimezone()
    since = (now - timedelta(days=float(days))).isoformat()
    sql = ("SELECT id, timestamp, source, crop_path, confidence, species, bbox_x1, bbox_y1,"
           " bbox_x2, bbox_y2, frame_w, frame_h, suppressed_at, suppress_ref_id, suppress_detail"
           " FROM detections WHERE suppressed_by=? AND suppressed_at >= ?")
    params = [SUPPRESSED_BY, since]
    if source:
        sql += " AND source=?"
        params.append(source)
    sql += " ORDER BY timestamp DESC"
    out = []
    for r in conn.execute(sql, params):
        out.append(dict(id=r[0], timestamp=r[1], source=r[2], crop_path=r[3], confidence=r[4],
                        species=r[5], box=(r[6], r[7], r[8], r[9]), frame_w=r[10], frame_h=r[11],
                        suppressed_at=r[12], ref_id=r[13],
                        decision=Decision.from_json(r[14])))
    return out


def _has_suppression_columns(conn) -> bool:
    if not _table_exists(conn, "detections"):
        return False
    have = {r[1] for r in conn.execute("PRAGMA table_info(detections)")}
    return {"suppressed_at", "suppressed_by", "suppress_detail"} <= have


def cluster_rows(rows, iou=None, cfg=None) -> list[list[dict]]:
    """Group suppressed rows into recurrence clusters, largest first.

    Grouped on purpose: a cluster is the unit of judgement. One glance at 40 crops of the same
    watering can settles the whole cluster, where a chronological sheet makes you re-decide the
    same object 40 times.
    """
    iou = float(iou if iou is not None else (cfg or config.CONFIG).refimg_recurrence_iou)
    clusters: list[list[dict]] = []
    anchors: list[tuple] = []
    for row in sorted(rows, key=lambda r: r["timestamp"]):
        box = scale_box(row["box"], row["frame_w"] or W, row["frame_h"] or H)
        best, score = None, 0.0
        for i, a in enumerate(anchors):
            v = box_iou(box, a)
            if v > score:
                best, score = i, v
        if best is None or score < iou:
            anchors.append(box)
            clusters.append([row])
        else:
            clusters[best].append(row)
            anchors[best] = tuple(0.9 * a + 0.1 * b for a, b in zip(anchors[best], box))
    return sorted(clusters, key=len, reverse=True)


def per_day_counts(rows) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        day = str(r["timestamp"])[:10]
        counts[day] = counts.get(day, 0) + 1
    return dict(sorted(counts.items()))


def _tile(row, root):
    """One crop with its scores burnt in, so the sheet is self-contained evidence."""
    img = None
    if row.get("crop_path"):
        p = Path(row["crop_path"])
        if not p.is_absolute():
            p = Path(root) / p
        img = cv2.imread(str(p))
    if img is None:
        img = np.zeros((TILE_H, TILE_W, 3), np.uint8)
        cv2.putText(img, "crop missing", (8, TILE_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (60, 60, 220), 1)
    else:
        img = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    tile = np.zeros((TILE_H + 46, TILE_W, 3), np.uint8)
    tile[:TILE_H] = img
    d = row["decision"]
    s, t = d.scores or {}, d.thresholds or {}
    lines = [
        str(row["timestamp"])[5:19].replace("T", " ") + f"  #{row['id']}",
        " ".join(f"{m[:2]}{s.get(m, float('nan')):.3g}/{t.get(m, float('nan')):.3g}"
                 for m in METRICS),
        f"rec {d.recurrence.get('events', 0)}ev/{d.recurrence.get('days', 0)}d  "
        f"age {d.age_s if d.age_s is not None else -1:.0f}s",
    ]
    for i, line in enumerate(lines):
        cv2.putText(tile, line[:CAPTION_CHARS], (4, TILE_H + 13 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (210, 210, 210), 1)
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H + 45), (70, 70, 70), 1)
    return tile


def _banner(text, width):
    b = np.zeros((26, width, 3), np.uint8)
    b[:] = (32, 40, 52)
    cv2.putText(b, text[:130], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
    return b


def contact_sheets(clusters, out_dir, root=None, prefix="refimg-review",
                   cols=SHEET_COLS, max_tiles=SHEET_MAX_TILES) -> list[Path]:
    """Render clusters to PNG contact sheets; returns the paths written.

    A long cluster spills onto further pages under a "(continued)" banner rather than being
    truncated. Silently dropping crops would defeat the point: the shadow week is only worth
    anything if the sheet holds EVERY row the veto flagged.
    """
    root = Path(root) if root else config.ROOT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width = cols * TILE_W
    sheets: list[Path] = []
    blocks, n_tiles, page = [], 0, 1

    def flush():
        nonlocal blocks, n_tiles, page
        if not blocks:
            return
        path = out_dir / f"{prefix}-{datetime.now():%Y%m%d}-{page:02d}.png"
        cv2.imwrite(str(path), np.vstack(blocks))
        sheets.append(path)
        blocks, n_tiles, page = [], 0, page + 1

    for ci, cluster in enumerate(clusters, 1):
        label = (f"cluster {ci}  --  {len(cluster)} suppressed  "
                 f"[{cluster[0]['timestamp'][:16]} .. {cluster[-1]['timestamp'][:16]}]")
        banner = label
        for i in range(0, len(cluster), cols):
            band = cluster[i:i + cols]
            if blocks and n_tiles + len(band) > max_tiles:
                flush()
                banner = label + "  (continued)"
            if banner:
                blocks.append(_banner(banner, width))
                banner = None
            tiles = [_tile(r, root) for r in band]
            tiles += [np.zeros_like(tiles[0])] * (cols - len(tiles))
            blocks.append(np.hstack(tiles))
            n_tiles += len(band)
    flush()
    return sheets


def review(conn, days=7, out_dir=None, root=None, source=None, now=None) -> dict:
    """The audit: per-day counts + contact sheets of everything suppressed. Read-only."""
    rows = load_suppressed(conn, days=days, source=source, now=now)
    clusters = cluster_rows(rows)
    sheets = contact_sheets(clusters, out_dir or store_dir(), root=root) if rows else []
    return dict(rows=rows, clusters=clusters, sheets=sheets, per_day=per_day_counts(rows))


def main(argv=None) -> int:
    cfg = config.CONFIG
    p = argparse.ArgumentParser(
        description="Reference-image veto: audit what shadow mode flagged as furniture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--review", action="store_true",
                   help="Render the recent suppressions as contact sheets and print the counts.")
    p.add_argument("--days", type=float, default=7.0, help="How far back to look.")
    p.add_argument("--db", default=str(cfg.db_path), help="SQLite database (opened READ-ONLY).")
    p.add_argument("--source", default=None, help="Limit to one detections.source.")
    p.add_argument("--out", default=None, help="Where to write the sheets.")
    args = p.parse_args(argv)

    if not args.review:
        p.print_help()
        return 0

    conn = read_only(args.db)
    try:
        if not _has_suppression_columns(conn):
            print("This database has no suppression columns yet -- nothing has run in shadow "
                  "mode. (config.refimg_enabled is False by default.)")
            return 0
        out = review(conn, days=args.days, out_dir=args.out or store_dir(cfg), source=args.source)
    finally:
        conn.close()

    rows = out["rows"]
    if not rows:
        print(f"No detections suppressed by {SUPPRESSED_BY} in the last {args.days:g} day(s).")
        return 0
    print(f"{len(rows)} suppressed detection(s) in {len(out['clusters'])} recurrence cluster(s):")
    for day, n in out["per_day"].items():
        print(f"  {day}  {n:5d}")
    print("\n  cluster   n  species seen")
    for i, cluster in enumerate(out["clusters"], 1):
        tally: dict[str, int] = {}
        for r in cluster:
            tally[r["species"] or "-"] = tally.get(r["species"] or "-", 0) + 1
        print(f"  {i:7d} {len(cluster):3d}  " +
              ", ".join(f"{k}:{v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1])))
    print("\nOpen these and look at every crop -- that is the whole point of shadow mode:")
    for s in out["sheets"]:
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
