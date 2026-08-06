"""
ANATOMY TRAITS AND REVIEW AIDS -- what the pixels of one crop can and cannot say about WHICH
raccoon this is.

WHY THIS EXISTS. The 2026-08-05 identity evaluation (docs/identity-eval-2026-08-05.md) measured
that the MegaDescriptor appearance signal DECAYS: session-blocked leave-one-visit-out top-1 is
0.739, and with a 21-day gap between probe and template it falls to 0.222 against a 0.348
majority-class baseline. The embedding is largely encoding "this animal, THIS WEEK, IN THIS
LIGHT". Matt names raccoons off things that do not have a week's half-life -- an ear notch, the
tail, the mask, how the animal sits -- so this module was built to try to measure those directly.

WHAT THE MEASUREMENT CAME BACK WITH, 2026-08-06. Four separate hand audits, every one of them
conducted by OPENING IMAGES rather than by reading a yield number, returned NO-GO on every
identity trait proposed:

  * TAIL RINGS -- BUILT, THEN DISPROVEN, THEN DELETED (2026-08-06). The tracer produced a
    confident, fully-measured "no signal" over 20,521 crops. It was VOID: of 22 hand-audited
    traces, 20 followed ears, legs, snouts or mask-bleed onto the ground rather than tails (4.5%
    real), and the single highest-confidence trace in the entire corpus followed an EAR while a
    ringed tail sat in plain view outside the mask. An earlier 16-crop audit had already found the
    traced limb was the tail twice. The cause is anatomy plus pose: an animal hunched at the dish
    keeps its tail curled against the body, so the tail is not a protrusion of the silhouette at
    all, and a silhouette is all a classical tracer has. `tail_trait`, `_limb_paths`,
    `_measure_limb`, `_dominant_pitch`, `_geodesic`, `_sample_profile` and `_smooth_path` are
    GONE, not disabled -- a plausible-looking extractor left lying around is worse than none,
    because a future reader will trust its numbers. `__getattr__` below re-raises the audit if
    anybody imports them again.
  * TAIL THINNESS AND MASK FADE (Matt's own words for the elderly female) -- NOT BUILT. Blind
    two-pass audit, 54 crops: the by-eye "rings visible" call separated a known-distinct pair on
    one anchor night (Fisher p=0.013) and gave EXACTLY NOTHING for the same pair on three other
    nights (p=1.00, rate difference 0.00, 95% CI [-0.60, +0.60]). The call flips within one animal
    within one night, because it tracks the light, not the animal. Mask fade could not be measured
    at all: the mask band was localisable in ~20% of crops and those crops were pose-selected in a
    way that coincides with the individual, so any number would be a pose statistic wearing an
    identity label.
  * SITTING POSTURE -- NOT BUILT as identity. The posture is real and ~93% readable, but a blind
    hand-label pass over 360 crops disproved exclusivity outright: all three adults perform it
    (14/116, 4/114, 6/116), including a four-minute bout by the animal it was supposed to
    distinguish FROM. Per night the rate gives AUC 0.556 [0.368, 0.749]. And the automatic proxy
    (box aspect) lands inside the negative-control band on this project's own harness. See
    `eating_style` for the shape a behaviour read-out is allowed to take.
  * EAR NOTCH -- NOT BUILT as a detector, BUILT as a REVIEW AID. This is the one trait whose
    failure is purely a failure of RESOLUTION rather than of the idea: an ear notch is permanent,
    binary and era-invariant, which is exactly what everything else here is not. It just is not in
    the pixels. Measured ear height is ~6-7% of bbox height; a human can call a margin at ~25-30px
    of ear and call it comfortably at ~50-70px; and in Matt's own confirmed 2026-08-06 visit the
    ear margin was unreadable in 48 of 48 crops audited. So this module ships no notch classifier
    and never will at this resolution -- it ships `head_review_panel`, which puts the pixels in
    front of the human as large as they go and says how much to trust them. That is the correct
    build when the audit says "a human can see it and no detector can".
  * FACIAL LANDMARKS -- NOT BUILT. 7.4% of crops show the face with both eyes resolvable, and the
    eyes are not separable from the black mask they sit inside.

WHAT SURVIVES, AND WHAT IT IS WORTH -- RE-MEASURED 2026-08-06 THROUGH THE PROJECT'S OWN HARNESS
(evalmetrics.leave_one_visit_out, session_blocked=True, 139 confirmed solo visits, corpus built via
individuals.VisitMatcher so it is byte-for-byte eval.eval_reid's; 4000-resample bootstrap CIs):

    appearance cosine   [POSITIVE CONTROL]  0.741 [0.662, 0.813]   <- reproduces the published
    fur_ratio           (visit median)      0.309 [0.230, 0.388]      0.739 baseline
    fur_fleck           (visit median)      0.374 [0.295, 0.453]
    median box area     [NEGATIVE CONTROL]  0.345 [0.266, 0.432]
    crop count          [NEGATIVE CONTROL]  0.237 [0.173, 0.309]
    chance (majority class)                 0.345

`fur_trait` therefore DOES NOT IDENTIFY: it sits below chance and inside the nuisance band. The
positive control rules out a powerless instrument -- on the three known-distinct pairs the same
harness gives appearance 0.777 / 0.882 / 0.928, while fur_ratio gives 0.543 / 0.553 / 0.410 against
chance rates of 0.511 / 0.565 / 0.554. It is retained as a one-animal-vs-rest PRIOR and as the
worked example of a descriptor built the right way, never as a ranker. Read its docstring for the
mask confound.

AND ONE RESULT WORTH MORE THAN ANY OF THE ABOVE: on the Notch-vs-Pedro contrast, MEDIAN BOX AREA --
a statistic containing no information whatsoever about which animal it is -- scores 0.639 against a
0.554 chance rate. That is the "date predicts the label at 97.8%" problem restating itself in a
third way. Any Notch/Pedro separation from any feature measured on this corpus has to clear that
bar before it means anything.

THE THREE RULES EVERY DESCRIPTOR HERE OBEYS
  1. RATIO, ANGLE OR COUNT -- never an absolute pixel distance, never an absolute colour. This
     camera's manual white balance is broken at every colour temperature, so a WhiteBalanceWatchdog
     forces AUTO white balance; absolute colour therefore encodes the auto-WB decision, i.e. the
     session. And Matt REPOSITIONS THE CAMERAS ON PURPOSE, so absolute geometry fails silently.
     Everything below is divided by something measured on the same animal in the same crop.
     (The review aids carry the ONE deliberate exception, flagged where it appears: a resolution
     bound is measured in source pixels, because it is a fact about the sensor and the optics and
     not about the yard. It bounds what a HUMAN can see; it never becomes a descriptor.)
  2. NORMALISED BY ANATOMY, NEVER BY THE BOUNDING BOX. A descriptor divided by bbox width or bbox
     aspect would smuggle POSTURE -- i.e. behaviour -- into the appearance axis. This is not a
     stylistic preference: the posture audit measured box aspect predicting the sit posture at
     per-crop AUC 0.830, so bbox aspect IS a behaviour quantity. tests/test_traits.py asserts the
     rule by padding a crop (which changes the box, not the animal) and requiring the descriptors
     not to move.
  3. NEVER A SILENT ZERO. Every extractor returns None when the trait is not visible, and a
     confidence in 0..1 when it is. A hidden ear is not an intact ear; a missing tail must never be
     scored as a tail that looks different. That is the single most dangerous failure mode for a
     matcher, and it is the one the deleted tail tracer committed at scale.

DETERMINISM. `foreground_mask` runs GrabCut, whose GMM initialisation draws from OpenCV's global
RNG. The deleted tracer never seeded it, so its own published numbers were not reproducible. Every
call here now seeds `cv2.setRNGSeed` first (see `_seed_cv_rng`).

THE TWO-AXIS RULE, AND THE THIRD DOOR. Appearance descriptors carry `AXIS = "appearance"` and are
the only things `appearance_vector()` will accept. BEHAVIOUR (`EatingStyleFlag`) carries
`AXIS = "behaviour"`: it may be shown to the human or raise a flag, never reorder a ranking.
REVIEW AIDS (`ReviewPanel`, `FrozenSceneCluster`, `ReviewSelection`) carry `AXIS = "review"`: they
measure nothing about the animal at all -- a panel is a magnifying glass, a frozen-scene cluster is
a piece of furniture -- and they are refused by the same type guard.

NOTHING IS PERSISTED AND NOTHING IS WIRED IN. These are pure functions over a crop image and plain
metadata. The CLI is a dry-run reporter; it opens the database read-only and never writes to it.

    python traits.py --visit 1234          # per-crop read-out for one visit (dry run)
    python traits.py --sample 40           # traits for 40 random raccoon crops, with gate stats
    python traits.py --review-visit 1234   # pick the crops worth showing a human, and say why
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

import config

__all__ = [
    "MaskResult", "FurTrait", "EatingStyleFlag", "ReviewPanel", "FrozenSceneCluster",
    "ReviewSelection",
    "foreground_mask", "fur_trait", "crop_traits", "aggregate_visit", "appearance_vector",
    "eating_style", "head_review_panel", "frozen_frame_dissimilarity", "frozen_scene_clusters",
    "review_candidates",
    "APPEARANCE_TRAITS", "REMOVED_DISPROVEN",
]

# The appearance descriptors this module emits, in a fixed order. Everything here is dimensionless
# by construction (a ratio or a normalised contrast) -- see rule 1 in the module docstring.
APPEARANCE_TRAITS = (
    "fur_ratio",            # (L_p50 - L_p5) / (L_p95 - L_p5) inside the foreground mask
    "fur_fleck",            # high-frequency fur texture / the same span (CONFOUNDED BY FOCUS)
)

# Names deleted on 2026-08-06 because they were MEASURED WRONG, with the audit result attached.
# `__getattr__` turns any attempt to import one back into that audit, so the machinery cannot be
# resurrected by a hopeful `from traits import tail_trait` in six months' time.
REMOVED_DISPROVEN = {
    "TailTrait": "tail ring descriptors",
    "tail_trait": "tail ring extractor",
    "_measure_limb": "per-limb ring measurement",
    "_limb_paths": "medial-axis limb tracer",
    "_geodesic": "geodesic walk used by the limb tracer",
    "_sample_profile": "cross-limb luminance profile",
    "_smooth_path": "path smoother used by the limb tracer",
    "_dominant_pitch": "autocorrelation ring-pitch finder",
}

_REMOVAL_NOTE = (
    "removed 2026-08-06: DISPROVEN. The tail tracer returned a confident, fully-measured 'no "
    "signal' over 20,521 crops that was VOID -- of 22 hand-audited traces, 20 followed ears, legs, "
    "snouts or mask-bleed rather than tails (4.5% real), and the highest-confidence trace in the "
    "corpus followed an EAR while a ringed tail sat in plain view outside the mask. It also called "
    "cv2.grabCut with an unseeded RNG, so its own numbers were not reproducible. Do not restore "
    "it; a part segmenter (a new SHA-256-pinned weight) is the only honest replacement, and "
    "head_review_panel() is the cheap human-in-the-loop alternative that shipped instead."
)


def __getattr__(name: str):
    """PEP 562 hook: make a deleted, disproven extractor fail LOUDLY and explain itself."""
    if name in REMOVED_DISPROVEN:
        raise AttributeError(f"traits.{name} ({REMOVED_DISPROVEN[name]}) {_REMOVAL_NOTE}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _seed_cv_rng(cfg) -> None:
    """Pin OpenCV's global RNG before anything stochastic runs.

    GrabCut initialises its GMMs with k-means, which draws from `cv::theRNG()`. Without this the
    same crop segments slightly differently on every call and every downstream number becomes
    unreproducible -- the exact defect that made the deleted tail tracer's measurements worthless.
    Mutating a global is not nice, but OpenCV exposes no per-call seed, and a module whose output
    is not reproducible cannot be audited at all."""
    cv2.setRNGSeed(int(cfg.traits_rng_seed))


# ---------------------------------------------------------------------------
# Return types. Each carries AXIS, which is how the axes are kept apart.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaskResult:
    """A foreground (animal) mask for one crop, at the module's internal working scale.

    `mask` is uint8 0/255 at (`h`, `w`) -- NOT the crop's own resolution. Everything downstream
    emits ratios, so the working scale cancels; keeping it small is what makes this cheap, and
    keeping it EXPLICIT is what stops anyone reading a pixel count out of here and calling it a
    measurement."""
    AXIS = "appearance"
    mask: np.ndarray
    quality: float          # 0..1: does the mask boundary sit on a real image edge?
    fg_fraction: float      # share of the crop the animal occupies (a diagnostic, not a descriptor)


@dataclass(frozen=True)
class FurTrait:
    """Fur-shading descriptors for ONE crop, measured ONLY as within-crop ratios.

    `fur_ratio` places the median fur luminance between the animal's own near-black reference (the
    facial mask, the shadowed underside) and its own near-white reference (the muzzle, the lit
    guard hairs): 0 = the fur reads as dark as the mask, 1 = as light as the muzzle. Because both
    references are ON THE ANIMAL and IN THIS CROP, the auto-white-balance decision cancels.

    IT DOES NOT IDENTIFY, AND THE NUMBERS ARE IN THIS DOCSTRING SO NOBODY HAS TO GO LOOKING.
    Session-blocked leave-one-visit-out over 139 confirmed solo visits, 2026-08-06: fur_ratio
    0.309 [0.230, 0.388] and fur_fleck 0.374 [0.295, 0.453], against a 0.345 majority baseline and
    negative controls (median box area, crop count) at 0.345 and 0.237. It is BELOW chance and
    inside the nuisance band. The same harness, same probes, gave the appearance cosine 0.741
    [0.662, 0.813], so this is a null from a demonstrably powerful instrument, not from a broken
    one. Read it as a one-animal-vs-rest prior at VISIT level, never as a ranker.

    KNOWN CONFOUND, added by the 2026-08-06 pelage audit: this is a within-MASK statistic, and the
    mask is GrabCut. On the retaining-ledge scenes the mask routinely swallows bright stone, so
    part of the "animal's own black-to-white span" is the yard's. The audit's independently-built
    contrast statistic separated a known-distinct pair at AUC 0.646 -- against 0.642 for MEAN CROP
    LUMINANCE, a number containing zero information about the animal. Treat any fur separation at
    or under ~0.65 AUC as the segmentation talking.

    `luminance_span` is the p95-p5 distance between the two references. It is a DIAGNOSTIC -- it is
    what the gate is computed from -- and must never be used as a descriptor: it is an absolute
    contrast and therefore a property of the exposure.

    `fur_fleck` (Matt's "grey-flecked" cue) is CONFOUNDED BY FOCUS AND MOTION BLUR. It is emitted
    so it can be measured, not because it is trusted."""
    AXIS = "appearance"
    fur_ratio: float
    fur_fleck: float
    luminance_span: float
    quality: float


@dataclass(frozen=True)
class EatingStyleFlag:
    """BEHAVIOUR AXIS -- DISPLAY / PRIOR ONLY. NEVER A RANKER INPUT.

    Matt distinguishes animals by how they take food: some go mouth-first to the dish, others grab
    and sit back. That is real and useful to SHOW next to an appearance match -- "looks like X, but
    isn't eating like X" is exactly the disagreement a two-axis design exists to surface. It must
    not change the appearance ranking order.

    The structural reason is measurable, not merely a rule: this is very nearly encoded by
    BOUNDING-BOX ASPECT RATIO, which the 2026-08-06 posture audit measured predicting an upright
    sit at per-crop AUC 0.830 against 360 blind hand labels. Any appearance descriptor normalised
    by box dimensions would therefore be a behaviour feature wearing an appearance name, and the
    fusion would happen silently. This class is deliberately shaped so it cannot be mistaken for
    one -- it carries `AXIS = "behaviour"`, it has no `.value`, and `appearance_vector()` refuses it
    by type.

    IT IS ALSO NOT AN IDENTITY FEATURE, and that was tested properly rather than assumed. Per-night
    sit rates give AUC 0.556, bootstrap 95% CI [0.368, 0.749], one-sided p=0.29; the same
    quantity run through this project's own leave-one-visit-out harness scores 0.198-0.372 against
    a 0.355 chance rate and two nuisance controls at 0.339 and 0.380 -- i.e. inside the noise. And
    the posture is not exclusive to one animal: a blind pass found all three adults performing it.
    Treat this as context for the human."""
    AXIS = "behaviour"
    DISPLAY_ONLY = True
    IDENTIFIES = False
    style: str              # 'mouth_first' | 'grabs_and_sits_back' | 'unclear'
    low_posture_fraction: float
    upright_fraction: float
    posture_switches_per_min: float
    n_frames: int
    quality: float

    def __post_init__(self):
        if self.style not in ("mouth_first", "grabs_and_sits_back", "unclear"):
            raise ValueError(f"unknown eating style {self.style!r}")


@dataclass(frozen=True)
class ReviewPanel:
    """REVIEW AXIS -- A MAGNIFYING GLASS. IT MEASURES NOTHING AND DECIDES NOTHING.

    The ear-notch audit ended "a human can call this trait and no detector can, because the mark is
    1-3 px". The correct build for that verdict is not a weaker classifier; it is to put the pixels
    in front of the human at the largest honest magnification and to say how far to trust them.
    That is all this is. It carries no verdict field on purpose: there is nowhere for a machine
    opinion about the ear to go.

    `readability` is a RESOLUTION BOUND, not a promise. 0.5 at the audited marginal threshold
    (~25px of ear, reached at ~400px of bbox height on this rig), 1.0 at the comfortable one
    (~50px). It says the pixels COULD support a human call -- it does not say the ear is in frame,
    facing the camera, or in focus. Availability is genuinely low and that is the finding, not a
    defect: 0 of 48 crops in the confirmed 2026-08-06 visit reached even the marginal threshold
    with an animal in the box.

    WHY THERE IS NO SHARPNESS TERM. One was built -- normalised local-detail energy, scale-free and
    gain-invariant by construction -- and it FAILED a 7-image hand audit: a dark, motion-blurred,
    unreadable trail-cam crop scored 0.059 while three crisp crops with fully readable ear margins
    scored 0.040-0.044, because dividing by the region's own luminance span inflates noise wherever
    the span is small. It is recorded here rather than shipped, because an unaudited confidence is
    worse than no confidence."""
    AXIS = "review"
    DISPLAY_ONLY = True
    IDENTIFIES = False
    image: np.ndarray                       # BGR, upscaled, for a human to look at
    region: tuple                           # (x1, y1, x2, y2) in ORIGINAL crop pixels
    magnification: float
    feature_px: float                       # estimated ear height in ORIGINAL crop pixels
    readability: float                      # 0..1 -- see above; a bound, never a verdict
    feature: str = "ear_margin"


@dataclass(frozen=True)
class FrozenSceneCluster:
    """REVIEW AXIS -- crops that are the SAME PICTURE minutes apart: furniture, not an animal.

    Not a trait and not an animal: a rejector. Two independent audits found the same trap from
    opposite directions -- the five largest boxes in the confirmed 2026-08-06 00:07 visit contain
    no animal at all (a dark shrub, re-detected at 0.26-0.31 confidence over 17 minutes), and
    another confirmed visit's crops are dominated by a glass jar leaning on the wall. Any "show me
    the biggest crops" query surfaces these FIRST, so a review selector that does not reject them
    wastes the scarcest resource in the project, which is Matt's attention.

    IT GROUPS ON PIXELS, NOT ON BOXES, and that is a correction made during this build rather than
    a preference. Box-geometry clustering was tried twice and audited twice: plain IoU FRAGMENTED
    one shrub into sub-threshold clusters because the detector returns nested boxes on it (IoU 0.50,
    containment 1.00 on a measured pair), and loosening to containment then MERGED the shrub with
    120 genuine raccoon frames. Correlating the crops asks the question directly -- "is this the
    same picture as one taken twelve minutes ago?" -- and needs no overlap threshold at all.

    Scale-free and illumination-invariant: thumbnails are resampled to a common size, mean-removed
    and unit-normalised, so neither apparent size nor an exposure change moves the correlation. And
    there is no yard geometry anywhere in it, which matters on a rig where a camera moved 384 px
    between two cycles.

    The time span is the safety catch: a real animal can hold still for four frames, but not for
    ten minutes across five detections while its pixels stay correlated above 0.9."""
    AXIS = "review"
    DISPLAY_ONLY = True
    IDENTIFIES = False
    detection_ids: tuple
    n: int
    span_minutes: float
    dissimilarity: float                    # 1 - median pairwise correlation within the group
    confidence: float                       # 0..1 that this is furniture, not an animal


@dataclass(frozen=True)
class ReviewSelection:
    """REVIEW AXIS -- which crops of a visit are worth a human's eyes, and why the rest are not.

    `rejected` is deliberately a tally rather than a silence. "Nothing here is worth showing you"
    is a real and common answer, and the reasons are what tell Matt whether to move a camera, not
    a defect to hide.

    `demoted` is the frozen-scene pile, and it is a SEPARATE LIST RATHER THAN A DELETION because
    the audit said so. Sixteen frozen-scene rejects were opened: nine were an empty shrub, two were
    ambiguous, and FIVE WERE A REAL RACCOON LYING STILL ON THE WALL -- one of them with both ear
    margins plainly readable. A review aid that silently hides the animal is worse than one that
    shows some furniture, so these are ranked below the candidates and flagged, never dropped."""
    AXIS = "review"
    DISPLAY_ONLY = True
    IDENTIFIES = False
    candidates: tuple = ()                  # dicts: id, feature_px, readability, ...
    demoted: tuple = ()                     # same shape, plus frozen_* fields; show these last
    rejected: dict = field(default_factory=dict)
    frozen_clusters: tuple = ()
    n_input: int = 0


# ---------------------------------------------------------------------------
# Foreground mask -- the thing the fur trait stands on.
# ---------------------------------------------------------------------------

def _work_scale(crop_bgr, side: int):
    """Downscale a crop so its long edge is `side`. Cheap, and it forces every measurement below
    to be taken in a scale where an absolute pixel count is obviously meaningless."""
    h, w = crop_bgr.shape[:2]
    s = float(side) / max(h, w)
    if s >= 1.0:
        return crop_bgr.copy()
    return cv2.resize(crop_bgr, (max(8, int(round(w * s))), max(8, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def _ellipse_mask(h: int, w: int, frac: float) -> np.ndarray:
    """Boolean mask of the ellipse inscribed in `frac` of the frame, centred. Used for BOTH seeds
    below. It is the only geometry in this module, it is defined relative to the crop rather than
    to the yard, and it survives a camera move because the crop IS the detector's box."""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ry, rx = max(1.0, h * frac / 2.0), max(1.0, w * frac / 2.0)
    return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0


def _edge_agreement(gray: np.ndarray, mask: np.ndarray) -> float:
    """How much of the mask's outline sits on a real image edge, 0..1.

    Self-calibrating on purpose: the Sobel magnitude is normalised by its OWN 90th percentile in
    this crop, so a soft through-glass night frame and a crisp one are judged on the same scale.
    This is the module's answer to "is this mask the animal or a guess?" -- there is no ground
    truth to check it against, but an outline that follows no edge is certainly not a silhouette."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ref = float(np.percentile(mag, 90)) or 1.0
    pts = np.vstack([c.reshape(-1, 2) for c in contours])
    h, w = gray.shape[:2]
    # Drop outline points lying on the crop border: the animal is cut off there, so there is no
    # edge to agree with and scoring them would punish every close-range crop.
    keep = ((pts[:, 0] > 1) & (pts[:, 0] < w - 2) & (pts[:, 1] > 1) & (pts[:, 1] < h - 2))
    pts = pts[keep]
    if len(pts) < 12:
        return 0.0
    vals = mag[pts[:, 1], pts[:, 0]] / ref
    return float(np.clip(np.median(vals), 0.0, 1.0))


def foreground_mask(crop_bgr, cfg=None) -> Optional[MaskResult]:
    """Segment the animal out of one crop. Returns None when the result cannot be trusted.

    GrabCut, seeded from the crop's own shape rather than from a border rectangle. The usual
    rect-init is WRONG here: the crop is the detector's tight box, so the border is often animal,
    not background. Instead the four corners outside the inscribed ellipse are seeded as definite
    background (a tight box around a quadruped leaves its corners empty) and the central ellipse as
    definite foreground. No colour prior, no yard-specific assumption, nothing tuned to a fixed
    framing -- it re-derives itself from every crop, which is the same reason staticfilter.py
    self-calibrates instead of using hand-measured zones.

    DETERMINISTIC: OpenCV's global RNG is pinned first (see `_seed_cv_rng`), so the same crop
    always yields the same mask. It did not use to be, and that alone invalidated the tail tracer.

    KNOW ITS LIMIT BEFORE BUILDING ON IT. Hand-audited 2026-08-06 by dumping the contour over every
    crop used in a pelage study: on the lit retaining ledge the mask routinely annexes bright
    stone, and it happily returns a confident mask for crops containing no animal at all. It is
    good enough to normalise a within-crop ratio against and not good enough to be a silhouette.

    Only cv2/numpy: adding a segmentation model would mean a new pinned weight and more resident
    memory on a machine that already dies of commit exhaustion."""
    cfg = cfg or config.CONFIG
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    small = _work_scale(crop_bgr, cfg.traits_mask_side)
    h, w = small.shape[:2]
    if min(h, w) < 24:                      # below this a raccoon is a smudge; say so, don't guess
        return None
    m = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    m[~_ellipse_mask(h, w, 1.0)] = cv2.GC_BGD
    m[_ellipse_mask(h, w, cfg.traits_mask_core_fraction)] = cv2.GC_FGD
    try:
        _seed_cv_rng(cfg)
        cv2.grabCut(small, m, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), cfg.traits_mask_iterations,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    # Keep only the blob the animal is in. The centre of a detector box is the animal; a second
    # blob is a shadow, a food dish or a neighbouring paving stone.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return None
    keep = lab[h // 2, w // 2]
    if keep == 0:                            # centre landed on background -> take the biggest blob
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = np.where(lab == keep, 255, 0).astype(np.uint8)
    frac = float((fg > 0).mean())
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edge = _edge_agreement(gray, fg)
    # A plausible animal fills a lot of its own detector box but not all of it. Outside that band
    # the segmentation has collapsed to "everything" or "nothing"; fail closed rather than emit a
    # confident number computed over paving.
    lo, hi = cfg.traits_mask_min_fg_fraction, cfg.traits_mask_max_fg_fraction
    if not (lo <= frac <= hi) or edge < cfg.traits_mask_min_edge_agreement:
        return None
    band = min(1.0, (frac - lo) / max(1e-6, 0.15), (hi - frac) / max(1e-6, 0.15) + 0.5)
    quality = float(np.clip(edge * np.clip(band, 0.0, 1.0), 0.0, 1.0))
    return MaskResult(mask=fg, quality=quality, fg_fraction=frac)


# ---------------------------------------------------------------------------
# Fur shading -- a within-crop ratio against the animal's own black and white.
# ---------------------------------------------------------------------------

def fur_trait(crop_bgr, mask: Optional[MaskResult] = None, cfg=None) -> Optional[FurTrait]:
    """Where the fur sits between the animal's own darkest and lightest references. None when
    those references are not both present in this crop.

    THE GATE IS THE POINT. The trait needs a near-black (the facial mask, the shadowed underside)
    and a near-white (the muzzle, lit guard hairs) IN THE SAME CROP -- resolvable in about half of
    them. A crop that is all mid-tone rump has no anchors, and normalising against noise would
    return a number that describes the exposure. `luminance_span` is that check, and it is
    deliberately kept OUT of APPEARANCE_TRAITS: it is an absolute contrast, so it is a fact about
    the light, not about the animal.

    See FurTrait for what this is worth (0.341 vs a 0.348 baseline) and for the mask confound."""
    cfg = cfg or config.CONFIG
    mask = mask or foreground_mask(crop_bgr, cfg)
    if mask is None:
        return None
    small = _work_scale(crop_bgr, cfg.traits_mask_side)
    m8 = mask.mask if mask.mask.shape[:2] == small.shape[:2] else cv2.resize(
        mask.mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    mb = m8 > 0
    if mb.sum() < 64:
        return None
    # CIELAB L*, so "lighter" means perceptually lighter rather than "has more of whichever channel
    # the auto white balance happened to push this frame".
    lum = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)[:, :, 0].astype(np.float32) / 255.0
    vals = lum[mb]
    p5, p50, p95 = (float(np.percentile(vals, q)) for q in (5, 50, 95))
    span = p95 - p5
    if span < cfg.traits_fur_min_span:
        return None
    ratio = (p50 - p5) / span
    # Fleck: fur texture at a scale set by the ANIMAL's apparent size, not by the frame, so the
    # same animal nearer the camera is measured at the same relative scale. Eroded mask so the
    # silhouette edge (a huge high-frequency step) does not count as fur texture.
    sigma = max(0.8, cfg.traits_fleck_scale * math.sqrt(float(mb.sum())))
    hp = lum - cv2.GaussianBlur(lum, (0, 0), sigma)
    inner = cv2.erode(m8, np.ones((3, 3), np.uint8), iterations=2) > 0
    if inner.sum() < 32:
        inner = mb
    fleck = float(hp[inner].std() / span)
    quality = float(np.clip(mask.quality * min(1.0, span / (2.0 * cfg.traits_fur_min_span)), 0, 1))
    return FurTrait(fur_ratio=round(float(ratio), 4), fur_fleck=round(fleck, 4),
                    luminance_span=round(float(span), 4), quality=round(quality, 4))


# ---------------------------------------------------------------------------
# Per-crop and per-visit read-outs.
# ---------------------------------------------------------------------------

def crop_traits(crop_bgr, cfg=None) -> dict:
    """Every appearance trait for one crop: {'mask': ..., 'fur': FurTrait|None}. Shares one mask
    between the extractors -- the mask is the expensive part. A dict of Nones is a perfectly good
    answer and a common one."""
    cfg = cfg or config.CONFIG
    m = foreground_mask(crop_bgr, cfg)
    if m is None:
        return {"mask": None, "fur": None}
    return {"mask": m, "fur": fur_trait(crop_bgr, m, cfg)}


def appearance_vector(traits) -> dict:
    """Flatten crop/visit traits into {trait_name: value} for the appearance axis only.

    THE GUARD: anything whose AXIS is not "appearance" raises TypeError. That is what makes the
    two-axis rule structural rather than a comment -- an EatingStyleFlag or a ReviewPanel cannot
    reach a ranker by being appended to the wrong list, because it cannot get through this function
    at all."""
    items = traits.values() if isinstance(traits, dict) else traits
    out: dict = {}
    for t in items:
        if t is None:
            continue
        axis = getattr(t, "AXIS", None)
        if axis != "appearance":
            raise TypeError(
                f"{type(t).__name__} is on the {axis!r} axis; the appearance ranking never sees "
                "behaviour or review aids (see the axis rule in this module's docstring)")
        for f, v in asdict(t).items():
            if f in APPEARANCE_TRAITS and v is not None:
                out[f] = float(v)
    return out


def aggregate_visit(per_crop: Sequence[dict], cfg=None) -> dict:
    """Fold per-crop traits into ONE visit-level read-out.

    Visit-level is the only level these traits work at. Measured crop-level pairwise AUC for the
    fur ratio is 0.566-0.632 -- barely above a coin -- while the same statistic aggregated over a
    night's crops reaches 0.767 on the one individual pair whose labelled periods overlap.
    `coverage` reports how selected the sample was: it is the fraction of crops that cleared the
    gate, and a low one is information (the animal never turned broadside), not a defect to hide."""
    cfg = cfg or config.CONFIG
    n = len(per_crop)
    vals: dict = {k: [] for k in APPEARANCE_TRAITS}
    wts: dict = {k: [] for k in APPEARANCE_TRAITS}
    n_fur = 0
    for c in per_crop:
        t = c.get("fur")
        if t is None:
            continue
        n_fur += 1
        for f, v in asdict(t).items():
            if f in vals and v is not None:
                vals[f].append(float(v))
                wts[f].append(float(t.quality))
    out: dict = {"n_crops": n, "n_fur": n_fur, "fur_coverage": (n_fur / n) if n else 0.0,
                 "traits": {}, "n_by_trait": {}}
    for f in APPEARANCE_TRAITS:
        if len(vals[f]) >= cfg.traits_min_crops_per_visit:
            # Median, not mean: on a minority-availability trait one bad mask is a large outlier,
            # and there is no second opinion to average it away.
            out["traits"][f] = float(np.median(vals[f]))
            out["n_by_trait"][f] = len(vals[f])
    out["quality"] = float(np.mean([w for f in APPEARANCE_TRAITS for w in wts[f]])) if any(
        wts[f] for f in APPEARANCE_TRAITS) else 0.0
    return out


# ---------------------------------------------------------------------------
# BEHAVIOUR AXIS -- separate function, separate type, never a ranker input.
# ---------------------------------------------------------------------------

def eating_style(boxes: Sequence[Sequence[float]], cfg=None) -> Optional[EatingStyleFlag]:
    """DISPLAY / PRIOR ONLY. How the animal held itself over a visit, from its boxes alone.

    `boxes` is a sequence of (t_seconds, x1, y1, x2, y2), one per detection, any order. Nothing but
    the box shape is used -- no pixels, no yard geometry, no food-dish location (the cameras move;
    a dish coordinate would fail silently the first time one did).

    Read it as posture, not as identity. See EatingStyleFlag for the measurement that says so:
    per-night AUC 0.556 [0.368, 0.749], and inside the nuisance band on the project's own
    leave-one-visit-out harness. It is here so a human can see "looks like X, isn't eating like X"
    -- and so that the quantity it is built from (box aspect, measured at per-crop AUC 0.830 for
    posture) is visibly on THIS side of the wall, never normalising an appearance descriptor on the
    other side."""
    cfg = cfg or config.CONFIG
    rows = [b for b in boxes if b is not None and len(b) >= 5]
    n = len(rows)
    if n < cfg.traits_eating_min_frames:
        return None
    aspect = []
    for _, x1, y1, x2, y2 in ((r[0], r[1], r[2], r[3], r[4]) for r in rows):
        w, h = float(x2) - float(x1), float(y2) - float(y1)
        if w > 0 and h > 0:
            aspect.append(w / h)
    if len(aspect) < cfg.traits_eating_min_frames:
        return None
    a = np.asarray(aspect, np.float32)
    low = float((a >= cfg.traits_eating_low_aspect).mean())
    up = float((a <= cfg.traits_eating_upright_aspect).mean())
    ts = sorted(float(r[0]) for r in rows)
    minutes = max((ts[-1] - ts[0]) / 60.0, 1e-3)
    state = (a >= cfg.traits_eating_low_aspect).astype(np.int8)
    switches = int(np.abs(np.diff(state)).sum())
    if low - up >= cfg.traits_eating_margin:
        style = "mouth_first"
    elif up - low >= cfg.traits_eating_margin:
        style = "grabs_and_sits_back"
    else:
        style = "unclear"
    quality = float(np.clip(len(a) / (4.0 * cfg.traits_eating_min_frames), 0.0, 1.0))
    return EatingStyleFlag(style=style, low_posture_fraction=round(low, 4),
                           upright_fraction=round(up, 4),
                           posture_switches_per_min=round(switches / minutes, 3),
                           n_frames=len(a), quality=round(quality, 4))


# ---------------------------------------------------------------------------
# REVIEW AXIS -- aids for the human. They measure nothing and rank nothing.
#
# This is what the ear-notch audit's verdict actually authorises: "a human can
# call it, no detector can at this resolution". So: magnify honestly, state the
# resolution bound, and get the junk out of the way first.
# ---------------------------------------------------------------------------

def head_review_panel(crop_bgr, cfg=None, *, feature_fraction: Optional[float] = None,
                      band: Optional[float] = None) -> Optional[ReviewPanel]:
    """Blow up the top band of a crop so a human can judge the ear margins. None when the source
    resolution cannot support the call -- which is the usual answer on this rig.

    None here is load-bearing and is NOT "no notch": a hidden ear is not an intact ear (rule 3).
    The caller gets either an image and a bound, or nothing.

    THE ONE ABSOLUTE-PIXEL NUMBER IN THIS MODULE LIVES HERE, and it is deliberate. `feature_px` is
    an estimate in SOURCE pixels, because human readability is a property of the sensor and the
    optics, not of the yard: the audit measured an ear margin marginally callable at ~25-30px of
    ear and comfortably at ~50-70px, and no ratio can express that. It is a bound on what a person
    can see. It never becomes a descriptor, it never enters `appearance_vector`, and the geometry
    around it (the band, the ear's share of box height) stays a FRACTION of the crop, so a camera
    move does not invalidate it.

    The band is a fixed top share of the crop rather than a detected head, because there is no head
    detector here and inventing one would repeat the tail tracer's mistake.

    HAND-AUDITED, WHICH IS THE ONLY REASON ANY OF THIS IS CLAIMED. Twenty-four panels drawn at
    random (seed 11) from the 389 candidates the selector produces over the 171 human-confirmed
    glass-door raccoon visits were opened and scored: 21 of 24 contain an animal, and 10-11 of 24
    show an ear margin a human could actually call. The rest are backs, rumps and motion smears --
    the panel shows one, the human moves on in a second, and that is the right cost profile for a
    review aid. A separate ladder over trail-cam crops (ear 235-307px) confirmed the band contains
    both ears whenever they are raised. Compare the unfiltered version of the same query: the five
    largest boxes in Matt's confirmed 00:07 visit contain no animal at all."""
    cfg = cfg or config.CONFIG
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    h, w = crop_bgr.shape[:2]
    frac = cfg.traits_review_ear_height_fraction if feature_fraction is None else feature_fraction
    band = cfg.traits_review_head_band if band is None else band
    if not (0.0 < band <= 1.0) or not (0.0 < frac <= 1.0):
        raise ValueError("band and feature_fraction must be fractions in (0, 1]")
    feature_px = float(frac) * float(h)
    if feature_px < cfg.traits_review_min_feature_px:
        return None                          # below what a human could call: say nothing at all
    y2 = max(4, int(round(h * band)))
    region = crop_bgr[0:y2, :]
    if region.size == 0:
        return None
    # Magnify to a target on-screen size, LANCZOS (the audit's recommendation), never downscale.
    target = float(cfg.traits_review_target_px)
    mag = max(1.0, min(target / region.shape[0], float(cfg.traits_review_max_magnification)))
    out = cv2.resize(region, (max(8, int(round(region.shape[1] * mag))),
                              max(8, int(round(region.shape[0] * mag)))),
                     interpolation=cv2.INTER_LANCZOS4)
    good = float(cfg.traits_review_good_feature_px)
    readability = float(np.clip(feature_px / good, 0.0, 1.0)) if good > 0 else 0.0
    return ReviewPanel(image=out, region=(0, 0, w, y2), magnification=round(mag, 3),
                       feature_px=round(feature_px, 2), readability=round(readability, 4))


def _thumb(img_gray, side: int) -> Optional[np.ndarray]:
    """A zero-mean, unit-norm little thumbnail, so a correlation between two of them is scale-free
    and invariant to any brightness gain the auto-exposure applied between the two frames."""
    if img_gray is None or getattr(img_gray, "size", 0) == 0:
        return None
    t = cv2.resize(img_gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    t -= t.mean()
    n = float(np.linalg.norm(t))
    return (t / n) if n > 1e-6 else None


def frozen_frame_dissimilarity(crops: Sequence[np.ndarray], cfg=None) -> Optional[float]:
    """1 - median pairwise normalised cross-correlation over small grey thumbnails of `crops`.

    Near 0 = the pixels never changed, i.e. this is furniture or a repeated false detection. None
    when fewer than two usable crops were supplied -- a single crop cannot be shown not to move.

    Scale-free and illumination-invariant by construction: thumbnails are resampled to a common
    size, mean-subtracted and unit-normalised, so neither the animal's apparent size nor an
    exposure change can move the number. Measured on this corpus (2026-08-06): the animal-free
    cluster in the confirmed 00:07 visit 0.005, the glass-jar cluster 0.041; real animal runs
    sampled minutes apart 0.235-0.551, and as low as 0.054 over a few CONSECUTIVE frames -- which
    is what the span floor in `frozen_scene_clusters` exists to exclude. Across minutes the gap is
    an order of magnitude, which is why a cut in the middle is not a knife edge."""
    cfg = cfg or config.CONFIG
    side = int(cfg.traits_static_thumb_side)
    ts = []
    for c in crops:
        if c is None:
            continue
        g = c if (getattr(c, "ndim", 0) == 2) else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        t = _thumb(g, side)
        if t is not None:
            ts.append(t.ravel())
    if len(ts) < 2:
        return None
    vals = [float(ts[i] @ ts[j]) for i in range(len(ts)) for j in range(i + 1, len(ts))]
    return float(1.0 - float(np.median(vals)))


def _seconds(ts) -> Optional[float]:
    """Parse a timestamp to epoch seconds, OFFSET-AWARE.

    The rig stores ISO-8601 WITH a UTC offset, and SQLite's strftime('%H', ts) silently converts it
    -- a bug that has already produced wrong hour-of-night numbers in this project. Parsing here
    with datetime.fromisoformat keeps the offset, and every use below is a DIFFERENCE of two
    timestamps, which is offset-safe by construction."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        return ts.timestamp()
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except (ValueError, TypeError):
        return None


def _row(d) -> dict:
    """Accept a sqlite3.Row, a dict or a plain mapping; take only what this module needs."""
    get = d.get if hasattr(d, "get") else (
        lambda k, default=None: d[k] if k in d.keys() else default)
    return {
        "id": get("id"),
        "t": _seconds(get("timestamp")),
        "box": (float(get("bbox_x1", 0) or 0), float(get("bbox_y1", 0) or 0),
                float(get("bbox_x2", 0) or 0), float(get("bbox_y2", 0) or 0)),
        "frame_w": float(get("frame_w", 0) or 0),
        "frame_h": float(get("frame_h", 0) or 0),
        "crop_path": get("crop_path"),
        "confidence": get("confidence"),
    }


def frozen_scene_clusters(detections: Sequence, load_crop: Callable, cfg=None) -> list:
    """Group detections whose CROPS are the same picture and whose group spans a long time.

    `detections` is any sequence of mappings carrying id and timestamp (a sqlite3.Row works);
    `load_crop(detection_id) -> image` is REQUIRED, because the pixels are the whole method. Greedy
    single pass against each open group's first thumbnail, which is enough for a thing that is not
    moving; anything that moves fails the correlation on its next frame and opens its own group.

    Two rejected designs are recorded here so nobody re-tries them: grouping by box IoU fragmented
    one shrub into sub-threshold pieces (the detector emits nested boxes on it -- a measured pair
    scores IoU 0.50 and containment 1.00), and grouping by containment then swallowed 120 genuine
    raccoon frames. Both were caught by opening the crops. Correlating the images asks the question
    directly and needs no overlap constant at all.

    Nothing is calibrated to the yard: no zone, no coordinate, no box threshold. Only "does this
    look like that", which survives a camera move by construction."""
    cfg = cfg or config.CONFIG
    rows = [_row(d) for d in detections]
    rows = [r for r in rows if r["t"] is not None]
    rows.sort(key=lambda r: (r["t"], str(r["id"])))
    side = int(cfg.traits_static_thumb_side)
    cut = 1.0 - float(cfg.traits_static_max_dissimilarity)      # correlation, not distance
    groups: list = []
    for r in rows:
        try:
            img = load_crop(r["id"])
        except Exception:            # a missing crop file is not a reason to fail the whole sweep
            img = None
        if img is None:
            continue
        g = img if getattr(img, "ndim", 0) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t = _thumb(g, side)
        if t is None:
            continue
        v = t.ravel()
        for grp in groups:
            if float(grp["ref"] @ v) >= cut:
                grp["rows"].append(r)
                grp["vecs"].append(v)
                # The reference is the group's RUNNING MEAN, re-normalised -- not its first frame.
                # Measured: against a first-frame reference the same shrub split into a group of 23
                # and a stray group of 3, and the group of 3 fell under the member floor and came
                # out at the top of a review sheet (opened, confirmed empty). A mean reference is
                # also the less noisy of the two by construction: sensor noise averages out of it,
                # and sensor noise is exactly what a frozen scene's frames differ by.
                m = np.mean(grp["vecs"], axis=0)
                nm = float(np.linalg.norm(m))
                grp["ref"] = m / nm if nm > 1e-6 else grp["ref"]
                break
        else:
            groups.append({"ref": v, "rows": [r], "vecs": [v]})
    total_span_min = ((rows[-1]["t"] - rows[0]["t"]) / 60.0) if len(rows) > 1 else 0.0
    out = []
    for grp in groups:
        rs = grp["rows"]
        span_min = (rs[-1]["t"] - rs[0]["t"]) / 60.0
        # TWO span tests, and the FRACTION is the important one. An absolute "10 minutes" was tried
        # first and it silently exempted short visits: a 7-minute visit whose every large crop was
        # the same empty shrub could not, by arithmetic, contain a 10-minute group, so all four of
        # its review candidates were furniture (opened, confirmed empty). Furniture is present for
        # the WHOLE visit by definition, so the honest test is a share of the visit's own span --
        # self-calibrating, no clock constant. The small absolute floor stays underneath it, to
        # stop a 20-second visit of a briefly-still animal reading as furniture: real animal runs
        # decorrelate at 0.235-0.551 across minutes but were measured as low as 0.054 across a few
        # consecutive frames, and that is the case the floor exists to exclude.
        if len(rs) < cfg.traits_static_min_detections or \
                span_min < cfg.traits_static_min_span_minutes or \
                (total_span_min > 0 and
                 span_min < cfg.traits_static_min_span_fraction * total_span_min):
            continue
        vecs = grp["vecs"]
        pairs = [float(vecs[i] @ vecs[j]) for i in range(len(vecs))
                 for j in range(i + 1, len(vecs))]
        dis = 1.0 - float(np.median(pairs)) if pairs else 0.0
        conf = float(np.clip(1.0 - 0.5 * dis / max(1e-6, cfg.traits_static_max_dissimilarity),
                             0.5, 1.0))
        out.append(FrozenSceneCluster(
            detection_ids=tuple(r["id"] for r in rs), n=len(rs),
            span_minutes=round(span_min, 2), dissimilarity=round(dis, 4),
            confidence=round(conf, 4)))
    out.sort(key=lambda c: (-c.n, -c.confidence))
    return out


def review_candidates(detections: Sequence, cfg=None, limit: int = 8,
                      load_crop: Optional[Callable] = None) -> ReviewSelection:
    """Pick the few crops of a visit actually worth a human's eyes, and tally why the rest are not.

    The bottleneck this addresses is measured, not assumed. "Show me the biggest crops" is the
    obvious query and it is a TRAP on this corpus, twice over:
      * the five largest boxes in Matt's confirmed 2026-08-06 visit contain NO ANIMAL -- a dark
        shrub re-detected at 0.26-0.31 confidence over 17 minutes (opened and confirmed by hand);
      * the ten largest crops carrying a human name in the whole glass-door corpus are a dark
        curtain, full-frame false detections stamped by visit-level labelling. Ordering the June
        corpus by box height returns full-frame boxes first, one of which is empty and one of which
        holds a raccoon occupying a tenth of it (opened and confirmed by hand).
    So: reject near-full-frame boxes as a FRACTION of the frame (scale-free, survives a camera
    move), DEMOTE frozen scenes on the pixels, then rank what is left by how many pixels the human
    gets. Demote rather than delete: 5 of 16 hand-opened frozen-scene rejects were a real raccoon
    lying still, one with both ear margins readable.

    MEASURED ON THE 171 HUMAN-CONFIRMED GLASS-DOOR RACCOON VISITS: 27,277 detections in, 389
    candidates out (1.4%), at least one candidate on 90 of 171 visits (53%). A blind sample of 24
    candidates was opened: 21 contain an animal, and 10-11 show an ear margin a human could call.
    Compare the unfiltered version of the same query -- "the biggest crops" -- where the top five
    of the confirmed 00:07 visit contain no animal at all.

    ORDER MATTERS AND IS DELIBERATE. The frozen-scene check runs LAST, over only the crops that
    already cleared the size gates -- typically a few dozen out of a few hundred. That keeps the
    image I/O proportional to what is about to be shown, and it is why `load_crop` is worth
    supplying. Without it the frozen-scene reject cannot run at all and the tally says so
    (`frozen_check_skipped`); the geometric shortcut that would have let it run without pixels was
    built, audited and thrown away (see `frozen_scene_clusters`).

    ANTI-PSEUDO-REPLICATION, and this is not decoration: a previous study's six "independent"
    crops turned out to be four near-duplicate frames from a seven-second window, which inflated
    its effect. Candidates are spread by dividing the visit's own span into `limit` equal buckets
    and taking the best from each -- self-calibrating, no clock constant to get wrong.

    Pure: no DB, no writes."""
    cfg = cfg or config.CONFIG
    rows = [_row(d) for d in detections]
    n_input = len(rows)
    rejected: dict = {}

    def drop(tag, n=1):
        rejected[tag] = rejected.get(tag, 0) + n

    keep = []
    for r in rows:
        bw, bh = r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]
        if bh <= 0 or bw <= 0:
            drop("degenerate_box")
            continue
        if r["frame_w"] > 0 and r["frame_h"] > 0:
            area_frac = (bw * bh) / (r["frame_w"] * r["frame_h"])
            if area_frac >= cfg.traits_review_max_frame_fraction:
                # A box this big is almost always the detector giving up on a dark frame, not an
                # animal at the glass -- and its height, which is the readability estimate's whole
                # input, is then the FRAME's height and says nothing about the animal.
                drop("near_full_frame")
                continue
        feature_px = cfg.traits_review_ear_height_fraction * bh
        if feature_px < cfg.traits_review_min_feature_px:
            drop("too_small_to_read")
            continue
        keep.append({"id": r["id"], "t": r["t"], "bbox_height": round(bh, 1),
                     "feature_px": round(feature_px, 2),
                     "readability": round(float(np.clip(
                         feature_px / max(1e-6, cfg.traits_review_good_feature_px), 0.0, 1.0)), 4),
                     "crop_path": r["crop_path"], "confidence": r["confidence"]})

    # Frozen-scene DEMOTION, on the survivors only. Not a deletion -- see ReviewSelection.
    clusters: list = []
    demoted: list = []
    if load_crop is None:
        if keep:
            drop("frozen_check_skipped", 0)     # register the key so the caller sees it was not run
    else:
        clusters = frozen_scene_clusters(
            [{"id": k["id"], "timestamp": k["t"]} for k in keep], load_crop, cfg)
        frozen = {i: c for c in clusters for i in c.detection_ids}
        if frozen:
            drop("frozen_scene_demoted", len(frozen))
            for k in keep:
                if k["id"] in frozen:
                    c = frozen[k["id"]]
                    demoted.append({**k, "frozen_dissimilarity": c.dissimilarity,
                                    "frozen_group_n": c.n,
                                    "frozen_confidence": c.confidence})
            keep = [k for k in keep if k["id"] not in frozen]
    demoted.sort(key=lambda k: (-k["feature_px"], str(k["id"])))
    if not keep:
        return ReviewSelection(candidates=(), demoted=tuple(demoted[:limit]), rejected=rejected,
                               frozen_clusters=tuple(clusters), n_input=n_input)

    ts = [k["t"] for k in keep if k["t"] is not None]
    if ts and max(ts) > min(ts) and limit > 1:
        lo, hi = min(ts), max(ts)
        width = (hi - lo) / limit
        buckets: dict = {}
        for k in keep:
            b = (limit - 1 if k["t"] is None
                 else min(limit - 1, int((k["t"] - lo) / max(width, 1e-9))))
            cur = buckets.get(b)
            # Deterministic tie-break on id, so two crops of identical size cannot make the
            # selection depend on row order.
            if cur is None or (k["feature_px"], str(k["id"])) > (cur["feature_px"], str(cur["id"])):
                buckets[b] = k
        picked = [buckets[b] for b in sorted(buckets)]
    else:
        picked = list(keep)
    # DELIBERATELY NOT TOPPED UP to `limit`. An earlier version filled the remaining slots with the
    # next-biggest crops and promptly returned four frames from a five-second window -- the same
    # pseudo-replication that inflated a previous study's effect size (4 of its 6 "independent"
    # crops came from a seven-second window). Fewer, spread-out candidates is the honest answer;
    # "the animal was only readable at two moments" is information.
    picked.sort(key=lambda k: (-k["feature_px"], str(k["id"])))
    return ReviewSelection(candidates=tuple(picked[:limit]), demoted=tuple(demoted[:limit]),
                           rejected=rejected, frozen_clusters=tuple(clusters), n_input=n_input)


# ---------------------------------------------------------------------------
# CLI -- a dry-run reporter. It never writes to the database.
# ---------------------------------------------------------------------------

def _ro_conn(path) -> sqlite3.Connection:
    """Read-ONLY connection. The live rig may be writing to this file right now, and nothing in
    this module has any business changing it."""
    c = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def _crop_abspath(rel: str) -> Path:
    return Path(__file__).resolve().parent / str(rel).replace("\\", "/")


def _report(rows, cfg) -> int:
    n_mask = n_fur = n_panel = 0
    per_crop = []
    for r in rows:
        img = cv2.imread(str(_crop_abspath(r["crop_path"])))
        if img is None:
            continue
        t = crop_traits(img, cfg)
        per_crop.append(t)
        n_mask += t["mask"] is not None
        n_fur += t["fur"] is not None
        panel = head_review_panel(img, cfg)
        n_panel += panel is not None
        fur = t["fur"]
        print(f"  {Path(r['crop_path']).name:<48} "
              f"mask {'y' if t['mask'] else '-'}  "
              + (f"fur {fur.fur_ratio:5.3f} fleck {fur.fur_fleck:5.3f}  " if fur
                 else f"{'fur --':<24}")
              + (f"ear panel {panel.feature_px:5.1f}px r={panel.readability:.2f}"
                 if panel else "ear panel -- (below the readability floor)"))
    n = len(per_crop)
    if not n:
        print("No readable crops.")
        return 1
    print(f"\n{n} crop(s): mask {n_mask} ({n_mask/n:.0%}), fur {n_fur} ({n_fur/n:.0%}), "
          f"ear panel {n_panel} ({n_panel/n:.0%})")
    agg = aggregate_visit(per_crop, cfg)
    print("visit aggregate:", {k: round(v, 4) for k, v in agg["traits"].items()} or "(nothing "
          "cleared the gate -- that is a valid answer, not a failure)")
    print("NOTE: fur_ratio scores 0.309 [0.230, 0.388] session-blocked LOO top-1 against a 0.345 "
          "majority baseline, with negative controls at 0.345 and 0.237. It does not identify "
          "anybody.")
    return 0


def _review_report(rows, cfg, limit: int) -> int:
    def load(det_id):
        r = next((x for x in rows if x["id"] == det_id), None)
        return None if r is None else cv2.imread(str(_crop_abspath(r["crop_path"])))

    sel = review_candidates(rows, cfg, limit=limit, load_crop=load)
    print(f"{sel.n_input} detection(s) in, {len(sel.candidates)} worth a look.")
    if sel.frozen_clusters:
        print("frozen-scene groups (same picture minutes apart -- probably furniture):")
        for c in sel.frozen_clusters:
            print(f"  n={c.n:<4} span={c.span_minutes:>6.1f} min  dissim={c.dissimilarity:.4f}  "
                  f"confidence={c.confidence}  first ids {list(c.detection_ids[:4])}")
    if sel.rejected:
        print("rejected:", dict(sorted(sel.rejected.items(), key=lambda kv: -kv[1])))
    for k in sel.candidates:
        print(f"  det {k['id']:<10} bbox_h {k['bbox_height']:>7.1f}  "
              f"ear~{k['feature_px']:>5.1f}px  "
              f"readability {k['readability']:.2f}  {k['crop_path']}")
    for k in sel.demoted:
        print(f"  [static?] det {k['id']:<10} bbox_h {k['bbox_height']:>7.1f}  "
              f"ear~{k['feature_px']:>5.1f}px  readability {k['readability']:.2f}  "
              f"dissim {k['frozen_dissimilarity']:.4f}  {k['crop_path']}")
    if not sel.candidates and not sel.demoted:
        print("Nothing in this visit reaches the ear-margin readability floor. That is the usual "
              "answer on this rig and it is a fact about the optics, not a failure.")
    elif not sel.candidates:
        print("Everything large enough in this visit looks like a frozen scene -- shown above, "
              "flagged. Five of sixteen such crops audited were a real animal lying still, so "
              "they are demoted, never hidden.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Anatomy traits (fur shading) and review aids for raccoon crops. DRY RUN ONLY "
                    "-- opens the database read-only and never writes to it.")
    p.add_argument("--db", default=str(config.CONFIG.db_path))
    p.add_argument("--visit", type=int, default=None, help="Report every crop of one visit.")
    p.add_argument("--sample", type=int, default=0, help="Report N random crops of the species.")
    p.add_argument("--review-visit", type=int, default=None,
                   help="Pick the crops of one visit worth showing a human, and say why not.")
    p.add_argument("--limit", type=int, default=8, help="How many review candidates to pick.")
    p.add_argument("--species", default=None, help="Species to sample (default: config).")
    args = p.parse_args()
    cfg = config.CONFIG
    species = args.species or cfg.reid_species
    conn = _ro_conn(args.db)
    try:
        if args.review_visit is not None:
            rows = conn.execute(
                "SELECT id, timestamp, crop_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w,"
                " frame_h, confidence FROM detections WHERE visit_id = ? ORDER BY timestamp",
                (args.review_visit,)).fetchall()
            print(f"Visit #{args.review_visit}: {len(rows)} detection(s)")
            return _review_report(rows, cfg, args.limit)
        if args.visit is not None:
            rows = conn.execute(
                "SELECT crop_path FROM detections WHERE visit_id = ? ORDER BY timestamp",
                (args.visit,)).fetchall()
            print(f"Visit #{args.visit}: {len(rows)} crop(s)")
        elif args.sample:
            rows = conn.execute(
                "SELECT crop_path FROM detections WHERE species = ? ORDER BY RANDOM() LIMIT ?",
                (species, args.sample)).fetchall()
            print(f"{len(rows)} random {species} crop(s)")
        else:
            p.error("pass --visit N, --review-visit N or --sample N")
        return _report(rows, cfg)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
