"""
Tests for the anatomy-trait extractor and the review aids (traits.py).

The traits exist because the appearance embedding was measured to encode "this animal, this week,
in this light" rather than "this animal". So the tests here are mostly INVARIANCE tests: they take
one synthetic animal and re-photograph it wrongly -- bigger, dimmer, colour-cast, in a differently
shaped box -- and demand the descriptors not move. In rough order of how much a failure would cost:

  * BOX-SHAPE INVARIANCE is the two-axis rule made executable. Posture here is very nearly encoded
    by bounding-box aspect ratio (measured at per-crop AUC 0.830 against 360 blind hand labels), so
    an appearance descriptor that moved when the box was padded would be a behaviour feature
    wearing an appearance name -- and the fusion would be silent. Padding changes the box and not
    the animal; the descriptors must not notice.
  * AXIS SEPARATION: appearance_vector() must refuse an EatingStyleFlag AND a ReviewPanel outright,
    and a ranking built from appearance must be identical for two animals whose behaviour differs.
  * THE DISPROVEN TRACER MUST STAY DEAD. The tail extractor was deleted on 2026-08-06 after a
    22-image hand audit found 20 of 22 traces followed ears, legs or snouts. A test pins the
    removal, because the failure mode of a resurrected extractor is a confident wrong number.
  * NONE IS NOT ZERO: an unreadable ear must come back as None, never as an intact ear. A silent
    zero is the one failure that quietly poisons a matcher.
  * DETERMINISM: GrabCut draws from OpenCV's global RNG. The deleted tracer never seeded it, so its
    own published numbers were unreproducible. Two identical calls must now agree exactly.
  * ILLUMINATION INVARIANCE: the camera's white balance is on AUTO, so a per-channel gain is a
    thing that happens between two frames of the same animal.
  * SCALE INVARIANCE: the cameras move. Nothing may be an absolute pixel measurement -- except the
    review panel's readability bound, which is a fact about the sensor and is tested as such.

Synthetic pixels only -- no database, no crops on disk, no model. Where a test is about the
MEASUREMENT rather than the segmentation it hands in a known-good mask, so a GrabCut wobble can
never turn into a red test about something else.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

import config
import traits


# ---------------------------------------------------------------------------
# A synthetic raccoon: pale body, dark facial mask, white muzzle, banded tail.
# ---------------------------------------------------------------------------

def _animal(scale: float = 1.0, *, fur: int = 150, n_rings: int = 5, tail: bool = True,
            seed: int = 7):
    """(crop_bgr, MaskResult) for a fake raccoon. `fur` sets the body grey, so a lighter animal and
    a darker one can be compared against the SAME black mask and white muzzle references."""
    rng = np.random.default_rng(seed)
    W, H = int(420 * scale), int(240 * scale)
    s = lambda v: int(round(v * scale))                                    # noqa: E731
    img = np.full((H, W, 3), (70, 62, 55), np.uint8)                       # cool background
    mask = np.zeros((H, W), np.uint8)
    body_c, body_ax = (s(130), s(120)), (s(105), s(78))
    for canvas, colour in ((img, (fur - 12, fur, fur + 8)), (mask, 255)):
        cv2.ellipse(canvas, body_c, body_ax, 0, 0, 360, colour, -1)
    if tail:
        y0, y1 = s(96), s(148)
        x0, x1 = s(215), s(400)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
        pitch = (x1 - x0) / float(n_rings)
        for i in range(n_rings):                                           # alternating rings
            a, b = int(x0 + i * pitch), int(x0 + (i + 0.5) * pitch)
            cv2.rectangle(img, (a, y0), (b, y1), (fur + 55, fur + 62, fur + 68), -1)
            cv2.rectangle(img, (b, y0), (int(x0 + (i + 1) * pitch), y1),
                          (fur - 62, fur - 58, fur - 52), -1)
    cv2.ellipse(img, (s(58), s(104)), (s(46), s(20)), 0, 0, 360, (18, 20, 22), -1)   # mask band
    cv2.ellipse(img, (s(38), s(146)), (s(22), s(13)), 0, 0, 360, (238, 240, 242), -1)  # muzzle
    img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img, traits.MaskResult(mask=mask, quality=1.0, fg_fraction=float((mask > 0).mean()))


def _pad(img, mask, right: int, bottom: int):
    """Re-frame the same animal in a taller box -- what a different detector box looks like."""
    img2 = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(70, 62, 55))
    mask2 = cv2.copyMakeBorder(mask.mask, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=0)
    return img2, traits.MaskResult(mask=mask2, quality=mask.quality,
                                   fg_fraction=float((mask2 > 0).mean()))


def _det(i, t, box, frame=(1920, 1080), path=None):
    """One detection row in the shape traits.py accepts (a dict; sqlite3.Row also works).
    `t` is an OFFSET-AWARE ISO timestamp, the form the rig actually stores."""
    return {"id": i, "timestamp": t, "bbox_x1": box[0], "bbox_y1": box[1], "bbox_x2": box[2],
            "bbox_y2": box[3], "frame_w": frame[0], "frame_h": frame[1],
            "crop_path": path or f"crops/{i}.jpg", "confidence": 0.5}


CFG = config.CONFIG


# ---------------------------------------------------------------------------
# The disproven tracer must stay dead.
# ---------------------------------------------------------------------------

def test_the_disproven_tail_tracer_is_gone_and_says_why():
    """Deleted 2026-08-06: 20 of 22 hand-audited traces followed ears, legs, snouts or mask-bleed
    rather than tails. Importing it again must fail LOUDLY with the audit attached, not quietly
    return a plausible number."""
    for name in ("tail_trait", "TailTrait", "_limb_paths", "_measure_limb", "_dominant_pitch",
                 "_geodesic", "_sample_profile", "_smooth_path"):
        assert name in traits.REMOVED_DISPROVEN
        assert not hasattr(traits, name)
        with pytest.raises(AttributeError) as e:
            getattr(traits, name)
        assert "DISPROVEN" in str(e.value)
        assert "22 hand-audited" in str(e.value)
    assert not any("ring" in t or "tail" in t for t in traits.APPEARANCE_TRAITS), \
        "no ring/tail descriptor may survive the deletion"


def test_no_config_knob_survives_the_deleted_tracer():
    """A dead knob is a trap: the next reader sets it True and nothing happens."""
    assert not [f for f in vars(CFG) if f.startswith("traits_tail")]
    assert not hasattr(CFG, "traits_trace_side")


# ---------------------------------------------------------------------------
# The axis rule, made executable. Three axes now, one door.
# ---------------------------------------------------------------------------

def test_appearance_vector_refuses_a_behaviour_flag():
    """The wall between the axes is a TYPE, not a comment: behaviour cannot reach a ranker."""
    flag = traits.EatingStyleFlag(style="mouth_first", low_posture_fraction=0.9,
                                  upright_fraction=0.0, posture_switches_per_min=1.0,
                                  n_frames=40, quality=1.0)
    with pytest.raises(TypeError):
        traits.appearance_vector([flag])
    img, m = _animal()
    fur = traits.fur_trait(img, m, CFG)
    with pytest.raises(TypeError):            # and it cannot ride along beside a real trait
        traits.appearance_vector([fur, flag])


def test_appearance_vector_refuses_a_review_aid():
    """A review panel is a magnifying glass. It measures nothing, so it must not be able to reach
    a ranker either -- the same guard, extended to the third axis added 2026-08-06."""
    crop = np.zeros((600, 400, 3), np.uint8)
    cv2.circle(crop, (200, 120), 90, (200, 200, 200), -1)
    panel = traits.head_review_panel(crop, CFG)
    assert panel is not None and panel.AXIS == "review"
    with pytest.raises(TypeError):
        traits.appearance_vector([panel])
    cluster = traits.FrozenSceneCluster(detection_ids=(1, 2), n=2, span_minutes=30.0,
                                        dissimilarity=0.01, confidence=0.95)
    with pytest.raises(TypeError):
        traits.appearance_vector([cluster])
    with pytest.raises(TypeError):
        traits.appearance_vector([traits.ReviewSelection()])


def test_ranking_is_invariant_to_behaviour():
    """Two visits identical in appearance but opposite in eating style must rank identically.

    The eval's stated principle: behaviour may raise a flag or be displayed, never reorder the
    appearance candidates. Here the same animal is given a mouth-first and a sits-back posture
    record and the ranking is recomputed; the order and the distances must be untouched."""
    img, m = _animal()
    probe = traits.appearance_vector(traits.crop_traits(img, CFG))
    others = [traits.appearance_vector(traits.crop_traits(_animal(fur=f)[0], CFG))
              for f in (110, 150, 190)]

    def rank(p, pool):
        keys = sorted(set(p) & set.intersection(*(set(o) for o in pool)))
        return sorted(range(len(pool)),
                      key=lambda i: sum((p[k] - pool[i][k]) ** 2 for k in keys))

    head_down = [(t, 0.0, 0.0, 100.0, 50.0) for t in range(40)]      # aspect 2.0 -> mouth-first
    sat_back = [(t, 0.0, 0.0, 50.0, 100.0) for t in range(40)]       # aspect 0.5 -> sits back
    a = traits.eating_style(head_down, CFG)
    b = traits.eating_style(sat_back, CFG)
    assert a.style == "mouth_first" and b.style == "grabs_and_sits_back"
    assert rank(probe, others) == rank(probe, others)                # deterministic to begin with
    # There is deliberately NO way to pass a or b into the ranking: the only route from traits to a
    # ranker is appearance_vector, and it raises on them. That is the invariance.
    for flag in (a, b):
        with pytest.raises(TypeError):
            traits.appearance_vector([flag])


def test_ranking_is_invariant_to_review_aids():
    """Same guarantee, extended: a review panel and a frozen-scene verdict cannot reorder anything.

    The panel is deliberately built from the crop the ranking also sees, so this is not vacuous --
    if a future edit let a review object through appearance_vector, the ranking WOULD move."""
    img, m = _animal()
    probe = traits.appearance_vector(traits.crop_traits(img, CFG))
    pool = [traits.appearance_vector(traits.crop_traits(_animal(fur=f)[0], CFG))
            for f in (110, 150, 190)]

    def rank(p):
        keys = sorted(set(p) & set.intersection(*(set(o) for o in pool)))
        return sorted(range(len(pool)), key=lambda i: sum((p[k] - pool[i][k]) ** 2 for k in keys))

    before = rank(probe)
    tall = cv2.resize(img, (img.shape[1], 800), interpolation=cv2.INTER_LANCZOS4)
    panel = traits.head_review_panel(tall, CFG)
    assert panel is not None
    with pytest.raises(TypeError):
        traits.appearance_vector({"fur": traits.fur_trait(img, m, CFG), "panel": panel})
    assert rank(probe) == before


def test_no_appearance_descriptor_is_a_function_of_box_shape():
    """Pad the crop: the BOX changes shape, the ANIMAL does not. Descriptors must not move.

    This is the structural guard against smuggling posture into the appearance axis. Everything is
    normalised by anatomy -- the animal's own black-to-white luminance span -- so re-framing is a
    no-op up to resampling noise."""
    img, m = _animal()
    before = traits.appearance_vector({"fur": traits.fur_trait(img, m, CFG)})
    assert before, "the synthetic animal should yield some traits to compare"
    img2, m2 = _pad(img, m, right=140, bottom=280)
    aspect_before = img.shape[1] / img.shape[0]
    aspect_after = img2.shape[1] / img2.shape[0]
    assert abs(aspect_after - aspect_before) > 0.5, "the box really did change shape"
    after = traits.appearance_vector({"fur": traits.fur_trait(img2, m2, CFG)})
    for k, v in before.items():
        assert k in after, f"{k} vanished when the box was re-framed"
        assert after[k] == pytest.approx(v, rel=0.20, abs=0.05), f"{k} moved with the box shape"


def test_every_axis_declares_itself():
    assert traits.EatingStyleFlag.AXIS == "behaviour"
    assert traits.EatingStyleFlag.DISPLAY_ONLY is True
    assert traits.EatingStyleFlag.IDENTIFIES is False
    assert traits.FurTrait.AXIS == "appearance"
    assert traits.MaskResult.AXIS == "appearance"
    for t in (traits.ReviewPanel, traits.FrozenSceneCluster, traits.ReviewSelection):
        assert t.AXIS == "review"
        assert t.DISPLAY_ONLY is True
        assert t.IDENTIFIES is False


def test_a_review_panel_carries_no_verdict():
    """There must be nowhere for a machine opinion about the ear to live. The audit's finding was
    that no detector works at this resolution; a `notch=True/False` field would quietly invite one
    back in."""
    crop = np.zeros((600, 400, 3), np.uint8)
    panel = traits.head_review_panel(crop, CFG)
    fields = set(vars(panel))
    assert not fields & {"notch", "has_notch", "verdict", "score", "label", "individual"}


# ---------------------------------------------------------------------------
# Determinism. The deleted tracer's numbers were unreproducible; ours must not be.
# ---------------------------------------------------------------------------

def test_segmentation_is_deterministic():
    """GrabCut seeds its GMMs with k-means off OpenCV's GLOBAL RNG. Unseeded, the same crop
    segments differently between calls and every number downstream drifts."""
    img, _ = _animal()
    a = traits.foreground_mask(img, CFG)
    other, _ = _animal(fur=90, seed=3)
    traits.foreground_mask(other, CFG)         # perturb the global RNG in between
    b = traits.foreground_mask(img, CFG)
    assert a is not None and b is not None
    assert np.array_equal(a.mask, b.mask)
    assert a.quality == b.quality
    fa = traits.fur_trait(img, None, CFG)
    fb = traits.fur_trait(img, None, CFG)
    assert fa == fb


def test_review_selection_is_deterministic_and_order_independent():
    """Same detections in a different row order must give the same answer -- otherwise the
    selection depends on however the caller's SQL happened to sort."""
    dets = [_det(i, f"2026-08-06T00:{i:02d}:00-07:00", (0, 0, 300, 500 + (i % 3)))
            for i in range(10)]
    a = traits.review_candidates(dets, CFG, limit=4)
    b = traits.review_candidates(list(reversed(dets)), CFG, limit=4)
    assert [c["id"] for c in a.candidates] == [c["id"] for c in b.candidates]


# ---------------------------------------------------------------------------
# None is not zero.
# ---------------------------------------------------------------------------

def test_an_unreadable_ear_is_none_not_an_intact_ear():
    """The whole point of the review panel. A crop too small for a human to call the margin gets
    NOTHING back -- not a panel with a low score, and certainly not "no notch found"."""
    small = np.zeros((int(CFG.traits_review_min_feature_px /
                         CFG.traits_review_ear_height_fraction) - 40, 200, 3), np.uint8)
    assert traits.head_review_panel(small, CFG) is None
    big = np.zeros((int(CFG.traits_review_min_feature_px /
                        CFG.traits_review_ear_height_fraction) + 40, 200, 3), np.uint8)
    assert traits.head_review_panel(big, CFG) is not None
    assert traits.head_review_panel(None, CFG) is None
    assert traits.head_review_panel(np.zeros((0, 0, 3), np.uint8), CFG) is None


def test_a_flat_crop_has_no_fur_trait():
    """No black reference and no white reference in the crop = no ratio. Not 0.5, not 0.0."""
    flat = np.full((200, 300, 3), 128, np.uint8)
    mask = np.zeros((200, 300), np.uint8)
    cv2.ellipse(mask, (150, 100), (110, 70), 0, 0, 360, 255, -1)
    m = traits.MaskResult(mask=mask, quality=1.0, fg_fraction=0.4)
    assert traits.fur_trait(flat, m, CFG) is None


def test_foreground_mask_declines_a_uniform_crop():
    assert traits.foreground_mask(np.full((160, 160, 3), 90, np.uint8), CFG) is None
    assert traits.foreground_mask(None, CFG) is None
    assert traits.foreground_mask(np.zeros((4, 4, 3), np.uint8), CFG) is None


def test_one_frame_cannot_be_shown_not_to_move():
    """A single crop has nothing to correlate against, so the answer is None -- never 0.0, which
    would read as 'perfectly frozen' and reject a lone real animal."""
    frame = np.random.default_rng(1).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    assert traits.frozen_frame_dissimilarity([frame], CFG) is None
    assert traits.frozen_frame_dissimilarity([], CFG) is None
    assert traits.frozen_frame_dissimilarity([frame, None], CFG) is None


def test_every_trait_carries_a_confidence():
    img, m = _animal()
    fur = traits.fur_trait(img, m, CFG)
    assert fur is not None and 0.0 <= fur.quality <= 1.0
    crop = np.zeros((600, 400, 3), np.uint8)
    panel = traits.head_review_panel(crop, CFG)
    assert panel is not None and 0.0 <= panel.readability <= 1.0


# ---------------------------------------------------------------------------
# Scale-free and illumination-invariant BY CONSTRUCTION.
# ---------------------------------------------------------------------------

def test_fur_ratio_survives_a_change_of_apparent_size():
    """The same animal twice as far away. The descriptor is a ratio, so nothing may move."""
    small_img, small_m = _animal(scale=0.62)
    big_img, big_m = _animal(scale=1.30)
    a = traits.appearance_vector({"fur": traits.fur_trait(small_img, small_m, CFG)})
    b = traits.appearance_vector({"fur": traits.fur_trait(big_img, big_m, CFG)})
    assert a and b
    for k in set(a) & set(b):
        assert b[k] == pytest.approx(a[k], rel=0.25, abs=0.06), f"{k} is not scale-free"


def test_fur_ratio_survives_an_auto_white_balance_swing():
    """A per-channel gain is exactly what the camera's forced-AUTO white balance does between two
    frames of the same animal. The within-crop ratio must ignore it -- that is the whole reason the
    trait is a ratio against the animal's own black and white rather than an absolute colour."""
    img, m = _animal()
    base = traits.fur_trait(img, m, CFG)
    warm = np.clip(img.astype(np.float32) * (0.75, 0.95, 1.45), 0, 255).astype(np.uint8)
    cool = np.clip(img.astype(np.float32) * (1.40, 1.00, 0.70), 0, 255).astype(np.uint8)
    for cast in (warm, cool):
        got = traits.fur_trait(cast, m, CFG)
        assert got is not None
        assert got.fur_ratio == pytest.approx(base.fur_ratio, abs=0.10)


def test_fur_ratio_orders_a_lighter_animal_above_a_darker_one():
    """The trait must still MEASURE something: with the same black mask and white muzzle in the
    crop, paler body fur has to read as a higher ratio. (It measures a fur tone; it does NOT
    identify anybody -- session-blocked LOO top-1 0.309 against a 0.345 chance rate.)"""
    dark, m = _animal(fur=105)
    pale, _ = _animal(fur=195)
    assert traits.fur_trait(pale, m, CFG).fur_ratio > traits.fur_trait(dark, m, CFG).fur_ratio


def test_the_panel_region_is_a_fraction_of_the_crop_not_a_yard_coordinate():
    """The cameras move on purpose, so the panel may not be pinned to any absolute position. Two
    crops of very different sizes must yield the same RELATIVE window."""
    for h, w in ((600, 400), (1200, 900), (700, 1500)):
        p = traits.head_review_panel(np.zeros((h, w, 3), np.uint8), CFG)
        assert p is not None
        x1, y1, x2, y2 = p.region
        assert (x1, y1, x2) == (0, 0, w)
        assert y2 / h == pytest.approx(CFG.traits_review_head_band, abs=0.01)


def test_readability_is_a_resolution_bound_and_never_exceeds_one():
    """It answers "could a human call this?", not "is there an ear here". Monotone in source
    pixels, saturating at the audited comfortable threshold."""
    frac = CFG.traits_review_ear_height_fraction
    heights = [CFG.traits_review_min_feature_px / frac + 5,
               CFG.traits_review_good_feature_px / frac,
               CFG.traits_review_good_feature_px / frac * 4]
    reads = [traits.head_review_panel(np.zeros((int(h), 300, 3), np.uint8), CFG).readability
             for h in heights]
    assert reads[0] < reads[1] <= reads[2] == 1.0
    assert all(0.0 <= r <= 1.0 for r in reads)


def test_the_panel_magnifies_and_never_shrinks():
    p = traits.head_review_panel(np.zeros((600, 200, 3), np.uint8), CFG)
    assert p.magnification >= 1.0
    assert p.image.shape[0] >= (p.region[3] - p.region[1])
    huge = traits.head_review_panel(np.zeros((6000, 400, 3), np.uint8), CFG)
    assert huge.magnification == pytest.approx(1.0), "a big crop must not be upsampled for show"


# ---------------------------------------------------------------------------
# Frozen scenes: furniture, and the animal that must not be mistaken for it.
# ---------------------------------------------------------------------------

def test_identical_frames_read_as_frozen_and_moving_ones_do_not():
    rng = np.random.default_rng(4)
    scene = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
    still = [np.clip(scene.astype(np.int16) + rng.normal(0, 2, scene.shape), 0, 255
                     ).astype(np.uint8) for _ in range(6)]
    moving = [rng.integers(0, 255, (120, 120, 3), dtype=np.uint8) for _ in range(6)]
    assert traits.frozen_frame_dissimilarity(still, CFG) < CFG.traits_static_max_dissimilarity
    assert traits.frozen_frame_dissimilarity(moving, CFG) > CFG.traits_static_max_dissimilarity


def test_frozen_dissimilarity_ignores_an_exposure_change():
    """Thumbnails are mean-removed and unit-normalised, so a brightness/contrast swing between two
    frames of the same furniture cannot masquerade as movement."""
    rng = np.random.default_rng(5)
    scene = rng.integers(40, 200, (120, 120, 3), dtype=np.uint8)
    brighter = np.clip(scene.astype(np.float32) * 1.6 + 20, 0, 255).astype(np.uint8)
    assert traits.frozen_frame_dissimilarity([scene, brighter], CFG) < \
        CFG.traits_static_max_dissimilarity


def test_a_frozen_scene_is_demoted_never_hidden():
    """5 of 16 hand-opened frozen-scene rejects were a real raccoon lying still, one with both ear
    margins readable. So the pile is returned, flagged and ranked last -- a review aid that hides
    the animal is worse than one that shows some furniture."""
    rng = np.random.default_rng(6)
    scene = rng.integers(0, 255, (500, 400, 3), dtype=np.uint8)
    frames = {i: np.clip(scene.astype(np.int16) + rng.normal(0, 2, scene.shape), 0, 255
                         ).astype(np.uint8) for i in range(6)}
    dets = [_det(i, f"2026-08-06T01:{i * 4:02d}:00-07:00", (0, 0, 400, 500)) for i in range(6)]
    sel = traits.review_candidates(dets, CFG, limit=6, load_crop=lambda i: frames[i])
    assert sel.candidates == ()
    assert {d["id"] for d in sel.demoted} == set(range(6))
    assert sel.rejected.get("frozen_scene_demoted") == 6
    assert sel.frozen_clusters and sel.frozen_clusters[0].dissimilarity < 0.1
    assert all("frozen_dissimilarity" in d for d in sel.demoted)


def test_a_short_burst_of_a_still_animal_is_not_furniture():
    """The absolute span floor exists for exactly this: real animal runs were measured as low as
    0.054 dissimilarity across a few consecutive frames, and would otherwise read as frozen."""
    rng = np.random.default_rng(7)
    scene = rng.integers(0, 255, (500, 400, 3), dtype=np.uint8)
    frames = {i: scene.copy() for i in range(6)}
    dets = [_det(i, f"2026-08-06T01:00:0{i}-07:00", (0, 0, 400, 500)) for i in range(6)]
    sel = traits.review_candidates(dets, CFG, limit=6, load_crop=lambda i: frames[i])
    assert sel.frozen_clusters == ()
    assert len(sel.candidates) >= 1


def test_the_frozen_check_reports_that_it_was_skipped_without_crops():
    """Silence would read as "nothing was static here". It has to say it never looked."""
    dets = [_det(i, f"2026-08-06T01:{i * 4:02d}:00-07:00", (0, 0, 400, 500)) for i in range(6)]
    sel = traits.review_candidates(dets, CFG, limit=6)
    assert "frozen_check_skipped" in sel.rejected
    assert sel.frozen_clusters == ()


# ---------------------------------------------------------------------------
# Review candidate selection.
# ---------------------------------------------------------------------------

def test_a_near_full_frame_box_is_rejected():
    """Ordering the June corpus by box height returns full-frame boxes first -- one containing no
    animal at all, one holding a raccoon that occupies a tenth of it. Their height is the FRAME's
    height, so the readability estimate would be pure fiction."""
    full = _det(1, "2026-08-06T01:00:00-07:00", (0, 0, 1900, 1070))
    tight = _det(2, "2026-08-06T01:05:00-07:00", (100, 100, 400, 700))
    sel = traits.review_candidates([full, tight], CFG, limit=4)
    assert [c["id"] for c in sel.candidates] == [2]
    assert sel.rejected["near_full_frame"] == 1


def test_a_crop_below_the_readability_floor_is_rejected_with_a_reason():
    tiny = _det(1, "2026-08-06T01:00:00-07:00", (0, 0, 100, 120))
    sel = traits.review_candidates([tiny], CFG, limit=4)
    assert sel.candidates == ()
    assert sel.rejected == {"too_small_to_read": 1}
    assert sel.n_input == 1


def test_candidates_are_spread_across_the_visit_not_clustered_in_one_second():
    """ANTI-PSEUDO-REPLICATION. A previous study's six "independent" crops were four near-duplicate
    frames from a seven-second window. Four huge crops one second apart plus two smaller ones
    minutes later must not return four of the burst."""
    burst = [_det(i, f"2026-08-06T01:00:0{i}-07:00", (0, 0, 400, 900)) for i in range(4)]
    later = [_det(10, "2026-08-06T01:10:00-07:00", (0, 0, 400, 600)),
             _det(11, "2026-08-06T01:20:00-07:00", (0, 0, 400, 550))]
    sel = traits.review_candidates(burst + later, CFG, limit=4)
    ids = [c["id"] for c in sel.candidates]
    assert len([i for i in ids if i < 4]) == 1, f"the burst should contribute one crop, got {ids}"
    assert 10 in ids and 11 in ids


def test_selection_does_not_pad_itself_up_to_the_limit():
    """Returning fewer is the honest answer -- "the animal was only readable at two moments" is
    information, and padding it out is how the pseudo-replication got back in."""
    dets = [_det(i, f"2026-08-06T01:0{i}:00-07:00", (0, 0, 400, 600)) for i in range(3)]
    sel = traits.review_candidates(dets, CFG, limit=8)
    assert len(sel.candidates) <= 3


def test_timestamps_are_parsed_offset_aware():
    """The rig stores ISO-8601 WITH a UTC offset and SQLite's strftime silently converts it -- a
    bug that has already produced wrong hour-of-night numbers in this project. Two timestamps that
    are one minute apart in real time must be one minute apart here, whatever offsets they carry."""
    a = traits._seconds("2026-08-06T01:00:00-07:00")
    b = traits._seconds("2026-08-06T09:01:00+01:00")
    assert b - a == pytest.approx(60.0)
    assert traits._seconds(None) is None
    assert traits._seconds("not a timestamp") is None


def test_review_candidates_never_touches_the_database_or_the_disk():
    """Pure over metadata plus an injected loader. Passing a loader that would explode proves the
    metadata path never reaches for a crop."""
    dets = [_det(i, f"2026-08-06T01:{i:02d}:00-07:00", (0, 0, 400, 600)) for i in range(3)]
    sel = traits.review_candidates(dets, CFG, limit=3)      # no load_crop at all
    assert len(sel.candidates) == 3


# ---------------------------------------------------------------------------
# Visit-level aggregation: the only level these traits work at.
# ---------------------------------------------------------------------------

def test_aggregate_reports_coverage_and_withholds_thin_traits():
    img, m = _animal()
    good = {"mask": m, "fur": traits.fur_trait(img, m, CFG)}
    blank = {"mask": None, "fur": None}
    thin = traits.aggregate_visit([good, blank, blank, blank], CFG)
    assert thin["n_crops"] == 4 and thin["n_fur"] == 1
    assert thin["fur_coverage"] == pytest.approx(0.25)
    assert thin["traits"] == {}, "one crop is not a visit-level measurement"
    rich = traits.aggregate_visit([good] * 5 + [blank] * 5, CFG)
    assert rich["fur_coverage"] == pytest.approx(0.5)
    assert "fur_ratio" in rich["traits"]
    assert rich["n_by_trait"]["fur_ratio"] == 5


def test_aggregate_of_nothing_is_empty_not_zero():
    out = traits.aggregate_visit([{"mask": None, "fur": None}] * 6, CFG)
    assert out["traits"] == {} and out["fur_coverage"] == 0.0 and out["quality"] == 0.0


def test_appearance_vector_omits_diagnostics():
    """luminance_span is an ABSOLUTE contrast -- a fact about the exposure, not the animal -- so it
    must never leave the module as a descriptor. Same for quality."""
    img, m = _animal()
    v = traits.appearance_vector(traits.crop_traits(img, CFG))
    assert "luminance_span" not in v and "quality" not in v
    assert set(v) <= set(traits.APPEARANCE_TRAITS)


# ---------------------------------------------------------------------------
# The behaviour axis, on its own terms.
# ---------------------------------------------------------------------------

def test_eating_style_reads_posture_from_boxes_alone():
    low = [(t, 0.0, 0.0, 120.0, 60.0) for t in range(30)]
    up = [(t, 0.0, 0.0, 60.0, 120.0) for t in range(30)]
    assert traits.eating_style(low, CFG).style == "mouth_first"
    assert traits.eating_style(up, CFG).style == "grabs_and_sits_back"
    mixed = [(t, 0.0, 0.0, 100.0, 100.0) for t in range(30)]        # aspect 1.0 -> neither
    assert traits.eating_style(mixed, CFG).style == "unclear"


def test_eating_style_declines_a_visit_that_is_too_short_to_judge():
    assert traits.eating_style([(0, 0, 0, 10, 5)], CFG) is None
    assert traits.eating_style([], CFG) is None


def test_eating_style_rejects_an_unknown_label():
    with pytest.raises(ValueError):
        traits.EatingStyleFlag(style="hungry", low_posture_fraction=0.0, upright_fraction=0.0,
                               posture_switches_per_min=0.0, n_frames=9, quality=1.0)


# ---------------------------------------------------------------------------
# The CLI must not be able to touch the live database.
# ---------------------------------------------------------------------------

def test_cli_connection_is_read_only(db_path):
    import db as dbmod
    dbmod.connect(db_path).close()
    conn = traits._ro_conn(db_path)
    try:
        with pytest.raises(Exception):
            conn.execute("INSERT INTO detections (timestamp, source, detection_class, confidence,"
                         " bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h, crop_path)"
                         " VALUES ('x','y','animal',1,0,0,1,1,10,10,'c.jpg')")
    finally:
        conn.close()
