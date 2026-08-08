"""
Tests for tiledetect -- the same-frame animal count.

The point of this module is that it must never invent an animal, so most of what is asserted
here is the NEGATIVE: two real animals side by side stay two, a kit inside its mother's box stays
a kit, and no rule anywhere adds a box. The one case that gets the most attention is the boundary
between "one animal boxed twice" and "two animals overlapping", because getting that wrong in the
generous direction manufactures co-presence that never happened -- which is worse than the
detector's merging bug, since it corrupts the cannot-link constraint the individual splitter
depends on.

Geometry only: nothing here loads a model, a camera or a database.
"""
from __future__ import annotations

import pytest

import tiledetect as td


# --- geometry primitives ------------------------------------------------------------------------
def test_iou_is_scale_free():
    """Doubling every coordinate must not change an IoU -- the property the whole module leans on
    when the camera is repositioned and the animals change size in frame."""
    a, b = (0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)
    big = tuple(v * 7.3 for v in a), tuple(v * 7.3 for v in b)
    assert td.iou(a, b) == pytest.approx(td.iou(*big))


def test_iou_disjoint_and_identical():
    assert td.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert td.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_containment_is_asymmetric():
    """A small box fully inside a huge one has near-zero IoU but containment 1.0. That asymmetry
    is exactly what the umbrella rule needs and IoU cannot express."""
    small, big = (10.0, 10.0, 20.0, 20.0), (0.0, 0.0, 100.0, 100.0)
    assert td.containment(small, big) == pytest.approx(1.0)
    assert td.containment(big, small) == pytest.approx(0.01)
    assert td.iou(small, big) < 0.02


def test_degenerate_box_has_no_area():
    assert td.area((5, 5, 5, 5)) == 0.0
    assert td.area((10, 10, 0, 0)) == 0.0        # inverted, not negative area
    assert td.iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


# --- the count ----------------------------------------------------------------------------------
def test_empty_frame():
    fc = td.distinct([])
    assert (fc.n_distinct, fc.n_raw, fc.n_collapsed) == (0, 0, 0)


def test_single_box_survives():
    fc = td.distinct([((10, 10, 100, 100), 0.9)])
    assert fc.n_distinct == 1 and fc.kept == (0,)


def test_count_is_always_flagged_as_a_lower_bound():
    """Not decoration. The detector's recall on multi-animal night frames is about 0.39, so a
    surface that prints this number without the qualifier is lying by roughly a factor of two."""
    assert td.distinct([((0, 0, 10, 10), 0.5)]).lower_bound is True
    assert td.distinct([]).lower_bound is True


def test_duplicate_boxes_collapse_to_one_animal():
    """The measured real case: one raccoon on the wall wearing two boxes at IoU 0.588, stored as
    two 'raccoon' rows."""
    a = (1063.0, 418.0, 1264.0, 622.0)
    b = (1059.0, 415.0, 1392.0, 624.0)
    assert td.iou(a, b) == pytest.approx(0.588, abs=0.01)
    fc = td.distinct([(a, 0.331), (b, 0.276)])
    assert fc.n_distinct == 1
    assert fc.n_raw == 2
    assert fc.dropped_duplicate == (1,)          # the weaker of the pair goes
    assert fc.kept == (0,)


def test_near_identical_boxes_collapse():
    """A single bird boxed twice at IoU 0.925 and stored as one 'American crow' plus one
    'house finch'."""
    fc = td.distinct([((550, 342, 650, 438), 0.459), ((548, 341, 652, 440), 0.453)])
    assert fc.n_distinct == 1


def test_disjoint_animals_are_never_merged():
    """63% of same-frame pairs on this camera are disjoint. Not one of them may be collapsed."""
    fc = td.distinct([((100, 100, 200, 200), 0.8), ((600, 300, 700, 420), 0.7)])
    assert fc.n_distinct == 2
    assert fc.n_collapsed == 0


# --- THE BOUNDARY CASE --------------------------------------------------------------------------
# Two animals standing shoulder to shoulder overlap. One animal boxed twice overlaps more. The cut
# has to sit between those two populations, and the consequence of getting it wrong is asymmetric:
# collapsing a real pair loses evidence (recoverable), while splitting one animal in two invents a
# companion that was never there (poisons the cannot-link constraint and every co-presence stat).
def test_touching_animals_at_the_boundary_stay_two():
    """Two boxes overlapping just BELOW the cut are two animals and must both survive."""
    a = (0.0, 0.0, 100.0, 100.0)
    b = (36.0, 0.0, 136.0, 100.0)                 # IoU = 64/136 = 0.47, just under 0.55
    assert 0.45 < td.iou(a, b) < td.DEFAULT_IOU_MAX
    assert td.distinct([(a, 0.9), (b, 0.8)]).n_distinct == 2


def test_same_animal_at_the_boundary_becomes_one():
    """Two boxes overlapping just ABOVE the cut are one animal."""
    a = (0.0, 0.0, 100.0, 100.0)
    b = (28.0, 0.0, 128.0, 100.0)                 # IoU = 72/128 = 0.5625, just over 0.55
    assert td.DEFAULT_IOU_MAX < td.iou(a, b) < 0.60
    assert td.distinct([(a, 0.9), (b, 0.8)]).n_distinct == 1


def test_the_certified_four_raccoon_huddle_stays_four():
    """Hand-drawn ground truth from the certified reference frame
    reference_crops/_frames/stans_kids/66ab1112_src-IMG_5175_f001200.jpg -- four raccoons at one
    bowl, heavily overlapping. The detector returns ONE box for this; if it ever returned four,
    de-duplication must not undo that. Scaled to the 4K frame's real coordinates.
    """
    boxes = [
        ((240, 950, 1900, 2320), 0.9),      # left animal, broadside
        ((1010, 1130, 2270, 2650), 0.8),    # animal at the bowl, overlapping the first
        ((2360, 230, 3660, 2350), 0.7),     # large animal, right
        ((2140, 1360, 3260, 2470), 0.6),    # face-on animal, overlapping the large one
    ]
    fc = td.distinct(boxes)
    assert fc.n_distinct == 4, f"collapsed a real huddle: {fc}"


def test_kit_inside_its_mothers_box_survives():
    """The single most important non-collapse in the module. A kit standing against its mother is
    fully contained in her box; a bare containment rule would delete it, and 'kit' is defined as
    'smaller than the adult beside her', so that pair IS the signal."""
    mother = (100.0, 100.0, 500.0, 460.0)
    kit = (330.0, 300.0, 440.0, 420.0)
    assert td.containment(kit, mother) == pytest.approx(1.0)
    fc = td.distinct([(mother, 0.85), (kit, 0.40)])
    assert fc.n_distinct == 2
    assert fc.dropped_umbrella == ()


def test_umbrella_box_over_two_animals_is_dropped():
    """The measured artifact behind the project's all-time 'four raccoons in one frame' record:
    two real animals plus two brackets drawn around both of them."""
    a = (100.0, 100.0, 300.0, 300.0)
    b = (320.0, 110.0, 520.0, 310.0)
    umbrella = (98.0, 98.0, 522.0, 312.0)
    fc = td.distinct([(umbrella, 0.95), (a, 0.60), (b, 0.55)])
    assert fc.n_distinct == 2
    assert 0 in fc.dropped_umbrella
    assert fc.kept == (1, 2)


def test_the_all_time_record_frame_is_two_animals_not_four():
    """Two animals wearing two umbrellas -- the exact shape reported for 2026-06-23T22:25:25,
    where boxes 2 and 3 have IoU 0.97 with each other and each contains boxes 0 and 1."""
    a = (100.0, 100.0, 300.0, 300.0)
    b = (320.0, 110.0, 520.0, 310.0)
    u1 = (98.0, 98.0, 522.0, 312.0)
    u2 = (99.0, 99.0, 521.0, 311.0)
    assert td.iou(u1, u2) > 0.97
    fc = td.distinct([(a, 0.7), (b, 0.65), (u1, 0.95), (u2, 0.93)])
    assert fc.n_distinct == 2


def test_a_big_animal_overlapping_two_small_ones_is_not_an_umbrella():
    """The umbrella rule must not fire just because a box contains two others. It fires only when
    the box is essentially the BRACKET around them -- a large animal that dwarfs their union stays."""
    small_a = (200.0, 200.0, 240.0, 240.0)
    small_b = (260.0, 200.0, 300.0, 240.0)
    big = (100.0, 100.0, 600.0, 600.0)           # far bigger than union(small_a, small_b)
    assert td.iou(big, td.union_box([small_a, small_b])) < td.DEFAULT_UNION_MIN
    fc = td.distinct([(big, 0.9), (small_a, 0.5), (small_b, 0.5)])
    assert fc.n_distinct == 3


def test_result_is_order_independent():
    """Deterministic and permutation-invariant. The previous trait effort was undone in part by an
    unseeded routine whose numbers could not be reproduced; this module has no randomness at all,
    and this test is what holds that."""
    boxes = [((100, 100, 300, 300), 0.7), ((320, 110, 520, 310), 0.65),
             ((98, 98, 522, 312), 0.95), ((105, 104, 298, 297), 0.5)]
    base = td.distinct(boxes)
    for perm in ([3, 1, 0, 2], [2, 0, 3, 1], [1, 3, 2, 0]):
        shuffled = [boxes[i] for i in perm]
        got = td.distinct(shuffled)
        assert got.n_distinct == base.n_distinct
        assert sorted(perm[i] for i in got.kept) == sorted(base.kept)


def test_distinct_never_adds_a_box():
    """The invariant that matters most. Whatever the rules do, the count can only ever go down --
    the module is structurally incapable of manufacturing co-presence."""
    cases = [
        [],
        [((0, 0, 10, 10), 0.3)],
        [((0, 0, 100, 100), 0.9), ((10, 10, 90, 90), 0.8), ((0, 0, 100, 100), 0.7)],
        [((i * 40.0, 0.0, i * 40.0 + 60, 60.0), 0.5) for i in range(6)],
    ]
    for boxes in cases:
        fc = td.distinct(boxes)
        assert fc.n_distinct <= fc.n_raw
        assert fc.n_distinct == len(fc.kept)
        assert len(fc.kept) + len(fc.dropped_duplicate) + len(fc.dropped_umbrella) == fc.n_raw


def test_detection_objects_and_bare_tuples_agree():
    """Accepts detector.Detection without importing it (and therefore without a model)."""
    class FakeDetection:
        def __init__(self, bbox, confidence):
            self.bbox = bbox
            self.confidence = confidence

    boxes = [((100, 100, 300, 300), 0.7), ((105, 104, 298, 297), 0.5)]
    objs = [FakeDetection(*b) for b in boxes]
    assert td.distinct(objs).n_distinct == td.distinct(boxes).n_distinct == 1


def test_degenerate_boxes_do_not_count_as_animals():
    fc = td.distinct([((10, 10, 100, 100), 0.9), ((50, 50, 50, 50), 0.4)])
    assert fc.n_distinct == 1


def test_count_frame_helper():
    assert td.count_frame([((0, 0, 10, 10), 0.5), ((100, 0, 110, 10), 0.5)]) == 2


# --- furniture ----------------------------------------------------------------------------------
def test_motion_ratio_is_one_for_a_static_box():
    """A box whose pixels change exactly as much as the rest of the frame is not an animal."""
    np = pytest.importorskip("numpy")
    prev = np.zeros((100, 200), np.float32)
    cur = np.ones((100, 200), np.float32)          # uniform change everywhere
    assert td.motion_ratio(prev, cur, (10, 10, 60, 60)) == pytest.approx(1.0)


def test_motion_ratio_rises_where_something_moved():
    np = pytest.importorskip("numpy")
    prev = np.zeros((100, 200), np.float32)
    cur = np.zeros((100, 200), np.float32)
    cur[20:40, 20:40] = 100.0                      # one small patch changed a lot
    assert td.motion_ratio(prev, cur, (20, 20, 40, 40)) > 20.0
    assert td.motion_ratio(prev, cur, (120, 60, 180, 90)) == pytest.approx(0.0)


def test_motion_ratio_is_exposure_free():
    """Scaling the whole frame's change scales numerator and denominator alike, so the ratio is
    unchanged. This is what lets one threshold serve day, night, IR and a moved camera."""
    np = pytest.importorskip("numpy")
    prev = np.zeros((80, 80), np.float32)
    cur = np.zeros((80, 80), np.float32)
    cur[10:30, 10:30] = 5.0
    a = td.motion_ratio(prev, cur, (10, 10, 30, 30))
    b = td.motion_ratio(prev, cur * 37.0, (10, 10, 30, 30))
    assert a == pytest.approx(b)


def test_motion_ratio_on_a_dead_frame_is_neutral():
    """No change anywhere: the honest answer is 'cannot tell', and 1.0 keeps the box."""
    np = pytest.importorskip("numpy")
    z = np.zeros((50, 50), np.float32)
    assert td.motion_ratio(z, z, (5, 5, 20, 20)) == 1.0


def test_furniture_needs_both_persistence_and_dead_pixels():
    """staticfilter.py keys on persistence alone and would delete 6,738 real raccoon rows on this
    camera, because a raccoon feeding at a fixed dish holds a near-identical box. Persistence
    alone must not be enough here."""
    dish_raccoon = td.Cluster(box=(0, 0, 10, 10), n_frames=20, ratios=[9.4] * 20)
    assert dish_raccoon.is_furniture(20) is False       # persistent, but its pixels move

    shrub = td.Cluster(box=(0, 0, 10, 10), n_frames=20, ratios=[0.98] * 20)
    assert shrub.is_furniture(20) is True

    brief_freeze = td.Cluster(box=(0, 0, 10, 10), n_frames=2, ratios=[1.0, 1.0])
    assert brief_freeze.is_furniture(20) is False      # dead pixels, but not persistent


def test_furniture_threshold_sits_in_the_measured_gap():
    """Measured over 70 glass-door clips, 36 clusters, all opened by eye: furniture 0.98-2.00,
    animals 2.56-25.6. The cut must fall between them with margin, not on either edge."""
    assert 2.00 < 2.56
    assert 1.00 < td.DEFAULT_MOTION_RATIO < 2.00
    still_crow = td.Cluster(box=(0, 0, 10, 10), n_frames=30, ratios=[2.56] * 30)
    assert still_crow.is_furniture(30) is False
    lantern = td.Cluster(box=(0, 0, 10, 10), n_frames=30, ratios=[1.10] * 30)
    assert lantern.is_furniture(30) is True


def test_cluster_median_ratio_ignores_outliers():
    c = td.Cluster(box=(0, 0, 10, 10), n_frames=5, ratios=[1.0, 1.0, 1.1, 1.0, 90.0])
    assert c.median_ratio == pytest.approx(1.0)


def test_cluster_boxes_groups_by_place_and_is_deterministic():
    here = (100.0, 100.0, 200.0, 200.0)
    nudged = (104.0, 102.0, 205.0, 201.0)
    elsewhere = (600.0, 300.0, 700.0, 400.0)
    per_frame = [[(here, 0.5, 1.0)], [(nudged, 0.5, 1.0)], [(elsewhere, 0.5, 9.0)]]
    a = td.cluster_boxes(per_frame)
    b = td.cluster_boxes(per_frame)
    assert len(a) == 2
    assert [c.n_frames for c in a] == [2, 1]
    assert [c.box for c in a] == [c.box for c in b]      # same input, same answer, every time


def test_cluster_of_one_frame_is_never_furniture():
    """A single sighting can never be called furniture -- false negatives are recoverable, a
    deleted animal is not."""
    assert td.Cluster(box=(0, 0, 10, 10), n_frames=1, ratios=[1.0]).is_furniture(10) is False


def test_furniture_share_guard_is_off_by_default():
    """Measured: over one night's 60 clips (1,139 sampled frames) the shrub-and-bucket cluster
    that reads as a raccoon appeared 29 times -- 2.5% of the window. A share test tuned for a
    single clip would silently switch the furniture arm off at the scale where it separates."""
    assert td.DEFAULT_STATIC_MIN_SHARE == 0.0
    sporadic_shrub = td.Cluster(box=(0, 0, 10, 10), n_frames=29, ratios=[1.07] * 29)
    assert sporadic_shrub.is_furniture(1139) is True

    # ... and the same window must still keep every real animal in it (measured 7.57-16.40).
    for ratio in (7.57, 8.68, 10.25, 16.40):
        busy = td.Cluster(box=(0, 0, 10, 10), n_frames=37, ratios=[ratio] * 37)
        assert busy.is_furniture(1139) is False, ratio


def test_night_scan_aggregates_clusters_across_clips():
    """scan_clips judges furniture over every clip at once. Built without a detector: NightScan is
    assembled by hand from two clips that each see the same static box a handful of times, which
    is exactly the pattern no single clip can call."""
    box = (1034.0, 396.0, 1279.0, 690.0)
    animal = (100.0, 240.0, 274.0, 375.0)
    per_clip = [[(box, 0.3, 1.07)] for _ in range(5)] + [[(animal, 0.8, 10.2)] for _ in range(5)]
    scans = [
        td.ClipScan(path=td.Path("a.mp4"), n_sampled=10, counts_raw=[1] * 10,
                    counts_distinct=[1] * 10, counts_clean=[1] * 10, clusters=[], furniture=[],
                    per_frame=per_clip),
        td.ClipScan(path=td.Path("b.mp4"), n_sampled=10, counts_raw=[1] * 10,
                    counts_distinct=[1] * 10, counts_clean=[1] * 10, clusters=[], furniture=[],
                    per_frame=per_clip),
    ]
    all_frames = [f for s in scans for f in s.per_frame]
    clusters = td.cluster_boxes(all_frames)
    night = td.NightScan(scans=scans, clusters=clusters,
                         furniture=[c for c in clusters if c.is_furniture(len(all_frames))])
    assert night.n_sampled == 20
    assert len(night.furniture) == 1
    assert td.iou(night.furniture[0].box, box) > 0.9
    # the animal frames still count; the furniture frames no longer do
    assert night.clean_counts(scans[0]) == [0] * 5 + [1] * 5
