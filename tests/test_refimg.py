"""
Tests for refimg -- the reference-image furniture veto.

The thing this module must never do is erase an animal, so almost everything asserted here is a
NEGATIVE: every gate, on its own, must be able to stop a suppression. Each test below removes
exactly one piece of evidence from an otherwise-suppressible box and demands ABSTAIN or KEEP.
The one positive test (a synthetic watering can that has fired for a week) exists mostly to prove
the negatives are not passing vacuously.

The other half is the machinery that decides WHICH reference is even eligible: the illumination
classifier, the view-epoch persistence rule (a wall clock, not a frame count -- the measured
correction in docs/refimg-design-2026-08-07.md §2), and the certification state machine, whose
whole job is to make a resident animal impossible to learn.

Synthetic pixels and an in-memory SQLite only: nothing here loads a camera, a model, or the real
backyard.db.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

import refimg


# --- synthetic scenes ---------------------------------------------------------------------------

def yard(seed=7, brightness=150, chroma=30, h=360, w=640):
    """A deterministic fake yard: textured noise plus a few hard edges, so DSSIM and the Sobel
    metrics have real structure to compare rather than flat grey.

    The STRUCTURE moves with the seed, not just the noise. That matters: the edge fingerprint is
    a map of where the edges are, and two frames that differ only in noise are correctly judged
    to be the same camera view -- so a "the camera moved" test needs the furniture to move too.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 55, size=(h, w), dtype=np.int16) + brightness
    ox, oy = int(rng.integers(0, w // 3)), int(rng.integers(0, h // 3))
    base[oy + 10:oy + 90, ox + 20:ox + 140] += 35          # a wall
    base[h - oy - 120:h - oy - 20, w - ox - 200:w - ox - 60] -= 40   # a shrub
    base[:, 40 + ox:48 + ox] += 50                          # a post
    if chroma:
        frame = np.stack([base - chroma, base, base + chroma], axis=2)
    else:
        frame = np.stack([base, base, base], axis=2)
    return np.clip(frame, 0, 255).astype(np.uint8)


def quiet_mask():
    return np.zeros((refimg.H, refimg.W), np.uint8)


def blob_mask(box):
    m = np.zeros((refimg.H, refimg.W), np.uint8)
    x1, y1, x2, y2 = (int(v) for v in box)
    m[y1:y2, x1:x2] = 255
    return m


def certify(mgr, frame, t0=1000.0, n=6, step=1.0, mask=None):
    """Run a clean certification: n frames the detector RAN on and found nothing in."""
    obs = None
    for i in range(n):
        obs = mgr.observe(frame, detections=[], motion_mask=quiet_mask() if mask is None else mask,
                          now=t0 + i * step)
    return obs


# =================================================================================================
# Illumination -- derived from the frame, never from a clock
# =================================================================================================

def test_illumination_day_night_ir():
    assert refimg.illumination(yard(brightness=150, chroma=30)) == "day"
    assert refimg.illumination(yard(brightness=25, chroma=30)) == "night"
    # No chroma at all is the IR illuminator, however bright the scene is.
    assert refimg.illumination(yard(brightness=150, chroma=0)) == "ir"
    assert refimg.illumination(yard(brightness=25, chroma=0)) == "ir"


def test_illumination_is_switched_not_blended():
    """A frame either IS night or IS day; there is no in-between state a reference could be keyed
    on. The boundary sits at median 90, so 89 and 91 must land on opposite sides."""
    dark = np.zeros((180, 320, 3), np.uint8)
    dark[:, :, 0], dark[:, :, 1], dark[:, :, 2] = 40, 70, 100   # grey ~76, chroma 60
    assert refimg.illumination(dark) == "night"
    light = np.clip(dark.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    assert refimg.illumination(light) == "day"


def test_illumination_of_a_grey_frame_is_never_ir():
    """A single-channel frame has no chroma to measure, and calling that 'ir' would silently key a
    daylight reference to the IR bucket. It must fall back to the luminance test."""
    assert refimg.illumination(np.full((180, 320), 200, np.uint8)) == "day"
    assert refimg.illumination(np.full((180, 320), 20, np.uint8)) == "night"


# =================================================================================================
# ViewWatcher -- the wall-clock persistence rule
# =================================================================================================

def fp(frame):
    return refimg.EdgeFingerprint.of(refimg._to_working_gray(frame))


def test_sustained_disagreement_bumps_the_epoch():
    """The camera really moved: five minutes of frames that no longer look like the template."""
    w = refimg.ViewWatcher(persist_s=300.0)
    here, moved = fp(yard(seed=1)), fp(yard(seed=99))
    for t in range(0, 60, 5):
        assert w.observe(t, "day", here) == 0
    t = 60
    while t <= 60 + 320:
        w.observe(t, "day", moved)
        t += 5
    assert w.epoch == 1
    assert w.changes and w.changes[0][0] >= 60 + 300


def test_transient_flare_does_not_bump_the_epoch():
    """Measured: a 3-frame debounce still produced 9 false epochs in one glass-door day, every one
    inside a known transient haze/flare regime, because 3 frames can be 3 seconds. A minute of
    disagreement followed by recovery is haze, not a reposition."""
    w = refimg.ViewWatcher(persist_s=300.0)
    here, flare = fp(yard(seed=1)), fp(yard(seed=99))
    for t in range(0, 60, 5):
        w.observe(t, "day", here)
    for t in range(60, 120, 5):          # 60 s of disagreement -- far more than 3 frames
        w.observe(t, "day", flare)
    assert w.epoch == 0
    for t in range(120, 200, 5):
        w.observe(t, "day", here)
    assert w.epoch == 0


def test_a_gap_in_observation_cancels_a_pending_disagreement():
    """Two disagreeing samples 400 s apart are not 400 s of sustained disagreement -- in between,
    nobody was watching. Silence is not evidence, the same rule certification uses."""
    w = refimg.ViewWatcher(persist_s=300.0, max_gap_s=60.0)
    here, moved = fp(yard(seed=1)), fp(yard(seed=99))
    w.observe(0, "day", here)
    w.observe(5, "day", here)
    w.observe(10, "day", moved)
    w.observe(410, "day", moved)         # a 400 s hole: the pending disagreement is dropped
    assert w.epoch == 0


def test_the_night_gap_re_seeds_the_template_instead_of_calling_sunrise_a_reposition():
    """THE DAWN BUG, measured on the live rig 2026-08-09T05:56:52.

    Day frames stop at dusk and start again at sunrise, so the template freezes on an evening
    frame. This fingerprint does not survive a change of daylight -- with the camera provably
    stationary, that rig's day references correlate 0.075-0.68 across one day -- so the first five
    continuous minutes of dawn read as five minutes of sustained disagreement and the epoch bumped
    at corr 0.261, while the two references four minutes either side of the bump correlate 0.987.
    A false bump retires every reference AND flushes the recurrence ledger, whose >= 2 calendar
    days of evidence then has to be re-earned. Every morning.
    """
    w = refimg.ViewWatcher(persist_s=300.0, max_gap_s=60.0)
    evening, dawn = fp(yard(seed=1)), fp(yard(seed=99))
    for t in range(0, 120, 5):
        w.observe(t, "day", evening)
    for t in range(9 * 3600, 9 * 3600 + 900, 5):     # nine hours later, dawn, frames flowing
        w.observe(t, "day", dawn)
    assert w.epoch == 0
    assert w.reseeds == 1


def test_a_reposition_while_the_yard_is_being_watched_still_bumps():
    """The re-seed must not disarm the detector it lives in: a move mid-stream is still caught."""
    w = refimg.ViewWatcher(persist_s=300.0, max_gap_s=60.0)
    here, moved = fp(yard(seed=1)), fp(yard(seed=99))
    for t in range(0, 600, 5):
        w.observe(t, "day", here)
    for t in range(600, 600 + 400, 5):               # no gap: this really is a reposition
        w.observe(t, "day", moved)
    assert w.epoch == 1


def test_ir_and_night_frames_never_bump_the_epoch():
    """Every IR frame across 12 days spanning confirmed repositions collapsed into ONE view
    cluster: IR cannot see that the camera moved, so it must never be allowed to vote."""
    w = refimg.ViewWatcher(persist_s=300.0)
    here, moved = fp(yard(seed=1)), fp(yard(seed=99))
    w.observe(0, "day", here)
    for t in range(10, 4000, 5):
        w.observe(t, "ir", moved)
        w.observe(t, "night", moved)
    assert w.epoch == 0


def test_fingerprint_correlation_separates_views():
    a, b = fp(yard(seed=1)), fp(yard(seed=99))
    assert a.correlate(a) == pytest.approx(1.0, abs=1e-4)
    assert a.correlate(b) < refimg.VIEW_CORR_MIN


def test_fingerprint_survives_a_round_trip_through_bytes():
    """reference_images.edge_fp is a BLOB; a reference that cannot re-check its own view is a
    reference that vetoes across a reposition."""
    a = fp(yard(seed=3))
    back = refimg.EdgeFingerprint.frombytes(a.tobytes())
    assert back.correlate(a) == pytest.approx(1.0, abs=1e-5)


# =================================================================================================
# Certification -- the state machine that makes a sleeping animal unlearnable
# =================================================================================================

def test_a_quiet_certified_run_produces_a_reference():
    mgr = refimg.ReferenceManager(hold_s=4.0)
    certify(mgr, yard())
    ref = mgr.get("day")
    assert ref is not None
    assert ref.provenance == refimg.PROVENANCE_MOTION_MASKED
    assert mgr.n_certified >= 1
    assert mgr.n_detector_runs == 6 and mgr.n_detector_empty == 6


def test_a_detection_resets_the_certification_clock():
    """The animal's own detections are what protect it. One box mid-run and the hold restarts."""
    mgr = refimg.ReferenceManager(hold_s=4.0)
    frame = yard()
    for i in range(4):
        mgr.observe(frame, detections=[], motion_mask=quiet_mask(), now=1000.0 + i)
    mgr.observe(frame, detections=[(100, 100, 200, 200)], motion_mask=quiet_mask(), now=1004.0)
    for i in range(3):
        mgr.observe(frame, detections=[], motion_mask=quiet_mask(), now=1005.0 + i)
    assert mgr.get("day") is None        # only 2 s held since the reset
    for i in range(3):
        mgr.observe(frame, detections=[], motion_mask=quiet_mask(), now=1008.0 + i)
    assert mgr.get("day") is not None


def test_a_frame_the_detector_never_ran_on_can_never_certify():
    """detections=None is NOT detections=[]. Treating "no detection row nearby" as certification is
    what stores a reference with the raccoon in it (design doc §3.1)."""
    mgr = refimg.ReferenceManager(hold_s=4.0)
    frame = yard()
    for i in range(20):
        mgr.observe(frame, detections=None, motion_mask=quiet_mask(), now=1000.0 + i)
    assert mgr.get("day") is None
    assert mgr.n_detector_runs == 0


def test_motion_during_the_hold_blocks_certification():
    mgr = refimg.ReferenceManager(hold_s=4.0)
    frame = yard()
    for i in range(20):
        mgr.observe(frame, detections=[], motion_mask=blob_mask((100, 60, 180, 130)),
                    now=1000.0 + i)
    assert mgr.get("day") is None


def test_a_hole_in_observation_does_not_bridge_a_certification_run():
    mgr = refimg.ReferenceManager(hold_s=4.0, max_gap_s=8.0)
    frame = yard()
    mgr.observe(frame, detections=[], motion_mask=quiet_mask(), now=1000.0)
    mgr.observe(frame, detections=[], motion_mask=quiet_mask(), now=1001.0)
    mgr.observe(frame, detections=[], motion_mask=quiet_mask(), now=2000.0)   # the rig was down
    mgr.observe(frame, detections=[], motion_mask=quiet_mask(), now=2001.0)
    assert mgr.get("day") is None


def test_illumination_flip_starts_a_new_certification_and_keys_its_own_reference():
    """References are keyed per illumination and switched, never blended -- the day<->night flip is
    the largest photometric event in this corpus."""
    mgr = refimg.ReferenceManager(hold_s=4.0)
    certify(mgr, yard(brightness=150), t0=1000.0)
    assert mgr.get("day") is not None
    assert mgr.get("night") is None
    certify(mgr, yard(brightness=25), t0=1010.0)
    assert mgr.get("night") is not None


def test_a_motion_blob_voids_coverage_where_it_moved():
    """Policy E: the reference is still a certified frame, but it disowns any pixel something moved
    over in the last hour. A raccoon the detector MISSED costs an abstention, not an erasure."""
    mgr = refimg.ReferenceManager(hold_s=4.0)
    frame = yard()
    certify(mgr, frame, t0=1000.0)
    assert mgr.get("day").cover.all()
    # Something moves on the wall. The frame is not quiet, so it re-certifies nothing -- but the
    # blob is remembered and subtracted from what the existing reference claims to know.
    mgr.observe(frame, detections=[], motion_mask=blob_mask((40, 30, 110, 90)), now=1010.0)
    ref = mgr.get("day")
    assert not ref.cover.all()
    assert not ref.covered((45, 35, 105, 85))
    assert ref.covered((200, 120, 260, 170))          # elsewhere is untouched


def test_coverage_is_forgotten_after_the_no_update_horizon():
    mgr = refimg.ReferenceManager(hold_s=4.0, no_update_s=3600.0)
    frame = yard()
    certify(mgr, frame, t0=1000.0)
    mgr.observe(frame, detections=[], motion_mask=blob_mask((40, 30, 110, 90)), now=1010.0)
    assert not mgr.get("day").covered((45, 35, 105, 85))
    mgr.observe(frame, detections=None, motion_mask=quiet_mask(), now=1010.0 + 3601)
    assert mgr.get("day").covered((45, 35, 105, 85))


def test_coverage_remembers_the_blob_and_not_its_bounding_rectangle():
    """A diagonal streak of motion is not a rectangle of motion.

    The first version remembered cv2.boundingRect() of each blob, which disowns pixels nothing ever
    moved over -- measured at 1.37x the blobs' own area over the 13,438 motion-positive frames of
    2026-08-09 00:00-05:30. The corner OFF the streak has to stay covered, or the reference is
    claiming ignorance it does not have.
    """
    mgr = refimg.ReferenceManager(hold_s=4.0)
    frame = yard()
    certify(mgr, frame, t0=1000.0)
    streak = np.zeros((refimg.H, refimg.W), np.uint8)
    cv2.line(streak, (40, 30), (140, 130), 255, 9)          # a bottom-left-to-top-right diagonal
    mgr.observe(frame, detections=[], motion_mask=streak, now=1010.0)
    ref = mgr.get("day")
    assert not ref.covered((80, 70, 100, 90))               # ON the streak: disowned, as before
    assert ref.covered((105, 35, 135, 60))                  # the rectangle's corner: never moved


def test_the_motion_memory_does_not_grow_with_a_busy_night():
    """Cost, not safety, and it is on the CAPTURE THREAD. The rectangle list was re-drawn in full
    on every frame: measured 4.9 ms at 2,500 remembered rectangles and 16.7 ms at 10,000, against
    the 7.6 ms the whole per-frame veto was budgeted at -- and the busiest hour measured on this
    rig ran 1,546 detector frames. One timestamp per pixel is flat in both memory and time."""
    mgr = refimg.ReferenceManager(hold_s=4.0)
    frame = yard()
    certify(mgr, frame, t0=1000.0)
    before = mgr._motion_at.nbytes
    for i in range(400):
        mgr.observe(frame, detections=[], motion_mask=blob_mask((i % 200, i % 100,
                                                                i % 200 + 30, i % 100 + 30)),
                    now=1010.0 + i)
    assert mgr._motion_at.nbytes == before


def test_an_epoch_change_flushes_every_reference():
    """A reference for a scene that no longer exists is the silent failure config.py warns about."""
    mgr = refimg.ReferenceManager(hold_s=4.0, view=refimg.ViewWatcher(persist_s=60.0))
    certify(mgr, yard(seed=1), t0=1000.0)
    assert mgr.get("day") is not None
    moved = yard(seed=99)
    t = 1010.0
    while t <= 1010.0 + 70:
        mgr.observe(moved, detections=None, motion_mask=quiet_mask(), now=t)
        t += 5
    assert mgr.view_epoch == 1
    assert mgr.get("day") is None


# =================================================================================================
# The recurrence ledger
# =================================================================================================

def make_recurrence(**kw):
    kw.setdefault("min_events", 5)
    kw.setdefault("min_days", 2)
    kw.setdefault("event_gap_s", 600.0)
    kw.setdefault("iou", 0.60)
    return refimg.Recurrence(**kw)


def fire(rec, box, moments):
    for m in moments:
        rec.observe(box, m)
    return rec.stats(box)


def test_recurrence_counts_independent_events_not_detections():
    """A hundred frames of one animal over ten minutes is ONE occasion, not a hundred."""
    rec = make_recurrence()
    day = datetime(2026, 8, 5, 22, 0, 0)
    stats = fire(rec, (10, 10, 50, 50), [day + timedelta(seconds=i) for i in range(100)])
    assert stats["n"] == 100
    assert stats["events"] == 1
    assert stats["days"] == 1


def test_recurrence_needs_events_across_days():
    rec = make_recurrence()
    d1 = datetime(2026, 8, 5, 22, 0, 0)
    box = (10, 10, 50, 50)
    fire(rec, box, [d1 + timedelta(seconds=700 * i) for i in range(5)])
    assert rec.stats(box)["events"] == 5
    assert not rec.satisfied(box)                 # five firings, but all on one day
    rec.observe(box, d1 + timedelta(days=1))
    assert rec.satisfied(box)


def test_recurrence_separates_spots_by_iou():
    rec = make_recurrence()
    d1 = datetime(2026, 8, 5, 22, 0, 0)
    here, elsewhere = (10, 10, 50, 50), (200, 100, 240, 140)
    fire(rec, here, [d1 + timedelta(seconds=700 * i) for i in range(5)])
    rec.observe(here, d1 + timedelta(days=1))
    assert rec.satisfied(here)
    assert not rec.satisfied(elsewhere)
    assert rec.stats(elsewhere) == dict(events=0, days=0, n=0)


def test_recurrence_survives_a_restart(tmp_path):
    """The ledger has to outlive the rig, or every restart resets furniture to 'first seen' and the
    veto goes quiet for two days."""
    path = tmp_path / "recurrence.json"
    rec = make_recurrence(path=path)
    d1 = datetime(2026, 8, 5, 22, 0, 0)
    box = (10, 10, 50, 50)
    fire(rec, box, [d1 + timedelta(seconds=700 * i) for i in range(5)])
    rec.observe(box, d1 + timedelta(days=1))
    assert rec.save(now=d1.timestamp() + 86400, force=True)

    back = make_recurrence(path=path).load()
    assert back.satisfied(box)
    assert back.stats(box) == rec.stats(box)


def test_recurrence_is_flushed_when_the_camera_moves():
    """Every stored box points at a spot that is no longer there."""
    rec = make_recurrence()
    d1 = datetime(2026, 8, 5, 22, 0, 0)
    box = (10, 10, 50, 50)
    fire(rec, box, [d1 + timedelta(seconds=700 * i) for i in range(5)])
    rec.observe(box, d1 + timedelta(days=1))
    assert rec.satisfied(box)
    rec.observe(box, d1 + timedelta(days=1, seconds=10), epoch=1)
    assert not rec.satisfied(box)


def test_recurrence_prunes_spots_that_went_quiet():
    rec = make_recurrence(retain_days=30)
    d1 = datetime(2026, 8, 5, 22, 0, 0)
    fire(rec, (10, 10, 50, 50), [d1 + timedelta(seconds=700 * i) for i in range(5)])
    rec.prune(now=(d1 + timedelta(days=45)).timestamp())
    assert rec.clusters == []


# =================================================================================================
# The veto conjunction -- one test per gate, each one removing a single piece of evidence
# =================================================================================================

def scene(now=1000.0, seed=7):
    frame = yard(seed=seed)
    obs = refimg.prepare(frame, now=now)
    ref = refimg.Reference(image=obs.gray.copy(), captured_at=now - 30.0, illumination="day",
                           view_epoch=0, fingerprint=obs.fingerprint, source="glass_door_cam",
                           id=42)
    return frame, obs, ref


def furniture_recurrence(box):
    rec = make_recurrence()
    d1 = datetime(2026, 8, 5, 22, 0, 0)
    fire(rec, box, [d1 + timedelta(seconds=700 * i) for i in range(5)])
    rec.observe(box, d1 + timedelta(days=1))
    assert rec.satisfied(box)
    return rec


BOX = (200.0, 110.0, 250.0, 150.0)      # working coordinates, a quiet patch of the fake yard


def test_a_recurring_spot_that_matches_the_empty_reference_is_suppressed():
    """The positive case: a watering can that has fired for a week, sitting exactly where the
    certified-empty frame says there is nothing."""
    _, obs, ref = scene()
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    assert d.decision == refimg.SUPPRESS
    assert d.reason == "matches_empty_reference_at_a_recurring_spot"
    assert d.suppressed and d.ref_id == 42
    assert d.recurrence["events"] >= 5 and d.recurrence["days"] >= 2


def test_a_first_time_match_is_kept():
    """Gate 6 alone. The pixels agree perfectly and it is still not suppressed, because one sighting
    is not a furniture signature -- and location recurrence alone would flag the food bowl, which
    is 27 days of real raccoons."""
    _, obs, ref = scene()
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, make_recurrence())
    assert d.decision == refimg.KEEP
    assert d.reason == "not_recurrent_enough"
    assert d.scores["lum"] < d.thresholds["lum"]     # the pixel gate DID pass


def test_recurrence_alone_cannot_suppress_when_the_pixels_differ():
    """Gate 5 alone. This is the animal-at-the-food-bowl case, and it is the reason the metric
    gate exists at all."""
    frame, _, ref = scene()
    animal = frame.copy()
    x1, y1, x2, y2 = (int(v * 2) for v in BOX)       # working -> the 640x360 source frame
    animal[y1:y2, x1:x2] = np.clip(animal[y1:y2, x1:x2].astype(np.int16) + 70, 0, 255)
    obs = refimg.prepare(animal, now=1000.0)
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    assert d.decision == refimg.KEEP
    assert d.reason == "pixels_differ_from_empty"


def test_no_reference_abstains():
    _, obs, _ = scene()
    d = refimg.ShadowVeto().evaluate(BOX, obs, None, furniture_recurrence(BOX))
    assert d.decision == refimg.ABSTAIN and d.reason == "no_reference"


def test_a_view_epoch_change_abstains():
    _, obs, ref = scene()
    obs.view_epoch = 1                      # the camera was seen to move since this was certified
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    assert d.decision == refimg.ABSTAIN and d.reason == "view_epoch_changed"


def test_a_reference_of_another_view_abstains():
    """Belt and braces under the epoch gate: a cross-view empty pair exceeded the safe metric
    threshold 100% of the time in the drift study, so such a reference vetoes at random."""
    _, obs, ref = scene()
    ref.fingerprint = fp(yard(seed=99))
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    assert d.decision == refimg.ABSTAIN and d.reason == "camera_moved"
    assert d.view_corr < refimg.VIEW_CORR_MIN


def test_a_stale_reference_abstains():
    _, obs, ref = scene()
    ref.captured_at = obs.ts - 3 * 3600.0
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    assert d.decision == refimg.ABSTAIN and d.reason == "reference_stale"
    assert d.age_s == pytest.approx(10800.0)


def test_uncovered_pixels_abstain():
    """An unknown pixel is not evidence of emptiness."""
    _, obs, ref = scene()
    ref.cover[100:160, 190:260] = False
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    assert d.decision == refimg.ABSTAIN and d.reason == "reference_has_no_pixels_here"


def test_an_unknown_illumination_abstains():
    """No measured thresholds for this camera in this light means no opinion -- never a guess."""
    _, obs, ref = scene()
    veto = refimg.ShadowVeto(thresholds={"night": {"lum": 11.406, "dssim": 0.2346,
                                                   "sobel": 0.4854}})
    d = veto.evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    assert d.decision == refimg.ABSTAIN and d.reason == "no_thresholds_for_illumination"


def test_every_metric_is_part_of_the_conjunction():
    """Each of lum/dssim/sobel on its own must be able to save a box: adding sobel to the
    conjunction moved one race case 36 -> 13 erased animals and another only 131 -> 120, so it is
    worth having and cannot be relied upon."""
    _, obs, ref = scene()
    rec = furniture_recurrence(BOX)
    for metric in refimg.METRICS:
        tight = {k: dict(v) for k, v in refimg.ShadowVeto().thresholds.items()}
        tight["day"][metric] = -1.0                  # nothing can score below this
        d = refimg.ShadowVeto(thresholds=tight).evaluate(BOX, obs, ref, rec)
        assert d.decision == refimg.KEEP, metric
        assert d.reason == "pixels_differ_from_empty"


def test_the_full_conjunction_is_recorded_even_when_it_abstains():
    """suppress_detail has to make a bad decision diagnosable months later without re-running
    anything, so the trace is written for KEEP and ABSTAIN too."""
    _, obs, ref = scene()
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, make_recurrence())
    detail = d.to_detail()
    assert detail["decision"] == refimg.KEEP
    assert set(detail["scores"]) >= set(refimg.METRICS)
    assert detail["thresholds"] and detail["provenance"] and detail["age_s"] is not None
    assert "view_corr" in detail


# =================================================================================================
# Decision serialisation and determinism
# =================================================================================================

def test_decision_round_trips_through_json():
    _, obs, ref = scene()
    for rec in (make_recurrence(), furniture_recurrence(BOX)):
        d = refimg.ShadowVeto().evaluate(BOX, obs, ref, rec)
        assert refimg.Decision.from_json(d.to_json()) == d


def test_decision_json_matches_the_designed_shape():
    _, obs, ref = scene()
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    payload = json.loads(d.to_json())
    assert payload["decision"] == "SUPPRESS"
    assert payload["provenance"] == refimg.PROVENANCE_MOTION_MASKED
    assert set(payload["recurrence"]) == {"events", "days", "n"}
    assert set(payload["thresholds"]) == set(refimg.METRICS)
    assert set(payload["scores"]) >= set(refimg.METRICS)


def test_unreadable_detail_degrades_to_an_abstention():
    d = refimg.Decision.from_json("{not json")
    assert d.decision == refimg.ABSTAIN and d.reason == "unreadable_detail"


def test_the_veto_is_deterministic():
    """Same inputs, twice, through two independent objects -- byte-identical traces. Anything
    stochastic in here would make the shadow week unauditable."""
    _, obs, ref = scene()
    rec = furniture_recurrence(BOX)
    a = refimg.ShadowVeto().evaluate(BOX, obs, ref, rec).to_json()
    b = refimg.ShadowVeto().evaluate(BOX, obs, ref, rec).to_json()
    assert a == b
    # ... and the per-frame FramePair cache must not change an answer either.
    veto = refimg.ShadowVeto()
    assert veto.evaluate(BOX, obs, ref, rec).to_json() == a
    assert veto.evaluate(BOX, obs, ref, rec).to_json() == a


def test_the_pair_cache_notices_a_new_reference():
    """The cache is keyed on the reference as well as the frame; reusing a stale FramePair would
    score a box against the wrong empty yard."""
    frame, obs, ref = scene()
    veto = refimg.ShadowVeto()
    rec = furniture_recurrence(BOX)
    assert veto.evaluate(BOX, obs, ref, rec).decision == refimg.SUPPRESS
    other = refimg.Reference(image=refimg._to_working_gray(yard(seed=99)), captured_at=obs.ts - 30,
                             illumination="day", view_epoch=0, fingerprint=obs.fingerprint)
    d = veto.evaluate(BOX, obs, other, rec)
    assert d.decision == refimg.KEEP and d.reason == "pixels_differ_from_empty"


# =================================================================================================
# End to end: certify, then veto against the reference the manager produced
# =================================================================================================

def test_a_manager_certified_reference_suppresses_its_own_furniture():
    mgr = refimg.ReferenceManager(hold_s=4.0)
    frame = yard()
    obs = certify(mgr, frame, t0=1000.0)
    ref = mgr.get("day")
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    assert d.decision == refimg.SUPPRESS


def test_a_manager_certified_reference_keeps_an_animal_that_walks_into_frame():
    """The one that matters. The reference is certified empty, the spot is a known recurring
    hotspot -- and the moment something is actually THERE, the box survives."""
    mgr = refimg.ReferenceManager(hold_s=4.0)
    frame = yard()
    certify(mgr, frame, t0=1000.0)
    ref = mgr.get("day")
    animal = frame.copy()
    x1, y1, x2, y2 = (int(v * 2) for v in BOX)
    rng = np.random.default_rng(3)
    animal[y1:y2, x1:x2] = rng.integers(0, 90, size=(y2 - y1, x2 - x1, 3), dtype=np.uint8)
    obs = refimg.prepare(animal, now=1006.0)
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    assert d.decision == refimg.KEEP


# =================================================================================================
# Persistence and the shadow-mode write
# =================================================================================================

DETECTIONS_DDL = """
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL, source TEXT NOT NULL, detection_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    frame_w INTEGER, frame_h INTEGER, crop_path TEXT, species TEXT
);
"""


@pytest.fixture
def memdb():
    """A throwaway in-memory DB with just enough of detections to exercise the shadow write. It
    deliberately does NOT go through db.py: the real backyard.db is never involved."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(DETECTIONS_DDL)
    refimg.ensure_tables(conn)
    try:
        yield conn
    finally:
        conn.close()


def insert_detection(conn, when, crop_path, box=(400, 220, 500, 300), species="raccoon"):
    cur = conn.execute(
        "INSERT INTO detections (timestamp, source, detection_class, confidence, bbox_x1, bbox_y1,"
        " bbox_x2, bbox_y2, frame_w, frame_h, crop_path, species) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (when, "glass_door_cam", "animal", 0.42, box[0], box[1], box[2], box[3], 640, 360,
         str(crop_path), species))
    conn.commit()
    return int(cur.lastrowid)


def test_ensure_tables_is_idempotent(memdb):
    refimg.ensure_tables(memdb)
    refimg.ensure_tables(memdb)
    cols = {r[1] for r in memdb.execute("PRAGMA table_info(detections)")}
    assert set(refimg.SUPPRESSION_COLUMNS) <= cols
    assert refimg._table_exists(memdb, "reference_images")
    assert refimg._table_exists(memdb, "view_epochs")


def test_shadow_mode_writes_metadata_and_nothing_else(memdb, tmp_path):
    """The binding property of this whole change: a suppression flags the row and leaves it, its
    crop and every existing query completely alone."""
    crop = tmp_path / "c.jpg"
    cv2.imwrite(str(crop), yard(h=80, w=80))
    det_id = insert_detection(memdb, "2026-08-07T22:10:00-07:00", crop)
    before = memdb.execute("SELECT timestamp, source, confidence, species, crop_path FROM "
                           "detections WHERE id=?", (det_id,)).fetchone()

    _, obs, ref = scene()
    d = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    assert refimg.mark_suppressed(memdb, det_id, d, ref_id=42) is True
    memdb.commit()

    row = memdb.execute("SELECT timestamp, source, confidence, species, crop_path, suppressed_at,"
                        " suppressed_by, suppress_ref_id, suppress_detail FROM detections WHERE"
                        " id=?", (det_id,)).fetchone()
    assert row[:5] == before                       # the detection itself is untouched
    assert crop.exists()
    assert row[5] and row[6] == refimg.SUPPRESSED_BY and row[7] == 42
    assert refimg.Decision.from_json(row[8]).decision == refimg.SUPPRESS


def test_a_kept_box_is_never_flagged(memdb, tmp_path):
    crop = tmp_path / "c.jpg"
    cv2.imwrite(str(crop), yard(h=80, w=80))
    det_id = insert_detection(memdb, "2026-08-07T22:10:00-07:00", crop)
    _, obs, ref = scene()
    keep = refimg.ShadowVeto().evaluate(BOX, obs, ref, make_recurrence())
    assert keep.decision == refimg.KEEP
    assert refimg.mark_suppressed(memdb, det_id, keep) is False
    assert memdb.execute("SELECT suppressed_at FROM detections WHERE id=?",
                         (det_id,)).fetchone()[0] is None


def test_reference_round_trips_through_the_store(memdb, tmp_path):
    """A suppression must stay replayable against the exact image that produced it."""
    mgr = refimg.ReferenceManager(source="glass_door_cam", hold_s=4.0)
    frame = yard()
    certify(mgr, frame, t0=1000.0)
    mgr.observe(frame, detections=[], motion_mask=blob_mask((40, 30, 110, 90)), now=1010.0)
    ref = mgr.get("day")
    rid = refimg.save_reference(memdb, ref, root=tmp_path)

    back = refimg.load_reference(memdb, "glass_door_cam", "day", 0)
    assert back is not None and back.id == rid
    assert np.array_equal(back.image, ref.image)
    assert np.array_equal(back.cover, ref.cover)
    assert back.fingerprint.correlate(ref.fingerprint) == pytest.approx(1.0, abs=1e-5)
    assert back.captured_at == pytest.approx(ref.captured_at, abs=1.0)


def test_retiring_an_epoch_hides_its_references_without_deleting_them(memdb, tmp_path):
    mgr = refimg.ReferenceManager(hold_s=4.0)
    certify(mgr, yard(), t0=1000.0)
    refimg.save_reference(memdb, mgr.get("day"), root=tmp_path)
    refimg.retire_references(memdb, "glass_door_cam", 0)
    assert refimg.load_reference(memdb, "glass_door_cam", "day", 0) is None
    assert memdb.execute("SELECT COUNT(*) FROM reference_images").fetchone()[0] == 1


def test_view_epochs_are_recorded_once(memdb):
    refimg.record_view_epoch(memdb, "glass_door_cam", 1, started_at=1000.0, corr=0.21)
    refimg.record_view_epoch(memdb, "glass_door_cam", 1, started_at=2000.0, corr=0.19)
    rows = memdb.execute("SELECT epoch, detected_by, corr FROM view_epochs").fetchall()
    assert rows == [(1, "edge_fp_corr", 0.21)]


# =================================================================================================
# --review : the audit loop
# =================================================================================================

def test_review_groups_by_cluster_and_counts_by_day(memdb, tmp_path):
    crop = tmp_path / "c.jpg"
    cv2.imwrite(str(crop), yard(h=90, w=90))
    _, obs, ref = scene()
    decision = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))

    now = datetime.now().astimezone()
    can = (400, 220, 500, 300)          # the same watering can, twice on two nights
    gap = (60, 40, 120, 100)            # a different spot
    for i, (box, when) in enumerate([(can, now - timedelta(hours=2)),
                                     (can, now - timedelta(days=1, hours=2)),
                                     (gap, now - timedelta(hours=3))]):
        det_id = insert_detection(memdb, when.isoformat(), crop, box=box)
        refimg.mark_suppressed(memdb, det_id, decision, ref_id=7, when=when.timestamp())
    memdb.commit()

    out = refimg.review(memdb, days=7, out_dir=tmp_path / "sheets", root=tmp_path)
    assert len(out["rows"]) == 3
    assert [len(c) for c in out["clusters"]] == [2, 1]
    assert sum(out["per_day"].values()) == 3
    assert len(out["per_day"]) == 2
    assert out["sheets"] and out["sheets"][0].exists()
    sheet = cv2.imread(str(out["sheets"][0]))
    assert sheet is not None and sheet.shape[0] > 100


def test_a_long_cluster_spills_onto_more_sheets_instead_of_being_truncated(tmp_path):
    """The sheet has to hold EVERY flagged row. A cluster that silently stopped at the page break
    would hide exactly the crops the shadow week exists to look at."""
    crop = tmp_path / "c.jpg"
    cv2.imwrite(str(crop), yard(h=90, w=90))
    _, obs, ref = scene()
    decision = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    rows = [dict(id=i, timestamp=f"2026-08-07T22:{i:02d}:00-07:00", crop_path=str(crop),
                 decision=decision) for i in range(23)]
    sheets = refimg.contact_sheets([rows], tmp_path / "sheets", root=tmp_path,
                                   cols=5, max_tiles=10)
    assert len(sheets) == 3                      # 23 tiles over pages of 10
    assert all(s.exists() for s in sheets)


def test_review_ignores_rows_outside_the_window(memdb, tmp_path):
    crop = tmp_path / "c.jpg"
    cv2.imwrite(str(crop), yard(h=90, w=90))
    _, obs, ref = scene()
    decision = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    old = datetime.now().astimezone() - timedelta(days=40)
    det_id = insert_detection(memdb, old.isoformat(), crop)
    refimg.mark_suppressed(memdb, det_id, decision, when=old.timestamp())
    memdb.commit()
    assert refimg.review(memdb, days=7, out_dir=tmp_path, root=tmp_path)["rows"] == []


def test_review_survives_a_missing_crop(memdb, tmp_path):
    """Clips and crops get pruned; the audit sheet must still render the row it is reporting."""
    _, obs, ref = scene()
    decision = refimg.ShadowVeto().evaluate(BOX, obs, ref, furniture_recurrence(BOX))
    when = datetime.now().astimezone() - timedelta(hours=1)
    det_id = insert_detection(memdb, when.isoformat(), tmp_path / "gone.jpg")
    refimg.mark_suppressed(memdb, det_id, decision, when=when.timestamp())
    memdb.commit()
    out = refimg.review(memdb, days=7, out_dir=tmp_path / "sheets", root=tmp_path)
    assert len(out["rows"]) == 1 and out["sheets"][0].exists()


def test_load_suppressed_is_silent_on_a_database_without_the_columns():
    """Shadow mode ships disabled, so --review on a stock database must say nothing rather than
    raise."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(DETECTIONS_DDL)
    try:
        assert refimg._has_suppression_columns(conn) is False
        assert refimg.load_suppressed(conn, days=7) == []
    finally:
        conn.close()


# =================================================================================================
# Geometry helpers
# =================================================================================================

def test_scale_box_maps_capture_resolution_into_the_working_frame():
    """Thresholds are measured at 320x180; a box that arrived in 1920x1080 pixels has to land in
    the same place as the same box from a 1280x720 session."""
    a = refimg.scale_box((960, 540, 1440, 810), 1920, 1080)
    b = refimg.scale_box((640, 360, 960, 540), 1280, 720)
    assert a == pytest.approx(b)
    assert a == pytest.approx((160.0, 90.0, 240.0, 135.0))


def test_box_slice_clamps_to_the_working_frame():
    """A box running off the edge of the frame must not produce an empty or wrapped slice."""
    sl = refimg._box_slice((-50, -50, 5000, 5000))
    assert sl[0].start == 0 and sl[1].start == 0
    assert sl[0].stop == refimg.H and sl[1].stop == refimg.W


def test_box_iou_basics():
    assert refimg.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert refimg.box_iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0
