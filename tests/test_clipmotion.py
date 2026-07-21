"""
Tests for clipmotion.py's pure logic -- NMS, tracklet association, gait extraction -- and the
multi-tracklet DB layer. No video, no detector: tracks are synthetic [[t, cx, cy, w, h, conf]]
lists built to known geometry, so every expectation is exact by construction.

The gait test encodes the physical model the extractor assumes: a walking quadruped's box
centre-y bobs once per stride, so a 2 Hz sine riding on a slow drift must come back as
stride_hz ~= 2.0 with high strength, while aperiodic jitter of the same magnitude must be
rejected (strength below threshold -> honest None).
"""
from __future__ import annotations

import numpy as np
import pytest

import db
from clipmotion import (build_tracks, gait_features, nms_boxes, track_features)


# ---------------------------------------------------------------------------
# NMS.
# ---------------------------------------------------------------------------

def test_nms_drops_double_box_keeps_real_pair():
    a = (0.50, 0.50, 0.20, 0.20, 0.90)
    a_dup = (0.51, 0.50, 0.20, 0.20, 0.70)     # near-identical lower-conf double-box
    b = (0.85, 0.50, 0.20, 0.20, 0.60)         # genuinely separate second animal
    kept = nms_boxes([a_dup, a, b])
    assert kept[0] == a                        # highest confidence survives, first
    assert a_dup not in kept and b in kept
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# Tracklet association.
# ---------------------------------------------------------------------------

def _walk(t0, t1, dt, x0, vx, y):
    """Sample times + a box walking horizontally at vx (normalized units/s)."""
    ts = np.arange(t0, t1 + 1e-9, dt)
    return {round(float(t), 3): (round(x0 + vx * (t - t0), 4), y, 0.1, 0.1, 0.9) for t in ts}


def test_build_tracks_two_parallel_animals():
    A = _walk(0.0, 2.0, 0.1, x0=0.2, vx=0.05, y=0.3)
    B = _walk(0.0, 2.0, 0.1, x0=0.8, vx=-0.05, y=0.7)
    samples = [(t, [A[t], B[t]]) for t in sorted(A)]
    tracks = build_tracks(samples)
    assert len(tracks) == 2
    assert all(len(tr) == 21 for tr in tracks)
    ys = sorted(tr[0][2] for tr in tracks)
    assert ys == [0.3, 0.7]                    # one track per animal, never swapped
    for tr in tracks:                          # x strictly monotonic per animal = no identity mixing
        xs = [p[1] for p in tr]
        assert xs == sorted(xs) or xs == sorted(xs, reverse=True)


def test_build_tracks_bridges_short_dropout_but_closes_long_gap():
    A = _walk(0.0, 4.0, 0.1, x0=0.2, vx=0.05, y=0.5)
    # Detector misses A between t=1.0 and t=1.8 (0.8 s < TRACK_GAP_S) -> same tracklet.
    samples = [(t, ([A[t]] if not (1.0 < t < 1.8) else [])) for t in sorted(A)]
    tracks = build_tracks(samples)
    assert len(tracks) == 1

    # A 2.0 s starvation (> TRACK_GAP_S 1.5) closes the tracklet; the animal's return starts a
    # NEW one. Both halves are long enough to survive min_hits.
    samples = [(t, ([A[t]] if (t <= 1.0 or t >= 3.0) else [])) for t in sorted(A)]
    tracks = build_tracks(samples)
    assert len(tracks) == 2


def test_build_tracks_drops_flicker():
    A = _walk(0.0, 2.0, 0.1, x0=0.2, vx=0.05, y=0.3)
    samples = [(t, [A[t]]) for t in sorted(A)]
    # One spurious far-away box in a single frame: too short to be an animal.
    samples[10] = (samples[10][0], samples[10][1] + [(0.9, 0.9, 0.05, 0.05, 0.5)])
    tracks = build_tracks(samples)
    assert len(tracks) == 1
    assert len(tracks[0]) == 21


# ---------------------------------------------------------------------------
# Gait.
# ---------------------------------------------------------------------------

def _gait_track(*, hz, bob=0.004, noise=0.0005, vx=0.06, T=8.0, dt=0.1, seed=0):
    """A walking track: x advances at vx, centre-y bobs at `hz` with detector noise on top."""
    rng = np.random.default_rng(seed)
    ts = np.arange(0.0, T + 1e-9, dt)
    return [[float(t), 0.1 + vx * float(t),
             0.5 + bob * np.sin(2 * np.pi * hz * float(t)) + float(rng.normal(0, noise)),
             0.1, 0.1, 0.9] for t in ts]


def test_gait_detects_two_hz_stride():
    g = gait_features(_gait_track(hz=2.0))
    assert g["walk_s"] >= 7.0
    assert g["stride_hz"] is not None
    assert 1.6 <= g["stride_hz"] <= 2.4
    assert g["stride_strength"] > 0.5


def test_gait_rejects_aperiodic_jitter():
    rng = np.random.default_rng(3)
    ts = np.arange(0.0, 8.0 + 1e-9, 0.1)
    track = [[float(t), 0.1 + 0.06 * float(t), 0.5 + float(rng.normal(0, 0.004)),
              0.1, 0.1, 0.9] for t in ts]
    g = gait_features(track)
    assert g["walk_s"] >= 7.0                  # it IS walking...
    assert g["stride_hz"] is None              # ...but there's no rhythm to report


def test_gait_needs_a_walking_run():
    # Stationary (eating): the bob alone may blip over the moving threshold for a step or two,
    # but there's no sustained walking run, so no stride may be reported.
    ts = np.arange(0.0, 8.0 + 1e-9, 0.1)
    track = [[float(t), 0.5, 0.5 + 0.004 * np.sin(2 * np.pi * 2.0 * float(t)), 0.1, 0.1, 0.9]
             for t in ts]
    g = gait_features(track)
    assert g["stride_hz"] is None and g["walk_s"] < 1.0


def test_track_features_still_work_on_tracklets():
    f = track_features(_gait_track(hz=2.0))
    assert f["duration_s"] == pytest.approx(8.0, abs=0.2)
    assert f["moving_frac"] > 0.9
    # A horizontal beeline whose centre-y bobs per stride: path > net by the bob, so
    # straightness lands high-but-not-1.
    assert f["straightness"] > 0.8


# ---------------------------------------------------------------------------
# DB layer: multi-tracklet insert + resumability.
# ---------------------------------------------------------------------------

def _clip(conn):
    return db.insert_clip(conn, source=db.SOURCE_GLASS_DOOR_CAM, clip_path="clips/x.mp4",
                          started_at="2026-06-10T21:00:00-07:00",
                          ended_at="2026-06-10T21:00:30-07:00", fps=10.0, width=1280,
                          height=720, frame_count=300, detection_count=12, max_confidence=0.9)


def test_insert_clip_tracks_multi_and_redo(conn):
    cid = _clip(conn)
    n = db.insert_clip_tracks(conn, clip_id=cid, model="m", n_samples=300, tracklets=[
        {"track_json": "[[0,0.5,0.5,0.1,0.1,0.9]]", "n_hits": 9,
         "features": {"duration_s": 5.0, "stride_hz": 2.1, "stride_strength": 0.6, "walk_s": 4.0}},
        {"track_json": "[[0,0.8,0.7,0.1,0.1,0.8]]", "n_hits": 6, "features": {}},
    ])
    assert n == 2
    rows = conn.execute("SELECT track_idx, n_hits, stride_hz FROM clip_tracks "
                        "WHERE clip_id=? ORDER BY track_idx", (cid,)).fetchall()
    assert [(r["track_idx"], r["n_hits"]) for r in rows] == [(0, 9), (1, 6)]
    assert rows[0]["stride_hz"] == pytest.approx(2.1)
    # Redo replaces, never duplicates.
    db.insert_clip_tracks(conn, clip_id=cid, model="m", n_samples=300, tracklets=[
        {"track_json": "[]", "n_hits": 7, "features": {}}])
    assert conn.execute("SELECT COUNT(*) FROM clip_tracks WHERE clip_id=?",
                        (cid,)).fetchone()[0] == 1


def test_empty_clip_gets_marker_row_and_leaves_the_queue(conn):
    cid = _clip(conn)
    assert len(db.clips_needing_tracks(conn, "m")) == 1
    db.insert_clip_tracks(conn, clip_id=cid, model="m", n_samples=120, tracklets=[])
    assert conn.execute("SELECT n_hits FROM clip_tracks WHERE clip_id=?",
                        (cid,)).fetchone()[0] == 0
    assert db.clips_needing_tracks(conn, "m") == []   # marker row keeps it out of the batch


# ---------------------------------------------------------------------------
# Solo-overlap linking: behaviour links stay grounded in HUMAN-confirmed names.
# ---------------------------------------------------------------------------

def test_link_tracks_requires_human_confirmed_visit(conn):
    from datetime import datetime, timedelta

    from clipmotion import link_tracks_to_individuals

    base = datetime(2026, 6, 10, 21, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)
    ts = lambda m: (base + timedelta(minutes=m)).isoformat()

    d = db.insert_detection(
        conn, timestamp=ts(1), source=db.SOURCE_GLASS_DOOR_CAM, detection_class="animal",
        confidence=0.9, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
        crop_path="crops/x.jpg", species="raccoon")
    vid = db.insert_visit(
        conn, source=db.SOURCE_GLASS_DOOR_CAM, species="raccoon", individual_id=None,
        started_at=ts(0), ended_at=ts(10), detection_count=1, max_confidence=0.9,
        representative_detection_id=d)
    db.assign_visit(conn, [d], vid)
    cid = db.insert_clip(
        conn, source=db.SOURCE_GLASS_DOOR_CAM, clip_path="clips/x.mp4", started_at=ts(1),
        ended_at=ts(2), fps=10.0, width=1280, height=720, frame_count=300,
        detection_count=10, max_confidence=0.9)
    db.insert_clip_tracks(conn, clip_id=cid, model="m", n_samples=300,
                          tracklets=[{"track_json": "[]", "n_hits": 40, "features": {}}])
    conn.commit()

    # An AUTO-assigned visit name is a prediction, not ground truth: no behaviour link.
    db.label_visit(conn, vid, "Stan", source="auto")
    r = link_tracks_to_individuals(conn, "raccoon")
    assert r["assigned"] == 0
    assert r["skipped"].get("visit_not_human_confirmed") == 1

    # The human promotes it -> the same solo tracklet links, tagged 'overlap'.
    db.label_visit(conn, vid, "Stan")
    r2 = link_tracks_to_individuals(conn, "raccoon")
    assert r2["assigned"] == 1
    row = conn.execute("SELECT individual_id, individual_source FROM clip_tracks "
                       "WHERE clip_id = ?", (cid,)).fetchone()
    assert row["individual_id"] == "Stan" and row["individual_source"] == "overlap"
