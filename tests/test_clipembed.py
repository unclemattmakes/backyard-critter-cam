"""
Tests for clipembed.py pure logic (frame selection, box cropping) and the clip_track_embeddings
DB layer. No video decode, no model: select_frames works on synthetic track points, _crop on a
hand-built numpy frame, and the DB round-trip on bytes.
"""
from __future__ import annotations

import numpy as np

import db
from clipembed import _crop, select_frames


def _pt(t, conf, cx=0.5, cy=0.5, w=0.2, h=0.2):
    return [t, cx, cy, w, h, conf]


def test_select_frames_returns_all_when_short():
    track = [_pt(0, 0.9), _pt(0.1, 0.8)]
    assert select_frames(track, 10) == track


def test_select_frames_spreads_and_prefers_confident():
    # 12 points over t=0..11; the sharpest in each of 4 even time-bins should be chosen.
    track = [_pt(t, 0.5 + (0.4 if t % 3 == 0 else 0.0)) for t in range(12)]
    out = select_frames(track, 4)
    assert len(out) == 4
    times = [p[0] for p in out]
    assert times == sorted(times)                      # spread across the timeline, in order
    assert times[0] < 3 and times[-1] >= 9             # first/last bins represented
    assert all(p[5] >= 0.5 for p in out)               # confidence-preferring within each bin


def test_select_frames_empty():
    assert select_frames([], 5) == []


def test_crop_denormalizes_pads_and_clamps():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[40:60, 90:110] = (0, 0, 255)                 # a red BGR block at the centre box
    im = _crop(frame, (0.5, 0.5, 0.2, 0.2), pad=0.0)   # box = x[80..120], y[40..60]
    assert im is not None
    assert im.size == (40, 20)                          # (w,h) in pixels
    # BGR->RGB: the red block (BGR 0,0,255) should read as RGB red.
    assert im.getpixel((20, 10))[0] > 200 and im.getpixel((20, 10))[2] < 60


def test_crop_padding_grows_box():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    base = _crop(frame, (0.5, 0.5, 0.2, 0.2), pad=0.0)
    padded = _crop(frame, (0.5, 0.5, 0.2, 0.2), pad=0.25)
    assert padded.size[0] > base.size[0] and padded.size[1] > base.size[1]


def test_crop_rejects_degenerate_box():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert _crop(frame, (0.5, 0.5, 0.0, 0.0)) is None
    # A box fully off the top-left clamps to nothing.
    assert _crop(frame, (-0.5, -0.5, 0.2, 0.2)) is None


# ---------------------------------------------------------------------------
# DB layer.
# ---------------------------------------------------------------------------

def _clip_with_track(conn, *, n_hits=40):
    cid = db.insert_clip(conn, source=db.SOURCE_GLASS_DOOR_CAM, clip_path="clips/x.mp4",
                         started_at="2026-06-11T21:00:00-07:00",
                         ended_at="2026-06-11T21:00:30-07:00", fps=10.0, width=1280, height=720,
                         frame_count=300, detection_count=12, max_confidence=0.9)
    db.insert_clip_tracks(conn, clip_id=cid, model="MDV6", n_samples=300, tracklets=[
        {"track_json": "[[0,0.5,0.5,0.2,0.2,0.9]]", "n_hits": n_hits, "features": {}}])
    return conn.execute("SELECT id FROM clip_tracks WHERE clip_id=?", (cid,)).fetchone()[0]


def _vec(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return (v / np.linalg.norm(v)).tobytes()


def test_clip_track_embedding_round_trip_and_resumable(conn):
    tid = _clip_with_track(conn, n_hits=40)
    assert len(db.clip_tracks_needing_embedding(conn, "megadescriptor-l-384", 30)) == 1
    db.insert_clip_track_embedding(conn, track_id=tid, model="megadescriptor-l-384",
                                   dim=3, embedding=_vec(1, 0, 0), n_frames=8)
    # Embedded -> drops out of the work-list.
    assert db.clip_tracks_needing_embedding(conn, "megadescriptor-l-384", 30) == []
    rows = db.load_clip_track_embeddings(conn, "megadescriptor-l-384")
    assert len(rows) == 1 and rows[0]["track_id"] == tid
    v = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
    assert v @ v == np.float32(1.0)                     # round-trips L2-normalized
    # Idempotent replace.
    db.insert_clip_track_embedding(conn, track_id=tid, model="megadescriptor-l-384",
                                   dim=3, embedding=_vec(0, 1, 0), n_frames=10)
    assert len(db.load_clip_track_embeddings(conn, "megadescriptor-l-384")) == 1


def test_clip_tracks_needing_embedding_respects_min_hits(conn):
    _clip_with_track(conn, n_hits=12)                   # below the sustained gate
    assert db.clip_tracks_needing_embedding(conn, "megadescriptor-l-384", 30) == []


def test_embedding_cascades_when_track_recomputed(conn):
    tid = _clip_with_track(conn, n_hits=40)
    cid = conn.execute("SELECT clip_id FROM clip_tracks WHERE id=?", (tid,)).fetchone()[0]
    db.insert_clip_track_embedding(conn, track_id=tid, model="megadescriptor-l-384",
                                   dim=3, embedding=_vec(1, 0, 0), n_frames=8)
    # Recomputing the clip's tracks deletes the old track row -> its embedding cascades away.
    db.insert_clip_tracks(conn, clip_id=cid, model="MDV6", n_samples=300, tracklets=[
        {"track_json": "[]", "n_hits": 40, "features": {}}])
    assert db.load_clip_track_embeddings(conn, "megadescriptor-l-384") == []
