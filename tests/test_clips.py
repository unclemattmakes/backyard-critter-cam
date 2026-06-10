"""
Tests for clips.py -- ClipRecorder (the phase-4 behaviour-clip writer).

ClipRecorder is driven by the capture loop with three calls, all taking `now` (a monotonic clock)
as a parameter -- so the whole lifecycle is testable with synthetic numpy frames and a hand-stepped
clock, no camera and no real time:

    recorder.note_frame(frame, now, loop_fps=)   # every frame: buffers pre-roll, writes, checks stop
    recorder.note_detection(now, detections)     # animal hits: start / extend the clip
    recorder.finalize()                          # flush + write the DB row

We use dataclasses.replace(config.CONFIG, ...) to point clips_dir / db_path at a tmp dir and set
small pre/post-roll, then verify: pre-roll buffering, start-on-detection, auto-stop after
clip_post_roll_s, the clip_max_s cap, and that finalize() writes BOTH a clips DB row AND a
decodable .mp4 (reopened with cv2.VideoCapture, frames counted). cv2 + numpy are installed.
"""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

import clips
import config
import db


# A fake detector.Detection: ClipRecorder only ever reads `.confidence` (and the capture loop --
# not the recorder -- filters on class), but we give it `.class_name` too to match the real object.
class FakeDetection:
    def __init__(self, confidence: float, class_name: str = "animal"):
        self.confidence = confidence
        self.class_name = class_name


def _frame():
    """A synthetic frame matching the task spec: 120x160 BGR, zeros."""
    return np.zeros((120, 160, 3), dtype=np.uint8)


@pytest.fixture
def clip_cfg(tmp_path):
    """A Config copy whose clips_dir/db_path live under tmp, with tiny pre/post-roll and a low
    max so the cap is reachable in a handful of synthetic frames. clip_fps is forced so the
    written file's playback rate is deterministic (no reliance on a measured live rate)."""
    return dataclasses.replace(
        config.CONFIG,
        clips_dir=tmp_path / "clips",
        db_path=tmp_path / "clips_test.db",
        clip_pre_roll_s=0.0,     # default: no pre-roll (clean frame counting); overridden per-test
        clip_post_roll_s=1.0,
        clip_max_s=5.0,
        clip_fps=15.0,
        clip_scale=1.0,
        clip_codec="mp4v",
    )


@pytest.fixture
def clip_conn(clip_cfg):
    c = db.connect(clip_cfg.db_path)
    try:
        yield c
    finally:
        c.close()


def _count_clip_rows(conn):
    return conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]


def _decode_frame_count(path) -> int:
    """Reopen the written .mp4 and count decodable frames -- proves the file is valid, not just
    present. (Reads frames rather than trusting CAP_PROP_FRAME_COUNT, which some backends estimate.)"""
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cv2 could not open {path}"
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


# --- start on detection -----------------------------------------------------------------

def test_starts_recording_on_detection(clip_cfg, clip_conn):
    rec = clips.ClipRecorder(clip_cfg, clip_conn)
    assert rec.recording is False
    rec.note_frame(_frame(), now=0.0, loop_fps=15.0)   # idle frame, nothing happens
    assert rec.recording is False

    rec.note_detection(now=0.1, detections=[FakeDetection(0.9)])
    assert rec.recording is True
    assert rec.clip_path is not None


def test_no_clip_without_detection(clip_cfg, clip_conn):
    """Frames alone never open a clip -- only a detection starts one."""
    rec = clips.ClipRecorder(clip_cfg, clip_conn)
    for i in range(5):
        rec.note_frame(_frame(), now=float(i), loop_fps=15.0)
    rec.finalize()
    assert rec.recording is False
    assert _count_clip_rows(clip_conn) == 0


# --- auto-stop after post-roll ----------------------------------------------------------

def test_auto_stops_after_post_roll(clip_cfg, clip_conn):
    """The clip ends once a frame arrives more than clip_post_roll_s after the last detection."""
    rec = clips.ClipRecorder(clip_cfg, clip_conn)   # post_roll = 1.0s
    rec.note_detection(now=0.0, detections=[FakeDetection(0.8)])
    rec.note_frame(_frame(), now=0.2, loop_fps=15.0)   # within post-roll -> keeps recording
    assert rec.recording is True
    rec.note_frame(_frame(), now=0.5, loop_fps=15.0)
    assert rec.recording is True
    # 1.5s since the last detection (> 1.0 post-roll) -> auto-finalize inside note_frame.
    rec.note_frame(_frame(), now=1.5, loop_fps=15.0)
    assert rec.recording is False
    assert _count_clip_rows(clip_conn) == 1


def test_detection_extends_the_clip(clip_cfg, clip_conn):
    """A fresh detection pushes last_det_t forward, so the post-roll window restarts."""
    rec = clips.ClipRecorder(clip_cfg, clip_conn)
    rec.note_detection(now=0.0, detections=[FakeDetection(0.8)])
    rec.note_frame(_frame(), now=0.8, loop_fps=15.0)
    rec.note_detection(now=0.9, detections=[FakeDetection(0.8)])   # extend
    # 1.5s after START but only 0.6s after the LAST detection -> still recording.
    rec.note_frame(_frame(), now=1.5, loop_fps=15.0)
    assert rec.recording is True
    rec.finalize()
    assert _count_clip_rows(clip_conn) == 1


# --- max-length cap ---------------------------------------------------------------------

def test_max_length_cap_finalizes(clip_cfg, clip_conn):
    """Continuous detections can't make an unbounded file: at clip_max_s the clip is force-cut,
    even though the animal is still being detected (last_det_t keeps advancing)."""
    rec = clips.ClipRecorder(clip_cfg, clip_conn)   # max_s = 5.0
    rec.note_detection(now=0.0, detections=[FakeDetection(0.8)])
    t = 0.0
    while t < 4.5:
        t += 0.5
        rec.note_detection(now=t, detections=[FakeDetection(0.8)])  # keep extending (no idle stop)
        rec.note_frame(_frame(), now=t, loop_fps=15.0)
        if not rec.recording:
            break
    # Once a frame lands past started_t + max_s (5.0), note_frame finalizes despite live detections.
    rec.note_detection(now=5.2, detections=[FakeDetection(0.8)])
    rec.note_frame(_frame(), now=5.2, loop_fps=15.0)
    assert rec.recording is False
    assert _count_clip_rows(clip_conn) == 1


# --- pre-roll buffering ------------------------------------------------------------------

def test_pre_roll_frames_are_prepended(clip_cfg, clip_conn):
    """With pre-roll on, frames seen BEFORE the first detection are buffered and written into the
    clip, so it opens on the animal arriving. The decoded frame count must exceed the number of
    live (post-detection) frames by the buffered pre-roll frames."""
    cfg = dataclasses.replace(clip_cfg, clip_pre_roll_s=3.0)
    rec = clips.ClipRecorder(cfg, clip_conn)

    # 4 pre-roll frames before any detection (these go into the ring).
    for i in range(4):
        rec.note_frame(_frame(), now=float(i) * 0.1, loop_fps=15.0)
    assert rec.recording is False
    assert len(rec.ring) == 4

    # Detection starts the clip -> the 4 buffered frames are written immediately.
    rec.note_detection(now=0.45, detections=[FakeDetection(0.9)])
    assert rec.frame_count == 4          # pre-roll dumped on start

    # 2 live frames, then stop via post-roll.
    rec.note_frame(_frame(), now=0.5, loop_fps=15.0)
    rec.note_frame(_frame(), now=0.6, loop_fps=15.0)
    rec.note_frame(_frame(), now=2.0, loop_fps=15.0)   # > 1.0s after last det -> finalize (also a frame)
    assert rec.recording is False

    # 4 pre-roll + 3 live = 7 frames written.
    assert rec.frame_count == 7
    clip_path = rec.clip_path
    assert _decode_frame_count(clip_path) == 7


# --- finalize writes a row AND a decodable file -----------------------------------------

def test_finalize_writes_db_row_and_decodable_mp4(clip_cfg, clip_conn):
    """The payoff: a finished clip leaves a clips row with the right metadata AND a real .mp4 that
    cv2 can reopen and decode frame-for-frame."""
    rec = clips.ClipRecorder(clip_cfg, clip_conn)   # pre_roll 0 -> frame count == live frames only
    rec.note_detection(now=0.0, detections=[FakeDetection(0.91)])
    n_live = 6
    for i in range(n_live):
        rec.note_detection(now=i * 0.1, detections=[FakeDetection(0.5)])  # keep alive, no idle stop
        rec.note_frame(_frame(), now=i * 0.1, loop_fps=15.0)
    clip_path = rec.clip_path
    rec.finalize()

    # DB row.
    assert _count_clip_rows(clip_conn) == 1
    row = clip_conn.execute("SELECT * FROM clips ORDER BY id DESC LIMIT 1").fetchone()
    # row is a tuple (clip_conn has no row_factory); pull by column index via cursor description.
    cols = [d[0] for d in clip_conn.execute("SELECT * FROM clips LIMIT 0").description]
    rec_row = dict(zip(cols, row))
    assert rec_row["source"] == clip_cfg.source
    assert rec_row["fps"] == 15.0
    assert (rec_row["width"], rec_row["height"]) == (160, 120)
    assert rec_row["frame_count"] == n_live
    assert rec_row["max_confidence"] == pytest.approx(0.91)   # best score across the clip
    assert rec_row["ended_at"] is not None                   # finalized, not cut off
    assert rec_row["clip_path"].endswith(".mp4")

    # The file itself is valid and decodes to exactly the frames written.
    assert clip_path.exists()
    assert _decode_frame_count(clip_path) == n_live


def test_finalize_is_safe_when_not_recording(clip_cfg, clip_conn):
    """finalize() is a no-op when idle and is safe to call repeatedly (shutdown calls it blindly)."""
    rec = clips.ClipRecorder(clip_cfg, clip_conn)
    rec.finalize()             # never recorded
    rec.finalize()             # again
    assert _count_clip_rows(clip_conn) == 0


def test_detection_count_and_max_conf_tracked(clip_cfg, clip_conn):
    """The clip tallies detector hits and the best confidence seen (a usability proxy)."""
    rec = clips.ClipRecorder(clip_cfg, clip_conn)
    rec.note_detection(now=0.0, detections=[FakeDetection(0.4), FakeDetection(0.7)])
    rec.note_frame(_frame(), now=0.1, loop_fps=15.0)
    rec.note_detection(now=0.2, detections=[FakeDetection(0.95)])
    rec.note_frame(_frame(), now=0.3, loop_fps=15.0)
    rec.finalize()
    row = clip_conn.execute(
        "SELECT detection_count, max_confidence FROM clips ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == 3                       # 2 + 1 detections
    assert row[1] == pytest.approx(0.95)
