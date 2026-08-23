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
import os
import shutil
import subprocess
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

import backup
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


# --- H.264 recording (browser-playable, via the ffmpeg pipe) ----------------------------

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_records_browser_playable_h264(clip_cfg, clip_conn):
    """With clip_codec='h264' the recorder pipes frames to ffmpeg and writes a real H.264 .mp4 --
    browser-playable, exact frame count, and still cv2-decodable for clipmotion."""
    cfg = dataclasses.replace(clip_cfg, clip_codec="h264")
    rec = clips.ClipRecorder(cfg, clip_conn)
    assert rec.use_ffmpeg is True
    rec.note_detection(now=0.0, detections=[FakeDetection(0.9)])
    n_live = 8
    for i in range(n_live):
        rec.note_detection(now=i * 0.1, detections=[FakeDetection(0.6)])  # keep alive
        rec.note_frame(_frame(), now=i * 0.1, loop_fps=15.0)
    clip_path = rec.clip_path
    rec.finalize()

    assert _count_clip_rows(clip_conn) == 1
    assert clip_path.exists() and clip_path.stat().st_size > 0
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "default=nw=1:nk=1", str(clip_path)],
        capture_output=True, text=True).stdout.strip()
    assert codec == "h264"                     # NOT mp4v -> plays in a browser <video>
    assert _decode_frame_count(clip_path) == n_live   # cv2 still reads it (clipmotion path)


def test_h264_falls_back_to_mp4v_without_ffmpeg(clip_cfg, clip_conn, monkeypatch):
    """If ffmpeg isn't available, clip_codec='h264' degrades to the cv2 mp4v writer rather than
    failing to record -- the dashboard transcodes those for playback."""
    monkeypatch.setattr(clips, "_FFMPEG", None)
    cfg = dataclasses.replace(clip_cfg, clip_codec="h264")
    rec = clips.ClipRecorder(cfg, clip_conn)
    assert rec.use_ffmpeg is False and rec.cv2_codec == "mp4v"
    rec.note_detection(now=0.0, detections=[FakeDetection(0.9)])
    for i in range(5):
        rec.note_detection(now=i * 0.1, detections=[FakeDetection(0.6)])
        rec.note_frame(_frame(), now=i * 0.1, loop_fps=15.0)
    rec.finalize()
    assert _count_clip_rows(clip_conn) == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg/ffprobe not on PATH")
def test_convert_legacy_mp4v_to_h264_in_place(clip_cfg, clip_conn):
    """convert_legacy_to_h264 re-encodes an mp4v clip to H.264 in place (same path), preserves the
    frame count, and is idempotent (a second pass skips the now-H.264 file)."""
    cfg = dataclasses.replace(clip_cfg, clip_codec="mp4v")
    rec = clips.ClipRecorder(cfg, clip_conn)
    assert rec.use_ffmpeg is False                         # records mp4v to start
    rec.note_detection(now=0.0, detections=[FakeDetection(0.9)])
    for i in range(6):
        rec.note_detection(now=i * 0.1, detections=[FakeDetection(0.6)])
        rec.note_frame(_frame(), now=i * 0.1, loop_fps=15.0)
    clip_path = rec.clip_path
    rec.finalize()

    def codec(p):
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "default=nw=1:nk=1", str(p)],
            capture_output=True, text=True).stdout.strip()

    assert codec(clip_path) != "h264"
    n_before = _decode_frame_count(clip_path)

    res = clips.convert_legacy_to_h264(cfg.clips_dir, verbose=False)
    assert res["converted"] == 1 and res["skipped"] == 0
    assert clip_path.exists() and codec(clip_path) == "h264"   # converted in place, same path
    assert _decode_frame_count(clip_path) == n_before          # frame count preserved

    res2 = clips.convert_legacy_to_h264(cfg.clips_dir, verbose=False)
    assert res2["converted"] == 0 and res2["skipped"] == 1     # idempotent


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


def test_bad_clip_codec_falls_back_to_mp4v(conn, tmp_path):
    """A non-4-character clip_codec (a config typo like 'vp9') must NOT reach
    cv2.VideoWriter_fourcc -- that raises a TypeError which, unguarded, crashed the whole capture
    rig on the first clip. The recorder coerces it to a safe 'mp4v' instead."""
    cfg = dataclasses.replace(config.CONFIG, clip_codec="vp9", clips_dir=tmp_path / "clips")
    rec = clips.ClipRecorder(cfg, conn)
    assert rec.cv2_codec == "mp4v"


# --- clip_scale must never hand the encoder odd dimensions ------------------------------

def test_clip_scale_output_dimensions_are_even(clip_cfg, clip_conn):
    """libx264 (yuv420p) refuses odd frame sizes, and 0.667 x 1920 = 1280.64 -- the old fx/fy
    resize rounded that to 1281 and ffmpeg died on the first frame of EVERY clip (2026-07-20,
    a full day of clips silently lost). _prep must round the scaled size down to even."""
    cfg = dataclasses.replace(clip_cfg, clip_scale=0.667)
    rec = clips.ClipRecorder(cfg, clip_conn)
    prepped = rec._prep(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert prepped.shape == (720, 1280, 3)          # not 1281 -- and both dimensions even

    # An awkward source size still comes out even (int(161*0.5)=80, int(121*0.5)=60).
    rec2 = clips.ClipRecorder(dataclasses.replace(clip_cfg, clip_scale=0.5), clip_conn)
    h, w = rec2._prep(np.zeros((121, 161, 3), dtype=np.uint8)).shape[:2]
    assert w % 2 == 0 and h % 2 == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_h264_records_at_1080p_with_clip_scale_0667(clip_cfg, clip_conn):
    """End-to-end regression for the 2026-07-20 outage: the LIVE geometry (1920x1080 capture,
    clip_scale 0.667, H.264 pipe) must produce a real, decodable clip -- not a silent drop."""
    cfg = dataclasses.replace(clip_cfg, clip_codec="h264", clip_scale=0.667)
    rec = clips.ClipRecorder(cfg, clip_conn)
    assert rec.use_ffmpeg is True
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    rec.note_detection(now=0.0, detections=[FakeDetection(0.9)])
    for i in range(6):
        rec.note_detection(now=i * 0.1, detections=[FakeDetection(0.6)])
        rec.note_frame(frame, now=i * 0.1, loop_fps=15.0)
    clip_path = rec.clip_path
    rec.finalize()

    assert rec.use_ffmpeg is True                  # the pipe survived -- no fallback needed
    assert _count_clip_rows(clip_conn) == 1
    row = clip_conn.execute(
        "SELECT width, height FROM clips ORDER BY id DESC LIMIT 1").fetchone()
    assert row == (1280, 720)                      # even dimensions, ~720p as configured
    assert clip_path.exists() and clip_path.stat().st_size > 0
    assert _decode_frame_count(clip_path) == 6


# --- a dying ffmpeg pipe must be caught, not silently counted into ----------------------

class _DyingPipeWriter:
    """Mimics ffmpeg's real first-write death (bad geometry): looks open until write() is
    first called, then reports dead. write() swallows everything, like the real pipe writer."""
    def __init__(self, path, fps, size):
        self.writes = 0

    def isOpened(self):
        return self.writes == 0

    def write(self, frame):
        self.writes += 1

    def release(self, timeout=None):
        pass


def test_dead_ffmpeg_pipe_falls_back_to_cv2_and_keeps_the_clip(clip_cfg, clip_conn,
                                                               monkeypatch, capsys):
    """If the ffmpeg pipe dies on the first write, _check_writer must rebuild the clip on the
    cv2 writer from the pre-roll ring -- LOUDLY -- instead of counting frames into a dead pipe
    and silently dropping the 0-byte file at finalize (the 2026-07-20 failure shape)."""
    monkeypatch.setattr(clips, "_FFMPEG", "ffmpeg-stub")          # "ffmpeg is available"
    monkeypatch.setattr(clips, "_FfmpegWriter", _DyingPipeWriter)  # ...but its pipe dies
    cfg = dataclasses.replace(clip_cfg, clip_codec="h264", clip_pre_roll_s=3.0)
    rec = clips.ClipRecorder(cfg, clip_conn)
    assert rec.use_ffmpeg is True

    for i in range(3):                             # pre-roll buffered before the visit
        rec.note_frame(_frame(), now=i * 0.1, loop_fps=15.0)
    rec.note_detection(now=0.35, detections=[FakeDetection(0.9)])
    assert rec.use_ffmpeg is False                 # fallback engaged at clip start
    assert rec.frame_count == 3                    # ring re-dumped into the cv2 writer
    rec.note_frame(_frame(), now=0.4, loop_fps=15.0)
    rec.note_frame(_frame(), now=0.5, loop_fps=15.0)
    clip_path = rec.clip_path
    rec.finalize()

    assert "ffmpeg pipe died" in capsys.readouterr().out
    assert _count_clip_rows(clip_conn) == 1        # the clip SURVIVED
    assert _decode_frame_count(clip_path) == 5     # 3 pre-roll + 2 live, all recovered


class _OpenButUselessWriter:
    """A writer that claims to be healthy but never puts a byte on disk -- the shape of a
    failure _check_writer can't see. finalize must then complain, not shrug."""
    def __init__(self, path, fps, size):
        pass

    def isOpened(self):
        return True

    def write(self, frame):
        pass

    def release(self, timeout=None):
        pass


def test_clip_dropped_with_no_file_is_loud(clip_cfg, clip_conn, monkeypatch, capsys):
    """The finalize branch that drops a frames-written-but-no-file clip used to be SILENT --
    which is how a completely dead recorder hid for a whole day. It must print."""
    monkeypatch.setattr(clips, "_FFMPEG", "ffmpeg-stub")
    monkeypatch.setattr(clips, "_FfmpegWriter", _OpenButUselessWriter)
    cfg = dataclasses.replace(clip_cfg, clip_codec="h264")
    rec = clips.ClipRecorder(cfg, clip_conn)
    rec.note_detection(now=0.0, detections=[FakeDetection(0.9)])
    rec.note_frame(_frame(), now=0.1, loop_fps=15.0)
    rec.finalize()

    assert "DROPPED" in capsys.readouterr().out
    assert _count_clip_rows(clip_conn) == 0        # no row for a clip that has no file


# --- soft prune: the derived re-ID/behaviour data outlives the video --------------------

def _stocked_clip(clip_cfg, conn, name, *, mtime, started_at, n_tracks=0, embed_first=False):
    """A fake on-disk clip (1 MiB, controlled mtime) + its DB row, optionally with sustained
    tracklets and an appearance vector on the first one. Returns the clips.id."""
    import os
    p = clip_cfg.clips_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * (1024 * 1024))
    os.utime(p, (mtime, mtime))
    cid = db.insert_clip(
        conn, source=db.SOURCE_GLASS_DOOR_CAM, clip_path=db.rel_to_root(p),
        started_at=started_at, ended_at=None, fps=10.0, width=160, height=120,
        frame_count=100, detection_count=5, max_confidence=0.9)
    if n_tracks:
        db.insert_clip_tracks(
            conn, clip_id=cid, model="m", n_samples=100,
            tracklets=[{"track_json": "[]", "n_hits": 40, "features": {}}] * n_tracks)
        if embed_first:
            tid = conn.execute("SELECT id FROM clip_tracks WHERE clip_id = ?", (cid,)).fetchone()[0]
            db.insert_clip_track_embedding(conn, track_id=tid, model="megadescriptor-l-384",
                                           dim=3, embedding=b"\0" * 12, n_frames=8)
    conn.commit()
    return cid


def test_prune_is_soft_and_derived_data_survives(clip_cfg, clip_conn):
    import sqlite3

    import stats
    clip_conn.row_factory = sqlite3.Row   # the shared conn fixture does this; clip_conn doesn't
    # Three 1 MiB clips, oldest first; a ~1.2 MiB budget forces the oldest TWO out.
    a = _stocked_clip(clip_cfg, clip_conn, "a.mp4", mtime=1_000, started_at="2026-06-01T21:00:00",
                      n_tracks=1, embed_first=True)      # mined: track + appearance vector
    b = _stocked_clip(clip_cfg, clip_conn, "b.mp4", mtime=2_000, started_at="2026-06-02T21:00:00",
                      n_tracks=1)                        # tracked but not yet embedded
    c = _stocked_clip(clip_cfg, clip_conn, "c.mp4", mtime=3_000, started_at="2026-06-03T21:00:00")
    cfg = dataclasses.replace(clip_cfg, clips_max_gb=1.2 * 1024 * 1024 / (1024 ** 3))

    assert clips.prune_clips(cfg, clip_conn) == 2
    assert not (clip_cfg.clips_dir / "a.mp4").exists()
    assert not (clip_cfg.clips_dir / "b.mp4").exists()
    assert (clip_cfg.clips_dir / "c.mp4").exists()

    # Every row survives; only the pruned ones carry the stamp.
    stamps = dict(clip_conn.execute("SELECT id, pruned_at FROM clips").fetchall())
    assert set(stamps) == {a, b, c}
    assert stamps[a] and stamps[b] and stamps[c] is None

    # The mined signal is intact -- this is the whole point of the soft prune.
    assert clip_conn.execute("SELECT COUNT(*) FROM clip_tracks WHERE clip_id = ?", (a,)).fetchone()[0] == 1
    assert clip_conn.execute(
        "SELECT COUNT(*) FROM clip_track_embeddings e JOIN clip_tracks t ON t.id = e.track_id "
        "WHERE t.clip_id = ?", (a,)).fetchone()[0] == 1
    assert len(db.load_clip_track_embeddings(clip_conn, "megadescriptor-l-384")) == 1

    # Work-lists that need the VIDEO skip pruned clips (b's unembedded track is unreachable now;
    # c still queues for extraction) -- and playback surfaces only offer what can actually play.
    assert [r["id"] for r in db.clips_needing_tracks(clip_conn, "m")] == [c]
    assert db.clip_tracks_needing_embedding(clip_conn, "megadescriptor-l-384", 30) == []
    assert ([r["clip_path"] for r in stats.load_clips(clip_conn)]
            == [db.rel_to_root(clip_cfg.clips_dir / "c.mp4").replace("\\", "/")])

    # A second pass finds the folder inside budget -- pruned ghosts can't loop.
    assert clips.prune_clips(cfg, clip_conn) == 0


def test_pruned_pair_clip_still_flags_co_presence(clip_cfg, clip_conn):
    import sqlite3

    import individuals
    clip_conn.row_factory = sqlite3.Row
    d = db.insert_detection(
        clip_conn, timestamp="2026-06-01T21:00:30-07:00", source=db.SOURCE_GLASS_DOOR_CAM,
        detection_class="animal", confidence=0.9, bbox=(0, 0, 10, 10), frame_w=100, frame_h=100,
        crop_path="crops/x.jpg", species="raccoon")
    vid = db.insert_visit(
        clip_conn, source=db.SOURCE_GLASS_DOOR_CAM, species="raccoon", individual_id=None,
        started_at="2026-06-01T21:00:00-07:00", ended_at="2026-06-01T21:10:00-07:00",
        detection_count=1, max_confidence=0.9, representative_detection_id=d)
    db.assign_visit(clip_conn, [d], vid)
    pair = _stocked_clip(clip_cfg, clip_conn, "pair.mp4", mtime=1_000,
                         started_at="2026-06-01T21:01:00-07:00", n_tracks=2)
    clip_conn.execute("UPDATE clips SET pruned_at = '2026-06-20T14:00:00-07:00' WHERE id = ?", (pair,))
    clip_conn.commit()

    # The co-presence SIGNAL must outlive the footage: the analytics join keeps pruned rows.
    assert individuals.clip_co_presence_by_visit(clip_conn, "raccoon") == {vid: 1}


# --- per-source prune budgets: an SD-card import must not evict the live rig ------------

def _sourced_clip(clip_cfg, conn, source, name, *, mtime, mib=1):
    """A fake on-disk clip under clips/<safe_source>/<name> with a controlled mtime + its row."""
    import os
    p = clip_cfg.clips_dir / clips._safe_source(source) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * (1024 * 1024 * mib))
    os.utime(p, (mtime, mtime))
    cid = db.insert_clip(
        conn, source=source, clip_path=db.rel_to_root(p), started_at="2026-07-27T21:00:00-07:00",
        ended_at=None, fps=30.0, width=1920, height=1080, frame_count=360,
        detection_count=1, max_confidence=0.9)
    conn.commit()
    return cid


def _mib_budget(n):
    return n * 1024 * 1024 / (1024 ** 3)


def test_prune_budgets_are_per_source(clip_cfg, clip_conn):
    """The regression this whole split exists for: importing a card's videos used to evict the
    live rig's clips, because one oldest-first budget covered the whole folder and trail-cam
    files carry OLDER mtimes than today's live footage."""
    live = [_sourced_clip(clip_cfg, clip_conn, db.SOURCE_GLASS_DOOR_CAM, f"live{i}.mp4",
                          mtime=9_000 + i) for i in range(3)]
    # Trail-cam clips are older (copy2 preserves the card's mtimes) -- under one shared,
    # oldest-first budget these would be *safe* and the live clips would be the ones deleted.
    card = [_sourced_clip(clip_cfg, clip_conn, db.SOURCE_TRAIL_CAM_SD, f"card{i}.mp4",
                          mtime=1_000 + i) for i in range(4)]
    cfg = dataclasses.replace(
        clip_cfg,
        clips_max_gb=_mib_budget(3),                              # live: 3 MiB, holds all 3
        clips_max_gb_by_source={db.SOURCE_TRAIL_CAM_SD: _mib_budget(2)},   # card: 2 MiB, sheds 2
        clips_irreplaceable_sources=(),   # this test is about the BUDGET split; the archive gate
    )                                     # is exercised on its own below

    assert clips.prune_clips(cfg, clip_conn) == 2
    # Every live clip survives even though they are the NEWEST files present ...
    for i in range(3):
        assert (clip_cfg.clips_dir / "glass_door_cam" / f"live{i}.mp4").exists()
    # ... and the card shed only its own two oldest.
    assert not (clip_cfg.clips_dir / "trail_cam_sd" / "card0.mp4").exists()
    assert not (clip_cfg.clips_dir / "trail_cam_sd" / "card1.mp4").exists()
    assert (clip_cfg.clips_dir / "trail_cam_sd" / "card2.mp4").exists()
    assert (clip_cfg.clips_dir / "trail_cam_sd" / "card3.mp4").exists()

    stamps = dict(clip_conn.execute("SELECT id, pruned_at FROM clips").fetchall())
    assert all(stamps[c] is None for c in live)
    assert stamps[card[0]] and stamps[card[1]]
    assert stamps[card[2]] is None and stamps[card[3]] is None
    assert clips.prune_clips(cfg, clip_conn) == 0        # stable second pass


def test_prune_zero_budget_source_is_never_pruned(clip_cfg, clip_conn):
    """A 0 budget means 'this source is archive-grade' -- the escape hatch for footage that
    cannot be re-recorded, and it must not fall back to the shared cap."""
    _sourced_clip(clip_cfg, clip_conn, db.SOURCE_TRAIL_CAM_SD, "keep.mp4", mtime=1_000, mib=4)
    cfg = dataclasses.replace(clip_cfg, clips_max_gb=_mib_budget(1),
                              clips_max_gb_by_source={db.SOURCE_TRAIL_CAM_SD: 0})
    assert clips.prune_clips(cfg, clip_conn) == 0
    assert (clip_cfg.clips_dir / "trail_cam_sd" / "keep.mp4").exists()


def test_prune_unbudgeted_source_uses_shared_cap(clip_cfg, clip_conn):
    """A source with no override still rolls on clips_max_gb -- the split adds a bucket, it
    doesn't quietly exempt everything that isn't listed."""
    for i in range(3):
        _sourced_clip(clip_cfg, clip_conn, db.SOURCE_GLASS_DOOR_CAM, f"g{i}.mp4", mtime=1_000 + i)
    cfg = dataclasses.replace(clip_cfg, clips_max_gb=_mib_budget(2),
                              clips_max_gb_by_source={db.SOURCE_TRAIL_CAM_SD: 99})
    assert clips.prune_clips(cfg, clip_conn) == 1
    assert not (clip_cfg.clips_dir / "glass_door_cam" / "g0.mp4").exists()


def test_prune_legacy_flat_clips_still_roll(clip_cfg, clip_conn):
    """Clips written before the per-source layout sit directly under clips/ with no source dir.
    They must land in the shared bucket, not be read as a source named 'a.mp4' (which would give
    every one of them its own private budget and stop pruning entirely)."""
    a = _stocked_clip(clip_cfg, clip_conn, "a.mp4", mtime=1_000, started_at="2026-06-01T21:00:00")
    _stocked_clip(clip_cfg, clip_conn, "b.mp4", mtime=2_000, started_at="2026-06-02T21:00:00")
    cfg = dataclasses.replace(clip_cfg, clips_max_gb=_mib_budget(1),
                              clips_max_gb_by_source={db.SOURCE_TRAIL_CAM_SD: 99})
    assert clips.prune_clips(cfg, clip_conn) == 1
    assert not (clip_cfg.clips_dir / "a.mp4").exists()
    assert dict(clip_conn.execute("SELECT id, pruned_at FROM clips").fetchall())[a]


# ---- The archive gate: the budget is a preference, "the only copy" is not ------------
# Until this shipped, protecting irreplaceable footage was a ritual -- remember --backup-first,
# keep the budget generous, remember backup.py runs weekly and skips today so the newest day
# always lags. A ritual is a thing you can forget once.

def _dated_clip(cfg, conn, source, day, name, *, mtime, mib=1):
    """A clip in the REAL on-disk layout: clips/<source>/<YYYY-MM-DD>/<file>.mp4."""
    d = cfg.clips_dir / source / day
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(bytes(mib * 1024 * 1024))
    os.utime(p, (mtime, mtime))
    cid = db.insert_clip(conn, source=source, clip_path=db.rel_to_root(p),
                         started_at=f"{day}T21:00:00-07:00", ended_at=f"{day}T21:00:05-07:00",
                         fps=10.0, width=64, height=36, frame_count=10,
                         detection_count=1, max_confidence=0.9)
    conn.commit()
    return cid, p


def test_prune_refuses_to_delete_the_only_copy(clip_cfg, clip_conn, tmp_path):
    """An irreplaceable source's clip is kept when its day-archive does not exist yet -- even
    though the budget says delete it. The card is formatted every cycle; this footage exists in
    exactly one place until backup.py has zipped it."""
    _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", "a.mp4", mtime=1_000, mib=2)
    dest = tmp_path / "drive"
    (dest / "clips").mkdir(parents=True)
    cfg = dataclasses.replace(clip_cfg, backup_dest=dest,
                              clips_max_gb_by_source={db.SOURCE_TRAIL_CAM_SD: _mib_budget(1)},
                              clips_irreplaceable_sources=("trail_cam_sd",))
    assert clips.prune_clips(cfg, clip_conn) == 0
    assert (clip_cfg.clips_dir / "trail_cam_sd" / "2026-08-01" / "a.mp4").exists()


def _member(p: Path) -> str:
    """The arcname backup.py stores a file under -- project-relative, posix separators."""
    return Path(db.rel_to_root(p)).as_posix()


@pytest.fixture
def archive_index(tmp_path, monkeypatch):
    """Point the prune guard's LOCAL archive index somewhere disposable, and hand back a writer.
    Without this a test would read the real rig's index out of the project root."""
    idx = tmp_path / "archive-index"
    monkeypatch.setattr(backup, "ARCHIVE_INDEX_DIR", idx)

    def write(stem: str, holds: dict) -> None:
        backup.write_archive_index(idx, stem, holds)
    return write


def _guarded_cfg(clip_cfg, dest):
    return dataclasses.replace(clip_cfg, backup_dest=dest,
                               clips_max_gb_by_source={db.SOURCE_TRAIL_CAM_SD: _mib_budget(1)},
                               clips_irreplaceable_sources=("trail_cam_sd",))


def test_prune_proceeds_once_the_archive_actually_holds_the_clip(
        clip_cfg, clip_conn, tmp_path, archive_index):
    _, p = _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", "a.mp4",
                       mtime=1_000, mib=2)
    dest = tmp_path / "drive"
    (dest / "clips").mkdir(parents=True)
    zip_name = "clips-trail_cam_sd-2026-08-01.zip"
    (dest / "clips" / zip_name).write_bytes(b"PK" + bytes([5, 6]) + bytes(18))
    archive_index("clips-trail_cam_sd-2026-08-01", {_member(p): zip_name})

    assert clips.prune_clips(_guarded_cfg(clip_cfg, dest), clip_conn) == 1
    assert not p.exists()


def test_a_clip_backfilled_into_an_already_archived_day_is_held(
        clip_cfg, clip_conn, tmp_path, archive_index):
    """THE hole this closes. The trail-cam card is dumped and goes straight back in the camera, so
    its dump day arrives in two batches -- the second lands in a date that was archived days ago.
    The guard used to ask only whether that day had a zip, which was true, and the budget was free
    to delete footage that had never been inside one. The card has since been formatted."""
    _, archived = _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", "a.mp4",
                              mtime=1_000, mib=2)
    _, backfilled = _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", "b.mp4",
                                mtime=2_000, mib=2)
    dest = tmp_path / "drive"
    (dest / "clips").mkdir(parents=True)
    zip_name = "clips-trail_cam_sd-2026-08-01.zip"
    (dest / "clips" / zip_name).write_bytes(b"PK" + bytes([5, 6]) + bytes(18))
    archive_index("clips-trail_cam_sd-2026-08-01", {_member(archived): zip_name})

    assert clips.prune_clips(_guarded_cfg(clip_cfg, dest), clip_conn) == 1
    assert not archived.exists(), "the archived clip should still be prunable"
    assert backfilled.exists(), "a clip no archive holds must survive the budget"


def test_prune_holds_a_day_with_no_index_at_all(clip_cfg, clip_conn, tmp_path, archive_index):
    """A fresh clone or a restored machine has archives but no local index yet. "I cannot prove
    it" keeps the footage and says so; one backup.py run rebuilds the index."""
    _, p = _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", "a.mp4",
                       mtime=1_000, mib=2)
    dest = tmp_path / "drive"
    (dest / "clips").mkdir(parents=True)
    (dest / "clips" / "clips-trail_cam_sd-2026-08-01.zip").write_bytes(
        b"PK" + bytes([5, 6]) + bytes(18))

    assert clips.prune_clips(_guarded_cfg(clip_cfg, dest), clip_conn) == 0
    assert p.exists()


def test_prune_holds_when_the_archive_the_index_names_has_gone_missing(
        clip_cfg, clip_conn, tmp_path, archive_index):
    """A zip written on 08-21 was simply GONE from Drive the next day. The index says which
    archive holds a clip; that archive still has to be there."""
    _, p = _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", "a.mp4",
                       mtime=1_000, mib=2)
    dest = tmp_path / "drive"
    (dest / "clips").mkdir(parents=True)
    archive_index("clips-trail_cam_sd-2026-08-01",
                  {_member(p): "clips-trail_cam_sd-2026-08-01.zip"})   # ...but no such file

    assert clips.prune_clips(_guarded_cfg(clip_cfg, dest), clip_conn) == 0
    assert p.exists()


def test_prune_proceeds_for_a_clip_archived_in_a_later_part(
        clip_cfg, clip_conn, tmp_path, archive_index):
    """A day is a set of archives; the guard must accept whichever part holds the clip."""
    _, p = _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", "b.mp4",
                       mtime=2_000, mib=2)
    dest = tmp_path / "drive"
    (dest / "clips").mkdir(parents=True)
    part2 = "clips-trail_cam_sd-2026-08-01.part2.zip"
    (dest / "clips" / part2).write_bytes(b"PK" + bytes([5, 6]) + bytes(18))
    archive_index("clips-trail_cam_sd-2026-08-01", {_member(p): part2})

    assert clips.prune_clips(_guarded_cfg(clip_cfg, dest), clip_conn) == 1
    assert not p.exists()


def test_the_day_index_is_read_once_per_day_not_once_per_clip(
        clip_cfg, clip_conn, tmp_path, archive_index, monkeypatch):
    """The reason this is answered from a local index at all is cost, so the cost is pinned. Even
    a local read per CLIP would be wasteful; against the cloud archives themselves it would be
    catastrophic -- opening a zip's central directory on Drive materialises the whole file, and a
    backup dry-run reading ~185 of them grew Drive's cache from 10 to 86 GiB."""
    for i in range(3):
        _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", f"c{i}.mp4",
                    mtime=1_000 + i, mib=2)
    dest = tmp_path / "drive"
    (dest / "clips").mkdir(parents=True)
    reads: list[str] = []
    real = backup.read_archive_index
    monkeypatch.setattr(backup, "read_archive_index",
                        lambda d, stem: (reads.append(stem), real(d, stem))[1])

    clips.prune_clips(_guarded_cfg(clip_cfg, dest), clip_conn)      # all three examined, all held

    assert reads == ["clips-trail_cam_sd-2026-08-01"]


def test_backup_and_the_prune_guard_agree_end_to_end(
        clip_cfg, clip_conn, tmp_path, monkeypatch):
    """The index is a contract BETWEEN two modules, so this drives both real functions rather than
    hand-writing one side: backup.archive_media records what its archives hold, and the guard
    answers out of that record. A hand-written index would pin the contract to itself."""
    monkeypatch.setattr(backup, "ROOT", tmp_path)
    monkeypatch.setattr(db, "_ROOT", tmp_path)
    idx = tmp_path / "archive-index"
    monkeypatch.setattr(backup, "ARCHIVE_INDEX_DIR", idx)

    stamp = 1_785_000_000                       # a real 2026 mtime: zipfile refuses pre-1980
    _, archived = _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", "a.mp4",
                              mtime=stamp, mib=2)
    dest = tmp_path / "drive"
    backup.archive_media(clip_cfg.clips_dir, dest / "clips", date(2026, 8, 2),
                               dry_run=False, index_dir=idx)
    # Imported after that archive was sealed -- the two-batch dump day, for real this time.
    _, backfilled = _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", "b.mp4",
                                mtime=stamp + 60, mib=2)

    assert clips.prune_clips(_guarded_cfg(clip_cfg, dest), clip_conn) == 1
    assert not archived.exists()
    assert backfilled.exists()

    # ...and once backup.py has caught up, it becomes prunable like anything else.
    backup.archive_media(clip_cfg.clips_dir, dest / "clips", date(2026, 8, 2),
                               dry_run=False, index_dir=idx)
    assert clips.prune_clips(_guarded_cfg(clip_cfg, dest), clip_conn) == 1
    assert not backfilled.exists()


def test_prune_gate_fails_closed_without_a_backup_destination(clip_cfg, clip_conn):
    """No drive configured, an unplugged drive, an unrecognised layout: all mean "I cannot prove
    a copy exists", and all keep the footage. A full disk is a problem you can see and fix."""
    _dated_clip(clip_cfg, clip_conn, "trail_cam_sd", "2026-08-01", "a.mp4", mtime=1_000, mib=2)
    cfg = dataclasses.replace(clip_cfg, backup_dest=None,
                              clips_max_gb_by_source={db.SOURCE_TRAIL_CAM_SD: _mib_budget(1)},
                              clips_irreplaceable_sources=("trail_cam_sd",))
    assert clips.prune_clips(cfg, clip_conn) == 0


def test_prune_gate_only_guards_the_sources_named_irreplaceable(clip_cfg, clip_conn, tmp_path):
    """The live rig can re-record tomorrow, so its rolling window must keep rolling whatever the
    backup drive is doing -- otherwise an unplugged drive quietly fills the disk."""
    _dated_clip(clip_cfg, clip_conn, "glass_door_cam", "2026-08-01", "a.mp4", mtime=1_000, mib=2)
    cfg = dataclasses.replace(clip_cfg, backup_dest=tmp_path / "nowhere",
                              clips_max_gb=_mib_budget(1),
                              clips_irreplaceable_sources=("trail_cam_sd",))
    assert clips.prune_clips(cfg, clip_conn) == 1


# --------------------------------------------------------------------------- codec reporting
# 2026-08-21: ffmpeg vanished off PATH and the rig recorded a full day of mp4v unnoticed, because
# the only signal was a warning printed solely in the broken case. These pin the always-printed
# replacement: what the recorder will ACTUALLY write, whatever the config asked for.

def test_effective_codec_reports_h264_when_ffmpeg_is_present(clip_cfg, monkeypatch):
    monkeypatch.setattr(clips, "_FFMPEG", "/usr/bin/ffmpeg")
    cfg = dataclasses.replace(clip_cfg, clip_codec="h264")
    assert clips.effective_codec(cfg) == "H.264"
    assert clips.ffmpeg_missing(cfg) is False


def test_effective_codec_reports_the_mp4v_fallback_when_ffmpeg_is_gone(clip_cfg, monkeypatch):
    """The silent-degradation case: config still says h264, but what lands on disk is mp4v."""
    monkeypatch.setattr(clips, "_FFMPEG", None)
    cfg = dataclasses.replace(clip_cfg, clip_codec="h264")
    assert clips.effective_codec(cfg) == "mp4v"
    assert clips.ffmpeg_missing(cfg) is True


def test_ffmpeg_missing_is_false_when_mp4v_was_actually_asked_for(clip_cfg, monkeypatch):
    """Deliberately choosing the cv2 writer is not a degradation -- don't cry wolf about it."""
    monkeypatch.setattr(clips, "_FFMPEG", None)
    cfg = dataclasses.replace(clip_cfg, clip_codec="mp4v")
    assert clips.effective_codec(cfg) == "mp4v"
    assert clips.ffmpeg_missing(cfg) is False


def test_effective_codec_mirrors_the_recorders_fourcc_length_fallback(clip_cfg, monkeypatch):
    """A 3-char codec makes ClipRecorder fall back to mp4v; the banner must say mp4v too, or it
    would report a codec that never gets written."""
    monkeypatch.setattr(clips, "_FFMPEG", "/usr/bin/ffmpeg")
    assert clips.effective_codec(dataclasses.replace(clip_cfg, clip_codec="vp9")) == "mp4v"


def test_missing_ffmpeg_is_announced_once_per_process_not_once_per_camera(
        clip_cfg, clip_conn, capsys, monkeypatch):
    """Three cameras used to mean three identical warnings per restart -- noise, not signal."""
    monkeypatch.setattr(clips, "_FFMPEG", None)
    monkeypatch.setattr(clips, "_WARNED_NO_FFMPEG", False)
    cfg = dataclasses.replace(clip_cfg, clip_codec="h264")
    for source in ("glass_door_cam", "yard_ir", "cam02"):
        clips.ClipRecorder(cfg, clip_conn, source=source)
    assert capsys.readouterr().out.count("ffmpeg not found on PATH") == 1
