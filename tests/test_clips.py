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
import shutil
import subprocess

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
