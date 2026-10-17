"""
Tests for capwatch: the read deadline, the per-camera heartbeat, and the capture watch that
writes the status file rigwatch reads.

A capture thread can stop inside one call that never returns, with the process up and the
dashboard answering. These tests stand in for that with fake captures whose read() blocks on an
Event, a real thread parked on an Event (so the reported stack is a real one), and a fake clock.
No camera, network or GPU.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

import capwatch


class _Cap:
    """A capture stand-in: read() returns frames, raises, or blocks until `unblock` is set."""

    def __init__(self, frames=(), block_after=None, raise_on_read=False):
        self.frames = list(frames)
        self.block_after = block_after          # reads served before the next one blocks
        self.raise_on_read = raise_on_read
        self.unblock = threading.Event()
        self.reading = threading.Event()        # set while a read is parked
        self.reads = 0
        self.released = threading.Event()
        self.settings = []

    def read(self):
        if self.raise_on_read:
            raise RuntimeError("decoder fell over")
        if self.block_after is not None and self.reads >= self.block_after:
            self.reading.set()
            self.unblock.wait(10)
            return True, "late frame"
        self.reads += 1
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def set(self, prop, val):
        self.settings.append((prop, val))
        return True

    def get(self, prop):
        return 42.0

    def isOpened(self):
        return True

    def release(self):
        self.released.set()


# ---- DeadlineReader ---------------------------------------------------------------------

def test_frames_pass_straight_through_and_so_do_set_and_get():
    cap = _Cap(frames=["a", "b"])
    r = capwatch.DeadlineReader(cap, timeout_s=5.0)
    assert r.read() == (True, "a")
    assert r.read() == (True, "b")
    assert r.read() == (False, None)            # an honest failed read is not a timeout
    assert r.timed_out is False
    assert r.set(5, 1.0) is True and cap.settings == [(5, 1.0)]
    assert r.get(5) == 42.0
    r.release()
    assert cap.released.is_set()


def test_a_read_that_never_returns_is_abandoned_at_the_deadline():
    cap = _Cap(frames=["a"], block_after=1)
    r = capwatch.DeadlineReader(cap, timeout_s=0.2)
    assert r.read() == (True, "a")
    t0 = time.monotonic()
    assert r.read() == (False, None)
    assert 0.15 <= time.monotonic() - t0 < 3.0
    assert r.timed_out is True and r.isOpened() is False
    t0 = time.monotonic()
    assert r.read() == (False, None)            # an abandoned capture fails at once
    assert time.monotonic() - t0 < 0.1
    cap.unblock.set()                           # let the parked helper finish


def test_release_of_a_stuck_read_is_left_to_the_helper_until_the_read_returns():
    cap = _Cap(block_after=0)
    r = capwatch.DeadlineReader(cap, timeout_s=0.1)
    assert r.read() == (False, None) and r.timed_out
    r.release()
    # Releasing an FFmpeg capture from another thread while it is mid-read is how you crash
    # the process, so the capture thread must not do it.
    assert not cap.released.wait(0.2)
    cap.unblock.set()                           # the stuck read finally comes back ...
    assert cap.released.wait(2.0)               # ... and the helper releases it then
    r._thread.join(2.0)
    assert not r._thread.is_alive()


def test_a_raising_read_is_a_failed_read_not_a_dead_helper():
    cap = _Cap(raise_on_read=True)
    r = capwatch.DeadlineReader(cap, timeout_s=2.0)
    assert r.read() == (False, None)
    assert r.timed_out is False
    assert r.read() == (False, None)            # the helper is still serving
    r.release()


def test_release_of_an_idle_reader_releases_once_and_stops_the_helper():
    cap = _Cap(frames=["a"])
    r = capwatch.DeadlineReader(cap, timeout_s=2.0)
    r.read()
    r.release()
    r.release()                                 # idempotent
    assert cap.released.is_set()
    r._thread.join(2.0)
    assert not r._thread.is_alive()
    assert r.read() == (False, None)


# ---- CaptureBeat + CaptureWatch ---------------------------------------------------------

class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return _Clock()


def _watch(tmp_path, beats, clock, stall_s=120.0):
    said, ledger = [], []
    w = capwatch.CaptureWatch(beats, stall_s, path=tmp_path / ".capture_status.json",
                              log=said.append, coverage=lambda *a: ledger.append(a),
                              clock=clock, wall=lambda: 1_760_000_000.0 + clock.t)
    return w, said, ledger


def _status(tmp_path):
    return json.loads((tmp_path / ".capture_status.json").read_text(encoding="utf-8"))


def test_a_camera_delivering_frames_is_up_and_the_watch_says_nothing(tmp_path, clock):
    beat = capwatch.CaptureBeat("glass_door_cam", clock=clock)
    w, said, ledger = _watch(tmp_path, {"glass_door_cam": beat}, clock)
    for _ in range(3):
        clock.t += 0.1
        beat.frame()
    clock.t += 1.0
    w.check()
    cam = _status(tmp_path)["cameras"]["glass_door_cam"]
    assert cam["state"] == "up" and cam["stuck"] is False and cam["where"] is None
    assert cam["frames"] == 3 and cam["frame_age_s"] == pytest.approx(1.0)
    assert said == [] and ledger == []


def test_a_thread_that_stops_moving_is_reported_once_with_where_it_is_stuck(tmp_path, clock):
    beat = capwatch.CaptureBeat("glass_door_cam", clock=clock)
    parked, release = threading.Event(), threading.Event()

    def stuck_in_a_read_that_never_returns():
        beat.bind()
        beat.frame()
        parked.set()
        release.wait(10)

    t = threading.Thread(target=stuck_in_a_read_that_never_returns, daemon=True)
    t.start()
    assert parked.wait(2.0)
    try:
        w, said, ledger = _watch(tmp_path, {"glass_door_cam": beat}, clock)
        clock.t += 60.0
        w.check()
        assert said == [] and _status(tmp_path)["cameras"]["glass_door_cam"]["stuck"] is False

        clock.t += 61.0                         # 121 s without a frame or a reconnect attempt
        w.check()
        assert len(said) == 1
        assert "CAPTURE STALLED" in said[0] and "glass_door_cam" in said[0]
        assert "stuck_in_a_read_that_never_returns" in said[0]   # the stack names the culprit
        assert ledger == [("glass_door_cam", "down", "stalled")]
        cam = _status(tmp_path)["cameras"]["glass_door_cam"]
        assert cam["stuck"] is True and cam["stuck_s"] == pytest.approx(121.0)
        assert "stuck_in_a_read_that_never_returns" in cam["where"]

        clock.t += 300.0
        w.check()                               # still stuck: no second report
        assert len(said) == 1 and len(ledger) == 1

        beat.frame()                            # it came back
        w.check()
        assert len(said) == 2 and "moving again" in said[1]
        assert ledger[-1] == ("glass_door_cam", "up", "unstalled")
        assert _status(tmp_path)["cameras"]["glass_door_cam"]["stuck"] is False
    finally:
        release.set()
        t.join(2.0)


def test_the_stall_clock_starts_when_the_capture_thread_does(tmp_path, clock):
    # First run: the detector download can take minutes before any camera thread exists.
    beat = capwatch.CaptureBeat("glass_door_cam", clock=clock)
    clock.t += 300.0
    beat.bind()
    w, said, _ = _watch(tmp_path, {"glass_door_cam": beat}, clock)
    clock.t += 10.0
    w.check()
    assert said == [] and _status(tmp_path)["cameras"]["glass_door_cam"]["stuck"] is False


def test_a_camera_that_is_away_is_reconnecting_not_stuck(tmp_path, clock):
    beat = capwatch.CaptureBeat("glass_door_cam", clock=clock)
    beat.frame()
    w, said, _ = _watch(tmp_path, {"glass_door_cam": beat}, clock)
    for _ in range(20):                         # ten minutes of reopen attempts, 30 s apart
        clock.t += 30.0
        beat.tick("reconnecting")
    w.check()
    cam = _status(tmp_path)["cameras"]["glass_door_cam"]
    assert cam["state"] == "reconnecting" and cam["stuck"] is False
    assert cam["frame_age_s"] == pytest.approx(600.0)    # rigwatch's call: dark, camera down
    assert cam["reconnects"] == 1                         # one outage, many attempts
    assert said == []


def test_a_camera_that_never_delivered_ages_from_when_its_thread_started(tmp_path, clock):
    beat = capwatch.CaptureBeat("yard_ir", clock=clock)
    w, _, _ = _watch(tmp_path, {"yard_ir": beat}, clock)
    clock.t += 45.0
    w.check()
    cam = _status(tmp_path)["cameras"]["yard_ir"]
    assert cam["last_frame_ts"] is None and cam["frame_age_s"] == pytest.approx(45.0)


def test_an_exited_thread_is_stopped_not_stuck(tmp_path, clock):
    beat = capwatch.CaptureBeat("glass_door_cam", clock=clock)
    beat.frame()
    beat.tick("stopped")
    w, said, _ = _watch(tmp_path, {"glass_door_cam": beat}, clock)
    clock.t += 3600.0
    w.check()
    cam = _status(tmp_path)["cameras"]["glass_door_cam"]
    assert cam["state"] == "stopped" and cam["stuck"] is False
    assert said == []


def test_the_status_file_carries_the_rig_pid_and_a_fresh_stamp(tmp_path, clock):
    import os
    w, _, _ = _watch(tmp_path, {}, clock)
    w.check()
    st = _status(tmp_path)
    assert st["pid"] == os.getpid()
    assert st["ts"] == pytest.approx(1_760_000_000.0 + clock.t)
    assert st["stall_s"] == 120.0 and st["cameras"] == {}


def test_the_watch_survives_a_broken_log_a_broken_ledger_and_an_unwritable_file(tmp_path, clock):
    beat = capwatch.CaptureBeat("glass_door_cam", clock=clock)

    def boom(*a):
        raise OSError("disk full")

    w = capwatch.CaptureWatch({"glass_door_cam": beat}, 1.0,
                              path=tmp_path / "no-such-dir" / "status.json",
                              log=boom, coverage=boom, clock=clock)
    clock.t += 5.0
    status = w.check()                          # nothing raises
    assert status["cameras"]["glass_door_cam"]["stuck"] is True


def test_the_watch_thread_writes_until_stopped_and_once_more_on_the_way_out(tmp_path):
    beat = capwatch.CaptureBeat("glass_door_cam")
    path = tmp_path / ".capture_status.json"
    w = capwatch.CaptureWatch({"glass_door_cam": beat}, 120.0, path=path, log=lambda m: None)
    stop = threading.Event()
    t = threading.Thread(target=w.run, args=(stop, 0.01), daemon=True)
    t.start()
    deadline = time.monotonic() + 2.0
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()
    beat.tick("stopped")
    stop.set()
    t.join(2.0)
    assert not t.is_alive()
    assert json.loads(path.read_text(encoding="utf-8"))["cameras"]["glass_door_cam"]["state"] == "stopped"
