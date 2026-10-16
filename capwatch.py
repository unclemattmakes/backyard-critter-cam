"""Is the rig still watching the yard? Per-camera frame heartbeats, a read with a deadline, and
the status file rigwatch reads.

A process that is up, with a dashboard that answers, says nothing about whether frames are still
arriving. A capture thread can stop inside one call that never returns (a network read, a GPU
call, a console write) without raising, and the rest of the rig carries on. So:

  DeadlineReader  runs a network camera's read() on a helper thread, so a read that never
                  returns costs capture_read_timeout_s and a reconnect, not the evening.
  CaptureBeat     one per camera. The capture thread stamps every good frame, every loop pass
                  and every reconnect attempt.
  CaptureWatch    its own thread. Every few seconds it reads the beats, writes
                  config.CAPTURE_STATUS_FILE, and when a camera has made no progress for
                  capture_stall_s it logs the stuck thread's stack, so a stall names its own cause.

Stdlib only, and nothing here touches a camera: the tests drive it with fake readers.
"""
from __future__ import annotations

import json
import os
import sys
import sysconfig
import threading
import time
import traceback
from pathlib import Path

import config

WATCH_INTERVAL_S = 5.0          # how often the status file is rewritten
STACK_DEPTH = 20                # innermost frames logged for a stuck thread (torch nests deep)


class DeadlineReader:
    """A capture whose read() gives up after `timeout_s`.

    The real read runs on one helper thread, one read at a time, and read() waits for it, so the
    capture thread and the helper never use the capture at the same moment (OpenCV captures are
    not thread-safe). A read that misses its deadline abandons the capture: read() returns
    (False, None) with `timed_out` set, later reads fail at once, and the real release is left to
    the helper, which does it if the stuck read ever returns. set/get pass straight through."""

    def __init__(self, cap, timeout_s: float, name: str = "capture-read"):
        self._cap = cap
        self.timeout_s = float(timeout_s)
        self.timed_out = False
        self._lock = threading.Lock()
        self._go = threading.Event()
        self._done = threading.Event()
        self._result = (False, None)
        self._busy = False              # the helper is inside the real read
        self._closed = False
        self._released = False
        self._thread = threading.Thread(target=self._serve, name=name, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            self._go.wait()
            self._go.clear()
            with self._lock:
                if self._closed and not self._busy:
                    return
            try:
                result = self._cap.read()
            except Exception:           # noqa: BLE001 -- a raising read is a failed read
                result = (False, None)
            with self._lock:
                self._result = result
                self._busy = False
                orphaned = self._closed
            self._done.set()
            if orphaned:                # abandoned mid-read: the release is ours now
                self._release_real()
                return

    def read(self):
        with self._lock:
            if self._closed:
                return False, None
            self._busy = True
            self._done.clear()
        self._go.set()
        if not self._done.wait(self.timeout_s):
            with self._lock:
                if self._busy:
                    self.timed_out = True
                    self._closed = True
                    return False, None
        with self._lock:
            result, self._result = self._result, (False, None)
        return result

    def release(self) -> None:
        with self._lock:
            busy = self._busy
            self._closed = True
        if busy:
            return                      # the helper releases it when (if) its read returns
        self._go.set()                  # let an idle helper exit
        self._release_real()

    def _release_real(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        try:
            self._cap.release()
        except Exception:               # noqa: BLE001 -- releasing a dead stream is best-effort
            pass

    def isOpened(self) -> bool:
        return not self._closed and bool(self._cap.isOpened())

    def __getattr__(self, name):
        return getattr(self._cap, name)


class CaptureBeat:
    """One camera's heartbeat: stamped by its capture thread, read by CaptureWatch.

    `progress` moves on every loop pass and every reconnect attempt; `last_frame` only on a good
    frame. The gap between them is the diagnosis: frames stopped but progress still moving is a
    camera that is away (the rig is reconnecting); both stopped is a capture thread that is stuck."""

    def __init__(self, source: str, clock=time.monotonic, wall=time.time):
        self.source = source
        self._clock, self._wall = clock, wall
        self._lock = threading.Lock()
        now = clock()
        self.state = "opening"          # opening | up | reconnecting | stopped
        self.thread_id: int | None = None
        self.started = now
        self.progress = now
        self.last_frame: float | None = None
        self.last_frame_wall: float | None = None
        self.frames = 0
        self.reconnects = 0
        self.abandoned_reads = 0

    def bind(self) -> None:
        """Call from the capture thread itself, so a stall can show that thread's stack. Counts
        as progress: the clock starts when the thread does, not when the beat was built."""
        with self._lock:
            self.thread_id = threading.get_ident()
            self.progress = self._clock()

    def frame(self) -> None:
        now = self._clock()
        with self._lock:
            self.progress = self.last_frame = now
            self.last_frame_wall = self._wall()
            self.frames += 1
            self.state = "up"

    def tick(self, state: str | None = None) -> None:
        with self._lock:
            self.progress = self._clock()
            if state is not None:
                if state == "reconnecting" and self.state != "reconnecting":
                    self.reconnects += 1
                self.state = state

    def abandoned(self) -> None:
        with self._lock:
            self.abandoned_reads += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": self.state, "thread_id": self.thread_id, "started": self.started,
                    "progress": self.progress, "last_frame": self.last_frame,
                    "last_frame_wall": self.last_frame_wall, "frames": self.frames,
                    "reconnects": self.reconnects, "abandoned_reads": self.abandoned_reads}


_LIBRARY_DIRS = tuple(os.path.normcase(str(Path(p))) for p in {
    sysconfig.get_paths()[k] for k in ("stdlib", "platstdlib", "purelib", "platlib")})


def thread_stack(thread_id, frames=sys._current_frames) -> list[traceback.FrameSummary]:
    """A live thread's stack, oldest frame first. [] if the thread is gone."""
    frame = frames().get(thread_id) if thread_id is not None else None
    return list(traceback.extract_stack(frame)) if frame is not None else []


def format_frame(fs: traceback.FrameSummary) -> str:
    text = f"{Path(fs.filename).name}:{fs.lineno} in {fs.name}"
    return f"{text}: {fs.line.strip()}" if fs.line else text


def where_stuck(stack: list[traceback.FrameSummary]) -> str | None:
    """The innermost frame of the rig's own code: the line that made the call that has not
    returned. A stack parked in threading.py or torch says little until you know who called it."""
    for fs in reversed(stack):
        if not os.path.normcase(str(Path(fs.filename))).startswith(_LIBRARY_DIRS):
            return format_frame(fs)
    return format_frame(stack[-1]) if stack else None


class CaptureWatch:
    """Reads every camera's beat, writes the status file, and reports a stuck capture thread.

    `log` gets one multi-line message per stall and one per recovery, never one per check.
    `coverage(source, event, reason)`, when given, records the stall in the coverage ledger, which
    otherwise reads "up" for as long as a stuck thread sits there."""

    def __init__(self, beats: dict, stall_s: float, *, path: Path | None = None, log=print,
                 coverage=None, clock=time.monotonic, wall=time.time,
                 frames=sys._current_frames):
        self.beats = beats
        self.stall_s = float(stall_s)
        self.path = Path(path) if path is not None else config.CAPTURE_STATUS_FILE
        self._log, self._coverage = log, coverage
        self._clock, self._wall, self._frames = clock, wall, frames
        self.stuck: set[str] = set()            # sources already reported as stuck

    def check(self) -> dict:
        now, wall = self._clock(), self._wall()
        cams = {}
        for source, beat in list(self.beats.items()):
            s = beat.snapshot()
            idle = now - s["progress"]
            since_frame = now - (s["last_frame"] if s["last_frame"] is not None else s["started"])
            stuck = s["state"] != "stopped" and idle > self.stall_s
            stack = thread_stack(s["thread_id"], self._frames) if stuck else []
            where = where_stuck(stack)
            if stuck and source not in self.stuck:
                self.stuck.add(source)
                self._report_stall(source, s, idle, since_frame, stack)
            elif not stuck and source in self.stuck:
                self.stuck.discard(source)
                self._say(f"[{source}] capture is moving again (state '{s['state']}').")
                self._note_coverage(source, "up", "unstalled")
            cams[source] = {
                "state": s["state"],
                "last_frame_ts": s["last_frame_wall"],
                "frame_age_s": round(since_frame, 1),
                "frames": s["frames"],
                "reconnects": s["reconnects"],
                "abandoned_reads": s["abandoned_reads"],
                "stuck": stuck,
                "stuck_s": round(idle, 1) if stuck else None,
                "where": where,
            }
        status = {"ts": wall, "pid": os.getpid(), "stall_s": self.stall_s, "cameras": cams}
        self._write(status)
        return status

    def run(self, stop_event, interval_s: float = WATCH_INTERVAL_S) -> None:
        """The watch thread: check, then wait, until stop_event; one last write on the way out."""
        while True:
            self._check_quietly()
            if stop_event.wait(interval_s):
                break
        self._check_quietly()

    def _check_quietly(self) -> None:
        try:
            self.check()
        except Exception as e:          # noqa: BLE001 -- the watch must outlive its own bugs
            self._say(f"[capture-watch] check failed: {type(e).__name__}: {e}")

    def _report_stall(self, source, s, idle, since_frame, stack) -> None:
        lines = "\n".join(f"    {format_frame(fs)}" for fs in stack[-STACK_DEPTH:])
        self._say(f"[{source}] CAPTURE STALLED: no frame for {since_frame:.0f}s and the capture "
                  f"thread has made no progress for {idle:.0f}s (state '{s['state']}'). The rig "
                  f"is up but this camera is not being watched. Stuck at "
                  f"{where_stuck(stack) or 'an unknown place (the thread is gone)'}; its stack, "
                  f"innermost last:\n{lines or '    (none)'}")
        self._note_coverage(source, "down", "stalled")

    def _note_coverage(self, source, event, reason) -> None:
        if self._coverage is None:
            return
        try:
            self._coverage(source, event, reason)
        except Exception:               # noqa: BLE001 -- the ledger is best-effort by contract
            pass

    def _say(self, msg: str) -> None:
        try:
            self._log(msg)
        except Exception:               # noqa: BLE001
            pass

    def _write(self, status: dict) -> None:
        # Atomic, best-effort (as classify.py's naming status): a reader mid-open on Windows can
        # make os.replace fail, and the next check rewrites it 5 s later anyway.
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(status), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass
