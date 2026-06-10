"""
Phase 4 capture -- record a short VIDEO clip around each visit.

Stills capture WHO and WHEN; a clip captures HOW -- gait, approach speed, dwell, vigilance,
who-defers-to-whom. That's the substance of behaviour (PLAN.md phase 4), and a confound-robust
second shot at individual ID: a limp or a gait reads the same from any angle, where a single
still frame (pose + soft glass imagery) does not. So clips feed BOTH the behaviour axis and a
future motion-based re-ID.

Design -- boring and robust, and entirely inside the capture thread (so the cv2.VideoWriter is
only ever touched from one thread; no locking):

  * a rolling PRE-ROLL ring buffer holds the last `clip_pre_roll_s` of frames in memory, so a
    clip includes the animal ARRIVING -- the seconds before the detector first fired;
  * the first animal detection opens an .mp4 writer, dumps the pre-roll, then keeps writing live
    frames;
  * the clip ends `clip_post_roll_s` after the LAST detection (the animal has left), or at the
    `clip_max_s` safety cap so a camped-out raccoon can't make a ten-minute file;
  * one `clips` DB row per clip (time span, fps, size, detection count) for phase-4 queries.

Opt-in (config.record_clips / --record-clips); off by default so the family one-click rig's disk
behaviour doesn't change until asked. Stills are still saved alongside -- clips are additive.

Memory note: the pre-roll ring holds raw frames, so it costs roughly
`clip_pre_roll_s * fps * frame_bytes` of RAM (e.g. 3 s * 30 fps * ~2.6 MB at 720p ~ 230 MB).
Lower `clip_pre_roll_s`, or set `clip_scale` < 1.0 to downscale the buffered/recorded frames,
if that's too much.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

import config
import db

# Upper bound assumed for sizing the pre-roll ring (frames = pre_roll_s * this). The real fps is
# measured for playback; this only bounds the buffer so memory can't grow without limit.
_RING_FPS_CAP = 30


def _rel(path: Path) -> str:
    """Project-root-relative path when possible (keeps the DB portable), like backyard_cam._rel."""
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


def prune_clips(cfg: config.Config, conn) -> int:
    """Keep clips/ under cfg.clips_max_gb by deleting the OLDEST clips (file + DB row; any
    clip_tracks rows cascade). This is what makes always-on recording safe on a family rig --
    the folder is a rolling window, never an unbounded grower. Returns clips removed.
    Best-effort: a locked/missing file is skipped, never fatal. 0/None budget = no cap."""
    budget = (cfg.clips_max_gb or 0) * (1024 ** 3)
    if budget <= 0 or not cfg.clips_dir.exists():
        return 0
    try:
        files = [(p.stat().st_mtime, p.stat().st_size, p)
                 for p in cfg.clips_dir.rglob("*.mp4")]
    except OSError:
        return 0
    total = sum(sz for _, sz, _ in files)
    if total <= budget:
        return 0
    removed = 0
    for _mt, sz, p in sorted(files):           # oldest first
        if total <= budget:
            break
        try:
            p.unlink()
        except OSError:
            continue                           # in use / already gone -- try the next one
        total -= sz
        removed += 1
        try:
            conn.execute("DELETE FROM clips WHERE clip_path = ?", (_rel(p),))
            conn.commit()
        except Exception:
            pass                               # orphan row is harmless; next prune retries
    if removed:
        print(f"[clips] pruned {removed} oldest clip(s) to stay under "
              f"{cfg.clips_max_gb:g} GB (rolling window).")
    return removed


class ClipRecorder:
    """Records a short clip around each visit. Drive it from the capture loop:
        recorder.note_frame(frame, now)              # every frame (buffers pre-roll, writes clip)
        recorder.note_detection(now, saved_dets)     # when animal detections land (start/extend)
        recorder.finalize()                          # on shutdown (flush any open clip)
    `now` is a time.monotonic() value; `saved_dets` are detector.Detection objects (only the
    classes you actually save -- so a person at the glass never starts a clip)."""

    def __init__(self, cfg: config.Config, conn):
        self.cfg = cfg
        self.conn = conn
        self.scale = min(1.0, max(0.05, cfg.clip_scale))
        self.pre_roll = max(0.0, cfg.clip_pre_roll_s)
        self.post_roll = max(0.0, cfg.clip_post_roll_s)
        self.max_s = max(1.0, cfg.clip_max_s)
        self.codec = cfg.clip_codec
        ring_len = max(1, int(self.pre_roll * _RING_FPS_CAP) + 5)
        self.ring: deque = deque(maxlen=ring_len)

        self.recording = False
        self.disabled = False           # set if the writer can't be opened (bad codec) -- degrade
        self.writer = None
        self.clip_path: Path | None = None
        self.started_t = 0.0            # monotonic clock, for durations
        self.last_det_t = 0.0
        self.started_iso: str | None = None
        self.fps: float | None = None
        self.size: tuple[int, int] | None = None   # (w, h) actually written
        self.frame_count = 0
        self.detection_count = 0
        self.max_conf = 0.0
        self.clips_saved = 0
        self._loop_fps = None          # the capture loop's sustained fps (best clip-rate estimate)
        prune_clips(cfg, conn)         # enforce the disk budget left over from previous runs

    # -- frame ingestion -----------------------------------------------------------
    def _prep(self, frame):
        if self.scale != 1.0:
            return cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
        return frame

    def note_frame(self, frame, now: float, loop_fps: float | None = None) -> None:
        """Call every captured frame. Buffers pre-roll and, while recording, writes + checks
        the stop conditions (idle since last detection, or the max-length cap). `loop_fps` is the
        capture loop's own rolling frame rate -- the most reliable clip-playback rate, so clips
        play at life speed (a momentary ring measurement under/over-shoots when the loop rate
        swings as the detector fires)."""
        if self.disabled:
            return
        if loop_fps and loop_fps > 3:
            self._loop_fps = loop_fps
        prepped = None
        if self.pre_roll > 0:
            prepped = self._prep(frame)
            self.ring.append((now, prepped))
        if self.recording:
            if prepped is None:
                prepped = self._prep(frame)
            self._write(prepped)
            if (now - self.last_det_t) > self.post_roll or (now - self.started_t) > self.max_s:
                self.finalize()

    def note_detection(self, now: float, detections) -> None:
        """Call when animal detections land. Starts a clip if idle, and extends the current one."""
        if self.disabled:
            return
        if not self.recording:
            self._start(now)
        self.last_det_t = now
        self.detection_count += len(detections)
        if detections:
            self.max_conf = max(self.max_conf, max(d.confidence for d in detections))

    # -- clip lifecycle ------------------------------------------------------------
    def _measure_fps(self) -> float:
        """Clip playback rate (so clips play at life speed). Prefer a forced value, then the
        capture loop's sustained rolling fps (most reliable), then a pre-roll-ring estimate, then
        a 15 fps fallback. The ring estimate alone proved unreliable (it under-shot a 14-23 fps
        rig to ~10 -> 1.4-2.3x slow-mo), because it samples only the brief, often-slow window
        around the triggering detection."""
        if self.cfg.clip_fps:
            return float(self.cfg.clip_fps)
        if self._loop_fps and self._loop_fps > 3:
            return min(float(_RING_FPS_CAP), self._loop_fps)
        if len(self.ring) >= 2:
            t0, t1 = self.ring[0][0], self.ring[-1][0]
            if t1 > t0:
                return min(float(_RING_FPS_CAP), max(5.0, (len(self.ring) - 1) / (t1 - t0)))
        return 15.0

    def _start(self, now: float) -> None:
        self.recording = True
        self.started_t = now
        self.last_det_t = now
        self.frame_count = 0
        self.detection_count = 0
        self.max_conf = 0.0
        self.writer = None
        self.size = None
        self.fps = self._measure_fps()
        wall = datetime.now().astimezone()
        self.started_iso = wall.isoformat()
        day_dir = self.cfg.clips_dir / wall.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        stamp = wall.strftime("%Y-%m-%dT%H-%M-%S-") + f"{wall.microsecond // 1000:03d}"
        self.clip_path = day_dir / f"{stamp}.mp4"
        # Prepend the buffered pre-roll so the clip opens on the animal arriving.
        for _t, f in list(self.ring):
            self._write(f)

    def _write(self, frame) -> None:
        if self.writer is None:
            h, w = frame.shape[:2]
            self.size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(str(self.clip_path), fourcc, self.fps, (w, h))
            if not writer.isOpened():
                print(f"[clips] could not open a video writer ('{self.codec}' codec) -- "
                      f"disabling clip recording for this run.")
                self.disabled = True
                self.recording = False
                return
            self.writer = writer
        self.writer.write(frame)
        self.frame_count += 1

    def finalize(self) -> None:
        """Close the current clip (if any) and write its DB row. Safe to call when not recording
        and to call repeatedly; the capture loop calls it on shutdown."""
        if not self.recording:
            return
        self.recording = False
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        self.writer = None

        if self.frame_count > 0 and self.clip_path is not None:
            rel = _rel(self.clip_path)
            ended = datetime.now().astimezone().isoformat()
            w, h = self.size if self.size else (None, None)
            try:
                db.insert_clip(self.conn, source=self.cfg.source, clip_path=rel,
                               started_at=self.started_iso, ended_at=ended, fps=self.fps,
                               width=w, height=h, frame_count=self.frame_count,
                               detection_count=self.detection_count, max_confidence=self.max_conf)
            except Exception as e:  # noqa: BLE001 -- a DB hiccup shouldn't crash the rig
                print(f"[clips] saved {rel} but could not write its DB row: {e}")
            dur = self.frame_count / self.fps if self.fps else 0.0
            self.clips_saved += 1
            print(f"[clips] saved {rel}  ({self.frame_count} frames, {dur:.1f}s, "
                  f"{self.detection_count} det)")
            prune_clips(self.cfg, self.conn)   # keep the rolling window inside the disk budget
        elif self.clip_path is not None:
            # Opened but nothing got written -- drop the empty file.
            try:
                if self.clip_path.exists():
                    self.clip_path.unlink()
            except Exception:
                pass
