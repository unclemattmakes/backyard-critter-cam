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

import argparse
import re
import shutil
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

import config
import backup
import db

# The day-folder shape backup.day_dirs archives. Kept here rather than imported so the pruner
# never drags backup.py (and its logging/zipfile setup) into the capture process.
_DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Upper bound assumed for sizing the pre-roll ring (frames = pre_roll_s * this). The real fps is
# measured for playback; this only bounds the buffer so memory can't grow without limit.
_RING_FPS_CAP = 30

# H.264 (libx264) encode settings for the ffmpeg-pipe writer -- matched to web.py's transcode so
# clips recorded here look identical to transcoded ones. 'veryfast' is far quicker than the live
# capture rate (no backpressure at 720p/~10-15 fps); '-bf 0' keeps the written frame count exact.
_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_WARNED_NO_FFMPEG = False       # module-level so the missing-ffmpeg warning is once per process
_X264_PRESET = "veryfast"
_X264_CRF = "23"
_H264_ALIASES = {"h264", "libx264", "avc1", "x264"}


def effective_codec(cfg) -> str:
    """The codec a recorder built from `cfg` will REALLY write -- 'H.264' only when h264 was asked
    for AND ffmpeg is on PATH, otherwise the OpenCV fourcc it falls back to.

    Exists so the startup banner can state the codec on EVERY run. 2026-08-21 ffmpeg vanished off
    PATH (chkdsk rebuilt a directory index after a 0x1E bugcheck and took the winget package tree
    with it) and the rig recorded mp4v for a day without anyone noticing: the only signal was a
    warning that appears solely in the broken case, and you cannot notice a line that ISN'T there.
    A codec printed every time is a line whose CHANGE is visible."""
    want_h264 = str(cfg.clip_codec).lower() in _H264_ALIASES
    if want_h264:
        return "H.264" if _FFMPEG is not None else "mp4v"
    codec = str(cfg.clip_codec)
    return codec if len(codec) == 4 else "mp4v"      # same fourcc-length fallback the recorder uses


def ffmpeg_missing(cfg) -> bool:
    """True in the one case that degrades silently: h264 configured, ffmpeg not on PATH."""
    return str(cfg.clip_codec).lower() in _H264_ALIASES and _FFMPEG is None


class _FfmpegWriter:
    """A cv2.VideoWriter-compatible sink that pipes raw BGR frames to ffmpeg -> H.264 .mp4.

    Why: OpenCV can only reliably WRITE 'mp4v' (MPEG-4 Part 2) on this rig, which browsers can't
    decode; libx264 plays in any <video>, is ~half the size, and cv2 still READS it for clipmotion.
    Same three-method surface as cv2.VideoWriter (isOpened / write / release) so the recorder
    doesn't care which one it's using. '+faststart' moves the moov atom to the front on close so
    the dashboard can stream the clip immediately."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        w, h = size
        self.proc = subprocess.Popen(
            [_FFMPEG, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{max(1.0, fps):.6f}",
             "-i", "-", "-an", "-c:v", "libx264", "-preset", _X264_PRESET, "-crf", _X264_CRF,
             "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._ok = self.proc.stdin is not None

    def isOpened(self) -> bool:
        return self._ok and self.proc.poll() is None

    def write(self, frame) -> None:
        try:
            self.proc.stdin.write(frame.tobytes())     # frame is a C-contiguous BGR uint8 ndarray
        except (BrokenPipeError, OSError, ValueError):
            self._ok = False                            # ffmpeg died -> recorder drops the clip

    def release(self, timeout: float = 30) -> None:
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=timeout)             # let ffmpeg flush + write the moov atom
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# Project-root-relative stored path -- the shared db helper (was a local copy of backyard_cam._rel).
_rel = db.rel_to_root


def _safe_source(source: str) -> str:
    """A filesystem-safe directory name for a `source` (any non-alphanumeric char -> '_'), so a
    source label can name a clips subdir. Mirrors import_trailcam.ledger_path's sanitising."""
    return "".join(c if c.isalnum() else "_" for c in str(source)) or "cam"


def _source_budgets(cfg: config.Config) -> dict[str, float]:
    """Map of clips/<subdir> -> budget in BYTES, keyed by the on-disk directory name (the
    _safe_source spelling), so a scan can look a file's budget up from its path alone. Sources
    without an override -- and legacy clips sitting directly under clips/, which predate the
    per-source layout -- fall into the shared `None` bucket on cfg.clips_max_gb."""
    out = {None: (cfg.clips_max_gb or 0) * (1024 ** 3)}
    for source, gb in (getattr(cfg, "clips_max_gb_by_source", None) or {}).items():
        out[_safe_source(source)] = (gb or 0) * (1024 ** 3)
    return out


def prune_clips(cfg: config.Config, conn) -> int:
    """Keep clips/ under its disk budget by deleting the OLDEST clip FILES. This is what makes
    always-on recording safe on a family rig -- the folder is a rolling window, never an
    unbounded grower. Returns clips removed.

    Budgets are PER SOURCE (cfg.clips_max_gb_by_source, else the shared cfg.clips_max_gb), and
    each source only ever prunes its own files. One shared oldest-first pool silently made the
    sources compete: importing a trail-cam card's 9 GB of video would have evicted nearly every
    glass-door clip, which is backwards -- the live rig can re-record tomorrow, an SD card's
    footage is gone the moment the card is formatted.

    SOFT prune: only the video file is deleted; the clips ROW stays, stamped `pruned_at`. The
    row's children -- clip_tracks, clip_track_embeddings, and their individual_id links -- are
    exactly the derived re-ID/behaviour data the nightly batch mined from the clip, and deleting
    the row used to CASCADE them away (June 2026: 550 tracklet vectors + every Notch/Elliot track
    link silently lost when the month aged out). The video is expendable -- and usually archived
    by backup.py before pruning reaches it anyway; what it taught us is not. Playback surfaces
    filter on `pruned_at IS NULL`; the analytical joins keep using the full table.
    Best-effort: a locked/missing file is skipped, never fatal. 0/None budget = no cap."""
    if not cfg.clips_dir.exists():
        return 0
    budgets = _source_budgets(cfg)
    buckets: dict[str | None, list] = {}
    for p in cfg.clips_dir.rglob("*.mp4"):
        try:
            st = p.stat()                  # one stat per file (was two), inside the try ...
        except OSError:
            continue                       # ... so a file vanishing mid-scan skips that file,
        try:                               # ... not the whole prune (live writer rotating).
            top = p.relative_to(cfg.clips_dir).parts[0]
        except (ValueError, IndexError):
            top = None
        # A file directly under clips/ has no source dir (its 'top' is the filename itself) --
        # bucket those with the other unbudgeted legacy clips rather than inventing a source.
        key = top if (top in budgets and top != p.name) else None
        buckets.setdefault(key, []).append((st.st_mtime, st.st_size, p))

    removed = 0
    for key, files in buckets.items():
        budget = budgets.get(key, budgets[None])
        if budget <= 0:                        # 0/None budget = this source is never pruned
            continue
        total = sum(sz for _, sz, _ in files)
        if total <= budget:
            continue
        guard = _archive_guard(cfg, key)
        n_here = n_held = 0
        for _mt, sz, p in sorted(files):       # oldest first, WITHIN this source only
            if total <= budget:
                break
            if guard is not None and not guard(p):
                n_held += 1
                continue                       # the only copy: the budget does not get to win
            try:
                p.unlink()
            except OSError:
                continue                       # in use / already gone -- try the next one
            total -= sz
            n_here += 1
            # Drop the browser transcode (clips_web mirror) too, so the web cache never outlives
            # its clip -- it stays inside the same rolling budget and is regenerated on demand.
            try:
                (cfg.clips_dir.parent / "clips_web" / p.relative_to(cfg.clips_dir)).unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
            try:
                conn.execute("UPDATE clips SET pruned_at = ? WHERE clip_path = ?",
                             (db.now_local_iso(), _rel(p)))
                conn.commit()
            except Exception:
                pass                           # unstamped row is harmless; next prune retries
        if n_here:
            removed += n_here
            print(f"[clips] pruned {n_here} oldest {key or 'legacy'} clip(s) to stay under "
                  f"{budget / 1024 ** 3:g} GB (rolling window).")
        if n_held:
            # LOUD, because the budget is now being exceeded on purpose and only a human can fix
            # it -- run backup.py (or `--backup-first`) and the next prune will proceed.
            print(f"[clips] KEPT {n_held} {key} clip(s) the budget wanted to delete: they are not "
                  f"in the day-archive yet, and this source's footage exists nowhere else. "
                  f"clips/ is over budget by {(total - budget) / 1024 ** 3:.1f} GB until "
                  f"backup.py archives them.")
    return removed


def _archive_guard(cfg: config.Config, key):
    """A `path -> bool` "is it safe to delete this?" test for an IRREPLACEABLE source, or None
    when this source is replaceable and the budget alone decides.

    Safe means: the day-archive on the backup destination actually CONTAINS this clip, and the
    archive holding it is still there.

    Membership, not merely the existence of the day's zip -- which is all this asked until
    2026-08-23, and it was not enough. A trail-cam import backfills clips into dates that were
    archived weeks ago (the card is dumped and put straight back in the camera, so its dump day
    always arrives in two batches), so "that day has a zip" was true for clips that had never
    been inside one. The budget was then free to delete footage from a card that has since been
    formatted -- the exact loss this guard exists to prevent. `import_trailcam.py --backup-first`
    covered the usual route in, which is why it never bit; it did not make the check correct.

    It answers from a LOCAL index, never by opening the archive. A zip's central directory lives
    at the END of the file, and reading it on a Drive-hosted archive materialises the whole file
    into Drive's cache -- a backup dry-run reading ~185 of them took content_cache from 10 to
    86 GiB. Doing that per prune, on the capture box, while it records, would cost far more than
    the hole it closes. backup.py writes the index as it archives, out of the central directories
    it already has to read, so this side of it is a couple of small local file reads per prune.

    FAILS CLOSED, five ways. No backup destination, an unreachable drive, an unrecognised path
    layout, NO INDEX for that day, or an index that does not list this clip all return "not
    safe", so the file survives and the budget is exceeded instead. That is the project's stated
    asymmetry as code: a full disk is a problem you can see and fix, and footage from a card that
    has since been formatted is not. The cost of the new failure mode is that a rig whose index
    has never been written -- a fresh clone, a restored machine -- holds every irreplaceable clip
    until backup.py runs once, and says so loudly while it does.
    """
    sources = getattr(cfg, "clips_irreplaceable_sources", ()) or ()
    if key is None or key not in {_safe_source(s) for s in sources}:
        return None
    dest = getattr(cfg, "backup_dest", None)
    archive = Path(dest) / "clips" if dest else None
    seen: dict[str, dict | None] = {}       # one index read per DAY per prune pass, not per clip

    def safe(p: Path) -> bool:
        if archive is None:
            return False
        try:
            rel = p.relative_to(cfg.clips_dir)
        except ValueError:
            return False
        parts = rel.parts
        if len(parts) == 3 and _DAY_DIR_RE.match(parts[1]):     # <source>/<date>/<file>.mp4
            stem = f"clips-{parts[0]}-{parts[1]}"
        elif len(parts) == 2 and _DAY_DIR_RE.match(parts[0]):   # legacy <date>/<file>.mp4
            stem = f"clips-{parts[0]}"
        else:
            return False        # a layout backup.day_dirs would not archive -> assume unarchived
        if stem not in seen:
            seen[stem] = backup.read_archive_index(backup.ARCHIVE_INDEX_DIR, stem)
        holds = seen[stem]
        if not holds:
            return False                          # nothing recorded for that day -> prove nothing
        # backup.py's arcnames are project-relative and posix; db.rel_to_root is native-separator.
        member = Path(_rel(p)).as_posix()
        holder = holds.get(member)
        if holder is None:
            return False        # the day is archived, this clip is not: an import backfilled it
        try:
            return (archive / holder).is_file()   # ...and the part holding it has not gone missing
        except OSError:
            return False                          # drive unplugged mid-prune -> keep the footage
    return safe


def _video_codec(path: Path) -> str:
    """Video stream codec_name via ffprobe ('' if unknown / ffprobe missing)."""
    if _FFPROBE is None:
        return ""
    try:
        r = subprocess.run(
            [_FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=30)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def convert_legacy_to_h264(clips_dir: Path, *, verbose: bool = True) -> dict:
    """One-time migration: re-encode every non-H.264 clip under `clips_dir` to H.264 IN PLACE, so it
    plays in a browser without the dashboard's on-demand transcode. Clips recorded before clip_codec
    defaulted to 'h264' are mp4v; so are any recorded during an ffmpeg-less stretch (the cv2 fallback)
    -- this fixes both. Idempotent (already-H.264 clips are skipped) and atomic (encode to a temp
    file, then swap), so an interrupted run never corrupts a clip. Frame count / fps / dimensions are
    preserved, so the clips DB rows -- keyed on the unchanged clip_path -- stay valid (no DB write).
    Needs ffmpeg + ffprobe on PATH. Returns {converted, skipped, failed, saved_bytes}."""
    res = {"converted": 0, "skipped": 0, "failed": 0, "saved_bytes": 0}
    if _FFMPEG is None or _FFPROBE is None:
        print("[clips] ffmpeg/ffprobe not on PATH -- cannot convert clips to H.264.")
        return res
    if not clips_dir.exists():
        print(f"[clips] no clips directory at {clips_dir}.")
        return res
    for p in sorted(clips_dir.rglob("*.mp4")):
        if p.name.endswith(".tmp.mp4"):
            continue                            # a leftover temp from an interrupted run -- ignore
        if _video_codec(p) == "h264":
            res["skipped"] += 1
            continue
        tmp = p.with_suffix(".h264.tmp.mp4")
        before = p.stat().st_size
        try:
            # -c:a aac, not -an: this swap REPLACES the source file, so stripping audio here
            # destroys it forever -- and a trail-cam MP4's microphone track (growls, kit chitter)
            # is the only copy that sound will ever have. The rig's own clips carry no audio
            # track, so for them nothing changes. AAC re-encode because trail-cam PCM/ADPCM
            # won't play in a browser anyway.
            subprocess.run(
                [_FFMPEG, "-y", "-loglevel", "error", "-i", str(p),
                 "-c:a", "aac", "-b:a", "96k", "-c:v", "libx264",
                 "-preset", _X264_PRESET, "-crf", _X264_CRF, "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(tmp)],
                check=True, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not tmp.exists() or tmp.stat().st_size == 0:
                raise RuntimeError("ffmpeg produced an empty file")
            after = tmp.stat().st_size
            tmp.replace(p)                      # atomic in-place swap; clip_path is unchanged
            res["converted"] += 1
            res["saved_bytes"] += max(0, before - after)
            if verbose:
                print(f"  converted {p.name}  ({before / 1e6:.1f} -> {after / 1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001 -- original is untouched (we swap only on success)
            res["failed"] += 1
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"  [clips] failed to convert {p.name}: {e}")
    if verbose:
        print(f"[clips] H.264 conversion: {res['converted']} converted, "
              f"{res['skipped']} already H.264"
              + (f", {res['failed']} failed" if res["failed"] else "")
              + f"  (freed {res['saved_bytes'] / 1e6:.0f} MB).")
    return res


class ClipRecorder:
    """Records a short clip around each visit. Drive it from the capture loop:
        recorder.note_frame(frame, now)              # every frame (buffers pre-roll, writes clip)
        recorder.note_detection(now, saved_dets)     # when animal detections land (start/extend)
        recorder.finalize()                          # on shutdown (flush any open clip)
    `now` is a time.monotonic() value; `saved_dets` are detector.Detection objects (only the
    classes you actually save -- so a person at the glass never starts a clip)."""

    def __init__(self, cfg: config.Config, conn, source: str | None = None):
        self.cfg = cfg
        self.conn = conn
        # The DB 'source' this recorder tags its clips with. Defaults to cfg.source for the
        # single-camera rig; a multi-camera rig passes each camera's own source so two cameras
        # never share a clip row -- and writes each camera's clips to its own clips/<source>/<date>/
        # subdir, so two cameras firing in the same millisecond can't collide on a filename.
        self.source = source or cfg.source
        self.scale = min(1.0, max(0.05, cfg.clip_scale))
        self.pre_roll = max(0.0, cfg.clip_pre_roll_s)
        self.post_roll = max(0.0, cfg.clip_post_roll_s)
        self.max_s = max(1.0, cfg.clip_max_s)
        self.codec = cfg.clip_codec
        # 'h264' -> pipe to ffmpeg (browser-playable). If ffmpeg is missing we fall back to the
        # OpenCV mp4v writer so recording still works (the dashboard transcodes those on demand).
        want_h264 = str(cfg.clip_codec).lower() in _H264_ALIASES
        self.use_ffmpeg = want_h264 and _FFMPEG is not None
        self.cv2_codec = "mp4v" if want_h264 else str(cfg.clip_codec)
        # An OpenCV fourcc must be exactly 4 characters; a typo'd or 3-char codec ('vp9', 'mjpg ')
        # would otherwise raise inside cv2.VideoWriter_fourcc and crash the whole capture rig on
        # the first clip. Fall back to a safe codec rather than take the rig down over a config typo.
        if len(self.cv2_codec) != 4:
            print(f"[clips] clip_codec '{cfg.clip_codec}' is not a 4-character fourcc -- "
                  f"recording 'mp4v' instead.")
            self.cv2_codec = "mp4v"
        # Once per PROCESS, not once per ClipRecorder: with three cameras this fired three times
        # per restart, which reads as noise rather than signal. The loud version lives in the
        # startup banner (backyard_cam.py) -- this stays for anyone building a recorder directly.
        global _WARNED_NO_FFMPEG
        if want_h264 and _FFMPEG is None and not _WARNED_NO_FFMPEG:
            _WARNED_NO_FFMPEG = True
            print("[clips] ffmpeg not found on PATH -- recording mp4v instead of H.264 "
                  "(clips still record; the dashboard will transcode them for playback).")
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
            h, w = frame.shape[:2]
            # Round the scaled size DOWN to EVEN: libx264 (yuv420p) refuses odd dimensions, and
            # an innocent-looking scale hits one easily -- 0.667 x 1920 = 1280.64, which fx/fy
            # rounding turned into 1281, killing ffmpeg on the first frame of every clip
            # (2026-07-20: a full day of clips lost to exactly this).
            nw = max(2, int(w * self.scale) & ~1)
            nh = max(2, int(h * self.scale) & ~1)
            return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
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
            self._check_writer()
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
        # Per-source subdir so a multi-camera rig keeps each camera's clips apart (and a same-
        # millisecond filename on two cameras can't collide). Single-camera rigs that pass no
        # source still get clips/<source>/<date>/ -- harmless, and clipmotion/web rglob clips_dir
        # so older flat clips/<date>/ clips are still found and served.
        day_dir = self.cfg.clips_dir / _safe_source(self.source) / wall.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        stamp = wall.strftime("%Y-%m-%dT%H-%M-%S-") + f"{wall.microsecond // 1000:03d}"
        self.clip_path = day_dir / f"{stamp}.mp4"
        # Prepend the buffered pre-roll so the clip opens on the animal arriving.
        for _t, f in list(self.ring):
            self._write(f)
        self._check_writer()   # ffmpeg rejects bad input by DYING on the first write -- catch it now

    def _open_writer(self, w: int, h: int):
        """Open the clip sink: an ffmpeg/H.264 pipe when configured + available, else cv2's mp4v
        writer. Returns the writer, or None if it couldn't be opened (caller disables recording)."""
        if self.use_ffmpeg:
            try:
                wr = _FfmpegWriter(self.clip_path, self.fps, (w, h))
                if wr.isOpened():
                    return wr
                wr.release()        # opened but immediately dead -- close the pipe/proc, don't leak it
            except Exception as e:  # noqa: BLE001 -- fall back to cv2 rather than lose the clip
                print(f"[clips] ffmpeg writer failed ({e}); falling back to OpenCV mp4v.")
            self.use_ffmpeg = False                 # don't retry ffmpeg for the rest of the run
        return cv2.VideoWriter(str(self.clip_path), cv2.VideoWriter_fourcc(*self.cv2_codec),
                               self.fps, (w, h))

    def _write(self, frame) -> None:
        if self.disabled:           # a prior open failed -- don't re-attempt for every buffered frame
            return
        if self.writer is None:
            h, w = frame.shape[:2]
            self.size = (w, h)
            writer = self._open_writer(w, h)
            if writer is None or not writer.isOpened():
                print(f"[clips] could not open a video writer ('{self.codec}' codec) -- "
                      f"disabling clip recording for this run.")
                self.disabled = True
                self.recording = False
                return
            self.writer = writer
        self.writer.write(frame)
        self.frame_count += 1

    def _check_writer(self) -> None:
        """Fall back to the cv2 writer if the ffmpeg pipe has DIED on this clip. ffmpeg rejects
        bad input by exiting on the very first frame (e.g. an odd frame size), and
        `_FfmpegWriter.write` deliberately swallows the broken pipe -- so without this check the
        recorder keeps counting frames into a dead pipe and finalize() finds only a 0-byte file
        to silently drop. That exact combination hid a completely dead recorder for a full day
        (2026-07-20). Rebuilding from the pre-roll ring means a death at clip start (the common
        case: encoder init rejects the frame geometry) loses nothing; a rare genuinely-mid-clip
        death keeps the last-few-seconds tail rather than dropping the clip outright."""
        if not isinstance(self.writer, _FfmpegWriter) or self.writer.isOpened():
            return
        print(f"[clips] ffmpeg pipe died on {_rel(self.clip_path)} -- rebuilding this clip on "
              f"the OpenCV '{self.cv2_codec}' writer (and using it for the rest of the run).")
        self.use_ffmpeg = False
        try:
            self.writer.release(timeout=1)
        except Exception:
            pass
        self.writer = None
        self.frame_count = 0
        for _t, f in list(self.ring):     # best-available rebuild from the buffer we still hold
            self._write(f)

    def finalize(self, shutdown: bool = False) -> None:
        """Close the current clip (if any) and write its DB row. Safe to call when not recording and
        to call repeatedly. Called inline mid-session when a clip auto-stops (shutdown=False -> a
        short writer-flush timeout, so a stuck ffmpeg can't stall the single capture thread) and
        once on rig shutdown (shutdown=True -> a generous timeout to finish the final flush)."""
        if not self.recording:
            return
        self.recording = False
        if self.writer is not None:
            try:
                if isinstance(self.writer, _FfmpegWriter):
                    self.writer.release(timeout=30 if shutdown else 5)
                else:
                    self.writer.release()
            except Exception:
                pass
        self.writer = None

        # A finalized clip must have frames AND a non-empty file on disk -- if ffmpeg died mid-write
        # the pipe writer leaves a 0-byte/missing file, which we drop rather than log a dead row.
        wrote_file = (self.clip_path is not None and self.clip_path.exists()
                      and self.clip_path.stat().st_size > 0)
        if self.frame_count > 0 and wrote_file:
            rel = _rel(self.clip_path)
            ended = datetime.now().astimezone().isoformat()
            w, h = self.size if self.size else (None, None)
            try:
                db.insert_clip(self.conn, source=self.source, clip_path=rel,
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
            if self.frame_count > 0:
                # Frames were handed to a writer but nothing usable landed on disk: the encoder
                # died and the fallback couldn't save it either. Say so LOUDLY -- this branch
                # dropping clips in silence is how a dead recorder went unnoticed for a whole
                # day (2026-07-20). Never make this quiet again.
                print(f"[clips] DROPPED {_rel(self.clip_path)} -- {self.frame_count} frame(s) "
                      "written but the file is missing/empty (video writer died).")
            # Drop the empty/partial file, if any.
            try:
                if self.clip_path.exists():
                    self.clip_path.unlink()
            except Exception:
                pass


def main() -> int:
    """Clip maintenance CLI. Currently: migrate legacy mp4v clips to browser-playable H.264."""
    p = argparse.ArgumentParser(description="Behaviour-clip maintenance utilities.")
    p.add_argument("--to-h264", action="store_true",
                   help="Re-encode legacy mp4v clips in --clips-dir to H.264 in place "
                        "(browser-playable; idempotent; preserves the clips DB rows).")
    p.add_argument("--clips-dir", default=str(config.CONFIG.clips_dir),
                   help="Clips directory to operate on (default: config.clips_dir).")
    args = p.parse_args()
    if args.to_h264:
        convert_legacy_to_h264(Path(args.clips_dir))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
