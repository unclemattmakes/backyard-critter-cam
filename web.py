"""
Local web dashboard server (Python stdlib http.server -- no web framework).

Serves the live MJPEG stream, stats + species JSON, crop images, AND now:
  * live camera controls   -- GET/POST /api/camera  (web thread queues changes; the capture
    thread applies them, since OpenCV VideoCapture is single-thread -- see CameraControlBridge)
  * per-species browsing    -- GET /api/species, /api/species/<name>
  * ID verification         -- POST /api/detection/<id>  {action: verify|reject|correct, species}

The page itself lives in dashboard.html (read from disk so the design can iterate without a
restart). Bound to localhost by default. No torch/cv2 here -- only stdlib + db/stats.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import behavior
import config
import db
import stats

DASHBOARD_FILE = config.ROOT / "dashboard.html"

# Camera settings the dashboard is allowed to POST. 'exposure'/'gain' are handled specially by
# apply_camera_settings (None = auto); the rest map to cv2.CAP_PROP_<NAME>.
_ALLOWED_CONTROLS = {
    "exposure", "gain", "EXPOSURE", "GAIN", "FOCUS", "BRIGHTNESS", "CONTRAST", "SATURATION",
    "SHARPNESS", "GAMMA", "HUE", "WB_TEMPERATURE", "AUTO_EXPOSURE", "AUTOFOCUS", "AUTO_WB",
    "BACKLIGHT",
}


class FrameBuffer:
    """Thread-safe holder of the latest annotated JPEG, with new-frame signalling."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame: bytes | None = None
        self._seq = 0

    def update(self, jpg: bytes) -> None:
        with self._cond:
            self._frame = jpg
            self._seq += 1
            self._cond.notify_all()

    def get(self):
        with self._cond:
            return self._frame, self._seq

    def wait(self, last_seq: int, timeout: float = 4.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._frame, self._seq


class CameraControlBridge:
    """Hand camera-control changes from web threads to the single capture thread, and publish
    the capture thread's current control readings back for the UI. (cv2.VideoCapture is not
    thread-safe, so the capture thread does all the cap.get/cap.set work.)"""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: dict = {}
        self._current: dict = {}

    def request(self, settings: dict) -> None:        # web -> capture
        with self._lock:
            self._pending.update(settings)

    def take_pending(self) -> dict:                   # capture drains
        with self._lock:
            p, self._pending = self._pending, {}
            return p

    def publish(self, current: dict) -> None:         # capture -> web
        with self._lock:
            self._current = dict(current)

    def snapshot(self) -> dict:                        # web reads
        with self._lock:
            return dict(self._current)


def _clean_settings(data: dict) -> dict:
    out = {}
    for k, v in (data or {}).items():
        if k in _ALLOWED_CONTROLS:
            out[k] = None if v is None else float(v)
    return out


def _is_within(target: Path, parent: Path) -> bool:
    try:
        target.relative_to(parent)
        return True
    except ValueError:
        return False


# Media types served from /media (crops, frames, and now behaviour clips). Video needs a correct
# Content-Type AND HTTP Range support, or a browser <video> won't stream or seek.
_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
}
_RANGE_CHUNK = 4 * 1024 * 1024   # cap an open-ended range so a clip is never read whole into RAM


def _parse_range(header: str, size: int):
    """Parse a single-range 'bytes=START-END' header against a `size`-byte file. Returns
    (start, end, open_ended) inclusive, or None for an unsatisfiable/garbled range. Handles an
    open-ended 'bytes=START-' (the common video probe) and a suffix 'bytes=-N' (last N bytes)."""
    try:
        spec = header.split("=", 1)[1].split(",", 1)[0].strip()
        lo, _, hi = spec.partition("-")
        open_ended = lo != "" and hi == ""
        if lo == "":                       # suffix range: the last N bytes
            n = int(hi)
            if n <= 0:
                return None
            start, end = max(0, size - n), size - 1
        else:
            start = int(lo)
            end = int(hi) if hi else size - 1
        end = min(end, size - 1)
        if start < 0 or start > end or start >= size:
            return None
        return start, end, open_ended
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Web-playable clips. The rig records clips with OpenCV's 'mp4v' fourcc (MPEG-4 Part 2) because
# that's the codec cv2 can reliably WRITE on Windows -- but browsers only DECODE H.264 in an .mp4,
# so an mp4v clip fails to play in a <video> (MEDIA_ERR_SRC_NOT_SUPPORTED). We therefore transcode
# each clip to H.264 ONCE, on first view, into a `clips_web/` cache (faststart = instant streaming),
# and serve that. Originals are left untouched for clipmotion/re-ID (cv2 reads mp4v fine). Needs
# ffmpeg on PATH; without it we serve the original and the browser simply can't play it.
# ---------------------------------------------------------------------------
_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_transcode_guard = threading.Lock()
_transcode_locks: dict = {}
_codec_cache: dict = {}        # clip path -> video codec_name, so each clip is probed at most once


def _is_h264(path: Path) -> bool:
    """True if `path`'s video stream is already H.264 (browser-playable) -- new clips are recorded
    this way, so they're served as-is and never re-transcoded. Probed once per path, then cached.
    Without ffprobe we assume NOT h264 (so a legacy mp4v clip still gets transcoded)."""
    if _FFPROBE is None:
        return False
    key = str(path)
    if key not in _codec_cache:
        try:
            r = subprocess.run(
                [_FFPROBE, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(path)],
                capture_output=True, text=True, timeout=15)
            _codec_cache[key] = (r.stdout or "").strip()
        except Exception:
            _codec_cache[key] = ""
    return _codec_cache[key] == "h264"


def _lock_for(key: str) -> threading.Lock:
    """One lock per output path, so concurrent range requests for the same clip transcode once."""
    with _transcode_guard:
        lk = _transcode_locks.get(key)
        if lk is None:
            lk = _transcode_locks[key] = threading.Lock()
        return lk


def _web_clip(src: Path, clips_root: Path, cache_root: Path):
    """H.264 version of clip `src`, transcoded once into `cache_root` mirroring its path under
    `clips_root`. Returns the cached Path, or None if it can't be made (no ffmpeg / transcode
    failed) -- the caller then serves the original. Re-transcodes if the source is newer."""
    if _FFMPEG is None:
        return None
    if _is_h264(src):
        return src                         # already browser-playable (new clips) -> serve as-is
    try:
        rel = src.relative_to(clips_root)
    except ValueError:
        rel = Path(src.name)
    out = cache_root / rel

    def _fresh() -> bool:
        try:
            return out.stat().st_size > 0 and out.stat().st_mtime >= src.stat().st_mtime
        except OSError:
            return False

    if _fresh():
        return out
    with _lock_for(str(out)):
        if _fresh():                       # another thread may have just built it
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.mp4")
        try:
            subprocess.run(
                [_FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)],
                check=True, timeout=180,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            tmp.replace(out)               # atomic publish; partial reads never see a half file
            return out
        except Exception:                  # noqa: BLE001 -- bad ffmpeg/codec -> fall back to original
            try:
                tmp.unlink()
            except OSError:
                pass
            return None


def make_server(cfg, frame_buffer: FrameBuffer, control_bridge: CameraControlBridge):
    allowed_dirs = [d.resolve() for d in (cfg.crops_dir, cfg.frames_dir, cfg.clips_dir,
                                          getattr(cfg, "clip_crops_dir", cfg.clips_dir))]
    stop_event = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, content_type, body: bytes, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, "application/json",
                       json.dumps(obj).encode("utf-8"), {"Cache-Control": "no-store"})

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    if DASHBOARD_FILE.exists():
                        self._send(200, "text/html; charset=utf-8", DASHBOARD_FILE.read_bytes())
                    else:
                        self._send(500, "text/plain", b"dashboard.html missing")
                elif path == "/dashboard.css":
                    # Stylesheet split out of dashboard.html (read fresh from disk, like the HTML,
                    # so the design still iterates without a server restart).
                    css = config.ROOT / "dashboard.css"
                    if css.exists():
                        self._send(200, "text/css; charset=utf-8", css.read_bytes())
                    else:
                        self._send(404, "text/plain", b"dashboard.css missing")
                elif path == "/api/stats":
                    self._json(stats.compute_stats(cfg) or {"total_crops": 0})
                elif path == "/api/species":
                    self._json(stats.species_overview(cfg) or {"species": [], "total": 0})
                elif path.startswith("/api/species/"):
                    name = urllib.parse.unquote(path[len("/api/species/"):])
                    self._json(stats.species_crops(cfg, name))
                elif path == "/api/labels":
                    self._json(_candidate_labels(cfg))
                elif path == "/api/denylist":
                    # The dashboard mirrors this denylist (NONCRITTER); serve it so the two never
                    # drift -- stats._NON_CRITTER is the single source of truth.
                    self._json(sorted(stats._NON_CRITTER))
                elif path == "/api/camera":
                    self._json(control_bridge.snapshot())
                elif path == "/api/naming":
                    self._json(_naming_status())
                elif path == "/api/crops":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(stats.crops_page(
                        cfg, day=(q.get("day") or [None])[0], species=(q.get("species") or [None])[0],
                        start=(q.get("start") or [None])[0], end=(q.get("end") or [None])[0],
                        offset=int((q.get("offset") or [0])[0]), limit=min(int((q.get("limit") or [60])[0]), 200)))
                elif path == "/api/visits":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(stats.visits_page(cfg, day=(q.get("day") or [None])[0]))
                elif path == "/api/digest":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(stats.period_digest(cfg, edition=(q.get("edition") or ["auto"])[0]))
                elif path == "/api/behavior":
                    self._json(behavior.overview(cfg))
                elif path == "/api/individuals":
                    self._json(stats.individuals_overview(cfg))
                elif path == "/api/reid/queue":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(_reid_queue(cfg, species=(q.get("species") or [cfg.reid_species])[0],
                                           limit=min(int((q.get("limit") or [30])[0]), 100)))
                elif path == "/api/reid/poses":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(_reid_poses(cfg, (q.get("individual") or [""])[0]))
                elif path == "/api/reid/unblend":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(_reid_unblend(cfg, (q.get("visit_id") or [""])[0]))
                elif path == "/api/reid/clips":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(_reid_clips(cfg, (q.get("individual") or [""])[0]))
                elif path == "/snapshot.jpg":
                    frame, _ = frame_buffer.get()
                    if frame is None:
                        self._send(503, "text/plain", b"no frame yet")
                    else:
                        self._send(200, "image/jpeg", frame, {"Cache-Control": "no-store"})
                elif path == "/stream.mjpg":
                    self._stream()
                elif path.startswith("/media/"):
                    self._media(path[len("/media/"):])
                else:
                    self._send(404, "text/plain", b"not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._send(500, "text/plain", f"error: {e}".encode())
                except Exception:
                    pass

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw or b"{}")
            except Exception:
                data = {}
            try:
                if path == "/api/camera":
                    control_bridge.request(_clean_settings(data))
                    self._json({"ok": True})
                elif path == "/api/individual":
                    self._individual_action(data)
                elif path == "/api/reid/confirm":
                    self._reid_confirm(data)
                elif path == "/api/visit/label":
                    self._visit_label(data)
                elif path == "/api/reid/unblend/label":
                    self._unblend_label(data)
                elif path.startswith("/api/detection/"):
                    self._detection_action(int(path.rsplit("/", 1)[-1]), data)
                else:
                    self._send(404, "text/plain", b"not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._json({"error": str(e)}, code=500)
                except Exception:
                    pass

        def _unblend_label(self, data):
            """Assign an individual to a cluster of clip tracklets (the un-blend action):
            {"track_ids": [...], "name": "Notch"}. name=""/null clears."""
            ids = data.get("track_ids") or []
            if not isinstance(ids, list) or not ids:
                self._json({"error": "missing track_ids"}, code=400)
                return
            name = (str(data.get("name") or "").strip()) or None
            conn = db.connect(cfg.db_path)
            try:
                n = db.set_clip_track_individual(conn, [int(t) for t in ids], name)
                self._json({"ok": True, "labelled": n, "name": name})
            finally:
                conn.close()

        def _visit_label(self, data):
            """Confirm/correct a visit's SPECIES and/or name its INDIVIDUAL in one call -- the
            shared backend for the Individuals queue (sends visit_id) and the Explorer Visits
            list (sends source+start+end). Body: {visit_id | source,start,end} plus any of
            {name (str|null), species (str), verify (bool)}. Deliberately does NOT rebuild visits
            (a rebuild renumbers ids and would invalidate an open queue; labels live on the
            detections and survive the next rebuild)."""
            kw = {}
            if data.get("visit_id") is not None:
                try:
                    kw["visit_id"] = int(data["visit_id"])
                except (TypeError, ValueError):
                    self._json({"error": "bad visit_id"}, code=400)
                    return
            elif data.get("source") and data.get("start") and data.get("end"):
                kw.update(source=str(data["source"]), start=str(data["start"]),
                          end=str(data["end"]))
            else:
                self._json({"error": "need visit_id or source+start+end"}, code=400)
                return
            if "name" in data:                      # present (incl. null) = act on identity
                kw["name"] = (str(data["name"]).strip() or None) if data["name"] is not None else None
            if data.get("species"):
                kw["species"] = str(data["species"]).strip()
            if data.get("verify"):
                kw["verify"] = True
            conn = db.connect(cfg.db_path)
            try:
                self._json({"ok": True, **db.apply_visit_label(conn, **kw)})
            finally:
                conn.close()

        def _reid_confirm(self, data):
            """Confirm (or clear) WHO one visit was: {"visit_id": 1014, "name": "Stan"}.
            name=""/null clears. Stamps the visit's species-matching crops (individual_source=
            'human') and mirrors the name onto the visits row. Deliberately does NOT rebuild the
            visits table -- a rebuild renumbers visit ids and would invalidate every other card
            in the open review queue; labels live on detections, so the next rebuild inherits
            them anyway."""
            try:
                vid = int(data.get("visit_id"))
            except (TypeError, ValueError):
                self._json({"error": "missing/bad 'visit_id'"}, code=400)
                return
            name = (data.get("name") or "").strip() or None
            conn = db.connect(cfg.db_path)
            try:
                n = db.label_visit(conn, vid, name)
                self._json({"ok": True, "visit_id": vid, "name": name, "stamped": n})
            finally:
                conn.close()

        def _individual_action(self, data):
            """Name / merge / clear an individual group: {"from": "raccoon_c01", "to": "Notch"}.
            to="" or null clears the label. Naming two groups the same name merges them. The visit
            ledger is refreshed afterwards so per-individual behaviour follows the rename."""
            old = (data.get("from") or "").strip()
            new = (data.get("to") or "").strip() or None
            if not old:
                self._json({"error": "missing 'from'"}, code=400)
                return
            import sqlite3 as _sq
            import visits as _visits
            conn = db.connect(cfg.db_path)
            try:
                n = db.rename_individual(conn, old, new)
                try:
                    conn.row_factory = _sq.Row
                    _visits.build_visits(conn, cfg.visit_gap_minutes, verbose=False)
                except Exception:
                    pass   # visit refresh is best-effort; the rename itself already committed
                self._json({"ok": True, "renamed": n, "to": new})
            finally:
                conn.close()

        def _detection_action(self, det_id, data):
            action = data.get("action")
            conn = db.connect(cfg.db_path)
            try:
                if action == "verify":
                    db.set_species_verified(conn, det_id, 1)
                elif action == "reject":
                    db.set_species_verified(conn, det_id, 0)
                elif action == "unset":
                    db.set_species_verified(conn, det_id, None)
                elif action == "correct" and data.get("species"):
                    db.correct_species(conn, det_id, str(data["species"]).strip())
                else:
                    self._json({"error": "bad action"}, code=400)
                    return
                self._json({"ok": True})
            finally:
                conn.close()

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.send_header("Cache-Control", "no-store, private")
            self.end_headers()
            last = -1
            while not stop_event.is_set():
                frame, last = frame_buffer.wait(last, timeout=4.0)
                if frame is None:
                    continue
                self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")

        def _media(self, rel):
            target = (config.ROOT / urllib.parse.unquote(rel)).resolve()
            if not any(_is_within(target, d) for d in allowed_dirs) or not target.is_file():
                self._send(404, "text/plain", b"not found")
                return
            ctype = _MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
            # Clips are recorded as mp4v (browser-undecodable); serve a cached H.264 transcode.
            serve = target
            if target.suffix.lower() == ".mp4":
                clips_root = cfg.clips_dir.resolve()
                if _is_within(target, clips_root):
                    web_ver = _web_clip(target, clips_root, clips_root.parent / "clips_web")
                    if web_ver is not None:
                        serve = web_ver
            size = serve.stat().st_size
            rng = self.headers.get("Range")
            if not rng:
                # Whole file. Advertise Accept-Ranges so a <video> knows it can seek next time.
                self._send(200, ctype, serve.read_bytes(),
                           {"Cache-Control": "max-age=86400", "Accept-Ranges": "bytes"})
                return
            parsed = _parse_range(rng, size)
            if parsed is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end, open_ended = parsed
            if open_ended:                          # serve a bounded chunk; the player asks for more
                end = min(end, start + _RANGE_CHUNK - 1)
            with open(serve, "rb") as f:
                f.seek(start)
                body = f.read(end - start + 1)
            self._send(206, ctype, body, {
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Accept-Ranges": "bytes", "Cache-Control": "max-age=86400",
            })

    server = ThreadingHTTPServer((cfg.web_host, cfg.web_port), Handler)
    server.daemon_threads = True
    server.stop_event = stop_event
    return server


def _naming_status() -> dict:
    """Read the live-naming helper's status file (written by classify.py --watch) so the
    dashboard can show warming-up vs naming vs stopped. A stale heartbeat is treated as stopped."""
    p = config.NAMING_STATUS_FILE
    try:
        if not p.exists():
            return {"state": "off"}
        data = json.loads(p.read_text())
        data["age"] = round(time.time() - float(data.get("ts", 0)), 1)
        if data.get("state") in ("loading", "ready") and data["age"] > 30:
            data["state"] = "stopped"      # heartbeat went stale -> the helper died
        return data
    except Exception:
        return {"state": "off"}


def _reid_queue(cfg, species: str = "raccoon", limit: int = 30) -> dict:
    """The Individuals tab's review queue: recent visits of `species` with a WHO suggestion each
    (nearest confirmed visit / novelty flag / 2+-animals badge), the confirmed cast, and -- while
    nothing is confirmed yet -- the cold-start visit-groups to name first. Read-only; the heavy
    lifting (prototype matching over the embedding matrix) lives in individuals.VisitMatcher and
    is rebuilt per call, which is fine for a tab that loads on demand."""
    import individuals

    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"species": species, "queue": [], "cast": [], "bootstrap": []}
    try:
        matcher = individuals.VisitMatcher(conn, species, cfg)

        def _rep_crop(det_id):
            if det_id is None:
                return None
            r = conn.execute("SELECT crop_path FROM detections WHERE id = ?", (det_id,)).fetchone()
            return r["crop_path"].replace("\\", "/") if r and r["crop_path"] else None

        def _dwell(a, b):
            try:
                from datetime import datetime as _dt
                return int((_dt.fromisoformat(b) - _dt.fromisoformat(a)).total_seconds())
            except (ValueError, TypeError):
                return 0

        queue = []
        rows = conn.execute(
            """SELECT id, started_at, ended_at, detection_count, representative_detection_id
               FROM visits WHERE species = ? ORDER BY started_at DESC LIMIT ?""",
            (species, limit)).fetchall()
        for v in rows:
            s = matcher.suggest(v["id"])
            queue.append({
                "visit_id": v["id"], "started_at": v["started_at"],
                "dwell_s": _dwell(v["started_at"], v["ended_at"]),
                "n_crops": v["detection_count"],
                "rep_crop": _rep_crop(v["representative_detection_id"]),
                "confirmed_as": s["confirmed_as"], "candidates": s["candidates"],
                "clip_candidates": s["clip_candidates"],
                "novel": s["novel"], "multi": s["multi"],
                "co_present_frames": s["co_present_frames"],
                "co_present_clips": s["co_present_clips"],
                "n_embedded": s["n_embedded"], "note": s["note"], "species": species,
            })

        # The confirmed cast, with how much template material backs each name.
        cast: dict = {}
        for vid, name in matcher.confirmed.items():
            c = cast.setdefault(name, {"name": name, "n_visits": 0, "last_seen": None})
            c["n_visits"] += 1
            started = matcher.visit_started.get(vid)
            if started and (c["last_seen"] is None or started > c["last_seen"]):
                c["last_seen"] = started

        def _group_crops(visit_ids):
            if not visit_ids:
                return []
            reps = conn.execute(
                f"""SELECT representative_detection_id FROM visits
                    WHERE id IN ({','.join('?' * len(visit_ids))})""", visit_ids).fetchall()
            return [c for c in (_rep_crop(r["representative_detection_id"]) for r in reps) if c]

        has_templates = bool(matcher.templates())

        # Cold start only: until something is confirmed, naming visit-GROUPS beats naming visits.
        bootstrap = []
        if not has_templates:
            for g in matcher.bootstrap_groups():
                bootstrap.append({**g, "crops": _group_crops(g["visits"])})

        # Once there's a cast, RE-FIT: sort the unconfirmed remainder into "looks like <name>"
        # buckets (for bulk-confirm) + candidate-new-individual groups, and flag any confirmed
        # individual that has no clean solo template yet.
        refit = None
        if has_templates:
            r = matcher.refit()
            started = matcher.visit_started
            fits = {name: {"visits": [{**x, "rep_crop": _rep_crop(conn.execute(
                        "SELECT representative_detection_id FROM visits WHERE id=?",
                        (x["visit_id"],)).fetchone()[0])} for x in lst]}
                    for name, lst in r["fits"].items()}
            refit = {
                "fits": fits,
                "novel_groups": [{**g, "crops": _group_crops(g["visits"])}
                                 for g in r["novel_groups"]],
                "untemplated": r["untemplated"], "n_fit": r["n_fit"], "n_novel": r["n_novel"]}

        # Heads-up when suggestions are running blind: high-conf crops still missing vectors.
        backlog = conn.execute(
            """SELECT COUNT(*) FROM detections d
               WHERE d.species = ? AND d.confidence >= ? AND NOT EXISTS
                 (SELECT 1 FROM detection_embeddings e
                  WHERE e.detection_id = d.id AND e.model = ?)""",
            (species, cfg.reid_suggest_min_conf, individuals.EMBED_MODEL)).fetchone()[0]

        return {"species": species, "queue": queue,
                "cast": sorted(cast.values(), key=lambda c: -c["n_visits"]),
                "bootstrap": bootstrap, "refit": refit, "unembedded": backlog,
                "novel_threshold": cfg.reid_novel_threshold}
    finally:
        conn.close()


def _reid_unblend(cfg, visit_id: str) -> dict:
    """Separate a multi-animal visit into its individuals via clip-tracklet clustering."""
    import individuals
    try:
        vid = int(visit_id)
    except (TypeError, ValueError):
        return {"visit_id": None, "groups": [], "note": "bad visit_id"}
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"visit_id": vid, "groups": []}
    try:
        # Cluster suggestions use EXPLICIT un-blend labels only (clean per-animal templates that
        # separate the bonded pair -- validated 5/7), NOT the solo-visit-attributed blob (which is
        # coarse and, for a never-solo animal, impossible). So pass an empty solo map: the
        # suggestions stay quiet until the first cluster is labelled, then they sharpen.
        templates = individuals.clip_templates(conn, {}, cfg=cfg)
        return individuals.unblend_visit(conn, vid, templates=templates, cfg=cfg)
    finally:
        conn.close()


def _reid_poses(cfg, individual: str, top: int = 12) -> dict:
    """Characteristic poses of one confirmed individual: clusters of its crops by appearance
    embedding (identity fixed -> the embedding varies by pose/viewpoint). Returns the biggest
    `top` pose-groups, each with representative crop paths."""
    import individuals
    if not individual:
        return {"individual": "", "poses": []}
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"individual": individual, "poses": []}
    try:
        groups = individuals.pose_groups(conn, individual, cfg=cfg)
        return {"individual": individual, "n_groups": len(groups), "poses": groups[:top]}
    finally:
        conn.close()


def _reid_clips(cfg, individual: str) -> dict:
    """Behaviour clips attributable to one confirmed individual (via the visits they overlap)."""
    import individuals
    if not individual:
        return {"individual": "", "clips": []}
    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"individual": individual, "clips": []}
    try:
        clips = individuals.clips_for_individual(conn, individual)
        return {"individual": individual, "n_clips": len(clips), "clips": clips}
    finally:
        conn.close()


def _candidate_labels(cfg) -> list:
    """Species options for the correction dropdown: the classifier's label list plus anything
    already seen in the DB, de-duplicated and sorted."""
    labels = set()
    try:
        import classify
        labels.update(classify.SPECIES_LABELS)
    except Exception:
        pass
    conn = db.connect_readonly(cfg.db_path)
    if conn is not None:
        labels.update(r[0] for r in conn.execute(
            "SELECT DISTINCT species FROM detections WHERE species IS NOT NULL"))
        conn.close()
    return sorted(labels)


def start(cfg, frame_buffer: FrameBuffer, control_bridge: CameraControlBridge):
    server = make_server(cfg, frame_buffer, control_bridge)
    threading.Thread(target=server.serve_forever, name="webdash", daemon=True).start()
    return server


def shutdown(server) -> None:
    try:
        server.stop_event.set()
        server.shutdown()
        server.server_close()
    except Exception:
        pass
