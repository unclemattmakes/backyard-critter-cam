"""
Local web dashboard server (Python stdlib http.server -- no web framework).

Serves the live MJPEG stream, stats + species JSON, crop images, AND now:
  * live camera controls   -- GET/POST /api/camera  (web thread queues changes; the capture
    thread applies them, since OpenCV VideoCapture is single-thread -- see CameraControlBridge)
  * per-species browsing    -- GET /api/species, /api/species/<name>
  * ID verification         -- POST /api/detection/<id>  {action: verify|reject|correct, species}
  * ignore zones            -- GET/POST /api/zones, POST /api/zones/delete (dashboard-drawn
    static false-fire spots; IgnoreZoneStore hands edits to the capture threads live)

The page itself lives in dashboard.html (read from disk so the design can iterate without a
restart). Bound to localhost by default. No torch/cv2 here -- only stdlib + db/stats.
"""
from __future__ import annotations

import gzip
import ipaddress
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import behavior
import config
import db
import reel
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
    """Thread-safe holder of the latest annotated JPEG, with new-frame signalling.

    Also tracks whether anyone is WATCHING -- live stream clients register themselves, and a
    snapshot request flags interest for a few seconds -- so the capture thread can skip JPEG
    encoding entirely when nobody would see the result (see config.display_max_fps)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame: bytes | None = None
        self._seq = 0
        self._clients = 0
        self._want_until = 0.0

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

    def client_started(self) -> None:
        with self._cond:
            self._clients += 1

    def client_stopped(self) -> None:
        with self._cond:
            self._clients = max(0, self._clients - 1)

    def request_frame(self, horizon_s: float = 3.0) -> None:
        """A one-shot consumer (snapshot) wants fresh frames for the next few seconds."""
        with self._cond:
            self._want_until = time.monotonic() + horizon_s

    def watched(self) -> bool:
        with self._cond:
            return self._clients > 0 or time.monotonic() < self._want_until


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


class IgnoreZoneStore:
    """The live copy of the ignore_zones table (static false-fire spots the detector should
    disregard), shared between web threads and capture threads the same way CameraControlBridge
    is. The DB row is the durable truth -- it survives restarts and holds the tombstones that
    keep a deleted config zone deleted -- while this object is what the capture loop actually
    reads, so a dashboard edit reaches the detector on the next frame with no DB traffic on the
    capture path (this rig has been bitten by 'database is locked' in that loop before).

    Mutations write the DB first, then swap the in-memory tuple for that source, so a crash
    between the two can only lose the in-memory copy -- which the next startup reloads from the
    table anyway."""

    def __init__(self, db_path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._by_source: dict = {}   # source -> ((x1, y1, x2, y2), ...) LIVE zones only

    @classmethod
    def load(cls, cfg) -> "IgnoreZoneStore":
        """Build the store at startup: seed config.ignore_zones into the table (once per exact
        rectangle -- see db.seed_ignore_zones for why deletions stick), then cache the live rows."""
        store = cls(cfg.db_path)
        conn = db.connect(cfg.db_path)
        try:
            db.seed_ignore_zones(conn, getattr(cfg, "ignore_zones", None) or {})
            for row in db.list_ignore_zones(conn):
                cur = store._by_source.get(row["source"]) or ()
                store._by_source[row["source"]] = cur + ((row["x1"], row["y1"],
                                                          row["x2"], row["y2"]),)
        finally:
            conn.close()
        return store

    def rects(self, source) -> tuple:
        """The hot-path read: this camera's zones as an immutable tuple of (x1, y1, x2, y2).
        Called every capture-loop iteration, so it must stay a lock + attribute fetch."""
        with self._lock:
            return self._by_source.get(source) or ()

    def counts(self) -> dict:
        """{source: live zone count}, sources with zones only -- for the rig's startup banner."""
        with self._lock:
            return {s: len(t) for s, t in self._by_source.items() if t}

    def _reload_source(self, conn, source) -> None:
        """Re-read one source's live zones from the DB and swap them in wholesale -- after any
        mutation, memory is whatever the table says, not an incremental guess."""
        fresh = tuple((r["x1"], r["y1"], r["x2"], r["y2"])
                      for r in db.list_ignore_zones(conn, source))
        with self._lock:
            self._by_source[source] = fresh

    def add(self, source, x1, y1, x2, y2, note=None) -> dict:
        """Validate + insert one zone (see db.add_ignore_zone), refresh the cache, return the
        stored row. Raises ValueError on a degenerate rectangle."""
        conn = db.connect(self._db_path)
        try:
            row = db.add_ignore_zone(conn, source, x1, y1, x2, y2, note=note)
            self._reload_source(conn, source)
        finally:
            conn.close()
        return row

    def remove(self, zone_id) -> bool:
        """Soft-delete one zone by id; True if a live zone was removed."""
        conn = db.connect(self._db_path)
        try:
            row = db.remove_ignore_zone(conn, zone_id)
            if row is not None:
                self._reload_source(conn, row["source"])
        finally:
            conn.close()
        return row is not None


# A few controls have small, platform-stable valid sets; clamp those so a LAN/DNS-rebind client
# can't push a nonsensical value. EXPOSURE/GAIN/FOCUS/BRIGHTNESS/etc. use camera- and OS-specific
# scales (negative log2-seconds on Windows DirectShow, positive absolute units on V4L2), so we do
# NOT pin those -- the driver clamps them and the effect is transient anyway. Non-finite values
# (NaN / +-inf) are rejected for EVERY control below, which is the genuinely dangerous garbage.
_CONTROL_RANGES = {
    "AUTO_EXPOSURE": (0.0, 4.0), "AUTOFOCUS": (0.0, 1.0),
    "AUTO_WB": (0.0, 1.0), "BACKLIGHT": (0.0, 3.0),
}


def _clean_settings(data: dict) -> dict:
    out = {}
    for k, v in (data or {}).items():
        if k not in _ALLOWED_CONTROLS:
            continue
        if v is None:                           # exposure/gain: None = auto-expose
            out[k] = None
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        if val != val or val in (float("inf"), float("-inf")):   # drop NaN / +-inf
            continue
        lo, hi = _CONTROL_RANGES.get(k, (None, None))
        if lo is not None:
            val = max(lo, min(hi, val))
        out[k] = val
    return out


def _is_within(target: Path, parent: Path) -> bool:
    try:
        target.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_lan_client(host: str) -> bool:
    """True if `host` (a client's IP) is loopback or on a private/local network -- the only
    addresses allowed when the dashboard is bound to the LAN (cfg.lan_only). A public internet
    address returns False and the request is refused, so a forwarded port never exposes the rig to
    the world. Covers loopback, RFC1918 private ranges, link-local, and IPv4-mapped IPv6."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _host_name(raw: str) -> str:
    """The hostname portion of a Host header, minus any :port and IPv6 brackets."""
    h = (raw or "").strip()
    if h.startswith("["):                      # [::1]:8000 -> ::1
        return h[1:].split("]", 1)[0]
    if h.count(":") == 1:                       # host:port -> host (a bare IPv6 literal has >1 colon)
        h = h.split(":", 1)[0]
    return h


def _is_allowed_host(raw: str, web_host: str = "") -> bool:
    """DNS-rebinding guard. The peer-IP check (_is_lan_client) can't stop a malicious site that
    resolves its OWN name to this rig's LAN IP: the request then comes from the victim's own (local)
    browser, so the peer IP looks fine, but the Host header still carries the ATTACKER's hostname.
    Accept only a Host that is 'localhost', a loopback/private/link-local IP literal, or the
    operator's configured web_host. A real browser/curl always sends Host, so a rebinding fetch
    can't omit it; an absent Host (rare, HTTP/1.0 tooling) is allowed through."""
    if not raw:
        return True
    host = _host_name(raw).lower()
    if host == "localhost" or host == str(web_host or "").lower():
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _origin_authority(raw: str):
    """(host, port) for an Origin header like 'http://192.168.1.50:8000', or None if it isn't a
    plain http(s) origin we can parse -- which covers the literal 'null' a sandboxed iframe or a
    file:// page sends, and any garbage."""
    u = urllib.parse.urlsplit((raw or "").strip())
    if u.scheme not in ("http", "https"):
        return None
    try:
        host, port = u.hostname, u.port
    except ValueError:                          # non-numeric port
        return None
    if not host:
        return None
    return host.lower(), port or (443 if u.scheme == "https" else 80)


def _host_authority(raw: str, default_port):
    """(host, port) for a Host header -- 'localhost', '192.168.1.50:8000', '[::1]:8000'. Borrows
    urlsplit's authority parser so the IPv6 brackets and the :port suffix are handled once."""
    u = urllib.parse.urlsplit("//" + (raw or "").strip())
    try:
        host, port = u.hostname, u.port
    except ValueError:
        return None
    if not host:
        return None
    return host.lower(), port or default_port


def _is_same_origin(origin: str, host_header: str, web_host: str = "", web_port=0) -> bool:
    """True only if `origin` names THIS dashboard. A cross-site page's Origin is its own name, so
    it never matches -- which is the whole point: the peer-IP and Host checks both PASS for a
    request the operator's own browser makes while sitting on someone else's site.
    Accepts the configured web_host, loopback, and -- the LAN case -- whatever name or IP the
    request's own Host header carries, since a browser only sends an Origin equal to its Host when
    it loaded the page from this very server. The Host branch is gated on _is_allowed_host so a
    DNS-rebinding name can't satisfy both headers at once."""
    got = _origin_authority(origin)
    if got is None:
        return False
    port = int(web_port or 0)
    allowed = {("localhost", port), ("127.0.0.1", port), ("::1", port)}
    wh = str(web_host or "").strip().lower()
    if wh and wh not in ("0.0.0.0", "::"):      # a wildcard bind isn't a name a browser can send
        allowed.add((wh, port))
    if _is_allowed_host(host_header, web_host):
        mine = _host_authority(host_header, got[1])
        if mine is not None:
            allowed.add(mine)
    return got in allowed


def _is_json_content_type(raw: str) -> bool:
    """True for 'application/json', with an optional '; charset=...' parameter. Requiring it on
    POST closes the no-preflight path on its own: without CORS a cross-origin fetch may only send
    text/plain, x-www-form-urlencoded or multipart/form-data, and the preflight that application/json
    triggers is one we never answer."""
    return (raw or "").split(";", 1)[0].strip().lower() == "application/json"


def _csrf_refusal(method: str, headers, web_host: str = "", web_port=0):
    """The 403 body for a state-changing request that didn't come from this dashboard, or None if it
    may proceed. GET/HEAD change nothing and pass untouched; every POST -- present and future --
    must carry an Origin naming us and a JSON Content-Type. A MISSING Origin is refused too: fetch
    and XHR always send one, so only hand-rolled tooling lacks it, and a curl user just adds
    -H 'Origin: http://127.0.0.1:8000' -H 'Content-Type: application/json'.
    Without this, any page the operator happens to be browsing could fetch
    /api/individual with {"from": "Notch", "to": ""} and blank months of hand-confirmed re-ID
    labels -- no preflight, no confirm, no undo. `headers` is anything with a .get()."""
    if method != "POST":
        return None
    if not _is_same_origin(headers.get("Origin"), headers.get("Host"), web_host, web_port):
        return b"forbidden: POST needs an Origin naming this dashboard (cross-site request guard)"
    if not _is_json_content_type(headers.get("Content-Type")):
        return b"forbidden: POST needs Content-Type: application/json (cross-site request guard)"
    return None


# Media types served from /media (crops, frames, and now behaviour clips). Video needs a correct
# Content-Type AND HTTP Range support, or a browser <video> won't stream or seek.
_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
}
_RANGE_CHUNK = 4 * 1024 * 1024   # cap an open-ended range so a clip is never read whole into RAM
_MAX_POST_BYTES = 1 << 20        # dashboard POST bodies are tiny JSON; reject anything larger (413)
_MAX_STREAMS = 6                 # base cap on concurrent MJPEG viewers (one LAN client can't exhaust
                                 # threads); make_server scales it up with the number of cameras.
_MAX_ZONES_PER_CAMERA = 50       # every live zone costs an IoU per detection and a rectangle per
                                 # streamed frame; a runaway client must not be able to stack up
                                 # thousands. 50 is far past any real yard's furniture count.


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
_CACHE_MAX = 4096              # bound the two caches above so a long session can't grow them forever


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
            val = (r.stdout or "").strip()
        except Exception:
            val = ""
        if len(_codec_cache) >= _CACHE_MAX:           # bound the cache over a long-running session
            _codec_cache.pop(next(iter(_codec_cache)), None)
        _codec_cache[key] = val
    return _codec_cache[key] == "h264"


def _lock_for(key: str) -> threading.Lock:
    """One lock per output path, so concurrent range requests for the same clip transcode once."""
    with _transcode_guard:
        lk = _transcode_locks.get(key)
        if lk is None:
            if len(_transcode_locks) >= _CACHE_MAX:   # bound the lock table over a long session
                _transcode_locks.pop(next(iter(_transcode_locks)), None)
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


# ---------------------------------------------------------------------------
# API response cache. The dashboard polls /api/stats + /api/species every 6
# seconds, and compute_stats is a full-table scan + visit clustering (~1.3s of
# CPU at 130k detections) -- polling was most of what this box spent its day
# on, on the same machine that runs the detector. Entries are keyed by a cheap
# DB-content signature (size + mtime of the .db and its -wal), so an idle
# database serves cached JSON indefinitely and any external write invalidates
# it; while capture IS writing, the signature churns with every detection, so a
# hold-down additionally caps rebuilds at one per _API_HOLD_S -- the Live tab
# lags a busy visit by at most that. Label/name POSTs clear the whole cache
# (do_POST), so a human correction always shows on the very next fetch.
# ---------------------------------------------------------------------------

_API_CACHE: dict = {}          # name -> (db_sig, built_at, payload)
_api_cache_guard = threading.Lock()
_API_HOLD_S = 10.0


def _db_sig(cfg) -> tuple:
    """(size, mtime_ns) of the database and its WAL -- changes whenever any writer commits
    (WAL append) or a checkpoint runs. Two stats instead of a query: no connection needed."""
    sig = []
    for suffix in ("", "-wal"):
        try:
            st = os.stat(f"{cfg.db_path}{suffix}")
            sig.append((st.st_size, st.st_mtime_ns))
        except OSError:
            sig.append(None)
    return tuple(sig)


def clear_api_cache() -> None:
    with _api_cache_guard:
        _API_CACHE.clear()


def _cached(cfg, name: str, build, hold_s: float = _API_HOLD_S):
    """`build()`'s result, rebuilt only when the DB signature changed AND the entry is older
    than `hold_s`. One build at a time per name (the lock table), so two browsers polling in
    step can't both pay for the same full-table scan."""
    sig = _db_sig(cfg)

    def fresh():
        hit = _API_CACHE.get(name)
        if hit is None:
            return None
        h_sig, built_at, payload = hit
        return payload if (h_sig == sig or (time.time() - built_at) < hold_s) else None

    with _api_cache_guard:
        payload = fresh()
    if payload is not None:
        return payload
    with _lock_for("api:" + name):
        with _api_cache_guard:                     # another thread may have just built it
            payload = fresh()
        if payload is not None:
            return payload
        payload = build()
        with _api_cache_guard:
            if len(_API_CACHE) >= 64:              # bound it (per-individual profile keys)
                _API_CACHE.pop(next(iter(_API_CACHE)), None)
            _API_CACHE[name] = (_db_sig(cfg), time.time(), payload)
        return payload


# ---------------------------------------------------------------------------
# Archived (soft-pruned) clips. backup.py zips each day of clips/ into
# <backup_dest>/clips/ -- one zip per day (legacy flat layout) or per camera-day
# (multi-camera layout) -- with members STORED (uncompressed) under arcnames
# identical to the DB's clip_path. A clip whose video the disk budget pruned
# (clips.pruned_at set) therefore usually still exists inside that day's zip.
# These helpers find it, restore it into a small local LRU cache, and let the
# ordinary clip-serving path (H.264 transcode + range requests) take over --
# so "in the archive" means playable, not a dead link.
# ---------------------------------------------------------------------------

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ARCHIVE_ROOT = config.ROOT / "archive_cache"   # restored under clips/, transcodes under web/
_ARCHIVE_CACHE_KEEP = 40                        # restored mp4s kept (LRU); the zips keep the originals


def _archive_zip_for(clip_path: str):
    """(zip filename, member arcname) for a clip_path, or None when the path doesn't follow a
    known day layout. Mirrors backup.day_dirs' naming exactly:
        clips/<date>/x.mp4          -> clips-<date>.zip
        clips/<source>/<date>/x.mp4 -> clips-<source>-<date>.zip"""
    parts = [p for p in (clip_path or "").replace("\\", "/").split("/") if p]
    if len(parts) == 3 and parts[0] == "clips" and _DAY_RE.match(parts[1]):
        return f"clips-{parts[1]}.zip", "/".join(parts)
    if len(parts) == 4 and parts[0] == "clips" and _DAY_RE.match(parts[2]):
        return f"clips-{parts[1]}-{parts[2]}.zip", "/".join(parts)
    return None


def _prune_archive_cache(cache_root: Path, keep: int = _ARCHIVE_CACHE_KEEP) -> None:
    """Cap the restored-clip cache to the `keep` most-recently-used mp4s (restored originals AND
    their transcodes). Deleting is always safe: the zips still hold the originals, so a re-click
    just restores again. Per-file try because Windows refuses to unlink a file mid-serve."""
    try:
        files = sorted(cache_root.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    for p in files[keep:]:
        try:
            p.unlink()
        except OSError:
            pass


def _restore_archived_clip(backup_dest, clip_path: str, cache_root: Path):
    """Copy one pruned clip out of its backup zip into cache_root/<clip_path>. Returns the
    restored Path, or None (no backup dest configured / zip or member missing / unrecognized
    layout). Members are stored uncompressed, so this is a straight file copy."""
    hit = _archive_zip_for(clip_path)
    if hit is None:
        return None
    zip_name, member = hit
    out = (cache_root / member).resolve()
    if not _is_within(out, cache_root.resolve()):   # belt & braces; member comes from our own DB
        return None
    try:
        if out.is_file() and out.stat().st_size > 0:
            out.touch()                             # LRU bump
            return out
    except OSError:
        pass
    if not backup_dest:
        return None
    zpath = Path(backup_dest) / "clips" / zip_name
    if not zpath.is_file():
        return None
    with _lock_for("archive:" + member):            # concurrent range requests restore once
        try:
            if out.is_file() and out.stat().st_size > 0:
                return out
            with zipfile.ZipFile(zpath) as zf, zf.open(member) as src:
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp.mp4")
                with open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                tmp.replace(out)                    # atomic publish, like the transcode cache
        except (KeyError, OSError, zipfile.BadZipFile):
            return None
    _prune_archive_cache(cache_root)
    return out


def _individual_profile(cfg, name: str) -> dict:
    """stats.individual_profile + a web-layer annotation: for each archived clip, whether it is
    actually reachable right now (`archive_ok` -- already restored locally, or its day's zip is
    present on the backup drive). The dashboard then shows an unreachable prune as gone instead
    of promising a play that would 404."""
    prof = stats.individual_profile(cfg, name)
    if not prof.get("found"):
        return prof
    have = None                                     # zip names on the backup drive, listed once
    for v in prof.get("visits", []):
        for c in (v.get("clips") or []):
            if not (c and c.get("archived")):
                continue
            if have is None:
                dest = getattr(cfg, "backup_dest", None)
                try:
                    have = {p.name for p in (Path(dest) / "clips").iterdir()} if dest else set()
                except OSError:
                    have = set()
            hit = _archive_zip_for(c.get("clip_path") or "")
            restored = hit and (_ARCHIVE_ROOT / hit[1]).is_file()
            c["archive_ok"] = bool(restored or (hit and hit[0] in have))
    return prof


def _zones_payload(cfg, source: str, bridge_snap: dict) -> dict:
    """GET /api/zones: this camera's live ignore zones, plus what the editor needs -- the true
    frame size when the capture thread has published one (the drawing surface maps 1:1 to the
    snapshot, so this is informational), and per-zone `stale`: True when the camera has been seen
    to move (view_epochs) since the zone was drawn. Stale is the whole failure mode of a
    hand-measured rectangle -- it keeps 'ignoring' a patch of scene that is no longer there --
    and it fails silently, so the one place a human looks at zones must say it out loud."""
    conn = db.connect(cfg.db_path)
    try:
        rows = db.list_ignore_zones(conn, source)
        moved_at = db.view_epoch_started(conn, source)
    finally:
        conn.close()
    moved = db.parse_local(moved_at) if moved_at else None
    for r in rows:
        created = db.parse_local(r["created_at"])
        r["stale"] = bool(moved and created and created < moved)
    fw, fh = bridge_snap.get("frame_w"), bridge_snap.get("frame_h")
    return {"source": source, "zones": rows,
            "frame": {"w": fw, "h": fh} if fw and fh else None,
            "iou": getattr(cfg, "ignore_zone_iou", 0.45)}


def make_server(cfg, frame_buffers: dict, control_bridges: dict, zone_store=None):
    """`frame_buffers` / `control_bridges` are dicts keyed by camera `source` -- one entry per
    live camera. A single-camera rig passes one-entry dicts; the Live tab then shows one pane.
    The dashboard discovers the cameras via /api/cameras and routes /stream.mjpg, /snapshot.jpg,
    /api/camera and /api/live/* with a ?source= query param (defaulting to the primary camera, so
    an old client with no source param still works).

    `zone_store` is the rig's shared IgnoreZoneStore (the capture threads read the same instance,
    so a zone edit takes effect on the next frame). None -- tests, or any caller without a rig --
    builds a private one from the DB; edits then persist but nothing live is watching them."""
    if zone_store is None:
        zone_store = IgnoreZoneStore.load(cfg)
    allowed_dirs = [d.resolve() for d in (cfg.crops_dir, cfg.frames_dir, cfg.clips_dir,
                                          getattr(cfg, "clip_crops_dir", cfg.clips_dir),
                                          # refcam's detector crops of the phone reference shots
                                          # (the individual profile's reference gallery). The raw
                                          # originals live OUTSIDE the project and stay unserved.
                                          getattr(cfg, "reference_crops_dir", cfg.crops_dir))]
    stop_event = threading.Event()

    # The primary camera (the Live tab's default / "Plate I"): the one matching cfg.source if it's
    # among the live cameras, else the first one. Insertion order of frame_buffers = camera order.
    primary = cfg.source if cfg.source in frame_buffers else next(iter(frame_buffers))
    _specs = {s.source: s for s in cfg.camera_specs()}
    _cam_order = [primary] + [s for s in frame_buffers if s != primary]
    cameras_meta = [{"source": s, "primary": s == primary,
                     "name": (_specs[s].display_name if s in _specs else s),
                     "network": bool(_specs[s].is_network) if s in _specs else False}
                    for s in _cam_order]
    # Each live pane opens its own MJPEG stream, so scale the concurrent-stream cap with the camera
    # count (the flat _MAX_STREAMS=6 was sized for one camera + a few viewers).
    stream_slots = threading.BoundedSemaphore(max(_MAX_STREAMS, 4 * len(frame_buffers)))

    class Handler(BaseHTTPRequestHandler):
        timeout = 30   # drop a stalled / slowloris connection instead of blocking a worker forever

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
            body = json.dumps(obj).encode("utf-8")
            extra = {"Cache-Control": "no-store"}
            # The big payloads (visits ~500 KB, stats ~90 KB) are highly compressible JSON;
            # gzip cuts them ~10x, which is what a phone on the far side of the house feels.
            if len(body) > 2048 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                body = gzip.compress(body, 5)
                extra["Content-Encoding"] = "gzip"
                extra["Vary"] = "Accept-Encoding"
            self._send(code, "application/json", body, extra)

        def _src(self):
            """The camera `source` this request targets, from a ?source= query param. Falls back
            to the primary camera for an unknown/absent value, so an old client (or a hand-typed
            /stream.mjpg with no param) still gets the main feed instead of a 404."""
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            s = (q.get("source") or [None])[0]
            return s if s in frame_buffers else primary

        def _lan_guard(self) -> bool:
            """Refuse non-local clients when bound to the network (cfg.lan_only). Returns True if
            the request may proceed. Localhost is always allowed; on the LAN launcher this keeps the
            dashboard reachable from your own devices but invisible to the wider internet.
            Both do_GET and do_POST call this first, so the cross-site check below covers every
            mutating endpoint by construction -- including the ones nobody has written yet."""
            # Runs before the lan_only shortcut: a rig deliberately opened past the LAN still must
            # not take POSTs from other people's pages.
            refusal = _csrf_refusal(self.command, self.headers,
                                    getattr(cfg, "web_host", ""), getattr(cfg, "web_port", 0))
            if refusal is not None:
                self._send(403, "text/plain", refusal)
                return False
            if not getattr(cfg, "lan_only", True):
                return True
            host = self.client_address[0] if self.client_address else ""
            if not _is_lan_client(host):
                self._send(403, "text/plain",
                           b"forbidden: this dashboard only accepts connections from your local network")
                return False
            # Peer is local -- but also validate the Host header, so a malicious site can't use DNS
            # rebinding (its name -> this rig's LAN IP) to drive the dashboard from your own browser.
            if not _is_allowed_host(self.headers.get("Host"), getattr(cfg, "web_host", "")):
                self._send(403, "text/plain",
                           b"forbidden: unrecognized Host header (DNS-rebinding guard)")
                return False
            return True

        def do_GET(self):
            if not self._lan_guard():
                return
            path = urllib.parse.urlparse(self.path).path
            # The three UI files are read fresh from disk per request so the design iterates
            # without a restart -- tell the BROWSER the same (no-cache = revalidate each load),
            # or an updated dashboard keeps rendering with last week's cached css/js.
            _fresh = {"Cache-Control": "no-cache"}
            try:
                if path in ("/", "/index.html"):
                    if DASHBOARD_FILE.exists():
                        self._send(200, "text/html; charset=utf-8", DASHBOARD_FILE.read_bytes(), _fresh)
                    else:
                        self._send(500, "text/plain", b"dashboard.html missing")
                elif path == "/dashboard.css":
                    # Stylesheet split out of dashboard.html (read fresh from disk, like the HTML,
                    # so the design still iterates without a server restart).
                    css = config.ROOT / "dashboard.css"
                    if css.exists():
                        self._send(200, "text/css; charset=utf-8", css.read_bytes(), _fresh)
                    else:
                        self._send(404, "text/plain", b"dashboard.css missing")
                elif path == "/dashboard.js":
                    # Behaviour script, likewise split out of dashboard.html and read fresh per
                    # request (no restart needed to iterate). The HTML loads it via <script src>.
                    js = config.ROOT / "dashboard.js"
                    if js.exists():
                        self._send(200, "application/javascript; charset=utf-8", js.read_bytes(), _fresh)
                    else:
                        self._send(404, "text/plain", b"dashboard.js missing")
                elif path == "/api/stats":
                    self._json(_cached(cfg, "stats",
                                       lambda: stats.compute_stats(cfg) or {"total_crops": 0}))
                elif path == "/api/species":
                    self._json(_cached(cfg, "species",
                                       lambda: stats.species_overview(cfg) or {"species": [], "total": 0}))
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
                    self._json(control_bridges[self._src()].snapshot())
                elif path == "/api/cameras":
                    # The live cameras, for the dashboard to build one feed pane per camera.
                    self._json({"cameras": cameras_meta, "primary": primary})
                elif path == "/api/zones":
                    src = self._src()
                    self._json(_zones_payload(cfg, src, control_bridges[src].snapshot()))
                elif path == "/api/naming":
                    self._json(_naming_status())
                elif path == "/api/live/now":
                    self._json(_live_now(cfg, self._src()))
                elif path == "/api/crops":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(stats.crops_page(
                        cfg, day=(q.get("day") or [None])[0], species=(q.get("species") or [None])[0],
                        start=(q.get("start") or [None])[0], end=(q.get("end") or [None])[0],
                        offset=max(0, int((q.get("offset") or [0])[0])),
                        limit=max(1, min(int((q.get("limit") or [60])[0]), 200)),
                        individual=(q.get("individual") or [None])[0]))
                elif path == "/api/individual/profile":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    nm = (q.get("name") or [""])[0]
                    self._json(_cached(cfg, "profile:" + nm,
                                       lambda: _individual_profile(cfg, nm), hold_s=30))
                elif path == "/api/visits":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    day = (q.get("day") or [None])[0]
                    if day:                    # day views are rare one-offs; cache only the default
                        self._json(stats.visits_page(cfg, day=day))
                    else:
                        self._json(_cached(cfg, "visits", lambda: stats.visits_page(cfg)))
                elif path == "/api/digest":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(stats.period_digest(cfg, edition=(q.get("edition") or ["auto"])[0],
                                                   date=(q.get("date") or [None])[0]))
                elif path == "/api/reel":
                    # The condensed highlight reel for a period: ready (manifest) | building |
                    # empty | unavailable | failed. Building happens in a background thread.
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(reel.reel_status(cfg, edition=(q.get("edition") or ["auto"])[0],
                                                date=(q.get("date") or [None])[0]))
                elif path == "/api/behavior":
                    self._json(_cached(cfg, "behavior", lambda: behavior.overview(cfg), hold_s=30))
                elif path == "/api/individuals":
                    self._json(_cached(cfg, "individuals", lambda: stats.individuals_overview(cfg)))
                elif path == "/api/rollcall":
                    self._json(_cached(cfg, "rollcall", lambda: stats.cast_rollcall(cfg), hold_s=30))
                elif path == "/api/visit/motion":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    try:                        # a bad/missing visit_id => empty shape, not a 500
                        vid = int((q.get("visit_id") or [""])[0])
                    except (TypeError, ValueError):
                        self._send(400, "text/plain", b"bad visit_id")
                    else:
                        self._json(stats.visit_motion(cfg, vid))
                elif path == "/api/individual/motion":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    self._json(stats.individual_motion(cfg, (q.get("individual") or [""])[0]))
                elif path == "/api/review":
                    self._json(_cached(cfg, "review", lambda: stats.review_queue(cfg), hold_s=30))
                elif path == "/api/reid/queue":
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    sp = (q.get("species") or [cfg.reid_species])[0]
                    lim = _qs_int(q, "limit", 30, 1, 100)
                    off = _qs_int(q, "offset", 0, 0, 1_000_000)
                    md = (q.get("mode") or ["recent"])[0]
                    # The heaviest endpoint there is (~10s: a full VisitMatcher rebuild), so it
                    # leans hardest on the cache. Label POSTs clear it, so the queue is always
                    # fresh right after a confirm -- the rebuild then happens once, not per open.
                    self._json(_cached(cfg, f"reidq:{sp}:{lim}:{off}:{md}",
                                       lambda: _reid_queue(cfg, species=sp, limit=lim,
                                                           offset=off, mode=md), hold_s=60))
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
                    fbuf = frame_buffers[self._src()]
                    _, seq = fbuf.get()
                    # Flag interest so the capture thread starts encoding, then wait for a FRESH
                    # frame (encoding is skipped while unwatched, so the stored one may be old).
                    # On timeout fall back to whatever is stored rather than failing the request.
                    fbuf.request_frame()
                    frame, _ = fbuf.wait(seq, timeout=1.5)
                    if frame is None:
                        self._send(503, "text/plain", b"no frame yet")
                    else:
                        self._send(200, "image/jpeg", frame, {"Cache-Control": "no-store"})
                elif path == "/stream.mjpg":
                    self._stream(frame_buffers[self._src()])
                elif path.startswith("/archive/clip/"):
                    self._archived_clip(path[len("/archive/clip/"):])
                elif path.startswith("/media/"):
                    self._media(path[len("/media/"):])
                else:
                    self._send(404, "text/plain", b"not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:                            # log detail server-side; don't leak it to the client
                    print(f"[web] GET {self.path} failed: {e}")
                except Exception:
                    pass
                try:
                    self._send(500, "text/plain", b"internal error")
                except Exception:
                    pass

        def do_POST(self):
            if not self._lan_guard():
                return
            path = urllib.parse.urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except (TypeError, ValueError):     # a non-numeric Content-Length must not 500 us
                length = 0
            if length > _MAX_POST_BYTES:
                self._send(413, "text/plain", b"payload too large")
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw or b"{}")
            except Exception:
                data = {}
            try:
                if path not in ("/api/camera", "/api/zones", "/api/zones/delete"):
                    stats.clear_digest_cache()   # label/name edits must show on the next Dispatch
                    clear_api_cache()            # ...and on the next poll of every cached endpoint
                if path == "/api/camera":
                    control_bridges[self._src()].request(_clean_settings(data))
                    self._json({"ok": True})
                elif path == "/api/zones":
                    self._zone_add(data)
                elif path == "/api/zones/delete":
                    self._zone_delete(data)
                elif path == "/api/individual/status":
                    self._individual_status(data)
                elif path == "/api/individual":
                    self._individual_action(data)
                elif path == "/api/reid/confirm":
                    self._reid_confirm(data)
                elif path == "/api/visit/label":
                    self._visit_label(data)
                elif path == "/api/live/sighting":
                    self._live_sighting(data)
                elif path == "/api/reid/unblend/label":
                    self._unblend_label(data)
                elif path.startswith("/api/detection/"):
                    self._detection_action(int(path.rsplit("/", 1)[-1]), data)
                else:
                    self._send(404, "text/plain", b"not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:                            # log detail server-side; don't leak it to the client
                    print(f"[web] POST {self.path} failed: {e}")
                except Exception:
                    pass
                try:
                    self._json({"error": "internal error"}, code=500)
                except Exception:
                    pass

        def _zone_add(self, data):
            """Add an ignore zone for the ?source= camera: {"x1","y1","x2","y2", "note"?} in
            FULL-RES frame pixels (the snapshot the dashboard draws on IS the full frame, so its
            naturalWidth maps 1:1). The shared store makes it live on the next captured frame."""
            src = self._src()
            if len(zone_store.rects(src)) >= _MAX_ZONES_PER_CAMERA:
                self._json({"error": f"too many zones (limit {_MAX_ZONES_PER_CAMERA}); "
                                     "remove one first"}, code=400)
                return
            try:
                x1, y1, x2, y2 = (float(data[k]) for k in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError):
                self._json({"error": "x1, y1, x2, y2 (numbers) are required"}, code=400)
                return
            # Clamp to the frame when the capture thread has told us its true size; a drag can
            # end a few px past the image edge and should land ON the edge, not 400.
            snap = control_bridges[src].snapshot()
            fw, fh = snap.get("frame_w"), snap.get("frame_h")
            if fw and fh:
                x1, x2 = (min(max(v, 0), fw) for v in (x1, x2))
                y1, y2 = (min(max(v, 0), fh) for v in (y1, y2))
            try:
                row = zone_store.add(src, x1, y1, x2, y2, note=data.get("note"))
            except ValueError as e:
                self._json({"error": str(e)}, code=400)
                return
            self._json({"ok": True, "zone": row})

        def _zone_delete(self, data):
            """Soft-delete one zone: {"id": n}. The tombstoned row keeps the config seed from
            resurrecting it on the next restart (see db.remove_ignore_zone)."""
            try:
                zid = int(data.get("id"))
            except (TypeError, ValueError):
                self._json({"error": "id (integer) is required"}, code=400)
                return
            removed = zone_store.remove(zid)
            self._json({"ok": removed} if removed
                       else {"ok": False, "error": "no such zone"}, code=200 if removed else 404)

        def _unblend_label(self, data):
            """Assign an individual to an un-blend cluster: {"track_ids": [...], "name": "Notch"}
            labels clip TRACKLETS (clip-space, the clips basis); {"detection_ids": [...], "name":
            ...} stamps still CROPS directly (individual_source='human' -- the stills basis, whose
            labels survive visit renumbering like every detection-level label). name=""/null
            clears either way."""
            tids = data.get("track_ids") or []
            dids = data.get("detection_ids") or []
            tids = tids if isinstance(tids, list) else []
            dids = dids if isinstance(dids, list) else []
            if not tids and not dids:
                self._json({"error": "missing track_ids or detection_ids"}, code=400)
                return
            name = (str(data.get("name") or "").strip()) or None
            conn = db.connect(cfg.db_path)
            try:
                n = 0
                if tids:
                    n += db.set_clip_track_individual(conn, [int(t) for t in tids], name)
                if dids:
                    n += db.set_individual_bulk(conn, [int(i) for i in dids], name)
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

        def _live_sighting(self, data):
            """Log who's visiting RIGHT NOW from the Live tab. Body: {names: [...], note?}. The
            visit span is resolved server-side from the live source's recent detections (the client
            only knows the names it recognises), so a stale client clock can't mislabel the span.
            One name also stamps that individual onto the span (a live solo confirm, feeding the
            re-ID templates); two+ names record co-presence only (no single contaminating stamp --
            the documented pair gotcha). Deliberately does NOT rebuild visits."""
            names = data.get("names")
            if isinstance(names, str):
                names = [names]
            if not isinstance(names, list) or not any(str(n).strip() for n in names):
                self._json({"error": "need at least one name"}, code=400)
                return
            # Which camera the user is naming, from the POST body (the Live tab sends the pane it's
            # watching). Falls back to the primary camera for an unknown/absent value.
            source = data.get("source")
            source = source if source in frame_buffers else primary
            visit = stats.current_live_visit(cfg, source)
            conn = db.connect(cfg.db_path)
            try:
                res = db.record_live_sighting(
                    conn, source=source, names=names,
                    span_start=visit.get("start"), span_end=visit.get("end"),
                    note=data.get("note"))
                if res.get("error"):
                    self._json({"error": res["error"]}, code=400)
                    return
                self._json({"ok": True, "visit": visit, **res})
            finally:
                conn.close()

        def _reid_confirm(self, data):
            """Confirm (or clear) WHO one visit was: {"visit_id": 1014, "name": "Stan"}.
            name=""/null clears; add "reject": true to clear AND leave the human's "not them"
            tombstone (individual_source 'human' with a NULL id), which stops the nightly
            auto-assign pass from re-naming the visit. Stamps the visit's species-matching crops
            (individual_source='human') and mirrors the name onto the visits row. Deliberately
            does NOT rebuild the visits table -- a rebuild renumbers visit ids and would
            invalidate every other card in the open review queue; labels live on detections, so
            the next rebuild inherits them anyway."""
            try:
                vid = int(data.get("visit_id"))
            except (TypeError, ValueError):
                self._json({"error": "missing/bad 'visit_id'"}, code=400)
                return
            name = (data.get("name") or "").strip() or None
            reject = bool(data.get("reject")) and name is None
            conn = db.connect(cfg.db_path)
            try:
                n = db.label_visit(conn, vid, name, reject=reject)
                self._json({"ok": True, "visit_id": vid, "name": name, "stamped": n,
                            "rejected": reject})
            finally:
                conn.close()

        def _individual_status(self, data):
            """THE ROSTER: record that an individual has left the yard, or take it back.
            {"name": "Notch", "status": "departed", "effective_date": "2026-06-30"} -- the date is
            the LAST DAY it was here. {"status": "resident"} undoes it.

            Changes exactly one thing: the nightly auto-assign pass stops writing that name onto
            visits that started AFTER that date (individuals.VisitMatcher.is_departed). Nothing is
            hidden, no label is touched, and older visits stay auto-nameable -- the animal really
            was here then. The date is what makes that possible, so it is not optional in spirit;
            omitting it fails closed (no auto-name at any time) rather than open."""
            conn = db.connect(cfg.db_path)
            try:
                row = db.set_individual_status(
                    conn, (data.get("name") or ""),
                    status=(data.get("status") or "departed"),
                    effective_date=data.get("effective_date"), note=data.get("note"))
            except ValueError as e:      # empty name / unknown status / unparseable date -> 400
                self._json({"error": str(e)}, code=400)
                return
            finally:
                conn.close()
            self._json({"ok": True, **row})

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

        def _stream(self, frame_buffer):
            if not stream_slots.acquire(blocking=False):
                self._send(503, "text/plain", b"too many active streams; try again shortly")
                return
            frame_buffer.client_started()   # capture thread encodes only while watched
            try:
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
            finally:
                frame_buffer.client_stopped()
                stream_slots.release()

        def _archived_clip(self, id_str):
            """Serve one clip by DB id, wherever its bytes now live: still on disk -> the
            ordinary path; soft-pruned -> restored out of that day's backup zip into
            archive_cache/, then transcoded/range-served like any other clip. The URL is
            stable across the prune, so a saved link keeps working after the disk budget
            eats the local copy."""
            try:
                cid = int(id_str)
            except (TypeError, ValueError):
                self._send(400, "text/plain", b"bad clip id")
                return
            conn = db.connect_readonly(cfg.db_path)
            if conn is None:
                self._send(503, "text/plain", b"no database yet")
                return
            try:
                row = conn.execute("SELECT clip_path FROM clips WHERE id = ?", (cid,)).fetchone()
            finally:
                conn.close()
            if not row or not row["clip_path"]:
                self._send(404, "text/plain", b"no such clip")
                return
            clip_path = row["clip_path"].replace("\\", "/")
            live = (config.ROOT / clip_path).resolve()
            clips_root = cfg.clips_dir.resolve()
            if _is_within(live, clips_root) and live.is_file():
                self._serve_video(live, clips_root, clips_root.parent / "clips_web")
                return
            restored = _restore_archived_clip(getattr(cfg, "backup_dest", None),
                                              clip_path, _ARCHIVE_ROOT)
            if restored is None:
                self._send(404, "text/plain",
                           b"this clip was pruned and its archive isn't reachable "
                           b"(backup drive missing, or the clip predates the backups)")
                return
            self._serve_video(restored, _ARCHIVE_ROOT / "clips", _ARCHIVE_ROOT / "web")

        def _serve_video(self, target: Path, clips_root: Path, cache_root: Path):
            """Range-serve a clip, preferring its cached H.264 transcode (mp4v-era clips are
            browser-undecodable; new clips are H.264 already and pass straight through)."""
            serve = target
            web_ver = _web_clip(target, clips_root, cache_root)
            if web_ver is not None:
                serve = web_ver
            self._serve_file(serve, "video/mp4")

        def _media(self, rel):
            target = (config.ROOT / urllib.parse.unquote(rel)).resolve()
            if not any(_is_within(target, d) for d in allowed_dirs) or not target.is_file():
                self._send(404, "text/plain", b"not found")
                return
            ctype = _MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
            # Clips are recorded as mp4v (browser-undecodable); serve a cached H.264 transcode.
            if target.suffix.lower() == ".mp4":
                clips_root = cfg.clips_dir.resolve()
                if _is_within(target, clips_root):
                    self._serve_video(target, clips_root, clips_root.parent / "clips_web")
                    return
            self._serve_file(target, ctype)

        def _serve_file(self, serve: Path, ctype: str):
            size = serve.stat().st_size
            rng = self.headers.get("Range")
            if not rng:
                # Whole file. Advertise Accept-Ranges so a <video> knows it can seek next time.
                if ctype.startswith("video/"):
                    # Stream a clip in chunks rather than buffering a whole (tens-of-MB) video in RAM.
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "max-age=86400")
                    self.end_headers()
                    with open(serve, "rb") as fsrc:
                        shutil.copyfileobj(fsrc, self.wfile, _RANGE_CHUNK)
                    return
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


def _qs_int(q: dict, key: str, default: int, lo: int, hi: int) -> int:
    """One integer query parameter, clamped to [lo, hi]. Garbage falls back to `default` rather
    than raising -- a hand-typed ?limit=abc should not become a 500."""
    try:
        return max(lo, min(int((q.get(key) or [default])[0]), hi))
    except (TypeError, ValueError):
        return default


# ---- review-queue modes (the queue is the only source of templates) --------------------
# `recent` is what the dashboard has always shown and stays the default; the other three are
# opt-in filters over the WHOLE species pool, because a recent-only window reaches a few dozen
# visits out of hundreds and human confirmations are the only thing that grows the template set.
# All four are computed from stored columns + the appearance prototypes, so nothing here is
# calibrated to a camera position and a camera move can't silently stale them out.
QUEUE_MODES = ("recent", "unreviewed_auto", "ambiguous", "stale")


def _days_since(ts, now=None):
    """Whole-ish days between an ISO timestamp and now (float, one decimal). None if unparseable."""
    t = db.parse_local(ts) if ts else None
    if t is None:
        return None
    from datetime import datetime as _dt
    ref = now or _dt.now().astimezone()
    return round((ref - t).total_seconds() / 86400.0, 1)


def _template_freshness(matcher, now=None) -> dict:
    """{name: {n_templates, newest_template, days_since_template}} over each individual's
    CONFIRMED SOLO visits -- exactly the set `templates()` hands the matcher.

    This is the priority list, not a decoration: measured on this corpus, leave-one-visit-out
    top-1 falls 0.818 -> 0.482 -> 0.222 as the newest usable template ages 0 -> 7 -> 21 days.
    An individual whose freshest template is a month old effectively cannot be recognised."""
    out: dict = {}
    for name, vid, _proto in matcher.templates():
        started = matcher.visit_started.get(vid)
        e = out.setdefault(name, {"n_templates": 0, "newest_template": None})
        e["n_templates"] += 1
        if started and (e["newest_template"] is None or started > e["newest_template"]):
            e["newest_template"] = started
    for e in out.values():
        e["days_since_template"] = _days_since(e["newest_template"], now)
    return out


def _appearance_rank(matcher, vid):
    """(name, similarity, lead) for one visit from the APPEARANCE templates only -- the same two
    numbers auto_assign gates on, so a mode built on them shows exactly what the auto tier sees.

    Two things it deliberately does NOT do. It takes no behaviour or adjacency input: the queue's
    filters must not become a second, sloppier ranker (two-axis principle). And it ranks through
    `templates_for`, the SOURCE-GUARDED pool -- a probe is only compared with templates from its
    own camera, so a visit from an untemplated camera ranks against nothing and drops out of the
    filters rather than being sorted by noise."""
    import individuals
    if vid not in matcher.protos:
        return None
    temps = (matcher.templates_for(vid) if hasattr(matcher, "templates_for")
             else matcher.templates())
    if not temps:
        return None
    ranked = individuals.rank_templates(matcher.protos[vid], temps)
    if not ranked:
        return None
    name, sim, _via = ranked[0]
    return name, sim, sim - (ranked[1][1] if len(ranked) > 1 else 0.0)


def _queue_filter(cfg, matcher, rows, mode, freshness):
    """Keep the visits `mode` addresses, newest first (the caller's row order is preserved).

      recent           -- everything; today's behaviour, unchanged.
      unreviewed_auto  -- machine-named and neither kept nor rejected by a human. These wear a
                          name nobody confirmed; each one is a single click from being a template
                          or a tombstone.
      ambiguous        -- the appearance match clears the similarity bar but its lead over the
                          runner-up INDIVIDUAL is under the margin. This is precisely what
                          auto_assign refuses (its `ambiguous` skip bucket), which makes it the
                          highest-information click on offer: the machine cannot resolve it and
                          says so; the eye usually can.
      stale            -- unreviewed visits whose top candidate is an individual whose newest
                          human template has aged past reid_queue_stale_days. Confirming one
                          refreshes the template the next week of matching stands on.
    """
    if mode not in QUEUE_MODES:
        mode = "recent"
    if mode == "recent":
        return mode, list(rows)
    # 'ambiguous' mirrors the auto tier's OWN two bars when the tier is live, so the mode shows
    # exactly the visits it refuses on the margin (its `ambiguous` skip bucket). While the tier is
    # disabled -- reid_auto_threshold 0.0, the shipped default, and the state this mode is most
    # useful in -- both bars go inert, so fall back to the novelty cut and the queue's own margin.
    auto_on = (cfg.reid_auto_threshold or 0) > 0
    clears = max(cfg.reid_auto_threshold or 0.0, cfg.reid_novel_threshold)
    margin = (cfg.reid_auto_margin if auto_on and (cfg.reid_auto_margin or 0) > 0
              else cfg.reid_queue_ambiguous_margin)
    out = []
    for v in rows:
        vid = v["id"]
        if mode == "unreviewed_auto":
            if vid in matcher.auto and vid not in matcher.confirmed and vid not in matcher.rejected:
                out.append(v)
            continue
        # ambiguous / stale both look at an UNREVIEWED visit's appearance ranking.
        if vid in matcher.confirmed or vid in matcher.rejected or matcher.is_multi(vid):
            continue
        r = _appearance_rank(matcher, vid)
        if r is None:
            continue
        name, sim, lead = r
        if mode == "ambiguous":
            if sim >= clears and lead < margin:
                out.append(v)
        elif mode == "stale":
            days = (freshness.get(name) or {}).get("days_since_template")
            if sim >= cfg.reid_novel_threshold and days is not None \
                    and days >= cfg.reid_queue_stale_days:
                out.append(v)
    return mode, out


def _adjacency_context(rows_by_source, matcher, v):
    """DISPLAY-ONLY: the nearest same-source visit within an hour that already carries a human
    name ("started 6 min after the visit you named Stan").

    Measured on this corpus, adjacent same-source visits inside 60 min are the same individual
    67-77% of the time against a 28% base rate -- genuinely worth showing a human. It is NOT
    worth scoring: as a ranking input under session blocking it fired 3 times and was wrong 3
    times, and adjacency + a same-night template are the same confound. So this rides in its own
    payload field, is never consulted by _appearance_rank, and never touches the sort order."""
    lst = rows_by_source.get(v["source"]) or []
    try:
        i = lst.index(v["id"])
    except ValueError:
        return None
    best = None
    # `lst` is newest-first, so i+1 is the OLDER neighbour -- this visit started AFTER it.
    for j, direction in ((i + 1, "after"), (i - 1, "before")):
        if j < 0 or j >= len(lst):
            continue
        other = lst[j]
        name = matcher.confirmed.get(other)
        if not name:
            continue
        a, b = matcher.visit_started.get(v["id"]), matcher.visit_started.get(other)
        ta, tb = db.parse_local(a) if a else None, db.parse_local(b) if b else None
        if ta is None or tb is None:
            continue
        gap = abs((ta - tb).total_seconds())
        if gap <= _ADJACENT_CONTEXT_S and (best is None or gap < best["gap_s"]):
            best = {"name": name, "visit_id": other, "gap_s": int(gap), "direction": direction}
    return best


_ADJACENT_CONTEXT_S = 3600.0   # the measured window; display only, never a ranking input


def _reid_funnel(matcher, pool, source_of, template_sources) -> dict:
    """Where this species' visits go, counted live: total -> has a prototype -> human-confirmed
    -> usable (solo) template, plus what the auto tier is looking at.

    Published on the panel because "automation contributes nothing" is useless as a feeling and
    actionable as a number. `addressable` is the auto tier's own arithmetic -- visits with a
    prototype that are not confirmed, not multi-animal and not tombstoned -- so the gap between
    it and `auto_named` is exactly the automation's shortfall. The per-source split matters for
    the same reason: a source with no templates can never be named from templates, and folding
    it into one denominator quietly understates every coverage figure."""
    protos = set(matcher.protos)
    confirmed = set(matcher.confirmed)
    multi = {v for v in protos if v not in confirmed and matcher.is_multi(v)}
    addressable = {v for v in protos
                   if v not in confirmed and v not in matcher.rejected and v not in multi}
    by_source: dict = {}
    for v in pool:
        s = by_source.setdefault(v["source"], {"source": v["source"], "visits": 0, "confirmed": 0,
                                               "auto": 0, "addressable": 0, "templated": False})
        s["visits"] += 1
        vid = v["id"]
        s["confirmed"] += int(vid in confirmed)
        s["auto"] += int(vid in matcher.auto)
        s["addressable"] += int(vid in addressable)
    for s in by_source.values():
        s["templated"] = s["source"] in template_sources
    return {
        "visits": len(pool),
        "with_prototype": len(protos),
        "confirmed": len(confirmed),
        "templates": len(matcher.templates()),
        "multi_animal": len(multi),
        "addressable": len(addressable),
        "auto_named": len(matcher.auto),
        "rejected": len(matcher.rejected),
        "by_source": sorted(by_source.values(), key=lambda s: -s["visits"]),
    }


def _reid_queue(cfg, species: str = "raccoon", limit: int = 30, offset: int = 0,
                mode: str = "recent") -> dict:
    """The Individuals tab's review queue: visits of `species` with a WHO suggestion each
    (nearest confirmed visit / novelty flag / 2+-animals badge), the confirmed cast, and -- while
    nothing is confirmed yet -- the cold-start visit-groups to name first. Read-only; the heavy
    lifting (prototype matching over the embedding matrix) lives in individuals.VisitMatcher and
    is rebuilt per call, which is fine for a tab that loads on demand.

    `mode` picks WHICH visits (see QUEUE_MODES); `offset`/`limit` PAGINATE within that mode.
    Pagination rather than a bigger limit on purpose: the per-visit work below (clip overlap +
    crop strip) and the matcher rebuild are what cost, so a page stays cheap however deep the
    filter reaches."""
    import individuals

    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {"species": species, "queue": [], "cast": [], "bootstrap": [], "mode": "recent",
                "offset": 0, "limit": limit, "n_matched": 0, "funnel": {}}
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
        all_clips = stats.load_clips(conn)   # once; overlap-match each visit's footage in memory
        # The WHOLE species pool, newest first. The mode filters it and offset/limit page it --
        # the expensive per-visit work below runs on the page only.
        pool = conn.execute(
            """SELECT id, source, started_at, ended_at, detection_count, representative_detection_id
               FROM visits WHERE species = ? ORDER BY started_at DESC""", (species,)).fetchall()
        freshness = _template_freshness(matcher)
        mode, matched = _queue_filter(cfg, matcher, pool, mode, freshness)
        rows = matched[offset:offset + limit]
        # Per-source visit order (newest first, same as `pool`), for the display-only temporal
        # context chip. Built from the whole pool so a page boundary can't hide a neighbour.
        by_source: dict = {}
        source_of: dict = {}
        for v in pool:
            by_source.setdefault(v["source"], []).append(v["id"])
            source_of[v["id"]] = v["source"]
        # Which sources the appearance templates actually come from. A visit from any OTHER source
        # cannot be matched -- not "is probably a new animal", CANNOT BE MATCHED -- and the card
        # says so instead of showing a meaningless top-1. Derived from the data, never a source
        # name in code, so it stays true when a camera is added, moved or retired.
        template_sources = {source_of.get(tvid) for _n, tvid, _p in matcher.templates()}
        template_sources.discard(None)
        for v in rows:
            s = matcher.suggest(v["id"])
            # Evidence for the human doing the naming: the clips that rolled during this visit
            # (busiest first, click to play) + a strip of its sharpest crops for extra angles.
            vclips = [stats._clip_out(c) for c in stats.clips_overlapping(
                all_clips, v["source"],
                db.parse_local(v["started_at"]), db.parse_local(v["ended_at"]))]
            rep = _rep_crop(v["representative_detection_id"])
            vcrops = [r["crop_path"] for r in conn.execute(
                "SELECT crop_path FROM detections WHERE source = ? AND species = ? "
                "AND timestamp >= ? AND timestamp <= ? AND crop_path IS NOT NULL "
                "ORDER BY crop_quality DESC LIMIT 7",
                (v["source"], species, v["started_at"], v["ended_at"])).fetchall()]
            vcrops = [c for c in vcrops if c != rep][:6]   # "other" crops -> drop the hero thumb
            # A visit from a source no template comes from is structurally unmatchable. Measured:
            # every trail-cam raccoon prototype scores a median 0.249 / max 0.363 against every
            # glass-door template, and trail-cam-to-trail-cam similarity is flat (0.510 near in
            # time vs 0.514 far) -- there is no identity structure to threshold. Saying "possibly
            # someone new" there states something about the ANIMAL when the truth is about the
            # CAMERA, so the flag rides here and the card swaps the wording.
            cross_source = bool(template_sources) and v["source"] not in template_sources
            queue.append({
                "visit_id": v["id"], "started_at": v["started_at"], "source": v["source"],
                "dwell_s": _dwell(v["started_at"], v["ended_at"]),
                "n_crops": v["detection_count"],
                "rep_crop": rep,
                "clips": vclips, "crops": vcrops,
                "confirmed_as": s["confirmed_as"], "auto_as": s["auto_as"],
                "rejected": v["id"] in matcher.rejected,
                "candidates": s["candidates"],
                "clip_candidates": s["clip_candidates"],
                "novel": s["novel"], "multi": s["multi"],
                "cross_source": cross_source,
                "co_present_frames": s["co_present_frames"],
                "co_present_clips": s["co_present_clips"],
                # DISPLAY ONLY -- see _adjacency_context. Never read by any ranking path, and
                # only offered where there is still a call to make (an already-confirmed visit
                # doesn't need a hint, it needs to stay out of the way).
                "context": None if s["confirmed_as"] else _adjacency_context(by_source, matcher, v),
                "n_embedded": s["n_embedded"], "note": s["note"], "species": species,
            })

        # The confirmed cast, with how much template material backs each name -- plus how many
        # visits the nightly pass has auto-named to it (pending the human's glance).
        cast: dict = {}
        for vid, name in matcher.confirmed.items():
            c = cast.setdefault(name, {"name": name, "n_visits": 0, "n_auto": 0, "last_seen": None})
            c["n_visits"] += 1
            started = matcher.visit_started.get(vid)
            if started and (c["last_seen"] is None or started > c["last_seen"]):
                c["last_seen"] = started
        for vid, name in matcher.auto.items():
            if name in cast:               # auto only ever assigns confirmed names, but be safe
                cast[name]["n_auto"] += 1
        # TEMPLATE FRESHNESS: how old this individual's newest USABLE (confirmed solo) template is.
        # n_visits counts confirmations; a confirmation on a 2+-animal visit is NOT a template, so
        # the two numbers differ and the difference is the point -- an individual can look busy and
        # still be unrecognisable. See _template_freshness for the decay curve this reads against.
        # THE ROSTER: what the human has said about who still lives here. A departed individual is
        # shown, ranked and suggested exactly as before -- the flag only tells the reader why the
        # stale-template warning next to it is not a to-do, and gives the auto tier the one fact it
        # can never infer (individuals.VisitMatcher.is_departed).
        statuses = db.individual_statuses(conn)
        by_key = {str(k).strip().casefold(): v for k, v in statuses.items()}
        for name, c in cast.items():
            f = freshness.get(name) or {}
            c["n_templates"] = f.get("n_templates", 0)
            c["newest_template"] = f.get("newest_template")
            c["days_since_template"] = f.get("days_since_template")
            st = by_key.get(str(name).strip().casefold()) or {}
            c["status"] = st.get("status") or "resident"
            c["departed_on"] = st.get("effective_date")
            c["status_note"] = st.get("note")

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
            def _rep_for_visit(vid):
                # The visit may have been renumbered/removed by a rebuild since refit() listed it,
                # so guard the row (fetchone() can be None) rather than index None[0] -> 500.
                row = conn.execute(
                    "SELECT representative_detection_id FROM visits WHERE id=?", (vid,)).fetchone()
                return _rep_crop(row[0] if row else None)
            fits = {name: {"visits": [{**x, "rep_crop": _rep_for_visit(x["visit_id"])} for x in lst]}
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
                "novel_threshold": cfg.reid_novel_threshold,
                "mode": mode, "modes": list(QUEUE_MODES),
                "offset": offset, "limit": limit, "n_matched": len(matched),
                "stale_days": cfg.reid_queue_stale_days,
                "funnel": _reid_funnel(matcher, pool, source_of, template_sources)}
    finally:
        conn.close()


def _reid_unblend(cfg, visit_id: str) -> dict:
    """Separate a multi-animal visit into its individuals: clip-tracklet clustering first (the
    full-frame-rate basis, validated on the Notch/Elliot pair), falling back to STILL-tracklet
    clustering when the clips can't separate (photo-only trail-cam cycles, pruned footage, or
    simply fewer than two embedded tracklets). Both bases return the same group shape; stills
    groups carry `detection_ids` (labels stamp the crops directly) where clip groups carry
    `track_ids`, and the payload's `basis` says which ran."""
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
        # The RICHER set (solo-attributed + explicit) is used only to break the tie between two
        # human-logged co-present names (unblend_visit restricts it to that pair, so the coarseness
        # that bars it from open suggestions is harmless here). Build it from a matcher's solo map;
        # best-effort, since it's only a disambiguation aid on top of the quick-pick.
        elim = templates
        matcher = None
        try:
            matcher = individuals.VisitMatcher(conn, cfg.reid_species, cfg=cfg)
            elim = matcher.clip_templates
        except Exception:
            pass
        out = individuals.unblend_visit(conn, vid, templates=templates,
                                        elim_templates=elim, cfg=cfg)
        out["basis"] = "clips"
        if len(out.get("groups") or []) >= 2:
            return out
        # The clips couldn't split this visit -- try the stills basis. Its suggestions rank
        # against the confirmed-visit STILL templates (human-only, like everything
        # template-shaped in this system).
        stills = individuals.unblend_visit_stills(
            conn, vid, templates=(matcher.templates() if matcher else []), cfg=cfg)
        return stills if len(stills.get("groups") or []) >= 2 else out
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
        clips = individuals.clips_for_individual(conn, individual, species=cfg.reid_species, cfg=cfg)
        return {"individual": individual, "n_clips": len(clips), "clips": clips}
    finally:
        conn.close()


def _live_now(cfg, source: str | None = None) -> dict:
    """Powers the Live tab's "who's here now?" control for one camera `source`: the current visit
    span ON THAT SOURCE, the named cast to pick from (human-confirmed individuals, so you tag a
    known animal in one tap), and a short log of recent live sightings so a just-logged note
    visibly lands. The cast/recent log is shared across cameras (you name the same animals
    everywhere); only the visit span is per-source."""
    source = source or cfg.source
    visit = stats.current_live_visit(cfg, source)
    cast, recent = [], []
    conn = db.connect_readonly(cfg.db_path)
    if conn is not None:
        try:
            cast = [r[0] for r in conn.execute(
                "SELECT DISTINCT individual_id FROM detections "
                "WHERE individual_id IS NOT NULL AND individual_source = 'human' "
                "ORDER BY individual_id COLLATE NOCASE")]
            recent = db.recent_live_sightings(conn, limit=8)
        finally:
            conn.close()
    return {"visit": visit, "cast": cast, "recent": recent, "source": source}


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


def _prewarm(cfg):
    """Warm the Dispatch's digest cache and kick the current period's highlight-reel build a
    little after startup, so the first person to open the page isn't the one who pays for it.
    Delayed so the rig finishes loading its models first; failures are silently irrelevant."""
    time.sleep(45)
    try:
        stats.period_digest(cfg)
        reel.reel_status(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[web] prewarm skipped: {e}")


def start(cfg, frame_buffers: dict, control_bridges: dict, zone_store=None):
    server = make_server(cfg, frame_buffers, control_bridges, zone_store)
    threading.Thread(target=server.serve_forever, name="webdash", daemon=True).start()
    threading.Thread(target=_prewarm, args=(cfg,), name="web-prewarm", daemon=True).start()
    return server


def shutdown(server) -> None:
    try:
        server.stop_event.set()
        server.shutdown()
        server.server_close()
    except Exception:
        pass
