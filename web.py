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
restart). Bound to localhost by default, on port 80 so the address can be a bare name (see
mdns.py) -- falling back to cfg.web_port_fallback when 80 is taken or forbidden, which is the
normal case off Windows. No torch/cv2 here -- only stdlib + db/stats.
"""
from __future__ import annotations

import gzip
import hmac
import ipaddress
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
import zipfile
from dataclasses import replace
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import behavior
import config
import db
import mdns
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


def _is_allowed_host(raw: str, web_host: str = "", mdns_host: str = "") -> bool:
    """DNS-rebinding guard. The peer-IP check (_is_lan_client) can't stop a malicious site that
    resolves its OWN name to this rig's LAN IP: the request then comes from the victim's own (local)
    browser, so the peer IP looks fine, but the Host header still carries the ATTACKER's hostname.
    Accept only a Host that is 'localhost', a loopback/private/link-local IP literal, the
    operator's configured web_host, or the mDNS name this rig publishes for itself (mdns.py) --
    which is the name every LAN visitor is now told to type, so refusing it would 403 the whole
    point. A real browser/curl always sends Host, so a rebinding fetch can't omit it; an absent
    Host (rare, HTTP/1.0 tooling) is allowed through.

    `mdns_host` is matched EXACTLY rather than blanket-allowing `*.local`. Blanket-allowing is
    tempting and nearly safe -- RFC 6762 reserves `.local` for multicast, so an internet resolver
    cannot point one at this rig -- but "nearly" leans on every client routing `.local` to mDNS,
    and a network whose unicast DNS answers for `.local` (an old AD domain, say) would hand the
    guard straight back to an attacker. One known name costs nothing and assumes nothing."""
    if not raw:
        return True
    host = _host_name(raw).lower()
    if host == "localhost" or host == str(web_host or "").lower():
        return True
    if mdns_host and host == str(mdns_host).lower():
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


def _is_same_origin(origin: str, host_header: str, web_host: str = "", web_port=0,
                    mdns_host: str = "") -> bool:
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
    if _is_allowed_host(host_header, web_host, mdns_host):
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


def _csrf_refusal(method: str, headers, web_host: str = "", web_port=0, mdns_host: str = ""):
    """The 403 body for a state-changing request that didn't come from this dashboard, or None if it
    may proceed. GET/HEAD change nothing and pass untouched; every POST -- present and future --
    must carry an Origin naming us and a JSON Content-Type. A MISSING Origin is refused too: fetch
    and XHR always send one, so only hand-rolled tooling lacks it, and a curl user just adds
    -H 'Origin: http://127.0.0.1' -H 'Content-Type: application/json'.
    Without this, any page the operator happens to be browsing could fetch
    /api/individual with {"from": "Notch", "to": ""} and blank months of hand-confirmed re-ID
    labels -- no preflight, no confirm, no undo. `headers` is anything with a .get()."""
    if method != "POST":
        return None
    if not _is_same_origin(headers.get("Origin"), headers.get("Host"), web_host, web_port,
                           mdns_host):
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

# The making-of site's static types (it is HTML + CSS + JS + baked JSON + the media above).
_MAKINGOF_TYPES = {
    **_MEDIA_TYPES,
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8", ".json": "application/json",
    ".svg": "image/svg+xml", ".ico": "image/x-icon", ".md": "text/plain; charset=utf-8",
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
            # -c:a aac, not -an: the live rig's clips carry no audio track (OpenCV frame pipe),
            # so for them this changes nothing -- but a trail-cam MP4 arrives with its microphone
            # track intact (import copies the file whole), and growls/kit-chitter are ID evidence
            # the owner actually uses. Re-encode rather than copy: trail cams like ADPCM/PCM
            # audio, which browsers refuse; AAC always plays. No audio in -> no audio out.
            subprocess.run(
                [_FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 "-c:a", "aac", "-b:a", "96k", str(tmp)],
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


def _archive_candidates(zdir: Path, zip_name: str):
    """Every archive that could hold this day's clip, base first then parts.

    backup.py writes a day as <stem>.zip and adds <stem>.part2.zip, .part3.zip, ... for whatever
    a later trail-cam import backfilled into that date, rather than rewriting the sealed archive.
    So the member is in exactly one of them and which one is not predictable from the path."""
    base = zdir / zip_name
    found = [base] if base.is_file() else []
    # Literal prefix, not a glob: the name carries a camera label off the disk, and a `[` in one
    # would make a glob pattern silently match nothing (backup.archive_parts says the same).
    prefix = zip_name[:-len(".zip")] + ".part"
    try:
        found += sorted(q for q in zdir.iterdir()
                        if q.name.startswith(prefix) and q.name.endswith(".zip"))
    except OSError:
        pass
    return found


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
    candidates = _archive_candidates(Path(backup_dest) / "clips", zip_name)
    if not candidates:
        return None
    with _lock_for("archive:" + member):            # concurrent range requests restore once
        if out.is_file() and out.stat().st_size > 0:
            return out
        for zpath in candidates:
            try:
                with zipfile.ZipFile(zpath) as zf, zf.open(member) as src:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".tmp.mp4")
                    with open(tmp, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    tmp.replace(out)                # atomic publish, like the transcode cache
                break
            except (KeyError, OSError, zipfile.BadZipFile):
                continue        # not in this part, or this part is unreadable -- try the next
        else:
            return None         # no part held it
    _prune_archive_cache(cache_root)
    return out


# ---------------------------------------------------------------------------
# Pack this rig from the dashboard ("move this rig", footer). Thin UI over
# `migrate.py pack`: the button spawns it as a SUBPROCESS (same pattern as the
# naming helper -- its heavyio waits and hour-long zip runs must not live inside
# a request thread, and a dashboard restart must not kill a half-written
# bundle), and status is read back by tailing the process's combined output.
# Starting a pack is LOOPBACK-ONLY, the camera-password rule for the same
# reason: this endpoint writes gigabytes to a filesystem path of the caller's
# choosing, over plain HTTP with no login beyond the operator token -- so to
# aim the rig's disk somewhere, be at the rig. Restore has no UI on purpose:
# it runs on the NEW machine, where no rig and therefore no dashboard exists
# yet -- it stays the one CLI command the restore checklist already teaches.
# ---------------------------------------------------------------------------

_pack_guard = threading.Lock()
_pack_job: dict | None = None    # the one pack at a time, kept after exit so /status can report
                                 # the outcome. A dashboard restart forgets it (state says idle)
                                 # while the pack itself runs on: the log file stays the record.


def _pack_start(dest_raw, no_weights: bool, *, root: Path | None = None) -> tuple[dict, int]:
    """Validate `dest_raw` and spawn `migrate.py pack` toward it. Returns (json body, status).
    Validation mirrors migrate.py's own refusals so the UI hears them as form errors instead of
    as a subprocess that exits 1 -- but the subprocess still re-checks everything, because this
    layer is a convenience, never the authority."""
    global _pack_job
    root = root or config.ROOT
    dest_raw = str(dest_raw or "").strip()
    if not dest_raw:
        return {"error": "give a destination folder -- a USB drive, a network share, or a "
                         "cloud-synced folder"}, 400
    dest = Path(dest_raw).expanduser()
    if not dest.is_absolute():
        return {"error": "give an absolute path -- a relative one would land wherever the "
                         "server happened to be started from"}, 400
    if root in [dest, *dest.parents]:
        return {"error": "that folder is inside the project -- pack the rig somewhere outside "
                         "itself"}, 400
    with _pack_guard:
        if _pack_job is not None and _pack_job["proc"].poll() is None:
            return {"error": "a pack is already running", "dest": _pack_job["dest"]}, 409
        log_path = root / "logs" / "pack-from-dashboard.log"
        log_path.parent.mkdir(exist_ok=True)
        cmd = [sys.executable, str(root / "migrate.py"), "pack", str(dest)]
        if no_weights:
            cmd.append("--no-weights")
        try:
            # "w", not append: each pack's log is exactly that pack, so the status tail can
            # never show a previous run's "finished" under a new run's spinner.
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.Popen(cmd, cwd=str(root), stdin=subprocess.DEVNULL,
                                        stdout=logf, stderr=subprocess.STDOUT)
        except OSError as e:
            return {"error": f"could not start migrate.py: {e}"}, 500
        _pack_job = {"proc": proc, "dest": str(dest), "log": log_path,
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return {"ok": True, "dest": str(dest)}, 200


def _pack_status(operator: bool, loopback: bool) -> dict:
    """State of the pack job for the dashboard. `can_pack` is the up-front capability flag
    (mirrors /api/cameras' can_set_credentials, so the form can say "do this at the rig" before
    a refused submit, not after). A viewer learns only that a pack is or isn't running -- the
    destination and the log say where this machine's disks are mounted, which is operator
    detail."""
    with _pack_guard:
        job = _pack_job
    out: dict = {"can_pack": bool(operator and loopback)}
    if job is None:
        out["state"] = "idle"
        return out
    rc = job["proc"].poll()
    out["state"] = "running" if rc is None else ("done" if rc == 0 else "failed")
    if not operator:
        return out
    out["dest"] = job["dest"]
    out["started"] = job["started"]
    if rc is not None:
        out["returncode"] = rc
    try:
        lines = job["log"].read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    out["log_tail"] = lines[-15:]
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
    # The individual's STORY (life_events ledger): dated free-text notes for the profile
    # timeline. Read-only here; POST /api/individual/event appends.
    conn = db.connect_readonly(cfg.db_path)
    if conn is not None:
        try:
            prof["events"] = db.life_events(conn, name)
            prof["lapse"] = _profile_lapse(conn, cfg, prof, name)
        finally:
            conn.close()
    return prof


def _profile_lapse(conn, cfg, prof, name) -> dict:
    """The LAPSE state for one individual's own page -- from individuals.lapse_by_name, which is
    the same definition the Individuals tab's matcher uses, computed WITHOUT loading a single
    embedding vector (a profile view must not pay for its species' whole matrix).

    One definition matters here more than it looks: the first cut of this used "human-confirmed
    solo visit" and skipped the requirement that the visit have enough embedded crops to BE a
    template. On real data that made this page call Notch 'fresh, confirmed 0.7 days ago' while
    the Individuals tab called the same animal 'lapsed, 45.6 days' -- both from a confirmation the
    nightly embed pass had not reached yet. Returns the `none` state on any failure, which is the
    loud direction."""
    import individuals
    try:
        species = (prof.get("species_mix") or [{}])[0].get("species")
        table = individuals.lapse_by_name(conn, species, cfg=cfg, names=[name])
        return (table.get(str(name).strip().casefold())
                or individuals.identity_lapse(None, 0, cfg=cfg))
    except (sqlite3.Error, ValueError, TypeError, KeyError, IndexError):
        return individuals.identity_lapse(None, 0, cfg=cfg)


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


def _cameras_admin(cfg, running_sources, started_at: str):
    """The stored camera list, plus whether the rig is running something different from it.

    `pending_restart` is the honest half of this feature. Unlike ignore zones, a camera change
    does NOT apply live -- the rig reads the list once at startup and gives each camera a capture
    thread -- so the dashboard has to say so, or an operator edits a URL, sees the old picture,
    and edits it again somewhere worse. It is true when the enabled set no longer matches what is
    actually running, or when any live row was written after this server started.

    Never returns a password: db.list_cameras does not select the column at all."""
    conn = db.connect(cfg.db_path)
    try:
        rows = db.list_cameras(conn)
    finally:
        conn.close()
    enabled = [r for r in rows if r["enabled"]]
    pending = sorted(r["source"] for r in enabled) != sorted(running_sources)
    if not pending:
        since = db.parse_local(started_at) if started_at else None
        for r in enabled:
            stamp = db.parse_local(r["updated_at"] or r["created_at"] or "")
            if since and stamp and stamp > since:
                pending = True
                break
    return rows, pending


def make_server(cfg, frame_buffers: dict, control_bridges: dict, zone_store=None,
                specs=None):
    """`frame_buffers` / `control_bridges` are dicts keyed by camera `source` -- one entry per
    live camera. A single-camera rig passes one-entry dicts; the Live tab then shows one pane.
    The dashboard discovers the cameras via /api/cameras and routes /stream.mjpg, /snapshot.jpg,
    /api/camera and /api/live/* with a ?source= query param (defaulting to the primary camera, so
    an old client with no source param still works).

    `zone_store` is the rig's shared IgnoreZoneStore (the capture threads read the same instance,
    so a zone edit takes effect on the next frame). None -- tests, or any caller without a rig --
    builds a private one from the DB; edits then persist but nothing live is watching them.

    `specs` is the CameraSpec list the rig is ACTUALLY RUNNING. It has to be passed in rather
    than re-read from cfg.camera_specs(), because since 2026-08-22 the camera list lives in the
    database and config only seeds it -- so a camera added from the dashboard exists in neither
    cfg.cameras nor anything derivable from it, and reading config here would give a pane the
    wrong name, the wrong network flag, or no pane at all. None falls back to config for
    serve-only callers and tests, which have no rig to ask."""
    if zone_store is None:
        zone_store = IgnoreZoneStore.load(cfg)
    if not frame_buffers:
        # No live camera at all (a serve-only caller, or a test passing bare dicts): synthesize
        # one dead pane for the primary source rather than dying on next(iter({})) below. The
        # Live tab then shows its normal "Camera feed unavailable" overlay, which is the truth.
        frame_buffers = {cfg.source: FrameBuffer()}
        control_bridges = dict(control_bridges or {})
        control_bridges.setdefault(cfg.source, CameraControlBridge())
    allowed_dirs = [d.resolve() for d in (cfg.crops_dir, cfg.frames_dir, cfg.clips_dir,
                                          getattr(cfg, "clip_crops_dir", cfg.clips_dir),
                                          # refcam's detector crops of the phone reference shots
                                          # (the individual profile's reference gallery). The raw
                                          # originals live OUTSIDE the project and stay unserved.
                                          getattr(cfg, "reference_crops_dir", cfg.crops_dir))]
    stop_event = threading.Event()

    # The name this rig publishes for itself, resolved once so every request checks a plain string.
    # Read from CONFIG rather than from a live registration: whether the announcement actually
    # succeeded is mdns.py's business, and a name nothing answers for is one no browser can put in
    # a Host header anyway -- so trusting config here costs nothing and keeps the guard a pure
    # function of settings.
    mdns_host = mdns.host_name(cfg) if mdns.enabled(cfg) else ""

    # The primary camera (the Live tab's default / "Plate I"): the one matching cfg.source if it's
    # among the live cameras, else the first one. Insertion order of frame_buffers = camera order.
    primary = cfg.source if cfg.source in frame_buffers else next(iter(frame_buffers))
    _specs = {s.source: s for s in (specs if specs is not None else cfg.camera_specs())}
    _cam_order = [primary] + [s for s in frame_buffers if s != primary]
    cameras_meta = [{"source": s, "primary": s == primary,
                     "name": (_specs[s].display_name if s in _specs else s),
                     "network": bool(_specs[s].is_network) if s in _specs else False}
                    for s in _cam_order]
    # Each live pane opens its own MJPEG stream, so scale the concurrent-stream cap with the camera
    # count (the flat _MAX_STREAMS=6 was sized for one camera + a few viewers).
    # When this process started serving. _cameras_admin compares stored rows against it to
    # decide whether a restart is pending -- a camera edit does not apply live, and the dashboard
    # has to say so rather than leaving someone staring at the old picture.
    server_started_at = db.now_local_iso()
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
                                    getattr(cfg, "web_host", ""), getattr(cfg, "web_port", 0),
                                    mdns_host)
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
            if not _is_allowed_host(self.headers.get("Host"), getattr(cfg, "web_host", ""),
                                    mdns_host):
                self._send(403, "text/plain",
                           b"forbidden: unrecognized Host header (DNS-rebinding guard)")
                return False
            return True

        def _is_operator(self) -> bool:
            """The operator/viewer split -- see _operator_decision for the whole rule."""
            return _operator_decision(getattr(cfg, "operator_token", None),
                                      self.client_address[0] if self.client_address else "",
                                      self.headers.get("X-Operator-Token"))

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
                    # `cameras` is exactly what it always was -- source/primary/name/network and
                    # never a URL -- because GETs are not gated and every LAN viewer reads this.
                    # The management half (`rows`) is added only for an operator, and even then
                    # carries has_password rather than any password.
                    body = {"cameras": cameras_meta, "primary": primary,
                            "manageable": self._is_operator()}
                    if body["manageable"]:
                        rows, pending = _cameras_admin(cfg, list(frame_buffers),
                                                       server_started_at)
                        body["rows"] = rows
                        body["pending_restart"] = pending
                        # Whether THIS client may set a password: credential writes are
                        # loopback-only, so the form can say "do this at the rig" up front
                        # instead of after a refused save.
                        body["can_set_credentials"] = self._is_loopback()
                    self._json(body)
                elif path == "/api/zones":
                    src = self._src()
                    self._json(_zones_payload(cfg, src, control_bridges[src].snapshot()))
                elif path == "/api/naming":
                    self._json(_naming_status())
                elif path == "/api/evalstatus":
                    self._json(_eval_status())
                elif path == "/api/role":
                    # Which tier THIS client is. The client uses it for comfort (hiding the
                    # curation chrome); the server refuses viewer writes regardless, in do_POST.
                    self._json({"operator": self._is_operator(),
                                "split": bool(getattr(cfg, "operator_token", None))})
                elif path == "/api/migrate/pack":
                    # The "move this rig" panel's poll: pack-job state, plus (operator only)
                    # the destination and the tail of migrate.py's own log.
                    self._json(_pack_status(self._is_operator(), self._is_loopback()))
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
                elif path == "/api/favorites":
                    # The gallery of kept things. Cached like every other read; every POST
                    # clears the whole API cache, so a star shows up here on the next open.
                    self._json(_cached(cfg, "favorites", lambda: stats.favorites_page(cfg)))
                elif path == "/api/behavior":
                    self._json(_cached(cfg, "behavior", lambda: behavior.overview(cfg), hold_s=30))
                elif path == "/api/seasons":
                    self._json(_cached(cfg, "seasons", lambda: stats.seasons_overview(cfg), hold_s=60))
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
                    since = _qs_int(q, "since_h", DEFAULT_QUEUE_WINDOW_H, 0, 24 * 400)
                    # The heaviest endpoint there is (~10s: a full VisitMatcher rebuild), so it
                    # leans hardest on the cache. Label POSTs clear it, so the queue is always
                    # fresh right after a confirm -- the rebuild then happens once, not per open.
                    self._json(_cached(cfg, f"reidq:{sp}:{lim}:{off}:{md}:{since}",
                                       lambda: _reid_queue(cfg, species=sp, limit=lim,
                                                           offset=off, mode=md,
                                                           since_h=since), hold_s=60))
                elif path == "/api/reid/dossier":
                    # One visit, everything about it (see _reid_dossier). Cached per visit id:
                    # it rebuilds the whole VisitMatcher, and the one-at-a-time flow steps
                    # BACKWARD as often as forward, so the second look must be free.
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    try:
                        vid = int((q.get("visit_id") or [""])[0])
                    except (TypeError, ValueError):
                        self._send(400, "text/plain", b"bad visit_id")
                    else:
                        self._json(_cached(cfg, f"dossier:{vid}",
                                           lambda: _reid_dossier(cfg, vid), hold_s=120))
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
                elif path == "/making-of" or path.startswith("/making-of/"):
                    # The explainer site, served from the rig itself: family on the LAN gets
                    # "what am I looking at?" without needing the public URL. Same folder the
                    # GitHub Pages deploy publishes; no build step, plain files.
                    self._makingof(path[len("/making-of"):].lstrip("/"))
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
            # THE VIEWER GATE (cfg.operator_token). A viewer's one allowed write is the "who's
            # here" log, which _live_sighting records as attributed, NO-STAMP testimony -- the
            # pair-path semantics, so a guest's enthusiasm can never contaminate a template.
            # Everything else mutating is refused HERE, at the same choke point as the CSRF
            # guard, so endpoints nobody has written yet are covered by construction.
            if not self._is_operator() and path != "/api/live/sighting":
                self._json({"error": "viewing only -- ask the operator for the token "
                                     "(dashboard footer) to edit", "viewer": True}, code=403)
                return
            try:
                if path not in ("/api/camera", "/api/zones", "/api/zones/delete",
                                "/api/cameras/save", "/api/cameras/delete",
                                "/api/migrate/pack"):
                    stats.clear_digest_cache()   # label/name edits must show on the next Dispatch
                    clear_api_cache()            # ...and on the next poll of every cached endpoint
                if path == "/api/camera":
                    control_bridges[self._src()].request(_clean_settings(data))
                    self._json({"ok": True})
                elif path == "/api/zones":
                    self._zone_add(data)
                elif path == "/api/zones/delete":
                    self._zone_delete(data)
                elif path == "/api/cameras/save":
                    self._camera_save(data)
                elif path == "/api/cameras/delete":
                    self._camera_delete(data)
                elif path == "/api/migrate/pack":
                    # Start `migrate.py pack` toward data["dest"]. Loopback-only, the camera-
                    # password rule (see the _pack_start section for the whole reasoning): this
                    # writes gigabytes to a path of the caller's choosing, so to aim the rig's
                    # disk somewhere, be at the rig.
                    if not self._is_loopback():
                        self._json({"error": "packing writes this rig's whole archive to a "
                                             "folder of your choosing, so it can only be "
                                             "started at the rig itself -- open this "
                                             "dashboard on the rig (http://127.0.0.1) "
                                             "and click it there"}, code=403)
                    else:
                        body, code = _pack_start(data.get("dest"),
                                                 bool(data.get("no_weights")))
                        self._json(body, code=code)
                elif path == "/api/individual/avatar":
                    # Pin (or clear, with crop=null) an individual's badge photo.
                    conn = db.connect(cfg.db_path)
                    try:
                        ok = db.set_individual_avatar(conn, str(data.get("name") or ""),
                                                      data.get("crop"))
                    finally:
                        conn.close()
                    self._json({"ok": True} if ok else
                               {"error": "that crop isn't one of this individual's"},
                               code=200 if ok else 400)
                elif path == "/api/individual/status":
                    self._individual_status(data)
                elif path == "/api/individual/event":
                    # Append one dated note to an individual's story: {"name","note","date"?}.
                    try:
                        conn = db.connect(cfg.db_path)
                        try:
                            row = db.add_life_event(
                                conn, data.get("name"), data.get("note"),
                                event_date=data.get("date") or None,
                                labeled_by=_labeler(data))
                        finally:
                            conn.close()
                        self._json({"ok": True, "event": row})
                    except ValueError as e:
                        self._json({"error": str(e)}, code=400)
                elif path == "/api/individual":
                    self._individual_action(data)
                elif path == "/api/reid/confirm":
                    self._reid_confirm(data)
                elif path == "/api/visit/label":
                    self._visit_label(data)
                elif path == "/api/favorite":
                    self._favorite(data)
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

        def _is_loopback(self) -> bool:
            """True when the request came from the rig box itself.

            This is the entire access control on a camera PASSWORD, and it is deliberate. This
            dashboard binds 0.0.0.0 on this rig, serves plain HTTP, and has no login at all
            unless cfg.operator_token is set -- so a password typed on a phone would cross the
            Wi-Fi in cleartext, to a server that would have accepted it from anyone on that
            Wi-Fi. Requiring loopback means "to give the rig a camera's password, be at the rig".
            It costs an operator one walk and removes the exposure rather than mitigating it.

            Everything else about a camera -- adding it, renaming, resolution, motion area,
            enabling, deleting -- stays editable from anywhere an operator can reach, because
            none of it is a secret."""
            return _is_loopback_client(
                self.client_address[0] if self.client_address else "")

        def _camera_save(self, data):
            """Create or edit one camera. With "id" it edits, without it creates.

            SOURCE IS WRITE-ONCE. It is the partition key of detections, visits, coverage_events,
            ignore_zones, view_epochs and the clips/<source>/ directory on disk, so an edit that
            changed it would orphan everything already recorded under the old name. An attempt to
            change it is refused with an explanation rather than silently ignored.

            A password in the body requires loopback (see _is_loopback). An ABSENT password on an
            edit leaves the stored one alone -- the form can never show the operator what is
            stored, so submitting it must not be able to wipe it. Clearing takes an explicit
            clear_password."""
            # What counts as "setting a password" must match db.update_camera EXACTLY, or the
            # gate and the writer disagree: db treats any non-None, non-empty value as a write,
            # so bool() here would wave a JSON `0` or `false` past the check and still store it.
            pw = data.get("password")
            wants_secret = (pw is not None and pw != "") or bool(data.get("clear_password"))
            fields = dict(
                kind=data.get("kind"), name=data.get("name"),
                device_index=data.get("device_index"), url_scheme=data.get("url_scheme"),
                url_host=data.get("url_host"), url_port=data.get("url_port"),
                url_path=data.get("url_path"), username=data.get("username"),
                frame_width=data.get("frame_width"), frame_height=data.get("frame_height"),
                motion_min_area=data.get("motion_min_area"),
                record_clips=data.get("record_clips"),
                enabled=bool(data.get("enabled", True)))
            conn = db.connect(cfg.db_path)
            try:
                # PRESENT, not merely truthy: an id of {} or [] or 0 is a client bug, and
                # falling through to the create branch would silently make a SECOND camera
                # instead of editing the one that was asked for.
                if data.get("id") is not None:
                    existing = db.get_camera(conn, data["id"])
                    if existing is None:
                        self._json({"error": "no such camera"}, code=404)
                        return
                    asked = str(data.get("source") or existing["source"])
                    if asked != existing["source"]:
                        self._json({"error": f"a camera's name is permanent once it has recorded "
                                             f"anything -- {existing['source']!r} is stamped on "
                                             f"its detections, visits and clip folder. Add a new "
                                             f"camera instead."}, code=400)
                        return
                    # MOVING A CREDENTIAL IS A CREDENTIAL OPERATION. The loopback rule below
                    # protects WRITING a password, but the stored one survives an edit that
                    # changes only the address -- so without this, any operator on the network
                    # could repoint a camera at a host they control, keep the password, and let
                    # the rig hand it over at the next restart. The URL is where the secret gets
                    # sent; changing it is changing the secret's destination.
                    if existing["has_password"] and not wants_secret:
                        def _same(a, b):
                            norm = lambda v: "" if v is None or v == "" else str(v)
                            return norm(a) == norm(b)
                        if any(not _same(data.get(k), existing.get(k)) for k in
                               ("kind", "url_scheme", "url_host", "url_port", "url_path",
                                "username")):
                            wants_secret = True
                    if wants_secret and not self._is_loopback():
                        self._json({"error":
                                    "This camera's login can only be changed from the rig "
                                    "itself, not over the network -- and that includes moving it "
                                    "to a different address, because its stored password would "
                                    "be sent to wherever it points. Open the dashboard on the "
                                    "rig machine to change this one."}, code=403)
                        return
                    if not fields["enabled"] and existing["enabled"]:
                        others = [r for r in db.list_cameras(conn)
                                  if r["enabled"] and r["id"] != existing["id"]]
                        if not others:
                            self._json({"error": "this is the only camera left -- turning it off "
                                                 "would leave the rig nothing to watch, and it "
                                                 "stops at the next restart."}, code=400)
                            return
                    row = db.update_camera(conn, existing["id"], password=pw,
                                           clear_password=bool(data.get("clear_password")),
                                           **fields)
                else:
                    if wants_secret and not self._is_loopback():
                        self._json({"error": "camera passwords can only be set from the rig "
                                             "itself, not over the network -- add the camera "
                                             "here without one, then set its password from the "
                                             "rig machine."}, code=403)
                        return
                    row = db.add_camera(conn, source=data.get("source"), password=pw, **fields)
            except (ValueError, TypeError) as e:
                # TypeError as well: a JSON id of {} or [] reaches int() and would otherwise be a
                # 500 with a traceback rather than a message about the field.
                self._json({"error": str(e)}, code=400)
                return
            finally:
                conn.close()
            # Deliberately not "ok, it's live": it is saved, and the rig picks it up on the next
            # start. Saying otherwise is how someone ends up re-typing a password that was right.
            self._json({"ok": True, "camera": row, "pending_restart": True})

        def _camera_delete(self, data):
            """Soft-delete one camera: {"id": n}. The row is tombstoned, not removed -- that is
            what stops config_local.py's seed putting it back on the next start, and what makes
            re-adding the same name reattach to the rows already recorded under it.

            Refuses the last enabled camera: a rig with none exits as soon as its final capture
            thread ends, so this would otherwise be a two-click way to make the box unstartable
            from a phone."""
            try:
                cam_id = int(data.get("id"))
            except (TypeError, ValueError):
                self._json({"error": "id (integer) is required"}, code=400)
                return
            conn = db.connect(cfg.db_path)
            try:
                enabled = [r for r in db.list_cameras(conn) if r["enabled"]]
                target = db.get_camera(conn, cam_id)
                if target is None:
                    self._json({"error": "no such camera"}, code=404)
                    return
                if target["enabled"] and len(enabled) <= 1:
                    self._json({"error": "this is the only camera left -- add another before "
                                         "removing it, or the rig has nothing to watch and will "
                                         "stop at the next restart."}, code=400)
                    return
                removed = db.remove_camera(conn, cam_id)
            finally:
                conn.close()
            self._json({"ok": True, "camera": removed, "pending_restart": True})

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
            kw["labeled_by"] = _labeler(data)
            conn = db.connect(cfg.db_path)
            try:
                self._json({"ok": True, **db.apply_visit_label(conn, **kw)})
            finally:
                conn.close()

        def _favorite(self, data):
            """Star / un-star one crop or one visit -- "keep this".

            Body: {"kind":"detection", "detection_id":n} or
                  {"kind":"visit", "source":.., "start":.., "end":..}, plus:
              "on":    false to un-star (default true).
              "note":  present (including "" or null) = set the caption exactly, the same
                       key-present idiom _visit_label uses for `name`.

            The visit form takes source+start+end -- _visitTarget's shape, the same one
            /api/visit/label takes -- because a visit id is not a durable handle here (the Visit
            Log re-clusters visits per request and visits.py renumbers the ledger). Only `start`
            is the key; `end` is recorded for display. See the `favorites` table in db.py.

            This is the one write in the dashboard that says nothing about WHAT the animal is,
            so it never touches a label, a species or an identity -- and nothing downstream
            reads it."""
            kind = str(data.get("kind") or "").strip().lower()
            kw = {}
            if kind == "detection":
                kw["detection_id"] = data.get("detection_id")
            elif kind == "visit":
                kw["source"] = data.get("source")
                kw["started_at"] = data.get("started_at") or data.get("start")
            # Any other kind falls through with an empty key and is refused by db._fav_key below,
            # so the validation lives in ONE place rather than being restated here.
            conn = db.connect(cfg.db_path)
            try:
                if data.get("on") is False:
                    self._json({"ok": True, "favorite": False,
                                "removed": db.remove_favorite(conn, kind, **kw)})
                    return
                row = db.add_favorite(conn, kind, note=data.get("note"),
                                      ended_at=(data.get("ended_at") or data.get("end")),
                                      labeled_by=_labeler(data), **kw)
                if "note" in data:      # present (incl. ""/null) = set the caption exactly
                    row = db.set_favorite_note(conn, kind, data.get("note"), **kw) or row
                self._json({"ok": True, "favorite": True, "row": row})
            except ValueError as e:
                self._json({"error": str(e)}, code=400)
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
            # Attribution + the viewer tier. `logged_by` is whoever this browser says is typing
            # (optional, self-reported, length-capped); a VIEWER's log additionally records with
            # stamp=False -- testimony in live_sightings, nothing written onto crops.
            operator = self._is_operator()
            logged_by = _labeler(data)
            conn = db.connect(cfg.db_path)
            try:
                res = db.record_live_sighting(
                    conn, source=source, names=names,
                    span_start=visit.get("start"), span_end=visit.get("end"),
                    note=data.get("note"), stamp=operator, labeled_by=logged_by)
                if res.get("error"):
                    self._json({"error": res["error"]}, code=400)
                    return
                self._json({"ok": True, "visit": visit, "as_viewer": not operator, **res})
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
                n = db.label_visit(conn, vid, name, reject=reject,
                                   labeled_by=_labeler(data))
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

        def _makingof(self, rel):
            """Serve the making-of explainer from making-of/ -- same containment discipline as
            _media (resolve, then prove the result is inside the one allowed root). Bare
            /making-of serves the index; a directory path serves its index.html."""
            root = (config.ROOT / "making-of").resolve()
            if not root.is_dir():
                self._send(404, "text/plain", b"the making-of site isn't in this checkout")
                return
            target = (root / urllib.parse.unquote(rel)).resolve() if rel else root
            if not _is_within(target, root):
                self._send(404, "text/plain", b"not found")
                return
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                self._send(404, "text/plain", b"not found")
                return
            ctype = _MAKINGOF_TYPES.get(target.suffix.lower(), "application/octet-stream")
            self._serve_file(target, ctype)

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

    server = _bind(cfg, Handler)
    # Rebind the closure's own `cfg` to the port actually taken. The Handler reads `cfg` from this
    # scope at REQUEST time, so this reaches it -- and it has to, because _is_same_origin compares
    # the Origin's port against cfg.web_port: a rig that asked for 80, fell back to 8000 and kept
    # believing it was on 80 would refuse every POST the dashboard made, from a page it served
    # itself. (It also makes web_port=0 honest for tests, which now get the ephemeral port back
    # instead of a literal zero.)
    bound = server.server_address[1]
    if bound != cfg.web_port:
        cfg = replace(cfg, web_port=bound)
    server.daemon_threads = True
    server.stop_event = stop_event
    return server


def _bind(cfg, handler):
    """Listen on cfg.web_port, falling back to cfg.web_port_fallback if that port is taken or
    forbidden. Returns the bound server; raises the ORIGINAL OSError if nothing worked.

    The fallback exists because the default port is now 80, and 80 is a port a bind can lose for
    ordinary reasons: it needs root on Linux/macOS, and on any OS something web-shaped may already
    hold it. Losing the whole dashboard over that would be a bad trade for a nicer address, and an
    unattended rig cannot be asked to notice and pass --port. The failure is printed rather than
    swallowed -- the address moved, and every printed URL downstream reads the real port from the
    socket, so the operator is told rather than left to discover it."""
    ports = mdns.local_candidates(cfg)
    first_error = None
    for port in ports:
        try:
            return ThreadingHTTPServer((cfg.web_host, port), handler)
        except OSError as e:
            first_error = first_error or e
            if port != ports[-1]:
                print(f"  [web] port {port} is not available ({e.strerror or e}); "
                      f"falling back to {ports[-1]}.")
    raise first_error


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


def _labeler(data) -> str | None:
    """Who is typing, self-reported by the browser (`logged_by`), length-capped. Free text and
    unverified on purpose -- this is a household name tag, not an account: it answers "whose
    verdict was that?" when several people label, and it is what a per-labeller agreement rate
    would key on later. NULL means the operator, before attribution existed. Attribution cannot
    be back-filled honestly (db.py refuses to invent provenance), so every write path that takes
    a human verdict passes through here."""
    return str((data or {}).get("logged_by") or "").strip()[:40] or None


def _is_loopback_client(peer_host) -> bool:
    """Is this peer the rig box itself? Pure, so the refusal path is testable without arranging a
    non-loopback socket (every test client is 127.0.0.1, which would always pass).

    Anything unparseable is NOT loopback. That direction matters: this gates writing a camera
    password, so an address we cannot understand must fail closed."""
    try:
        return ipaddress.ip_address(str(peer_host or "")).is_loopback
    except ValueError:
        return False


def _operator_decision(token_cfg, peer_host, sent_token) -> bool:
    """The operator/viewer rule, pure (unit-testable without a socket). No token configured =
    everyone is an operator -- the historical behaviour and the fresh-clone default. With a
    token: loopback is implicitly operator (you are at the rig), and any other client is
    operator only when its X-Operator-Token header matches -- entered once per browser in the
    dashboard footer, then attached to every request. Constant-time compare; a wrong token is
    simply a viewer, never an error."""
    if not token_cfg:
        return True
    try:
        if ipaddress.ip_address(peer_host or "").is_loopback:
            return True
    except ValueError:
        pass
    return hmac.compare_digest(str(sent_token or ""), str(token_cfg))


_EVAL_STATUS_CACHE = {"key": None, "value": None}


def _eval_status() -> dict:
    """The nightly regression gate's verdict, read off the newest reports/eval_*.json (written by
    run_clipmotion.bat's eval step). The dashboard shows this so a metric slide is a chip on the
    masthead the morning after, not a discovery weeks later. Cached on (path, mtime) -- artifacts
    only change when the nightly batch writes one. Degrades to {available: False} on a machine
    that has never run eval; that is a fact, not an error."""
    d = config.ROOT / "reports"
    try:
        files = sorted(d.glob("eval_*.json")) if d.exists() else []
    except OSError:
        files = []
    if not files:
        return {"available": False}
    p = files[-1]
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return {"available": False}
    if _EVAL_STATUS_CACHE["key"] == key:
        return _EVAL_STATUS_CACHE["value"]
    try:
        art = json.loads(p.read_text())
    except (OSError, ValueError):
        return {"available": False}
    diff = art.get("baseline_diff") or {}
    out = {
        "available": True,
        "artifact": p.name,
        "run_at": (art.get("meta") or {}).get("run_at"),
        # ok True/False when the run diffed a baseline; None = no gate ran (first artifact).
        "ok": diff.get("ok") if diff else None,
        "regressions": [r.get("metric") for r in (diff.get("regressions") or [])],
        "baseline_run_at": diff.get("baseline_run_at"),
    }
    _EVAL_STATUS_CACHE["key"], _EVAL_STATUS_CACHE["value"] = key, out
    return out


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


def _template_freshness(matcher, now=None, cfg=None) -> dict:
    """Freshness + the LAPSE state per individual. The computation lives in individuals.py, next
    to the matcher whose templates it describes, so the profile page, the suggestion payload and
    the roll call all read one definition instead of three that drift."""
    import individuals
    return individuals.template_freshness(matcher, now=now, cfg=cfg)


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


#: How far either side of a visit to look for the SAME MOMENT on another camera. Ten minutes,
#: because the trail cam's clock is set by hand each cycle and drifts a few minutes against the
#: rig's, and because a raccoon that leaves the door frame reaches the far camera within that.
CROSS_CAMERA_PAD_S = 600
#: Cap on how many crops one dossier ships. The whole point of the one-at-a-time flow is that the
#: eye gets EVERYTHING, but a 2,000-crop visit would hang the tab; sharpest-first means the cap
#: only ever removes the crops least worth looking at.
DOSSIER_MAX_CROPS = 60


def _reid_dossier(cfg, visit_id: int, species: str = "raccoon") -> dict:
    """EVERYTHING known about ONE visit -- the payload behind the one-at-a-time review flow.

    The queue card is deliberately small: it has to render 30 of them. This is the opposite --
    one visit, every crop, every clip, and the same moment as seen by any OTHER camera. That last
    part is the reason this endpoint exists rather than the card just growing:

      a trail-cam visit CANNOT be appearance-matched (measured: trail-cam prototypes score a
      median 0.249 against every glass-door template, and trail-cam-to-trail-cam similarity is
      flat -- there is no identity structure to threshold). So the only evidence that can ever
      name one is a human noticing that the glass door saw the same animal in the same minutes.
      109 of 521 glass-door raccoon visits have a trail-cam visit within CROSS_CAMERA_PAD_S, so
      that pairing is available for a fifth of the corpus and is currently shown nowhere.

    Read-only. Returns {} for an unknown visit."""
    import individuals

    conn = db.connect_readonly(cfg.db_path)
    if conn is None:
        return {}
    try:
        v = conn.execute(
            """SELECT id, source, species, started_at, ended_at, detection_count,
                      representative_detection_id, individual_id
               FROM visits WHERE id = ?""", (int(visit_id),)).fetchone()
        if v is None:
            return {}
        species = v["species"] or species
        matcher = individuals.VisitMatcher(conn, species, cfg)
        all_clips = stats.load_clips(conn)

        def _crop(det_id):
            if det_id is None:
                return None
            r = conn.execute("SELECT crop_path FROM detections WHERE id = ?",
                             (det_id,)).fetchone()
            return r["crop_path"].replace("\\", "/") if r and r["crop_path"] else None

        def _evidence(row):
            """Crops + clips for one visit, sharpest crop first."""
            crops = [{"path": r["crop_path"].replace("\\", "/"),
                      "at": r["timestamp"], "conf": r["confidence"]}
                     for r in conn.execute(
                         """SELECT crop_path, timestamp, confidence FROM detections
                            WHERE visit_id = ? AND crop_path IS NOT NULL
                            ORDER BY crop_quality DESC, confidence DESC LIMIT ?""",
                         (row["id"], DOSSIER_MAX_CROPS)).fetchall()]
            clips = [stats._clip_out(c) for c in stats.clips_overlapping(
                all_clips, row["source"], db.parse_local(row["started_at"]),
                db.parse_local(row["ended_at"]))]
            return crops, clips

        crops, clips = _evidence(v)
        s = matcher.suggest(v["id"])

        # THE SAME MOMENT ON ANOTHER CAMERA. Compared as instants via db.parse_local, never as
        # ISO strings: the two sources' rows can carry different UTC offsets (the trail cam's
        # timestamps are reconstructed at import) and a string compare would silently mis-order
        # them across a DST boundary.
        start, end = db.parse_local(v["started_at"]), db.parse_local(v["ended_at"])
        neighbours = []
        if start and end:
            pad = timedelta(seconds=CROSS_CAMERA_PAD_S)
            lo, hi = start - pad, end + pad
            day = (lo - timedelta(days=1)).strftime("%Y-%m-%d")
            for n in conn.execute(
                    """SELECT id, source, species, started_at, ended_at, detection_count,
                              representative_detection_id, individual_id
                       FROM visits WHERE id != ? AND source != ? AND started_at >= ?
                       ORDER BY started_at""",
                    (v["id"], v["source"], day)).fetchall():
                ns, ne = db.parse_local(n["started_at"]), db.parse_local(n["ended_at"])
                if not ns or not ne or ns > hi or ne < lo:
                    continue
                ncrops, nclips = _evidence(n)
                neighbours.append({
                    "visit_id": n["id"], "source": n["source"], "species": n["species"],
                    "started_at": n["started_at"], "ended_at": n["ended_at"],
                    "n_crops": n["detection_count"], "individual_id": n["individual_id"],
                    "rep_crop": _crop(n["representative_detection_id"]),
                    "crops": ncrops, "clips": nclips,
                    "offset_s": int((ns - start).total_seconds()),
                })

        # The answer buttons: who the human could plausibly say. Confirmed cast, busiest first,
        # each carrying how stale its template is so "who is this?" and "who needs a fresh
        # template?" are the same glance.
        freshness = _template_freshness(matcher)
        counts: dict = {}
        for vid, name in matcher.confirmed.items():
            counts[name] = counts.get(name, 0) + 1
        statuses = db.individual_statuses(conn)
        by_key = {str(k).strip().casefold(): st for k, st in statuses.items()}
        cast = []
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            f = freshness.get(name) or {}
            st = by_key.get(str(name).strip().casefold()) or {}
            cast.append({"name": name, "n_visits": n,
                         "n_templates": f.get("n_templates", 0),
                         "days_since_template": f.get("days_since_template"),
                         "status": st.get("status") or "resident"})

        return {
            "visit_id": v["id"], "source": v["source"], "species": species,
            "started_at": v["started_at"], "ended_at": v["ended_at"],
            "n_crops": v["detection_count"],
            "rep_crop": _crop(v["representative_detection_id"]),
            "crops": crops, "clips": clips,
            "confirmed_as": s["confirmed_as"], "auto_as": s["auto_as"],
            "rejected": v["id"] in matcher.rejected,
            "candidates": s["candidates"], "clip_candidates": s["clip_candidates"],
            "novel": s["novel"], "multi": s["multi"],
            "co_present_frames": s["co_present_frames"],
            "co_present_clips": s["co_present_clips"],
            "cross_source": s.get("cross_source", False),
            "n_embedded": s["n_embedded"], "note": s["note"],
            "neighbours": neighbours, "cast": cast,
        }
    finally:
        conn.close()


#: How far back the review queue reaches by DEFAULT. The queue used to open on the entire corpus
#: -- 500+ visits -- which is not a work list, it is a wall, and the honest reaction to a wall is
#: to close the tab. Two nights is what a person can actually still remember seeing.
DEFAULT_QUEUE_WINDOW_H = 48


def _reid_queue(cfg, species: str = "raccoon", limit: int = 30, offset: int = 0,
                mode: str = "recent", since_h: int = DEFAULT_QUEUE_WINDOW_H) -> dict:
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
        # THE WINDOW, applied before the mode filter so "12 ambiguous" means 12 in the window the
        # reader is looking at, not 12 somewhere in two months. since_h <= 0 means "all of it",
        # which is what the UI's "load older" walks out to.
        n_all = len(pool)
        cutoff = None
        if since_h and since_h > 0:
            newest = max((db.parse_local(v["started_at"]) for v in pool), default=None)
            if newest is not None:
                # Anchored on the NEWEST VISIT, not on wall-clock now: after a quiet spell (or a
                # rig that was down, which happens here) a now-anchored window shows an empty
                # queue and reads as "nothing to review" when the truth is "nothing last night".
                cutoff = newest - timedelta(hours=float(since_h))
                pool = [v for v in pool
                        if (db.parse_local(v["started_at"]) or newest) >= cutoff]
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
                # Every candidate carries the LAPSE state of the name it proposes. A suggestion is
                # only as good as the freshest template behind it, and a two-week-old one is at or
                # below "just say the commonest name" -- so the card can stop presenting those two
                # cases as though they were the same offer. It is a LABEL, never a filter: gating
                # on template age was measured and rejected (coverage fell, wrong names rose).
                "candidates": [dict(c, lapse=(freshness.get(c.get("name")) or {}).get("lapse"))
                               for c in s["candidates"]],
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
            c["lapse"] = f.get("lapse") or individuals.identity_lapse(None, 0, cfg=cfg)
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
                # The window, so the UI can say what it is hiding rather than just hiding it.
                "since_h": since_h, "n_in_window": len(pool), "n_all": n_all,
                "window_from": cutoff.isoformat() if cutoff else None,
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
    # Case-insensitive: labels mix cases ("American crow" vs "band-tailed pigeon"), and a plain
    # sorted() split the dropdown into two alphabets -- capitals first, then a second A-Z run.
    return sorted(labels, key=str.casefold)


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


def start(cfg, frame_buffers: dict, control_bridges: dict, zone_store=None,
          specs=None):
    server = make_server(cfg, frame_buffers, control_bridges, zone_store, specs)
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
