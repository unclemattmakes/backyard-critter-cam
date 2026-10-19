"""Full-resolution crops for naming and re-ID, from a camera's snapshot endpoint.

A network camera's live stream is usually its low-resolution sub-stream: decoding a 4K main stream
costs ~75% of a core around the clock, and this host hangs under sustained load. But the stream
frame is also what every crop is cut from, so a raccoon across the yard reaches species naming and
re-ID at ~50 px. Many cameras (Reolink among them) serve a single full-resolution JPEG on request.

So the stream keeps doing the watching, and when it saves an animal the rig asks the camera for one
still, finds the animal in it again with the same detector, and swaps the detection's crop_path to
a cut from the still. The stream crop is kept in stream_crop_path. Nothing changes for a camera
without a snapshot URL, and every failure (timeout, wrong size, the animal gone) leaves the row
exactly as the stream wrote it.

Runs in its own thread per camera, so the capture loop never waits on HTTP. One job at a time: a
detection that arrives while a fetch is in flight keeps its stream crop rather than queueing behind.
"""
from __future__ import annotations

import queue
import ssl
import threading
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

import cameras
import db

# The still and the stream must share an aspect ratio, or a stream box cannot be mapped onto the
# still. 2% covers rounding (640x360 vs 3840x2160 is exact; 896x512 vs 2560x1440 is 1.6% off).
ASPECT_TOLERANCE = 0.02


@dataclass
class Job:
    """One detector pass worth of saved detections: [(detection_id, Detection, stream_crop_rel)],
    with the stream frame size their boxes are measured in."""
    dets: list
    frame_w: int
    frame_h: int


# Encrypted but unverified, for https snapshot URLs. A camera's certificate is self-signed, so
# verifying it fails every fetch; skipping the check still keeps the password (which a Reolink
# takes in the query string, on every detection) away from anyone passively sniffing the LAN.
# It does not stop an active impersonator, which would need the camera's certificate pinned.
# Scoped to snapshot fetches only: nothing else in the rig uses this context.
_CAMERA_TLS = ssl.create_default_context()
_CAMERA_TLS.check_hostname = False
_CAMERA_TLS.verify_mode = ssl.CERT_NONE


def fetch_jpeg(url: str, timeout: float):
    """GET one still and decode it, or None. A Reolink answers a bad login with HTTP 200 and a JSON
    error body, so the bytes are decoded rather than trusting the status."""
    tls = _CAMERA_TLS if url.lower().startswith("https://") else None
    with urllib.request.urlopen(url, timeout=timeout, context=tls) as resp:
        data = resp.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img


def scale_box(bbox, sx: float, sy: float):
    x1, y1, x2, y2 = bbox
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(stream_dets, snap_dets, sx: float, sy: float, min_iou: float) -> dict:
    """Pair stream detections with snapshot detections: {stream_index: snapshot Detection}.

    Snapshot boxes are scaled DOWN into stream coordinates (sx, sy = stream / still) and paired
    greedily, best IoU first, same class only, each box used once. Two raccoons side by side must
    not both get the crop of the nearer one, so a box that is already taken is not reused."""
    pairs = []
    for i, sd in enumerate(stream_dets):
        for j, nd in enumerate(snap_dets):
            if nd.class_name != sd.class_name:
                continue
            v = iou(sd.bbox, scale_box(nd.bbox, sx, sy))
            if v >= min_iou:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    out, used = {}, set()
    for _, i, j in pairs:
        if i in out or j in used:
            continue
        out[i] = snap_dets[j]
        used.add(j)
    return out


def cut(img, bbox, padding: float, max_side: int):
    """The padded, clamped box from `img`, shrunk (never enlarged) to max_side; None if empty."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = img[cy1:cy2, cx1:cx2]
    long_side = max(crop.shape[:2])
    if max_side and long_side > max_side:
        f = max_side / long_side
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), max(1, round(crop.shape[0] * f))),
                          interpolation=cv2.INTER_AREA)
    return crop


class SnapshotCropper:
    """One camera's snapshot worker. `detect` is the shared detector's detect(), called under
    `det_lock`; `fetch` is injectable so the tests never touch a network."""

    def __init__(self, spec, cfg, detect, det_lock, *, fetch=None, tag: str = ""):
        self.spec, self.cfg, self.detect, self.det_lock = spec, cfg, detect, det_lock
        self.fetch = fetch
        self.tag = tag or f"[{spec.source}]"
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        self._conn = None
        self.n_swapped = self.n_unmatched = self.n_failed = self.n_skipped = 0
        self._warned_aspect = False

    def start(self) -> "SnapshotCropper":
        self._thread = threading.Thread(target=self._run, name=f"snapshot-{self.spec.source}",
                                        daemon=True)
        self._thread.start()
        print(f"{self.tag} hi-res crops on ({cameras.safe_src(self.spec.snapshot_url)}).")
        return self

    def submit(self, job: Job) -> bool:
        """Hand a job to the worker without blocking. False (and counted) if it is still busy."""
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            self.n_skipped += len(job.dets)
            return False

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        if self.n_swapped or self.n_unmatched or self.n_failed or self.n_skipped:
            print(f"{self.tag} hi-res crops: {self.n_swapped} swapped, {self.n_unmatched} not "
                  f"found in the still, {self.n_failed} fetch failure(s), {self.n_skipped} "
                  f"skipped while busy.")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    job = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    self.process(job)
                except Exception as e:  # a bad still must never end the worker
                    self.n_failed += 1
                    print(f"{self.tag} hi-res crop failed: {e}")
        finally:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass

    def _db(self):
        # Opened on the worker's own thread: a sqlite connection is not shared across threads.
        if self._conn is None:
            self._conn = db.connect(self.cfg.db_path)
        return self._conn

    def process(self, job: Job) -> int:
        """Fetch, re-detect, cut and swap. Returns how many detections got a hi-res crop."""
        try:
            img = (self.fetch or fetch_jpeg)(self.spec.snapshot_url, self.cfg.snapshot_timeout_s)
        except Exception as e:
            img = None
            err = e
        else:
            err = None
        if img is None:
            self.n_failed += 1
            # The first failure explains itself; after that, every 50th, so a camera with HTTP off
            # cannot flood the log at one line per detection.
            if self.n_failed == 1 or self.n_failed % 50 == 0:
                why = f": {err}" if err else " (the reply was not an image -- wrong login?)"
                print(f"{self.tag} hi-res still unavailable{why} [{self.n_failed} so far]; "
                      f"keeping the stream crop.")
            return 0
        sh, sw = img.shape[:2]
        if abs(sw / sh - job.frame_w / job.frame_h) > ASPECT_TOLERANCE * (job.frame_w / job.frame_h):
            self.n_failed += 1
            if not self._warned_aspect:
                self._warned_aspect = True
                print(f"{self.tag} hi-res still is {sw}x{sh} but the stream is "
                      f"{job.frame_w}x{job.frame_h}; different shapes cannot be matched, so "
                      f"hi-res crops are idle until they agree.")
            return 0
        with self.det_lock:
            snap_dets = self.detect(img)
        sx, sy = job.frame_w / sw, job.frame_h / sh
        found = match([d for _, d, _ in job.dets], snap_dets, sx, sy,
                      self.cfg.snapshot_match_min_iou)
        swapped = 0
        for i, (det_id, det, stream_rel) in enumerate(job.dets):
            hit = found.get(i)
            if hit is None:
                self.n_unmatched += 1
                continue
            crop = cut(img, hit.bbox, self.cfg.crop_padding, self.cfg.snapshot_crop_max_side)
            if crop is None:
                self.n_unmatched += 1
                continue
            # Named after its stream crop, in the same day folder, so the pair sorts together.
            stream_abs = db.crop_abspath(stream_rel)
            path = stream_abs.with_name(stream_abs.stem + "_hires.jpg")
            if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality]):
                self.n_failed += 1
                continue
            try:
                if db.set_snapshot_crop(self._db(), det_id, db.rel_to_root(path)):
                    swapped += 1
            except Exception as e:
                # "database is locked" while the namer holds a long write: the stream crop stands.
                self.n_failed += 1
                print(f"{self.tag} hi-res crop {path.name} saved but not linked: {e}")
        self.n_swapped += swapped
        return swapped
