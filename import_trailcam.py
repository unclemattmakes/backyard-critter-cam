r"""
Trail-cam batch importer -- the SECOND capture source, folded into the SAME pipeline.

The live rig (backyard_cam.py) captures from the glass-door webcam frame by frame. The wider-
yard weatherproof trail cam (Voopeak/Campark TC02/TC08 family) is the opposite: its "WiFi" is a
short-range self-broadcast hotspot, it never joins the home network, and it has no push to a
server. So -- per PLAN.md ("Phase Later -- Trail-cam batch importer") -- we do NOT reverse-
engineer the camera. We dump its SD card into a folder and process that folder in BATCH, writing
rows with source='trail_cam_sd' (db.SOURCE_TRAIL_CAM_SD). Everything downstream (species ID,
re-ID, behaviour) already treats the `source` column as the only difference between rigs.

Same detector, same crop convention, same DB. The only real differences from the live rig:
  * frames come from FILES, not a webcam (no motion gate -- a trail cam already motion-triggered
    every shot, so every image is worth detecting on);
  * the timestamp comes from the image's EXIF DateTimeOriginal (the moment the cam fired), with
    the file's modification time as a fallback; and
  * trail-cam NIGHT frames are often IR GRAYSCALE -- the detector handles those fine (structure
    survives even when colour washes out; see PLAN.md "Day vs. night / color vs. IR").

VIDEOS (added 2026-07-27). In hybrid mode the camera writes a JPG burst at the trigger and an
MP4 a couple of seconds later. The stills are still the pipeline food -- they carry the crops,
species and re-ID -- but the MP4s are the BEHAVIOUR signal, and they used to be ignored entirely:
9 GB of gait, dwell and who-defers-to-whom sat on a card that gets formatted every cycle. They
now import as `clips` rows, the same table the live recorder writes, so clipmotion/clipembed and
the dashboard's player treat them exactly like glass-door clips (the TC02 already writes H.264,
so nothing is transcoded). Two things are deliberately different from a still import:
  * a video is COPIED into clips/<source>/<date>/ -- the card is its only home, and a row
    pointing at D:\ would dangle the moment the card is pulled;
  * only clips whose trigger produced an animal crop are kept (--all-videos overrides). The
    stills already answered "was anything there", and on a sun-triggered card that gate halved
    the disk cost. Empty triggers are common: see the 227-frame sun storm of 2026-07-13.
Disk is bounded per SOURCE (config.clips_max_gb_by_source), because one shared oldest-first
budget let an SD-card import evict the live rig's rolling window -- backwards, since the rig can
re-record tomorrow and the card's footage dies at the next format. That same asymmetry is what
--backup-first is for: where the card is formatted every cycle, a prune does not TRIM the clip
history, it DESTROYS it, so the flag archives (backup.py) before the import and skips the prune
outright if that archive fails. Enforcing a disk budget against the only surviving copy is never
the right trade.

STATIC FALSE-FIRES are dropped between the stills and the videos (staticfilter.py, added
2026-08-05). A detector pointed at a yard fires on furniture: on the 2026-08-04 card a covered
Weber grill, its chimney starter, a dark gap that read as eyeshine and a shrub lit by the IR
flash produced 395 of 789 detections -- half the cycle, labelled 'brown rat' and 'Virginia
opossum' by a classifier doing its job on a barbecue. The live rig fixes this with hand-measured
config.ignore_zones, which is wrong HERE: this camera is repositioned on purpose, and a stale
zone fails silently. So the filter derives the spots per batch instead -- boxes that repeat in
one place for hours are furniture, because an animal's box changes shape as it moves -- and the
ORDER matters: it runs before the video pass so the clip gate never buys disk for footage whose
only 'animal' was the grill. --no-static-filter keeps everything.

  python import_trailcam.py D:\DCIM\100MEDIA                 # ingest one SD-card dump (GPU)
  python import_trailcam.py D:\dump --recursive              # walk subfolders too
  python import_trailcam.py D:\dump --device cpu             # no GPU (slower; fine for batch)
  python import_trailcam.py D:\dump --processed-dir D:\done  # move each file after import
  python import_trailcam.py D:\dump --no-videos              # stills only, ignore the .MP4s
  python import_trailcam.py D:\dump --all-videos             # keep empty-trigger clips too
  python import_trailcam.py D:\drop --watch                  # poll a drop folder forever
  python import_trailcam.py D:\DCIM\100MEDIA --backup-first  # archive before anything can prune
  python import_trailcam.py D:\dump --no-static-filter       # keep static false-fires too

SPECIES COME AFTER THE IMPORT (same as live crops): rows land with species NULL, and the visit
ledger is refreshed right away so the Behaviour tab shows the new visits -- unlabeled at first.
classify.py fills the labels (the rig's naming helper picks them up within moments if it's
running; otherwise run `python classify.py`) and refreshes the ledger AGAIN when it does, so the
pipeline ends with labeled visits either way -- no manual `python visits.py`. (Before classify
did that second refresh, 90 of 113 trail-cam visits sat species-less until a manual rebuild --
hit for real 2026-07-22.)

IDEMPOTENCY (re-runs must not double-import). Three layers, all boring on purpose:
  * ledger: a plain-text sidecar next to the DB (backyard.db.imported-<source>.txt) records every
    file imported for this `source`, one 'basename|capture-second' key per line (e.g.
    'IMAG0001.JPG|2026-07-13T06-42-11'), appended as files land. At startup we read it back into
    a skip-set, so a plain re-run over the same folder is a no-op -- even for images that produced
    NO crop (a frame with no animal still gets logged, so it isn't re-detected). This is the
    primary, always-on default; it survives whether or not a crop was saved.
  * crop_paths: every crop's name leads with the capture stamp and embeds 'src-<orig-stem>', so
    the same stem|capture-second skip keys are rebuilt from the DB's crop_paths too -- a belt-
    and-braces recovery if the ledger is deleted (crops that exist won't be re-imported).
  * --processed-dir: after a file imports cleanly, MOVE it there, so a re-run never even sees it.
    The recommended way to run an ongoing drop folder (and what --watch leans on to drain it).

  (Idempotency keys on BASENAME + CAPTURE SECOND, scoped to `source` -- NOT the basename alone.
  TC02-family cams restart numbering at IMAG0001.JPG after every in-camera card format, so a
  later cycle's dump reuses an earlier cycle's filenames -- keyed on the name alone, the ledger
  silently skipped those real new files (hit for real 2026-07-19: 235 of 558 would have been
  dropped). The capture second comes from image_timestamp() -- EXIF DateTimeOriginal, which trail
  cams stamp reliably and which survives copies and re-mounts -- so the SAME photo is recognised
  wherever it's rescanned from, while a recycled NAME from a new cycle imports cleanly. Blind
  spot: an EXIF-less file keys on its mtime, so re-copying such a file can re-import it;
  --processed-dir sidesteps even that.

  Ledger lines written before this fix hold the bare basename. A bare name cannot tell "same
  file re-scanned" from "new cycle reused the name", and guessing 'skip' is exactly how real
  photos got lost -- so legacy lines no longer skip anything by themselves. Files they recorded
  stay covered by the crop_paths recovery (the capture stamp was always in the crop name); a
  pre-fix file that saved no crop just costs one extra no-op detector pass if its dump is ever
  re-scanned, and its ledger line is re-appended in the new format as it goes.)

Robust by design (PLAN.md "boring and robust"): an unreadable/corrupt image is warned about and
skipped; one bad file never aborts the batch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import cv2

import clips
import config
import db
import staticfilter
import visits
from config import CONFIG
# Reuse the live rig's exact crop + path logic so trail-cam crops are byte-for-byte the same
# convention as glass-door crops (same padded/clamped box, same filename, same crops_dir/<date>/
# layout). Importing them keeps a single source of truth -- if save_crop changes, both rigs move.
from backyard_cam import _rel, save_crop
from detector import CudaUnavailableError, Detection, Detector

# Image extensions we ingest. Trail cams write JPEG; PNG is accepted for completeness / exports.
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Video extensions we ingest as behaviour clips. The TC02 writes H.264 .MP4 -- the same codec
# clips.py records for the live rig -- so an imported clip plays in the dashboard with no
# transcode and clipmotion.py reads it exactly like a glass-door clip.
VIDEO_EXTS = {".mp4"}

# How far outside a clip's own time span a still's detection may sit and still count as the same
# trigger event. Generous on purpose, and measured: on a 599-clip card every clip either had a
# detection within a couple of seconds or none within a minute, so the exact value isn't delicate.
VIDEO_PAIR_WINDOW_S = 30.0

# Largest image we'll decode from an (untrusted) SD card. cv2.imread allocates a full uncompressed
# buffer sized from the file header, so a 'decompression bomb' (tiny file, huge declared dimensions)
# can OOM the import. We reject anything over this via a cheap Pillow header read (no full decode)
# before imread -- a real trail-cam frame is far under it.
MAX_DECODE_PIXELS = 120_000_000   # ~120 MP; typical trail cams are <= ~12 MP


def _within_pixel_budget(path: Path) -> bool:
    """True if `path`'s declared dimensions are sane to decode; False if oversized. Uses Pillow's
    lazy header read (.size does NOT decode the pixels), so a bomb image costs microseconds. An
    unreadable header returns True -- let cv2.imread try and the existing corrupt-file path handle it
    (this guard targets only the 'huge declared size' OOM, not general decode failures)."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS   # also trips Pillow's own bomb guard on decode
        with Image.open(path) as im:
            w, h = im.size
        return (w * h) <= MAX_DECODE_PIXELS
    except Exception:
        return True

# A short marker baked into each crop's filename component so the original SD-card filename is
# recoverable from crop_path; together with the capture stamp that leads the crop name, that's
# what powers the DB-recovery skip-set (see module docstring). Kept filesystem-safe and unlikely
# to collide with a real stem.
SRC_TAG = "src-"

# Skip keys pair the file's NAME with its CAPTURE SECOND ('IMAG0001.JPG|2026-07-13T06-42-11').
# The name alone is not unique across card-format cycles (a TC02 restarts at IMAG0001.JPG), and
# the second alone is not unique within a burst -- together they are, for any sane camera.
KEY_TS_FMT = "%Y-%m-%dT%H-%M-%S"   # capture second, colon-free; the crop stamp is this + '-<ms>'
LEDGER_KEY_SEP = "|"               # illegal in Windows filenames, so it never collides with a name


def skip_key(name: str, ts: str) -> str:
    """The idempotency key for one source image: filename (or stem) + its KEY_TS_FMT capture
    second. The ledger stores name-keyed lines; imported_keys() rebuilds stem-keyed ones (crop
    names drop the extension) -- import_folder checks both spellings against the skip-set."""
    return f"{name}{LEDGER_KEY_SEP}{ts}"


def list_images(folder: Path, recursive: bool) -> list[Path]:
    """Every image file in `folder` (optionally walking subfolders), sorted for a stable,
    chronological-ish import order. Case-insensitive on extension (SD cards love .JPG)."""
    walker = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in walker if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _exif_datetime(path: Path):
    """The moment the trail cam fired, read from EXIF DateTimeOriginal (tag 36867) via Pillow,
    as a naive local datetime; None if absent/unparseable. Trail cams stamp this reliably, and
    it beats the file mtime (which is really "when the SD card was copied"). Best-effort: any
    failure (no EXIF, odd format, unreadable header) just falls back to mtime upstream."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
        if not exif:
            return None
        raw = exif.get(36867) or exif.get(306)  # DateTimeOriginal, else DateTime
        if not raw:
            return None
        # EXIF format is "YYYY:MM:DD HH:MM:SS" (note the ':' date separators).
        return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def image_timestamp(path: Path) -> datetime:
    """Local, tz-aware capture time for an image: EXIF DateTimeOriginal if present, else the
    file's modification time. Either way it's made tz-aware with the local UTC offset via
    .astimezone(), matching db.now_local_iso()'s ISO 8601 'local time WITH offset' convention."""
    dt = _exif_datetime(path)
    if dt is None:
        dt = datetime.fromtimestamp(path.stat().st_mtime)
    return dt.astimezone()


def imported_keys(conn, source: str) -> set[str]:
    """Skip keys ('<orig-stem>|<capture-second>') for everything already imported for `source`,
    recovered from the crop_paths in the DB -- each crop name leads with the capture stamp and
    embeds 'src-<stem>'. The belt-and-braces fallback that keeps re-runs idempotent if the ledger
    sidecar is deleted, and what still covers files imported back when ledger lines were bare
    basenames (the stamp was always in the crop name, even then). Only full stamp+name PAIRS are
    recovered -- a row whose stamp doesn't parse contributes nothing, because a bare name must
    never skip a file (see module docstring). Empty set on a fresh DB."""
    out: set[str] = set()
    for (cp,) in conn.execute(
        "SELECT crop_path FROM detections WHERE source = ?", (source,)
    ):
        if not cp:
            continue
        stem = Path(cp).name
        i = stem.find(SRC_TAG)
        if i == -1:
            continue
        rest = stem[i + len(SRC_TAG):]
        # crop name = '<capture-stamp>_src-<orig-stem>_<idx>_<class>_<conf>.jpg'. save_crop appends
        # exactly THREE underscore-fields (idx, class, conf), so strip those from the RIGHT to
        # recover the full original stem -- a left split mangles any source name that itself
        # contains an underscore (e.g. IMG_0042 -> 'IMG'), which then never matches the file again
        # on a DB-recovery re-run.
        orig_stem = rest.rsplit("_", 3)[0]
        # The stamp before '_src-' is KEY_TS_FMT plus a '-<ms>' field (see ingest_file). Drop the
        # milliseconds to get key granularity, and validate: garbage in, no key out.
        ts = stem[:i].rstrip("_").rsplit("-", 1)[0]
        try:
            datetime.strptime(ts, KEY_TS_FMT)
        except ValueError:
            continue
        out.add(skip_key(orig_stem, ts))
    return out


def ledger_path(db_path: Path, source: str) -> Path:
    """The plain-text import ledger that sits beside the DB, scoped to one `source`. Filesystem-
    safe: any non-alphanumeric char in the source name becomes '_'."""
    safe = "".join(c if c.isalnum() else "_" for c in source)
    return Path(db_path).with_name(Path(db_path).name + f".imported-{safe}.txt")


def read_ledger(db_path: Path, source: str) -> set[str]:
    """Skip keys ('<basename>|<capture-second>') already recorded for `source`, per the sidecar
    ledger. The always-on default skip-set; survives even for frames that produced no crop. Empty
    set if the ledger is absent. Pre-fix lines holding a bare basename (no '|') are read but
    IGNORED: a name alone can't tell a re-scan from a new cycle reusing the name, and skipping on
    it loses real photos -- files those lines recorded stay covered by imported_keys() (see module
    docstring)."""
    p = ledger_path(db_path, source)
    if not p.exists():
        return set()
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return set()
    return {ln.strip() for ln in lines if LEDGER_KEY_SEP in ln}


def append_ledger(db_path: Path, source: str, key: str) -> None:
    """Record one imported file's skip key in the sidecar ledger (append + flush, so a crash
    mid-batch still leaves earlier files marked). Best-effort: a write failure never fails the
    import."""
    try:
        with open(ledger_path(db_path, source), "a", encoding="utf-8") as f:
            f.write(key + "\n")
    except Exception as e:
        print(f"  [warn] could not update import ledger for {key}: {e}")


def ingest_file(path: Path, detector: Detector, conn, cfg: config.Config,
                source: str) -> tuple[int, int]:
    """Import ONE image into the pipeline: load it, run the detector, and for each detection whose
    class is in cfg.save_classes, save a crop (live-rig convention) and write a DB row tagged with
    `source`. Returns (n_detections_reported, n_saved). The frame is loaded with cv2.imread, which
    returns None on a missing/corrupt file -- handled by the caller (warn + skip).

    Mirrors backyard_cam.run()'s per-detection block, minus the motion gate / preview: a trail cam
    already motion-triggered every shot, so every frame is worth a detector pass.
    """
    if not _within_pixel_budget(path):
        print(f"  skip (image too large to decode safely): {path.name}")
        return -1, 0
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"  skip (unreadable/corrupt): {path.name}")
        return -1, 0  # sentinel: distinguishes a bad file from a readable file with 0 detections

    try:
        dets = detector.detect(frame)
    except Exception as e:  # never let one bad frame kill the batch
        print(f"  [detector] error on {path.name} (skipping): {e}")
        return -1, 0

    saved_dets = [d for d in dets if d.class_name in cfg.save_classes]
    if not saved_dets:
        return len(dets), 0

    dt = image_timestamp(path)
    iso = dt.isoformat()
    day = dt.strftime("%Y-%m-%d")
    # Filename stamp matches the live rig (timestamp_index_class_conf), with the source image's
    # stem spliced in as 'src-<stem>' so the import is traceable AND the default skip-set works.
    # It leads with the capture second in KEY_TS_FMT (plus milliseconds) precisely so
    # imported_keys() can parse the skip key back out of the crop name -- keep them in lockstep.
    base_stamp = dt.strftime(KEY_TS_FMT) + f"-{dt.microsecond // 1000:03d}"
    stamp = f"{base_stamp}_{SRC_TAG}{path.stem}"
    h, w = frame.shape[:2]

    saved = 0
    for i, det in enumerate(saved_dets):
        result = save_crop(frame, det, cfg, day, stamp, i)
        if result is None:
            continue
        crop_path, crop_q = result   # save_crop returns (Path, shot-quality), exactly like the live rig
        db.insert_detection(
            conn,
            timestamp=iso,
            source=source,
            detection_class=det.class_name,
            confidence=det.confidence,
            bbox=det.bbox,
            frame_w=w,
            frame_h=h,
            crop_path=_rel(crop_path),
            frame_path=None,   # the SD card already holds the original full frame; don't copy it.
            crop_quality=crop_q,   # score trail-cam crops too, so the dashboard can lead with the cutest
            # species / individual_id stay NULL -- classify.py / reid.py fill them later, same as
            # the live rig's crops. The 'source' column is the only thing marking these as trail-cam.
        )
        saved += 1
        print(f"  [{iso}] {det.class_name} {det.confidence:.2f} -> {_rel(crop_path)}  ({path.name})")
    return len(dets), saved


def list_videos(folder: Path, recursive: bool) -> list[Path]:
    """Every video file in `folder` (optionally walking subfolders), sorted like list_images.
    Case-insensitive on extension (SD cards write .MP4)."""
    walker = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in walker if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def _probe_video(path: Path) -> dict | None:
    """{duration_s, fps, width, height, frame_count} for one video, or None if it can't be read.
    ffprobe first (exact container duration), else OpenCV, which every install already has --
    cv2 reports frame count and fps, and duration is derived from them. Never raises."""
    import shutil as _sh
    ffprobe = _sh.which("ffprobe")
    if ffprobe:
        try:
            import subprocess
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,r_frame_rate,nb_frames", "-show_entries",
                 "format=duration", "-of", "json", str(path)],
                capture_output=True, text=True, timeout=30)
            d = json.loads(r.stdout or "{}")
            st = (d.get("streams") or [{}])[0]
            num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
            fps = float(num) / float(den or 1) if float(den or 1) else 0.0
            dur = float((d.get("format") or {}).get("duration") or 0.0)
            n = int(st.get("nb_frames") or 0) or (int(round(dur * fps)) if fps else 0)
            if dur > 0 and st.get("width"):
                return {"duration_s": dur, "fps": fps or None, "width": int(st["width"]),
                        "height": int(st["height"]), "frame_count": n}
        except Exception:
            pass                                   # fall through to the cv2 path
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if not (n and fps and w):
            return None
        return {"duration_s": n / fps, "fps": fps, "width": w, "height": h, "frame_count": n}
    except Exception:
        return None


def video_span(path: Path, meta: dict) -> tuple[datetime, datetime]:
    """(started_at, ended_at) for a trail-cam video, both tz-aware local. A video file's mtime is
    when the camera FINISHED writing it, not when recording began -- verified on this TC02 against
    the burned-in overlay clock, where mtime minus duration matched the visible start time. So the
    end is the mtime and the start is derived, which is the opposite of the stills (whose EXIF
    stamp is the trigger instant)."""
    ended = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    return ended - timedelta(seconds=float(meta["duration_s"])), ended


def video_skip_key(path: Path) -> str:
    """Idempotency key for one video: filename + its mtime second. Deliberately keyed on the RAW
    mtime rather than the derived start time -- the skip check then costs one stat() and never has
    to probe a file it is about to skip (probing 599 clips on every re-run to decide they're all
    duplicates is exactly the kind of slow no-op this ledger exists to avoid)."""
    ts = datetime.fromtimestamp(path.stat().st_mtime).astimezone().strftime(KEY_TS_FMT)
    return skip_key(path.name, ts)


def imported_video_keys(conn, source: str) -> set[str]:
    """Video skip keys recovered from the `clips` table -- the belt-and-braces fallback for videos,
    mirroring imported_keys() for stills. The original filename comes from the 'src-<stem>' tag in
    clip_path and the mtime second from the row's `ended_at` (which IS the mtime by video_span's
    contract), so the pair reconstructs exactly what video_skip_key would produce. Rows whose path
    or timestamp doesn't parse contribute nothing: a bare name must never skip a file."""
    out: set[str] = set()
    for cp, ended_at in conn.execute(
        "SELECT clip_path, ended_at FROM clips WHERE source = ?", (source,)
    ):
        if not cp or not ended_at:
            continue
        name = Path(cp).name
        i = name.find(SRC_TAG)
        if i == -1:
            continue
        stem = Path(name[i + len(SRC_TAG):]).stem
        try:
            ts = datetime.fromisoformat(ended_at).strftime(KEY_TS_FMT)
        except ValueError:
            continue
        # The ledger keys on the full filename; recovery only knows the stem, so emit both
        # spellings and let import_videos accept either (same trick as the stills path).
        out.add(skip_key(stem, ts))
    return out


def paired_detections(conn, source: str, start: datetime, end: datetime,
                      window_s: float) -> tuple[int, float | None]:
    """(n_detections, max_confidence) for the stills that belong to this clip's trigger event --
    every `source` detection landing within `window_s` of the clip's own span. The window reaches
    BACKWARDS from the start because the camera writes the JPG burst at the trigger and only
    begins the MP4 a couple of seconds later, so the animal's stills sit just before the video."""
    lo = (start - timedelta(seconds=window_s)).isoformat()
    hi = (end + timedelta(seconds=window_s)).isoformat()
    n, mx = conn.execute(
        "SELECT COUNT(*), MAX(confidence) FROM detections "
        "WHERE source = ? AND timestamp >= ? AND timestamp <= ?", (source, lo, hi)).fetchone()
    return int(n or 0), mx


def ingest_video(path: Path, conn, cfg: config.Config, source: str, *,
                 window_s: float, require_animal: bool) -> str:
    """Import ONE trail-cam video as a behaviour clip: probe it, decide whether its trigger caught
    an animal, copy it into clips/<source>/<date>/ and write the `clips` row. Never raises on a
    bad file. Returns one of:
      'stored'     -- copied and rowed;
      'no-animal'  -- deliberately skipped, a settled decision;
      'unreadable' / 'copy-failed' -- TRANSIENT, so the caller must NOT mark it imported.
    That last distinction matters: a video still being written (or on a flaky card) probes as
    unreadable, and ledgering it would retire the file forever on the strength of a torn read.

    The copy is deliberate: the card is the ONLY home for these files and gets formatted every
    cycle, so a clips row pointing at D:\\ would dangle the moment the card is pulled. copy2
    preserves the mtime so an imported clip ages honestly in the rolling window."""
    meta = _probe_video(path)
    if meta is None:
        print(f"  skip (unreadable video, will retry next run): {path.name}")
        return "unreadable"
    start, end = video_span(path, meta)
    n_det, max_conf = paired_detections(conn, source, start, end, window_s)
    if require_animal and n_det == 0:
        # No still from this trigger produced an animal crop. On this camera that means sun,
        # wind or heat tripped the PIR -- the clip is empty. Cheap to re-import later with
        # --all-videos if that assumption ever proves wrong.
        return "no-animal"

    day = start.strftime("%Y-%m-%d")
    stamp = start.strftime(KEY_TS_FMT) + f"-{start.microsecond // 1000:03d}"
    dest_dir = cfg.clips_dir / clips._safe_source(source) / day
    dest = dest_dir / f"{stamp}_{SRC_TAG}{path.stem}{path.suffix.lower()}"
    try:
        import shutil as _sh
        dest_dir.mkdir(parents=True, exist_ok=True)
        _sh.copy2(path, dest)                       # copy2 keeps mtime -> honest prune ordering
    except OSError as e:
        print(f"  [warn] could not copy {path.name} into clips/: {e}")
        return "copy-failed"

    db.insert_clip(
        conn, source=source, clip_path=_rel(dest),
        started_at=start.isoformat(), ended_at=end.isoformat(),
        fps=meta.get("fps"), width=meta.get("width"), height=meta.get("height"),
        frame_count=int(meta.get("frame_count") or 0),
        detection_count=n_det, max_confidence=max_conf,
    )
    print(f"  [{start.isoformat()}] clip {meta['duration_s']:.1f}s "
          f"{meta.get('width')}x{meta.get('height')} ({n_det} det) -> {_rel(dest)}  ({path.name})")
    return "stored"


# ingest_video outcomes that are SETTLED -- the file has been dealt with and its ledger line can
# be written. Anything else is transient (torn/locked file) and must stay unledgered so the next
# run retries it.
_VIDEO_DONE = {"stored", "no-animal"}


def import_videos(folder: Path, conn, cfg: config.Config, *, source: str, recursive: bool,
                  skip: set[str], window_s: float, require_animal: bool,
                  processed_dir: Path | None = None) -> tuple[int, int, int]:
    """Import every video in `folder` once, as behaviour clips. Same idempotency contract as
    import_folder (the shared ledger, keyed per file), and the same never-fail-the-batch posture.
    Returns (stored, skipped_already, skipped_no_animal).

    Videos are imported AFTER the stills on purpose: the animal gate reads the detections the
    stills just wrote, so the two passes must not be interleaved."""
    videos = list_videos(folder, recursive)
    if not videos:
        return 0, 0, 0
    print(f"\nFound {len(videos)} video(s) in {folder}. Importing as clips "
          f"(source='{source}'{'' if require_animal else ', ALL videos'}) ...")
    stored = already = empty = 0
    for path in videos:
        try:
            key = video_skip_key(path)
        except OSError:
            print(f"  skip (vanished mid-scan): {path.name}")
            continue
        stem_key = skip_key(path.stem, key.split(LEDGER_KEY_SEP, 1)[1])
        if key in skip or stem_key in skip:
            already += 1
            continue
        status = ingest_video(path, conn, cfg, source, window_s=window_s,
                              require_animal=require_animal)
        if status not in _VIDEO_DONE:
            continue                    # transient -- leave it unledgered so a re-run retries it
        stored += (status == "stored")
        # A no-animal video is still ledgered: that verdict came from its trigger's stills, which
        # don't change on a re-run, so re-probing it every cycle would burn minutes to reach the
        # same answer. --all-videos is the way to bring those in later.
        empty += (status == "no-animal")
        skip.add(key)
        append_ledger(cfg.db_path, source, key)
        if processed_dir is not None:
            move_to_processed(path, processed_dir)
    return stored, already, empty


def move_to_processed(path: Path, processed_dir: Path) -> None:
    """Move an imported file into processed_dir so re-runs (and --watch) skip it. On a name
    collision there, suffix '_1', '_2', ... rather than clobbering an existing file. Best-effort:
    a move failure is warned about but doesn't fail the import that already succeeded."""
    try:
        processed_dir.mkdir(parents=True, exist_ok=True)
        dest = processed_dir / path.name
        n = 1
        while dest.exists():
            dest = processed_dir / f"{path.stem}_{n}{path.suffix}"
            n += 1
        path.replace(dest)
    except Exception as e:
        print(f"  [warn] imported but could not move {path.name} to {processed_dir}: {e}")


def run_backup_first(cfg: config.Config) -> bool:
    """Archive clips/crops/db (backup.py) BEFORE this import can prune anything. True on success.

    The ORDERING is the whole point. prune_clips enforces a per-source disk budget by deleting the
    OLDEST clips, and for a trail cam that is not a trim but a destruction: the card is formatted
    every cycle, so clips/<source>/ holds the only copy that footage will ever have. Archiving
    first means the clips actually at risk -- the old ones -- are already in the backup before
    anything can delete them. (backup.py skips TODAY's folder, but today's clips are the newest
    and so the last thing a prune would touch; they're archived on the next run.)

    Shelling out rather than importing backup.py keeps its argparse/logging setup to itself, and
    means a crash in the archiver can't take the import down with it.
    """
    import subprocess
    script = Path(__file__).with_name("backup.py")
    # flush before handing the terminal to the child: backup.py writes to this same stream, and an
    # unflushed buffer here would print our banner AFTER its output in a redirected log.
    print(f"\n[backup] archiving to {cfg.backup_dest} before importing, so a prune can never "
          "evict un-archived footage ...", flush=True)
    try:
        r = subprocess.run([sys.executable, str(script)], cwd=str(script.parent))
    except Exception as e:
        print(f"  [backup] could not run {script.name}: {e}")
        return False
    if r.returncode != 0:
        print(f"  [backup] FAILED (exit {r.returncode}) -- see the output above.")
        return False
    print("  [backup] done.")
    return True


def refresh_visits(conn, cfg: config.Config) -> None:
    """Fold the just-imported detections into the visit ledger, so the dashboard's Behaviour tab
    shows the trail-cam visits right away. At this point they are UNLABELED -- ingest_file leaves
    species NULL on purpose -- and they stay that way until classify.py names the crops, which
    refreshes the ledger AGAIN itself (that second refresh is what ends the pipeline with labeled
    visits; _print_next_step tells the user which way it'll happen). Best-effort by visits.refresh's
    contract: an error never fails an import that already succeeded."""
    visits.refresh(conn, cfg.visit_gap_minutes)


def _naming_helper_alive() -> bool:
    """True when a live species-naming helper (classify.py --watch, normally spawned by the rig)
    has a FRESH heartbeat in its status file -- the same loading/ready-under-30s rule the
    dashboard header uses (web._naming_status). Read-only and best-effort: any problem (no file,
    stale, unparseable) reads as 'not running'."""
    try:
        data = json.loads(config.NAMING_STATUS_FILE.read_text())
        return (data.get("state") in ("loading", "ready")
                and (time.time() - float(data.get("ts", 0))) <= 30)
    except Exception:
        return False


def _print_next_step(saved: int) -> None:
    """One line on how the new crops get their species. The import itself leaves species NULL, so
    the visits just refreshed are unlabeled until classify.py runs -- and classify refreshes the
    ledger again when it labels them, so neither path needs a manual `python visits.py`."""
    if _naming_helper_alive():
        print(f"  {saved} new crop(s) await species labels -- the rig's naming helper is running "
              "and will label them shortly (the visit ledger refreshes itself again then).")
    else:
        print(f"  {saved} new crop(s) await species labels -- run `python classify.py` to name "
              "them (it refreshes the visit ledger itself when done).")


def import_folder(folder: Path, detector: Detector, conn, cfg: config.Config, *,
                  source: str, recursive: bool, processed_dir: Path | None,
                  skip: set[str]) -> tuple[int, int, int]:
    """Import every image in `folder` once. `skip` is the set of skip keys (basename|capture-
    second) already imported (default idempotency); files moved to --processed-dir won't reappear
    anyway. Updates `skip` in place as it goes so a single pass never imports the same file twice.
    Returns (files_imported, crops_saved, files_skipped)."""
    images = list_images(folder, recursive)
    if not images:
        print(f"No images ({'/'.join(sorted(IMAGE_EXTS))}) found in {folder}"
              f"{' (recursive)' if recursive else ''}.")
        return 0, 0, 0

    print(f"Found {len(images)} image(s) in {folder}"
          f"{' (recursive)' if recursive else ''}. Importing as source='{source}' ...")
    imported = saved_total = skipped = 0
    for path in images:
        # The capture second is a cheap EXIF header read (no pixel decode) -- fine to do even for
        # files we're about to skip. stat() can still race a --watch drainer moving the file out
        # from under us; treat a vanished file like a corrupt one (warn-ish and move on).
        try:
            ts = image_timestamp(path).strftime(KEY_TS_FMT)
        except OSError:
            print(f"  skip (vanished mid-scan): {path.name}")
            continue
        # The ledger keys on the full filename; the DB-recovery fallback keys on the stem (the
        # extension isn't stored in the crop name) -- accept either so both skip-sources match.
        if skip_key(path.name, ts) in skip or skip_key(path.stem, ts) in skip:
            skipped += 1
            continue
        n_reported, n_saved = ingest_file(path, detector, conn, cfg, source)
        if n_reported < 0:
            continue  # unreadable/corrupt -- already warned, leave it in place to inspect
        skip.add(skip_key(path.name, ts))
        # Mark imported (even if 0 crops) -> idempotent; also how a pre-fix bare line gets
        # upgraded to the keyed format when its file is re-scanned.
        append_ledger(cfg.db_path, source, skip_key(path.name, ts))
        imported += 1
        saved_total += n_saved
        if n_saved == 0:
            print(f"  [{path.name}] no saveable detections "
                  f"({n_reported} reported, none in save_classes).")
        if processed_dir is not None:
            move_to_processed(path, processed_dir)
    return imported, saved_total, skipped


def watch_folder(folder: Path, detector: Detector, conn, cfg: config.Config, *,
                 source: str, recursive: bool, processed_dir: Path | None,
                 skip: set[str], interval: float) -> int:
    """Ongoing drop-folder mode: import the folder, then poll it every `interval`s for newly
    dropped files and import those too, until Ctrl-C. Reuses the one-shot import_folder per pass;
    the `skip` set (seeded from the DB) carries across passes so nothing re-imports. Pairs well
    with --processed-dir, which also drains the folder as it goes."""
    print(f"Watching {folder} for new trail-cam images every {interval:g}s "
          f"(source='{source}'; Ctrl-C to stop).")
    print("  [watch] stills only -- videos are imported by a one-shot run over the card, where "
          "every clip is finished being written.")
    total_imported = total_saved = 0
    try:
        while True:
            imported, saved, _ = import_folder(
                folder, detector, conn, cfg, source=source, recursive=recursive,
                processed_dir=processed_dir, skip=skip)
            total_imported += imported
            total_saved += saved
            if imported:
                print(f"  [watch] +{imported} file(s), +{saved} crop(s) "
                      f"(session: {total_imported} files / {total_saved} crops).")
                if saved:
                    refresh_visits(conn, cfg)
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n[watch] stopped. Imported {total_imported} file(s), "
              f"{total_saved} crop(s) this session.")
    return 0


def parse_args() -> tuple[config.Config, argparse.Namespace]:
    c = CONFIG
    p = argparse.ArgumentParser(
        description="Batch-import a trail-cam SD-card folder into the backyard pipeline "
                    "(source='trail_cam_sd').",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("folder", help="Folder of trail-cam images (the SD-card dump / drop folder).")
    p.add_argument("--recursive", action="store_true",
                   help="Walk subfolders too (SD cards often nest under DCIM/100MEDIA/...).")
    p.add_argument("--device", default=c.device, choices=["cuda", "cpu", "auto"],
                   help="Inference device: auto (GPU when it genuinely runs, else CPU) | cuda "
                        "(REQUIRE an NVIDIA GPU) | cpu (no GPU, slower -- fine for an "
                        "overnight batch).")
    p.add_argument("--min-confidence", type=float, default=c.min_confidence,
                   help="Minimum detector confidence to save a crop.")
    p.add_argument("--source", default=db.SOURCE_TRAIL_CAM_SD,
                   help="Value written to detections.source (default 'trail_cam_sd').")
    p.add_argument("--db", default=str(c.db_path),
                   help="SQLite database to write to (default config.CONFIG.db_path).")
    p.add_argument("--crops-dir", default=str(c.crops_dir),
                   help="Where crops are written (foldered by date), like the live rig.")
    p.add_argument("--processed-dir", default=None,
                   help="If set, MOVE each file here after a successful import so re-runs skip it "
                        "(recommended for an ongoing drop folder).")
    p.add_argument("--watch", action="store_true",
                   help="Poll the folder forever, importing newly dropped files (Ctrl-C to stop).")
    p.add_argument("--interval", type=float, default=10.0,
                   help="Seconds between polls in --watch mode.")
    p.add_argument("--no-videos", dest="videos", action="store_false", default=True,
                   help="Import stills only; leave the card's .MP4 clips alone.")
    p.add_argument("--backup-first", action="store_true",
                   help="Run backup.py BEFORE importing, so any clip this run might prune is "
                        "archived first. Recommended whenever the card is formatted every cycle, "
                        "because a prune then deletes the ONLY copy. If the backup fails the "
                        "import still runs, but the prune is skipped for that run.")
    p.add_argument("--all-videos", action="store_true",
                   help="Import EVERY video, not just the ones whose trigger produced an animal "
                        "crop. Doubles the disk cost on a sun-triggered card.")
    p.add_argument("--video-pair-window", type=float, default=VIDEO_PAIR_WINDOW_S,
                   help="Seconds around a clip's span to look for its trigger's detections.")
    p.add_argument("--no-static-filter", dest="static_filter", action="store_false", default=True,
                   help="KEEP static false-fires (a grill, a post, foliage lit by the IR flash) "
                        "instead of dropping boxes that sit in one spot for hours. The filter "
                        "measures those spots per batch rather than from config, so it survives "
                        "moving the camera -- see staticfilter.py.")
    p.add_argument("--static-min-count", type=int, default=staticfilter.DEFAULT_MIN_COUNT,
                   help="Detections in one spot before the static filter calls it furniture.")
    p.add_argument("--static-min-span-minutes", type=float,
                   default=staticfilter.DEFAULT_MIN_SPAN_MINUTES,
                   help="How long one spot must keep firing before it counts as static.")
    p.add_argument("--static-iou", type=float, default=staticfilter.DEFAULT_IOU,
                   help="How identical two boxes must be to count as the same spot.")
    args = p.parse_args()

    cfg = replace(
        c,
        device=args.device,
        min_confidence=args.min_confidence,
        db_path=Path(args.db),
        crops_dir=Path(args.crops_dir),
    )
    return cfg, args


def main() -> int:
    cfg, args = parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[ERROR] Not a folder: {folder}")
        return 1
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    # Archive BEFORE anything else, while we still hold no DB connection of our own (backup.py
    # snapshots the database) and before a single new clip can push the budget over. A FAILED
    # backup does not stop the import -- it disables the prune instead, which is the fail-safe
    # direction: worst case clips/ runs over budget until the next successful backup, where the
    # other way round costs footage that exists nowhere else.
    prune_ok = True
    if args.backup_first:
        prune_ok = run_backup_first(cfg)
        if not prune_ok:
            print("  [backup] PRUNE DISABLED for this run: without a fresh archive, evicting the "
                  "oldest clips could destroy their only copy. clips/ may exceed "
                  "clips_max_gb_by_source until the next successful backup.")

    cfg.crops_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(cfg.db_path)

    # Build the detector ONCE up front (like backyard_cam.run): this resolves the device -- a real
    # GPU compute-check for 'cuda'/'auto' that fails loud on a wrong-arch torch build (the
    # Blackwell sm_120 trap) -- and downloads the model weights on first use.
    print(f"Loading MegaDetector v6 ({cfg.model_version}) on {cfg.device} ...")
    print("  (first run downloads the model weights from Zenodo -- one time only)")
    try:
        detector = Detector(cfg.model_version, cfg.device, cfg.min_confidence,
                            classes=cfg.detect_classes)
    except CudaUnavailableError as e:
        print(f"\n[CUDA ERROR]\n{e}")
        conn.close()
        return 2
    if detector.device == "cuda":
        print(f"  detector ready on GPU: {detector.device_name}")
    else:
        print("  detector ready on CPU -- slower per frame, but a batch import isn't time-critical.")

    # Seed the duplicate skip-set from what's already been imported for this source (default
    # idempotency, independent of --processed-dir): the ledger (covers files that produced no crop)
    # UNION the DB crop_paths (belt-and-braces if the ledger was deleted, and what covers files
    # imported back when ledger lines were bare basenames). See module docstring.
    skip = read_ledger(cfg.db_path, args.source) | imported_keys(conn, args.source)
    if args.videos:
        skip |= imported_video_keys(conn, args.source)   # the clips-table half of the recovery
    if skip:
        print(f"  {len(skip)} file(s) already imported for source='{args.source}' -- will skip them.")

    try:
        if args.watch:
            rc = watch_folder(folder, detector, conn, cfg, source=args.source,
                              recursive=args.recursive, processed_dir=processed_dir,
                              skip=skip, interval=args.interval)
        else:
            # Watermark BEFORE the pass: every detections row above this id is one THIS run wrote,
            # which is what scopes the static filter to a single card -- i.e. to one camera
            # placement, the only span over which "the same spot" means anything.
            first_new_id = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM detections").fetchone()[0]
            imported, saved, skipped = import_folder(
                folder, detector, conn, cfg, source=args.source, recursive=args.recursive,
                processed_dir=processed_dir, skip=skip)
            # Static false-fires go BEFORE the videos, not after: the clip gate keeps a video whose
            # trigger produced an animal crop, so a grill scoring 'animal' would otherwise buy
            # disk space for footage of nothing happening -- and on a card that is the only copy,
            # that space is spent against real footage.
            dropped_static = 0
            if saved and args.static_filter:
                dropped_static = staticfilter.sweep_batch(
                    conn, cfg, args.source, min_id=first_new_id, iou=args.static_iou,
                    min_count=args.static_min_count,
                    min_span_minutes=args.static_min_span_minutes)
                saved -= dropped_static
            # Videos AFTER the stills: the animal gate reads the detections the stills just wrote.
            v_stored = v_already = v_empty = 0
            if args.videos:
                v_stored, v_already, v_empty = import_videos(
                    folder, conn, cfg, source=args.source, recursive=args.recursive, skip=skip,
                    window_s=args.video_pair_window, require_animal=not args.all_videos,
                    processed_dir=processed_dir)
                if v_stored and prune_ok:
                    clips.prune_clips(cfg, conn)    # honour this source's own rolling budget
            extra = f" (skipped {skipped} already-imported)" if skipped else ""
            print(f"\nDone. Imported {imported} file(s); saved {saved} crop(s) to "
                  f"{cfg.db_path}{extra}.")
            if dropped_static:
                print(f"  Static filter dropped {dropped_static} false-fire detection(s) -- "
                      f"listed in "
                      f"{staticfilter.manifest_path(cfg.db_path, args.source).name}.")
            if args.videos:
                v_extra = f", {v_already} already-imported" if v_already else ""
                print(f"  Clips: stored {v_stored} video(s); skipped {v_empty} with no animal "
                      f"in the trigger{v_extra}.")
            if saved:
                refresh_visits(conn, cfg)
                _print_next_step(saved)
            rc = 0
    finally:
        conn.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
