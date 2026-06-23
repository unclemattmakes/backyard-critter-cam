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

  python import_trailcam.py D:\DCIM\100MEDIA                 # ingest one SD-card dump (GPU)
  python import_trailcam.py D:\dump --recursive              # walk subfolders too
  python import_trailcam.py D:\dump --device cpu             # no GPU (slower; fine for batch)
  python import_trailcam.py D:\dump --processed-dir D:\done  # move each file after import
  python import_trailcam.py D:\drop --watch                  # poll a drop folder forever

IDEMPOTENCY (re-runs must not double-import). Three layers, all boring on purpose:
  * ledger: a plain-text sidecar next to the DB (backyard.db.imported-<source>.txt) records every
    basename imported for this `source`, one per line, appended as files land. At startup we read
    it back into a skip-set, so a plain re-run over the same folder is a no-op -- even for images
    that produced NO crop (a frame with no animal still gets logged, so it isn't re-detected).
    This is the primary, always-on default; it survives whether or not a crop was saved.
  * crop_paths: the original filename is also encoded into every crop's name, so the skip-set is
    rebuilt from the DB's crop_paths too -- a belt-and-braces recovery if the ledger is deleted
    (crops that exist won't be re-imported).
  * --processed-dir: after a file imports cleanly, MOVE it there, so a re-run never even sees it.
    The recommended way to run an ongoing drop folder (and what --watch leans on to drain it).

  (Idempotency keys on the BASENAME, scoped to `source`. Two different folders holding a same-
  named file is the one case the ledger/crop-path layers can't tell apart -- use --processed-dir,
  or unique filenames, if that matters to you.)

Robust by design (PLAN.md "boring and robust"): an unreadable/corrupt image is warned about and
skipped; one bad file never aborts the batch.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2

import config
import db
import visits
from config import CONFIG
# Reuse the live rig's exact crop + path logic so trail-cam crops are byte-for-byte the same
# convention as glass-door crops (same padded/clamped box, same filename, same crops_dir/<date>/
# layout). Importing them keeps a single source of truth -- if save_crop changes, both rigs move.
from backyard_cam import _rel, save_crop
from detector import CudaUnavailableError, Detection, Detector

# Image extensions we ingest. Trail cams write JPEG; PNG is accepted for completeness / exports.
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

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
# recoverable from crop_path -- that's what powers the default duplicate skip-set (see module
# docstring). Kept filesystem-safe and unlikely to collide with a real stem.
SRC_TAG = "src-"


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


def imported_basenames(conn, source: str) -> set[str]:
    """Original SD-card filenames already imported for `source`, recovered from the crop_paths in
    the DB (each crop name embeds 'src-<stem>'). This is the default skip-set that makes a plain
    re-run idempotent without --processed-dir. Empty set on a fresh DB."""
    out: set[str] = set()
    for (cp,) in conn.execute(
        "SELECT crop_path FROM detections WHERE source = ?", (source,)
    ):
        if not cp:
            continue
        stem = Path(cp).name
        i = stem.find(SRC_TAG)
        if i != -1:
            rest = stem[i + len(SRC_TAG):]
            # crop name = 'src-<orig-stem>_<idx>_<class>_<conf>.jpg'. save_crop appends exactly THREE
            # underscore-fields (idx, class, conf), so strip those from the RIGHT to recover the full
            # original stem -- a left split mangles any source name that itself contains an underscore
            # (e.g. IMG_0042 -> 'IMG'), which then never matches the file again on a DB-recovery re-run.
            out.add(rest.rsplit("_", 3)[0])
    return out


def ledger_path(db_path: Path, source: str) -> Path:
    """The plain-text import ledger that sits beside the DB, scoped to one `source`. Filesystem-
    safe: any non-alphanumeric char in the source name becomes '_'."""
    safe = "".join(c if c.isalnum() else "_" for c in source)
    return Path(db_path).with_name(Path(db_path).name + f".imported-{safe}.txt")


def read_ledger(db_path: Path, source: str) -> set[str]:
    """Basenames already imported for `source`, per the sidecar ledger. The always-on default
    skip-set; survives even for frames that produced no crop. Empty set if the ledger is absent."""
    p = ledger_path(db_path, source)
    if not p.exists():
        return set()
    try:
        return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    except Exception:
        return set()


def append_ledger(db_path: Path, source: str, basename: str) -> None:
    """Record one imported basename in the sidecar ledger (append + flush, so a crash mid-batch
    still leaves earlier files marked). Best-effort: a write failure never fails the import."""
    try:
        with open(ledger_path(db_path, source), "a", encoding="utf-8") as f:
            f.write(basename + "\n")
    except Exception as e:
        print(f"  [warn] could not update import ledger for {basename}: {e}")


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
    base_stamp = dt.strftime("%Y-%m-%dT%H-%M-%S-") + f"{dt.microsecond // 1000:03d}"
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


def refresh_visits(conn, cfg: config.Config) -> None:
    """Re-collapse detections into visit events after an import lands, so the dashboard's
    Behaviour tab includes the trail-cam visits without a manual `python visits.py`. Best-effort
    and subsecond at this scale; an error never fails an import that already succeeded."""
    try:
        conn.row_factory = sqlite3.Row
        visits.build_visits(conn, cfg.visit_gap_minutes, verbose=False)
        print("  visit ledger refreshed.")
    except Exception as e:
        print(f"  [visits] could not refresh visit events (run `python visits.py`): {e}")


def import_folder(folder: Path, detector: Detector, conn, cfg: config.Config, *,
                  source: str, recursive: bool, processed_dir: Path | None,
                  skip: set[str]) -> tuple[int, int, int]:
    """Import every image in `folder` once. `skip` is the set of original basenames already
    imported (default idempotency); files moved to --processed-dir won't reappear anyway. Updates
    `skip` in place as it goes so a single pass never imports the same name twice. Returns
    (files_imported, crops_saved, files_skipped)."""
    images = list_images(folder, recursive)
    if not images:
        print(f"No images ({'/'.join(sorted(IMAGE_EXTS))}) found in {folder}"
              f"{' (recursive)' if recursive else ''}.")
        return 0, 0, 0

    print(f"Found {len(images)} image(s) in {folder}"
          f"{' (recursive)' if recursive else ''}. Importing as source='{source}' ...")
    imported = saved_total = skipped = 0
    for path in images:
        # The ledger keys on the full filename; the DB-recovery fallback keys on the stem (the
        # extension isn't stored in the crop name) -- accept either so both skip-sources match.
        if path.name in skip or path.stem in skip:
            skipped += 1
            continue
        n_reported, n_saved = ingest_file(path, detector, conn, cfg, source)
        if n_reported < 0:
            continue  # unreadable/corrupt -- already warned, leave it in place to inspect
        skip.add(path.name)
        append_ledger(cfg.db_path, source, path.name)  # mark imported (even if 0 crops) -> idempotent
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
                   help="Inference device: cuda (default, like the live rig; requires an NVIDIA "
                        "GPU) | cpu (no GPU, slower -- fine for an overnight batch) | auto.")
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
    # UNION the DB crop_paths (belt-and-braces if the ledger was deleted). See module docstring.
    skip = read_ledger(cfg.db_path, args.source) | imported_basenames(conn, args.source)
    if skip:
        print(f"  {len(skip)} file(s) already imported for source='{args.source}' -- will skip them.")

    try:
        if args.watch:
            rc = watch_folder(folder, detector, conn, cfg, source=args.source,
                              recursive=args.recursive, processed_dir=processed_dir,
                              skip=skip, interval=args.interval)
        else:
            imported, saved, skipped = import_folder(
                folder, detector, conn, cfg, source=args.source, recursive=args.recursive,
                processed_dir=processed_dir, skip=skip)
            extra = f" (skipped {skipped} already-imported)" if skipped else ""
            print(f"\nDone. Imported {imported} file(s); saved {saved} crop(s) to "
                  f"{cfg.db_path}{extra}.")
            if saved:
                refresh_visits(conn, cfg)
            rc = 0
    finally:
        conn.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
