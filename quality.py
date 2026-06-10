"""
Per-crop SHOT QUALITY -- "how good a photo is this?" -- so the dashboard can lead a visit with its
cutest, sharpest frame instead of merely the most confident detection. (This is what turns "show me
the best pic" into something real without a face detector -- see PLAN.md / the dashboard reel.)

The score is IMAGE-derived and pure cv2/numpy:
  * SHARPNESS -- variance of the Laplacian. The dominant signal: a crisp, in-focus crop scores high;
    the soft, flared through-glass night crops (the whole reason the crops looked bad) score low.
  * night EYESHINE boost -- in a DARK crop, a few very-bright specular pixels are almost always eyes
    catching the light, i.e. the animal looking toward the glass. A small reward for "facing you".
Crop SIZE and CENTEREDNESS are deliberately NOT folded in here -- the bounding box is already in the
DB, so stats.py blends those in cheaply (and can be retuned) without re-reading any image.

The live rig scores each crop as it's saved (backyard_cam.save_crop); this module's CLI backfills
crops captured before the feature existed:

    python quality.py                 # score every crop missing a quality value
    python quality.py --redo          # rescore everything
    python quality.py --db other.db   # point at a specific database

stats.py only READS the stored number, so it stays cv2/torch-free.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import cv2

import config
import db


def score_crop(crop_bgr) -> float:
    """Image-derived quality of one crop (>= 0; higher = sharper / more 'a good shot'). Returns
    Laplacian-variance sharpness, boosted up to +50% by night eyeshine. 0.0 for an empty crop."""
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    boost = 1.0
    if gray.mean() < 90:                       # dark crop -> bright specks are eyes, not glare
        bright_frac = float((gray > 240).mean())
        boost = 1.0 + min(bright_frac * 60.0, 0.5)
    return round(sharp * boost, 2)


def score_file(path: Path) -> float | None:
    """Score a crop on disk; None if it can't be read (missing / corrupt file)."""
    img = cv2.imread(str(path))
    return None if img is None else score_crop(img)


def backfill(cfg: config.Config, *, redo: bool = False, limit: int = 0, batch: int = 200) -> int:
    """Score crops that don't yet have a quality value (or all of them with --redo). Returns the
    number scored. Resumable: stops and resumes cleanly because only NULLs are selected by default."""
    conn = db.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row              # read selected columns by name
    try:
        where = "crop_path IS NOT NULL" + ("" if redo else " AND crop_quality IS NULL")
        rows = conn.execute(
            f"SELECT id, crop_path FROM detections WHERE {where} ORDER BY id"
            + (f" LIMIT {int(limit)}" if limit else "")
        ).fetchall()
        total = len(rows)
        if not total:
            print("All crops already scored. (Use --redo to rescore.)")
            return 0
        print(f"Scoring {total} crop(s){' (rescore)' if redo else ''} ...")
        done, missing, pending = 0, 0, []
        for r in rows:
            q = score_file(config.ROOT / r["crop_path"])
            if q is None:
                missing += 1
                continue
            pending.append((r["id"], q))
            if len(pending) >= batch:
                db.set_crop_quality_bulk(conn, pending)
                done += len(pending)
                pending = []
                print(f"  {done}/{total} scored ...", end="\r")
        if pending:
            db.set_crop_quality_bulk(conn, pending)
            done += len(pending)
        print(f"\nDone. Scored {done} crop(s)"
              + (f"; skipped {missing} unreadable file(s)." if missing else "."))
        return done
    finally:
        conn.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Score crop shot-quality (sharpness + night eyeshine).")
    p.add_argument("--db", default=str(config.CONFIG.db_path), help="Database to backfill.")
    p.add_argument("--redo", action="store_true", help="Rescore crops that already have a value.")
    p.add_argument("--limit", type=int, default=0, help="Only score the first N (0 = all).")
    args = p.parse_args()
    import dataclasses
    cfg = dataclasses.replace(config.CONFIG, db_path=Path(args.db))
    backfill(cfg, redo=args.redo, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
