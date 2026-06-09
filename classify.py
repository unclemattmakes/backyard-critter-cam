"""
Phase 2 -- species classification on the saved crops (fills detections.species).

Zero-shot via BioCLIP 2 (no training): each animal crop is matched against SPECIES_LABELS
below, and the top match + its score are written to detections.species / .species_confidence.
Because it's zero-shot, EDITING THE LABEL LIST IS FREE -- tweak it for your yard and re-run,
no retraining. Re-runnable and resumable: by default only rows with species IS NULL are done.

  python classify.py                 # backfill all unlabeled crops (GPU)
  python classify.py --device cpu    # ... on CPU (slower, but no GPU contention with the rig)
  python classify.py --redo          # re-classify everything (e.g. after editing the labels)
  python classify.py --limit 200     # just the first 200 (quick check)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import config
import db

# --- Your yard's candidate species. Common names work well. Keep it to species you actually
# get (plus a few plausibles); a tighter list gives sharper zero-shot results. This is a
# STARTER list (PNW backyard) -- edit it and re-run. Validated 2026-06-08: BioCLIP 2 nailed
# raccoon / American crow / domestic cat and a dark-eyed junco from this set.
SPECIES_LABELS = [
    # --- Mammals: common in Pacific Northwest / Puget Sound lowland backyards ---
    "raccoon", "Virginia opossum", "eastern gray squirrel", "Douglas squirrel",
    "eastern cottontail", "Townsend's chipmunk", "brown rat", "domestic cat",
    # --- Birds: common Puget Sound backyard / ground feeders (frequency-ranked from WA data:
    #     song sparrow, dark-eyed junco, black-capped chickadee and American crow are the top 4) ---
    "song sparrow", "dark-eyed junco", "golden-crowned sparrow", "white-crowned sparrow",
    "house sparrow", "spotted towhee", "American crow", "Steller's jay", "California scrub-jay",
    "black-capped chickadee", "chestnut-backed chickadee", "house finch", "American goldfinch",
    "American robin", "varied thrush", "northern flicker", "band-tailed pigeon",
    "European starling", "Bewick's wren", "Anna's hummingbird", "bushtit",
]


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 2: BioCLIP 2 species classification on crops.")
    p.add_argument("--device", default="cuda", help="cuda or cpu.")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Only classify crops with detector confidence >= this (usability gate).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="Max crops to process (0 = all).")
    p.add_argument("--redo", action="store_true",
                   help="Re-classify rows that already have a species (e.g. after editing labels).")
    args = p.parse_args()

    conn = db.connect(config.CONFIG.db_path)  # also runs the species_confidence migration
    # Never overwrite a human-confirmed/corrected label, even with --redo.
    where = "detection_class = 'animal' AND confidence >= ? AND COALESCE(species_verified, 0) != 1"
    params = [args.min_confidence]
    if not args.redo:
        where += " AND species IS NULL"
    rows = conn.execute(
        f"SELECT id, crop_path FROM detections WHERE {where} ORDER BY id", params
    ).fetchall()
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("Nothing to classify -- all caught up.")
        return 0
    print(f"Classifying {len(rows)} crop(s) with BioCLIP 2 on {args.device} "
          f"(against {len(SPECIES_LABELS)} candidate species)...")

    from bioclip import CustomLabelsClassifier
    try:
        clf = CustomLabelsClassifier(SPECIES_LABELS, device=args.device)
    except Exception as e:
        if args.device != "cpu":
            print(f"  {args.device} init failed ({e}); falling back to CPU.")
            args.device = "cpu"
            clf = CustomLabelsClassifier(SPECIES_LABELS, device="cpu")
        else:
            raise

    tally = Counter()
    done = 0
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        valid = [(rid, str(config.ROOT / cp.replace("\\", "/"))) for rid, cp in chunk]
        valid = [(rid, pth) for rid, pth in valid if os.path.exists(pth)]
        if not valid:
            continue
        try:
            preds = clf.predict([pth for _, pth in valid])
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and args.device != "cpu":
                print("  GPU out of memory -- switching to CPU for the remainder.")
                import torch
                torch.cuda.empty_cache()
                args.device = "cpu"
                clf = CustomLabelsClassifier(SPECIES_LABELS, device="cpu")
                preds = clf.predict([pth for _, pth in valid])
            else:
                raise

        best: dict[str, tuple[str, float]] = {}
        for d in preds:
            fn, sc = d["file_name"], float(d.get("score", 0.0))
            if fn not in best or sc > best[fn][1]:
                best[fn] = (d["classification"], sc)
        for rid, pth in valid:
            if pth in best:
                label, score = best[pth]
                db.set_species(conn, rid, label, score)
                tally[label] += 1
        conn.commit()
        done += len(valid)
        print(f"  {done}/{len(rows)} classified ...")

    conn.close()
    print("\nDone. Species tally this run:")
    for sp, n in tally.most_common():
        print(f"  {sp:26} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
