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

  python classify.py --watch         # run forever beside the live rig: classify new crops as
                                     # they land, so the dashboard's "Most Recent Visitor"
                                     # shows a species (not "animal") within a few seconds.
                                     # Defaults to CPU so it never fights the live detector for
                                     # the GPU; pass --device cuda to override.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
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


def build_classifier(device: str):
    """Construct a BioCLIP CustomLabelsClassifier, falling back to CPU if GPU init fails.
    Returns (classifier, device_actually_used)."""
    from bioclip import CustomLabelsClassifier
    try:
        return CustomLabelsClassifier(SPECIES_LABELS, device=device), device
    except Exception as e:
        if device != "cpu":
            print(f"  {device} init failed ({e}); falling back to CPU.")
            return CustomLabelsClassifier(SPECIES_LABELS, device="cpu"), "cpu"
        raise


def fetch_pending(conn, min_confidence: float, redo: bool, limit: int = 0):
    """Rows still needing a species: animal crops above the usability gate, never human-verified.
    Unless --redo, only rows with species IS NULL (so it's resumable and watch-safe)."""
    where = "detection_class = 'animal' AND confidence >= ? AND COALESCE(species_verified, 0) != 1"
    params = [min_confidence]
    if not redo:
        where += " AND species IS NULL"
    rows = conn.execute(
        f"SELECT id, crop_path FROM detections WHERE {where} ORDER BY id", params
    ).fetchall()
    return rows[:limit] if limit else rows


def classify_rows(conn, clf, device: str, rows, batch_size: int, total: int | None = None):
    """Classify (id, crop_path) rows in batches, writing species back. Returns
    (tally, clf, device) -- clf/device may change if a GPU OOM forces a CPU fallback mid-run."""
    tally = Counter()
    done = 0
    total = total if total is not None else len(rows)
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        valid = [(rid, str(config.ROOT / cp.replace("\\", "/"))) for rid, cp in chunk]
        valid = [(rid, pth) for rid, pth in valid if os.path.exists(pth)]
        if not valid:
            continue
        try:
            preds = clf.predict([pth for _, pth in valid])
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device != "cpu":
                print("  GPU out of memory -- switching to CPU for the remainder.")
                import torch
                from bioclip import CustomLabelsClassifier
                torch.cuda.empty_cache()
                device = "cpu"
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
        print(f"  {done}/{total} classified ...")
    return tally, clf, device


def run_watch(conn, args) -> int:
    """Poll forever for freshly-saved crops and classify them as they land, so the live
    dashboard names the most-recent visitor instead of showing the coarse 'animal'."""
    clf, device = build_classifier(args.device)
    print(f"[watch] BioCLIP 2 ready on {device}; polling every {args.interval:.0f}s for new "
          f"crops to classify (Ctrl-C to stop).")
    session = Counter()
    try:
        while True:
            rows = fetch_pending(conn, args.min_confidence, redo=False)
            if rows:
                print(f"[watch] {len(rows)} new crop(s) to classify...")
                tally, clf, device = classify_rows(conn, clf, device, rows, args.batch_size)
                session.update(tally)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[watch] stopped. Session tally:")
        for sp, n in session.most_common():
            print(f"  {sp:26} {n}")
    finally:
        conn.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 2: BioCLIP 2 species classification on crops.")
    p.add_argument("--device", default=None,
                   help="cuda or cpu. Default: cuda for a one-shot run, cpu for --watch "
                        "(so the poller never fights the live detector for the GPU).")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Only classify crops with detector confidence >= this (usability gate).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="Max crops to process (0 = all).")
    p.add_argument("--redo", action="store_true",
                   help="Re-classify rows that already have a species (e.g. after editing labels).")
    p.add_argument("--watch", action="store_true",
                   help="Run continuously: classify new unclassified crops as they arrive "
                        "(load the model once, poll the DB). For running beside the live rig.")
    p.add_argument("--interval", type=float, default=5.0,
                   help="Seconds between polls in --watch mode (default 5).")
    args = p.parse_args()

    if args.device is None:
        args.device = "cpu" if args.watch else "cuda"

    conn = db.connect(config.CONFIG.db_path)  # also runs the species_confidence migration

    if args.watch:
        return run_watch(conn, args)

    rows = fetch_pending(conn, args.min_confidence, args.redo, args.limit)
    if not rows:
        print("Nothing to classify -- all caught up.")
        conn.close()
        return 0
    print(f"Classifying {len(rows)} crop(s) with BioCLIP 2 on {args.device} "
          f"(against {len(SPECIES_LABELS)} candidate species)...")

    clf, args.device = build_classifier(args.device)
    tally, clf, args.device = classify_rows(conn, clf, args.device, rows, args.batch_size)

    conn.close()
    print("\nDone. Species tally this run:")
    for sp, n in tally.most_common():
        print(f"  {sp:26} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
