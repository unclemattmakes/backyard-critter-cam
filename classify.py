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
import json
import os
import sys
import threading
import time
from collections import Counter

import config
import db
import detector
from clipfilter import NONANIMAL_LABEL

# --- Your yard's candidate species. Common names work well. Keep it to species you actually
# get (plus a few plausibles); a tighter list gives sharper zero-shot results. This is a
# STARTER list (PNW backyard) -- edit it and re-run. Validated 2026-06-08: BioCLIP 2 nailed
# raccoon / American crow / domestic cat and a dark-eyed junco from this set.
_DEFAULT_SPECIES_LABELS = [
    # --- Mammals: common in Pacific Northwest / Puget Sound lowland backyards ---
    "raccoon", "Virginia opossum", "eastern gray squirrel", "Douglas squirrel",
    "eastern cottontail", "Townsend's chipmunk", "brown rat", "domestic cat", "domestic dog",
    # --- Birds: common Puget Sound backyard / ground feeders (frequency-ranked from WA data:
    #     song sparrow, dark-eyed junco, black-capped chickadee and American crow are the top 4) ---
    "song sparrow", "dark-eyed junco", "golden-crowned sparrow", "white-crowned sparrow",
    "house sparrow", "spotted towhee", "American crow", "Steller's jay", "California scrub-jay",
    "black-capped chickadee", "chestnut-backed chickadee", "house finch", "American goldfinch",
    "American robin", "varied thrush", "northern flicker", "band-tailed pigeon",
    "European starling", "Bewick's wren", "Anna's hummingbird", "bushtit",
    # NB: non-animal "decoy" labels (plate of food / food bowl / empty ground) were tried here to
    # absorb MegaDetector false-fires, and were INERT -- BioCLIP 2 is an organism-only model, so
    # its text encoder won't embed non-organism prompts strongly enough to ever out-score a real
    # species. Food false-positives must be filtered another way (detector-confidence gate or a
    # general-CLIP pre-filter), not by adding labels here.
]

# The active candidate set: a per-yard override from config_local.py (cfg.species_labels) if set,
# else the PNW starter list above -- so a friend retunes for their region without editing source.
SPECIES_LABELS = config.CONFIG.species_labels or _DEFAULT_SPECIES_LABELS


def build_classifier(device: str):
    """Construct a BioCLIP CustomLabelsClassifier on the resolved device (GPU when it genuinely
    runs, else CPU; see detector.build_with_fallback). Returns (classifier, device_used)."""
    from bioclip import CustomLabelsClassifier
    return detector.build_with_fallback(
        lambda dev: CustomLabelsClassifier(SPECIES_LABELS, device=dev),
        device, what="species namer")


def build_nonanimal_filter(device: str):
    """Build the general-CLIP non-animal gate (clipfilter.AnimalFilter) from config, or return None
    if it's disabled (cfg.nonanimal_filter) or fails to load. Fail-open by design: if the gate
    can't load, species naming still runs (just without the food/empty-frame filter) rather than
    taking the whole live rig down."""
    cfg = config.CONFIG
    if not getattr(cfg, "nonanimal_filter", False):
        return None
    try:
        from clipfilter import AnimalFilter
        af = AnimalFilter(cfg.nonanimal_model, cfg.nonanimal_pretrained, device,
                          cfg.nonanimal_threshold)
        print(f"[naming] non-animal prefilter ON ({cfg.nonanimal_model}/{cfg.nonanimal_pretrained}"
              f", reject>={cfg.nonanimal_threshold}) on {af.device}.")
        return af
    except Exception as e:
        print(f"[naming] non-animal prefilter unavailable ({e}); naming without it.")
        return None


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


def classify_rows(conn, clf, device: str, rows, batch_size: int, total: int | None = None,
                  afilter=None):
    """Classify (id, crop_path) rows in batches, writing species back. Returns
    (tally, clf, device) -- clf/device may change if a GPU OOM forces a CPU fallback mid-run.

    If `afilter` (a clipfilter.AnimalFilter) is given, each crop is first asked "is this even an
    animal?"; crops it rejects are labelled NONANIMAL_LABEL (source 'clip-filter') and skip BioCLIP
    entirely, so MegaDetector's food / empty-frame false-fires never get forced onto a real
    species the way they used to pile into 'brown rat'."""
    tally = Counter()
    done = 0
    total = total if total is not None else len(rows)
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        valid = [(rid, str(config.ROOT / cp.replace("\\", "/"))) for rid, cp in chunk]
        valid = [(rid, pth) for rid, pth in valid if os.path.exists(pth)]
        if not valid:
            continue
        n_batch = len(valid)   # crops actually processed this batch (animal + non-animal)

        # Stage 0: general-CLIP non-animal gate. Rejected crops are labelled NONANIMAL_LABEL and
        # dropped from the batch, so BioCLIP only ever sees things that are plausibly animals.
        if afilter is not None:
            kept = []
            for (rid, pth), (is_animal, p_non) in zip(valid, afilter.judge([p for _, p in valid])):
                if is_animal:
                    kept.append((rid, pth))
                else:
                    db.set_species(conn, rid, NONANIMAL_LABEL, p_non, source="clip-filter")
                    tally[NONANIMAL_LABEL] += 1
            valid = kept

        if valid:                          # anything left after the gate -> name it with BioCLIP
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
        done += n_batch
        print(f"  {done}/{total} classified ...")
    return tally, clf, device


def _write_naming_status(state: str, **extra) -> None:
    """Write a tiny status file the web dashboard reads, so it can show whether live naming is
    'loading' (model warming up), 'ready' (naming new crops), or 'stopped'. Atomic, best-effort."""
    try:
        data = {"state": state, "ts": time.time(), "pid": os.getpid(), **extra}
        tmp = config.NAMING_STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, config.NAMING_STATUS_FILE)
    except Exception:
        pass


def watch_loop(conn, *, device="cpu", interval=5.0, min_confidence=0.0, batch_size=64,
               stop_event=None, session=None) -> Counter:
    """Name freshly-saved crops by species as they land, until `stop_event` is set (or forever
    if it's None). Builds the BioCLIP classifier once, then polls the DB every `interval`s.

    This is the shared engine for live naming. It runs two ways:
      * standalone   -- `classify.py --watch`, stopped with Ctrl-C; and
      * folded in    -- backyard_cam.py runs it in a background thread, so ONE process / ONE
                        window does detection AND naming, and a single 'q' stops both.
    `session` (a Counter) accumulates the running tally -- pass one in to read it after a stop."""
    stop_event = stop_event if stop_event is not None else threading.Event()
    session = session if session is not None else Counter()
    _write_naming_status("loading", device=device)   # dashboard shows "warming up" until ready
    clf, device = build_classifier(device)
    afilter = build_nonanimal_filter(device)
    print(f"[naming] BioCLIP 2 ready on {device}; naming new crops as they arrive "
          f"(checking every {interval:.0f}s).")
    _write_naming_status("ready", device=device, named=sum(session.values()))
    while not stop_event.is_set():
        rows = fetch_pending(conn, min_confidence, redo=False)
        if rows:
            print(f"[naming] {len(rows)} new crop(s) to name...")
            tally, clf, device = classify_rows(conn, clf, device, rows, batch_size, afilter=afilter)
            session.update(tally)
        _write_naming_status("ready", device=device, named=sum(session.values()))  # heartbeat
        stop_event.wait(interval)   # interruptible sleep -- wakes instantly when stop is set
    _write_naming_status("stopped", device=device, named=sum(session.values()))
    return session


def run_watch(conn, args) -> int:
    """Standalone `--watch`: name new crops beside the live rig until Ctrl-C. The live rig now
    names crops in-process (see watch_loop), so you only need this to run naming on a DIFFERENT
    machine, or to push naming onto the GPU with `--device cuda`."""
    session = Counter()
    stop_event = threading.Event()
    try:
        watch_loop(conn, device=args.device, interval=args.interval,
                   min_confidence=args.min_confidence, batch_size=args.batch_size,
                   stop_event=stop_event, session=session)
    except KeyboardInterrupt:
        print("\n[naming] stopped. Session tally:")
        for sp, n in session.most_common():
            print(f"  {sp:26} {n}")
    finally:
        conn.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 2: BioCLIP 2 species classification on crops.")
    p.add_argument("--device", default=None, choices=["cuda", "cpu", "auto"],
                   help="cuda | cpu | auto. Default: auto for a one-shot run, cpu for --watch "
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
    p.add_argument("--tag", default=None,
                   help="Internal marker the live rig puts on this helper's command line so it "
                        "can find and cleanly stop it (and any subprocess) on shutdown.")
    args = p.parse_args()

    if args.device is None:
        args.device = config.CONFIG.classify_device if args.watch else config.CONFIG.device

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
    afilter = build_nonanimal_filter(args.device)
    tally, clf, args.device = classify_rows(conn, clf, args.device, rows, args.batch_size,
                                            afilter=afilter)

    conn.close()
    print("\nDone. Species tally this run:")
    for sp, n in tally.most_common():
        print(f"  {sp:26} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
