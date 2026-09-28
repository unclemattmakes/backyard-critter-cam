"""
Phase 2 -- species classification on the saved crops (fills detections.species).

Zero-shot via BioCLIP 2 (no training): each animal crop is matched against SPECIES_LABELS
below, and the top match + its score are written to detections.species / .species_confidence.
Because it's zero-shot, EDITING THE LABEL LIST IS FREE -- tweak it for your yard and re-run,
no retraining. Re-runnable and resumable: by default only rows with species IS NULL are done.

Labeling here is what gives VISITS their species (a visit carries the dominant label of its
crops), so whenever this module writes labels it also refreshes the visit ledger -- at the end
of a one-shot run, and in --watch whenever a naming backlog drains. That's what lets a trail-cam
batch import end with LABELED visits with no manual `python visits.py` afterwards.

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
from datetime import datetime, timedelta

import config
import db
import detector
import visits
from clipfilter import NONANIMAL_LABEL

# What species_source records for a crop whose FILE will not decode. It is labelled
# NONANIMAL_LABEL so it leaves the naming queue for good (see _decodes), but under its own source
# so the provenance survives: these are damaged files, not crops the model judged empty, and
# `SELECT * FROM detections WHERE species_source = 'unreadable-crop'` is the audit.
UNREADABLE_SOURCE = "unreadable-crop"

# How many pending crops one --watch poll takes on. The loop used to swallow the entire backlog in
# a single classify_rows call, which froze the status heartbeat (and the dashboard's "ready" pill)
# for as long as that took -- hours, on the 48k backfill.
WATCH_CHUNK = 512

# Refresh the visit ledger at least this often while unstamped detections exist, even if naming
# wrote nothing. See the second trigger in watch_loop for why "even if" is the load-bearing word.
LEDGER_MAX_STALE_S = 1800.0

# --- Your yard's candidate species. Common names work well. Keep it to species you actually
# get (plus a few plausibles); a tighter list gives sharper zero-shot results. This is a
# STARTER list (PNW backyard) -- edit it and re-run. Validated 2026-06-08: BioCLIP 2 nailed
# raccoon / American crow / domestic cat and a dark-eyed junco from this set.
_DEFAULT_SPECIES_LABELS = [
    # --- Mammals: common in Pacific Northwest lowland backyards ---
    "raccoon", "Virginia opossum", "eastern gray squirrel", "Douglas squirrel",
    "eastern cottontail", "Townsend's chipmunk", "brown rat", "domestic cat", "domestic dog",
    # --- Birds: common Pacific Northwest backyard / ground feeders (frequency-ranked from
    #     regional feeder-count data: song sparrow, dark-eyed junco, black-capped chickadee
    #     and American crow are the top 4) ---
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


def count_pending(conn, min_confidence: float = 0.0) -> int:
    """How many crops are still waiting to be named. Cheap (COUNT over the same indexed predicate
    as fetch_pending) and reported in the naming status file, so "is the namer alive" and "is the
    namer getting anywhere" stop being the same question -- they were, and the answer read green
    for two weeks while the backlog grew to 48,275."""
    try:
        return int(conn.execute(
            "SELECT COUNT(*) FROM detections WHERE detection_class = 'animal' "
            "AND confidence >= ? AND COALESCE(species_verified, 0) != 1 AND species IS NULL",
            (min_confidence,)).fetchone()[0])
    except Exception:       # noqa: BLE001 -- a status number is never worth failing a poll over
        return -1


def _unstamped(conn, older_than_minutes: float) -> bool:
    """True if a detection older than `older_than_minutes` still has no visit_id -- the test that
    decides whether the ledger's staleness timer has anything to do.

    The age bound is what makes this mean 'the ledger has fallen behind' rather than 'the camera is
    working'. db.insert_detection never writes visit_id, so a live rig has unstamped rows within
    seconds of any rebuild; gating on their bare existence would rebuild the whole ledger every
    30 minutes forever. A row older than the visit gap belongs to a visit that has already CLOSED
    and still has not been stamped, which is the real condition."""
    try:
        cutoff = (datetime.now().astimezone()
                  - timedelta(minutes=float(older_than_minutes))).isoformat()
        return conn.execute(
            "SELECT 1 FROM detections WHERE visit_id IS NULL AND timestamp < ? LIMIT 1",
            (cutoff,)).fetchone() is not None
    except Exception:       # noqa: BLE001
        return False


def _decodes(path: str) -> bool:
    """True if `path` actually opens as an image. Mirrors embed.py's _load_image guard, which
    classify.py lacked -- and that asymmetry cost two weeks of naming.

    A file that EXISTS is not a file that DECODES. When the host hard-hangs it loses the OS write
    cache, and NTFS leaves the crops it was mid-write on at their full allocated length filled with
    NUL bytes. Twenty such files survived the 2026-08-23 freeze and one more the 2026-08-24 one.
    BioCLIP opens paths itself, deep inside predict(), and raises PIL.UnidentifiedImageError --
    which subclasses OSError, so the GPU-OOM guard in classify_rows never caught it. Because
    fetch_pending orders by id, those rows sat in the FIRST batch of every run, so the helper died
    before its first commit on every single start: 48,275 crops went unnamed and every downstream
    stage that gates on species went with them.

    Image.verify() reads only enough to validate the container, so this is cheap next to the
    inference it protects."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:       # noqa: BLE001 -- any failure to decode means "do not hand this to the model"
        return False


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
        valid = [(rid, str(db.crop_abspath(cp))) for rid, cp in chunk]
        valid = [(rid, pth) for rid, pth in valid if os.path.exists(pth)]
        if not valid:
            continue

        # Run ALL the slow inference first and stash the labels in memory -- do NOT touch the DB
        # yet. Writing here would open a transaction and hold SQLite's WAL write lock across the
        # (CPU, multi-second-per-image) CLIP/BioCLIP passes, which locks out the live capture
        # thread long enough to crash it ("database is locked"). We instead commit everything in
        # one quick burst at the end, so the write lock is held for milliseconds, not minutes.
        pending: list = []                 # (id, label, score, source) -- source None = bioclip default

        # Screen out files that will not decode BEFORE the model ever sees them, and LABEL them so
        # they leave the queue rather than being retried (and re-crashing) forever -- see _decodes.
        # Marking beats merely skipping: a skipped row stays pending, so the watcher would re-select
        # it every poll for the life of the rig.
        readable = []
        for rid, pth in valid:
            if _decodes(pth):
                readable.append((rid, pth))
            else:
                pending.append((rid, NONANIMAL_LABEL, 0.0, UNREADABLE_SOURCE))
                tally[NONANIMAL_LABEL] += 1
                print(f"  unreadable crop, marked '{UNREADABLE_SOURCE}': {pth}")
        n_batch = len(valid)   # crops actually processed this batch (animal + non-animal + damaged)
        valid = readable

        # Stage 0: general-CLIP non-animal gate. Rejected crops are labelled NONANIMAL_LABEL and
        # dropped from the batch, so BioCLIP only ever sees things that are plausibly animals.
        if afilter is not None:
            kept = []
            for (rid, pth), (is_animal, p_non) in zip(valid, afilter.judge([p for _, p in valid])):
                if is_animal:
                    kept.append((rid, pth))
                else:
                    pending.append((rid, NONANIMAL_LABEL, p_non, "clip-filter"))
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
            except OSError as e:
                # _decodes screens the known failure, but the model opens each path a SECOND time
                # inside predict(), so a race (a prune, a crop still being written) can still land
                # here. Degrade to one-at-a-time: a bad path then costs its own row instead of the
                # whole helper. Rows that fail stay pending and are re-screened on the next poll.
                print(f"  batch failed to decode ({e}) -- retrying one crop at a time.")
                preds = []
                for _rid, pth in valid:
                    try:
                        preds.extend(clf.predict([pth]))
                    except OSError as e2:      # noqa: PERF203 -- the slow path, only after a failure
                        print(f"  unreadable crop, skipped: {pth} ({e2})")

            best: dict[str, tuple[str, float]] = {}
            for d in preds:
                fn, sc = d["file_name"], float(d.get("score", 0.0))
                if fn not in best or sc > best[fn][1]:
                    best[fn] = (d["classification"], sc)
            for rid, pth in valid:
                if pth in best:
                    label, score = best[pth]
                    pending.append((rid, label, score, None))
                    tally[label] += 1

        # Take the write lock only for this quick burst of UPDATEs, then release it with commit().
        for rid, label, score, source in pending:
            if source is None:
                db.set_species(conn, rid, label, score)
            else:
                db.set_species(conn, rid, label, score, source=source)
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
    _write_naming_status("ready", device=device, named=sum(session.values()), backlog=0)
    ledger_dirty = False   # labels written that the visit ledger hasn't folded in yet
    last_refresh = time.monotonic()
    try:
        while not stop_event.is_set():
            try:
                # A BITE, not the whole backlog. fetch_pending used to return every pending row and
                # classify_rows chewed the lot before returning, so on a 48k backfill the heartbeat
                # below did not run for hours and the dashboard showed a frozen "ready". Taking
                # WATCH_CHUNK at a time keeps the heartbeat honest and lets a stop land promptly.
                rows = fetch_pending(conn, min_confidence, redo=False, limit=WATCH_CHUNK)
                if rows:
                    print(f"[naming] {len(rows)} new crop(s) to name...")
                    tally, clf, device = classify_rows(conn, clf, device, rows, batch_size,
                                                       afilter=afilter)
                    session.update(tally)
                    if tally:
                        ledger_dirty = True
                stale = (time.monotonic() - last_refresh) >= LEDGER_MAX_STALE_S
                if stale:
                    # Advance the clock whenever the CHECK runs, not only when a rebuild follows.
                    # Assigning it solely inside the body let `stale` latch True forever on a
                    # ledger with nothing to do, and _unstamped is an unindexed scan of a 269k-row
                    # table -- so a quiet rig would run one every 5 seconds for the life of the
                    # process instead of one every 30 minutes.
                    last_refresh = time.monotonic()
                if (not rows and ledger_dirty) or (stale and _unstamped(conn, config.CONFIG.visit_gap_minutes)):
                    # Two triggers, deliberately. The first is the original trailing edge: a quiet
                    # poll after a naming burst means the backlog is drained, so fold the fresh
                    # labels into the ledger (a visit carries its crops' dominant species) -- that
                    # is what lets a trail-cam import end with LABELED visits by itself.
                    #
                    # The second exists because the first is not enough. ledger_dirty is only ever
                    # set when naming WRITES something, so a namer that produces nothing leaves it
                    # False forever and the ledger simply stops -- which is exactly what happened
                    # for the nine days after 2026-08-28: 37,703 detections with no visit_id and
                    # nothing downstream able to see them. build_visits handles species-less crops
                    # fine (it writes species NULL), so the ledger must not depend on the namer.
                    visits.refresh(conn, config.CONFIG.visit_gap_minutes)
                    ledger_dirty = False
                    last_refresh = time.monotonic()
                # The heartbeat carries the BACKLOG, not just liveness. "ready" on its own said
                # nothing about whether work was happening -- it read green for eight days while
                # named stayed 0 -- so rigwatch.py can now alarm on a backlog that never falls.
                _write_naming_status("ready", device=device, named=sum(session.values()),
                                     backlog=count_pending(conn, min_confidence))
            except Exception as e:   # noqa: BLE001 -- a poll must never be able to kill the helper
                # The whole reason this loop exists is to run unattended for weeks. Anything that
                # escapes one poll (a locked DB, a vanished crop, a model hiccup) is logged and
                # retried on the next one; only stop_event ends this loop.
                print(f"[naming] poll failed, continuing: {type(e).__name__}: {e}")
                _write_naming_status("error", device=device, named=sum(session.values()),
                                     detail=f"{type(e).__name__}: {e}")
            stop_event.wait(interval)   # interruptible sleep -- wakes instantly when stop is set
    finally:
        # Stopped mid-burst (standalone --watch Ctrl-C during a backlog): don't strand the labels
        # already written -- best-effort, so shutdown never gets slower than one quick rebuild.
        if ledger_dirty:
            visits.refresh(conn, config.CONFIG.visit_gap_minutes)
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

    if tally:
        # New labels change what visits would say (a visit carries its crops' dominant species),
        # so end the run by folding them into the visit ledger. This is what completes the
        # trail-cam pipeline (import_trailcam.py -> classify.py) -- and the --redo path after a
        # label-list edit -- without a manual `python visits.py`.
        visits.refresh(conn, config.CONFIG.visit_gap_minutes)
    conn.close()
    print("\nDone. Species tally this run:")
    for sp, n in tally.most_common():
        print(f"  {sp:26} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
