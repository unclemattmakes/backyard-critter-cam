# Runbook: importing a trail-cam SD card

The running order and the traps for a batch import (`import_trailcam.py`) — the things you
only learn by having done it wrong. The importer's module docstring is the authority on *how
it works* (idempotency keys, the video pairing gate, EXIF vs mtime); this is the operational
sequence that keeps a prune from eating footage that exists nowhere else. It generalizes to
any card-based camera; the numbers cited are this project's TC02 history, kept as calibration.

## The one rule

**The card is the only copy, and it gets formatted every cycle.**

This workflow formats the card in-camera each cycle and leaves hybrid video ON on purpose
(video is re-ID and behaviour data). So `clips/<source>/` is the only home that footage will
ever have, and the clip pruner does not *trim* it — it **destroys** it. Every check below
exists because of that one asymmetry. The live rig can record again tomorrow; the card cannot.

## Phase 1: orient before you import anything

- **Find the card without assumptions.** Drive letter and DCIM subfolder both change between
  cycles (this camera has used `100MEDIA` and `100HUNTI`). List removable drives, then glob
  `<drive>/DCIM/*`.
- **Size up the cycle**: count files by extension, total the video bytes, read the mtime range
  (first trigger → last). The budget check needs these.
- **Read the import ledger's tail** — `backyard.db.imported-<source>.txt`, one
  `basename|capture-second` per line — to see where the previous cycle stopped. There is a
  *second* sidecar next to the DB, `backyard.db.static-dropped-<source>.txt`: the record of
  every row a furniture sweep deleted (see Phase 4). It is the only route back to a bad call.
- **Don't panic when numbering goes backwards.** A formatted card restarts at `IMAG0001`; the
  ledger keys on basename **plus capture second**, so recycled names import cleanly. (Keyed on
  name alone — as before 2026-07-19 — 235 of 558 real photos would have been silently skipped.)
- **Check the power before you trust the window.** A card that stopped early stopped for a
  reason, and on a solar/battery camera it is usually not the card. Read the trigger histogram
  by hour and look for the tell: **video drops out before stills do** (the IR illuminator is the
  heavy load and video holds it longest), then everything stops, then it resumes in daylight.
  This camera burns its status bar into the frame — crop the bottom strip of a still and read
  the battery percentage directly. On 2026-08-11 it read 65% at deployment, **<5%** at 23:00,
  and the camera was dark until 12:44 the next day: a whole night lost, and the card still had
  4.29 GB free. The card is rarely the limit; the battery is.

## Phase 2: check the clip budget BEFORE you import

A prune fires at the *end* of the import, after the videos land. Work out whether it will:

- The budget: `cfg.clips_max_gb_by_source[<source>]` — **read the config_local override, not
  the config.py default** (this project runs 60 GB where the default ships 15).
- Current usage: total the `clips/<source>/` tree.
- Incoming: the card's video bytes × the keep rate. **Only clips whose trigger produced an
  animal crop are kept**, and the rate swings hard — measured at 72%, 59%, 52% and 33% across
  four cycles. Budget against the raw bytes; don't plan on 70%.

If current + incoming lands anywhere near the budget, **stop and raise the budget first**
rather than letting the prune choose. A 15 GB budget was nearly fatal once: one dump landed at
12.74/15 GB, meaning the *next* dump would have evicted the oldest days mid-import.

## Phase 3: back up first, always

Run with **`--backup-first`**. It archives to the backup destination before a single clip can
be pruned, and if the archive fails it disables the prune for that run rather than proceeding.

Not ceremony: `backup.py` skips today's folder and runs weekly, so the newest trail-cam days
routinely sit unarchived. On 2026-08-02 the newest archive was three days behind a 2.9 GB day
that had no second copy anywhere; `--backup-first` swept it up seconds before the import.

## Phase 4: run it

```bash
.venv/Scripts/python import_trailcam.py "<drive>/DCIM/<folder>" --backup-first
```

- Point it **straight at the media folder on the card**; `--recursive` is only for a nested dump.
- It is long (backup + a detector pass over 1,500–2,500 stills + copying several GB of video);
  run it in the background and tail the output.
- **Leave the card mounted until it finishes** — stills and videos import in two passes, and
  the video pass needs the detections the still pass just wrote.
- **It deletes rows, on purpose.** Between the two passes `staticfilter.py` sweeps the batch and
  drops furniture — a spot that either held one box for 60+ minutes, or whose crops never change
  across the cluster's whole life. The crop JPEGs stay on disk and every dropped row is appended
  to the manifest from Phase 1, so the sweep is auditable, but the `crops saved` number the
  importer prints is **net** of it. `--no-static-filter` keeps everything.

## Phase 5: species labels land after the import

Rows arrive with `species` NULL — by design, same as live crops. The visit ledger refreshes
immediately; then either the rig's live naming helper labels them within moments (and
refreshes the ledger again), or you run `python classify.py`, which does that second refresh
itself. **No manual `visits.py` step either way** — the importer prints which path applies.

**Whatever furniture survived Phase 4 gets a species here**, confidently and wrongly. The
classifier is doing its job on a barbecue. On this rig a dark upright object reliably comes back
as `brown rat` (272 of them off one bin on 2026-08-11; 306 off a grill and chimney starter in an
earlier cycle) and a bright one as `raccoon`. Treat a species tally as *unaudited* until Phase 6.

## Phase 6: verify before you report — and before you format

Check the importer's own summary (imported / crops / clips kept / skipped / already-imported /
static dropped), then confirm rows actually landed: new `detections` and `clips` for the source
in the cycle's window, and new visits. Compare against previous cycles **per day, not per
cycle** — the first and last day of a dump are always partial. This yard's full-day crop counts
have run 203–1,852; an order-of-magnitude miss means something silently broke, a 2× swing is
just weather.

**Then audit the furniture the sweep missed**, because the sweep is not sufficient on its own.
The cheap tell is a species tally that has gone strange: a rat explosion, a songbird species
list longer than the yard has songbirds, a nocturnal mammal in daylight. Cluster the cycle's
boxes and look at the repeat offenders — `python staticfilter.py --since <start> --explain`
prints every spot with its rigidity score, and anything scoring low that the sweep still kept
is worth opening. On the 2026-08-12 cycle this pass found 669 furniture detections across 10
spots (a bin, a lamp, a paver edge) that the automatic sweep could not reach.

Report the window the cycle covers, not just totals — and derive it from the **detections**,
not from when you pulled the card. If the camera died overnight those are hours apart, and the
card's last file timestamp will confidently report a window the data does not cover.

**Only after verifying is the card safe to format and put back in the camera.**

## What not to do

- **Never import from a renamed or copied folder.** A staging copy once came in as 323
  duplicate rows that had to be deduped by hand. Import in place, off the card.
- **Never format the card before the import is verified.** There is no second copy.
- **Never skip `--backup-first`** because the budget looks roomy — the weekly-backup lag is a
  separate reason to archive.
- **Never hardcode the drive letter or the DCIM folder name.** Both have already changed.
- **Do not silently drop part of the card.** If videos were skipped or the run errored partway,
  say exactly what did not import.
- **Never lower the span threshold to catch furniture that stopped.** It is the obvious response
  to a cycle like 2026-08-11 — the bin fired for 29 minutes against a 60-minute rule — and it is
  measurably impossible on this corpus: in that same cycle real raccoons foraged one spot for 23
  and 38 minutes, and were verified by eye. There is no span that separates them. The pixel test
  (`--rigid-maxpair`) is the safe half of that fix, and it only reaches rigid objects; the rest
  is a human looking at crops.
- **Do not delete a cluster you have not looked at.** A manifest and the surviving JPEGs make a
  bad sweep recoverable, but only if someone notices. Species labels are not evidence: the lamp
  that fired 99 times was labelled `raccoon:78`.
