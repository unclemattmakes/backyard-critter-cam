# Deferred work — the backlog the 2026-08-08 evaluation left behind

**This one is LIVING, not a dated snapshot** (unlike its neighbours in this folder). It is the
honest remainder of the whole-project evaluation of 2026-08-08: everything that review surfaced
and the same-day implementation pass did **not** build, with the reason it was left, the plan as
it stands after adversarial review, and the first step that would move it.

Keep it honest the way the rest of this project stays honest: when an item ships, delete it and
let the code speak; when an item is **measured dead**, do not delete it — move it to
[Killed, with reasons](#killed-with-reasons) so nobody spends a week re-discovering the same
negative. Numbers here come from the evaluation, from
[the identity eval](identity-eval-2026-08-05.md), and from measurements run against the live
database on 2026-08-08 and 08-09 while writing this — those say so where they appear, and two of
them killed a design that was on this list a day earlier.

What the same-day pass *did* ship (so nothing below re-proposes it): group-scope for family
labels, the nightly eval regression gate, the surprising-species flag, the sun-anchored arrival
clock, the Seasons tab, yard politics, the moon chart, coarse behaviour tags, the coverage
(effort) ledger, the operator/viewer split with `labeled_by` attribution, `--serve-only`, the
weights mirror, the label ledger, the CSV export, the `STATUS.txt` heartbeat, the accessibility
and progressive-loading pass, and the adoption rewrite of the README. See commits
`eebdcff`…`d9b53d9`. Several of those shipped as a **first half** — [§6](#6-half-built-the-parts-of-the-2026-08-08-burst-that-stop-short)
is the honest accounting of where they stop.

---

## 1. Waiting on a human, not on code

These cannot be written. They are the operator's own eyes and decisions, and several of the code
items below are gated on them.

### 1.1 The blind Notch audit — the largest label correction available
`Notch` returned 2026-08-06 after five weeks away, and `identity_references` now holds three
**certified** photographs of him. Both facts postdate the eval that left the Notch↔Pedro question
open, and **46–53 templates ride on the answer**. The suspicion on record is that the two were
mislabelled for each other early on; date alone predicts which name a visit carries at 97.8%,
which is exactly what a systematic swap looks like.

**Protocol (blind, or it proves nothing):** run `traits.head_review_panel` over the pre-06-30
visits labelled `Notch`, compare against the reference photographs with the model's suggestion
**hidden**, and record the verdict per visit. The ear notch is the one era-invariant marker this
cast has; it is the only evidence that can settle a question appearance embeddings cannot.

### 1.2 The D1 un-blend session — the cheapest template growth on the board
The still-tracklet splitter shipped 2026-07-31, is validated at corpus scale (co-present raccoon
pairs sit at median 0.19 cosine vs 0.72 frame-to-frame for one animal), and has produced
**zero** human labels: `clip_tracks.individual_source` is still 100% `'overlap'` (399 auto rows).

The D1 pass needs no new identification — only "which of these separated clusters is the animal
you already named" — across the **22 already-named multi-animal visits**. It takes CutiePie from
3 usable templates to 6 and Elliot from 4 to 6, which is the difference between "below the
5-template auto floor" and "findable".

### 1.3 Re-sweep and apply the auto-assign operating point
`config_local.py` still carries **0.76 / 0.12**, the point the eval replayed to *zero*
assignments (60 candidates died on the margin). The validated **0.88 / 0.02** (16 visits named,
0 wrong observed, ~15% rule-of-three upper bound) was never applied.

Do not simply flip it — **re-run the sweep first**, because the corpus moved underneath it:
Notch returned, the kits arrived, and group stamps entered the label space (they are now excluded
from the eval corpus by `db.is_group_label`, so the numbers will differ from the 08-05 run). The
machinery is `eval.py --reid` + `evalmetrics.py`; the nightly batch now writes a fresh artifact
every night, so a current baseline already exists.

### 1.4 The refimg shadow review — **done 2026-08-09**, and it needs one human decision
Full write-up with every number: [the shadow review](refimg-review-2026-08-09.md). The short
version, because two things this entry said the same morning turned out to be wrong:

- **The veto is not inert. It has flagged 18 rows and all 18 are furniture** — the tipped watering
  can against the shrub, 01:44–01:48 on 08-09, which landed a few hours after this entry was
  written. Every crop was opened. The original review question was answerable and its answer is
  the good one.
- **0.9 is not the wrong bar.** Replaying all 5,402 glass-door detections of the shadow week
  against the exact banked references, at bars 0.9 / 0.7 / 0.5 / off, gives 26 / 34 / 68 / 117
  suppressions — and every one of the 117, opened, is the same watering can. Lowering the bar buys
  more copies of an object already caught while spending the only gate that answers the design's
  own photographed failure (a certified reference with an undetected raccoon in it). It stays.

What is true is that coverage binds: 94.9% of boxes die on `reference_has_no_pixels_here`, and **no
daytime box has ever cleared the bar** (max 0.825 over 1,577), so on this camera the veto is a
night instrument. Shipped in response, all still shadow mode: the coverage memory is now a
per-pixel map of filled contours rather than a list of bounding rectangles (correctness, and it
takes the capture thread off a 4.9 ms-at-2,500-rectangles redraw); `ViewWatcher` re-seeds its
template across an unwatched gap, because **the rig was calling sunrise a camera reposition**
(§4 of the write-up — corr 0.987 across the "move", and the false bump flushes the recurrence
ledger every morning); and a `VetoCensus` line every hour now counts the abstentions, which is the
number whose absence made this dig necessary at all.

**The human decision that remains** is design §8 item 2, the measurement that says whether the
coverage gate is essential or overhead: run MegaDetector at conf 0.1 over a day of banked certified
reference frames and count the boxes the live 0.25 threshold rejected. It was deliberately not run
here — it means a second MegaDetector process on the GPU beside the live rig, and this box has
already lost the rig once to memory exhaustion. It wants a quiet window.

Still worth doing on the ordinary schedule: `python refimg.py --review --days 7` around 2026-08-14,
now that there is a week of flags and an hourly census line to read beside them.

### 1.5 One decision left (the restart already happened)
- ~~Restart the rig~~ — **done**: the rig came up at 2026-08-08T21:54 running the new code, and
  the coverage ledger wrote its first `up` row. Everything from the burst is live.
- **Set `cfg.operator_token`** if the household wants the viewer split on. Unset is a deliberate
  no-op — every LAN device keeps full edit rights, exactly as before.

---

## 2. Measurements that should come before features

Both are cheap, both are one-off scripts rather than shipped features, and both change what is
worth building next. This is the project's own order of operations: measure, then believe.

### 2.1 The background-identity diagnostic (the eval's most consequential unrun test)
**The question:** does the *background* identify the animal? Embed crops with the animal blanked
and run the identical session-blocked leave-one-visit-out. If background alone scores well above
chance, then every AUC and the whole decay interpretation is partly a story about where an animal
stood, and the priority shifts from modelling to **capture**.

**The "nearly free" shortcut does not exist — checked, 2026-08-09.** The plan was to cut patches
from `refimg`'s certified-empty frames at the exact bbox positions instead of inpainting. Four
facts kill that route:

- the banked references are **320×180 greyscale** (`refimg.py` pins `W, H = 320, 180` and reads
  them back as `IMREAD_GRAYSCALE`), so a median 313×309 raccoon box is a ~72×73 grey patch, and
  MegaDescriptor wants ~384×384 RGB. A low score would be uninterpretable — "background carries
  no identity" and "grey low-res patches embed to mush" look identical;
- there is **no full-resolution copy anywhere**: `detections.frame_path` is NULL for every row
  (`save_full_frame` is off by design);
- the bank covers about **a day**; the labelled corpus covers two months, so only ~9 of ~181
  confirmed visits have a contemporaneous reference;
- `view_epochs` is **empty for every camera but one, and that one row is a false bump** (§8), so
  "same camera epoch" is a rubber stamp across a door camera that gets repositioned every few days.

**Do it the plain way instead**, which needs nothing new: blank the animal's box in the crop
itself (fill with the crop's own border statistics, or simply mask it) and re-embed. Validate the
harness first by reproducing the *known* number — rebuild the unit list exactly as `eval.py` does,
re-embed the unmodified crops, and confirm the session-blocked LOO comes back at the published
top-1 before a single pixel is blanked. Keep it a scratch script that writes to no table.

**Read the outcome honestly, both ways.** Near chance → the embeddings are about the animal and
the decay result stands as an animal fact. Well above chance → a large share of what has been
called identity is scene memory, and the next move is capture geometry, not a better backbone.

### 2.2 The un-blend threshold sweep (`reid_track_match_threshold = 0.55`)
Shipped without its specificity sweep. The measurement is **free and needs no new labels**: on
confirmed **solo** visits the splitter must produce exactly one cluster, so the over-split rate
per threshold falls straight out of data already on disk. Do this before 1.2, so the D1 session
is not fighting a badly-cut threshold.

---

## 3. Identity: the designs, as corrected by review

### 3.1 Chain-of-eras identity — embrace the decay curve
**The idea.** The embargo curve is not only a limitation, it is a design spec: links are strong at
short range (0.708 top-1 at 1 day, 0.628 at 3 days) and worthless long (0.222 at 21 days, against
0.348 chance). Today `rank_templates` is age-blind — a 41-day-old template competes as an equal
with yesterday's. Restructure identity as a **chain** of week-scale era nodes: a probe matches the
recent end of the chain, and a high-confidence match *extends* it.

**Three corrections from the adversarial pass — the design is wrong without them:**

1. **Additive only, never a filter.** A pure recency gate was already implemented, measured and
   **rejected**: wrong-name rate 0.119 → 0.137 while coverage *fell* 18.0% → 16.1%
   (identity-eval Part 5; restated in `db.py`, `individuals.py` and a test). The measured-harmful
   part was *removing* old templates at match time. Keep the full age-blind human pool exactly as
   it is and **add** machine link anchors beside it.
2. **Admit what an anchor is.** An anchor is functionally a machine-made template, which relaxes
   `individuals.py`'s load-bearing rule that *a wrong auto name can't teach the matcher*. The
   compensating controls are not optional: store link provenance (an additive `via_visit_id` on
   the auto label — `auto_assign` already computes the via-visit and throws it away), invalidate
   descendants when a link is severed (reuse the reject-tombstone semantics), and show the chain
   in the dashboard so a human can see what is holding a name up.
3. **Replay before building.** `leave_one_visit_out` is embargo-LOO — the wrong shape. Write a new
   **forward-chronological** simulation in `evalmetrics.py`, session-blocked within links, with the
   operating point chosen on training folds, and include **both** documented rejections (the pure
   recency gate and pure adjacency propagation) as baselines so the comparison shows the composite
   beats each half. Build the real thing only on a win: roughly **≥2× auto-named coverage at zero
   observed wrong names**.

**Four things moved under this design on 2026-08-08–09, one of them a direct hit:**

- **Group exclusion removed the freshest link.** All nine `+ Kits` visits fall in the single
  densest labelling night in the corpus (2026-08-07 21:31 → 08-08 02:50, ~2,142 crops) and are
  now — correctly — invisible to `templates()`. A design whose whole value is *fresh short hops*
  just lost its freshest one. Net template count barely moved (141 solo vs the eval's 138), so
  every number in the identity eval still stands unamended.
- **The kits are a failure mode aimed straight at this mechanism.** Nine animals with zero
  labels. A forward chain would anchor a kit's visit onto its mother's name and propagate it for
  a week, and leave-one-visit-out **cannot see that** (the same blind spot the doc already
  records for departed animals). Scope the chain to Stan / Notch / Pedro explicitly; kits are out.
- **`labeled_by` is not a substitute for `via_visit_id`.** It is *attribution* (which human), not
  machine provenance (which visit vouched for this name). It does supply the exact additive-column
  migration pattern to copy.
- **The coverage ledger is empty of history** — one row, written when the rig restarted. A chain
  break is this design's core signal, and for every night before 2026-08-08 the ledger cannot
  distinguish "the animal was away" from "the camera was down". Useful in a month; not yet.

**First step: write the replay and run the arms — nothing else.** Add a forward-chronological
`forward_chain_replay` to `evalmetrics.py`, reusing `eval.py`'s existing corpus builder (it
already yields the solo units with group/multi/thin excluded) and its similarity scorer; select
the operating point on a seed half; run direct-match vs chained at link horizons of 3, 5 and 7
days, printing assigned / wrong / coverage / anchor-depth / descendant-contamination per arm.
No DB writes, no column, no UI.

**Expect it to be refuted, and count that as the win.** The honest prediction from the corpus:
Stan and Pedro already have confirmed nights 1–4 days apart, so the age-blind pool always holds a
fresh template for them and chaining changes nothing; the arms diverge only for Notch across his
42-day gap and for Elliot/CutiePie, who have too few templates to measure. If that is what comes
back, the design dies for the cost of a replay — and the replay itself is the first tool this
project has ever had for asking "does identity propagate forward?"

### 3.2 Say "this identity has lapsed" wherever a name is used — the quick one
Carved out of the chain design by review as **worth shipping regardless of whether the chain ever
happens.** Today the only place that admits an identity has gone stale is the operator-only
Template Freshness panel; everywhere else — the profile page, the suggestion payload, the roll
call — a 40-day-old template ranks as an equal to yesterday's without comment.

The data is already computed. What is missing is a first-class per-individual **lapsed** state,
surfaced wherever a name is consumed, that says the matcher can no longer vouch for this animal
and a human re-anchor is what fixes it. It turns the eval's honest ceiling — you cannot reliably
name an animal nobody has confirmed in a week or two — from a fact buried in a document into a
fact the interface tells you.

### 3.3 Family-by-structure: the half of the group-label fix that is still missing
The **defensive** half shipped (group labels can no longer poison the template pool). The
**identifying** half did not: nothing counts how many animals a family visit actually held.

Review's corrections, which matter because the obvious version is wrong: every count must be a
**lower bound**, aggregated per night from clip sustained tracklets *plus* still cannot-link
cliques — not from stills alone. Measured on the 08-07/08 family night, stills gave CutiePie
{4, 4}, Stan {2, 3}, Pedro {1, 2, 2, 2, 3} against a truth of 5 / 4 / 3: CutiePie separates,
Stan and Pedro collide. So the queue's wording has to be "CutiePie + Kits (at least 4 animals
seen at once)", never a headcount. And it is a **seasonal** signal — it needs re-validating as
the kits grow into adult-sized bodies.

### 3.4 Buy the resolution the ear notch needs
The notch is the one **era-invariant** marker in this cast, and it fails only on *resolution* — not
on contrast, not on pose. Anything that puts more pixels on an ear at the dish (a closer angle, a
higher-resolution crop path for close range) converts the single most durable identity signal from
"visible when lucky" into "readable on demand". Related and unbuilt: the trail cam is the one camera
that sometimes gets **close-range daylight** frames, and it remains structurally identity-free (144
raccoon visits, none named, temporal similarity 0.510 vs 0.514 — no structure to threshold).

---

## 4. New signals: hardware and new data

### 4.1 Weigh them — a load cell under the dish (~$15)
The single strongest idea in the whole review, and the only one that adds an identity signal
appearance decay **cannot touch**. Body mass is era-invariant like the ear notch: it separates
"chonky" Notch from Pedro permanently, gives kit growth in **grams** instead of pixel ratios, and
produces per-visit weight-class evidence that fuses with the suggest-confirm loop as a **prior**,
never as a classifier.

**The yield is real — measured, not assumed.** Of 502 glass-door raccoon visits, **330 pass a
solo gate at ≥30 seconds**, median dwell 291 s, about 37 a week (Stan 40, Notch 40, Pedro 39, and
199 unnamed). The instrument would out-supply the human labeller several times over. Expect
~5–13 *gated weighings* a week after placement realities bite, not 37.

**Three risks that change the design, not just the build:**

- **Partial loading is the design-killer, and it was not in the review.** A raccoon at a dish
  stands on two feet; partial weight is *posture*, not mass. The platform must fully support the
  animal, and the plateau-quality test (flat trace, no step-changes mid-plateau) has to be
  validated against known loads before any reading is called a mass. If clean plateaus turn out
  to be rare, what you have built is a posture sensor.
- **The melee gate is inverted from what the review assumed.** Detector co-presence reads *solo*
  during exactly the family melees it is meant to catch — that is the whole finding in
  [§6](#killed-with-reasons). So **the scale must gate itself first** (a second animal stepping
  on is a step-change in the trace); vision is the secondary check, not the primary one.
- **Mass is seasonal, not invariant.** Autumn fat deposition moves every adult. Profiles must be
  dated and drift-aware, or the prior starts rejecting correct names in October.

Plus the boring ones, each with a handle: weatherproof the amplifier (the cell body is the exposed
part); log **raw counts alongside grams** so a re-calibration re-derives history instead of losing
it; write **one row per settled plateau, never per sample**, through a short-lived connection —
an 80 Hz logger writing rows is the "database is locked" incident that killed the rig once
already; and put the cell under something the animals already use, since a strange board may
suppress visits outright.

**Hard prerequisite: the Notch audit (§1.1) comes first.** Mass profiles are built from
human-confirmed labels. Building them on a possibly-swapped Notch/Pedro pair would bake that swap
into a new, slower-decaying, more durable signal — and mass would then look like *independent
confirmation* of the wrong answer. If you would rather not wait, store only the event rows and
rebuild profiles on demand, so a label correction propagates.

### 4.2 Capture sound
Half of this shipped: the two re-encode paths no longer strip audio, so **trail-cam microphone
audio survives and plays** in the dashboard. Unbuilt: the live rig captures no audio at all (the
OpenCV frame pipe carries none), and nothing analyses it. The owner's own cast notes use
vocalisation as an identity marker ("Cutie growls"), and **kit chitter is a litter-present signal
that works off-frame**, which no visual method can match.

Smallest increment: a parallel `ffmpeg` audio capture (dshow on Windows) into timestamped segments
keyed to clip boundaries, muxed at finalize, leaving the frame pipe untouched; then band-energy
features — not a model — to separate growl / chitter / silence.

**Two findings that change the plan before any code:**

- **Do not use the free microphone.** The one mic already on this box is *on the webcam's own USB
  device* — which shoots through a **closed glass door**, so the mic is indoors. A pane of glass
  costs roughly 20–35 dB across 1–4 kHz, exactly where kit chitter lives, while the living room
  is unattenuated. It is the worst possible ratio: human conversation clearly, raccoons faintly.
  Worse, a long-lived audio handle on that composite device can block the **pnputil
  disable/enable cycle that is the only USB-wedge recovery that has ever worked here**. Use a
  separate USB audio device, sited at the glass.
- **Measure siting before building anything.** Half an hour: record ten minutes from the built-in
  mic and from a glass-taped USB mic during a known raccoon visit, and compare 1–4 kHz yard
  energy against indoor energy. If the room is louder than the yard, the free mic is disqualified;
  if the glass-taped mic cannot clear the yard by ~10 dB either, the whole acoustic idea dies for
  the cost of an afternoon instead of a season.

**The privacy decision comes first, in public, and it cannot be retrofitted** — audio captured
before the policy exists is already captured. What is recorded, for how long, what is never
published. (The README's Responsible-use section has been corrected in the meantime: it claimed
the project recorded no audio at all, which stopped being true when the trail-cam video importer
landed in July 2026 and became audible when the transcode stopped stripping audio.)

### 4.3 Weather join
Roadmap item H since 2026-06-28, still unbuilt, and the cheapest covariate available: latitude and
longitude are already configured, and Open-Meteo's historical endpoint can backfill the entire
archive in one call. It answers the most-asked backyard question ("do they come out in the rain?")
and, more usefully, it **explains variance that currently pollutes every other insight** — a wet
night and a quiet night look identical in the data today.

---

## 5. Surfaces and sharing

### 5.1 Attributed family labelling — the queue as a game
Now unblocked: the viewer split ships, a viewer's "who's here" lands as no-stamp testimony, and
as of 2026-08-09 **every human verdict records who made it** (the column had shipped the day
before with nothing writing to it — not even the operator's own confirms). The unbuilt half is
surfacing the **suggest-confirm queue** to viewers as a lightweight "who is this?" card game,
whose verdicts land in a **review tier** the operator promotes with one tap — never straight into
ground truth.

Why it is more than a toy: the project's #1 measured constraint is that identity decays in about a
week and the label supply is *one human*. Family who love these raccoons are the only untapped
supply, and attribution is what makes their answers safe to accept. The trust model (per-labeller
agreement rate against the operator's promotions) falls out of the attribution column for free.

Two things review pinned down that make the build smaller and safer than it looks:

- **The read side already works.** Viewers get the full review-queue payload today — `GET` has no
  operator gate. The game needs no new read endpoint, only a write path.
- **A review-tier verdict must never touch `detections`.** Template selection would ignore it,
  but the roll call, the profile pages, crop browsing and the visit log all read
  `detections.individual_id` with *no* source filter — so a viewer's guess written there would
  immediately corrupt every count on the site. The proposals need their own table, joined only
  where they are being reviewed.

### 5.2 The annual yearbook — and the artifact a child can hold
`makingof_export.py` already solves the hard parts: privacy-screened, people-free static export of
real database content with playable media. Repoint that machinery at a per-season **yearbook** — the
cast roster with portraits and quirks, litters and debuts as a timeline, first/last species dates,
the best clip per individual, the year's headline numbers — written as one self-contained folder
into the backup destination.

It is the human-readable complement to the new CSV export: in five years, with the rig dead and the
venv unbuildable, **a browser still opens it**. It is also the kid-facing surface the project
entirely lacks — a cast of named raccoon mothers with nine kits is, editorially, children's content,
and the cast-card page *is* the printable trading cards.

### 5.3 Epoch-aware occupancy map — and the epoch segmenter underneath it
Trail cam only: the door cam was **measured** degenerate for spatial structure (near-uniform,
centre-frame). Bucket detection bbox centres into a coarse grid **per camera epoch**, and render
a heat overlay on that epoch's own background — "the rabbits hug the fence line; the raccoons own
the middle path". By construction it respects the moving-camera rule: a move starts a new epoch
rather than silently corrupting the map.

**Promote this above "ranked last", for a reason the review missed.** The blocker is not the map,
it is that **`view_epochs` is empty for the trail cam** — and the *same* missing epoch segmenter
also blocks the trail-cam half of the furniture veto, whose design requires a reference pool
from one view epoch. One piece of work unblocks two features.

**The trap that probably caused the pessimism.** A single-frame edge fingerprint reads *an animal*
as a reposition — on this corpus that is 18 raw "breaks" against **5** after taking a rolling
median over consecutive daylight frames. Anyone reusing the live rig's `ViewWatcher` naively on
batch media gets ~16 junk epochs, most holding a handful of detections, and concludes the batches
are too thin to draw. They are not; the segmenter was wrong. Use daylight frames only (infrared
cannot see a reposition), a small rolling per-pixel median, and break on the fingerprint
correlation falling below the existing threshold.

Three constraints when it comes time to persist: epochs must be allocated **chronologically and
idempotently** (bumping an epoch also retires that source's reference images — free today at zero
trail-cam references, not free later); the ~2,764 detections that predate the first daylight clip
belong in an explicit **"epoch unknown"** bucket, never folded into epoch 0; and **run
`staticfilter` per epoch before drawing**, or the map will render the barbecue cover as a
confident hot spot — half of one import cycle was furniture, and a heat map is very good at
making furniture look like habitat.

Cheapest first step: a read-only reporter that writes before/after contact sheets for each
candidate boundary, for a human to confirm against their own memory of when they moved the
camera. No DB writes until the boundaries are confirmed.

---

## 6. Half-built: the parts of the 2026-08-08 burst that stop short

Honest accounting of features that shipped as a *first half*. Each is usable today; none is
finished.

- **The effort ledger only knows one camera.** `coverage_events` is written by the live capture
  loop, so the trail cam contributes nothing, `powerguard` does not write a `down` when it
  detects a wedge, and the three absence surfaces that most need it — the roll call's *overdue*,
  the digest's *quiet regulars*, novelty's *days since* — still read the ledger only in the
  digest's coverage line. Effort-correcting those three is the remainder.
- **The sun-anchored clock stops at the Behaviour tab.** No weekly drift panel is drawn (the data
  is computed and returned), the digest's species roll and the individual profiles still speak in
  clock hours, and the histogram itself is still clock-binned — only the summary line is
  sun-relative.
- **Behaviour tags are per visit, never totalled.** "Stan: 84% feeding visits; the cat: 91%
  transit" is one `GROUP BY` away and would say more about an animal than any single visit does.
- **`labeled_by` now records who, but nothing reads it.** No per-labeller view, no agreement rate,
  no review tier — see §5.1. (Fixed on 2026-08-09: the column was shipped the day before with
  *nothing* writing to it, operator confirms included.)
- **The archive-gated prune was never built.** The only-copy protection for trail-cam footage is
  still a ritual (`--backup-first`, a generous budget, remembering the weekly lag) rather than an
  invariant. Making the pruner refuse to delete a file the day-archive does not contain — for
  sources marked irreplaceable — turns the project's stated asymmetry into code.
- **Shadow-mode reviews still depend on memory.** `STATUS.txt` reports the flag count, which is
  most of the value, but there is no `shadow_reviews` record (feature, shipped date, review due,
  reviewed date) and no dashboard nag. §1.4 is the live example of why that matters — and half of
  the specific gap it exposed is now closed: `VetoCensus` logs every refimg decision hourly,
  abstentions included, so "it flagged nothing" can no longer hide the reason. The *scheduling*
  half is still a date in a comment in `config_local.py`.
- **The never-fired safety paths are still never-fired.** See §7.

## 7. Killed, with reasons

**Do not rebuild these.** Each was proposed, tested against real data, and died — recorded here so
the next enthusiastic reading of the same idea meets the measurement that killed it.

### Per-family kit headcount from track overlap — killed on the detector, not the logic
The proposal: count the maximum number of *simultaneous* sustained tracklets in family visits to
get nightly kit counts ("did all four of CutiePie's kits show tonight?"). It correctly sidestepped
identity decay — counting bodies, never naming them.

**It dies on detection.** Tested against the one certified four-animal window (`live_sightings`,
2026-07-31 00:43–00:47, "Stan with three kits, seen live at the glass door"): the 13 covering
glass-door clips hold **zero** sustained tracklets, and even at a loosened gate the detector
produced a **maximum of one box at any single instant** in every clip. The metric reads 1 where a
human personally counted 4. Glass-door clips average 7.2 s against a sustained-track requirement of
~3 s of continuous boxes, and the melee defeats the detector before any counting logic runs.

**The salvageable kernel:** a yard-level, trail-cam-only "at least N animals at once" lower bound.
That data genuinely exists — six simultaneous raccoons on 2026-08-04 22:39, seven mixed animals on
07-31. Ship it as a per-night curiosity with **no** mother attribution and **no** litter-survival
claim. For actual kit counts the only working instrument is the human eye via the live-sighting log.

### Kit age-class from same-frame box-size ratios — killed by its own measurement
The proposal was to flag kit-vs-adult from the within-frame box-area ratio between a kit and a
co-present adult, dodging the eval's adult-vs-adult size negative (Kruskal–Wallis p = 0.59) and
staying camera-move-safe by being scale-free. It was measured against the live database on
2026-08-09 **before** anything was built, and it fails on its own terms:

- **"Same frame removes the depth confound" is false.** Same frame is not same depth: in the 68
  usable family cannot-link pairs the two animals' foot lines sit a median 0.078 of frame height
  apart (p90 0.457). Depth still has to be modelled — and the ground-plane fit that would do it
  was **destroyed by the July camera swap** (R² 0.835 on 22,614 frames in the 720p era, down to
  0.405 on the 1,698 solo-adult frames of the 1080p era, which is the only era that has kits).
- **The raw feature does not separate at all.** Same-frame box-area ratio: family (`X + Kits`)
  visits median **0.508** (n = 246); solo-adult visits **0.474** (n = 444); unlabelled **0.458**
  (n = 2,081). The kit era is not shifted.
- **Done properly it still does not separate.** Tracklet-median, depth-normalised, furniture
  removed: family pairs median 0.831 / p10 0.649 (n = 68) against kit-free-era adult–adult pairs
  0.866 / p10 0.654 (n = 34). Discrimination AUC — "are these two bodies different sizes?" —
  comes out **0.886 [0.834, 0.937]** in the kit era and **0.914 [0.832, 0.973]** in the adult-only
  era. The intervals overlap: the measure is detecting *two separate bodies*, not *two age
  classes*. Family-vs-adult discrimination is **0.564**, i.e. coin-flip.

What this means beyond the feature: the kits cannot be age-classed from geometry in this corpus,
so the family-era questions ("did all four show tonight?", per-litter growth) need a different
instrument — the human eye via the live-sighting log today, and mass if a load cell ever lands.
It also flags a real casualty worth remembering: **the camera swap invalidated the depth
calibration**, so any future pixel-size reasoning has to be refit per camera era.

### Lowering `COVER_MIN_FRACTION` to unblock the refimg veto — killed by the replay
The obvious reading of §1.4's first draft: coverage abstains on 94.9% of boxes, so lower the bar.
Measured 2026-08-09 by replaying all 5,402 glass-door detections of the shadow week against the
banked references, each box scored on its own crop so nothing depends on clip frame timing: bars
0.9 / 0.7 / 0.5 / **off entirely** give 26 / 34 / 68 / **117** suppressions, and **all 117, opened
and looked at, are the same tipped watering can.** So the bar is not what is holding the veto back
— it is not dangerous to lower on this corpus, it is simply *pointless*, and it spends the one gate
that answers the design's photographed failure mode (a properly certified reference with an
undetected raccoon walking the wall in it). Thirty-eight hours without that failure recurring is not
evidence about a rare unrecoverable event. Full numbers: [the shadow review](refimg-review-2026-08-09.md) §3.

### "The bounding-box amplification is why coverage never reaches the bar" — killed at 1.37×
The suspicion was that `_blobs` remembering `cv2.boundingRect()` instead of the blob itself was
inflating the disowned area enough to explain a 95% abstention rate. Re-running the rig's own
`MotionGate` over the 13,438 motion-positive frames of 2026-08-09 00:00–05:30 says the
amplification is **median 1.33, mean 1.37, p90 1.53** — real, but an order of magnitude short of
the explanation. Switching to filled contours moves the cover at real detection boxes from median
0.000 to 0.009 and leaves **0.0% of boxes** reaching the bar either way. It shipped anyway, as a
correctness and capture-thread-cost fix, and is recorded here so nobody re-proposes it as the thing
that will make the veto fire. The real ceiling is structural: a detector box is, for anything that
moves, exactly the pixels that just moved.

### Already-dead elsewhere, listed so they stay dead
- **Gait / stride as an identity signal** — 27 of 25,041 tracks resolve a stride, and the values
  pile at the band edges. At ~10 fps effective sampling, raccoon stride (1.5–2.5 Hz) is
  Nyquist-marginal. The estimator is arguably broken, but the cost/benefit of fixing it is poor.
- **Facial landmarks** — raccoon eyes sit inside a black mask; the problem is contrast, not
  resolution, so a better camera does not rescue it.
- **Silhouette tail tracing** — 20 of 22 audited traces followed ears, mask bleed or legs. A
  silhouette tracer cannot reach a curled tail. (`traits.py` deletes the tracer and re-raises the
  audit if anyone re-imports it.)
- **SAM background removal before embedding** — tested, no improvement.
- **Rolling background models (MOG2) as an empty-scene reference** — absorbs a still animal in ~23 s
  against 823 s of verified residency.
- **Non-animal "decoy" labels in the species list** — inert. BioCLIP is organism-only and will never
  rank a non-organism prompt above a real species; that is what the general-CLIP gate is for.
- **Tiled/sliced inference and proposal-anchored zoom** — built and measured 2026-08-06: +44% boxes,
  of which 24 of 29 were lanterns, the barbecue cover, bare wall and shrubs.

---

## 8. Smaller leftovers

- **Exercise the never-fired safety paths on purpose.** The auto-assign reject tombstone has never
  been triggered (`rejected_visit_ids` is empty), and `powerguard`'s self-heal is on record as
  untested against a live wedge. Both will first execute during an incident, which is the worst
  moment to discover a bug. Reject one auto name deliberately and confirm the tombstone survives a
  nightly run.
- **The cast bible as data.** `life_events` now exists and the profile page writes to it, but the
  knowledge that justifies the labels still lives in the operator's head: which mother is which,
  the ear-notch marker, "pre-07 labels suspect". A per-individual metadata surface (markers with
  dates, family links, era notes) would make the Notch audit tractable for a future reader.
- **Date-stamp empirical numbers in code comments.** Every threshold comment is a number that was
  true once. A one-line convention (`# measured 2026-07-18, n=113`) plus an `eval.py` check that
  lists dated comments older than the newest artifact would extend the docs' honesty-with-provenance
  to the live surfaces, where the identity eval already caught it drifting.
- **Linux/macOS operational parity.** Everything that keeps the rig alive unattended is
  Windows-shaped: `run_clipmotion.bat`, `schtasks`, `setup_selfheal.bat`, the Power Request API. A
  Linux user gets no cron equivalents and, worse, no statement of what silently stops working
  without them.
- **A furniture bin worth purging.** The kit-age-class measurement turned up ~730 glass-door
  detections clustered on one static box at high IoU — the same class of false-fire
  `staticfilter` was written for, and the same class that produced 308 phantom "brown rats" on
  the trail cam. Worth a look and a purge, and a reminder that a heat map or a size statistic
  will happily describe a barbecue cover with great confidence.
- **`view_epochs` holds exactly one row, and it is wrong** (corrected 2026-08-09; it was empty when
  this list was written). `glass_door_cam` epoch 1, 05:56:52, corr 0.261 — written by the dawn
  false-reposition bug, 35 minutes after dawn at the camera, between two references that correlate
  0.987. The bug is fixed, the row is not: epoch 1 is in force and the current references are keyed
  to it, so deleting it would orphan them. It is a one-line correction to make deliberately. The
  table is still empty for the trail cam, where it remains a prerequisite hiding under §2.1, §5.3
  and the trail-cam half of the furniture veto.
