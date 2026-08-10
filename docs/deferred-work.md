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

### 2.1 The background-identity diagnostic — **RUN 2026-08-09. It is scene memory.**
**The question was:** does the *background* identify the animal? **The answer is yes, and it is
most of the signal.** Session-blocked leave-one-visit-out over the same 139 confirmed-solo raccoon
visits, same protocol, same corpus — the only difference is which pixels the embedder sees:

Full write-up, with the harness validation and the paired statistics:
[the background-identity diagnostic](background-identity-2026-08-09.md).

| arm | what the embedder sees | blocked | 7-day embargo | 21 days |
|---|---|---|---|---|
| **intact** (reproduces the published number) | the crop as saved | **0.741** | 0.482 | 0.122 |
| **animal** (control: ring blanked, box kept) | 59% of the crop | **0.727** | 0.489 | 0.108 |
| **background** (the detector's box blanked) | 41% of the crop | **0.597** | **0.489** | 0.094 |
| chance | — | 0.345 | 0.345 | 0.345 |

**The control passed**, which is what makes the rest readable: blanking 41% of every crop costs
nothing measurable (intact vs animal, 7 probes one way and 5 the other, exact binomial p = 0.77),
so MegaDescriptor is not embedding mutilated images to mush and 0.597 is information rather than
vandalism.

**With the animal removed from its own photograph the matcher still names it right 60% of the
time.** The animal does carry more (25 vs 5 discordant, p = 0.0003) — but the two channels are
**redundant, not additive**: 78 of the background's 83 correct answers are ones the animal also
gets, and only 5 probes are background-right where the animal is wrong. Who an animal is and where
it was photographed are confounded in this corpus.

**And at a week they are the same number.** The animal's margin over pure scene matching is +0.144
same-week and **0.000 at a 7-day embargo** (0.482 vs 0.489, one probe). Whatever survives seven
days in this embedding is not animal-specific.

**What it changes.** The doc's own decision rule fires the second way: the next move is capture
geometry, not a better backbone — a backbone swap cannot separate two channels carrying the same
information. It also raises the value of the era-invariant signals already on this list (the ear
notch §3.4, mass §4.1) for a second reason: they cannot be confounded with the yard. And the null
is wrong — "better than chance" has meant better than 0.345, when the honest floor is the
background arm at the same embargo. **`eval.py` should carry a background arm**, because the number
that matters is the margin and it is currently uncomputed and unwatched.

<details><summary>The plan as written, and the shortcut that did not exist</summary>

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

</details>

### 2.2 The un-blend threshold sweep — **run 2026-08-09; there is no operating point**
Both halves are done and both came back negative, which is the useful kind of answer: **the D1
session (§1.2) is not being held up by a badly-cut threshold, so go and do it.** Numbers and the
reason they are two different questions are in [Killed, with reasons](#killed-with-reasons); the
short version:

- This entry conflated two numbers. Over-split is decided by the **clustering distance**
  (`unblend_visit_stills(distance=0.45)`); `reid_track_match_threshold` gates whether a cluster
  gets a name **suggestion** and cannot change how many clusters there are.
- Sweeping the distance 0.20 → 0.70 moves solo over-split only 72.2% → 54.3%, against a
  **structural floor of 46%** that no threshold can touch: 56 visits contain a tracklet with no
  embedding at all (its own group by construction) and 13 contain a same-frame cannot-link. The
  lever is embedding coverage — 22.0% of still-tracklets carry no vector — not the cut.
- Sweeping the suggestion threshold on its own terms: at the shipped 0.55 it names the **wrong
  animal 46.9% of the time**. Written into `config.py` next to the number.

What it changes downstream: over-splitting a solo visit costs the D1 labeller extra clicks, not
correctness (every group is the same animal and gets the same name). The *suggestion* is the part
to distrust — treat it as a prompt, never as a default.

---

## 3. Identity: the designs, as corrected by review

### ~~3.1 Chain-of-eras identity~~ — **replayed 2026-08-09, refuted, and that was the plan**
The replay was written and the arms were run. `evalmetrics.forward_chain_replay` ships (with
`adjacency_propagation` beside it as the calendar baseline) — no DB writes, no column, no UI, as
specified. Against the design's own bar of **≥ 2× auto-named coverage at zero observed wrong
names**, the best coverage ratio anywhere on the grid is **1.27×, and it adds wrong names**.
Numbers, the two documented rejections re-run as baselines, and the methodological trap the
specified experiment would have walked into: [Killed, with reasons](#killed-with-reasons).

The tool stays, because it is the first thing this project has ever had for asking *does identity
propagate forward?* — and it caught something the design pass could not: the experiment as
specified is structurally unable to show a gain. Everything below is kept as the record of what
was proposed and why it looked right.

<details><summary>The design as it stood before the replay</summary>

#### Chain-of-eras identity — embrace the decay curve
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

*(Measured: the prediction was right, and the median gap to the same animal's last confirmed
visit is **0.90 days**.)*

</details>

### ~~3.2 Say "this identity has lapsed" wherever a name is used~~ — **shipped 2026-08-09**
`individuals.identity_lapse` is the state (**fresh / fading / lapsed / none**) plus the expected
top-1 at that template age, and every boundary is a measurement rather than a choice: re-measured
at finer granularity, top-1 runs 0.741 same-night → 0.482 at a week → 0.403 at ten days → **0.259
at a fortnight, below the 0.345 majority baseline**. Somewhere between 10 and 14 days the matcher
stops beating "just say Stan"; `reid_queue_stale_days` was already 14 for that reason, so the
state reads that number instead of inventing a second one to drift from.

It is surfaced in all four places a name is consumed — the cast panel, **every queue suggestion**
(the offer now carries the lapse of the name it proposes), the profile page, and the roll call,
where it sits beside *overdue* as the other half of the question: overdue is about the raccoon not
coming, lapsed is about nobody having confirmed it. **Labelled, never filtered** — gating on
template age is a measured-dead idea (§7).

Two things fell out of wiring it to several surfaces at once, both now fixed: the dashboard's
hard-coded accuracy table was quoting the **session-leaked** 0.82 for a same-night template, and
a first cut had the roll call calling Notch *fresh, 0.7 days* while the Individuals tab called the
same animal *lapsed, 45.6 days* — a confirmation the nightly embed pass had not reached is not a
template, and the multi-animal test needs all three of `is_multi`'s channels (dropping the clip one
cost three weeks of error on Stan). One definition now, `individuals.usable_template_visits`.

### 3.3 Family-by-structure — the counting half shipped 2026-08-09; attribution stays dead
`stats.crowd_peak` now reports the busiest INSTANT of each period — "at least N animals at once",
with the time, the camera and the species mix — in the digest and on the Dispatch tab. It is a
lower bound three times over (detector recall on a huddle ~0.39, the counting is greedy, the
stills only see instants something was saved), it carries **no attribution** ever, and the wording
says "at least" everywhere it appears. Measured on the corpus: the trail cam demonstrably held
**5 raccoons at once** on 2026-08-04 01:19 and again on 08-07 00:59; 14 of its 21 nights reach 4+.
The glass door reaches 3+ on 33 of 63 nights — mostly crows, but 4 raccoons at once on 08-06.

**One correction to this entry, measured.** It said to aggregate "from clip sustained tracklets";
taken literally that is wrong. `clip_tracks.n_sustained` counts TRACKLETS, not simultaneous
animals — one animal fragments into many — and summing it per night reports *19 animals* on a
trail-cam night whose stills say 5. A real clip-side count needs per-sample overlap from the
stored `track` polyline (which does carry sample times and boxes, so it is buildable). Until it
is, the shipped count is stills-only and says so.

Still missing, and still gated on a human: per-VISIT wording in the suggest-confirm queue
("CutiePie + Kits (at least 4 animals seen at once)"). And it is a **seasonal** signal — it needs
re-validating as the kits grow into adult-sized bodies.

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

**The method is now measured on the trail cam, and it works — 2026-08-09.** 284 daylight
trail-cam clip frames, 2026-07-23 → 08-07, against unusually good ground truth (the card is
formatted and the camera dismounted every import cycle, so 07-22 / 07-27 / 07-30 / 08-02 / 08-05
are all real repositions):

| pairs | n | p05 | median | p95 |
|---|---|---|---|---|
| same DAY (same mount, hours apart) | 3,745 | 0.441 | **0.815** | 0.973 |
| across a known remount | 30,168 | 0.101 | **0.260** | 0.529 |

At the shipped `VIEW_CORR_MIN` = 0.55 that is 8.7% false "it moved" against 4.3% missed, and the
**best balanced accuracy at any threshold is 0.936 at corr 0.536** — the shipped number is already
within noise of optimal. Daylight change costs real correlation but not enough to matter here:
same-day pairs run 0.917 at an hour apart, 0.712 at 4–8 h.

**Two corrections that came out of measuring it.** First, mine: grouping "same mount" by *import
cycle* gave a much gloomier 0.799, because Matt evidently repositions the trail cam **within** a
cycle too (07-30 and 07-31 are visibly different framings) — so the epoch count will exceed the
cycle count, and "same cycle" is not a control. Second, do not carry the glass door's result over:
there the same fingerprint runs 0.075–0.68 across one provably stationary day
([refimg review](refimg-review-2026-08-09.md) §4), because that camera shoots through glass and
its floodlit evening frames classify as `day`. **This signal is camera-specific.** It is sound on
the trail cam, which is the camera §5.3 is about.

So the first step below is de-risked, not blocked: build the reporter.

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
- ~~**Behaviour tags are per visit, never totalled.**~~ **Shipped 2026-08-09**:
  `stats.behaviour_profile` totals the tag per individual and per species, on the profile page.
  Raccoons 55% lingered / 35% fed here / 9% passed through over 332 readable visits; the squirrel
  59/17/25. Two things the build had to learn: it must count **pruned** clips (the prune is soft —
  filtering them left Stan with 0 tagged visits out of 64), and about **half of all visits have no
  clip overlapping them**, so the surface prints how many it could not read and refuses a
  percentage below three tagged visits. It is also NOT built on `clip_tracks.individual_id`, which
  is still 0-of-897 populated — so `stats.individual_motion`, which reads that column, still
  returns nothing for everybody. That one is the remaining half.
- **`labeled_by` now records who, but nothing reads it.** No per-labeller view, no agreement rate,
  no review tier — see §5.1. (Fixed on 2026-08-09: the column was shipped the day before with
  *nothing* writing to it, operator confirms included.)
- ~~**The archive-gated prune was never built.**~~ **Shipped 2026-08-09.**
  `cfg.clips_irreplaceable_sources` (default `("trail_cam_sd",)`) makes `prune_clips` refuse to
  delete a file whose day-archive zip is not already on the backup drive. It **fails closed** three
  ways — no destination configured, an unreachable drive, an unrecognised path layout all mean
  "I cannot prove a copy exists", so the file survives and the budget is exceeded — and it says so
  loudly, because only a human running `backup.py` can clear it. Replaceable sources are untouched:
  the live rig's rolling window must keep rolling whatever the backup drive is doing, or an
  unplugged drive quietly fills the disk.
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

### Chain-of-eras identity — killed by its own replay (§3.1), at 1.27× against a 2× bar
`evalmetrics.forward_chain_replay`, run 2026-08-09 over the 139 confirmed-solo raccoon visits
(2026-06-07 → 08-06, 6 individuals) on a snapshot.

**First, the trap the specified experiment was walking into.** The plan said to reuse `eval.py`'s
corpus builder, which yields *labelled* units. On an all-labelled corpus **an anchor can never
reach anywhere a human template does not already reach** — the anchor's own visit is a labelled
unit sitting in the same past, with the same similarity — so chaining is redundant by construction
and can only add a wrong candidate. Run that way it loses at **0 of 28 operating points**, which
reads like a result and is really a property of the corpus. A fair test has to withhold labels, so
the replay takes an `unlabelled` set: those visits become "nobody has named this", stay scorable
against their true label, and may become anchors. In reality **79% of this yard's 657 raccoon
visits are unnamed**, so 75% withheld is the realistic corner.

| withheld | operating point | direct named / wrong | chained @7d named / wrong | contaminated |
|---|---|---|---|---|
| 25% | 0.76 / 0.12 | 8.0 / 1.0 | 7.8 / 1.0 | 0.0 |
| 50% | 0.76 / 0.12 | 18.4 / 3.6 | 18.6 / 3.8 | 0.0 |
| **75%** | **0.76 / 0.12** | **24.6 / 4.8** | **28.8 / 7.0** | **1.0** |
| 75% | 0.88 / 0.02 | 8.2 / 1.0 | 10.4 / 1.2 | 0.4 |

(mean of 5 random withholding seeds, scored only on the withheld visits — the population
auto-assign actually faces.) **Best coverage ratio anywhere on the grid: 1.27×, and it buys that
by adding wrong names.** The design's bar was ≥ 2× at zero wrong. It is not close, and the
contamination column is the compensating control the design said was mandatory, doing its job:
names standing on ground a previous mistake laid.

**Why, measured:** the median gap to the same animal's last confirmed visit is **0.90 days**
(p90 3.26), and **88.2% of probes already have a human template within 3 days**. There is nothing
for a fresh anchor to add — exactly the refutation the design pass predicted.

**The two documented rejections, re-run as baselines on the same corpus.** The pure recency gate
(a filter, not an addition) costs coverage as it always did: 0.257 / 0.386 / 0.429 at 3 / 7 / 14
days against 0.429 age-blind. Pure adjacency propagation — "name it after whoever came last, no
appearance test" — names 136 of 139 at a **39.7% wrong-name rate**, against the direct appearance
arm's 19.5% at a quarter of the coverage. Appearance halves the error rate of the calendar. That
is a real margin and a thin one, and it is the honest size of what the embedding contributes in
forward mode.

### An operating point for the un-blend thresholds — killed by its own sweep (§2.2)
Measured 2026-08-09 against a **snapshot** of the live DB, over 151 human-confirmed solo raccoon
visits (487 still-tracklets, 341 groups, 139 templates across 6 names).

**Snapshot, not the live DB, and that is a finding in itself.** The first run of this sweep was
void: `visits.refresh` DELETEs and re-INSERTs the whole visits table, so every visit id changes
each time the rig collapses detections into visits. Ids climbed past the corpus mid-sweep and later
passes silently read "no such visit" — producing a clean-looking table in which the *tracklet*
count moved with the *clustering* parameter, which is impossible. Anything measuring this database
across more than one pass must snapshot first, or key on something that is not a visit id.

**Half one — the clustering distance, which is what "over-split" actually tests.** On a confirmed
solo visit the splitter must return exactly one group:

| distance | 0.20 | 0.35 | **0.45** | 0.55 | 0.70 |
|---|---|---|---|---|---|
| solo visits over-split | 72.2% | 70.9% | **68.9%** | 64.2% | 54.3% |
| multi visits under-split | 11.4% | 11.4% | **11.4%** | 14.3% | 14.3% |

Eighteen points of over-split across the whole usable range, three of under-split. There is no
knee. And 69 of the 111 fragmenting solo visits have a **structural floor** of ≥ 2 groups that no
distance can move: 56 contain a tracklet with no embedding at all (vectorless tracklets are their
own group by construction) and 13 contain a same-frame cannot-link. Of the 42 where the distance
genuinely decides, 35 still over-split at the shipped value. **The lever is embedding coverage —
22.0% of still-tracklets carry no MegaDescriptor vector — not the cut.**

**Half two — `reid_track_match_threshold`, which gates the name suggestion.** Leave-one-visit-out
with a same-day embargo: shown 56.3 / 46.9 / 40.5 / 34.3 / 18.2% and right 45.3 / 53.1 / 60.1 /
63.2 / 72.6% at 0.40 / 0.55 / 0.65 / 0.70 / 0.80. Accuracy is bought one-for-one with coverage all
the way up, and the shipped 0.55 sits at 53.1% against a ~34.5% most-frequent-name baseline. The
confusion is corpus-wide rather than one bad pair (Notch→Stan 25, Stan→Notch 12, Stan→Pedro 8,
Pedro→Notch 8), which is the **2026-08-05 identity decay showing up in still-tracklet space** —
where prototypes average 5–30 crops from one session and are thinner than the visit prototypes the
eval measured. Nothing here is fixable by choosing a better number.

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
