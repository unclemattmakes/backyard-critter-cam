# Reference-image veto: design, prototype, and the race results

**Prototype:** `scratchpad/refimg/proto/` — CPU only, DB read-only, no project file touched, no GPU.
**Question:** can a "ground truth of empty from this cam view" reference image veto detector boxes
that are furniture, without ever erasing an animal?

**Answer: yes on the glass door, and it is worth building. No on the trail cam as the pipeline
stands today, for a reason that has nothing to do with the metric.** The measured operating point on
the glass door is **588 of 652 furniture box-evaluations suppressed (90.2%), 0 of 4,649 animal
box-evaluations touched**, including 623 near-miss evaluations where an animal and the furniture were
in frame at the same instant. Every one of the 588 suppressions was opened and looked at: all 588 are
the tipped watering can.

Three results changed the design as I built it, and each is evidenced by an image below:

1. A "certified empty" reference built from frames the detector never ran on **contains the raccoon**.
2. The pixel test **alone** erases real raccoons — with the gates off it fires on up to 131 of 372
   confident animal boxes, and even with all gates on, 422 of 4,649 animal boxes matched an empty
   reference on all three metrics and were saved **only** by the recurrence gate.
3. A genuinely certified reference **still** caught an undetected raccoon walking the wall, because
   the detector missed it. That is what policy E exists to fix, and it fixes it by reusing the motion
   mask `MotionGate` already computes and throws away.

---

## 1. The measured constraints this design has to obey

Every number below is a measurement, from the stationarity agent, the drift agent, or this
prototype. Nothing here is a guess, and where something is unmeasured it says so.

| Constraint | Value | Source |
|---|---|---|
| Verified stationary residency (eye-verified) | **823 s** (13 m 43 s) | stationarity, `tc_20260731_2203_feeder_823s` |
| Densely-observed residency (holes ≤ 3.3 s) | **443 s** (7 m 23 s) | stationarity, `gd_20260629_2309_bowl_443s` |
| Merged same-spot occupancy, worst case | **12,657 s** (3 h 31 m) | stationarity |
| Longest observable still period | capped ~300 s by MotionGate | stationarity — every number is a **floor** |
| Required no-update horizon | **≥ 3600 s** | stationarity conclusion |
| Glass-door reposition rate | 1 in 4 days (256 px at 1280 wide) | drift |
| Trail-cam reposition rate | ~1 per **1.3 days** | drift |
| Day↔night flip cost | lum 94.3, DSSIM 0.87 — the largest event | drift → key on **(view, illumination)** |
| IR cannot see a reposition | all IR frames collapse to one view cluster over 12 days | drift |
| Consecutive-frame noise floor | GD day 0.41 / GD night 0.91 / TC IR **0.18** | drift |
| Detector recall | 0.73 (given) | brief; and see §4.3 |

**The two most important consequences.** A rolling background model absorbs an object in ~`history`
frames; at 21.6 fps a 500-frame MOG2 history is 23 seconds — three orders of magnitude below the
823 s floor. So a reference may never be derived from a decaying model. And a reference must be keyed
on `(camera, view epoch, illumination state)`, switched not blended, with illumination derived from
the frame itself (chroma < 6 ⇒ IR, else median < 90 ⇒ night, else day) rather than from a clock.

---

## 2. The candidate policies as built

`proto/policies.py`. All expose `observe(frame) / get(mode) -> Ref`, where `Ref` carries `img`,
`captured_at`, `provenance`, an edge fingerprint, and — load-bearing — a **`cover` mask of the pixels
the reference actually knows**. The veto abstains where `cover` is false. An unknown pixel is not
evidence of emptiness.

| | policy | reference is | protects the sleeper by |
|---|---|---|---|
| **A** | `CertifiedSnapshot` | the most recent single frame the detector certified empty for ≥ hold seconds, per illumination | the animal's own detections resetting the certification clock |
| **B** | `MaskedEMA` | slow per-pixel EMA that never updates inside a detector box or motion blob seen in the last 3600 s | the animal's own detections **and** its own motion |
| **C** | `RankBlock` | per-pixel median over time blocks spanning ≥ 6 h, view-gated, photometrically aligned | needing > 50% block occupancy to win the vote |
| **D** | `Hybrid` | A while fresh (≤ 2 h), C as the fallback | A's guarantee, C's availability |
| **E** | `CertifiedMotionMasked` | **A's frame, with every pixel that blobbed in the last 3600 s marked NOT COVERED** | A's guarantee **plus** a detector-independent motion channel |

Policy E was not in the brief. It exists because of finding 3 below.

Two control variants were run alongside to show failure modes rather than argue about them:
`A_naive_nearby` (treats "no detection row nearby" as certification) and, in race 1, the raw metric
test with no gates.

### The illumination and view machinery

`ViewWatcher` (proto/harness.py) detects repositions from **day frames only** — because the drift
agent showed IR frames cluster into one view across 12 days spanning confirmed repositions, so a
night reference cannot learn it has been invalidated by looking at itself. A day-frame view change
bumps an epoch that flushes **every** mode's reference.

Measured: with no debounce it fired **30 times** across 08-06; with a 3-frame debounce, **9 times**;
the real reposition on 08-05 (drift agent: between 22:01 and 22:43) is found exactly once. The 9
residual false positives land on 08-06 15:18–15:21, 18:11, 19:15–19:23 and 21:30 — which are the
drift agent's three named transient haze/flare regimes. **A frame-count debounce is the wrong shape**;
the recommendation is a wall-clock persistence test (disagreement sustained over ≥ 5 minutes),
because frame arrival is bursty and 3 frames can be 3 seconds.

---

## 3. RACE 1 — the sleeping animal

Warm each policy on real history, drive it through a sequence where an animal is *verifiably*
resident, and at every frame carrying a confident animal detection ask: **would the veto fire on this
animal's own box right now?** The worst frame is what matters, not the average.

### 3.1 The headline failure: a certified-empty reference with a raccoon in it

`proto/race1/tc_20260731_2203_feeder_823s.jpg` — the trail cam's eye-verified 823 s residency case.

- **`A_naive_nearby` FAILS.** Its reference, **1 minute old**, contains the raccoon, plainly visible.
  It fires on the animal at conf **0.79** — `lum 10.9 / 49.5`, `dssim 0.341 / 0.465`. 3 of 8 confident
  animal frames would have been erased. On `tc_20260728_0235_feeder_825s` the same policy scores
  `dssim 0.008 / 0.465` — the reference *is* the animal.
- **`A_certified` returns NO REFERENCE**, which is the correct answer, for the reason in §5.
- **`B_masked_ema` PASSES.** Its reference has a protected hole exactly where the animal is; it
  abstains on 7 of 8 evaluations and never fires.

This is the whole sleeping-animal problem in one picture, and the difference between the two A
variants is a single question: *did the detector actually run on the frame you stored?*

### 3.2 The pixel test on its own erases raccoons

Race 1 applies the metric test **with no gates at all** — no recurrence, no age limit, no view check.
On the glass door that is not survivable, and it is not survivable for *any* policy:

| case (all night, conf ≥ 0.70 animal boxes) | evals | A_certified fires | +sobel | B_masked_ema fires | worst (× threshold) |
|---|---|---|---|---|---|
| `gd_20260806_0007_wall_211s` | 1,470 | **36** | 13 | 0 | lum 0.079, dssim 0.007 |
| `gd_20260806_0155_stan_kits` | 372 | **131** | 120 | **37** | lum 0.128, dssim 0.033 |
| `gd_20260807_0249_wall_141s` | 113 | **20** | 14 | **15** | lum 0.077, dssim 0.039 |

`race1/gd_20260806_0155_stan_kits_A_certified_KILLED.jpg` and the matching `B_masked_ema_KILLED.jpg`
are contact sheets of the crops each would have erased. **They are raccoons** — unambiguous,
frame-filling raccoons walking the wall, from both policies.

Worst-case scores reach **0.007× the threshold**: not marginal, not a near miss. Adding sobel to the
conjunction helps unevenly (36→13, but 131→120), so it is worth having and cannot be relied on. The
drift agent's warning — `animal_min / empty_p95` never exceeds 1 on this camera — is not theoretical.
**No reference policy makes the bare metric safe on the glass door.** What makes it safe is §4.

### 3.3 Policy C is not available where it is needed

`C_rank` returned NO REFERENCE in every race-1 case, on both cameras. This is not a bug, it is the
requirement biting:

- C is only safe if the pool spans **more than twice the worst-case residency**: 2 × 12,657 s ≈ **7 h**
  of *same view, same illumination* samples. That is why `min_span` is 6 h and cannot go lower.
- On the trail cam, IR exists only at night, and a reposition (~1 per 1.3 days) flushes the pool. In
  the 823 s case the last reposition was 07-31 13:50, leaving 1.2 h of IR before the animal arrived.
- On the glass door, night is ~9 h/day, so C only becomes available late in a night and dies at dawn.

C is a real policy, but it needs multi-night pooling with a reposition-aware epoch, and the trail cam
moves too often for it to be reliable. It is the fallback in D, not the primary anywhere.

---

## 4. RACE 2 + 3 — furniture suppression and the near-miss

`proto/run_veto.py`, glass door, 2026-08-06 00:00–03:00 (which contains the Stan+kits visit), warmed
from 08-05 00:00. Furniture ground truth was derived with staticfilter's own rule and then **opened
and looked at** (`proto/labels_gd.py`, sheets `proto/gdc_NN.jpg`).

> **Why the labels needed eyes.** At the glass door's ~1 Hz cadence the static rule is not a furniture
> test. Cluster 11 — 34 detections, span 1,646 min, spread over 2 days, in one place — is a raccoon on
> the wall on two consecutive nights (`gdc_11.jpg`). The stationarity agent's warning holds exactly:
> location recurrence does not separate furniture from animals in this yard. Four clusters are
> furniture (a static bracket labelled *Anna's hummingbird* ×404; a dark gap in foliage labelled
> *Bewick's wren*; the tipped watering can labelled **raccoon ×60**; a shadow on the grass). Thirteen
> are raccoons.

### 4.1 Results

| policy | ANIMAL n=4,649 | FURNITURE n=652 | near-miss box-evals n=623 |
|---|---|---|---|
| | suppressed | suppressed / kept / abstained | animals suppressed |
| **A_certified** | **0** | **588 / 61 / 3**  (90.2%) | **0** (154 furniture killed) |
| **D_hybrid** | **0** | **588 / 61 / 3** | **0** |
| **E_certified_motion_masked** | **0** | **384 / 28 / 240**  (58.9%) | **0** |
| B_masked_ema | 0 | 0 / 516 / 136 | 0 |
| C_rank | 0 | 0 / 0 / 652 | 0 |

**Race 3 is passed by every policy: zero animals suppressed, in 4,649 evaluations and in all 623
near-miss evaluations where a real raccoon and the watering can were in frame together.**

Verification, not just counting: `race23/gd_A_certified_SUPPRESSED_FURNITURE.jpg` and
`gd_E_..._SUPPRESSED_FURNITURE.jpg` are contact sheets of every suppressed crop. All watering can.

### 4.2 Which gate is doing the work

From the decision-reason breakdown for A/D:

- **4,223** animal boxes were kept because *pixels differ from empty*. The metric handled them.
- **422** animal boxes (**9.1%**) passed the full three-metric pixel test and were kept **only** by the
  recurrence gate. Without gate (ii) those are 422 erased detections.
- Of 652 furniture evaluations, 588 needed both gates; 50 failed recurrence, 11 failed the metric.

So the conjunction is not decoration. The pixel gate is necessary and insufficient; the recurrence
gate is necessary and (per the stationarity agent's food-bowl finding, 27 days at conf 0.95)
catastrophically insufficient alone. Neither ordering of these two facts lets you drop either gate.

### 4.3 Why policy E exists — the detector-recall hole

Look at `race23/gd_A_certified.jpg`, top-right panel. That is a **certified** reference — detector ran,
returned nothing, motion quiet, four consecutive verdicts — and there is **a raccoon walking the wall
in it**. The detector simply missed it. The project has this exact blind spot on record: 2026-07-20
dusk, MDV6 missed the raccoon in the dark bokeh above the wall (0.89 when lit).

Holding longer does not fix a miss that repeats in the same place on every frame. Policy A's ceiling
is the detector's recall, full stop.

**E fixes it without the detector.** Furniture does not move; an animal does. `MotionGate` already
computes a foreground mask every frame and collapses it to a single number (`backyard_cam.py:687–728`).
E keeps the mask: the snapshot is still a detector-certified frame, but any pixel that blobbed in the
last 3,600 s is marked NOT COVERED and the veto **abstains** there. A missed animal costs an
abstention, never an erasure — and the watering can, which fires the detector but never moves, stays
vetoable.

The price is measured: **58.9% furniture suppression instead of 90.2%**, with 240 abstentions where
something moved near the can within the hour. That is the correct trade under this project's stated
asymmetry.

### 4.4 Policy B is a good guard and a useless reference

B suppressed **0 of 652** furniture: 136 abstentions (its protection holes sit exactly on the furniture,
which fires the detector constantly) and 516 keeps. B's holes are the right behaviour for safety and
the wrong behaviour for a veto. Use B's *idea* (protection), not B's *output*, which is what E does.

---

## 5. The trail cam is a different problem, and it is not the metric's fault

**`DET_COVERS_CLIP_FRAMES` is the finding.** The glass-door rig runs the detector on the frames it is
capturing. The trail-cam importer runs the detector on the SD card's **still photos**; it never runs on
video (clipmotion last ran 2026-06-16, tracks→individual is 0/897). So:

- A trail-cam clip frame carries **no detector evidence at all**. `A_certified` therefore has no input
  and correctly returns NO REFERENCE. `A_naive_nearby` fabricates one, and §3.1 is the picture of what
  that fabrication looks like.
- Worse, the detections that need vetoing come from stills, and `import_trailcam` saves **crops only** —
  `save_full_frame` has never been on. There is no still full frame to compare.

Running the full frame-based veto on the trail cam anyway (`run_veto.py tc`, 2026-08-03 20:00 →
08-04 08:00, with staticfilter's deleted grill/chimney rows re-injected) produces exactly what that
implies: **every policy abstains on all 19 furniture evaluations**, on `no_reference` for A/C/D and on
coverage/epoch for B. Twelve hours of that night yielded 418 clip frames across 16 clips — the trail
cam's clip coverage barely overlaps its own still bursts.

### 5.1 The crop route works geometrically, and the published IR threshold does not

The crop *is* the box's pixels, and stills (2560×1440) and video (1920×1080) share the framing. So a
crop can be compared directly against the corresponding region of a clip-derived reference.
`proto/run_tc_crops.py` builds a per-pixel median IR reference from 54 clip frames over 08-03..08-04
and scores 72 ground-truth furniture crops (from staticfilter's applied-sweep manifest — the grill and
chimney starter) against 60 confident raccoon/opossum crops from the same nights.

`race23/tc_crop_vs_ref_FURNITURE.jpg` and `..._ANIMAL.jpg` show crop | reference-region side by side.
**The geometry lines up** — the grill in the crop is the grill in the reference.

At the drift agent's IR thresholds the veto fires on 60 of 72 furniture crops (83.3%) — **and on 7 of
60 real animals (11.7%)**. `race23/tc_VETOED_ANIMALS.jpg` shows those seven: they are raccoons at a
feeder, unmistakable. The drift agent flagged this precisely: the IR threshold came from **n = 5**
animal boxes and "will only move down with more data". It moved down.

### 5.2 The corrected trail-cam operating point — and it is a good one

With 60 animal crops instead of 5 (`lum` animal minimum 14.06, `sobel` animal minimum 1.040):

| rule | furniture suppressed | animals erased |
|---|---|---|
| `dssim` at the published 0.4648 | 60/72 (83.3%) | **7/60** |
| `lum < 14.06` (animal minimum) | 68/72 (94.4%) | **0/60** |
| `sobel < 1.040` (animal minimum) | 72/72 (100%) | **0/60** |
| `lum < 12.65 AND sobel < 0.936` (10% margin) | 56/72 (77.8%) | **0/60** |
| `lum < 11.25 AND sobel < 0.832` (20% margin) | 48/72 (66.7%) | **0/60** |

**On the trail cam in IR, `lum` and `sobel` separate cleanly and `dssim` does not** — the opposite of
the glass door, where structure outlives luminance. `census` is inverted (furniture scores *higher*
than animals) and must not be used in the dark, exactly as the drift agent found. The IR flash is a
non-issue: it is the quietest mode on either camera (consecutive-frame floor 0.18).

---

## 6. Recommended architecture

### Glass door — **policy E, with D's fallback**

```
reference key   : (camera, view_epoch, illumination ∈ {day, night, ir})
primary         : E — detector-certified snapshot, cover = NOT(motion blobs in last 3600 s)
fallback        : C — per-pixel median, ≥ 6 h span, same view epoch, same illumination
                      (fallback takes the strict path in the veto: 4 metrics, 10 events, 3 days)
metrics         : lum AND dssim AND sobel, all below threshold
thresholds      : day   lum 3.869  dssim 0.1019  sobel 0.2218
                  night lum 11.406 dssim 0.2346  sobel 0.4854
max age         : 2 h
recurrence      : ≥ 5 independent firings (≥ 600 s apart) at IoU ≥ 0.60, over ≥ 2 calendar days
```

Start at E (58.9% of furniture, provably animal-safe against a detector miss). A/D's 90.2% is
available by dropping the motion mask, but only once §8's detector-in-loop recall test says the blind
spot is closed.

### Trail cam — **do not ship the veto yet; ship the two prerequisites**

The veto is not implementable on trail-cam detections today. Two additive changes make it so:

1. **Keep one full frame per still burst on import** (or per N stills) — the reference needs a frame
   from the same sensor path as the detection. Without it, everything below is the crop workaround.
2. **Reference from clips, veto crops** — works today, geometry verified (§5.1). Reference = policy C
   (rank median), because A/E are impossible without a detector on video.

```
metrics         : lum AND sobel      (NOT dssim, NOT census — §5.2)
thresholds      : ir  lum < 12.65  sobel < 0.936     (10% below the observed animal minimum,
                                                      n = 60 animals / 72 furniture)
max age         : 4 h — but see the view-epoch caveat: IR cannot detect a reposition,
                  so an IR reference must be invalidated by the previous DAY frame's epoch
pool            : ≥ 6 h span, ≥ 8 blocks, same view epoch. Frequently unavailable (~1
                  reposition per 1.3 days), so expect abstention to be the common outcome.
```

**Do not port the glass door's thresholds to the trail cam, or vice versa.** They disagree about which
metric works, and both disagreements are measured.

### Availability — the honest floor

`proto/availability.py`, over the whole clip corpus:

| | glass door | trail cam |
|---|---|---|
| clip coverage | 4.2 h / 2,042 intervals | 6.3 h / 891 intervals |
| clips with zero detections | **0** | 743 |
| certification opportunities (≥ 4 s) | 72 | 889 |
| detections with a certified ref < 2 h old | **6.5%** | 64.0% |

The glass-door number is a **floor, not a measurement of the live rig**. Its clips are 3–9 s long and a
clip exists only while the detector is hitting, so the DB structurally cannot show a quiet stretch.
Live, the rig captures at 21.6 fps around the clock and certifies whenever motion fires and the
detector returns nothing — which the DB never records, because nothing is written. In the race-1
replay, A had a reference for 480 of 537 animal evaluations. **True availability is unmeasurable
offline and must be instrumented (§8, item 1).**

---

## 7. Additive schema for logging suppressed boxes

Nothing is deleted. This follows the project's existing soft-delete convention (`clips.pruned_at`,
`detections` outliving pruned clips): the row is written, flagged, and filtered at read time by the
co-presence and analytics consumers. **Not implemented — DDL only, no DB writes in this prototype.**

```sql
-- 1. Soft suppression on the detection itself. NULL suppressed_at == a live row, so every
--    existing query keeps working unchanged until it opts in.
ALTER TABLE detections ADD COLUMN suppressed_at    TEXT;    -- local ISO 8601 w/ offset
ALTER TABLE detections ADD COLUMN suppressed_by    TEXT;    -- 'refimg_veto' | 'staticfilter' | ...
ALTER TABLE detections ADD COLUMN suppress_ref_id  INTEGER REFERENCES reference_images(id);
ALTER TABLE detections ADD COLUMN suppress_detail  TEXT;    -- JSON gate trace, see below

-- suppress_detail is the whole decision, so a bad suppression is diagnosable months later
-- without re-running anything:
--   {"decision":"SUPPRESS","reason":"matches_empty_reference_at_a_recurring_spot",
--    "provenance":"certified+motion_masked","view_corr":0.93,"age_s":312.4,
--    "scores":{"lum":4.70,"dssim":0.0245,"sobel":0.0922,"census":0.1802},
--    "thresholds":{"lum":11.406,"dssim":0.2346,"sobel":0.4854},
--    "recurrence":{"events":9,"days":3,"n":214}}

-- 2. The references themselves, so a suppression can be replayed against the exact image.
CREATE TABLE reference_images (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT    NOT NULL,   -- matches detections.source
    illumination  TEXT    NOT NULL,   -- 'day' | 'night' | 'ir', DERIVED FROM THE FRAME
    view_epoch    INTEGER NOT NULL,   -- bumped when the camera is seen to move
    captured_at   TEXT    NOT NULL,
    provenance    TEXT    NOT NULL,   -- 'certified' | 'certified+motion_masked' | 'rank_p50'
    image_path    TEXT    NOT NULL,   -- PNG, 320x180 grey (the resolution the thresholds are for)
    cover_path    TEXT,               -- PNG mask of KNOWN pixels; NULL == fully covered
    edge_fp       BLOB,               -- 320x180 float32 fingerprint, for the view gate
    n_frames      INTEGER,            -- 1 for a snapshot, pool size for a rank reference
    span_s        REAL,               -- 0 for a snapshot
    retired_at    TEXT                -- set on epoch change; never deleted
);
CREATE INDEX idx_refimg_lookup ON reference_images(source, illumination, view_epoch, captured_at);

-- 3. Camera view epochs, so "the camera moved" is a first-class recorded event rather than
--    something inferred after the fact (the failure mode config.py warns about: a stale zone
--    fails SILENTLY).
CREATE TABLE view_epochs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT    NOT NULL,
    epoch        INTEGER NOT NULL,
    started_at   TEXT    NOT NULL,
    detected_by  TEXT    NOT NULL,   -- 'edge_fp_corr'
    corr         REAL,               -- the correlation that triggered it
    UNIQUE(source, epoch)
);
```

**Read-side contract.** Suppressed rows must be excluded from exactly three places and left alone
everywhere else: `individuals.still_tracklets`' cannot-link constraint (the thing this is for),
co-presence badges, and species statistics. They stay in the dashboard behind a "show suppressed"
toggle, and they stay in `eval.py` — because the only way to learn the veto's precision is to keep
scoring the boxes it removed.

**Ship it in shadow mode first.** Write `suppressed_at` for a week without any consumer honouring it,
then look at a week of suppressed crops on one contact sheet. That is exactly the loop that produced
§4's verification, and it costs one column.

---

## 8. What is NOT validated, because the GPU was off limits

The import was running MegaDetector throughout, so nothing below could be run. Each item names the
test to run once it finishes.

1. **Real certification availability (the biggest unknown).** §6's 6.5% is a floor derived from clips,
   and the live rig's true rate is unmeasurable from the DB. **Test:** add two counters to the capture
   loop — detector invocations, and invocations returning zero boxes — and log the longest
   detector-empty run per hour for a day. If certified references are available >90% of the time, ship
   policy E as primary. If not, D's C-fallback becomes load-bearing and needs its own threshold study.

2. **Detector recall at the reference moment.** §4.3 shows a certified reference containing an
   undetected raccoon. I could not measure how often that happens, because measuring it means running
   the detector. **Test:** take the certified frames policy E stores over a day, run MegaDetector on
   them at conf 0.1, and count how many contain a box the live 0.25 threshold rejected. That number is
   E's motion mask's actual workload.

3. **Detector-in-loop veto, end to end.** Everything here replays boxes from the `detections` table.
   The live path has to run the veto *between* the detector and the DB write. Cost is **7.6 ms** per
   frame for the `Pair` construction and **0.056 ms** per box for `.on(box)` (measured on this CPU at
   320×180) — so it is one 7.6 ms hit per frame that has any box at all, amortised over that frame's
   boxes by the pair cache. At the 21.6 fps loop that is ~16% of one frame's budget on frames that
   have detections, which is probably fine and is **not measured inside the real loop**.

4. **The trail-cam full-frame prerequisite.** §6 recommends keeping one full frame per still burst.
   Nothing was changed in `import_trailcam.py`, so the crop workaround in §5.1 is the only path tested,
   and it was tested on one camera position over two nights.

5. **The glass-door daytime thresholds are untested here.** Every glass-door race ran at night (the
   adversarial cases are all nocturnal). Day thresholds are the drift agent's, unexercised by this
   prototype.

6. **`min_span` for policy C was never varied.** 6 h comes from the 2 × 12,657 s argument, not from a
   sweep. A sweep needs more clip history than the 4 unpruned glass-door days on disk.

7. **The reposition detector needs re-tuning** (§2). A 3-frame debounce still yields 9 false epochs a
   day on the glass door, all on the drift agent's known transient regimes. The wall-clock persistence
   test proposed there is untested.

8. **`E`'s `no_update = 3600 s` and `blob_min = 25 px²` were not swept.** The horizon comes from the
   stationarity floor; the blob threshold was set once and left. Both want a sweep against a week of
   real motion masks.

---

## 9. Files

```
scratchpad/refimg/proto/
  common.py           DB (read-only), clip streaming, illumination test, metrics, thresholds
  policies.py         A / B / C / D / E + the Ref type with its coverage mask
  harness.py          replay driver, MotionGate transplant, ViewWatcher, drawing
  veto.py             the six gates, the online Recurrence ledger
  furniture.py        ground-truth furniture: staticfilter manifest (TC) + derived clusters (GD)
  labels_gd.py        the eyeballed glass-door labels, and why the derivation alone is not enough
  run_sleeper.py      RACE 1        -> race1/*.jpg, race1/*_KILLED.jpg, race1/results_*.json
  run_veto.py         RACE 2 + 3    -> race23/gd_*.jpg, race23/results_gd.json
  run_tc_crops.py     trail-cam crop route -> race23/tc_*.jpg, race23/tc_crop_veto.json
  availability.py     certified-reference staleness over the whole corpus -> availability.json

  key images to open:
    race1/tc_20260731_2203_feeder_823s.jpg              a raccoon inside a "certified" reference
    race1/gd_20260806_0155_stan_kits_A_certified_KILLED.jpg   raccoons the ungated metric erases
    race23/gd_A_certified.jpg                           a real certified reference, and the miss in it
    race23/gd_E_certified_motion_masked_SUPPRESSED_FURNITURE.jpg   all 384 suppressions, all the can
    race23/tc_VETOED_ANIMALS.jpg                        the 7 raccoons the published IR threshold kills
```

DB accessed read-only throughout; no project file modified; no GPU used; no D: drive access.
