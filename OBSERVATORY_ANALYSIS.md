# The Backyard Observatory — Data, UX & Product Analysis

> Generated 2026-06-28 from a multi-agent analysis of ~3 weeks of live data (6 specialist
> analysts → independent data-truth verifier (92 claims re-derived, 20 corrected) →
> synthesis). Scope: the data we've collected, the ML pipeline, and the dashboard UX.
> Companion to [PLAN.md](PLAN.md) (design philosophy) and [REVIEW.md](REVIEW.md)
> (code-review / sharing-readiness). This one is about the *observatory*, not the code.
>
> *Window: 2026-06-07 → 2026-06-28, one camera. The DB is still being written, so raw
> counts carry a small live drift — numbers below are verifier-confirmed to ±drift.*

## State of the Observatory

In 22 days the glass-door cam has logged **~36,300 animal detections** (single source
`glass_door_cam`, all class `animal`, still growing), resolving to **29 real species** plus
9 junk labels. The yard is a **raccoon station with a diverse supporting cast**: raccoon =
**55.6% of all detections** (56.8% of real-species), top-5 species = **95.2%** (raccoon,
American crow, gray squirrel, Virginia opossum, dark-eyed junco), Shannon **H=1.42**, Pielou
evenness **J=0.43**.

It runs as **two near-disjoint guilds** that swap the stage at twilight: nocturnal mammals
(raccoon peak 22h, opossum peak 04h) hand off to a diurnal bird/squirrel guild within ~1–2
hours — mammal share collapses **100% → 4%** between 04h and 07h and rebounds **43% → 99.8%**
between 19h and 21h. **11,028 detections (30.4%)** carry an individual_id across **7 named
animals** (Notch 5,255, Stan 3,657, Elliot 928, Miss B. 431, Pedro 157, Pepsi 65, White Cat
14) — but re-ID is effectively **raccoon-only** (crows, squirrels, juncos at 0% coverage).
There are **727 visits** and **~3,310 behaviour clips (~9.6h)**, with **26,744 appearance
embeddings** already computed. Species labels are **~90% unaudited** (32,800+ NULL
`species_verified`; only 3,436 confirmed, 15 rejected), and recurring nightly craters
(raccoon/day dropping to 22, 34, 9) are **camera-standby outages, not quiet wildlife**.

---

## 0. Lead finding: the behaviour goldmine is mined, then thrown on the floor

This is the single most consequential thing in the database, and it spans three failures that
compound:

| Layer | What exists | What's wrong | Number |
|---|---|---|---|
| **Extraction** | `clipmotion.py` turns clips into kinematic tracks | **Stopped on 2026-06-16T21:31** and never restarted | **~2,806 of 3,310 clips (85%) have zero tracks**; every day 06-17→06-28 = 0 |
| **Attribution** | 897 tracks with 11 rich kinematic columns (`avg_speed`, `straightness`, `moving_frac`, `area_trend`, `peak_speed`, `net_disp`…) | `individual_id` is **NULL on 100% of them** | 0 of 897 |
| **Gait** | `stride_hz` / `stride_strength` cadence | Computed on **7 tracks**; only 41 cleared `walk_s≥2.5` | 7 of 897 |

**Why it's frozen, mechanistically:** `backyard_cam.py` has **zero references** to
`clipmotion`/`extract_tracks`. The live rig only *records* clips; track extraction is a
hand-run batch that was last run on 06-16. It is resumable (only untracked clips are
processed), so the fix is one command plus a cron.

**Why it matters:** PLAN.md's signature feature is the **two-axis "looks like X but isn't
acting like X"** readout, where behaviour is the *strong* axis for raccoons (appearance is
weak through glass). The behaviour axis is starved at every level — the ore isn't mined
(06-16 cutoff), and even the 506 tracked clips are never tied to Notch or Stan. The
`unblend` UI that would label them is **fully wired** (`web.py:452/510/538 →
db.set_clip_track_individual`) and **has never been used once**.

> **Caveat before shipping any per-individual kinematic numbers** (e.g. "Notch avg_speed
> 0.104 vs Stan 0.083"): the track→named-visit attribution rests on a time-overlap heuristic
> the verifier *could not independently reproduce*, and cross-species speed comparisons are
> **size-confounded** (a raccoon at the door fills the frame; a crow farther out doesn't).
> Treat per-individual gait as *promising and unvalidated* until the backfill writes
> `individual_id` and you normalize speed by bbox size. **Arrival-hour separation is solid**
> and needs no kinematics: **Stan clusters 20–21h, Notch clusters 00–04h** — disjoint
> windows, a free temporal prior for re-ID.

---

## 1. What could we do better? (UX, ML quality, correctness)

### UX — the daily loop opens on the wrong thing

- **The first screen leads with junk.** The Live tab (default landing) renders "Rarely Seen
  top 3" from `species_overview` (`stats.py:286`), which — unlike Calendar and Dispatch —
  **does not apply the `_NON_CRITTER` denylist**. The three "rare visitors" a guest sees are
  literally **chair, bricks, porch**. Real rarities (Steller's jay 6, Townsend's chipmunk 6,
  bushtit 12) are buried, and the Specimen Catalogue lists all 9 junk labels as species.
  **One-line server fix**, identical to the filter `period_digest` already uses at
  `stats.py:635`. This is the worst first-impression bug for a share-with-friends site.
- **The app boots into a livestream of an empty yard.** Activity peaks 21:00–02:00; the
  default Live tab during daylight shows a dead feed and camera sliders, not "who visited
  overnight." The **Dispatch** is the perfect daily artifact (`stats.py:600` already computes
  novel/quiet flags, plate-of-the-night, highlight reel) but it's the *second* tab. Land on
  the Dispatch, or banner "Last night: 12 raccoon visits, 1 new →" atop Live.
- **No "who's back / who's overdue."** Every named animal has a precise last-seen (Stan
  06-27, Notch 06-26 = 2d ago, Miss B. 06-23 = 5d ago) but the Individuals tab shows none of
  it. For someone who *names raccoons*, "Notch hasn't been seen in 2 days" is the line that
  makes them open the app daily. Data is one `MAX(timestamp)` per individual.
- **No per-individual diary, no Clips library, no notifications.** `visits.individual_id`
  already attributes 48 visits to Notch, 27 to Stan — but there's no profile page, no
  chronological timeline, and no browsable clip library for 9.6h of footage (clips are only
  reachable per-visit). And every interesting signal (new species, a regular returning) is
  render-time only — no push, no tab badge. The compelling moments are missed unless you
  happen to be looking.

### ML quality / correctness

- **`species_confidence` is useless for triage.** BioCLIP's forced-choice softmax is
  systematically overconfident: of 2,092 raccoons the *detector* barely saw (det conf <0.4),
  **98% still got species_confidence >0.9**; 94% of all unverified bioclip labels sit ≥0.9. A
  "review the low-confidence ones" queue surfaces only **648 rows** and misses the real
  errors. Rank the review queue by **detector confidence** (weak but real signal, r=0.25) ×
  **CLIP non-animal margin** × **circadian-implausible label** instead.
- **The long tail is contaminated by circadian-impossible misIDs.** Diurnal birds
  first-detected deep at night — varied thrush 06-07 **21:24**, white-crowned sparrow 06-09
  **01:44**, eastern cottontail 06-11 **01:18** — are almost certainly nocturnal mammals
  misclassified. A cheap **circadian-plausibility prior** would flag these before they
  pollute richness/novelty.
- **`domestic dog` (177) and `brown rat` (206) are NOT denylisted.** Both flow into richness,
  the catalogue, and species cards unsuppressed and 100% unverified. Dog is **entirely
  diurnal** (0 rows 21:00–05:59, peaks 9–10am and 4–6pm) → real dogs or daytime confusion,
  *not* nocturnal-raccoon misID. Rat is **bimodal** (9am food-plate false-fires + 3am/10pm
  real rats) with the **lowest mean species_confidence of any common species (0.557)**. These
  ~380 crops are the **highest-value manual-review target** in the whole DB.
- **The two-axis verdict fires on arrival-hour alone.** `twoaxis.py:75` sets `verdict='FITS'
  if arrival_ok else 'DISAGREES'` — `dwell_ok` is computed (line 72) and discarded. A
  "raccoon" that shows up at a normal hour but bolts in 3s reads as FITS. Fold the
  already-computed dwell (and motion, once tracks are backfilled) into the verdict.
- **Roster contamination.** "Pepsi" is **65 detections, 100% `domestic dog`**, diurnal
  arrivals — a dog in the raccoon roster. Nothing ties an `individual_id` to a species. Even
  Notch carries **44 `not an animal`** frames alongside his 5,211 raccoon crops. Per-individual
  aggregates built from `detections.individual_id` inherit both — standardize them on confirmed
  **solo visits** instead.

### Correctness / data hygiene

- **Duplicate boxes inflate every count.** ~**1,426 within-frame pairs at IoU>0.5** are
  near-identical boxes of *one* animal — MegaDetector emits two boxes, `backyard_cam.py:681`
  inserts one row per box with no NMS. So ~4% of all detections (and ~7% of raccoons) are
  double-counts that bias raccoon share, diversity indices, *and* prototype weighting. Add
  greedy IoU>0.6 dedup at insert.
- **Junk at the visit level is unaudited.** 34 of 727 visits (4.7%) are non-critter,
  including a **36-minute, 128-detection "not an animal" visit** (a stuck glass-door
  reflection). Any "switch headline metrics to visits" change must apply `_NON_CRITTER` at the
  visit layer, or a reflection becomes the longest-dwelling "visitor" of its night.
- **Perf landmines that scale with the DB.** `visits_page` (`stats.py:407`) `SELECT`s the
  *entire* detection table with no LIMIT and recomputes visits in Python on every request;
  `compute_stats` does the same on the 6s `/api/stats` poll. Serve from the materialized
  `visits` table with `LIMIT/OFFSET` and cache the poll.

---

## 2. Latent-signal inventory — data we have but don't surface

Everything below is **already computed and stored**; it just never reaches a live surface.

| Signal | Where it lives | Coverage | Where it should appear |
|---|---|---|---|
| **Per-track kinematics** (speed, straightness, moving_frac, path_len, net_disp, peak_speed) | `clip_tracks.*` | 845/897 rows | Per-visit "motion strip" via a new `/api/visit/<id>/motion`; per-individual fingerprint |
| **Approach/retreat** | `clip_tracks.area_trend` | **213 approach (>1.15), 202 retreat (<0.85)** | Per-visit glyph + digest line ("a raccoon walked right up to the door at 1am") |
| **Track → individual link** | `clip_tracks.individual_id` | **0/897 (empty)** | Backfill from solo-visit overlap → unlocks behaviour axis + motion re-ID |
| **Appearance map** (PCA-2D + cosine matrix) | code exists in `makingof_export.py:341` (frozen) | 24,949 distinct embeddings | Live `/api/appearance` + Explorer tab (the "wow" artifact, currently public-page-only) |
| **SAM `-seg` embeddings** | `detection_embeddings` model `*-seg` | **1,795, all dual-embedded** | A *controlled paired A/B* sitting unrun: within- vs between-individual cosine separation seg-vs-plain, then wire in or prune |
| **Last-seen / return interval** | `MAX(timestamp)` per `individual_id` | all 7 named | "Cast roll call" with overdue chips |
| **Per-individual visit history** | `visits.individual_id` | Notch 48, Stan 27, Pedro 5… | Per-animal dossier / timeline |
| **crop_quality** | `detections.crop_quality` | **36,228 scored** | "Sharpest crops of Stan" sort + quality pip; never shown as a number |
| **Co-occurrence** | `behavior.co_occurrence` | **318/727 visits (44%) multi-species** | Force-directed graph split diurnal/nocturnal; raccoon+opossum 31, crow+starling 57 |
| **Pair-clips for un-blend** | clips with ≥2 sustained tracks | **78 clips** | Explicit "tell these two apart" queue (the only path to template Elliot) |
| **Auto-clusters** | `individual_source='cluster'` | 63 ids / 521 dets, **0 promoted** | Top-of-queue naming candidates (c01=75, c04=73, c02=63) via `individuals.py --refit` |
| **Species accumulation curve** | derivable | saturates day 3 (25/29 by 06-09) | Biodiversity chart — but **reframe novelty as "rarely-seen returned"**, not "first-ever" |
| **`frame_path`** | schema column | **0/36,300 (dead)** | Drop it, or populate for "see the whole scene" |

**Degenerate / don't-bother (measured negatives, worth recording):** entry-direction from the
door cam is near-uniformly center-frame (raccoon 101/133, crow 122/166 center) —
**spatial/approach-path analytics are not extractable from one door cam** (and this is the
strongest data-grounded argument for a second camera). Twilight-*drift* charts are noise over a
21-day solstice window. `gait` at 3 of 7 values pinned to the 3.2 Hz ceiling looks like
**stride-harmonic aliasing**, not real cadence.

---

## 3. How to improve the observatory — prioritized roadmap

| # | Move | Value | Effort | Tied to |
|---|---|---|---|---|
| A | **Re-run + cron `clipmotion.py`** | Very high | Low | Unfreezes 12 days / ~2,806 clips of behaviour |
| B | **Backfill `clip_tracks.individual_id`** from solo-visit overlap | Very high | Low | 0/897 → real motion fingerprints; feeds two-axis |
| C | **Filter `_NON_CRITTER` in `species_overview`** | High | Trivial | Kills chair/bricks/porch on the landing screen |
| D | **Land on the Dispatch + "Cast roll call" with last-seen/overdue** | High | Low | "Notch: 2 days ago" |
| E | **Manual-review the 380 dog+rat + ~720 non-animal crops** | High | Low | Audits nearly all suspect labels in minutes |
| F | **Per-individual dossier / character cards** | High | Medium | Consumes `visits`+`behavior`+`clips` already computed |
| G | **One-tap arrival push → existing `/api/live/sighting`** | High | Medium | 5 live_sightings in 21 days; fuels re-ID |
| H | **Weather/temperature enrichment** (Open-Meteo, lat/lon on file) | High | Medium | Explains 60× daily swings; third covariate |
| I | **Observatory-health / uptime strip** | High | Low | Stops "first raccoon in 3 days" lying when the rig was asleep |
| J | **iNaturalist/eBird export of `species_verified=1`** | Medium | Medium | Gives the 3,436 verified rows a second life |
| K | **Second camera** (config already supports it) | Medium | Low | Breaks the pose/glass confound capping similarity at ~0.5 |
| L | **Gamified swipe-naming + verify fast-lane** | High | Medium | Drains 9,693 unnamed raccoons + 32,800+ unverified |

Keep appearance and behaviour on **separate axes** throughout (per project philosophy): the
win is surfacing *both* next to each other ("looks like Notch, but arrived at 2pm and bee-lined
the door"), never collapsing to one oracle answer.

---

## If you do five things

1. **`python clipmotion.py` now, then cron it nightly + a "tracks N days behind" badge.**
   Unfreezes **~2,806 of 3,310 clips** and 12 days of behaviour. The whole Phase-4 arm has
   been on a 5-day snapshot for two weeks. *(Value: very high / Effort: low.)*
2. **Backfill `clip_tracks.individual_id`** from each track's solo-visit overlap. Turns
   **0/897** linked tracks into per-individual motion fingerprints and finally feeds the
   two-axis readout the behaviour data it was designed around. *(Very high / low.)*
3. **Add the `_NON_CRITTER` filter to `species_overview` (`stats.py:286`).** One line removes
   **chair/bricks/porch** from the "Rarely Seen" cards and the catalogue — the worst first
   impression for a site you're sharing. *(High / trivial.)*
4. **Make the daily loop open on last night:** land on the Dispatch and add a "Cast roll call"
   with last-seen + overdue chips (Stan 06-27, Notch 06-26, Miss B. 06-23). All data is
   `MAX(timestamp)` per individual. *(High / low.)*
5. **Hand-review the ~1,100 suspect crops** — 206 brown rat (mean conf **0.557**), 177
   all-diurnal domestic dog, ~720 "not an animal." A few minutes of clicking audits nearly
   every shaky label and decides the dog/rat denylist question on evidence. *(High / low.)*

**Relevant files:** `stats.py` (286 species_overview, 407 visits_page, 600 period_digest),
`twoaxis.py` (75 verdict), `clipmotion.py`, `backyard_cam.py` (673/681 insert, no clipmotion
ref), `web.py` (452/510/538 unblend), `dashboard.js` (252 rarely-seen, 1072 indivRow),
`individuals.py`, `makingof_export.py` (341 appearance map).
