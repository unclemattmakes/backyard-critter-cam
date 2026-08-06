# Backyard Critter Cam — Why Automatic Naming Isn't Working, and What To Do About It

> Evaluation date: 2026-08-05. Scope: the whole detection → species → individual-identification
> stack, measured against the live `backyard.db` (127,021 detections, 2,416 visits, 2026-06-07 to
> 2026-08-05). Every number below was computed from the database or read out of the code at
> `file:line`. Claims that did not reproduce under a second, adversarial pass were dropped, and
> the ones that survived but were *overstated* are marked with their corrected scope.

---

## The one-paragraph answer

The appearance signal that separates one raccoon from another **decays to nothing in about a
week**, and the entire system is built as though it were stable. Leave-one-visit-out top-1
identification is 0.818 when a probe may match a template from the same night, 0.482 when the
nearest allowed template is 7 days old, and 0.222 at 21 days — against a 0.348 majority-class
baseline. Meanwhile the newest confirmed template is 41 days old for Notch and 29 days old for
Stan. The model is being asked to do the one thing this measurement says it cannot. The fix is
not a better backbone; it is a faster human loop, a validated operating point, and honest
measurement — in that order.

---

## Part 1 — The funnel

Where 598 raccoon visits go:

```
598 raccoon visits
├── 134 trail_cam_sd ─────────────── structurally unnameable (see §4). 0 ever named.
└── 464 glass_door_cam
    ├── 490 (both sources) have a usable prototype
    ├── 169 human-confirmed
    │   └── 138 usable solo templates ── Stan 48 · Notch 46 · Pedro 36 · Elliot 4 · CutiePie 3 · The Dude 1
    ├── 122 excluded as multi-animal (unconfirmed)
    └── 201 addressable candidates for the auto tier
        └── 0 assigned at the shipped operating point
```

And the label ledger, over detections:

| source | detections | share |
|---|---|---|
| `human` | 21,110 | 96% |
| `cluster` | 521 | 2% |
| `auto` | 425 | 2% |

**The automation contributes 2% of the labels.** That is the problem stated as a number.

---

## Part 2 — The five causes, ranked by contribution

### 1. Appearance identity decays within a week (dominant, and not a bug)

Measured by leave-one-visit-out over the 138 human-confirmed solo templates, with an *embargo*:
a minimum enforced time gap between a probe visit and any template it may match against.

| embargo | top-1 |
|---|---|
| none | 0.818 |
| 1 day | 0.708 |
| 3 days | 0.628 |
| 7 days | 0.482 |
| 14 days | 0.350 |
| 21 days | 0.222 |
| *(majority class)* | *0.348* |

The same story from the other direction: probes whose winning template is **less than a day old
are 92.5% correct**; probes whose nearest template is **14+ days old are 33.3% correct** — below
chance-by-majority.

MegaDescriptor, running on a near-uniform agouti animal at 90–400 px, through glass, under IR at
night, is largely encoding *this animal, this week, in this light* — not *this animal*.
Crop-level cross-night retrieval makes it explicit: **mAP 0.317 against a 0.271 chance rate.** A
single crop carries almost no cross-session identity. The system's entire accuracy is an
averaging effect over a visit's crops, and averaging cannot manufacture a signal that is not in
the crops.

### 2. The review queue starves the only thing that works

Human confirmations are the *sole* source of templates (`db.confirmed_visit_labels` is human-only
by design), and templates are what the matcher stands on. But `web.py:921-953` builds the queue as
`ORDER BY started_at DESC LIMIT ?`, and `dashboard.js:1169` calls it with no params, so the limit
defaults to 30.

Of the **208 addressable candidates, 6 are reachable.** Of the 7 visits the auto tier has ever
named, **1 is reviewable** — the other 6 wear a machine-written name nobody can confirm or reject.

Raising the limit does not fix it: the cap is already a parameter maxing at 100 (`web.py:585`),
and a top-100 *recent* window still reaches only 26 of 208. The fix is a filter mode, not a
bigger number.

### 3. The auto tier's operating point is stale

`config_local.py` ships 0.76/0.12. Replaying `individuals.VisitMatcher.auto_assign(dry_run=True)`
against the live DB:

```
{'assigned': [], 'skipped': {'confirmed': 160, 'multi_animal': 122,
                             'below_threshold': 141, 'ambiguous': 60, 'already_auto': 7}}
```

Sixty candidates clear the similarity bar and **not one clears the margin.** The bars were swept
when the cast was two raccoons; the cast is now three-plus and the near-ties got closer.
Re-swept on today's corpus, **0.88/0.02 names 16 visits (2,817 detections): Notch 8, Stan 4,
Pedro 3, Elliot 1.**

> **Corrected scope.** An earlier draft of this evaluation attached an 11.9% wrong-name rate to
> that recommendation. That figure is the error of the *sweep procedure* under 5-fold — averaged
> over whatever point each fold selects, most of them looser than 0.88/0.02. Evaluated directly
> at the recommended point, session-blocked: **20 of 138 probes assigned, 0 wrong** (unblocked:
> 23 of 138, 0 wrong). The honest statement is 0/20 observed, ~15% upper bound by rule-of-three.
> This makes Phase A materially safer than first written, which is why it can ship on its own.

### 4. Every number the project steers by is measured with the session leaked in

`eval.py:442-496` (`_auto_assign_sweep`) selects its operating point on the same probes it scores,
and lets a probe match its own same-night twin. Blocking same-night same-individual templates
drops LOO top-1 from **0.812 to 0.739** on the identical probe set.

Nothing re-runs the harness. `reports/` was last written `eval_20260718T030508Z.json`; `eval.py`
appears in no `.bat` and no scheduled task; and the `--baseline` regression check its docstring
promises (`eval.py:10`) does not exist — `grep -n baseline eval.py` returns that one docstring
line. The documented AUC slide 0.81 → 0.635 → 0.617 accumulated unremarked, and some of it is the
leak's benefit shrinking as sessions got busier, not a real regression.

### 5. The labelled corpus is three animals

Stan 48, Notch 46, Pedro 36 — then Elliot 4, CutiePie 3, The Dude 1. With three effective classes
and 138 usable visits, the bootstrap CI on most AUC deltas is wider than the deltas. Anything that
looks like it buys 0.01 is fitting noise.

And **the Notch↔Pedro question is still open.** They have never been co-observed; Notch's labelled
crops stop dead on 2026-06-30; Pedro's own cross-night self-similarity (0.324) is *lower* than his
similarity to Notch (0.372). If they are one animal, 46 of 138 templates are mislabelled and every
number above moves. Note that **8 of the 16 names the new operating point would write tonight are
Notch.**

---

## What is *not* the cause

Each of these was proposed, measured, and killed:

- **`crop_quality` selection.** Top-40 by quality is statistically indistinguishable from
  *bottom*-40 (bootstrap CI on the delta `[-0.010, +0.011]`). Keep it for thumbnails; stop
  believing it helps re-ID. (Deleting the selector entirely *does* hurt — safe auto coverage
  collapses 20 → 4 probes — so leave it in place.)
- **Embedding coverage.** 99.8% of above-gate raccoon crops have a vector.
- **Static furniture.** Removing 4,805 junk rows changes exactly one visit's naming outcome.
- **Gait, box size, arrival hour.** Nine motion features give LOO balanced accuracy 0.422 vs 0.333
  chance (p=0.107); depth-normalized size separates the three adults at Kruskal-Wallis p=0.59;
  arrival hour is a real *group* difference but points at the wrong individual on 52.2% of visits.
  `stride_hz` exists on 27 of 21,860 tracks and its values pile at the band edges — an estimator
  picking noise. (Also forbidden by the two-axis principle: behaviour may raise a flag, never
  change the ranking.)
- **The trail cam as a threshold problem.** It is a domain problem — see below.

---

## §4 — The trail cam is a separate universe, and that is a live hazard

Glass-door prototypes have real temporal structure: median similarity 0.647 for visit pairs under
an hour apart, 0.223 for pairs a day-plus apart. Trail-cam prototypes: **0.510 vs 0.514. Flat.**
No identity structure to threshold. Cross-source, all 93 trail-cam raccoon prototypes score a
median 0.249 and a **maximum 0.363** against every glass-door template.

The hazard: **83.7% of trail-cam visit pairs already exceed the 0.31 novelty cut.** The moment one
trail-cam visit is confirmed, `refit()` would propose a median of 83 of the other 92 under that
single name. That is a mass-mislabel one click away, and the queue changes below make that click
more likely.

This also corrupts the north-star metric. `auto_coverage` over all 598 raccoon visits includes 134
trail-cam visits that can never be named — and in August the trail cam has already produced *more*
raccoon visits (19) than the glass door (13). **The denominator must be the 464 glass-door raccoon
visits, or be reported per-source.**

---

## Part 3 — The plan

Ordered by measured gain per hour of work. Every phase carries a metric computable against
`backyard.db` under one protocol: **leave-one-visit-out, session-blocked** (a probe may not match a
template from the same night, night key = timestamp shifted −12h), **operating point selected on
training folds only.**

### Phase 0 — Answer the Notch/Pedro question, and start timestamping labels (hours)

This is step zero because 46 of 138 templates and half of tonight's proposed auto-names ride on it,
and because the instrument for answering it is the same one that bounds every other number here.

Every accuracy figure in this document is optimistic by an unknown amount, because the confirmed
corpus was built by a human *agreeing with matches the embedding proposed*. There is currently no
way to size that bias — there is no `labelled_at` column anywhere (verified across `detections`,
`visits`, `live_sightings`: every timestamp is an observation time, not a labelling time).

But a model-independent held-out set already exists and is unexploited. The Live tab
(`dashboard.js:397-433`) logs sightings from name chips **with no suggestion and no ranking
shown**, and `db.record_live_sighting` stamps solo spans through `apply_visit_label`. Measured: 36
confirmed visits overlap a solo live-sighting span, 14 of them are usable templates, 29 agree with
the stored label and **7 disagree.**

1. Add a nullable `labelled_at` column to `detections` (additive, per the project's schema rule)
   and write it from `db.label_visit`.
2. Run a blind pass: surface N visits with the suggestion hidden, record the answer *and the
   time*. This yields three things nothing else can — a human self-agreement rate (the Bristol red
   fox study's 99% figure, which the ceiling below leans on, has never been measured here), a first
   bound on the confirmation bias, and the Notch/Pedro answer.
3. Notch/Pedro is a human question, not a model question. Sighting id 40 shows a "Notch" logged and
   immediately corrected to "Elliot" — the label history is worth reading before deciding.

**Metric:** `labelled_at` non-null on all new labels; blind-pass agreement rate reported with n.

### Phase A — Re-open the auto tier at a validated point (minutes, then half a day)

Split this, because the naming gain and the harness work are independent and one is 100× cheaper:

**A1 (minutes).** Set `reid_auto_threshold` / `reid_auto_margin` in `config_local.py` to
**0.88/0.02**. Leave `config.py`'s shipped defaults disabled — the operating point is a property of
*this* corpus, camera and cast, and must not travel with a public repo.

**A2 (hours, do before A1 goes nightly).** Add a per-individual **template floor** to
`individuals.auto_assign`: refuse to auto-write a name backed by fewer than N templates. This
document says nothing tuned on Elliot (4), CutiePie (3) or The Dude (1) is trustworthy — the tier
has *already* auto-named a CutiePie and a The Dude visit off a single template, and 0.88/0.02 would
name an Elliot. The guard implements the stated ceiling; without it the ceiling is just prose.

**A3 (half a day, no user-visible effect).** Fix the measurement:
- Add session blocking to `eval.py:_auto_assign_sweep` and the LOO reporting. Print blocked and
  unblocked side by side — the gap (0.739 vs 0.812) is the cleanest available measure of how much
  "identity" is really session signal.
- Replace the in-sample sweep with repeated 5-fold: select on training folds, score the held-out
  fold, report both the recommendation and its held-out coverage/error.
- Implement the `--baseline` flag `eval.py:10` already advertises. `_save_artifact` (`eval.py:623`)
  writes the artifact; nothing reads it.
- Add a **report-only** eval step to `run_clipmotion.bat`, before the auto-assign step. It must not
  change any config.
- Have `auto_assign` log a WARNING when the configured point drifts from the artifact's
  recommendation — a warning, never a self-adopt (see Rejected).
- Update the re-ID comment block in `config.py` (still citing 2026-07-18 numbers) and add the
  crop-level line, so the next reader knows a single crop is worth nearly nothing.

**Gain:** +16 auto-named visits immediately (2,817 detections), auto tier 425 → ~3,200 labelled
detections. **Metric:** `auto_coverage` = distinct auto-named **glass-door** raccoon visits ÷ 464.
Today 7/464 = 1.5%; target ≥ 20/464 = 4.3% after the first nightly run.

### Phase B — Unthrottle the review queue (hours to 2 days)

The queue is the multiplier on everything else; it creates no labels itself.

1. In `web.py:_reid_queue`, add a **mode** parameter over the whole species pool: `recent`
   (today's behaviour, keep as default), `unreviewed_auto`, `ambiguous` (clears threshold but not
   margin — 60 visits today, the highest-information clicks available), and `stale` (recent visits
   whose top candidate's newest human template is older than N days). Paginate rather than raising
   the limit. All four are computed from stored columns and survive a camera move.
2. Add a **template-freshness** panel to the cast block (`web.py:~985` already computes
   `n_visits`/`n_auto`/`last_seen`): days since newest template per individual. Today: Notch 41,
   Stan 29, CutiePie 19, Elliot 18, Pedro 5, The Dude 3. Per the embargo curve, this panel *is* the
   priority list.
3. Publish the funnel from Part 1 on the re-ID panel, so "automation contributes nothing" becomes a
   specific, visible diagnosis.
4. Add a temporal-context chip ("started 6 min after the visit you named Stan") — **for the human
   only.** Adjacency must not enter the ranking (see Rejected).

> **Do not build:** keep/reject on auto-named visits. It already exists — `dashboard.js:1201-1202`
> renders ✓ keep / ✗ not them, and `web.py:763-781` accepts `reject`. Worth *testing*, though:
> `rejected_visit_ids` is empty, so the tombstone path has never once been exercised.

**Metric:** addressable candidates reachable in the queue, today 6, target ~200 (glass-door only —
51 of the 208 are trail-cam and Phase C's guard makes them unnameable by construction). Plus
`max(days_since_newest_template)` across the cast: today 41, target ≤ 14 for anyone seen this
month. Note that the obvious label-velocity metric is **not computable today** — grouping
`individual_source='human'` by week groups by *when the animal visited*, not when the label was
applied. It becomes computable once Phase 0 ships `labelled_at`.

### Phase C — Guardrails, before the label set gets contaminated (days)

None of these buys naming today. Each prevents a hard-to-undo corruption of the 21,110-detection
human label set — the project's only irreplaceable asset — and each becomes *likely* precisely as
Phases A and B increase labelling activity.

1. **Source guard** in `individuals.py`: refuse to rank a prototype against a template from a
   different source, and refuse to `auto_assign` or `refit` across sources. ~10 lines, no config,
   no calibration — a source-equality test survives a camera move, unlike any threshold. Costs
   nothing today (max cross-source similarity in the entire 397×93 matrix is 0.363) and defuses the
   mass-mislabel described in §4.
2. **Tell the truth about the trail cam** in the queue: show "no cross-camera match is possible",
   not a 0.249 top-1 and a "possibly someone new" badge. That badge currently reads as a claim
   about the animal when it is a claim about the camera.
3. **Fix `live_sightings`** (`db.record_live_sighting`, `db.py:866`) — 48 rows stamping 3,758 crops,
   the highest-value and noisiest ground truth in the DB. Supersede rather than duplicate when a
   span is re-logged. There are **five** conflicting span groups: (6,7), (8,9,10), (19,20),
   (25,26,27), (40,41,42). The clearest is 25/26/27 — "Clippy", then "Clippy Friend", then "Stan",
   over the *same two crops* within 33 seconds, each `apply_visit_label` overwriting the last, so
   the stored label is simply whatever was logged last. `sighting_multi` misses this because
   `multi_name_sighting_spans` only reads rows carrying 2+ names, and these arrived as three
   single-name rows.
4. **Bound the live solo stamp by the still tracklet**, not the visit span, using
   `individuals.still_tracklets` — the box-continuity reasoning that diagnosed the 2026-07-31 kit
   mis-stamp by hand. Keep the solo/pair asymmetry; it is deliberate and correct.
5. **Confidence-weight the species vote** (`visits.py:65-70` takes an unweighted crop-count mode).
   A ≥0.8-confidence-only vote flips 108 of 2,416 visits (4.5%), 12 of them *into* raccoon. A
   static artifact firing 800 low-confidence frames currently outvotes a real animal's 100
   high-confidence ones. Report the flipped set; do not apply it silently.
6. **Namespace the 63 `raccoon_cNN` cluster ids** (521 detections, every one spanning exactly one
   night). They contribute nothing, are excluded from templates, and inflate the apparent cast
   tenfold on every individual surface. Write clusters to their own column rather than the identity
   column everything else reads. This is the only item that mutates existing identity rows — route
   it through `label_visit(..., None)`, not a bulk UPDATE.

**Metric:** cross-source assignments = 0 by construction (assert it in a test); conflicting
`live_sightings` groups 5 → 0 unresolved; cast surface shows 10 real names + 0 `raccoon_cNN`, with
size reported as "individuals with ≥2 distinct nights and ≥3 embedded crops" (currently 6,
effectively 3).

### Phase D — Recover already-named visits locked out of the template set (1 day)

Two disjoint populations, cheapest first.

**D1 — the 22 already-named visits excluded from `templates()` because `is_multi` fired**
(Pedro 9, Stan 7, CutiePie 3, Elliot 2, Notch 1; 16 on the clips arm alone). These already carry a
human name. Un-blending them requires no new identification — only "which cluster is the one you
already named". That alone takes templates from 138 toward ~160, **doubles CutiePie (3→6) and adds
50% to Elliot (4→6)** — the two individuals this document calls unmeasurable. It is the single
cheapest template-set growth available.

**D2 — the 122 unconfirmed multi-animal visits.** The machine already exists:
`individuals.still_tracklets` + `unblend_visit_stills` (`individuals.py:148-224, 416+`, shipped
2026-08-02) split per-animal identity inside a visit using same-frame co-presence as a *hard*
cannot-link constraint, and work with no clips at all.

> **Corrected scope.** An earlier draft claimed the un-blend button is gated off for these visits.
> It is not — `dashboard.js:1230` gates on `v.multi`, which is true for all 122, so every one
> already shows the button. The blocker is simply that the labelling work is unstarted. Ungating it
> further is still worth doing, but it recovers a *different* population (visits that swallowed two
> animals **sequentially** and never tripped the flag), and must be justified on that basis.
> Relatedly: `is_multi` (`individuals.py:816-825`) is *not* simultaneity-only — it has a
> `sighting_multi` arm added for exactly the non-simultaneous case. Over the 122 it fires 32
> frames-only, 66 clips-only, 24 both, **0 sighting**. The arm exists; it has no data.

Before any labelling session, run the deferred sweep on `reid_track_match_threshold` (0.55, shipped
without one): on confirmed **solo** visits the splitter should produce exactly one cluster, so the
over-split rate at each threshold is a real specificity measurement needing no new labels.

**Metric:** `templates()` length, today 138, target > 155; splitter over-split rate on confirmed
solo visits, must stay near zero; `auto_coverage` re-measured after the new templates land.

**Risk:** this is the phase that can poison the label set. An over-split makes two templates for one
animal; an under-split makes a blended template that corrupts everything downstream. Sweep first,
treat the first ten as a hand-checked pilot. `clip_tracks.individual_source` contains only
`overlap` (347 auto-linked rows) and **zero** explicit un-blend labels — this feature has never
produced a single human label.

### Phase E — Learn a metric on the vectors already banked (weeks, uncertain)

First phase where modelling is justified; fifth in line because it is the most expensive and least
certain. It needs no re-embedding, no GPU batch job, no new dependency, and it improves as labels
accumulate — which B and D exist to make happen.

1. **Build the harness first**, reusing Phase A3's session-blocked held-out protocol. Nothing ships
   without beating the baseline on it: session-blocked LOO top-1 **0.739**. Report bootstrap CIs.
2. **Shrinkage whitening (WCCN)** over the frozen vectors — seconds of CPU, the cheapest supervised
   candidate. Measured 0.766 top-1 at λ=0.9 *unblocked*; re-measure blocked before believing it.
3. If WCCN clears the bar, an **ArcFace or triplet head** on the frozen features over the 21k
   labelled crops — minutes on the 5050. Split by **visit** and by **night**, never by crop; a
   crop-level split leaks the session and reports a fantasy.
4. **Global mean-centering**, honestly: under session blocking it moves zero-error coverage 20 → 21
   and leaves top-1 at 0.739. The impressive earlier figures were in-sample and leaked. If it ships
   it must re-sweep **all four** `reid_*` thresholds (centering collapses the similarity scale, mean
   0.342 → 0.040) and be applied consistently at `individuals.py:853` and `web.py:1089-1090`, or
   those two paths silently compare across spaces.
5. **MiewID as a second row** in `detection_embeddings` keyed by `model` — not a replacement. The
   schema was designed for this. Embed the confirmed corpus only (~22k crops) to answer the question
   cheaply; fuse by isotonic-calibrating each model onto a common probability scale.
6. **Two diagnostics that change what everything else is worth, run before any backbone work:**
   - *Day-vs-night cross-illumination retrieval.* 134 of 138 labelled visits start between 18:00 and
     08:00 — the stack has only ever been validated in the IR regime.
   - *Background-only retrieval.* Embed the crop with the animal box blanked and measure how well it
     still "identifies". **If background alone scores well above chance, every AUC in this document
     is inflated and the priority shifts from modelling to capture.**

**Metric:** session-blocked LOO top-1 (baseline 0.739), held-out coverage, held-out wrong-name rate.
Ships only if it beats the baseline on coverage at equal-or-lower error, with a bootstrap CI
excluding zero.

### Phase F — Speculative: viewpoint-conditioned prototypes (weeks, gated on E)

The one idea with a strong published precedent on an unpatterned mammal: the 2026 brown-bear work
(PoseSwin, 72,940 images / 109 bears) conditions the embedding on head pose and targets stable
facial geometry; GorillaWatch independently finds face beats body. The current pipeline averages a
rear view and a face-on view of the same animal into one prototype, ranked only by sharpness.
(`quality.py:42-46`'s night-eyeshine boost is already an accidental frontal-face proxy — a hint,
not evidence.)

Hand-tag a few hundred crops with front/side/rear/obscured, train a 4-class head on the
**already-stored** vectors (an afternoon of CPU, no new weights to pin), store the tag additively,
build viewpoint-conditioned prototypes, measure.

**Hard gate:** every previous attempt to stratify the template pool on this corpus *lost* accuracy —
day/night conditioning went 0.783 → 0.761 — because with 138 visits a narrower pool starves the
per-individual `max()` faster than nuisance removal buys anything. If illumination stratification is
still losing at that point, cancel this rather than attempt it. If the viewpoint head itself scores
below 80% held-out, stop there. If it does not clear the bar, record the number in a comment and
**do not iterate.**

---

## Part 4 — The ceiling, stated plainly

There is **no published deep-learning re-ID work on raccoons or any procyonid.** The ecological
literature still identifies them by clipping tail fur. The closest analogue — the Bristol red fox
study — had one trained human identify 168,417 of 170,923 records at 99% self-agreement, needed a
mean of **877 photos per fox**, used a *different* feature set for each individual, and concluded
that automated systems were unlikely to match it. State of the art for video re-ID of unpatterned
mammals (meerkats, polar bears) is **49–55% top-1**.

Matt's manual naming is not a stopgap that the model will eventually replace. On this class of
animal it is roughly the only thing that works.

**Achievable:** raise automatically-and-correctly-named glass-door raccoon visits from today's 1.5%
into the **5–15% range**, with a stated and measured error rate, on the three animals with enough
labelled visits to support a threshold. That is review-by-exception working as designed — the
machine takes the confident head of the distribution, the human keeps the rest.

**Not achievable on this data:** naming most visits automatically; naming an animal not confirmed in
the last week or two; naming anything on the trail cam from glass-door templates; naming Stan's kits
(there are **zero** labelled kit detections in the entire DB — the only kit ground truth is one
deliberately unstamped `live_sightings` row); or trusting any threshold tuned on Elliot, CutiePie or
The Dude.

---

## Part 5 — Rejected, with reasons

Ideas a reader would expect to see here, deliberately not proposed:

- **Fixing or replacing the `crop_quality` selector.** Measured dead — top-40 is indistinguishable
  from bottom-40. Keep it for thumbnails. Add a comment recording that the knob was measured; that
  is the whole appropriate action.
- **Self-calibrating `auto_assign`.** Reverses a deliberate, documented decision
  (`config.py:466-478`, `README.md:562-570`) that ships the tier disabled so a stranger cloning this
  repo does not get machine-written names off a four-visit corpus. The `staticfilter.py` analogy does
  not transfer: that self-calibrates a nuisance *rejector*; this self-calibrates something that
  **writes names.** Phase A3 uses the compatible version — a report step and a drift warning.
- **Lowe's ratio or a z-score instead of the absolute margin.** Measured: absolute 23 named probes,
  Lowe's 24, z-score 27. The claim that scale-free margins give a usable region where the absolute
  margin gives none is false — the region was always usable, the shipped constant just wasn't in it.
- **Propagating labels to temporally adjacent visits.** Under session blocking it fires 3 times and
  is wrong 3 times. The apparent gain unblocked (36 fires / 1 error) *is* the session leak — a
  same-night neighbour and a same-night template are the same confound. Context chip only.
- **A recency gate on auto-assign.** Implemented and measured: moves the wrong-name rate 0.119 →
  0.137 while cutting coverage 18.0% → 16.1%. Intuitive and measurably useless. Template staleness
  is real; the fix is confirming fresh templates, not filtering at match time.
- **Per-source thresholds, or hand-seeding trail-cam identities.** No structure to threshold (0.510
  vs 0.514, flat), and actively dangerous — see §4. Phase C ships the guard instead.
- **Gait, motion, size, arrival hour as identity signals.** All at or near chance; also forbidden by
  the two-axis principle. A test asserting match ordering is invariant to the motion features would
  be worth adding.
- **Swapping in DINOv3/DINOv2/CLIP as the primary embedder.** MegaDescriptor beats DINOv2 and CLIP
  on all 29 of its benchmark datasets, sometimes 66% vs 15%. DINOv3 also carries a bespoke license
  with a personal-information gate — a poor fit for a repo that has SHA-256-pinned every weight.
- **WildFusion's local keypoint matching (ALIKED/LightGlue/LoFTR).** Its +8.5 pp headline comes
  entirely from patterned species; its own authors note the matchers were trained on static objects.
  A raccoon at night through glass at ~397×324 px has no repeatable keypoints. Only the calibration
  half transfers, and that is folded into Phase E.
- **Adopting PytorchWildlife / wildlife-tools.** Already evaluated and rejected once
  (`README.md:99-112`, `docs/plan.md:218-222`) on dependency weight, on a machine that dies of commit
  exhaustion. Pull single self-contained components; do not rewrite the matcher onto a library.
- **Re-keying visits by (source, species), or moving `visit_gap_minutes` off 5.0.** Simulated: 7,625
  visits (3.2×), 42% singletons, 1,871 time-overlapping — destroys the timeline partition
  `behavior.py:75/187` and the dashboard depend on. Don't move the boundary; split inside it.
- **Running `staticfilter.py` on `glass_door_cam` at shipped defaults.** Simulated: deletes 14,771 of
  34,997 1080p rows (42%), including 6,738 raccoon rows and 484 already carrying a human
  `individual_id`. Its premise — nothing holds one box for 60+ minutes — is true for sparse trail-cam
  stills and false for a fixed-framing camera where raccoons re-occupy the same dish nightly.
  Measured naming benefit: +1 visit across two months.
- **Full-frame retention as a route to better naming.** Adds no labels, changes no embedding. The
  cited benefit (re-segmenting a merged visit from pixels) is already solved without pixels. Cheap
  insurance, but hygiene, not naming.
- **Fusing the clip channel into `auto_assign` as an OR arm.** Adds 22 assignments, 2 wrong (~9%),
  and both errors are exactly the near-ties the margin exists to reject — with the clip channel
  voting for the *wrong* name both times. It is not an independent view: `clip_track_embeddings` use
  the same MegaDescriptor model and the clip labels are attributed from the still confirmations.
  Notch and Elliot have no clip template at all.
- **Lowering `reid_co_presence_min` 3 → 1 for template eligibility.** Buys +0.013 AUC,
  indistinguishable from dropping 12 random Stan-heavy visits (p=0.12–0.25). If the badge is
  revisited, hand-label the 35 low-co-frame visits rather than moving the constant.
- **TTA and multi-model ensembling as a headline.** The AnimalCLEF26 ablation puts ensembling at
  +0.002 ARI against ~+0.44 from preprocessing and clustering. Flip-TTA is ~10 lines and free —
  add it opportunistically inside Phase E, reorder nothing around it.

---

## Method note

Produced by a multi-agent evaluation: eight parallel investigations (detection pipeline, re-ID
algorithm, data ground truth, an embeddings benchmark, species stage + eval harness, the behaviour
axis, project history, and a literature sweep), 110 findings, adversarial verification of the 14
critical/high ones (8 held, 5 refuted), then synthesis and an independent critique pass that
reproduced the numeric spine against the live DB and corrected eight errors in the first draft —
including the misattributed 11.9% error rate, the un-blend gating claim, and the north-star
denominator. All database access was read-only. The five refuted findings are recorded here so they
are not resurrected:

1. "Capture conditions outweigh identity inside the crop, and the detection stage records nothing
   that would let re-ID normalize them."
2. "Visits are cut at a fixed 5-minute, source-wide, species-blind gap, and that cut lands in the
   middle of the data."
3. "The multi-animal badge is simultaneity-only and its 3-frame threshold leaks blended visits into
   the template set."
4. "`rank_templates`' max-over-visits is biased by template count: individuals with few templates
   have 0% recall."
5. "`auto_assign` ignores three signals the same module already computes."
