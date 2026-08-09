# The refimg shadow review, run 2026-08-09 — and it was not inert

`docs/deferred-work.md` §1.4 recorded, on the morning of 2026-08-09, that the reference-image veto
had flagged **nothing** after a full night live, and reframed the review question from "are these
flags furniture?" to "why does motion-mask coverage never reach the bar?".

Re-measured against the live database the same afternoon, **both halves of that need correcting**:

- **The veto has flagged 18 rows, and all 18 are furniture.** They landed at 01:44–01:48 on 08-09,
  after §1.4 was written. The original review question was answerable after all, and its answer is
  the good one.
- **Coverage really is the binding gate** — 94.9% of every box judged in the shadow window died on
  it — but **0.9 is not the wrong bar**, and the accumulation policy is wrong in a *smaller* way
  than §1.4 assumed. The thing that actually cripples the veto turned out to be a different bug in
  a different gate: **the rig calls sunrise a camera reposition.**

Everything below is measured against `backyard.db` read-only while the rig kept writing, plus the
banked PNGs in `refimg_store/` and the recorded clips. Scratch scripts only; nothing wrote a row.

---

## 1. What the veto did

| | |
|---|---|
| detections, all sources | 138,561 |
| `suppressed_at IS NOT NULL` | **18**, all `suppressed_by = 'refimg_veto'` |
| `reference_images` | 161 (day 109, night 52; every one `certified+motion_masked`) |
| `view_epochs` | 1 row — and it is a false one, see §4 |
| glass-door detections since the first reference | 5,382 (08-07 21:21 → 08-09 11:39) |

`python refimg.py --review --days 7` renders all 18 into one cluster. **Every crop was opened.**
All 18 are the same static scene — retaining wall, shrub, a tipped white bin — at 01:44:47 through
01:48:10, box ≈ (1478, 589)–(1885, 1025) of 1920×1080. No animal is in any of them. BioCLIP labels
all 18 "raccoon", which is exactly the failure the veto exists to remove: the classifier is
organism-only, so furniture always comes back as an animal.

Their scores are nowhere near the line — `lum` 1.42–1.89 against 11.406, `dssim` 0.006–0.020
against 0.2346, `sobel` 0.061–0.087 against 0.4854 — with recurrence at 13 firings over 3 days and
cover 0.936–0.944.

**What that one spot is worth.** Boxes matching that cluster at IoU ≥ 0.60 number **435 of the
5,406** detections in the shadow window (8.0%) and **1,029 since the 1080p era began on
2026-07-22**, labelled 601 raccoon / 271 Virginia opossum / 65 eastern gray squirrel. A 40-crop
sample spread across that whole era is 35 furniture, one cat's face, and four frames of a person's
legs (labelled "eastern gray squirrel"). So the target is real and large — and the fact that a cat
and a person share the spot is a local, unplanned confirmation of design §4.2: **recurrence alone
would have erased them, and only the pixel test saves them.**

## 2. Coverage: the bar is right, the accumulation is not (but only by 1.37×)

Cover fraction of the banked masks, whole frame:

| illumination | n | median | mean | ≥ 0.9 |
|---|---|---|---|---|
| day | 109 | **0.000** | 0.074 | 1 |
| night | 52 | 0.178 | 0.323 | 4 |

Cover fraction **at the 5,382 real detection boxes**, against the reference in force:

| | n | median | p90 | max | ≥ 0.9 |
|---|---|---|---|---|---|
| day | 1,577 | 0.000 | 0.057 | **0.825** | **0 (0.0%)** |
| night | 3,805 | 0.000 | 0.781 | 1.000 | 274 (7.2%) |

Which gate stops each box: **`reference_has_no_pixels_here` 5,108 (94.9%)**, reaches the pixel test
274 (5.1%). No box was ever refused for age (median reference age 343 s against a 7,200 s limit).

**No daytime box has ever cleared the coverage bar, and the highest one ever recorded is 0.825.**
On this camera the veto is a night instrument. That is consistent with the design, whose races were
all nocturnal and which lists the daytime thresholds as unexercised (§8 item 5).

### Why the mask saturates

Overlaying the cover masks on their own references shows it: dawn-hour day references are red edge
to edge (cover exactly 0.000), and the rest are stair-stepped rectangles growing into full-width
bands. Three things compound:

1. **The veto only ever sees frames that already have motion.** `backyard_cam` runs the detector —
   and therefore `shadow.observe` — under `if motion and ...`, so every frame that reaches the
   coverage channel carries a blob ≥ `motion_min_area` by construction. The hourly detector census
   shows 300–1,546 such frames per hour.
2. **One global photometric event disowns the whole frame for an hour.** A dawn ramp, a cloud, an
   auto-exposure or auto-WB re-latch makes MOG2 return the entire frame as foreground; one such
   frame zeroes the cover mask for the full `refimg_no_update_s`.
3. **A detector box is, for anything that moves, exactly the pixels that just moved.** This part is
   the gate *working*: animals abstain because animals move. The thin part is that the furniture's
   own pixels get disowned too whenever anything passes near it within the hour.

### The bounding-box amplification is real and is not the explanation

`_blobs` remembered `cv2.boundingRect()` of each blob, which disowns pixels nothing ever moved
over. Measured by re-running the rig's own `MotionGate` over the **13,438 motion-positive frames**
of 2026-08-09 00:00–05:30:

| per frame, pixels disowned (of 57,600) | median | p90 | mean |
|---|---|---|---|
| bounding boxes (as shipped) | 850 | 3,119 | 1,547 |
| filled contours | 631 | 2,085 | 1,099 |

Amplification **median 1.33, mean 1.37, p90 1.53** — not the 10× that would have explained a 95%
abstention rate. Accumulated over that night the cover fraction improves from e.g. 0.230 → 0.311
and 0.618 → 0.689, and **at real detection boxes it moves the median from 0.000 to 0.009 with 0.0%
reaching the bar under either policy.** It is worth fixing as correctness, and it does not unblock
anything. Say so plainly rather than shipping it as a solution.

There *is* a second, non-safety reason to fix it: the rectangle list was re-drawn in full on every
frame, on the capture thread. Measured on this CPU: **0.79 ms at 500 remembered rectangles, 4.89 ms
at 2,500, 16.7 ms at 10,000** — against the 7.6 ms the whole per-frame veto was budgeted at.

## 3. Would a lower bar help? No — and it spends the only guard on nothing

Every one of the 5,402 glass-door detections in the shadow window was replayed against the exact
banked reference that was in force, with the coverage bar swept. The first attempt at this was
**wrong and is worth recording**: it pulled the clip frame nearest the detection's timestamp, but
clips prepend a pre-roll ring and their stored `fps` is the loop's own estimate, so `index / fps`
lands on the wrong instant. Drawing the box on the frame showed it sitting on empty wall — and
scoring empty wall against an empty reference "matches", which manufactured 29 raccoon
suppressions that never happened. The corrected harness pastes each detection's **own crop** back
into the frame at the exact rect `save_crop` cut it from, so every pixel scored is a pixel the rig
actually judged. It reproduces all 18 live suppressions exactly.

| coverage bar | suppressions | what the additions are |
|---|---|---|
| 0.90 (shipped) | 26 | the watering can |
| 0.70 | 34 | the watering can |
| 0.50 | 68 | the watering can |
| **off entirely** | **117** | **the watering can — every single one** |

All 117 crops were opened. **Not one animal, at any bar.** The conjunction below is what does the
work: with coverage off, 4,927 boxes are still kept by `pixels_differ_from_empty` and **294 more —
5.4% — pass all three pixel metrics and are saved ONLY by recurrence**, closely matching the
design's 9.1%.

So lowering the bar is not *dangerous on this corpus*. It is *pointless*: it buys more copies of one
object already caught, while spending the one gate that answers the design's own photographed
failure — a properly certified reference with an undetected raccoon walking the wall in it (§4.3).
Thirty-eight hours without that failure recurring is not evidence that a rare, unrecoverable event
does not happen. **`COVER_MIN_FRACTION` stays at 0.9.**

> The replay is slightly *more* permissive than the rig (26 vs 18 at the same bar) because it uses
> the banked cover PNGs, which are up to `PERSIST_EVERY_S` = 600 s staler than the live mask. That
> is the conservative direction for a safety test, and the 8 extra are all the same can.

## 4. The bug that actually cripples it: sunrise reads as a camera move

`view_epochs` holds one row: **glass_door_cam epoch 1 at 2026-08-09T05:56:52, `edge_fp_corr`,
corr 0.261.** The camera did not move.

- The two day references four minutes either side of that bump — 05:52:50 (epoch 0) and 05:57:09
  (epoch 1) — correlate **0.987**, and side by side they are the same wall, same pots, same tipped
  bin, same shrub.
- The template the bump was measured against was the last frame of the previous evening, 21:03:03,
  four minutes before dusk at the camera. It correlates **0.259** with the dawn frame.
- Dawn at this camera (`stats._sun`, from the yard's longitude) was 05:21:52. The bump lands 35
  minutes into the sunrise ramp.
- And the general fact underneath: **this fingerprint does not survive a change of daylight, it only
  tracks it.** With the camera provably stationary, 2026-08-08's day references correlate
  0.075–0.68 with that morning's first one — far below `VIEW_CORR_MIN` = 0.55 for most of the day.
  The template survives only because it blends 10% toward every agreeing frame, i.e. it tracks the
  sun as long as frames keep arriving. Across a night, none do.

`VIEW_MAX_GAP_S` already drops the *pending disagreement* across a gap in observation — "silence is
not evidence" — but leaves the *template* frozen, so the first five continuous minutes of dawn are
five minutes of sustained disagreement and the epoch moves.

**What a false bump costs.** It retires every reference (minor — they expire in 2 h anyway), it
**flushes the recurrence ledger**, and it writes a junk row into a table that three other backlog
items depend on (§2.1, §5.3, and the trail-cam half of the veto). The ledger is the expensive one:
recurrence requires ≥ 5 firings over **≥ 2 calendar days**, so after a dawn flush the evidence has
to be re-earned from zero every morning. The sidecar as found holds 169 clusters, 560 observations,
and **0 clusters satisfying recurrence** — every one of them stamped with a single day. The veto
can only ever fire between local midnight and the next dawn, on the second day of a fresh ledger.

## 5. What was changed, all of it still shadow mode

Nothing reads `suppressed_at`. Every detection is still saved, cropped and clipped exactly as
before. The audit gate is unchanged: no consumer honours the flag until a human has opened the
contact sheets.

1. **`ViewWatcher` re-seeds its template across an unwatched gap** (`VIEW_TEMPLATE_MAX_GAP_S`, =
   `VIEW_PERSIST_S`), the same rule already applied to the pending disagreement. A template from
   before an interval nobody watched cannot start the clock that says the camera moved. **The cost
   is real and deliberate:** a reposition performed during a lull longer than that is never
   detected. That is the safe direction — a missed bump leaves a stale reference the pixel test and
   `refimg_max_age_s` still have to get past, while a false bump destroys the ledger every morning.
2. **Coverage remembers a filled contour, per pixel, instead of a bounding rectangle in a list.**
   One `(180, 320)` float64 "last blobbed at" map: 460 KB flat, O(1) per frame, and it disowns
   exactly the pixels something moved over. The outline is *filled* rather than used raw because
   MOG2's foreground on a night animal is ragged, and an unfilled silhouette would leave holes
   inside the very body the gate exists to protect.
3. **`COVER_MIN_FRACTION` unchanged at 0.9**, with §3's numbers written next to it so the next
   reader does not have to re-derive them.
4. **`VetoCensus`: one log line an hour, counting every decision including the abstentions**, with
   the cover quantiles. `db.record_suppression` writes a row only for SUPPRESS, which is precisely
   why an inert veto and a perfectly precise one are indistinguishable in the database, and why
   answering §1.4 took a full clip replay. One line an hour would have answered it in a `grep`.
   (This is the concrete case for `docs/deferred-work.md` §6's missing `shadow_reviews` record.)

## 6. Still unmeasured, and it is the one that decides coverage's future

**Design §8 item 2 — detector recall at the reference moment — remains unrun**, and it is the
measurement that says whether the coverage gate is essential or overhead. The recipe: take the
certified frames the rig banks over a day, run MegaDetector on them at conf 0.1, and count how many
contain a box the live 0.25 threshold rejected. It was **not** run here on purpose: it means putting
a second MegaDetector process on the GPU beside the live rig, and this box has already lost the rig
once to memory exhaustion. It wants a quiet window and Matt's say-so.

Two smaller ones:

- The junk `view_epochs` row is left in place. Epoch 1 is in force and the current references are
  keyed to it, so deleting the row would orphan them; it is a one-line correction to make
  deliberately, not a side effect of this change.
- The daytime arm has never once cleared the coverage bar. Whether that should become an explicit
  "night only on this camera" statement in `config.py`, or stay an emergent property of the gate,
  is a judgement call left open.
