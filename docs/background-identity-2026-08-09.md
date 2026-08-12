# Does the background identify the animal? Yes — and past a week, it is all there is

`docs/deferred-work.md` §2.1 called this the evaluation's most consequential unrun test. It is run.

**The question.** Every re-ID number this project reports is a top-1 against a 0.345 majority
baseline ("always say Stan"). But a crop is not just an animal: it is an animal *somewhere*, and a
given raccoon tends to be photographed in the same corners of the same yard. If the background
alone identifies the animal, then the baseline is wrong, the decay curve is partly a story about
scene drift rather than animal drift, and the priority moves from modelling to **capture**.

**The method.** The same 139 confirmed-solo raccoon visits, the same prototypes, the same
session-blocked leave-one-visit-out. The *only* thing that varies is which pixels MegaDescriptor
sees. 4,168 crops re-embedded per arm, on CPU — the GPU is carrying the live rig, and this box has
lost the rig to memory exhaustion before.

| arm | what the embedder sees | blocked top-1 | 7-day embargo | 21-day |
|---|---|---|---|---|
| **intact** | the crop as saved | **0.741** | 0.482 | 0.122 |
| **animal** | the detector's box; the surrounding ring blanked | **0.727** | 0.489 | 0.108 |
| **background** | the ring; the detector's box blanked | **0.597** | 0.489 | 0.094 |
| chance | majority baseline | 0.345 | 0.345 | 0.345 |

`crop_padding` is 0.15 per side, so the box is 1/1.3² = 59% of a crop's area and the ring is
**41%**. Both arms therefore see a substantial, real part of the photograph; the blanked region is
filled flat with the median colour of whatever survives.

---

## 1. The harness was validated before a pixel was blanked

Because a wrong answer here would be invisible. Three checks, all passed:

- **Reproduce the published number.** The intact arm returns 0.741 blocked and 0.482 at a 7-day
  embargo, against `docs/identity-eval-2026-08-05.md`'s 0.739 and 0.482 — the corpus has gained one
  Notch return since.
- **Reproduce the stored vectors.** Re-embedding unmodified crops through this pipeline matches the
  vectors already in `detection_embeddings` at **cosine 1.00000** over a sample. So the embedder,
  the transform and the prototype builder are the shipped ones.
- **Look at the pixels.** The blanking geometry was rendered for a spread of crops and eyeballed
  before the run started: the animal is gone in the `background` arm and present in the `animal`
  arm, with the padding ring behaving as intended.

## 2. The control does its job: mutilation is not what moves the number

The obvious objection to a low background score would be that MegaDescriptor simply embeds
heavily-blanked images to mush. The `animal` arm is that control, and it kills the objection:

> **intact 0.741 vs animal 0.727 — 7 probes differ one way, 5 the other, exact binomial p = 0.77.**

Blanking 41% of every crop costs **nothing measurable**. The model is entirely comfortable with a
mutilated image, so the `background` arm's 0.597 is a measurement of information, not an artifact
of vandalism.

## 3. The background carries real identity, and it is redundant rather than additive

Paired per-probe, over all 139:

| comparison | right / wrong | reverse | exact binomial p |
|---|---|---|---|
| intact vs background | 25 | 5 | **0.0003** |
| animal vs background | 23 | 5 | **0.0009** |
| intact vs animal | 7 | 5 | 0.77 |

So the animal genuinely carries more than the background — that part is significant and should not
be overstated in the other direction. But look at the overlap:

- background-only gets **83** of 139 right; animal-only gets **101**;
- **78 of the background's 83 correct answers are ones the animal also gets right** (94%);
- only **5** probes are background-right where the animal is wrong.

The two channels are **not additive, they are redundant**. The background is very rarely telling
you something the animal does not; it is standing in for the animal, because in this corpus who an
animal is and where it was photographed are heavily confounded.

## 4. The finding that matters: at a week, all three arms are the same

| embargo | intact | animal | background | the animal's margin over background |
|---|---|---|---|---|
| none (session-blocked) | 0.741 | 0.727 | 0.597 | **+0.144** |
| 7 days | 0.482 | 0.489 | **0.489** | **0.000** |
| 21 days | 0.122 | 0.108 | 0.094 | +0.028 |

At same-week range the animal beats scene matching by 0.144. **At a week's remove it does not beat
it at all** — 0.482 against 0.489 is one probe, noise. Whatever survives seven days in this
embedding is not animal-specific information.

That reframes the decay curve recorded on 2026-08-05. It is not only "appearance identity decays in
about a week". It is: *the animal-specific part of the signal is gone in about a week, and what is
left of the number is the yard.*

## 5. What this changes

1. **The null is wrong, and every future identity number wants restating against it.** "Better than
   chance" here has meant better than 0.345. The honest floor for "can we recognise this animal" is
   what you get from the **background alone at the same embargo** — 0.597 same-week, 0.489 at seven
   days. Measured that way the shipped 0.741 is a +0.144 achievement, not a +0.396 one, and the
   7-day 0.482 is not an achievement at all.
2. **The next move is capture geometry, not a better backbone** — which is exactly the branch
   `deferred-work` §2.1 wrote in advance. More pixels on the body and fewer on the wall behind it;
   a tighter, more consistent framing at the dish. A backbone swap cannot separate two channels
   that are carrying the same information.
3. **It raises the value of era-invariant signals** already on the backlog — the ear notch (§3.4),
   and mass from a load cell (§4.1) — for a new reason. Their appeal was that they do not decay;
   the added reason is that they cannot be confounded with the yard.
4. **`eval.py` should carry a background arm.** The number that matters is the *margin*, and it is
   currently uncomputed and unwatched. This run was a scratch script; the diagnostic deserves to be
   a nightly line.

## 6. What this does NOT say, and the limits worth keeping

- **It does not say the embedding "reads the wall instead of the raccoon".** It says the two are
  not separable in this corpus. The animal arm is significantly better than the background arm; the
  channels overlap because the data does.
- **The `animal` arm is not animal-only.** A detector box is a rectangle, not a segmentation mask,
  so it contains a good deal of ground and wall. A segmentation-masked arm would draw the line
  properly and would be the natural follow-up.
- **The `background` arm is not structure-only.** Its blanked region is filled with the ring's
  median colour, so scene *colour* leaks in alongside scene *structure*. A constant-grey fill arm
  would separate those two.
- **139 probes.** Differences under ~0.02 are noise, which is precisely why the paired counts and
  exact binomial tests are quoted above rather than the accuracies alone.
- Everything here is the **glass door's** corpus and the raccoon cast. The trail cam has no
  confirmed identities at all.

---

*Method: `scratchpad/m_bg_identity.py` (scratch, wrote no table). DB read from a snapshot, because
`visits.refresh` renumbers every visit id and a multi-pass measurement against the live database
silently reads a different corpus part-way through.*
