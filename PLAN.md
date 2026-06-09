# Backyard Critter Cam — Project Plan

> Canonical project plan (authored by Matt, 2026-06-07). The code in this repo is **Phase 1**.
> This document is the single source of truth for where the system is going; module/README
> comments defer to it.

*A live + batch wildlife detection and individual-identification system for the backyard.
Crows, raccoons, opossums. Built on the NVIDIA laptop, logging to SQLite, augmenting an
existing instinct rather than replacing it.*

---

## The core idea (read this before building anything else)

The point of this project is **not** to build an autonomous "who is this animal?" oracle.
A hobby rig will never be confident enough to trust blindly, and chasing that goal throws
away the most interesting signal you already have.

When a critter shows up and you think *"you look like Notch, but you're not acting like
Notch,"* you're running **two separate classifiers in your head**:

- an **appearance** model ("you look like X")
- a **behavior** model ("but you don't act like X")

The friction you feel is the moment those two disagree — and **that disagreement is the
information.** It's either a new individual who resembles a known one, or a known individual
in an unusual state. Which of those it is, is exactly what you want to know.

So the design principle for the whole system is: **keep appearance and behavior on separate
axes, and surface both.** Never collapse them into one answer. The tool's job is to give your
existing instinct a *memory* and a *second opinion* — e.g.:

> closest appearance match: **Notch 78%**, Gimpy 41%
> …but this one arrived at **2pm, alone**; Notch is usually 4–5pm, with the pair.

Augment the critter-knower. Don't pretend to be one.

A second guiding principle, carried over from how the rest of your tooling works: **boring
and robust over clever.** Build the smallest thing that works, watch it run, then add the
next layer. Most of the value is in capture and accumulation, not in the fancy model.

---

## Hardware: two cameras, one brain

Two capture sources feed **one** pipeline and **one** database. The pipeline doesn't care
where a crop came from — there's a `source` column and that's the only difference.

### 1. Glass-door webcam — the live, owned, primary rig
USB webcam pointed through the sliding glass door, plugged straight into the NVIDIA laptop.
Fully owned end-to-end (sensor → script), so it's the real-time playground: live detection,
live overlay, eventually live "that's not Notch" alerts while you're sitting right there.

**Revised understanding (important):** this is NOT a daytime-only rig. The textbook
"glass = mirror at night" assumption doesn't hold for your conditions, because:
- at least one raccoon visits **before dark** (pure daytime capture, no asterisk), and
- your nocturnal visitors come **right up to the glass with lights on them**.

The mirror effect is about *which side is brighter*, not darkness per se. Animals pressed
against the pane with light on them read clearly. So the door cam is plausibly the **primary
rig for all three species** — crows by day, the early raccoon at dusk, lit mammals at night.

*Near-zero-effort upgrade if after-dark crops come out murky:* a motion-sensor porch light
or a cheap light aimed at the feeding spot tips the lighting ratio decisively.

### 2. VOOPEAK trail cam — the batch, wide-yard, weatherproof corpus-builder
Model is the Voopeak/Campark TC02/TC08 family (same OEM, rebadged). **Key constraint:** its
"WiFi" is a short-range self-broadcast hotspot activated over Bluetooth — the app connects
directly within ~30–65 ft. It does **not** join your home network and has **no** automatic
push to a server. Getting images off it = a human + the app in range, or pulling the SD card.

Do **not** try to reverse-engineer its protocol — it's a real RE project with uncertain
payoff and the opposite of the small-tool philosophy. Treat it as a **batch source**: dump
the SD card every few days into a watch folder; process overnight. Batch is the correct
architecture for individual-ID accumulation anyway.

Its role is demoted from "the only thing that works at night" to **"covers the wider yard
and the angles the door can't see,"** with weatherproofing and 850nm IR for night.

**This plan focuses on the webcam. The trail-cam importer is a later add-on** that drops
crops into the same pipeline with `source = 'trail_cam_sd'`.

---

## Per-species reality (sets expectations for individual ID)

Individual ID difficulty is **wildly** species-dependent. Setting this now saves a
frustrating week later.

**Raccoons — best case.** Mask pattern, tail-ring count/spacing, body size, ear notches,
scars. Variable and stable over time. Appearance embeddings do real work here. *(Already
vindicated: the first front-on crop showed readable mask geometry and ear shape.)*

**Opossums — good.** They get visibly wrecked — frostbitten/torn ears, tail scarring, fight
damage — and that damage is distinctive and persistent. Females carrying joeys = strong
transient tag. Appearance carries decent signal.

**Crows — hardest on appearance, richest on behavior.** Near-uniform black; discriminating
marks (missing toe, molt gap, bill quirk) are mostly transient. Appearance embeddings alone
will struggle. **But** crows run in family groups, so *co-occurrence* (who shows up with
whom) is a real individuating feature, and your "doesn't act like X" instinct is already
doing most of the work. For crows, the system should **lean on behavior, not looks.**

**Day vs. night / color vs. IR:** daytime crows = color, lean behavioral. Nighttime mammals
(IR grayscale on the trail cam) = color washes out but *structure* survives — tail rings,
ear notches, mask geometry. So IR mammals lean visual. The feature mix differs by species
**and** by time of day, and it lines up neatly: color-rich daytime birds → behavior;
structure-rich IR mammals → vision.

---

## Data model

One SQLite database, shared by both camera sources and all phases. Design the schema for
the full system now; populate only what each phase can.

**`detections` table** (the spine):
- `id` (pk)
- `timestamp` — local time, ISO 8601, tz convention documented in comments
- `source` — `'glass_door_cam'` (v1) / `'trail_cam_sd'` (later) / etc.
- `bbox` — detection box coords
- `confidence` — detector confidence
- `crop_path` — path to saved crop (mandatory)
- `frame_path` — path to full frame (nullable)
- `species` — TEXT, nullable. NULL in v1; phase 2 fills it.
- `individual_id` — TEXT, nullable. NULL until phase 3; re-ID fills it.

**Future stores (stub/comment now, don't implement early):**
- embeddings for re-ID — either an `embedding` BLOB column or a separate table keyed to
  `detection.id`.
- a `visits` concept (see "Visit-event collapsing" below).
- per-individual behavior profiles (phase 4).

---

## Phased build

### Phase 1 — Live capture skeleton ✅ DONE
USB webcam (OpenCV, `CAP_DSHOW` on Windows) → MOG2 motion gate → MegaDetector on gated
frames (CUDA) → crop + SQLite row → **live preview window with bounding boxes**. The
overlay is the point: you watch it work. Species and individual ID deliberately deferred.

*Status: working end-to-end. First crops captured a raccoon at dusk, correctly boxed, with
confidence in the filename. Mirror problem never materialized.*

### Phase 2 — Species ID on the crops
Run a classifier on the animal crops to fill the `species` column. Let crops accumulate for
a few days first so there's a corpus to work with. **Tooling moves fast here — verify the
current best backyard/wildlife species classifier options at build time rather than
assuming.** Filter on confidence before classifying (high-confidence crops are the readable
ones — the filename confidence is effectively a crop-*usability* score).

### Phase 3 — Individual re-identification
Crop → feature embedding → cluster embeddings → hand-label clusters once ("Notch," "Gimpy,"
"three-legged"). Turn this on **per species**, starting with raccoons (best case), then
opossums. For crows, expect appearance to underperform — lean on phase 4 instead. There's a
small ecosystem of wildlife-specific re-ID tooling; **check what's current when you get
here.** Store embeddings per the stubbed schema. Filter by confidence before clustering.

### Phase 4 — Behavior profiles + the disagreement alert
The differentiator nobody builds. Accumulate a per-individual profile: typical arrival
window, co-occurrence partners, boldness at the food, dwell time. Then the payoff —
implement the **two-axis readout**: appearance match score *next to* behavior fit, and fire
the *"this looks like X but isn't behaving like X"* alert when they disagree. This is your
original instinct, given a memory.

### Later — Trail-cam batch importer
Watch-folder script that ingests SD-card dumps with `source = 'trail_cam_sd'`. Same pipeline
downstream. Handles IR grayscale frames (structure-based features for night mammals).

---

## Gotchas & notes (things that will quietly bite if ignored)

**Visit-event collapsing — plan for it before phase 4.** One animal lingering throws many
triggers. The first capture session logged 3 crops of (almost certainly) one raccoon inside
a 60-second window. Raw crop counts will be ~10x inflated as "visit" counts. Fix: collapse
detections of the same individual separated by < N minutes into a single **visit event**.
Count visits, not crops, for any behavior/frequency stat. Add a `visits` concept to the
schema's future-work comments now.

**Confidence ≈ crop usability.** The detector confidence tracks how readable the crop is,
not just "is it an animal." High-confidence crops are the ones worth feeding to species ID
and re-ID. The filter threshold roughly sets itself from real data.

**MegaDetector packaging changes.** It's commonly distributed now via Microsoft's
Pytorch-Wildlife / `PytorchWildlife`. **Verify the current recommended package, model
version, and GPU-inference API at build time** — don't assume a snapshot. *(Phase-1 note:
PytorchWildlife's eager imports pull a web UI + audio stack, so this build loads the
identical official MDV6 weights directly via Ultralytics — see `detector.py` / README.)*

**Lighting beats optics theory.** Don't write off after-dark frames before seeing them. Let
the crop folder be the judge of what the door cam can actually capture; add a porch/spot
light only if real crops come out murky.

**Robustness.** The capture loop must survive webcam disconnects, missing CUDA, and empty
frames without crashing. Boring and robust.

---

## Open decisions

- **Webcam after-dark performance** — collect a few nights of real crops, then decide
  whether to add a light. (Data-driven, not now.)
- **Trail-cam transport** — SD-card dump vs. app-export-then-sync, decided when the importer
  gets built. The watch folder makes this an interchangeable detail.
- **Re-ID tooling** — defer the choice to phase 3; the field moves.

---

## One-line summary

Two cameras, one SQLite brain, four phases. Capture first (done), then name the species,
then name the individuals (raccoons first), then learn their behavior — and surface
appearance and behavior *separately* so the system can tell you when a familiar-looking
visitor isn't acting like themselves.
