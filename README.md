# Backyard Critter Cam

A live backyard-critter detection rig. A USB webcam points through a sliding glass door at
the yard; the program watches the feed, wakes a real animal detector only when something
moves, draws live bounding boxes, and saves a cropped image + a database row for every
critter it sees — crows, raccoons, opossums.

This glass-door cam is the **primary rig for all three species, day and night** (the
"glass = mirror after dark" worry didn't pan out — lit animals at the pane read clearly; a
raccoon already turned up at dusk). A second source — a wider-yard weatherproof trail cam,
imported in batches off its SD card — plugs into the same pipeline later. **V1 (this code)
is the live capture skeleton.** The full four-phase vision lives in [PLAN.md](PLAN.md).

---

## How it works

```
USB webcam (OpenCV / DirectShow)
   -> MOG2 motion gate        cheap background subtraction; skips still frames
   -> MegaDetector v6 (CUDA)  runs only on frames with real motion, rate-limited
   -> per detection:          draw box  +  save crop (+ optional full frame)  +  SQLite row
   -> live preview window      boxes + confidence; press 'q' to quit
```

The motion gate means the GPU detector runs a tiny fraction of the time (only when
something actually moves), so the rig is light enough to sit running next to you all day.

- **Detector:** [MegaDetector v6](https://github.com/microsoft/MegaDetector) — Microsoft AI
  for Good Lab's camera-trap model. The official MDV6 weights are
  [Ultralytics](https://docs.ultralytics.com/) YOLO models, so we run them **directly via
  Ultralytics** on the GPU rather than through the heavier PytorchWildlife wrapper (see
  [Notes on the detector](#notes-on-the-detector)). It classifies coarse `animal` /
  `person` / `vehicle` and rejects empty frames. Fine-grained **species** and **individual
  ID** are deliberately *not* done in V1 (future phases).
- **Saved by default:** animals only. People and vehicles are still *drawn* in the preview
  (wave at the camera to confirm it's working) but not written to disk, so you sitting next
  to the camera don't fill `crops/` with selfies. Change `save_classes` in `config.py`.

---

## Notes on the detector

MegaDetector v6 is distributed via Microsoft's
[PytorchWildlife](https://github.com/microsoft/Pytorch-Wildlife) framework, which is the
package most docs point to. In practice PytorchWildlife eagerly imports a bioacoustics stack
(`soundfile`), a classifier stack (`timm`), the legacy `yolov5`, and a `gradio` web UI —
heavy, brittle on Python 3.14, and at odds with this project's lean / no-web-server rule.

The MDV6 weights themselves are ordinary [Ultralytics](https://docs.ultralytics.com/) YOLO
models — PytorchWildlife runs them through Ultralytics internally. So this rig downloads the
**identical official MDV6 weights** (Microsoft's Zenodo release) and runs them directly
through Ultralytics: same model, same GPU inference, a fraction of the dependencies. The
`version → weight` mapping lives in `detector.py` (`MDV6_WEIGHTS`).

---

## Requirements

- **Windows or Linux** (macOS too). Windows uses the fast DirectShow capture backend; off
  Windows the rig automatically falls back to OpenCV's native backend (V4L2 on Linux), so no
  config change is needed.
- **NVIDIA GPU + CUDA recommended, but not required.** By default (`--device cuda`) inference
  runs on the GPU and the app fails loud if the GPU can't execute. Pass **`--device cpu`** to
  run without a GPU — slower per frame, but the motion gate only wakes the detector on real
  motion, so a backyard rig stays usable on a laptop CPU — or **`--device auto`** to use the
  GPU when present and fall back to CPU otherwise.
- **Python 3.14** (what this project was built and tested against).

> **Blackwell / RTX 50-series note (important).** This machine's RTX 5050 is compute
> capability **sm_120**. A stock `pip install torch` (a CUDA 12.6 build) *appears* to work
> (`torch.cuda.is_available()` returns `True`) but then dies at the first real GPU op with
> `CUDA error: no kernel image is available for execution on the device`. You need a
> **CUDA 12.8+ build of torch**. We use the **cu130** wheels below. The app also verifies
> this at startup by running a real GPU op, so a wrong build fails loudly and early.

---

## Setup

From the project folder (`C:\Users\you\projects\backyard`):

```powershell
# 1. Create an isolated virtual environment (keeps your ComfyUI torch untouched)
C:\Python314\python.exe -m venv .venv

# 2. Install a Blackwell-capable torch (CUDA 13.0 build) -- NOT the default PyPI torch
.\.venv\Scripts\python.exe -m pip install torch==2.12.0 torchvision==0.27.0 `
    --index-url https://download.pytorch.org/whl/cu130

# 3. Install the rest (camera, detector engine, detector package)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The MegaDetector v6 weights (~tens of MB) download automatically from Zenodo the first time
you run a detection.

> **Running without an NVIDIA GPU (CPU / Linux box).** Skip the cu130 step above and install a
> CPU-only torch instead:
>
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```
>
> then `pip install -r requirements.txt` as usual, and run with **`--device cpu`**. If the box
> *does* have a (non-Blackwell) NVIDIA GPU, install the matching CUDA torch build and use
> **`--device auto`** instead. The camera backend switches to V4L2 automatically on Linux.

---

## Running

### Easiest way (no typing) — for the whole family

Double-click **`start_critter_cam.bat`** (or **`start_critter_cam_lan.bat`** to also watch from
a phone/tablet on the same Wi-Fi). A live **video window** opens and the **dashboard** opens in
your browser; species names are added automatically — there's nothing else to start.

**To stop:** click the live **video window** and press **`q`** — or just **close that window**.
Everything (camera, dashboard, *and* species naming) shuts down together, in one step.

### From the command line

```powershell
# Find your webcam's index if 0 isn't it
.\.venv\Scripts\python.exe backyard_cam.py --list-cameras

# Run the rig (default camera index 0, 1280x720, live preview)
.\.venv\Scripts\python.exe backyard_cam.py
```

Press **`q`** in the preview window — or **close the window** — to quit cleanly; that also stops
the species-naming helper. The window is **resizable** — drag any edge.

Prefer a browser? Add **`--serve`** for a one-stop local dashboard (live feed + stats + gallery):

```powershell
.\.venv\Scripts\python.exe backyard_cam.py --serve        # then open http://127.0.0.1:8000
```

### Common flags

All defaults live in `config.py`; these override them per-run:

| Flag | Meaning |
|------|---------|
| `--camera-index N` | Which webcam (default 0). |
| `--width W --height H` | Requested capture resolution. |
| `--model-version V` | `MDV6-yolov10-c` (default, fast) · `MDV6-yolov9-c` · `MDV6-rtdetr-c` · `MDV6-yolov10-e` / `MDV6-yolov9-e` (heavier, more accurate). |
| `--device D` | `cuda` (default, needs an NVIDIA GPU) · `cpu` (no GPU, slower) · `auto` (GPU if usable, else CPU). |
| `--min-confidence F` | Minimum detector confidence to draw/save (default 0.25). |
| `--motion-min-area N` | Largest motion blob (px) needed to wake the detector (default 800). Raise to ignore small twitches; lower to catch smaller/farther critters. |
| `--detector-interval S` | Min seconds between detector runs while motion continues (default 1.0). |
| `--save-full-frame` | Also save the whole frame per detection event (default off; crops always saved). |
| `--record-clips` / `--no-record-clips` | Record a short video clip around each visit (default ON, disk-capped to `clips_max_gb` with oldest-first pruning). |
| `--clip-classes C…` | Detector classes that trigger a clip (default = saved = `animal`); e.g. `--clip-classes animal person` to record yourself as a test. |
| `--db PATH` / `--crops-dir PATH` | Override output locations. |
| `--no-preview` | Headless; quit with Ctrl+C. |
| `--no-classify` | Detection only — don't start the live species-naming helper. The rig launches it (and stops it) automatically by default; this turns that off. You can still fill species later with `python classify.py`. |
| `--stats` | Print a DB summary (crops vs. visits, per-hour activity, latest catches) and exit. Read-only. |
| `--list-cameras` | Probe camera indices and exit (find the right `--camera-index`). |
| `--serve` | Also serve the local web dashboard (live stream + stats) at `http://host:port`. |
| `--port N` / `--host H` | Dashboard port (default 8000) / bind host (default `127.0.0.1`; `0.0.0.0` = LAN). |

### Checking what it's caught

`python backyard_cam.py --stats` prints a read-only summary (safe to run while the rig is
live) — crops vs. **visits**, per-day and per-hour activity, and the latest catches:

```
Detections (crops): 166     Visits (est, >5 min gap): 3
By day:
  2026-06-07     166 crops      3 visits   [animal:166]
Arrivals by hour (visit starts, local time):
  20h  #################### 2
  21h  ########## 1
```

It counts **visits, not crops** — one lingering raccoon logged 166 crops across just 3
visits. Tune the collapse window with `--visit-gap-min N` (default 5).

### Web dashboard (`--serve`)

`python backyard_cam.py --serve` runs the same capture loop **and** a local web page (open
`http://127.0.0.1:8000`) with the **live annotated feed** (MJPEG), the live **stats**, and a
**gallery** of recent crops — a one-stop shop you can leave open in a browser tab.

- Built on Python's stdlib `http.server` — **no web framework, no new dependencies**.
- **Localhost only** by default (it shows your camera). To watch from a phone on the same
  network add `--host 0.0.0.0` (and mind who's on your Wi-Fi).
- Runs in the same process as capture; combine with `--no-preview` for a headless,
  browser-only rig, or keep the native window too.
- **Species names appear on their own:** the rig starts a small naming **helper** (`classify.py
  --watch`) as a child process and stops it with the app, so the recent-visitor card fills in a
  species by itself — nothing extra to launch or close. The helper takes a minute to warm up its
  model at startup; the header shows **"Identifier: warming up… / on"** so you can tell it's
  working. (Disable with `--no-classify`; see `classify.py` for bulk re-naming after you edit the
  label list.)

---

## Individual re-identification (phase 3)

Two batch tools turn the accumulated crops into an **appearance** signal — *who does this
look like?* — kept deliberately separate from behaviour (phase 4), per the plan. Run them
after a species (raccoons first) has banked a few hundred readable crops.

```powershell
# 1. Embed: one MegaDescriptor appearance vector per readable raccoon crop -> detection_embeddings
.\.venv\Scripts\python.exe embed.py --species raccoon          # GPU, resumable

# 2. Re-ID: cluster the vectors into candidate individuals + write contact-sheet montages
.\.venv\Scripts\python.exe reid.py --species raccoon           # -> reid/raccoon/cluster_*.jpg

# Look at the montages, then name a cluster that's clearly one animal:
.\.venv\Scripts\python.exe reid.py --name cluster_03 Notch     # sets individual_id on its crops

# "Who else looks like this crop?" — the second-opinion lookup (cross-session, burst-filtered):
.\.venv\Scripts\python.exe reid.py --neighbors 2765
```

- **Model:** [MegaDescriptor-L-384](https://huggingface.co/BVRA/MegaDescriptor-L-384) (the
  wildlife re-ID foundation model), loaded via `timm`. Vectors are L2-normalized, so cosine
  similarity is a dot product. The schema keys vectors by `model`, so a second embedder can be
  added later without a migration.
- **What works well:** near-duplicate / same-visit grouping is excellent (a lingering
  raccoon's burst scores 0.97+), so clustering reliably collapses a visit, and the
  neighbour lookup is a useful "second opinion."
- **What it is *not*:** a push-button "name every raccoon" oracle. On whole-body crops with
  an identical patio background, the embedding keys partly on **pose + background**, so
  cross-session *same-individual* similarity (~0.5) overlaps with *different-individual*
  similarity — there's no clean threshold that maps clusters straight to individuals. That's
  by design: appearance is **one axis**, surfaced to *augment* your eye, not replace it (see
  [PLAN.md](PLAN.md)). Hand-labelling the obvious clusters + the neighbour lookup is the
  intended workflow; behaviour (phase 4) is the other axis.

---

## Behaviour clips (phase 4 capture)

Stills capture *who* and *when*; a short **video clip** captures *how* — gait, approach speed,
dwell, vigilance, who-defers-to-whom. That's the substance of behaviour, and a confound-robust
second shot at individual ID (a limp reads the same from any angle, where a single still — pose
+ soft glass — does not). **On by default**, and safe to leave on: the clips folder is a
**rolling window** — past the `clips_max_gb` disk budget (default 10 GB ≈ two busy weeks) the
oldest clips are pruned automatically, file and DB row both.

```powershell
.\.venv\Scripts\python.exe backyard_cam.py                           # clips record by default
.\.venv\Scripts\python.exe backyard_cam.py --no-record-clips         # ...this run, stills only
.\.venv\Scripts\python.exe backyard_cam.py --clip-classes animal person   # also record yourself (test)
```

- A rolling **pre-roll buffer** means each clip opens on the animal *arriving* (the seconds
  before the detector first fired); recording ends a few seconds after the last detection, or at
  a safety cap so a camped-out raccoon can't make a giant file.
- **`clipmotion.py`** turns each clip into a **motion fingerprint** — the detector tracks the
  animal through sampled frames, and the trajectory yields duration, path length, straightness
  (beeline = 1, milling about ≈ 0), avg/peak speed, moving fraction (vs head-down eating), and
  an approach/retreat cue. The raw track is stored as JSON (`clip_tracks` table) so richer gait
  work (stride rhythm, limp detection) can re-derive later without re-running the detector.
  Batch + resumable: `python clipmotion.py` after clips accumulate; `--show` to read them back.
- One `.mp4` per visit under `clips/<date>/`, plus a row in the **`clips`** table (time span,
  fps, size, detection count) for later behaviour queries. Crops are still saved alongside.
- All knobs (pre/post-roll, max length, fps, downscale, codec, trigger classes, disk budget)
  live in `config.py`; `clip_scale < 1.0` trims the in-RAM buffer and file size if memory is
  tight; `record_clips = False` turns recording off permanently (incl. the family launchers).

---

## Behaviour analysis (phase 4)

The second axis. Once crops are classified, a chain of small tools turns raw detections into
*behaviour* — and pairs it with the appearance axis for the payoff read-out:

```powershell
.\.venv\Scripts\python.exe visits.py --stats        # collapse detections -> visit events (count visits, not crops)
.\.venv\Scripts\python.exe behavior.py              # per-species arrival windows, dwell, co-occurrence
.\.venv\Scripts\python.exe reid.py --match 457      # "closest appearance match: raccoon_c02 0.74, ..."
.\.venv\Scripts\python.exe twoaxis.py               # appearance NEXT TO behaviour; flag the disagreements
```

- **`visits.py`** — collapses consecutive same-source detections (< gap minutes apart) into one
  **visit** (`visits` table + a `visit_id` on each detection). One lingering raccoon = ~200
  crops but **one** visit; everything downstream counts visits, not crops.
- **`behavior.py`** — per-species (and per-individual) profiles: arrival-hour window (computed on
  the 24-h circle so a midnight-spanning crepuscular animal gets one sensible window), dwell
  time, visits/day, and **co-occurrence** (which species share a visit — crows run in family
  groups, so who-arrives-with-whom is real signal).
- **`reid.py --match`** — ranks the labelled individuals a crop looks most like (cosine to each
  individual's appearance centroid). The appearance axis as a number.
- **`twoaxis.py`** — the payoff. For each visit it puts the **appearance** match *next to* the
  **behaviour** fit and flags when they **disagree** ("labelled raccoon, but arrived at 11am —
  raccoons are nocturnal"). Catches both genuinely unusual visits and mis-classifications; gets
  sharper per-individual as you hand-label with `reid.py --name`.

The dashboard surfaces all of this: a **Behaviour** tab (profiles, off-pattern flags,
co-occurrence) and an **Individuals** tab where you *name the cast* — each look-alike group
shows its best crops with a name box; naming two groups the same name merges them, and the
per-individual axes sharpen as the cast grows.

The visit ledger refreshes itself on rig shutdown and after trail-cam imports; `visits.py` is
only needed manually after offline label edits.

---

## Output

```
backyard/
  backyard.db          SQLite database (one row per detection; clips in the clips table)
  crops/2026-06-07/    cropped detection JPEGs, foldered by date
  frames/2026-06-07/   full frames (only if --save-full-frame)
  clips/2026-06-07/    behaviour video clips (only if --record-clips)
  reid/raccoon/        re-ID contact-sheet montages (regenerated by reid.py)
```

Crop filenames look like `2026-06-07T19-25-59-123_0_animal_0.94.jpg`
(`timestamp_index_class_confidence`).

---

## Database schema

One row per detection in the **`detections`** table. Timestamps are **local time with UTC
offset, ISO 8601** (e.g. `2026-06-07T19:25:59.123456-07:00`) — the wall-clock time you'd
read off a window-side clock, but globally unambiguous and sortable.

| Column | Notes |
|--------|-------|
| `id` | PK. |
| `timestamp` | Local ISO 8601 with offset. |
| `source` | `glass_door_cam` in V1. Future: `trail_cam_sd`. |
| `detection_class` | Coarse label: `animal` / `person` / `vehicle`. |
| `confidence` | Detector score 0–1. |
| `bbox_x1..bbox_y2` | Box in absolute pixels. Stored as 4 columns so it's queryable. |
| `frame_w`, `frame_h` | Frame size, so a box stays interpretable / re-normalizable. |
| `crop_path` | Path to the saved crop (relative to project root). Always set. |
| `frame_path` | Full-frame path, or NULL. |
| `species` | Phase 2 (species classification, `classify.py`) fills it. |
| `individual_id` | Phase 3 (re-identification): set when you hand-label a cluster with `reid.py`. NULL until then. |
| `visit_id` | Phase 4: which `visits` row this crop belongs to (stamped by `visits.py`). |

A second table, **`detection_embeddings`** (keyed to `detections.id`), holds the phase-3
appearance vectors: `embed.py` writes one L2-normalized MegaDescriptor embedding per readable
crop, keyed by `model`, and `reid.py` reads them back to cluster crops into individuals.

A third table, **`clips`**, holds one row per recorded behaviour clip (`--record-clips`):
`clip_path`, `started_at`/`ended_at`, `fps`, size, and detection count. A clip spans a time
window on one `source`, so the detections captured during it join by timestamp — no FK needed.

A fourth table, **`visits`** (phase 4, built by `visits.py`), holds one row per collapsed visit
event — `source`, dominant `species`/`individual_id`, `started_at`/`ended_at` (→ dwell),
`detection_count`, and a `representative_detection_id`. Detections point back via `visit_id`.

Example queries:

```sql
-- Today's animal sightings, newest first
SELECT timestamp, confidence, crop_path FROM detections
WHERE source='glass_door_cam' AND detection_class='animal'
  AND timestamp >= date('now','localtime')
ORDER BY timestamp DESC;

-- Sightings per hour (rough activity profile)
SELECT substr(timestamp,1,13) AS hour, count(*) FROM detections
GROUP BY hour ORDER BY hour;
```

---

## Configuration

`config.py` is the single source of truth for every knob: camera index/resolution, motion
sensitivity, model version, confidence threshold, which classes get saved, output paths,
and the `--save-full-frame` default. The CLI flags above just override these per run.

### Camera tuning & day/night profiles

The camera's best settings differ with the light, so two pieces handle it:

- **`tune.py`** — finds good settings for the current scene: sweeps exposure, scores each
  frame for sharpness + highlight clipping, writes a labeled contact sheet + `metrics.csv`
  under `tuning/<timestamp>/`, and prints a recommendation. Re-run after you move the camera.
- **Sun-driven profiles** — the live app selects a `camera_profiles` entry (`day` / `night`)
  by **sun position** (civil dawn → dusk = day), so the day/night switch tracks the seasons,
  and it re-applies the profile automatically when the sun crosses. Workflow: run `tune.py`
  during the day → paste into the `day` profile; run it at night (ideally with a feeding-spot
  light) → paste into `night`; the rig switches between them on its own. Set
  `latitude`/`longitude` in `config_local.py` (copy `config_local.example.py`) to your location.

---

## Roadmap

Full plan and design philosophy: **[PLAN.md](PLAN.md)**. The short version:

- **Phase 1 — Live capture skeleton:** ✅ this code.
- **Phase 2 — Species ID:** classify the accumulated crops to fill `detections.species` (let
  a few days of crops pile up first; pick the current best wildlife classifier at build time).
- **Phase 3 — Individual re-ID:** ⚙️ tooling built (`embed.py` → `reid.py`): embed → cluster →
  hand-label ("Notch", "Gimpy"), per species, **raccoons first**. Same-visit grouping is
  strong; confident *cross-session* individual ID from appearance alone is not (background +
  pose dominate) — so it's a labelling assistant + second opinion, paired with phase-4 behaviour.
- **Phase 4 — Behaviour + the disagreement alert:** ⚙️ built (see *Behaviour analysis* above):
  `visits.py` (visit events), `behavior.py` (arrival windows, dwell, co-occurrence), `reid.py
  --match` (appearance score), and `twoaxis.py` — the **two-axis readout** that flags "looks like
  X but isn't acting like X." Plus `--record-clips` banks the video for gait/motion analysis.
  Sharpens per-individual as you hand-label. Still to come: motion/gait features off the clips.
- **Later — trail-cam batch importer:** ✅ `import_trailcam.py` — dump the SD card into a folder,
  `python import_trailcam.py <folder>` runs the same detector over it and writes crops with
  `source='trail_cam_sd'`. EXIF timestamps, idempotent re-runs, `--watch` a drop folder. Same
  pipeline downstream (species ID, re-ID, behaviour all key off the `source` column).

Guiding principle: keep **appearance and behaviour on separate axes** and surface both —
augment the critter-knower, don't replace them. And: boring and robust over clever; most of
the value is in capture and accumulation, not the fancy model.

The schema and module layout are arranged so each phase is an addition, not a rewrite.

---

## License

Copyright © 2026 Matt Scott. Licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE).

The rig runs MegaDetector v6 through [Ultralytics](https://github.com/ultralytics/ultralytics), which is **AGPL-3.0**, so this project is AGPL-3.0 as well. Note the network-use clause: if you run a modified version as a network service (for example, exposing the dashboard to others), you must offer those users the corresponding source.

---

## Troubleshooting

- **`[CUDA ERROR] ... no kernel image is available` / capability sm_120:** your torch is the
  wrong CUDA build. Reinstall with the cu130 wheels (see [Setup](#setup) step 2).
- **`Could not open camera index 0`:** run `--list-cameras`; another app (Zoom, Camera,
  ComfyUI) may be holding the webcam. Close it or pick a different index.
- **No detections though motion shows:** lower `--min-confidence`, or check lighting; the red
  dot (top-right of the preview) confirms the motion gate is firing.
- **Too many / too few motion triggers:** tune `--motion-min-area` (lower = more sensitive).
```
