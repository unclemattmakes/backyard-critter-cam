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

- **Windows** (uses the DirectShow capture backend for fast camera init).
- **NVIDIA GPU + CUDA.** Inference is GPU-only by design; the app refuses to run on CPU and
  prints an actionable error if the GPU can't execute.
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

---

## Running

```powershell
# Find your webcam's index if 0 isn't it
.\.venv\Scripts\python.exe backyard_cam.py --list-cameras

# Run the rig (default camera index 0, 1280x720, live preview)
.\.venv\Scripts\python.exe backyard_cam.py
```

Press **`q`** in the preview window to quit cleanly. The window is **resizable** — drag any edge.

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
| `--min-confidence F` | Minimum detector confidence to draw/save (default 0.25). |
| `--motion-min-area N` | Largest motion blob (px) needed to wake the detector (default 800). Raise to ignore small twitches; lower to catch smaller/farther critters. |
| `--detector-interval S` | Min seconds between detector runs while motion continues (default 1.0). |
| `--save-full-frame` | Also save the whole frame per detection event (default off; crops always saved). |
| `--db PATH` / `--crops-dir PATH` | Override output locations. |
| `--no-preview` | Headless; quit with Ctrl+C. |
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

---

## Output

```
backyard/
  backyard.db          SQLite database (one row per detection)
  crops/2026-06-07/    cropped detection JPEGs, foldered by date
  frames/2026-06-07/   full frames (only if --save-full-frame)
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
| `species` | **NULL in V1.** Phase 2 (species classification) fills it. |
| `individual_id` | **NULL in V1.** Phase 3 (re-identification) fills it. |

A second table, **`detection_embeddings`** (keyed to `detections.id`), is created but never
written to in V1 — it's the stub for phase-3 re-ID vectors.

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
- **Phase 3 — Individual re-ID:** embed → cluster → hand-label ("Notch", "Gimpy"), per
  species, **raccoons first** (mask / tail-rings / ear notches carry signal; crows barely do).
- **Phase 4 — Behaviour + the disagreement alert:** per-individual arrival windows,
  co-occurrence, dwell time. The payoff is a **two-axis readout** — appearance match score
  *next to* behaviour fit — that flags "looks like X but isn't acting like X."
- **Later — trail-cam batch importer:** `source='trail_cam_sd'`, same pipeline downstream.

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
