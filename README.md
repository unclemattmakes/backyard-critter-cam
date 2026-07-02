# Backyard Critter Cam

A live backyard-critter detection rig. A USB webcam points through a sliding glass door at
the yard; the program watches the feed, wakes a real animal detector only when something
moves, draws live bounding boxes, and saves a cropped image + a database row for every
critter it sees — crows, raccoons, opossums.

This glass-door cam is the **primary rig for all three species, day and night** (the
"glass = mirror after dark" worry didn't pan out — lit animals at the pane read clearly; a
raccoon already turned up at dusk). A second source — a wider-yard weatherproof trail cam,
imported in batches off its SD card — plugs into the same pipeline later.

What began as a live-capture skeleton is now the **full four-phase system** the plan called
for. It captures, **names the species** on every crop (BioCLIP 2), **re-identifies individual
animals** across nights — down to telling apart two raccoons that only ever show up together —
and reads their **behaviour** off short video clips, all surfaced in a local web
**dashboard** you can leave open in a browser tab. The design philosophy and the rest of the
roadmap live in [PLAN.md](PLAN.md).

---

## How it works

```
USB webcam (OpenCV / DirectShow)
   -> MOG2 motion gate          cheap background subtraction; skips still frames
   -> MegaDetector v6 (CUDA)    runs only on frames with real motion, rate-limited
   -> per detection:            draw box + save crop + score shot-quality + SQLite row,
                                and record a short video clip around the visit
   -> species namer (helper)    label each new crop (BioCLIP 2), behind a general-CLIP
                                "is this even an animal?" gate that drops food/empty frames
   -> live preview + dashboard  boxes & confidence in a window ('q' to quit) and/or a
                                local web page (--serve): live feed, stats, gallery, and more
```

The motion gate means the GPU detector runs a tiny fraction of the time (only when
something actually moves), so the rig is light enough to sit running next to you all day.

- **Detector:** [MegaDetector v6](https://github.com/microsoft/MegaDetector) — Microsoft AI
  for Good Lab's camera-trap model. The official MDV6 weights are
  [Ultralytics](https://docs.ultralytics.com/) YOLO models, so we run them **directly via
  Ultralytics** on the GPU rather than through the heavier PytorchWildlife wrapper (see
  [Notes on the detector](#notes-on-the-detector)). It classifies coarse `animal` /
  `person` / `vehicle` and rejects empty frames; **fine-grained species and individual ID are
  separate phases** layered on top (see [Species ID](#species-identification-phase-2) and
  [re-ID](#individual-re-identification-phase-3) below).
- **Detector class filter:** the street outside throws off a lot of `vehicle` boxes, so
  `detect_classes` in `config.py` tells the detector to report only `animal` + `person`. One
  YOLO pass scores every class regardless, so dropping the rest costs nothing and just
  declutters the live view.
- **Saved by default:** animals only. A person at the glass is still *drawn* in the preview
  (wave at the camera to confirm it's working) but not written to disk, so you sitting next
  to the camera don't fill `crops/` with selfies. Change `save_classes` in `config.py`.
- **Shot quality:** every saved crop is scored for sharpness — with a bump for night
  eyeshine, i.e. an animal looking toward the glass — by `quality.py`, so the dashboard can
  lead a visit with its *cutest, sharpest* frame rather than merely the highest-confidence one.

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
- **NVIDIA GPU + CUDA recommended, but not required.** By default (`--device auto`) inference
  runs on the GPU when it's usable and falls back to the CPU otherwise — so the rig runs on a
  laptop with no GPU out of the box (slower per frame, but the motion gate only wakes the
  detector on real motion, so it stays usable). Pass **`--device cpu`** to force CPU, or
  **`--device cuda`** to require the GPU and fail loud if a wrong torch build can't use it.
- **Python 3.10 or newer** (built and tested on 3.14; the app checks this at startup).

> **Blackwell / RTX 50-series note (important).** This machine's RTX 5050 is compute
> capability **sm_120**. A stock `pip install torch` (a CUDA 12.6 build) *appears* to work
> (`torch.cuda.is_available()` returns `True`) but then dies at the first real GPU op with
> `CUDA error: no kernel image is available for execution on the device`. You need a
> **CUDA 12.8+ build of torch**. `setup.bat` installs the right **cu130** wheels for you when it
> detects an NVIDIA GPU. The app verifies this at startup by running a real GPU op: with
> `--device cuda` a wrong build fails loudly and early; with the default `--device auto` it falls
> back to the CPU instead (never stuck — just slower).

---

## Setup

**Easiest — run the setup script.** It creates the virtual environment and installs the torch
build that matches your hardware (CUDA if you have an NVIDIA GPU, CPU otherwise), then the rest:

- **Windows:** double-click **`setup.bat`**
- **Linux / macOS:** `bash setup.sh`

Then start the app (see [Running](#running)); the MegaDetector weights download on first run.

### Manual setup (alternative)

From the project folder, with Python 3.10+ on your PATH:

```powershell
# 1. Create an isolated virtual environment (keeps any system torch untouched)
py -m venv .venv

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
| `--camera-index N` | Which webcam (default 0). Single-camera mode; for several at once see [Multiple cameras](#multiple-cameras-usb--networked). |
| `--source NAME` | DB `source` label for this camera's rows (single-camera mode; default `glass_door_cam`). |
| `--width W --height H` | Requested capture resolution. |
| `--model-version V` | `MDV6-yolov10-c` (default, fast) · `MDV6-yolov9-c` · `MDV6-rtdetr-c` · `MDV6-yolov10-e` / `MDV6-yolov9-e` (heavier, more accurate). |
| `--device D` | `auto` (default; GPU if usable, else CPU) · `cuda` (require an NVIDIA GPU, fail loud) · `cpu` (force CPU, slower). |
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

### Multiple cameras (USB + networked)

The glass-door webcam is the primary rig, but you can watch **several cameras at once** — a USB
webcam *plus* networked cameras around the yard. They run on one process (one capture thread per
camera, all sharing the single MegaDetector and the one naming helper), and the dashboard's **Live
Observation** tab shows a **grid of feeds**. Each camera writes its own `source`, so the whole
downstream — species ID, re-ID, behaviour, visits, the calendar — keeps the cameras separate
automatically (the schema was multi-source from day one).

List the cameras in `config_local.py` (copy `config_local.example.py`):

```python
from config import CameraSpec
def apply(cfg):
    cfg.latitude, cfg.longitude = 47.6, -122.3
    cfg.cameras = [
        CameraSpec("glass_door_cam", 0, name="Glass door"),                 # USB webcam, index 0
        CameraSpec("yard_ir", "rtsp://user:pass@192.168.1.50:554/h264Preview_01_sub",
                   name="Yard (night IR)"),                                  # RTSP/PoE IP camera
        CameraSpec("feeder_esp32", "http://192.168.1.51:81/stream",
                   name="Feeder (ESP32)", frame_width=640, frame_height=480, motion_min_area=300),
    ]
```

`src` is anything OpenCV's `VideoCapture` accepts: an **int** webcam index, or a **URL** — `rtsp://…`
for an IP/PoE camera, or `http://…/stream` for an ESP32-CAM's MJPEG server. Per-camera fields
(resolution, `motion_min_area`, day/night profile, `record_clips`) default to the global config;
only override what differs. Each camera needs a **unique `source`**.

- **Networked cameras** are opened through OpenCV's FFMPEG backend with a 1-frame buffer (so the
  stream can't lag behind the detector), forced to **TCP** transport, and reconnected with
  **indefinite backoff** (a network blip is transient — unlike a USB unplug, which gives up after
  a few tries). Use a camera's **sub-stream** (lower resolution) for the motion gate to keep
  decode cheap. In the dashboard, click a pane to make the **Instrument Panel** and **Who's
  visiting now?** act on that camera; a networked camera's exposure/focus are set on the camera
  itself, so its panel shows a note instead of sliders.
- **What to buy (for a nocturnal yard).** The targets here are mostly night animals, and that's
  where camera choice matters most. A **PoE IR camera** (e.g. Reolink RLC-510A, ~$60 — clean RTSP,
  IR night vision, one cable for power + data) is the night workhorse. An **ESP32-CAM** (ESP32-S3
  with PSRAM is steadier) is a cheap, fun **daytime** angle — its sensor is weak in the dark.
  Two notes from experience: IR night vision is **monochrome**, which costs the colour cues your
  species labels and re-ID rely on (a *colour-at-night* camera over a lit feeding spot keeps them);
  and mount any IR illuminator **off-axis** from the lens or eyeshine blows out the eyes.

Clips from each camera are written under `clips/<source>/<date>/`. Single-camera mode is just the
N=1 case — if you don't set `cfg.cameras`, nothing changes.

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
`http://127.0.0.1:8000`) — a field journal for the yard you can leave open in a browser tab. It
has grown from a single live feed into **six tabs**:

- **Live Observation** — the live annotated MJPEG feed, the most-recent-visitor card (species,
  how-long-ago, confidence, a ▶ badge to play its clip, and a live *off-pattern* flag if it
  arrived at an odd hour for its species), running tallies, and most-/least-common species. A
  **Who's visiting right now?** panel lets you name the live visit as it happens: tap a known
  critter (or add a new name) and **Log who's here** — one name tags that visit's crops with the
  individual (a live solo confirm that feeds the [re-ID](#individual-re-identification-phase-3)
  templates), two or more record who came *together* (co-presence, without mislabelling both
  animals as one). A short *recently logged* list shows your notes landing.
- **The Dispatch** — a newspaper-style **period digest** (☾ Night / ☀ Day): a back-to-back
  highlight reel of clips, the "plate of the night" hero shot, novelty & quiet flags ("first
  raccoon in 9 days"), moon phase, and a full species roll with per-hour activity clocks.
- **Behaviour** — per-species field notes (visits/day, median dwell, typical arrival window,
  hourly chart), **off-pattern** alerts, and **seen-together** co-occurrence pairs.
- **Individuals** — the "name the cast" workspace (see
  [phase 3](#individual-re-identification-phase-3)): a *Who is this?* review queue with
  one-click confirm / correct / clear, bulk **Fit to the Cast**, per-individual **Poses** and
  **Clips**, and **Un-blend** for multi-animal visits. If you named the visit live (above),
  Un-blend reads that log: it shows *"📓 you logged Notch + Elliot here"* and, once one cluster
  matches a known template, names the **other by elimination** — so a never-solo pair member gets
  identified from co-presence alone, before he has any template of his own.
- **Calendar** — a month grid; each day shows its visit count and top-species emoji, click
  through to a day's crops and visits.
- **Specimen Catalogue** — every species as a card; open one to confirm (✓) / reject (✗) /
  correct (✎, free-text allowed) each crop. Crops are **click-to-enlarge** everywhere, with
  ← / → to step through the strip.

There's also an **Instrument Panel** (the ⚙ modal): live camera controls — exposure, gain,
focus, white balance — read from and pushed to the *running* camera, so you can dial in the
glass-door shot without restarting (a live companion to `tune.py`).

- Built on Python's stdlib `http.server` — **no web framework, no new dependencies**.
- **Localhost only** by default (it shows your camera). To watch from a phone on the same
  network add `--host 0.0.0.0`. Even then the dashboard accepts connections **only from your
  local network** (loopback + private/LAN addresses): it refuses *direct* connections from the
  wider internet **and** validates the `Host` header to block DNS-rebinding from a malicious site
  you visit. That stops the common exposure paths — but it is **not a login**, so anyone already
  on your Wi-Fi has full access (including label edits). Don't port-forward it or put it on an
  untrusted network; for real remote access, front it with a VPN or an authenticating reverse
  proxy. Set `lan_only = False` in `config_local.py` only if you've done exactly that.
- Runs in the same process as capture; combine with `--no-preview` for a headless,
  browser-only rig, or keep the native window too.
- **Species names appear on their own:** the rig starts the naming **helper** (`classify.py
  --watch`) as a child process and stops it with the app, so the recent-visitor card fills in a
  species by itself — nothing extra to launch or close. The helper takes a minute to warm up its
  model; the header shows **"Identifier: warming up… / on"** so you can tell it's working.
- **Clips just play.** Clips recorded in legacy `mp4v` aren't browser-playable, so the server
  transcodes them to H.264 **on demand** into a `clips_web/` cache (with range-request seeking);
  H.264 clips are served as-is and the originals are never touched, so `clipmotion.py` still
  reads them.

---

## Species identification (phase 2)

Every saved crop gets a **species** name, not just the coarse `animal` label — zero-shot, via
**[BioCLIP 2](https://pypi.org/project/pybioclip/)** (`classify.py`). Because it's zero-shot,
**editing the candidate-species list is free**: the list in `classify.py` (`SPECIES_LABELS`, a
Pacific-Northwest backyard starter set) is just text — tweak it for your yard and re-run, no
retraining. It's resumable and re-runnable; by default only crops without a species are named.

```powershell
.\.venv\Scripts\python.exe classify.py              # name every unlabeled crop (GPU)
.\.venv\Scripts\python.exe classify.py --redo       # re-name everything (after editing labels)
.\.venv\Scripts\python.exe classify.py --watch      # name new crops live, beside the rig (CPU)
```

- **Live by default.** The rig runs `classify.py --watch` itself as a background helper, so the
  dashboard's "Most Recent Visitor" card fills in a species within seconds. It names on the
  **CPU** by default so it never fights the live detector for the GPU. (Disable with
  `--no-classify`; you can still backfill later with `python classify.py`.)
- **The non-animal gate (`clipfilter.py`).** MegaDetector's coarse `animal` class sometimes
  false-fires on a plate of food, a pet bowl, or bare deck — and BioCLIP, being an
  *organism-only* model, can't reject those: it forces every crop onto the nearest real species
  (in this DB they piled up as "brown rat"). So a **general CLIP** runs *first* and answers the
  one question BioCLIP can't — *"is this even an animal?"* — labelling non-animal crops
  `not an animal` and skipping BioCLIP entirely. It's zero-shot too (edit the prompt lists in
  `clipfilter.py`); tune the cut with `python clipfilter.py --sample`, which prints a threshold
  sweep over your own crops and writes nothing.
- **You have the final say.** In the dashboard's **Specimen Catalogue** you confirm (✓), reject
  (✗), or correct (✎, with a free-text option) any label. Human verdicts are *sticky* — never
  overwritten by a re-run — and they feed the per-individual behaviour profiles. Non-critter
  corrections ("cat food", "blur", …) live on a denylist so they vanish from the stats and the
  digest, exactly like `not an animal`.

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

### The suggest-confirm loop (`individuals.py`)

Single crops can't be matched across sessions — but **whole visits can**. Averaging a visit's
best crops into one *prototype* washes out the pose/burst noise: on real data, the same
raccoon's visits on different nights match at **0.83–0.93** while two *different* raccoons
photographed **in the same frame** (same light, same glass — the perfect controlled test,
courtesy of a two-raccoon visit) score only ~0.36–0.42. `individuals.py` builds on that:
every unconfirmed raccoon visit gets a suggestion — the nearest *human-confirmed* visit's
name ("looks like Stan 0.84"), or a *"possibly someone new"* flag — and each confirmation
becomes a new template, so **suggestions sharpen as you confirm**. No training; just
accumulating verified prototypes. Visits with two raccoons present at once get a "2+ raccoons"
badge — detected from simultaneous separated boxes in the stills *and* from clips holding two
sustained motion tracks (`clipmotion.py`), which catch pairs the sparse stills miss (co-arrival
is behaviour signal; their blended prototype never teaches).

The **dashboard's Individuals tab** is the intended surface: a "Who is this?" review queue
with one-click confirm / correct / clear, plus cold-start *visit-groups* to name before
anything is confirmed. Once you've named a few, it adds:

- **Fit to the Cast** — re-fits the unconfirmed remainder against your confirmed individuals:
  "12 more visits look like Stan" with a one-click **bulk confirm**, plus the leftovers that
  *look like nobody on file* clustered into **candidate new individuals**. It also flags anyone
  confirmed **only on a multi-animal visit** (their template is a blend of two raccoons and
  can't be matched) so you know to confirm a *solo* visit for them.
- **Poses** — per individual, clusters that animal's crops by appearance embedding. With identity
  held fixed, the embedding varies by *posture/viewpoint*, so the clusters are the animal's
  characteristic poses (the same pose-binding that defeats cross-individual ID, used in reverse).
- **Clips** — per individual, the behaviour clips that overlap its visits, so you can watch each
  named raccoon move (clips during a 2+-raccoon visit are flagged).
- **Un-blend** — on a 2+-raccoon visit, separate the animals using the **clips**. A clip tracks
  each animal independently, so `clipembed.py` embeds each motion tracklet's frames into the same
  vector space as the stills; clustering a pair visit's tracklets splits the two animals (the peak
  Notch+Elliot visit splits 36/29), each shown with frame-crop thumbnails to name. This is how the
  pair member who is **never solo** finally gets a clean appearance template — the labels land on
  the tracklets (clip-space, separate from the still-space `detections.individual_id`).
- **Clip-space match (findable again)** — once a pair member's tracklets are labelled, that
  clip-space template *finds* them in new visits: un-blend a fresh pair visit and each cluster
  comes **pre-suggested** ("✓ Elliot 86%"), and the review queue shows a distinct **clip-match**
  flag. Measured: the bonded pair separate cleanly given one label each (5/7 new pair visits split
  Notch vs Elliot, margins ~0.25); clip vectors sit in their own similarity regime, so they're
  matched with their own threshold (`reid_clip_match_threshold`), never the still-still cut.

Crops are **click-to-enlarge** everywhere, with ← / → to step through the strip you clicked.

The same calls work from the CLI:

```powershell
.\.venv\Scripts\python.exe embed.py --min-confidence 0.5       # keep vectors fresh first
.\.venv\Scripts\python.exe individuals.py --bootstrap          # cold start: nameable visit-groups
.\.venv\Scripts\python.exe individuals.py --queue              # recent visits + suggestions
.\.venv\Scripts\python.exe individuals.py --refit              # fit the rest to the named cast
.\.venv\Scripts\python.exe individuals.py --confirm 1014 Stan  # confirm one visit
```

---

## Behaviour clips (phase 4 capture)

Stills capture *who* and *when*; a short **video clip** captures *how* — gait, approach speed,
dwell, vigilance, who-defers-to-whom. That's the substance of behaviour, and a confound-robust
second shot at individual ID (a limp reads the same from any angle, where a single still — pose
+ soft glass — does not). **On by default**, and safe to leave on: the clips folder is a
**rolling window** — past the `clips_max_gb` disk budget (default 10 GB ≈ two busy weeks) the
oldest clips are pruned automatically, file and DB row both. (Want to keep them all? `backup.py`
archives each day before the pruner reaches it — see [Backups](#backups).)

```powershell
.\.venv\Scripts\python.exe backyard_cam.py                           # clips record by default
.\.venv\Scripts\python.exe backyard_cam.py --no-record-clips         # ...this run, stills only
.\.venv\Scripts\python.exe backyard_cam.py --clip-classes animal person   # also record yourself (test)
```

- A rolling **pre-roll buffer** means each clip opens on the animal *arriving* (the seconds
  before the detector first fired); recording ends a few seconds after the last detection, or at
  a safety cap so a camped-out raccoon can't make a giant file.
- **`clipmotion.py`** turns each clip into **motion fingerprints, one per animal** — the
  detector runs over every frame, double-boxes are suppressed (NMS), and boxes are associated
  into per-animal *tracklets* (a pair visit = two tracks). Each trajectory yields duration, path
  length, straightness (beeline = 1, milling about ≈ 0), avg/peak speed, moving fraction (vs
  head-down eating), an approach/retreat cue — plus a **gait estimate**: stride cadence (Hz)
  from the body-bob periodicity while walking, with the autocorrelation strength saying how much
  to trust it. The raw track is stored as JSON (`clip_tracks` table) so richer gait work (limp
  asymmetry) can re-derive later without re-running the detector. Batch + resumable:
  `python clipmotion.py` after clips accumulate; `--show` reads tracks back; `--report` groups
  motion features by individual/visit and prints the pair-clip same-conditions comparisons.
- One `.mp4` per visit under `clips/<camera>/<date>/`, plus a row in the **`clips`** table (time span,
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
  clips/glass_door_cam/2026-06-07/   behaviour video clips, per camera (only if --record-clips)
  reid/raccoon/        re-ID contact-sheet montages (regenerated by reid.py)
```

Crop filenames look like `2026-06-07T19-25-59-123_0_animal_0.94.jpg`
(`timestamp_index_class_confidence`).

---

## Backups

The code lives on GitHub and the model weights re-download themselves, but the **clips, crops
and database exist only on your disk**. `backup.py` archives exactly that generated content
into a folder of your choosing — point it at one your **cloud client already syncs** (Google
Drive, Dropbox, OneDrive) and off-site backup is just the sync client doing its job. Set the
destination once in `config_local.py` (`cfg.backup_dest = Path(r"C:\my-cloud-folder\backyard")`),
then:

```powershell
.\.venv\Scripts\python.exe backup.py             # archive everything new
.\.venv\Scripts\python.exe backup.py --dry-run   # ...or just say what it would do
```

- **One zip per camera per day** for media. mp4 and JPEG are already compressed, so the media
  zips are *stored*, not re-compressed — the zip exists because one ~75 MB file syncs to the
  cloud far faster than ~2,000 loose JPEGs. The **database** is snapshotted with SQLite's
  backup API on a read-only connection (safe while the rig is running), integrity-checked,
  then deflated (~40% smaller); a `meta-<date>.zip` picks up the small stuff (re-ID artifacts,
  tuning shots, logs, your `config_local.py`).
- **The backup outlives the clip pruner.** `clips/` is a rolling window (`clips_max_gb`), so
  run the backup at least **weekly**: each day folder is archived the morning after it
  completes, well inside the ~two-week prune horizon, and an existing clips archive is never
  rebuilt from a (possibly since-pruned) source. Crops archives *do* refresh if a past day
  gains files — a trail-cam import backfilling old dates.
- **Idempotent** — run it as often as you like; finished days are skipped in seconds. Restore
  instructions land in a `README.txt` beside the archives (short version: unzip everything
  into the project root).

Schedule it weekly with Windows Task Scheduler (runs as you, no admin needed):

```powershell
schtasks /Create /TN "Backyard critter-cam backup" /SC WEEKLY /D MON /ST 03:30 `
    /TR "C:\path\to\backyard\.venv\Scripts\pythonw.exe C:\path\to\backyard\backup.py"
```

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
| `crop_quality` | Phase 2: image shot-quality (sharpness × night eyeshine; `quality.py`) — picks the "cutest" thumbnail. |
| `species` | Phase 2 (`classify.py`, BioCLIP 2) fills it; `not an animal` for crops the prefilter rejects. |
| `species_confidence` | Phase 2 classifier score 0–1 for `species`. |
| `species_verified` | Human review in the dashboard: NULL = unreviewed, 1 = confirmed, 0 = wrong. |
| `species_source` | `bioclip` / `clip-filter` (auto) or `human` (corrected in the dashboard). |
| `individual_id` | Phase 3 (re-ID): set when you confirm a visit or hand-label a cluster. NULL until then. |
| `individual_source` | How `individual_id` was set (e.g. `human`, `refit`). |
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

Two further tables back the clip-based behaviour and un-blend work. **`clip_tracks`** (built by
`clipmotion.py`) holds one **motion fingerprint per animal per clip** — duration, path length,
straightness, average/peak speed, moving fraction, an approach/retreat cue, and a gait estimate
(stride cadence + how much to trust it) — plus the raw trajectory as JSON, so richer gait
features can be re-derived later without re-running the detector. **`clip_track_embeddings`**
(`clipembed.py`) holds one MegaDescriptor appearance vector per tracklet, in the *same* vector
space as `detection_embeddings` — that shared space is what lets a pair visit be split into its
two animals and a never-solo individual finally get a clean template.

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
sensitivity, model version, confidence threshold, which detector classes are reported
(`detect_classes`) and which get saved (`save_classes`), output paths, the behaviour-clip
settings (pre/post-roll, disk budget, codec), the species and non-animal-prefilter settings,
and the re-ID thresholds. The CLI flags above just override the common ones per run; the live
camera controls are also adjustable from the dashboard's **Instrument Panel** without a restart.

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

## Tests

A pure-logic test suite covers the data-shaping code — visit collapsing, behaviour profiles,
the two-axis readout, the non-animal gate's scoring, clip motion + embeddings, shot-quality, and
the DB layer — with **no GPU, camera, or model download needed**:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -q
```

---

## Roadmap

Full plan and design philosophy: **[PLAN.md](PLAN.md)**. The short version:

- **Phase 1 — Live capture skeleton:** ✅ MOG2 gate → MegaDetector → crop + clip + SQLite row,
  with a live preview window and the web dashboard.
- **Phase 2 — Species ID:** ✅ `classify.py` names every crop zero-shot with **BioCLIP 2**, live
  beside the rig, behind a general-CLIP non-animal gate (`clipfilter.py`); confirm/correct in the
  dashboard.
- **Phase 3 — Individual re-ID:** ✅ built. `embed.py` → `reid.py` for appearance clustering, then
  the **suggest-confirm loop** (`individuals.py`): single crops can't be matched across sessions,
  but **visit prototypes can** (same raccoon 0.83–0.93 across nights), so every visit gets a
  "looks like Stan" suggestion that sharpens as you confirm. Pair visits are **un-blended** from
  the clips, and once labelled those clip-space templates find the animal again.
- **Phase 4 — Behaviour + the disagreement alert:** ✅ built (see *Behaviour analysis* above):
  `visits.py` (visit events), `behavior.py` (arrival windows, dwell, co-occurrence), `reid.py
  --match` (appearance score), and `twoaxis.py` — the **two-axis readout** that flags "looks like
  X but isn't acting like X." Clips record by default and `clipmotion.py` turns them into
  motion/gait fingerprints. Sharpens per-individual as you hand-label.
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
