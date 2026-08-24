# Backyard Critter Cam

[![tests](https://github.com/unclemattmakes/backyard-critter-cam/actions/workflows/tests.yml/badge.svg)](https://github.com/unclemattmakes/backyard-critter-cam/actions/workflows/tests.yml)
[![license: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)

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
roadmap live in [the plan](docs/plan.md).

**No camera handy? Start with the making-of site, live at
[unclemattmakes.github.io/backyard-critter-cam](https://unclemattmakes.github.io/backyard-critter-cam/).**
It's a static, camera-free walk-through of the whole system — eight interactive demos built on a
frozen slice of this rig's real database (real crops, real embeddings, real mistakes). No GPU, no
install, no clone — it's a web page. (Offline, or hacking on it? It's the [making-of/](making-of/)
folder; serve it locally with `python -m http.server 8011 --directory making-of` and open
`http://localhost:8011` — it fetches its data, so the GitHub file viewer can't run it.)

**Have a folder of photos from any camera?** You don't need the live rig to try the pipeline —
see [Try it on footage you already have](#try-it-on-footage-you-already-have).

---

## Contents

- [How it works](#how-it-works) — the pipeline, one box at a time
- [The making-of site](https://unclemattmakes.github.io/backyard-critter-cam/) — the system
  explained through eight interactive demos; **no hardware needed** (its source and export
  workflow: [making-of/README.md](making-of/README.md))
- [Try it on footage you already have](#try-it-on-footage-you-already-have) — a folder of
  photos from any camera or phone, straight to a populated dashboard
- [Notes on the detector](#notes-on-the-detector) — why Ultralytics directly, not PytorchWildlife
- [Requirements](#requirements) · [Setup](#setup) · [Running](#running) — including
  [several cameras at once](#multiple-cameras-usb--networked) and the
  [web dashboard](#web-dashboard---serve)
- [Species identification (phase 2)](#species-identification-phase-2) — BioCLIP 2, and how far
  to trust a label
- [Individual re-identification (phase 3)](#individual-re-identification-phase-3) — the
  suggest-confirm loop
- [Behaviour clips (phase 4 capture)](#behaviour-clips-phase-4-capture) ·
  [Behaviour analysis (phase 4)](#behaviour-analysis-phase-4)
- [Output](#output) · [Backups](#backups) ·
  [Moving the rig to a new machine](#moving-the-rig-to-a-new-machine) ·
  [A morning email](#a-morning-email) ·
  [Database schema](#database-schema) · [Configuration](#configuration)
- [Security & privacy](#security--privacy) — there is no login; read this before the dashboard
  leaves your machine
- [Responsible use](#responsible-use) — it is a camera pointed at the outdoors
- [Tests](#tests) · [Roadmap](#roadmap) · [Contributing](#contributing) ·
  [License & attribution](#license--attribution) — the model weights have their own terms
- [Troubleshooting](#troubleshooting)

---

## How it works

```
Camera feed (OpenCV: DirectShow for a USB webcam, FFMPEG for an RTSP/HTTP stream)
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
- **`ffmpeg` and `ffprobe` on your PATH — strongly recommended, not required.** Two features are
  built on them. Clips are recorded by piping frames to ffmpeg (`clip_codec = "h264"`); with no
  ffmpeg the recorder falls back to OpenCV's `mp4v` writer, and no browser decodes `mp4v`, so
  those clips won't play in the dashboard — the server's on-demand transcode is *also* ffmpeg, so
  there's nothing to fall back to. And the Dispatch's stitched highlight reel is an ffmpeg concat,
  so without it the reel returns `ffmpeg not found on PATH` and the panel stays empty. Everything
  else — detection, crops, species, re-ID, motion tracks — works fine without it.

  ```
  winget install Gyan.FFmpeg     # Windows (or: choco install ffmpeg)
  sudo apt install ffmpeg        # Debian / Ubuntu
  brew install ffmpeg            # macOS
  ```

> **Which torch build fits which GPU (important — two silent traps).** CUDA wheels drop kernels
> for old GPU generations and don't yet have them for the newest, and both failure modes are
> quiet: `torch.cuda.is_available()` says `True` and then either the first real op dies
> (`no kernel image is available`) or — worse — `--device auto` silently runs the CPU forever
> while you believe the GPU is working. The setup scripts read your card's compute capability
> (`nvidia-smi --query-gpu=compute_cap`) and pick the right index for you; this table is the
> same decision on paper:
>
> | Your NVIDIA GPU | Compute capability | torch build to install |
> |---|---|---|
> | RTX 50-series (Blackwell — this rig's 5050) | sm_120 | **cu130** (a stock cu126 dies at the first real op) |
> | RTX 20/30/40-series, GTX 16 (Turing→Ada) | sm_75–sm_89 | cu126 **or** cu130 — both carry kernels |
> | GTX 9/10-series, Titan V (Maxwell/Pascal/Volta) | sm_50–sm_70 | **cu126** — cu130 has NO kernels for these; installing it silently costs you the GPU |
> | none / AMD / Intel | — | CPU wheels (`--index-url .../whl/cpu`); macOS uses the default wheel (MPS/CPU) |
>
> The app verifies the build at startup by running a real GPU op: with `--device cuda` a wrong
> build fails loudly and early; with the default `--device auto` it falls back to the CPU
> instead (never stuck — just slower, so check the startup line if you expected the GPU). If
> you'd rather not chase CUDA wheels at all, a CPU-only install works everywhere.

---

## Setup

Get the code first (or grab the ZIP from GitHub):

```bash
git clone https://github.com/unclemattmakes/backyard-critter-cam.git
cd backyard-critter-cam
```

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

**The full first-run download budget.** The detector's weights are small, but the naming helper
starts with the rig by default and the re-ID pass runs nightly — and their models are
*gigabytes*, fetched once from Hugging Face. On a slow or metered connection the first run can
look like a hang while a checkpoint streams down; it isn't. Sizes are approximate:

| Model | ~Size | Downloads when | From | Integrity |
|---|---|---|---|---|
| MegaDetector v6 (yolov10-c) | tens of MB | first detection | Zenodo (archival, DOI) | SHA-256-pinned |
| BioCLIP 2 (species names) | ~2 GB | first crop the naming helper labels (it starts with the rig) | Hugging Face | unpinned — see SECURITY.md |
| general CLIP ViT-B-32 (is-it-an-animal gate) | ~0.6 GB | alongside BioCLIP | Hugging Face | unpinned |
| MegaDescriptor-L-384 (re-ID appearance) | ~1 GB | first `embed.py` run (the nightly batch) | Hugging Face | unpinned |

Each is one-time; they land in the Hugging Face cache (`~/.cache/huggingface`) and the
`weights/` folder. Don't want the big ones yet? `--no-classify` runs capture + detection only.

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

#### Telling someone else how to connect

The LAN launcher gives the rig a **name on your network**, so nobody has to be handed an IP
address — or a port:

> **`http://critter-cam.local`**

That's the whole address. Nothing after it. It works from any phone, tablet or laptop on the same
Wi-Fi, and — unlike the numeric address — it keeps working after your router hands the rig a
different DHCP lease, so it's safe to write down. The rig also advertises itself as a web service,
so it turns up **by name** in network browsers (macOS Finder's *Network*, Discovery on iOS,
`avahi-browse` on Linux) without anyone typing anything.

The missing `:8000` is the other half of the trick, and it's why the dashboard now defaults to
**port 80**: that's the port a browser assumes, so serving there is what lets the address be a
bare name. Port 80 isn't always available — it needs root on Linux/macOS, and anything web-shaped
may already hold it — so a failed bind **falls back to `web_port_fallback`** (default 8000),
prints why, and every address the rig prints afterwards reads the port off the actual socket.
A rig that can't have 80 keeps its dashboard; it just has a longer address.

The launcher window prints the name **and** the numeric address once the dashboard is actually up,
and then stays open so you can go back and read it. To ask at any other time:

```bash
.venv\Scripts\python.exe mdns.py --host 0.0.0.0
```

A few things worth knowing:

- **Android is the weak link.** iOS, macOS and Windows resolve `.local` reliably; some Android
  browsers don't do mDNS at all. That's why the numeric address is always printed next to the
  name — if the name doesn't work on a phone, use the number.
- **Rename it per rig** with `cfg.mdns_name` in `config_local.py` (`"front-yard-cam"`) if you end
  up with two. Spaces and punctuation are fine to type — the name is sanitised to a legal DNS
  label. `cfg.mdns = False` turns the announcement off entirely.
- **Windows will not prompt about the firewall again**, and that's correct — its "Allow access"
  rules are scoped to the *program* with `LocalPort: Any`, not to one port, so the permission
  `python.exe` already has covers 80 too. What *does* decide whether other devices can reach the
  rig is the **network category**: those rules are per-profile, and the usual pair is *Allow* on
  Private and *Block* on Public. Windows classifies new Wi-Fi as Public by default, and Block
  wins over Allow — so on a Public-classified network the dashboard is unreachable from any other
  device no matter which port it serves. Check with `Get-NetConnectionProfile`; if it says
  `Public`, set that network to Private (Settings → Network & Internet → Wi-Fi → your network →
  *Private network*). This is not new with port 80 — it applied equally on 8000.
- **Only in LAN mode.** A localhost-bound rig publishes nothing — there'd be nobody to tell, and
  a name resolving to an address that refuses every caller is worse than no name.
- **If the name is already taken**, the rig checks who has it. Its own leftover record — what a
  crash or a power cut leaves cached on the network, since a rig that dies never sends its goodbye
  packets — is reclaimed, because `rigwatch.py` restarts this rig after it dies and the name must
  survive the reboots nobody is watching. A *different* device holding the name is left alone: the
  rig says so, names the address, and serves on numbers rather than giving two machines one name.

Needs the `zeroconf` package (in `requirements.txt`). Without it the rig runs exactly as before,
says so in one line at startup, and serves on numbers.

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
.\.venv\Scripts\python.exe backyard_cam.py --serve        # then open http://127.0.0.1
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
| `--record-clips` / `--no-record-clips` | Record a short video clip around each visit (default ON, disk-capped to `clips_max_gb` — or a per-camera budget from `clips_max_gb_by_source` — with oldest-first pruning). |
| `--clip-classes C…` | Detector classes that trigger a clip (default = saved = `animal`); e.g. `--clip-classes animal person` to record yourself as a test. |
| `--db PATH` / `--crops-dir PATH` | Override output locations. |
| `--no-preview` | Headless; quit with Ctrl+C. |
| `--no-classify` | Detection only — don't start the live species-naming helper. The rig launches it (and stops it) automatically by default; this turns that off. You can still fill species later with `python classify.py`. |
| `--stats` | Print a DB summary (crops vs. visits, per-hour activity, latest catches) and exit. Read-only. |
| `--list-cameras` | Probe camera indices and exit (find the right `--camera-index`). |
| `--serve` | Also serve the local web dashboard (live stream + stats) at `http://host:port`. |
| `--port N` / `--host H` | Dashboard port (default 80, falling back to 8000 if 80 is taken) / bind host (default `127.0.0.1`; `0.0.0.0` = LAN). |

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
    cfg.latitude, cfg.longitude = 40.7128, -74.0060
    cfg.cameras = [
        CameraSpec("glass_door_cam", 0, name="Glass door"),                 # USB webcam, index 0
        # ...or the same camera over the network, once it outgrows one PC -- a Raspberry Pi
        # running ustreamer forwards a USB webcam's own JPEG frames untouched:
        # CameraSpec("glass_door_cam", "http://192.168.1.60:8080/stream", name="Glass door"),
        CameraSpec("yard_ir", "rtsp://user:pass@192.168.1.50:554/h264Preview_01_sub",
                   name="Yard (night IR)"),                                  # RTSP/PoE IP camera
        CameraSpec("feeder_esp32", "http://192.168.1.51:81/stream",
                   name="Feeder (ESP32)", frame_width=640, frame_height=480, motion_min_area=300),
    ]
```

**A camera does not have to stay bolted to the rig.** Moving a USB webcam onto a cheap
single-board computer and pointing the rig at it over HTTP is the same three-line change as any
other network camera — and if the bridge forwards the camera's **own JPEG frames** rather than
re-encoding them, the pixels the detector sees do not change, so `motion_min_area`, the ignore
zones and `crop_quality` all stay comparable across the move. That is how this rig's glass-door
camera survived being separated from the machine that watches it.

`src` is anything OpenCV's `VideoCapture` accepts: an **int** webcam index, or a **URL** — `rtsp://…`
for an IP/PoE camera, or `http://…/stream` for an ESP32-CAM's MJPEG server. Per-camera fields
(resolution, `motion_min_area`, day/night profile, `record_clips`) default to the global config;
only override what differs. Each camera needs a **unique `source`**.

**Or add one without editing Python.** Since 2026-08-22 the camera list lives in the database and
the block above only **seeds** it the first time. The dashboard's **cameras** button (top right of
Live Observation) adds, edits and removes cameras — name, address, stream path, login, resolution,
motion area. Two things it deliberately does not pretend:

- **A change applies at the next restart, not immediately.** Each camera gets its own capture
  thread when the rig starts, so unlike ignore zones a camera cannot be attached to a running rig.
  The panel says so.
- **A camera's short name is permanent.** It is stamped on every detection, visit and clip folder
  recorded under it, so renaming would orphan all of it. Removing a camera keeps those rows, and
  re-adding the same short name reattaches to them.

A camera **password** can only be set from the rig machine itself, never over the network — see
[SECURITY.md](SECURITY.md#camera-credentials--the-one-exception). Everything else is editable from
any device an operator can reach.

**Adding one, step by step:** [docs/runbook-add-network-camera.md](docs/runbook-add-network-camera.md)
is the operational sequence — including the two that look like faults and aren't (RTSP ships
*disabled* on some cameras and needs a **reboot** after enabling; the vendor app's login is
often an account with the vendor, not a user on the camera, so RTSP rejects it). Probe a camera
before you write its URL into a config with:

```
python tools/camprobe.py <ip> --user <camera-user>      # CAM_PASS from the environment
```

It walks port → RTSP challenge → credentials → which stream paths exist, measures what each
stream costs to decode, suggests a `motion_min_area` for its resolution, and saves a test frame.
Passwords are masked in everything it prints.

**`camprobe` speaks RTSP only** — it refuses any other scheme rather than guessing, so it cannot
check an HTTP MJPEG source (a Pi bridge, an ESP32-CAM, a phone app). Those have a simpler test
anyway: **open the stream URL in a browser tab.** If it plays there, it will open for the rig,
and you have also answered the Protocol dropdown's question.

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

### No yard camera yet? Two zero-hardware test drives

Both fall straight out of `src` being "anything `VideoCapture` accepts":

- **Your phone is a working yard camera tonight.** Any free IP-webcam app that serves MJPEG over
  HTTP turns a spare phone into a camera the rig treats like any other:

  ```python
  cfg.cameras = [CameraSpec("phone", "http://192.168.1.23:8080/video", name="Phone on the sill",
                            frame_width=1280, frame_height=720)]
  ```

  (Match `motion_min_area` to the resolution — it's pixels of the largest motion blob.)

- **A downloaded video is a full pipeline demo.** A plain **local file path** works as a source,
  so you can point the rig at any wildlife video and watch the whole thing run — motion gate,
  boxes, crops, species names — before owning hardware:

  ```python
  cfg.cameras = [CameraSpec("test", r"C:/downloads/raccoon_visit.mp4", name="Canned test")]
  ```

  The rig reads the file like a camera — end-of-file looks like a dropped feed, so the
  reconnect logic reopens it and the video **replays on a loop** until you quit (fine for a
  demo; just expect duplicate visits per lap). Crops and visits land in the DB exactly as live
  ones would (delete the DB after, or keep it — it's just rows with that `source`).

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
`http://127.0.0.1`) — a field journal for the yard you can leave open in a browser tab. It
has grown from a single live feed into **nine tabs**:

- **Live Observation** — the live annotated MJPEG feed, the most-recent-visitor card (species,
  how-long-ago, confidence, a ▶ badge to play its clip, and a live *off-pattern* flag if it
  arrived at an odd hour for its species), running tallies, and most-/least-common species. A
  **Who's visiting right now?** panel lets you name the live visit as it happens: tap a known
  critter (or add a new name) and **Log who's here** — one name tags that visit's crops with the
  individual (a live solo confirm that feeds the [re-ID](#individual-re-identification-phase-3)
  templates), two or more record who came *together* (co-presence, without mislabelling both
  animals as one). A short *recently logged* list shows your notes landing.
- **Visit Log** — *the landing page*: the yard's comings and goings as scrollable cards (species
  mix, any named individuals, dwell, ▶ its clips), with the named cast across the top so any
  individual is one tap from their profile. The "scroll around and see what happened" surface.
- **Favourites** — the album. A ♡ on any visit card or photograph keeps it, and this tab
  collects them: kept visits as their own playable cards, kept photographs as a grid, each with
  an optional note ("the night the kits came out") and the name of whoever kept it. Unlike every
  other verdict on the dashboard, a ♡ is **only taste** — it changes no label and nothing
  downstream reads it, so keeping the cutest photo of Stan can never become evidence that it *is*
  Stan. A kept visit is remembered as a moment on a camera rather than a visit id (those get
  renumbered), so it survives re-clustering — star a visit while the animal is still on camera
  and the card grows to the finished span. A favourite whose crops are later purged is listed as
  gone rather than quietly disappearing.
- **The Dispatch** — a newspaper-style **period digest** (☾ Night / ☀ Day,
  with ‹ › arrows to walk back through earlier days). It leads with a **condensed highlight
  reel** — the period's best moments auto-cut into one short stitched video (~1 minute for a
  busy night, built by `reel.py` from motion tracks + shot quality, one beat per species /
  named individual / activity burst; chapters in the player jump between moments, and there's
  a ⤓ save link for sharing). Below it, **The Visits** — the period's comings and goings in
  order (time, species, any named individuals, dwell, motion one-liner, ▶ its clips) — then
  the cast roll call (recent faces as cards, the long-gone in one quiet line), novelty & quiet
  flags ("first raccoon in 9 days"), moon phase, the "plate of the night" hero shot, and the
  full species roll with per-hour activity clocks. The full every-clip playlist is still one
  click away ("watch every clip").
- **Behaviour** — per-species field notes (visits/day, median dwell, typical arrival window,
  hourly chart, and the **sun-anchored** arrival — "~1h48 after dusk", which stays true as the
  season walks sunset around), **off-pattern** alerts, **seen-together** co-occurrence pairs,
  **yard politics** (who avoids whom, who yields the yard — observational, with sample floors),
  and a **moonlight** scatter of nocturnal traffic against illumination.
- **Seasons** — the longitudinal view the calendar can't give: per-species weekly sparklines,
  first/last dates, and the yard's **species-accumulation curve** (is it still introducing
  itself, or is the cast known?).
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
focus, white balance — read from and pushed to the *running* camera, so you can dial in the shot
without restarting (a live companion to `tune.py`). **USB cameras only.** A networked camera
owns its own exposure and white balance, and there is no UVC control channel over a stream, so
the panel shows a note instead of sliders for those — which is also the quickest way to confirm
the rig really is treating a camera as networked.

- Built on Python's stdlib `http.server` — **no web framework, no new dependencies**.
- **Localhost only** by default (it shows your camera), with LAN access an explicit opt-in and
  **no login at any point** — read [Security & privacy](#security--privacy) before you make it
  reachable from anything but the machine it runs on.
- Runs in the same process as capture; combine with `--no-preview` for a headless,
  browser-only rig, or keep the native window too.
- **Species names appear on their own:** the rig starts the naming **helper** (`classify.py
  --watch`) as a child process and stops it with the app, so the recent-visitor card fills in a
  species by itself — nothing extra to launch or close. The helper takes a minute to warm up its
  model; the header shows **"Identifier: warming up… / on"** so you can tell it's working. (If a
  previous run was killed hard and left its helper behind, the next launch sweeps it up — see
  [Troubleshooting](#troubleshooting).)
- **Clips just play.** Clips recorded in legacy `mp4v` aren't browser-playable, so the server
  transcodes them to H.264 **on demand** into a `clips_web/` cache (with range-request seeking);
  H.264 clips are served as-is and the originals are never touched, so `clipmotion.py` still
  reads them. Both halves of that need [ffmpeg](#requirements) — with none on PATH the original
  is served and the browser refuses it.

---

## Try it on footage you already have

`import_trailcam.py` is named for this rig's trail cam, but it is really a **generic
batch importer**: point it at any folder of JPG/PNG photos — an old trail-cam dump, a phone's
camera roll from the garden, doorbell-cam exports — and it runs the same detector over them and
writes the same crops, visits and database rows the live rig would. Then serve the dashboard
over the result. No camera, no GPU required (CPU just takes longer), and your first populated
Dispatch/Calendar/Catalogue is on **your own animals**:

```bash
python import_trailcam.py my_folder --source my_yard
python classify.py
python backyard_cam.py --serve-only
```

then open `http://localhost`. Three honest caveats:

- **Photos are the input that matters.** Detection runs on stills; MP4s alongside them ride in
  as playable behaviour clips (paired to nearby animal photos), but a videos-only folder
  populates nothing.
- **Timestamps come from EXIF** (`DateTimeOriginal`), falling back to file modification time —
  so import from the original files, not copies whose mtimes were rewritten by the copy.
- **Name the `--source`** and keep using the same name for the same camera: everything
  downstream — visits, re-ID template scoping, prune budgets — keys off it.

---

## Species identification (phase 2)

Every saved crop gets a **species** name, not just the coarse `animal` label — zero-shot, via
**[BioCLIP 2](https://pypi.org/project/pybioclip/)** (`classify.py`). Because it's zero-shot,
**editing the candidate-species list is free**: the list in `classify.py` (`SPECIES_LABELS`, a
Pacific-Northwest backyard starter set) is just text — tweak it for your yard and re-run, no
retraining. Not in the PNW? **[docs/species-lists.md](docs/species-lists.md)** has ready-made
starter lists (US NE/SE/SW, UK & Ireland, Central Europe, Australia east coast) plus the
phrasing rules that make labels resolve well — set yours via `cfg.species_labels` in
`config_local.py`, no source edit. It's resumable and re-runnable; by default only crops without a species are named.
Whenever it writes labels it also refreshes the **visit ledger**, so each visit's dominant
species tracks the new labels with no manual `visits.py` step.

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
- **How far to trust a label — and the day/night catch.** A read-only eval harness (`eval.py`)
  grades the auto labels against your confirmed/corrected verdicts and writes a timestamped JSON
  to `reports/` (gitignored — so these are numbers to *reproduce*, not to read out of the repo).
  Its headline finding: species labels at **`species_confidence` ≥ 0.8 are ~94% accurate overall —
  but that trust does not survive the dark.** Day rows grade **96%**; **night rows ~63%**, and even
  the *confident* night rows only reach 68%, because through-glass IR is monochrome and soft, so a
  confident night label is *far* likelier to be wrong than a confident day one. Read night labels
  with more suspicion.
  **Know how thin that is.** On the **2026-07-01** run those percentages come from **165 gradable
  rows** — only 165, because a human correction overwrites `species`, so the model's own call
  survives on the prediction-intact slice alone. Of those, **138 are day and 27 night**, and
  **129 are raccoon**. So "94% / 63%" is really *one species, one camera, three weeks*, and the
  night figure rests on 27 rows. Treat it as the shape of the problem (confidence lies in the
  dark), not as a benchmark, and re-run `eval.py` on your own corpus — after you change the label
  list or a threshold — to see your numbers. (The grading only works because `model_species`
  snapshots the model's original call, so your corrections *teach* the eval instead of erasing the
  thing it scores.)

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
  [the plan](docs/plan.md)). Hand-labelling the obvious clusters + the neighbour lookup is the
  intended workflow; behaviour (phase 4) is the other axis.

### The suggest-confirm loop (`individuals.py`)

Single crops can't be matched across sessions — but **whole visits can**. Averaging a visit's
best crops into one *prototype* washes out the pose/burst noise: when the cast was **two**
raccoons, the same raccoon's visits on different nights matched at **0.83–0.93** while the two
*different* raccoons photographed **in the same frame** (same light, same glass — the perfect
controlled test, courtesy of a two-raccoon visit) scored only ~0.36–0.42. `individuals.py` builds
on that: every unconfirmed raccoon visit gets a suggestion — the nearest *human-confirmed* visit's
name ("looks like Stan 0.84"), or a *"possibly someone new"* flag — and each confirmation
becomes a new template, so **suggestions sharpen as you confirm**. No training; just
accumulating verified prototypes. Visits with two raccoons present at once get a "2+ raccoons"
badge — detected from simultaneous separated boxes in the stills *and* from clips holding two
sustained motion tracks (`clipmotion.py`), which catch pairs the sparse stills miss (co-arrival
is behaviour signal; their blended prototype never teaches). The *"possibly someone new"* cut
(`reid_novel_threshold`) is set to the similarity `eval.py` measured as best-separating
same-from-different across *all* your confirmed visits — currently **0.31**, not the eyeballed 0.55
it began with — so re-run `python eval.py --reid` as the cast grows and it reports the new optimum.

**That gap narrowed as the cast grew — the honest, dated version.** 0.83–0.93 against 0.36–0.42
was measured with **two** raccoons on file. The most recent sweep (**2026-07-18**, five raccoons,
113 leave-one-out probes over solo visits) reports **ROC-AUC 0.635** and **top-1 identification
0.637**: same-individual visit pairs now average 0.44 and different-individual pairs 0.32, which is
a real signal and a badly overlapping one. The code didn't regress — the problem got harder, and
five look-alike raccoons through one pane of glass is a truer measure of the difficulty than two
were. It's lopsided per animal, too: the two best-templated raccoons score 40/46 and 31/48 while
the three thin ones manage 1/19 between them, and some of that spread is probably label noise (one
misnamed visit poisons its own template), which is another reason to re-measure rather than
believe. `reports/` is gitignored, so these aren't numbers you can read out of a checkout — run
`python eval.py --reid` and get your own.

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

### Keeping it automatic (the nightly batch + auto-assign)

Suggestions are only as fresh as the vectors behind them, so the nightly batch
(`run_clipmotion.bat`) keeps the whole loop fed without you thinking about it: motion tracks for
new clips → still embeddings (all species, down to the 0.5 suggestion gate) → clip-tracklet
embeddings → solo-track linking → **auto-assign**. Every step is resumable, so schedule it daily
in an activity trough (like the backup task, no admin needed):

```powershell
schtasks /Create /TN "BackyardCritterCam-MotionTracks" /SC DAILY /ST 14:00 `
         /TR "\"$PWD\run_clipmotion.bat\"" /F
```

**Auto-assign** is the *review by exception* tier: a solo visit whose best match clears **both**
an eval-measured similarity bar *and* a lead-over-the-runner-up margin gets named automatically,
stamped `individual_source='auto'`. Auto names count on tracking surfaces (roll call, last-seen)
but deliberately **never feed the suggestion templates** and never ground behaviour links — a
wrong auto name can't teach the matcher anything. In the queue each one shows as **auto: Stan**
with **✓ keep** (promotes it to a real, template-feeding confirmation) and **✗ not them** (clears
it *and* pins the visit so the nightly pass won't re-name it).

It ships **disabled** — `reid_auto_threshold = 0.0` — and that is deliberate: an operating point is
a property of one corpus, one camera and one cast, so the default must not write machine-made names
onto a stranger's animals. To turn it on, **measure it on your own data first**: `python eval.py
--reid` ends with an auto-assign sweep that reports the max-coverage bars making **zero wrong calls
and zero novel-animal false-accepts across your confirmed visits**, and you set
`reid_auto_threshold` / `reid_auto_margin` from that. For reference, on a five-raccoon corpus
(2026-07-18) the zero-error point was **0.76 / 0.12**, and even there it only named 17 of 113
visits — 15% coverage. Useful as a place to start a sweep, not a default to trust: those bars are
tuned to that cast's look-alikes, not yours. Re-run the eval as the cast grows.

---

## Behaviour clips (phase 4 capture)

Stills capture *who* and *when*; a short **video clip** captures *how* — gait, approach speed,
dwell, vigilance, who-defers-to-whom. That's the substance of behaviour, and a confound-robust
second shot at individual ID (a limp reads the same from any angle, where a single still — pose
+ soft glass — does not). **On by default**, and safe to leave on: the clips folder is a
**rolling window** — past the `clips_max_gb` disk budget (default 10 GB ≈ two busy weeks) the
oldest clip **files** are pruned automatically. The budget is **per source** where you say so
(`clips_max_gb_by_source`, default `trail_cam_sd: 15 GB`), because the sources aren't equally
replaceable: the live rig can always record tomorrow, but a trail-cam clip exists only until the SD
card is formatted, and one shared oldest-first pool let a 9 GB card import evict an equal slice of
glass-door footage. Sources you don't list share `clips_max_gb`. The prune is *soft*: the DB row
stays, stamped
`pruned_at`, so everything the clip taught the system — motion tracks, tracklet appearance
vectors, individual links — **outlives the video** instead of resetting every budget cycle.
Watch/play surfaces simply stop offering the pruned footage. (Want the videos themselves kept
too? `backup.py` archives each day before the pruner reaches it — see [Backups](#backups).)

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
  motion features by individual/visit and prints the pair-clip same-conditions comparisons. Those
  fingerprints now also reach the dashboard live — a per-visit strip and a per-individual summary —
  through `/api/visit/motion` and `/api/individual/motion`, so the motion axis is visible without
  the CLI.
- One `.mp4` per visit under `clips/<camera>/<date>/`, plus a row in the **`clips`** table (time span,
  fps, size, detection count) for later behaviour queries. Crops are still saved alongside.
- All knobs (pre/post-roll, max length, fps, downscale, codec, trigger classes, disk budget)
  live in `config.py`; `clip_scale < 1.0` trims the in-RAM buffer and file size if memory is
  tight; `record_clips = False` turns recording off permanently (incl. the family launchers).
- **If the encoder dies, you'll know.** Scaled clip sizes are rounded down to *even* dimensions
  (libx264 refuses odd ones — `0.667 × 1920 = 1281` once killed ffmpeg on the first frame of
  every clip and a full day of footage vanished in silence); a pipe that dies mid-clip is caught,
  the clip is rebuilt on the OpenCV fallback writer from the pre-roll buffer, and a clip that
  still can't reach disk is dropped with a **loud** `[clips] DROPPED …` log line, never quietly.

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
  raccoons are nocturnal"). The behaviour fit weighs **both** the arrival hour **and** the dwell
  time against the species profile, so a raccoon that shows up at a normal hour but bolts in three
  seconds reads as *off* too. Catches both genuinely unusual visits and mis-classifications; gets
  sharper per-individual as you hand-label with `reid.py --name`.

The dashboard surfaces all of this: a **Behaviour** tab (profiles, off-pattern flags,
co-occurrence) and an **Individuals** tab where you *name the cast* — each look-alike group
shows its best crops with a name box; naming two groups the same name merges them, and the
per-individual axes sharpen as the cast grows. The **motion** signal is visible here too: each
visit in the *Who is this?* queue carries a one-line **motion strip** (approached or retreated,
how direct the path, how much it moved) and each named individual a **motion fingerprint**
aggregated across its clips. Speed stays hedged — it's in frame-fraction units, so a nearer animal
reads faster; direction and straightness are the trustworthy parts.

The visit ledger refreshes itself at every step that changes what it would say: on rig shutdown,
after a trail-cam import, and whenever `classify.py` writes species labels (at the end of a
one-shot run, and in `--watch` each time a naming backlog drains) — so a batch import ends with
*labeled* visits on its own. `visits.py` is only needed manually after offline edits (say,
hand-run SQL).

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
  tuning shots, logs, your `config_local.py`, the certified reference photos and their crops,
  and the database's import/static-dropped ledgers — the import ledger is what stops a
  restored rig from double-importing the trail cam's card).
- **The backup outlives the clip pruner.** `clips/` is a rolling window (`clips_max_gb`), so
  run the backup at least **weekly**: each day folder is archived the morning after it
  completes, well inside the ~two-week prune horizon, and an existing archive is never rebuilt
  from a (possibly since-pruned) source. It *is* topped up, though: files a past day has gained
  since it was archived get merged in — a trail-cam import backfills old dates, and the day you
  dump the card arrives in two batches, because the card goes straight back in the camera and
  the rest of that day comes off it next time. Archives only ever grow.
- **Idempotent** — run it as often as you like; finished days are skipped in seconds. Restore
  instructions land in a `README.txt` beside the archives — and restoring is automated:
  `python migrate.py restore <backup folder>` from a fresh clone reassembles the whole rig
  (see [Moving the rig to a new machine](#moving-the-rig-to-a-new-machine)).
- **Four things beyond the media**, because "the weights re-download themselves" is only true
  for some of them and a database is not the same thing as a readable record:
  - `weights-archive.zip` — a **one-time** mirror of the model weights (MegaDetector plus the
    Hugging Face checkpoints), never rebuilt. MegaDetector comes from Zenodo, which is
    archival; MegaDescriptor, BioCLIP and the CLIP gate come from a hub where repos get pulled,
    gated or re-licensed. If MegaDescriptor vanished, every stored vector would survive and the
    embedding space could never be extended again — the whole cross-time identity archive would
    freeze. Local archival for continuity, not redistribution (see [NOTICE.md](NOTICE.md)).
  - `labels-<date>.jsonl` — every **human verdict** as an append-only ledger, diffed against
    last week's. The label set is the one irreplaceable asset here, and a mass relabel now logs
    loudly instead of silently.
  - `export-<date>.zip` — the observation record as **plain CSV plus a `DATA.md` dictionary**
    (`export.py`). Data longevity shouldn't equal codebase longevity: this opens in a
    spreadsheet in ten years with no Python at all.
  - `STATUS.txt` — a weekly heartbeat: rig freshness, newest human label against the decay
    horizon, shadow-review flag count, trail-card import age, disk headroom. Its **absence or
    staleness** in your cloud app is itself the alarm, which is the only notification channel
    this project has.

Schedule it weekly with Windows Task Scheduler (runs as you, no admin needed):

```powershell
schtasks /Create /TN "Backyard critter-cam backup" /SC WEEKLY /D MON /ST 03:30 `
    /TR "C:\path\to\backyard\.venv\Scripts\pythonw.exe C:\path\to\backyard\backup.py"
```

---

## Moving the rig to a new machine

A rig accumulates the one thing a fresh clone can't give you — months of database, crops,
clips, hand-certified reference photos and labels — so sooner or later the question is how to
carry all of it to a new PC. `migrate.py` is that move, as two halves of one operation:

```powershell
# OLD machine — write a complete, current bundle (USB drive, network share, synced folder):
.\.venv\Scripts\python.exe migrate.py pack F:\rig-move

# NEW machine — clone the repo, run setup.bat/setup.sh, then from the project root:
.\.venv\Scripts\python.exe migrate.py restore F:\rig-move
```

- **`pack` writes the exact same layout as the weekly backup** (per-day media zips, an
  integrity-checked database snapshot, the meta zip, the one-time weights mirror) — plus
  *today's* folders, which the weekly run leaves for tomorrow. One format, so `restore` also
  works pointed straight at your **weekly cloud backup folder**: recovering from a dead
  machine is the same command, you just lose whatever the last backup missed. And because
  these archives are append-only, re-running `pack` only adds the delta.
- **Or pack from the dashboard.** The footer's **move this rig** link (operators) runs the
  same pack from the browser: pick a destination, watch the log live, and get the second-pass
  reminder the moment it finishes. Like a camera password, *starting* one only works from the
  rig machine itself (`http://127.0.0.1`) — it writes gigabytes to a path of your
  choosing, so to aim the rig's disk somewhere, be at the rig. Restore deliberately has no
  UI: it runs on the **new** machine, where no rig — and so no dashboard — exists yet.
- **Packing beside a running rig is safe, but do it in two passes.** The database snapshot
  uses SQLite's backup API (consistent even mid-write) and anything written in the last
  minute is deliberately left for the next pass — the recorder writes clips in place at
  their final name, and a half file sealed into an append-only archive would block the whole
  one forever. So: pack while the rig runs (the slow bulk), **stop the rig, pack once more**
  (seconds), then restore from that.
- **`restore` is the judgement a 2 a.m. hand-restore forgets.** It `quick_check`s the
  database snapshot *before* installing it (falling back to the next-oldest snapshot if the
  newest is corrupt), never overwrites an existing file (so an interrupted restore is just
  re-run, and a `config_local.py` you already wrote wins over the old machine's), installs
  the database **last and atomically**, then cross-checks the rows against the restored
  files and prints the new-machine checklist.
- **Restoring where a rig already lives never merges and never quietly replaces.** Restore
  says what's here (rows, newest detection, which trees), warns that restoring replaces it,
  and — only on an explicit yes — offers to **pack the previous rig into a portable backup
  first**, then moves it *whole* into a `replaced-rig-<timestamp>/` folder before restoring.
  Nothing is ever deleted: the replaced folder is yours to remove once the restored rig has
  proven itself. Scripted use must say it outright (`--replace` plus `--backup-to FOLDER` or
  `--no-backup`); with no terminal and no flags, the old hard refusal stands.
- **Only the clips the old machine still held are restored.** The archive deliberately
  outlives the clip pruner, so it holds more footage than any one disk should; pruned clips
  keep their DB rows and stay playable straight out of the backup zips (the dashboard's
  archive path), exactly as before the move.
- **What never migrates, on purpose:** the `.venv` and CUDA build (per-machine — run setup),
  scheduled tasks (they live in the OS; re-register the weekly backup on the new machine and
  **disable it on the old one**, or the two rigs will interleave writes into one archive),
  and caches that rebuild themselves (`clips_web/`, `archive_cache/`, `refimg_store/`).
  `--dry-run` on either half says exactly what it would do first; `--no-weights` skips the
  ~1.3 GB weights mirror if you'd rather re-download.
- **The new machine doesn't need the old machine's camera.** A configured camera that isn't
  there is retried forever, never fatal — and a rig with no camera at all still serves the
  entire migrated archive via `--serve-only`. Drop the departed camera from
  `config_local.py`, and your history is untouched either way: everything keys off the
  `source` column, so a retired camera simply stops adding rows. A replacement camera on the
  **same view** should reuse the old source name (continuous timeline); a new angle gets a
  new name.

---

## A morning email

`newsletter.py` mails you last night's Dispatch as a small newspaper — the same
`period_digest` the dashboard's Dispatch tab renders, laid out for an inbox: a lede of who
came and when, a hero photo re-judged for *cuteness* rather than raw sharpness (how much
animal is in frame, whether the focus is on the animal or the yard behind it, night eyeshine
as a "facing you" signal — and never an upscaled crop), the visit timeline with thumbnails,
the species roll, and the named cast's roll call. The digest's honesty rules travel with it: crowd counts
are floors, "surprising" species are listed as questions, and a night the camera wasn't
watching says so.

Everything in it **deep-links back into the dashboard** — a visit opens that day, a species
opens its catalogue sheet, a cast member opens their profile, the hero opens that Dispatch.
Those links reach the rig **over your own network only** (the dashboard has no login — see
[Security & privacy](#security--privacy)), so they work from the sofa and not from the bus.
The address is auto-detected per issue: your machine's **LAN IP**, not its hostname, because
phones resolve mDNS rather than NetBIOS and `http://your-pc:8000` simply fails on iOS and
Android. Re-deriving it every morning means a DHCP change heals itself; override with
`cfg.email_dashboard_url` for a fixed name, a different port, or a reverse proxy.

The email keeps using the **number** even though the rig now publishes `critter-cam.local`
(above), and that is deliberate: the paper carries exactly one link per item, it is read on
phones, and some Android browsers don't resolve `.local` at all — a name would trade a link that
always works for one that's prettier and sometimes doesn't. The startup banner can print both
side by side and let the reader pick; an email link can't. Set
`cfg.email_dashboard_url = "http://critter-cam.local"` if everyone in your household is on
iOS.

Sending uses [Resend](https://resend.com)'s REST API (free tier is plenty for one email a
day) via a single stdlib HTTP call — no SDK. Photos are embedded as inline attachments
because the alternatives genuinely fail in mail clients: your dashboard's image URLs are
LAN-only, and Gmail strips `data:` URIs. Set three values in `config_local.py` (never
`config.py` — the key is a secret and this repo is public; see
`config_local.example.py`):

```python
cfg.email_to = "you@example.com"     # or "you@example.com, someone@else.com" (or a list)
cfg.email_from = "The Backyard Dispatch <dispatch@your-domain.com>"  # a Resend-verified domain
cfg.email_resend_api_key = "re_..."
```

Several recipients each get their own copy, addressed only to them — nobody sees anyone
else's address. That is one send per recipient: right for a household list, the wrong shape for a
real mailing list.

Try it, then schedule it (runs as you, no admin needed):

```powershell
python newsletter.py --no-send   # render one issue to reports\mail\ and open it in a browser
```

```powershell
schtasks /Create /TN "Backyard critter-cam morning mail" /SC DAILY /ST 07:00 `
    /TR "C:\path\to\backyard\.venv\Scripts\pythonw.exe C:\path\to\backyard\newsletter.py"
```

Details that take care of themselves: in midwinter a 07:00 task can fire before dawn, so the
script sleeps until the night is actually complete rather than mailing the wrong one; a
visitor-less night still sends a short "quiet night" issue (absence is information — set
`cfg.email_send_quiet = False` to skip those); every issue is also written to
`reports/mail/` as a browser-viewable copy, so nothing is lost while email is unconfigured;
and until all three values are set the scheduled run is a polite no-op. A back-issue is
`python newsletter.py --date 2026-08-10 --edition night`.

---

## Database schema

One row per detection in the **`detections`** table. Timestamps are **local time with UTC
offset, ISO 8601** (e.g. `2026-06-07T19:25:59.123456-07:00`) — the wall-clock time you'd
read off a window-side clock, but globally unambiguous and sortable.

| Column | Notes |
|--------|-------|
| `id` | PK. |
| `timestamp` | Local ISO 8601 with offset. |
| `source` | Which camera wrote the row — `glass_door_cam` for the live rig, `trail_cam_sd` for an SD card run through `import_trailcam.py`, or whatever you name your own cameras in `cfg.cameras`. Everything downstream keys off it. |
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
| `model_species` | The classifier's **original** prediction, snapshotted at classify time so a later human correction to `species` no longer destroys what the model actually said. |
| `model_species_confidence` | The model's own score for `model_species` (unlike `species_confidence`, a human correction doesn't force this to 1.0). |
| `model_species_source` | Which model made that call (`bioclip` / `clip-filter`). Read by `eval.py` to grade the classifier against human verdicts. |
| `individual_id` | Phase 3 (re-ID): set when you confirm a visit or hand-label a cluster. NULL until then. A name containing `" + "` (e.g. `Stan + Kits`) is a **group** label — one archive name over several animals; it is never treated as single-animal ground truth (`db.is_group_label`). |
| `individual_source` | How `individual_id` was set: `human` (the only kind that becomes a template), `auto` (the nightly assigner), `cluster` (a look-alike proposal). |
| `labelled_at` | *When* a label was applied, as opposed to when the animal was seen. Deliberately not backfilled — there is no honest value for rows labelled before the column existed. |
| `labeled_by` | *Which human*. NULL = the operator, before attribution existed. What lets a household eventually supply attributed, reviewable labels. |
| `visit_id` | Phase 4: which `visits` row this crop belongs to (stamped by `visits.py`). |
| `suppressed_at`, `suppressed_by`, `suppress_ref_id`, `suppress_detail` | The furniture veto (`refimg.py`), **shadow mode**: rows it *would* remove are flagged, never deleted, and nothing reads these yet. NULL = live, so every existing query is unchanged. |

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

Smaller tables, each one a fact the code could not otherwise know:

| Table | What it records |
|-------|-----------------|
| `live_sightings` | Your real-time "who's here NOW" log — the strongest label class. Two-plus names (or one `X + Kits` group string) mean several animals, so no single name is stamped across them. Re-logging a span supersedes the earlier row rather than deleting it: the correction sequence is itself signal. |
| `individual_status` | Residency — `departed` with the last day the animal was resident, or `resident` written back to undo it. Keeps the nightly assigner from naming an animal that has left. |
| `life_events` | The cast's story as dated free text ("kits first emerged", "limping on the left front"). Append-only; nothing machine-side reads it. |
| `favorites` | What a human kept — a crop (by `detection_id`) or a visit (by `source` + the moment it started, because visit ids are renumbered on every rebuild), plus an optional note and who kept it. Taste, not evidence: nothing machine-side reads it either. |
| `coverage_events` | When each camera was actually **watching** (`up`/`down` at open, read-failure, reconnect, stop). The effort ledger every absence claim needs — without it a wedged camera reads as an empty yard. Windows before the ledger existed are *unknown*, never "covered". |
| `ignore_zones` | Dashboard-drawn spots the detector should disregard, with tombstoned deletes so a config seed can't resurrect one. |
| `reference_images`, `view_epochs` | Certified-empty frames of each camera and the "the camera moved" events that retire them (`refimg.py`). |
| `identity_references` | Hand-certified photographs of a known individual (`refcam.py`) — identity evidence that does not decay, and that never becomes a matcher template. |

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

## Security & privacy

**There is no login.** No accounts, no passwords. By default anyone who can reach the
dashboard's port sees the live feed, browses every crop and clip, and can edit species labels
and individual names — a deliberate trade for a single-household tool, and the one thing to
keep in mind before the page leaves the machine it runs on. The one optional split:
**`operator_token`** in `config_local.py` turns un-tokened devices into **viewers** — they read
and play everything and can log "who's here" as reviewable testimony, but every label/settings
write is refused server-side until the token is entered once in that browser (dashboard
footer). Localhost is always the operator; leaving the token unset keeps the historical
everyone-operates behaviour.

- **Localhost by default.** The server binds `127.0.0.1` (port 80, or 8000 if 80 is
  unavailable), so nothing off the machine can reach it.
- **LAN access is an explicit opt-in** — `start_critter_cam_lan.bat`, or `--host 0.0.0.0` — and
  even then two guards apply. The peer's IP must be loopback or private/link-local (`lan_only`,
  default `True`), which refuses *direct* connections from the wider internet; and the `Host`
  header must be `localhost`, a private IP literal, your configured `web_host`, or the exact
  mDNS name this rig publishes for itself (`cfg.mdns_name`, default `critter-cam.local`) — which
  is what stops **DNS rebinding**, a malicious site you happen to visit pointing its own hostname
  at your rig's LAN IP and driving the dashboard through your own browser. Note *exact*: `.local`
  is not allowed as a class. Blanket-allowing it would be nearly safe — RFC 6762 reserves `.local`
  for multicast, so an internet resolver can't point one at your rig — but "nearly" leans on every
  client routing `.local` to mDNS, and a network whose unicast DNS answers for `.local` would hand
  the guard back to the attacker.
- **The name is not a door.** mDNS publishes a nicer spelling of an address the guards above
  already allowed; it opens nothing, and `.local` is resolvable only from your own link.
- **Writes require a same-origin request.** Every `POST` — renaming an individual, assigning a
  visit, editing a species, deleting a clip, changing a camera control — must carry an `Origin`
  header matching the dashboard's own, and `Content-Type: application/json`. Without that, any page
  you happened to have open could quietly `fetch()` your rig and, say, blank the individual off
  months of hand-confirmed detections; the Host guard above does not stop a plain cross-origin
  POST. If you drive the API by hand, curl needs both headers:
  `-H 'Origin: http://127.0.0.1' -H 'Content-Type: application/json'`.
- **Do not port-forward this to the internet.** The guards above close the common accidental paths;
  they are not a login and not a substitute for one. Anyone already on your Wi-Fi has full access.
  If you want real remote access, put it behind a VPN or an authenticating reverse proxy — and only
  then set `lan_only = False` in `config_local.py`.
- **`config_local.py` holds the sensitive bits** — your latitude/longitude, and any RTSP camera
  credentials. It's gitignored, so it never rides along in a commit; note that `backup.py` *does*
  copy it into the `meta-<date>.zip`, which usually lands in a cloud-synced folder.
- **Retention is asymmetric, and only half of it is bounded.** Clips roll off on their own
  (`clips_max_gb` / `clips_max_gb_by_source`), but **`crops/` and the SQLite database grow without
  bound** — there is no crop pruner and no DB retention policy. Measured on one camera after about
  seven weeks: `backyard.db` ~810 MB, `crops/` ~4.7 GB. Deleting old material is a decision the rig
  deliberately leaves to you, so make it consciously rather than discovering it as a full disk.
- **Reporting a problem:** see [SECURITY.md](SECURITY.md).

---

## Responsible use

It's a camera pointed at the outdoors, which makes a few things worth saying plainly.

- **Point it at your own property.** Frame the yard you own or rent, not a neighbour's windows,
  their garden, or a public sidewalk.
- **Recording people is a different question from recording raccoons.** Depending on where you
  live, a camera that covers a neighbour's property or a public way can raise consent obligations
  and sometimes signage or registration ones. Check your local rules before a camera covers
  anywhere a person would reasonably expect privacy; nothing here is legal advice.
- **Audio: know which of your cameras has a microphone.** This project never *captures* audio
  itself — the live rig pipes bare video frames, so glass-door clips are silent. But a
  **trail-cam MP4 imported off an SD card arrives whole, microphone track included**, and since
  the clip transcode stopped stripping it that audio now *plays* in the dashboard. (This README
  previously claimed the project recorded no audio at all. That stopped being true when the
  video importer landed in July 2026, and it is corrected here rather than quietly.) Audio is
  where the rules are usually strictest — several jurisdictions require *all-party* consent for
  recording conversation, with no outdoors exception — so if your camera has a mic, decide
  deliberately: switch it off in the camera's own settings, or know that you are keeping voices.
  Deleting a clip deletes its audio with it; there is no separate audio store to purge.
- **Clips record whole frames.** Person *crops* are filtered by default (`save_classes` is animals
  only, so a person at the glass is drawn in the preview but never written to `crops/`) — but a
  clip is video of the **entire frame**. Anyone who walks through an animal-triggered clip is
  recorded, and `--save-full-frame` writes whole frames to `frames/` as well. The live MJPEG feed
  shows whatever the camera sees, filtered by nothing.
- **The showcase exporter filters by *label*.** `makingof_export.py` bakes real DB rows into the
  frozen page under `making-of/`, and it keeps person-labelled and non-critter detections out **by
  label** — which works exactly as well as the labels do. A person crop the classifier called
  something else walks straight through. Look at anything you're about to publish with your own eyes
  first.
- **Feeding.** This rig was built watching a feeding spot, and that is a choice with consequences:
  putting food out for wild animals is regulated or prohibited in some jurisdictions, and it
  habituates animals to people and concentrates them in one place, which is how disease spreads.
  Worth knowing before you set a plate out for the data.

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

Full plan and design philosophy: **[docs/plan.md](docs/plan.md)**. The short version:

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
  pipeline downstream (species ID, re-ID, behaviour all key off the `source` column): the rig's
  naming helper labels the new crops if it's running (else run `python classify.py`), and the
  visit ledger refreshes itself again when the labels land. Running a card whose contents get
  **formatted away each cycle**? Read **[the import runbook](docs/runbook-trailcam-import.md)**
  first — it is the budget-and-backup sequence that keeps a prune from eating the only copy.

Guiding principle: keep **appearance and behaviour on separate axes** and surface both —
augment the critter-knower, don't replace them. And: boring and robust over clever; most of
the value is in capture and accumulation, not the fancy model.

The schema and module layout are arranged so each phase is an addition, not a rewrite.

Two longer write-ups sit in `docs/` if you want the reasoning rather than the summary: a
[sharing-readiness review](docs/review-2026-06-19.md) of what had to change before this could go
public, and an [observatory analysis](docs/observatory-2026-06-28.md) of the first weeks of data
(where the pipeline was quietly wrong, and what the numbers actually supported).

---

## Contributing

Bug reports, camera-and-yard reports from a different climate, and patches are all welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for how the repo is laid out, what the tests cover, and the
house style (boring and robust over clever). Security issues go through
[SECURITY.md](SECURITY.md) instead of a public issue.

---

## License & attribution

Copyright © 2026 Matt Scott. Licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE).

The rig runs MegaDetector v6 through [Ultralytics](https://github.com/ultralytics/ultralytics), which is **AGPL-3.0**, so this project is AGPL-3.0 as well. Note the network-use clause: if you run a modified version as a network service (for example, exposing the dashboard to others), you must offer those users the corresponding source.

**The model weights are licensed separately, and the code's licence does not cover them.** They
download at first run and never enter this repo, so it is easy to miss that each carries its own
terms — and one of them is non-commercial:

| Weights | Licence | What that means here |
|---------|---------|----------------------|
| **MegaDetector v6** (detector) | CC-BY-4.0 | Attribution is a *condition*, not a courtesy — credit Microsoft AI for Good Lab wherever you use its output. |
| **BioCLIP 2** (species) | MIT | Permissive; the authors ask that you cite the paper. |
| **MegaDescriptor-L-384** (re-ID) | CC-BY-**NC**-4.0 | **Non-commercial.** No commercial use of the model or its embeddings, whatever the code's licence says. |

So "AGPL-3.0, commercial use permitted" describes the *code* only. A commercial deployment has to
drop the re-ID phase or swap in a differently-licensed embedder — `detection_embeddings` is keyed
by `model` precisely so a second embedder can be added without a migration. Full attribution text,
citations and links: **[NOTICE.md](NOTICE.md)**.

---

## Troubleshooting

- **`[CUDA ERROR] ... no kernel image is available` / capability sm_120:** your torch is the
  wrong CUDA build. Reinstall with the cu130 wheels (see [Setup](#setup) step 2).
- **`could not open camera src=0 yet -- will keep trying to connect` repeating:** the rig never
  gives up on a camera, so this line repeats instead of erroring out. Another app (Zoom, Camera,
  ComfyUI) may be holding the webcam, or the index is wrong — close the other app, or run
  `--list-cameras` and pick a different index.
- **No detections though motion shows:** lower `--min-confidence`, or check lighting; the red
  dot (top-right of the preview) confirms the motion gate is firing.
- **Too many / too few motion triggers:** tune `--motion-min-area` (lower = more sensitive).
- **The detector keeps boxing the same static spot** (a hard shadow, a dark opening in a wall,
  a knot in the fence — saved over and over, then force-named as some improbable species): that
  patch simply *looks* like an animal to MegaDetector, and the motion gate is no defense — it
  only decides *when* the detector runs, and anything else moving (a real visitor, dusk light
  fading) makes the stateless full-frame detector re-report the patch. Open the dashboard's
  Instrument Panel → **Ignored Spots** and drag a box over the patch (removing one is a click;
  edits apply on the next frame, no restart — `cfg.ignore_zones` in config_local.py still works
  as a one-time seed): a detection whose box mostly *is* the zone is dropped before drawing,
  saving, or clip-triggering, while a real animal passing through carries a much bigger box and
  is kept. Zones show as faint gray "ignored" outlines on the preview, and the dashboard flags
  any zone drawn before the camera last moved, so a stale one is visible.
- **App dies overnight / camera "vanishes" while idle:** a running rig **deliberately keeps the
  machine awake** — on Windows `backyard_cam.py` holds a **Power Request**
  (`PowerRequestExecutionRequired`, which is *Modern-Standby-aware*) plus the legacy
  `SetThreadExecutionState` as a belt-and-suspenders, so the box can't idle into standby and
  **USB-suspend the webcam** underneath the capture loop (a `Kernel-Power` standby event mid-run
  was the old overnight-death cause; the legacy flag alone did **not** hold a Modern-Standby box).
  It's Windows-only and a clean no-op elsewhere; a failure to set the request is now **logged
  loudly** at startup (`power: WARNING …`) rather than failing silently. If the machine still
  sleeps, check that no power policy is force-sleeping it and read that startup log line.
  **Caveat: the request only wins on AC power.** On battery, Windows naps into Modern Standby at
  the DC idle timeout no matter what — so a rig running on battery warns hard on all three
  surfaces (console, preview HUD, a dashboard banner) until you plug the charger in.
- **The preview turns into torn color bands / pink-and-red garbage while the camera stays
  "connected":** the webcam's USB stream has **wedged** — usually after standby suspend-cycled it
  (see the battery caveat above). The camera keeps delivering frames, but they're corrupt, and
  **no app-level recovery works**: reopening the capture just reattaches the same broken stream
  (one real wedge shrugged off 19 reopens and 5 white-balance resets; a physical unplug/replug
  fixed it instantly). The rig detects the state (`CAMERA WEDGE detected` in the log, plus a HUD
  line and dashboard banner) and — if you've run **`setup_selfheal.bat` once, as admin** — fixes
  it alone: that setup registers an elevated scheduled task the unprivileged rig can fire, which
  `pnputil` disable/enable-cycles the camera (the software version of a replug; device-side story
  in `logs/usb_reset.log`). Resets are budgeted (default 2/hour); past that, or without the
  setup, the banner asks for a human replug. Preview `usb_reset.ps1 -DryRun` first if you want to
  confirm which device it would cycle; knobs live under `wedge_*` in `config.py`.
- **Killed it with Task Manager / `taskkill /F` (or it crashed)?** The species-naming helper is
  its own process and never exits by itself, so a hard-killed rig used to leave it running
  invisibly — and the **next** launch then ran *two* BioCLIP workers fighting over CPU and the
  SQLite write lock. The rig now **sweeps for orphaned helpers at startup**: each helper's
  command-line tag names the rig that spawned it, and one is reaped only when that rig is
  *provably* gone. Two rigs sharing the DB are safe (a live rig's helper is never touched — when
  in doubt, nothing is killed), and both process rows of a helper (the venv launcher + the real
  interpreter) go together. You'll see `naming: reaping N stale helper process(es)…` in the
  console/log when it fires. Windows-only, best-effort, well under a second at startup.
