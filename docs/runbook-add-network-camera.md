# Runbook: adding a network camera

The running order for putting a second (or third) live camera into the rig, and the traps that
cost time the first time. `CameraSpec` in `config.py` is the authority on *what the fields mean*;
the README's [Multiple cameras](../README.md#multiple-cameras-usb--networked) section is the
authority on *what the rig does with them*. This is the operational sequence — what to do, in
what order, and how to tell which step you are stuck on.

Calibrated against the first one: a Reolink RLC-810A added 2026-08-21, alongside the glass-door
USB webcam. The numbers cited are that camera's; the sequence is not vendor-specific.

## The shape of it

**The side-by-side view is not a setting.** The dashboard builds one live pane per camera from
`/api/cameras` and grids them automatically (`.live-grid` is `auto-fit, minmax(300px, 1fr)`);
the `single` class only exists to keep a one-camera rig looking exactly as it did. So there is
nothing to switch on in the UI, and no "combined" view to configure. Everything is decided by
one thing: **the camera list the rig reads at startup**, `cfg.cameras`.

Two consequences worth knowing before you start:

- **The rig reads that list once, at launch.** Every change here needs a restart. Ignore zones
  are the exception — those went DB-backed and dashboard-editable in 2026-08-07 and apply live.
- **Nothing composites cameras.** Each camera gets its own capture thread, its own clip
  recorder writing to `clips/<source>/<date>/`, and stamps its own `source` on every row.
  `detections.source` and `visits.source` mean a visit belongs to exactly one camera; the
  two-up grid is a live *layout*, and never becomes a recorded double-view.

## Phase 1: get it on the network, and find it

- **Find the address in the camera's own app**, not by scanning. Reolink: Settings → Network →
  Advanced shows Connection Type and the current IP.
- **The ARP cache is not a device inventory.** `arp -a` lists hosts this box has recently
  *talked to*, so a camera nobody has contacted is simply absent from it. On 2026-08-21 the new
  camera answered ping at 5 ms and did not appear in ARP at all.
- **Note whether it is on DHCP.** If it is, the address in `config_local.py` is a lease, and a
  lease change kills the pane silently — the rig reconnects forever against an address nobody
  answers on, and the dashboard just shows "Camera feed unavailable". Pin it with a **DHCP
  reservation on the router** (one place to look, no chance of colliding with the pool) rather
  than a static IP set on the camera. Do this once the camera is in its final position.

## Phase 2: turn RTSP on — it ships OFF

This is the step that looks like a network fault and is not.

- Reolink: Settings → Network → Advanced → **Server Settings**, tick **RTSP** (554), Save.
- **Then reboot the camera.** On firmware v3.1.0.4972 the tick alone did nothing: port 554
  stayed closed for minutes afterwards, and only started listening after Settings → System →
  Reboot. A camera that answers ping with 554 closed has almost always not been rebooted.
- **ONVIF is not required** — the rig opens the RTSP URL directly through FFMPEG and never does
  ONVIF discovery. Leave it off. HTTP/HTTPS only enable the camera's own web UI; useful for
  tweaking IR and exposure from a browser later, but not needed to capture, and fewer open
  services is the better default.

## Phase 3: make a dedicated camera user

- **The app login is usually not a camera account.** On a cloud-managed camera you sign into
  the *vendor's* service; the camera's own user table is separate, and RTSP only checks the
  latter. On 2026-08-21 `admin` plus the password used in the app was refused with a clean 401,
  by both FFmpeg and a hand-written Digest DESCRIBE.
- **Make a user for the rig** (Reolink: User Management → Add User) rather than reusing admin.
  That password ends up in a file on disk; it should not also be the camera's admin credential.
- **Prefer a plain alphanumeric password.** Not because punctuation should break anything — it
  does not, and `!` is legal in a URL's userinfo — but because it removes a variable from a
  problem you may already be two rounds deep into isolating.

## Phase 4: probe before you configure

```
set CAM_PASS=...
python tools/camprobe.py 192.168.0.105 --user rig
```

`tools/camprobe.py` walks the failure modes in the order they happen — port, then RTSP
challenge, then credentials, then which stream paths exist — and ends with a ready-to-paste
`CameraSpec`. With no `--path` it tries known vendor paths, so an unfamiliar camera does not
need its manual found first. Passwords are masked in everything it prints.

Reading its output:

- **`port 554: CLOSED`** → Phase 2. Almost always the reboot.
- **`401 Unauthorized` in the server line** → *correct*. That is the camera proving RTSP is
  answering and telling you which auth scheme it wants.
- **Every path fails** → read FFmpeg's own line. `401` is credentials (Phase 3); `404` is the
  path; opened-but-no-frames means that channel is disabled on the camera.

**Then open the test frame.** It is the most valuable thing the tool produces and it is not a
number. On 2026-08-21 the frame showed a closet, a vacuum cleaner and a carpet — the camera was
still on the floor indoors, which is what decided sub-stream over main and saved tuning an aim
that was about to change.

### Sub-stream or main

Measured on the RLC-810A, 15 s of sustained decode after a warmup, on a 12-core box:

| Stream | Resolution | Sustained | Decode cost |
|---|---|---|---|
| `h264Preview_01_sub` | 640×360 | 10.0 fps | 1.2% of one core |
| `h264Preview_01_main` | 3840×2160 | 25.0 fps | 68.4% of one core |

Main is affordable — but decode is not the whole cost, and two other things matter more:

- **Crops are cut from whatever frame the rig sees.** A raccoon across the yard in a 640×360
  frame is ~50 px: enough for MegaDetector to draw a box around, nowhere near enough for
  species ID or re-ID. For a wide yard view, plan on the main stream.
- **`clip_scale` is global**, not per-camera. At 0.667 a 4K camera records 2560×1440 clips.

Start on the sub-stream to prove the plumbing; switch when the camera is in its final position.

## Phase 5: write the spec

```python
from config import CameraSpec

cfg.cameras = [
    CameraSpec("glass_door_cam", 1, name="Glass door"),
    CameraSpec("yard_ir", "rtsp://rig:PASSWORD@192.168.0.105:554/h264Preview_01_sub",
               name="Yard (night IR)", frame_width=640, frame_height=360,
               motion_min_area=200),
]
```

- **Setting `cfg.cameras` replaces `camera_index` as the thing the rig reads**, so the existing
  camera must now be spelled out explicitly. It does not lose its tuning: every field a spec
  does not name inherits the same-named `Config` value, so the glass door keeps its 1920×1080,
  its `CONTRAST: 48`, its `motion_min_area` 1800 — and `backend=None` on an int `src` still
  resolves to DirectShow on Windows.
- **Each camera needs a unique `source`.** It is the key everything downstream splits on.
- **`motion_min_area` is full-frame pixels** and therefore resolution-dependent. The gate runs
  on a ~640 px-wide downscaled copy but converts blob areas *back* to full-frame pixels before
  comparing, so the knob always means the same thing — which is exactly why a new camera at a
  different resolution needs its own value. `camprobe` computes it by holding the reference
  camera's trigger fraction (1800 px at 1920×1080 = 0.087% of frame).
- **Set `frame_width`/`frame_height` explicitly** on a network camera. They are a no-op for an
  RTSP stream (the stream dictates resolution), but leaving them to inherit makes the config
  claim 1080p for a 640×360 camera.
- **The primary camera** — the Live tab's default, and where the masthead's period and sun
  times come from — is whichever spec matches `cfg.source`, else the first in the list.

### Give it its own clip budget — before it records anything

**A new camera with no entry in `clips_max_gb_by_source` does not get its own budget. It joins
a shared one, and starts evicting the other cameras' clips.**

Sources without an override all fall into a single `None` bucket on `cfg.clips_max_gb` (10 GB),
pruned oldest-first *across every source in it*. On 2026-08-21 `glass_door_cam` alone was
already at **9.89 GB of that 10** — so the first gigabyte the new camera recorded would have
deleted the oldest glass-door clips to make room. Silently: the prune logs a count, not a
victim, and that bucket gets **no archive guard** (the "never delete the only copy" protection
is scoped to `clips_irreplaceable_sources`, i.e. the trail cam). `backup.py` runs weekly and
skips today, so a fast enough prune destroys clips before they are ever archived — the same
cadence collision that lost the 2026-07-30 trail-cam clips.

```python
cfg.clips_max_gb_by_source["yard_ir"] = 10.0
```

One line, and the existing camera goes back to owning the shared bucket alone. Check what the
current cameras actually use before picking the number; `clips/<source>/` is the whole story.

### Where the password lives

`config_local.py`, in the URL. Three paths it could escape by, all closed:

- **Git** — `config_local.py` is gitignored.
- **The dashboard** — `/api/cameras` serves only `source`, `name`, `network` and `primary`. The
  `src` URL never leaves the process, so LAN viewers cannot read it.
- **The log** — the startup banner, the could-not-open retry and every open/reconnect notice
  print the src, and `logs/` is swept into the backup by `backup.py`. All three go through
  `backyard_cam._safe_src`, which masks the password and keeps the username (a rejected login
  is the failure you most need that line to explain). If you add a new print of a `src`, use
  it — a test fails the build if a raw `src!r` reappears.

## Phase 6: restart and verify

Press **`q`** in the live video window (or close it), then run `start_critter_cam.bat`. That
order matters: a clean stop drops a `.rig_pause` marker so `rigwatch.py` does not race you back
up, and the launcher clears it on the way in.

Check, in this order:

1. **The banner**: `2 cameras: Glass door (1), Yard (night IR) ('rtsp://rig:***@...')` — the
   count, and the mask.
2. **Each camera opens**: one `[<source>] open (...)` line per camera.
3. **The dashboard**: two panes in Live Observation. Click the new one — the Instrument Panel
   and "Who's visiting now?" should re-scope to it, and a networked camera shows a note instead
   of exposure sliders, because those live on the camera.
4. **Rows arrive** under the new `source` once something moves.

## Phase 7: once it is in position

Everything below is deliberately deferred until the camera stops moving — a filter tuned to a
temporary aim fails silently after the move.

- **DHCP reservation** (Phase 1).
- **Turn the OSD off** — watermark, timestamp and device name are burned into the frame. Two
  reasons: they land in every crop, and the clock changes every second, which at 640×360 is a
  few hundred pixels of permanent motion against a 200 px trigger. If a new camera fires
  constantly on an empty scene, the burned-in clock is the first suspect.
- **Ignore zones**, drawn in the dashboard's Instrument Panel — and remember zones match by
  IoU ≥ 0.45, so a zone drawn *around* an area is inert. Fit them to the repeated box.
- **Revisit sub vs main** now that the framing is final.
- **Re-check the clip budget** against what the camera actually records in a real week — the
  number set in Phase 5 was a guess made before it had seen anything move.
