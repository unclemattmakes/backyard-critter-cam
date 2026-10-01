# docs/ — the project's paper trail

Two kinds of document live here, and the difference matters when you read them. For how the
system works and how to run it, read the [README](../README.md) at the repo root.

## Current — keep these true

- [deferred-work.md](deferred-work.md) — the ranked backlog, rewritten 2026-10-01: what's worth
  doing next, why, and roughly how big. Links to the measured-dead ideas in the archived August
  version.
- [host-instability-2026-08.md](host-instability-2026-08.md) — **an open investigation, updated
  in place with dated entries**: the rig's host hard-hangs under load, what has been ruled out
  (the PSU), what is being tried (the GPU power cap, the AVX2 cap on the naming helper), and what
  to try next. Read its status line first.
- [runbook-add-network-camera.md](runbook-add-network-camera.md) — the operational sequence for
  adding a network camera (RTSP or HTTP/MJPEG), including the two steps that look like faults and
  aren't, and what this rig actually runs.
- [runbook-trailcam-import.md](runbook-trailcam-import.md) — the operational sequence for
  importing an SD card whose contents get formatted away each cycle: orient, check the clip
  budget, archive first, verify before you format. The checks exist because the card is the only
  copy.
- [species-lists.md](species-lists.md) — ready-made zero-shot species lists for other regions
  (US NE/SE/SW, UK & Ireland, Central Europe, Australia east coast), plus the phrasing rules that
  make labels resolve well.

## Dated snapshots — do not read as current documentation

Each was written on the date in its name or banner and is kept because it explains *why* the
code looks the way it does. Findings in them have largely been addressed since, and their numbers
are frozen at that date.

- [plan.md](plan.md) — the original project plan (2026-06-07), written before a line of it existed:
  the two-axis "looks like X but isn't acting like X" idea, the four phases, and the per-species
  reality check on individual ID.
- [review-2026-06-19.md](review-2026-06-19.md) — a whole-rig code review at the point the project
  turned from a private tool into something a friend could install, plus the fix roadmap it drove.
- [observatory-2026-06-28.md](observatory-2026-06-28.md) — a data/UX/ML analysis of the first 22
  days of live capture: what the yard actually logged, what the pipeline got wrong, and which
  signals were being computed but never surfaced.
- [identity-eval-2026-08-05.md](identity-eval-2026-08-05.md) — the full re-ID evaluation that
  found appearance identity decays in about a week (top-1 0.818 → 0.482 → 0.222 at a 0/7/21-day
  probe-to-template embargo), that every earlier number was session-leaked, and that the fix is
  fresher labels rather than a better backbone. Its "Rejected, with reasons" section is the
  reason several ideas stay dead.
- [refimg-design-2026-08-07.md](refimg-design-2026-08-07.md) — the empty-scene reference veto,
  designed and then raced against real footage before shipping: why the bare-pixel metric erases
  raccoons, why rolling background models are forbidden here, and why it went out in shadow mode.
- [background-identity-2026-08-09.md](background-identity-2026-08-09.md) — does the background
  identify the animal? Yes: with the animal blanked out the matcher is still right 60% of the
  time, and at a 7-day embargo the animal adds nothing over the scene.
- [refimg-review-2026-08-09.md](refimg-review-2026-08-09.md) — the first review of the veto's
  shadow-mode flags (18, all furniture), why coverage keeps it from firing by day, and the dawn
  "camera moved" false alarm it uncovered.
- [archive/deferred-work-2026-08.md](archive/deferred-work-2026-08.md) — the August backlog in
  full, kept verbatim: every plan as adversarial review left it, and **Killed, with reasons**,
  the ideas that were measured dead.

Developer utilities live in [../tools/](../tools/). `check_endpoints.py` cross-checks the
dashboard's `/api/` calls against the routes `web.py` actually defines. `eval_rigidity.py`
re-sweeps `staticfilter.DEFAULT_RIGID_MAXPAIR` — the threshold below which a repeat-firing spot
is judged furniture on its pixels rather than its persistence — against this yard's own
furniture and animal populations. **Run it before changing that constant**: it deletes rows,
and the number it defends is the distance to the lowest-scoring real animal on record.
`camprobe.py` probes an RTSP camera before you add it (see the network-camera runbook).
