# Deferred work: the ranked backlog

**Living document, rewritten 2026-10-01** after the 2026-09-30 project evaluation and the ten PRs
(#28–#37) merged that night. Ranked by value per unit of effort, most useful first. Each item is
what, why, and a rough size (S = an evening, M = a few sessions, L = a project; *human* = needs the
owner's eyes or hands, not code).

When an item ships, delete it and let the code and the commit say what happened. When an item is
**measured dead**, don't delete it: add it to the *Killed, with reasons* list (below) so nobody
re-runs the same negative.

The August version of this file, with the full plans, numbers and adversarial reviews, is
archived verbatim at [archive/deferred-work-2026-08.md](archive/deferred-work-2026-08.md). Code
comments and older docs that cite "deferred-work §1.3", "§2.1" or "Killed" mean the section
numbers in that file.

## The backlog

1. **A durable identity cue that doesn't depend on the scene.** Appearance re-ID holds for about
   a week: blocked leave-one-out top-1 ~0.70, ~0.46 at a 7-day embargo, ~0.16 at 21 days, against
   0.34 for always guessing the commonest raccoon, and what survives a week is mostly the
   background ([background-identity-2026-08-09.md](background-identity-2026-08-09.md)). A better
   backbone can't fix that. Candidates: put more pixels on the ear notch (a closer angle or a
   higher-resolution crop path), or a load cell under the dish (~$15; partial loading and
   seasonal weight change are the design risks). *L, hardware.*
2. **Review the species-mismatch audit.** `individuals.py --audit-species-mismatch` lists 403
   labelled crops whose species differs from their individual's (43 auto, the 2026-06-26 Notch
   noise rows; 360 human, 223 under group labels; 149 `not an animal`). Many are misclassified
   crops of the right animal, so it's a review list, not a delete list. The 43-row auto repair is
   spelled out in PR #30 and has not been run. *S, human.*
3. **Scheduled tasks need a logged-in user.** Every critter-cam task runs only while the owner is
   logged on, so a reboot to the lock screen stops everything (24.5 h once). Re-register the tasks
   to run whether or not anyone is logged on (the rig itself needs a desktop session for its
   preview window, so headless config comes with it), or set auto-logon. *S, human (needs the
   account password).*
4. **Rotate the credentials that rode into old backups.** Until #36, the camera password and the
   mail API key sat in `config_local.py`, so every meta zip in the cloud backup folder carries
   them. Rotate both, then thin or delete the old meta zips. *S, human.*
5. **Host stability: confirm the AVX2 cap, then the real fix.** No hard hang since the naming
   helper was capped to AVX2 / 4 threads on 2026-09-23. If that holds, update the BIOS; if not,
   reseat or move the GPU and get HWiNFO logging working first.
   ([host-instability-2026-08.md](host-instability-2026-08.md)). Once the host is trusted, revert
   `detector_min_interval_s` 3.0 → 1.0 (it cost about two thirds of the crops per visit). *M.*
6. **Set `operator_token`.** It is unset, so every device on the LAN can relabel, rename and
   delete. One line in the secrets file turns other devices into viewers. *S, human.*
7. **The motion gate wakes the detector for nothing.** 82–92% of detector runs find no box
   (2026-09-30 eval). Tighten per camera (`motion_min_area`, ignore zones, a minimum blob
   persistence) and measure recall against the same week. Fewer GPU wakes also helps item 5.
   *M.*
8. **The reference-image veto has been inert since 2026-09-05.** It shipped in shadow mode and
   stopped flagging anything. Read the hourly `VetoCensus` lines to see why (coverage, no
   references, or switched off), then fix it or switch it off and say so. *S–M.*
9. **Use species accuracy by confidence band.** Graded against human labels: 0.906 right at
   ≥ 0.8, 0.225 at 0.5–0.8, 0.026 below 0.5. #35 hides < 0.5 on display. Still open: split the
   bands by day and night (night labels have been much weaker), and decide whether the 0.5–0.8
   band should also read as uncertain anywhere it drives a count. *S.*
10. **Put a background arm in `eval.py`.** The honest baseline for re-ID is scene matching at the
    same embargo, not 0.34 chance, and that margin is uncomputed and unwatched. *M.*
11. **The DB and crops grow without bound.** `backyard.db` is ~3.8 GB, `crops/` has no pruner,
    and every weekly snapshot is ~2.5 GB compressed. Decide a retention policy (for example,
    drop crops of old, unlabelled, unembedded noise detections) and turn on
    `backup_snapshot_retention` (the recommended policy would free ~15 GiB today). *M, plus an
    owner decision.*
12. **The human labelling sessions.** The blind Notch audit (a suspected early Notch/Pedro swap;
    46–53 templates ride on it; must come before any mass profile) and the D1 un-blend pass over
    the 22 named multi-animal visits (the cheapest template growth). Both are protocols in the
    archive, §1.1 and §1.2. *M, human.*
13. **Re-sweep the auto-assign operating point** on the current corpus before trusting the bars
    in `config_local.py`; group labels and new animals have moved it since August. *S.*
14. **Refactor the three biggest files.** `dashboard.js` (~3,600 lines), `web.py` (~2,800) and
    `stats.py` (~2,200) each hold several unrelated surfaces. Split by tab / route family /
    surface, behind the existing tests, one file per PR. *L, low urgency.*
15. **Exercise the never-fired safety paths on purpose.** The auto-assign reject tombstone and
    the USB-wedge self-heal have never run outside tests; trigger each once deliberately. *S.*
16. **Small doc/code drift found on 2026-10-01.** The `README.txt` that `backup.py` drops in the
    destination still calls `STATUS.txt` "the weekly heartbeat"; `config.py`'s `heartbeat_url`
    comment says to set it in `config_local.py` though it is a secrets-file field; the morning
    email's link uses the configured `web_port`, so it is wrong when the rig fell back to 8000.
    *S.*
17. **Trail-cam view epochs.** `view_epochs` is empty for the trail cam (and holds one wrong row
    for the glass door), which blocks both an occupancy map and the trail-cam half of the
    furniture veto. The segmenter method is measured and works; build the read-only reporter
    first. *M.*
18. **Linux/macOS ops parity.** Everything that keeps the rig alive unattended is
    Windows-shaped (`.bat`, `schtasks`, Power Requests). Document or provide cron/systemd
    equivalents. *M.*
19. **Weather join.** Backfill Open-Meteo history by date so a wet night stops looking like a
    quiet one. *S.*

## Killed, with reasons

Ideas measured dead, kept so nobody re-runs them: lowering `staticfilter`'s span, kit headcount
from track overlap, kit age class from box sizes, chain-of-eras identity, an un-blend operating
point, lowering `COVER_MIN_FRACTION`, the bounding-box amplification theory, gait as identity,
facial landmarks, silhouette tail tracing, SAM background removal, MOG2 as an empty-scene
reference, decoy labels, and tiled inference. The numbers for each are in
[the archive's *Killed, with reasons*](archive/deferred-work-2026-08.md#7-killed-with-reasons).
