# Deferred work: the ranked backlog

**Living document, rewritten 2026-10-01** after the 2026-09-30 project evaluation and the ten PRs
(#28–#37) merged that night, and updated 2026-11-05 after a three-part audit. Ranked by value per unit of effort, most useful first. Each item is
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
   logged on, so a reboot to the lock screen stops everything: 24.5 h once, then 64 h from
   2026-10-07 and 17.5 h on 2026-10-28, with no email either time. Re-register the tasks
   to run whether or not anyone is logged on (the rig itself needs a desktop session for its
   preview window, so headless config comes with it), or set auto-logon. The morning email's
   *Rig health* box now flags a watchdog that has stopped running, but the email is one of those
   tasks too, so it goes quiet at the same moment; only the off-host `heartbeat_url` covers that.
   *S, human (needs the account password).*
4. **Rotate the credentials that rode into old backups.** Until #36, the camera password and the
   mail API key sat in `config_local.py`, so every meta zip in the cloud backup folder carries
   them. Rotate both, then thin or delete the old meta zips. *S, human.*
5. **Host stability: the AVX2 cap was not the fix.** It gave the two longest runs on record (14
   and 17.6 days), then six unclean boots from 2026-10-07 to 10-28, three with no power button
   press, and the one with telemetry was an instant reset at idle. Nothing was logged. Update the BIOS (still 1401, from 2018), get HWiNFO logging working,
   then reseat or move the GPU
   ([host-instability-2026-08.md](host-instability-2026-08.md)). Once the host is trusted, revert
   `detector_min_interval_s` 3.0 → 1.0 (it cost about two thirds of the crops per visit). *M,
   human.*
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
   ≥ 0.8, 0.225 at 0.5–0.8, 0.026 below 0.5. #35 hides < 0.5 on display. Still open: surface the
   day and night bands the artifact already holds (`by_daynight`; `eval.py` prints only accuracy),
   and decide whether the 0.5–0.8
   band should also read as uncertain anywhere it drives a count. *S.*
10. **Put a background arm in `eval.py`.** The honest baseline for re-ID is scene matching at the
    same embargo, not 0.34 chance, and that margin is uncomputed and unwatched. *M.*
11. **The DB and crops grow without bound.** `backyard.db` is ~6.2 GB (2026-11-05), up from
    ~3.8 GB five weeks earlier, `crops/` has no pruner, and every weekly snapshot is ~5 GB compressed
    (DB and meta zips together). Decide a retention policy (for example,
    drop crops of old, unlabelled, unembedded noise detections) and turn on
    `backup_snapshot_retention` (the recommended policy would free ~15 GiB today). *M, plus an
    owner decision.*
12. **The human labelling sessions.** The blind Notch audit (a suspected early Notch/Pedro swap;
    46–53 templates ride on it; must come before any mass profile) and the D1 un-blend pass over
    the 22 named multi-animal visits (the cheapest template growth). Both are protocols in the
    archive, §1.1 and §1.2. *M, human.*
13. **Re-sweep the auto-assign operating point** on the current corpus before trusting the bars
    in `config_local.py`; group labels and new animals have moved it since August. *S.*
14. **Refactor the biggest files.** `dashboard.js` (~3,700 lines), `db.py` (~2,900),
    `backyard_cam.py` (~2,400), `web.py` (~2,300, after #47 moved the review queue out) and
    `stats.py` (~2,200) each hold several unrelated surfaces. The real blocker is `make_server`, one
    1,091-line closure: hoist one route family per PR, as #47 did. *L, low urgency.*
15. **Exercise the never-fired safety path on purpose.** The auto-assign reject tombstone has
    never run outside tests; trigger it once deliberately. (The USB-wedge self-heal cannot fire
    here: no USB camera, and `setup_selfheal.bat` was never run on this host.) *S.*
16. **Trail-cam view epochs.** `view_epochs` is empty for the trail cam (and holds one wrong row
    for the glass door), which blocks both an occupancy map and the trail-cam half of the
    furniture veto. The segmenter method is measured and works; build the read-only reporter
    first. *M.*
17. **Linux/macOS ops parity.** Everything that keeps the rig alive unattended is
    Windows-shaped (`.bat`, `schtasks`, Power Requests). Document or provide cron/systemd
    equivalents. *M.*
18. **Weather join.** Backfill Open-Meteo history by date so a wet night stops looking like a
    quiet one. *S.*

## Found by the 2026-11-05 audit, still open

Ranked the same way. Each was traced through the code; none has been seen to fire yet except
where it says so.

1. **Outages leave no record.** A hang writes no `down` coverage row, so the ledger reads the gap
   as covered, and the mail task has `StartWhenAvailable` off, so a missed morning is never made
   up. 2026-10-28 cost 17.5 h and the next email said nothing. Have rigwatch log an incident when
   `checked_at` jumps or the boot time changes, show it in *Rig health* for 48 h, and alarm when
   the last email is over 30 h old. *S.*
2. **Nothing reads a backup back.** `quick_check` runs on the local copy before zipping, nothing
   opens the published zip, and the last real restore was 2026-08-23. Add `backup.py --verify` as a
   monthly drill: `quick_check` the newest DB zip and CRC-test one random media zip. *S.*
3. **"Cloud upload: finished" only means Drive went idle.** A paused client, a sign-out or a full
   quota all read as drained, and the verdict goes only to `STATUS.txt`, inside the folder that is
   not syncing. Record it in the backup state and alarm on it. The real signal is the DriveFS
   `operations` table. Nothing checks destination space. *M.*
4. **A dead detector looks like a quiet yard.** `_run_camera` catches detector exceptions per
   frame, so a sticky CUDA failure leaves capture healthy and the email calling it a quiet night.
   Count consecutive failures, publish them in the capture status, exit after N. *S.*
5. **Cheap GETs start unbounded CPU work on a fragile host.** `/api/reel` builds run in parallel
   with no `-threads`, legacy `/media/*.mp4` transcodes in the request, `/api/visits?day=%25` loads
   the whole table, and every `limit`/`offset` of `/api/reid/queue` is a parallel matcher rebuild.
   None consult `heavyio`. One `BoundedSemaphore` around ffmpeg, `-threads 2`, validate `day`,
   clamp the cache keys. *S–M.*
6. **Nightly step failures never leave the batch log.** `batch_health` only asks whether a recent
   eval exists, and the eval runs even when `visits`, `clipmotion` or `embed` failed. Have it count
   `***` lines since the last `BATCH START`. *S.*
7. **Unreadable day archives are a warning, not a failure,** and block that day from later
   top-ups. `lost_clip_days` also undercounts: a partly pruned day is never reported, though
   `clips.pruned_at` could say so exactly. *S.*
8. **A removed camera clears its alarm as if it recovered.** Right on 2026-10-27, when the removal
   was deliberate, but the code cannot tell, and a lost `secrets.json` key looks the same. Remember
   last-seen sources and raise `camera_gone` for 24 h. *S.*
9. **Viewer testimony can change re-ID state** once `operator_token` is set: a two-name viewer log
   drops a visit from the templates, supersedes an operator's sighting, and has no length cap. Only
   matters after backlog item 6. *M.*
10. **Test gaps on incident paths.** No tests for `_supervise_naming`, `_naming_env` (the AVX2 cap),
    `run_clipmotion.bat`, the stale-ledger trigger in `classify.watch_loop`, the rigwatch backlog
    alarm, or `reconnect_capture` past its fast retries. *S each.*

## Killed, with reasons

Ideas measured dead, kept so nobody re-runs them: lowering `staticfilter`'s span, kit headcount
from track overlap, kit age class from box sizes, chain-of-eras identity, an un-blend operating
point, lowering `COVER_MIN_FRACTION`, the bounding-box amplification theory, gait as identity,
facial landmarks, silhouette tail tracing, SAM background removal, MOG2 as an empty-scene
reference, decoy labels, and tiled inference. The numbers for each are in
[the archive's *Killed, with reasons*](archive/deferred-work-2026-08.md#7-killed-with-reasons).
