@echo off
title Backyard Critter Cam - nightly re-ID + motion batch
cd /d "%~dp0"
REM ---------------------------------------------------------------------------
REM The nightly "keep individual-tracking automatic" batch. Task Scheduler runs
REM this daily (task BackyardCritterCam-MotionTracks, ~2pm -- the activity trough,
REM so it doesn't fight the overnight capture peak for the GPU). Order matters:
REM   1) clipmotion.py          -- motion tracks for every NEW behaviour clip
REM   2) embed.py               -- appearance vectors for new still crops (all
REM                                species, down to the 0.5 suggestion gate;
REM                                batch 16 stays VRAM-safe beside the live rig)
REM   2b) embed.py --co-present -- LOW-conf vectors, only in plausibly multi-
REM                                animal visits: the second animal is nearly
REM                                always the low-conf box, and the still-
REM                                tracklet splitter is blind without it
REM   3) clipembed.py           -- appearance vectors for new sustained tracklets
REM   4) clipmotion.py --link   -- attach solo-clip tracks to their HUMAN-named
REM                                individual (auto names don't ground behaviour)
REM   5) individuals.py --auto-assign -- name the unambiguous new solo visits
REM                                (bars from eval.py's sweep; auto names never
REM                                feed the suggestion templates; review ✓/✗ in
REM                                the dashboard queue)
REM Every step is RESUMABLE (only new/untouched rows are processed), so a missed
REM night just catches up the next day.
REM ---------------------------------------------------------------------------
echo [%date% %time%] motion tracks for new clips...
".venv\Scripts\python.exe" clipmotion.py --device auto
echo [%date% %time%] appearance embeddings for new crops...
".venv\Scripts\python.exe" embed.py --species all --min-confidence 0.5 --batch-size 16
echo [%date% %time%] low-conf embeddings for multi-animal visits...
".venv\Scripts\python.exe" embed.py --species all --co-present --min-confidence 0.25 --batch-size 16
echo [%date% %time%] appearance embeddings for new clip tracklets...
".venv\Scripts\python.exe" clipembed.py --device auto
echo [%date% %time%] linking solo tracks to named individuals...
".venv\Scripts\python.exe" clipmotion.py --link
echo [%date% %time%] auto-naming unambiguous visits...
".venv\Scripts\python.exe" individuals.py --auto-assign
echo [%date% %time%] done.
