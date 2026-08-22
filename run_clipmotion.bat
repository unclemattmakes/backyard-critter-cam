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
REM   5) eval.py --baseline latest -- the REGRESSION GATE, run nightly so a
REM                                metric slide is noticed the day it happens
REM                                (the 0.81 -> 0.635 AUC drift accumulated for
REM                                weeks because this step was a docstring, not
REM                                a scheduler line). Read-only over the DB, no
REM                                GPU: it scores the embeddings already stored.
REM   6) individuals.py --auto-assign -- name the unambiguous new solo visits
REM                                (bars from eval.py's sweep; auto names never
REM                                feed the suggestion templates; review ✓/✗ in
REM                                the dashboard queue). SKIPPED when the gate
REM                                just reported a regression: a matcher that
REM                                measurably got worse today should not spend
REM                                tonight writing names.
REM   7) warm the dashboard's re-ID queue cache -- the first Individuals-tab
REM                                visit after a big night otherwise eats the
REM                                full rebuild at the browser (measured 24s on
REM                                2026-08-08). Rig down = curl fails = fine.
REM Every step is RESUMABLE (only new/untouched rows are processed), so a missed
REM night just catches up the next day.
REM ---------------------------------------------------------------------------
REM --- Re-arm TOMORROW's start time from the sun, before doing any work -------------------
REM Deliberately first: if the box bugchecks mid-batch (it has, repeatedly -- 07-20, 07-22,
REM 08-13 all landed inside this window), tomorrow's trigger is already correct. See sunsched.py
REM for why this tracks sunset instead of a fixed clock time.
".venv\Scripts\python.exe" sunsched.py --arm --date tomorrow

REM --- Wait for a clear box ---------------------------------------------------------------
REM Steps 1-4 are all GPU, running beside the live rig on ONE 8 GB card. What we will NOT do is
REM run them on top of a multi-gigabyte Google Drive upload: that combination preceded both of
REM the destructive 0x1E crashes (08-19, which corrupted the DB, and 08-21, which cost 1,032
REM files to chkdsk). --ttl because the lock outlives the python that takes it; the release at
REM the bottom is the normal path, the TTL only covers a run the machine kills.
".venv\Scripts\python.exe" heavyio.py --acquire batch --wait 3600 --ttl 360 --note "nightly re-ID + motion batch"

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
echo [%date% %time%] nightly eval + regression gate...
set EVAL_REGRESSED=0
if exist "reports\eval_*.json" (
    ".venv\Scripts\python.exe" eval.py --baseline latest --tolerance 0.02
    if errorlevel 1 set EVAL_REGRESSED=1
) else (
    REM First run on this machine: nothing to diff against yet; this run WRITES the baseline.
    ".venv\Scripts\python.exe" eval.py
)
if "%EVAL_REGRESSED%"=="1" (
    echo [%date% %time%] *** EVAL REGRESSION -- auto-assign SKIPPED tonight. See the diff above
    echo [%date% %time%] *** and the newest reports\eval_*.json; the dashboard shows the verdict.
) else (
    echo [%date% %time%] auto-naming unambiguous visits...
    ".venv\Scripts\python.exe" individuals.py --auto-assign
)
echo [%date% %time%] warming the dashboard re-ID queue cache...
curl -s -o NUL --max-time 180 "http://127.0.0.1:8000/api/reid/queue?mode=recent&offset=0&limit=30"
".venv\Scripts\python.exe" heavyio.py --release batch
echo [%date% %time%] done.
