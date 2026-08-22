@echo off
title Backyard Critter Cam - nightly re-ID + motion batch
cd /d "%~dp0"
REM ---------------------------------------------------------------------------
REM The nightly "keep individual-tracking automatic" batch. Task Scheduler runs
REM this daily (task BackyardCritterCam-MotionTracks). The start time is NOT a
REM fixed clock time: sunsched.py re-aims it at the late-afternoon glare window
REM (sunset-2.5h), when the sun is on the camera's bearing and the rig is not
REM merely idle but BLIND -- so the GPU steps cost nothing. See sunsched.py.
REM Order matters:
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
REM                                feed the suggestion templates; review in the
REM                                dashboard queue). SKIPPED when the gate just
REM                                reported a regression: a matcher that
REM                                measurably got worse today should not spend
REM                                tonight writing names.
REM   7) warm the dashboard's re-ID queue cache -- the first Individuals-tab
REM                                visit after a big night otherwise eats the
REM                                full rebuild at the browser (measured 24s on
REM                                2026-08-08). Rig down = curl fails = fine.
REM Every step is RESUMABLE (only new/untouched rows are processed), so a missed
REM night just catches up the next day.
REM
REM WHY THIS LOGS TO A FILE
REM Task Scheduler gives a scheduled run no console, so until 2026-08-21 every
REM word this batch printed went nowhere. On 08-21 the run at 17:39 ended with
REM exit 0xC000013A having done NO work, and the only way to establish that was
REM to query the database for what it had touched -- the batch itself left not
REM one byte of evidence. (Cause: the box was in Modern Standby from 17:12 to
REM 18:01 and Task Scheduler killed the task. The header below records the power
REM state for exactly that reason.)
REM
REM The footer line is load-bearing: every completed run ends with BATCH
REM COMPLETE, so a log that just stops IS the signal that something killed the
REM run rather than the run failing honestly.
REM ---------------------------------------------------------------------------

if not exist "logs" mkdir "logs"
set "LOG=logs\clipmotion_batch.log"
REM One rotation at ~4 MB. A run is a few KB, so this holds many months, and a
REM single .1 is enough: the DB is the record of what happened, this is only the
REM record of how it went.
for %%A in ("%LOG%") do if %%~zA GTR 4000000 move /y "%LOG%" "%LOG%.1" >NUL 2>&1

echo Running the nightly batch; full output goes to %LOG%
call :run >> "%LOG%" 2>&1
set "BATCH_RC=%ERRORLEVEL%"
echo Batch finished ^(exit %BATCH_RC%^). Log: %LOG%
exit /b %BATCH_RC%


REM --- everything below runs with stdout+stderr redirected into the log --------------------
:run
echo.
echo ===============================================================================
echo [%date% %time%] BATCH START
for /f "delims=" %%G in ('git rev-parse --short HEAD 2^>NUL') do echo   commit   %%G
for /f "delims=" %%G in ('powershell -NoProfile -Command "$b=Get-CimInstance Win32_Battery -EA SilentlyContinue; if($b){'battery status ' + $b.BatteryStatus + ' (2=AC, 1=discharging), ' + $b.EstimatedChargeRemaining + '%% charge'}else{'no battery detected'}" 2^>NUL') do echo   power    %%G
for /f "delims=" %%G in ('powershell -NoProfile -Command "$p=@(Get-Process python,pythonw -EA SilentlyContinue); 'python processes: ' + $p.Count" 2^>NUL') do echo   rig      %%G
echo ===============================================================================

REM --- Re-arm TOMORROW's start time from the sun, before doing any work -------------------
REM Deliberately first: if the box bugchecks mid-batch (it has, repeatedly -- 07-20, 07-22,
REM 08-13 all landed inside this window), tomorrow's trigger is already correct. See sunsched.py
REM for why this tracks sunset instead of a fixed clock time.
".venv\Scripts\python.exe" sunsched.py --arm --date tomorrow
if errorlevel 1 (echo [%date% %time%]   *** sunsched --arm FAILED -- continuing with the remaining steps) else (echo [%date% %time%]   sunsched --arm ok)

REM --- Wait for a clear box ---------------------------------------------------------------
REM Steps 1-4 are all GPU, running beside the live rig on ONE 8 GB card. What we will NOT do is
REM run them on top of a multi-gigabyte Google Drive upload: that combination preceded both of
REM the destructive 0x1E crashes (08-19, which corrupted the DB, and 08-21, which cost 1,032
REM files to chkdsk). --ttl because the lock outlives the python that takes it; the release at
REM the bottom is the normal path, the TTL only covers a run the machine kills.
".venv\Scripts\python.exe" heavyio.py --acquire batch --wait 3600 --ttl 360 --note "nightly re-ID + motion batch"
if errorlevel 1 (echo [%date% %time%]   *** heavyio --acquire FAILED -- continuing with the remaining steps) else (echo [%date% %time%]   heavyio --acquire ok)

echo [%date% %time%] motion tracks for new clips...
".venv\Scripts\python.exe" clipmotion.py --device auto
if errorlevel 1 (echo [%date% %time%]   *** clipmotion FAILED -- continuing with the remaining steps) else (echo [%date% %time%]   clipmotion ok)
echo [%date% %time%] appearance embeddings for new crops...
".venv\Scripts\python.exe" embed.py --species all --min-confidence 0.5 --batch-size 16
if errorlevel 1 (echo [%date% %time%]   *** embed FAILED -- continuing with the remaining steps) else (echo [%date% %time%]   embed ok)
echo [%date% %time%] low-conf embeddings for multi-animal visits...
".venv\Scripts\python.exe" embed.py --species all --co-present --min-confidence 0.25 --batch-size 16
if errorlevel 1 (echo [%date% %time%]   *** embed --co-present FAILED -- continuing with the remaining steps) else (echo [%date% %time%]   embed --co-present ok)
echo [%date% %time%] appearance embeddings for new clip tracklets...
".venv\Scripts\python.exe" clipembed.py --device auto
if errorlevel 1 (echo [%date% %time%]   *** clipembed FAILED -- continuing with the remaining steps) else (echo [%date% %time%]   clipembed ok)
echo [%date% %time%] linking solo tracks to named individuals...
".venv\Scripts\python.exe" clipmotion.py --link
if errorlevel 1 (echo [%date% %time%]   *** clipmotion --link FAILED -- continuing with the remaining steps) else (echo [%date% %time%]   clipmotion --link ok)
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
    if errorlevel 1 (echo [%date% %time%]   *** individuals --auto-assign FAILED -- continuing with the remaining steps) else (echo [%date% %time%]   individuals --auto-assign ok)
)
echo [%date% %time%] warming the dashboard re-ID queue cache...
curl -s -o NUL --max-time 180 "http://127.0.0.1:8000/api/reid/queue?mode=recent&offset=0&limit=30"
".venv\Scripts\python.exe" heavyio.py --release batch
if errorlevel 1 (echo [%date% %time%]   *** heavyio --release FAILED -- continuing with the remaining steps) else (echo [%date% %time%]   heavyio --release ok)
echo [%date% %time%] BATCH COMPLETE
goto :eof


