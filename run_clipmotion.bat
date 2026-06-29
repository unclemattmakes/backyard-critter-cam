@echo off
title Backyard Critter Cam - nightly motion tracks
cd /d "%~dp0"
REM ---------------------------------------------------------------------------
REM Phase 4 batch, run nightly by Task Scheduler (see setup_motion_schedule.bat).
REM   1) clipmotion.py        -- turn every NEW behaviour clip into motion tracks
REM   2) clipmotion.py --link -- attach the solo-clip tracks to their named individual
REM Both steps are RESUMABLE (only untracked clips / unlinked solo tracks are touched),
REM so a missed night just catches up the next day. Scheduled for the ~2pm activity
REM trough so it doesn't fight the 9pm-2am capture peak for the GPU.
REM ---------------------------------------------------------------------------
echo [%date% %time%] building motion tracks for new clips...
".venv\Scripts\python.exe" clipmotion.py --device auto
echo [%date% %time%] linking solo tracks to named individuals...
".venv\Scripts\python.exe" clipmotion.py --link
echo [%date% %time%] done.
