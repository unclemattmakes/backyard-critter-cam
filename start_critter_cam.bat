@echo off
title Backyard Critter Cam
cd /d "%~dp0"
echo.
echo   Starting the Backyard Critter Cam...
echo.
echo   In a few seconds, two things open:
echo     1. a live VIDEO window (the camera feed), and
echo     2. the dashboard, in your web browser.
echo   Species names are added automatically -- there is nothing else to start.
echo.
echo   ===  TO STOP THE APP  ==========================================
echo      Click the live VIDEO window, then press the  Q  key.
echo      (Or just close that window.)  Everything stops together.
echo   ===============================================================
echo.
echo   You can leave this little log window alone -- it closes by itself
echo   when you stop the app normally. If something goes wrong it STAYS
echo   open with an error so you (or Claude) can read what happened.
echo.
REM Starting by hand clears the "stopped on purpose" marker, so rigwatch.py will bring the rig
REM back if it dies from here on. The marker is re-created below ONLY on a clean exit -- see the
REM else-branch of the errorlevel test, and the DELIBERATE STOPS section of rigwatch.py.
if exist ".rig_pause" del /q ".rig_pause"

REM Run python INSIDE a cmd /k so the log window stays open on an abnormal exit (nonzero exit
REM code = a crash or a clean-but-unexpected self-exit), instead of vanishing without a trace --
REM that silent disappearance is exactly what hid the overnight deaths. On a normal stop ('q' /
REM closing the video window) python exits 0 and we close the window ourselves.  start /min so it
REM doesn't steal focus from the video window + browser; the full log is also in logs\.
start "Backyard Critter Cam - log (you can ignore or minimize this)" /min cmd /k ".venv\Scripts\python.exe backyard_cam.py --serve & if errorlevel 1 (echo. & echo   *** The app exited abnormally -- the error is above and in logs\backyard_cam.log. & echo   *** This window is kept open on purpose; close it when you are done reading. & echo.) else (echo stopped from the video window > .rig_pause & exit)"

REM Wait until the dashboard is actually answering (poll it) rather than guessing a fixed delay --
REM the very first run downloads the detector model and can take a while, and we never want to open
REM the browser to a "can't reach this page". Gives up after ~60s and opens anyway.
REM
REM The poll used to be inline PowerShell against a hardcoded 8000. It can't be: the dashboard now
REM asks for port 80 and falls back if it can't have it, so the port is not knowable until the
REM socket exists. mdns.py --wait-local polls the real candidates and prints the URL that answered.
set "DASHURL="
for /f "usebackq delims=" %%u in (`.venv\Scripts\python.exe mdns.py --wait-local`) do set "DASHURL=%%u"
if not defined DASHURL set "DASHURL=http://127.0.0.1"
start "" "%DASHURL%"
