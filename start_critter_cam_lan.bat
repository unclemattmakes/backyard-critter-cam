@echo off
title Backyard Critter Cam  (LAN -- also viewable on phones and tablets)
cd /d "%~dp0"

REM The address block is NOT printed here any more. It used to be: this file detected the LAN IP
REM with PowerShell, echoed "http://<ip>:8000" among the startup banner, and then -- having opened
REM the browser -- ENDED, closing the window and taking the only copy of the address with it. You
REM got one glance at it while the rig was still booting.
REM
REM Two changes fix that. The block is printed AFTER the dashboard answers, by mdns.py, which
REM knows the name the rig published for itself ("critter-cam.local") as well as the number and
REM prints whichever ones are actually true. And this window then STAYS, so the address is
REM somewhere you can go back and look instead of something you had to catch.

echo.
echo   Starting the Backyard Critter Cam  (LAN mode -- other devices on your Wi-Fi can watch)...
echo.
echo   On THIS PC, in a few seconds: a live VIDEO window opens, and the dashboard
echo   opens in your browser. Species names are added automatically.
echo   The address for phones and tablets is printed below once it is up.
echo   (First launch may show a Windows Firewall prompt -- click Allow on Private networks.)
echo.
echo   Only devices on your own network can connect (you can confirm and even correct
echo   sightings from them); it refuses direct internet connections, but has no password.
echo.
echo   ===  TO STOP THE APP  ==========================================
echo      On this PC, click the live VIDEO window and press  Q.
echo      (Or just close that window.)  Everything stops together.
echo   ===============================================================
echo.
REM Starting by hand clears the "stopped on purpose" marker, so rigwatch.py will bring the rig
REM back if it dies from here on -- the same contract start_critter_cam.bat has always had. This
REM launcher was missing BOTH halves of it: it never cleared the marker and never wrote one, so a
REM LAN start left a stale marker standing (rig up but unguarded) and a LAN 'q' was not respected
REM as a deliberate stop. rigwatch also self-heals a stale marker now, but write it correctly here.
if exist ".rig_pause" del /q ".rig_pause"

REM Run python INSIDE a cmd /k so the log window stays open on an abnormal exit instead of
REM vanishing without a trace; on a normal stop ('q' / closing the video window) python exits 0,
REM we drop the pause marker so the watchdog leaves it alone, and close the window ourselves.
start "Backyard Critter Cam - log (you can ignore or minimize this)" /min cmd /k ".venv\Scripts\python.exe backyard_cam.py --serve --host 0.0.0.0 & if errorlevel 1 (echo. & echo   *** The app exited abnormally -- the error is above and in logs\backyard_cam.log. & echo   *** This window is kept open on purpose; close it when you are done reading. & echo.) else (echo stopped from the video window > .rig_pause & exit)"

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

REM Now the rig is up and has announced itself, so the addresses below are the real ones. --host
REM mirrors the flag python was launched with above: config.py alone still says localhost, and a
REM block that quietly disagreed with the running rig would be worse than no block.
echo.
echo   ===  HOW OTHERS CONNECT  =======================================
echo.
.venv\Scripts\python.exe mdns.py --host 0.0.0.0
echo.
echo      Same Wi-Fi only. If the name does not work on an Android phone,
echo      use the numeric address -- some Android browsers do not do mDNS.
echo   ===============================================================
echo.
echo   Leave this window open to keep the address handy, or close it --
echo   closing it does NOT stop the rig (stop with Q in the video window).
echo.
pause
