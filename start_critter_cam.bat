@echo off
title Backyard Critter Cam launcher
cd /d "%~dp0"
echo.
echo   Starting the Backyard Critter Cam...
echo.
echo   - A camera window will open. Press  q  in that window to stop the rig.
echo   - A second window names each visitor's species. Close it to stop naming.
echo   - The dashboard will open in your browser in a few seconds.
echo.
start "Backyard Critter Cam  (press q here to stop)" ".venv\Scripts\python.exe" backyard_cam.py --serve
start "Critter classifier  (names species; close to stop)" ".venv\Scripts\python.exe" classify.py --watch
timeout /t 16 /nobreak >nul
start "" "http://127.0.0.1:8000"
