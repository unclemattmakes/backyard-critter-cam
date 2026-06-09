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
echo   when you stop the app.
echo.
start "Backyard Critter Cam - log (you can ignore or minimize this)" ".venv\Scripts\python.exe" backyard_cam.py --serve
timeout /t 16 /nobreak >nul
start "" "http://127.0.0.1:8000"
