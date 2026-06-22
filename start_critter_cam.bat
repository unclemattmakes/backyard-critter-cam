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

REM Wait until the dashboard is actually answering (poll the port) rather than guessing a fixed
REM delay -- the very first run downloads the detector model and can take a while, and we never
REM want to open the browser to a "can't reach this page". Gives up after ~60s and opens anyway.
powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){ try{ $c=New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',8000); if($c.Connected){$c.Close();exit} }catch{} Start-Sleep -Milliseconds 500 }"
start "" "http://127.0.0.1:8000"
