@echo off
title Backyard Critter Cam  (LAN -- also viewable on phones and tablets)
cd /d "%~dp0"

rem Find this PC's LAN IPv4 (the adapter that has a default gateway = your live Wi-Fi/Ethernet)
rem so we can print the URL other devices should open. Falls back to the hostname if detection
rem fails (works PC-to-PC; phones prefer the numeric IP).
set "LANIP="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPConfiguration | Where-Object {$_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up'} | Select-Object -First 1 -ExpandProperty IPv4Address).IPAddress"`) do set "LANIP=%%i"
if not defined LANIP set "LANIP=%COMPUTERNAME%"

echo.
echo   Starting the Backyard Critter Cam  (LAN mode -- other devices on your Wi-Fi can watch)...
echo.
echo   On THIS PC, in a few seconds: a live VIDEO window opens, and the dashboard
echo   opens in your browser. Species names are added automatically.
echo.
echo   On a phone or tablet on the SAME Wi-Fi, open:   http://%LANIP%:8000
echo   (First launch may show a Windows Firewall prompt -- click Allow on Private networks.)
echo.
echo   ===  TO STOP THE APP  ==========================================
echo      On this PC, click the live VIDEO window and press  Q.
echo      (Or just close that window.)  Everything stops together.
echo   ===============================================================
echo.
start "Backyard Critter Cam - log (you can ignore or minimize this)" ".venv\Scripts\python.exe" backyard_cam.py --serve --host 0.0.0.0
timeout /t 16 /nobreak >nul
start "" "http://127.0.0.1:8000"
