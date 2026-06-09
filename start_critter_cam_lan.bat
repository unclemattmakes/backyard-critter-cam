@echo off
title Backyard Critter Cam launcher  (LAN)
cd /d "%~dp0"

rem Detect this PC's LAN IPv4 so we can show the URL other devices should open. The address
rem is the one on the adapter that actually has a default gateway (your live Wi-Fi/Ethernet).
rem Falls back to the hostname if detection fails (works PC-to-PC; phones prefer the IP).
set "LANIP="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPConfiguration | Where-Object {$_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up'} | Select-Object -First 1 -ExpandProperty IPv4Address).IPAddress"`) do set "LANIP=%%i"
if not defined LANIP set "LANIP=%COMPUTERNAME%"

echo.
echo   Starting the Backyard Critter Cam  (LAN mode -- visible to other devices on your Wi-Fi)...
echo.
echo   - A camera window will open. Press  q  in that window to stop the rig.
echo   - A second window names each visitor's species. Close it to stop naming.
echo   - The dashboard opens here in your browser in a few seconds.
echo.
echo   From another device on the SAME Wi-Fi, open:   http://%LANIP%:8000
echo   (First launch may pop a Windows Firewall prompt -- click Allow on Private networks.)
echo.
start "Backyard Critter Cam  (press q here to stop)" ".venv\Scripts\python.exe" backyard_cam.py --serve --host 0.0.0.0
start "Critter classifier  (names species; close to stop)" ".venv\Scripts\python.exe" classify.py --watch
timeout /t 16 /nobreak >nul
start "" "http://127.0.0.1:8000"
