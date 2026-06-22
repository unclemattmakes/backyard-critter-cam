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
echo   Only devices on your own network can connect (you can confirm and even correct
echo   sightings from them); the dashboard refuses connections from the wider internet.
echo.
echo   ===  TO STOP THE APP  ==========================================
echo      On this PC, click the live VIDEO window and press  Q.
echo      (Or just close that window.)  Everything stops together.
echo   ===============================================================
echo.
start "Backyard Critter Cam - log (you can ignore or minimize this)" ".venv\Scripts\python.exe" backyard_cam.py --serve --host 0.0.0.0

REM Wait until the dashboard is actually answering (poll the port) rather than guessing a fixed
REM delay -- the very first run downloads the detector model and can take a while, and we never
REM want to open the browser to a "can't reach this page". Gives up after ~60s and opens anyway.
powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){ try{ $c=New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',8000); if($c.Connected){$c.Close();exit} }catch{} Start-Sleep -Milliseconds 500 }"
start "" "http://127.0.0.1:8000"
