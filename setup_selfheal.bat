@echo off
title Backyard Critter Cam - USB self-heal setup
REM ---------------------------------------------------------------------------
REM One-time setup for the rig's USB self-heal (see usb_reset.ps1 / powerguard.py).
REM Registers the elevated scheduled task the rig fires when it detects the
REM camera's USB stream has wedged (the 2026-07-29..31 evening failures).
REM Needs admin ONCE -- this wrapper asks for elevation (UAC) and re-runs itself.
REM After setup, the rig heals wedges unattended; no elevation ever again.
REM ---------------------------------------------------------------------------
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Requesting administrator rights ^(UAC^)...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0usb_reset.ps1" -Setup
echo.
pause
