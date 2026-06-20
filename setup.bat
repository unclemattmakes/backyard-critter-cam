@echo off
REM ============================================================================
REM  Backyard Critter Cam - one-time setup (Windows).
REM  Creates the .venv, installs a torch build matched to your hardware (NVIDIA
REM  GPU -> CUDA build; otherwise CPU), then the rest of the requirements.
REM  Just double-click this file. Re-running it is safe.
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo === Backyard Critter Cam - setup ===
echo.

REM 1) Find a Python interpreter. Prefer the 'py' launcher, then 'python' on PATH.
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] No Python found on PATH. Install Python 3.10 or newer from
  echo         https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^),
  echo         then run this script again.
  pause & exit /b 1
)
echo Using Python: %PY%
%PY% -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)"
if errorlevel 1 (
  echo [ERROR] Your Python is older than 3.10. Install a newer one and re-run.
  pause & exit /b 1
)

REM 2) Create the virtual environment (local to this folder; leaves any system torch alone).
if not exist ".venv" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv || (echo [ERROR] venv creation failed. & pause & exit /b 1)
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip >nul

REM 3) Install torch. NVIDIA GPU present -> CUDA 12.8+ (cu130) build; otherwise the CPU build.
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo NVIDIA GPU detected -- installing the CUDA ^(cu130^) torch build ...
  "%VPY%" -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130 || (echo [ERROR] torch install failed. & pause & exit /b 1)
) else (
  echo No NVIDIA GPU detected -- installing the CPU torch build ^(slower, but it works^) ...
  "%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || (echo [ERROR] torch install failed. & pause & exit /b 1)
)

REM 4) The rest of the dependencies.
echo Installing the remaining requirements ...
"%VPY%" -m pip install -r requirements.txt || (echo [ERROR] requirements install failed. & pause & exit /b 1)

echo.
echo === Setup complete! ===
echo Start the app by double-clicking  start_critter_cam.bat
echo (or run:  .venv\Scripts\python.exe backyard_cam.py --serve )
echo.
pause
