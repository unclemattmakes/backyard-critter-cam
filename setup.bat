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

REM 3) Install torch, matched to the GPU GENERATION, not just GPU-or-not. CUDA wheels drop
REM    kernels for old generations and gain them late for new ones, and both mistakes are
REM    SILENT (torch.cuda.is_available() lies, then either the first op dies or --device auto
REM    quietly runs the CPU forever). Ask the driver for the card's compute capability:
REM      >= 12.0 (Blackwell, RTX 50)         -> cu130 is REQUIRED (cu126 has no sm_120 kernels)
REM      7.5 - 8.9 (Turing/Ampere/Ada)       -> cu130 fine (cu126 also fine)
REM      <  7.5 (Maxwell/Pascal/Volta)       -> cu126: cu130 dropped these; installing it would
REM                                             silently cost this machine its GPU
REM    Unreadable capability (odd driver) -> cu130, the pre-2026-08 behaviour.
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  set "CC="
  for /f "usebackq tokens=1 delims=." %%c in (`nvidia-smi --query-gpu=compute_cap --format=csv^,noheader 2^>nul`) do if not defined CC set "CC=%%c"
  if not defined CC set "CC=99"
  if !CC! LSS 7 (
    echo NVIDIA GPU detected ^(compute capability !CC!.x -- Maxwell/Pascal/Volta era^).
    echo The newest CUDA wheels dropped this generation, so installing the cu126 build
    echo -- with cu130 this card would silently go unused.
    "%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || (echo [ERROR] torch install failed. & pause & exit /b 1)
  ) else (
    echo NVIDIA GPU detected ^(compute capability !CC!.x^) -- installing the CUDA ^(cu130^) torch build ...
    "%VPY%" -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130 || (echo [ERROR] torch install failed. & pause & exit /b 1)
  )
) else (
  echo No NVIDIA GPU detected -- installing the CPU torch build ^(slower, but it works^) ...
  "%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || (echo [ERROR] torch install failed. & pause & exit /b 1)
)

REM 4) The rest of the dependencies.
echo Installing the remaining requirements ...
"%VPY%" -m pip install -r requirements.txt || (echo [ERROR] requirements install failed. & pause & exit /b 1)

REM 4b) Record what this machine actually resolved. "Setup chooses a build per machine" must not
REM     mean "nobody knows which build produced these numbers" -- the eval artifacts and the
REM     stored embeddings were computed by SOME torch/ultralytics, and this file says which.
REM     Machine-specific (gitignored); backup.py carries it in the weekly meta snapshot.
echo # resolved by setup.bat on %date% %time% > environment.lock.txt
"%VPY%" -m pip freeze | findstr /i "torch ultralytics numpy opencv open_clip open-clip timm" >> environment.lock.txt
echo Recorded the resolved versions to environment.lock.txt

REM 5) ffmpeg is not a Python dependency, so pip can't install it -- but the rig quietly loses
REM    two features without it: clips fall back to the mp4v codec no browser will play, and the
REM    Dispatch highlight reel stitches with ffmpeg or not at all. Warn, don't fail.
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo.
  echo [WARNING] ffmpeg is not on PATH. Everything still runs, but behaviour clips fall back
  echo           to an mp4v codec browsers refuse to play, and the Dispatch highlight reel
  echo           will report "ffmpeg not found on PATH" instead of stitching a reel.
  echo           Install it with:  winget install Gyan.FFmpeg
  echo           then open a NEW terminal so the updated PATH is picked up.
)

echo.
echo === Setup complete! ===
echo Start the app by double-clicking  start_critter_cam.bat
echo (or run:  .venv\Scripts\python.exe backyard_cam.py --serve )
echo.
pause
