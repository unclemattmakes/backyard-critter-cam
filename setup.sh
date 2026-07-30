#!/usr/bin/env bash
# ============================================================================
#  Backyard Critter Cam - one-time setup (Linux / macOS).
#  Creates the .venv, installs a torch build matched to your hardware (NVIDIA
#  GPU -> CUDA build; macOS -> default MPS/CPU wheel; otherwise CPU), then the
#  rest of the requirements.  Run with:  bash setup.sh
# ============================================================================
set -e
cd "$(dirname "$0")"

echo
echo "=== Backyard Critter Cam - setup ==="
echo

# 1) Find a Python 3.10+ interpreter.
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "[ERROR] No python3 found. Install Python 3.10+ and re-run." >&2
  exit 1
fi
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)'; then
  echo "[ERROR] Your Python is older than 3.10. Install a newer one and re-run." >&2
  exit 1
fi
echo "Using Python: $PY ($("$PY" --version 2>&1))"

# 2) Create the virtual environment.
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv
fi
VPY=".venv/bin/python"
"$VPY" -m pip install --upgrade pip >/dev/null

# 3) Install torch matched to the hardware.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected -- installing the CUDA (cu130) torch build ..."
  "$VPY" -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
elif [ "$(uname -s)" = "Darwin" ]; then
  echo "macOS detected -- installing the default torch build (CPU/MPS) ..."
  "$VPY" -m pip install torch torchvision
else
  echo "No NVIDIA GPU detected -- installing the CPU torch build (slower, but it works) ..."
  "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# 4) The rest of the dependencies.
echo "Installing the remaining requirements ..."
"$VPY" -m pip install -r requirements.txt

# 5) ffmpeg is not a Python dependency, so pip can't install it -- but the rig quietly loses
#    two features without it: clips fall back to the mp4v codec no browser will play, and the
#    Dispatch highlight reel stitches with ffmpeg or not at all. Warn, don't fail.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo
  echo "[WARNING] ffmpeg is not on PATH. Everything still runs, but behaviour clips fall back"
  echo "          to an mp4v codec browsers refuse to play, and the Dispatch highlight reel"
  echo "          will report 'ffmpeg not found on PATH' instead of stitching a reel."
  if [ "$(uname -s)" = "Darwin" ]; then
    echo "          Install it with:  brew install ffmpeg"
  else
    echo "          Install it with:  sudo apt install ffmpeg   [or your distro's equivalent]"
  fi
fi

echo
echo "=== Setup complete! ==="
echo "Run:  .venv/bin/python backyard_cam.py --serve     then open http://127.0.0.1:8000"
echo
