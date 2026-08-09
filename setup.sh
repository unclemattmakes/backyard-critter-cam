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

# 3) Install torch matched to the GPU GENERATION, not just GPU-or-not. CUDA wheels drop kernels
#    for old generations and gain them late for new ones, and both mistakes are SILENT
#    (torch.cuda.is_available() lies, then either the first op dies or --device auto quietly
#    runs the CPU forever). Ask the driver for the card's compute capability:
#      >= 12.0 (Blackwell, RTX 50)      -> cu130 REQUIRED (cu126 has no sm_120 kernels)
#      7.5 - 8.9 (Turing/Ampere/Ada)    -> cu130 fine
#      <  7.5 (Maxwell/Pascal/Volta)    -> cu126: cu130 dropped these; installing it would
#                                          silently cost this machine its GPU
#    The cut is 7.5, and it MUST compare the minor version too: Volta is sm_70 and Turing is
#    sm_75, so a major-only test (the first version of this, 2026-08-08) sent every Volta card
#    to cu130 -- precisely the silent GPU loss the check exists to prevent.
if command -v nvidia-smi >/dev/null 2>&1; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
  CC10="$(printf '%s' "$CC" | awk -F. 'NF{printf "%d", $1*10 + ($2=="" ? 0 : $2)}')"
  if [ -n "$CC10" ] && [ "$CC10" -lt 75 ]; then
    echo "NVIDIA GPU detected (compute capability ${CC} -- Maxwell/Pascal/Volta era)."
    echo "The newest CUDA wheels dropped this generation, so installing the cu126 build"
    echo "-- with cu130 this card would silently go unused."
    "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  else
    echo "NVIDIA GPU detected (compute capability ${CC:-unknown}) -- installing the CUDA (cu130) torch build ..."
    "$VPY" -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
  fi
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

# 4b) Record what this machine actually resolved. "Setup chooses a build per machine" must not
#     mean "nobody knows which build produced these numbers" -- the eval artifacts and stored
#     embeddings were computed by SOME torch/ultralytics, and this file says which.
#     Machine-specific (gitignored); backup.py carries it in the weekly meta snapshot.
{
  echo "# resolved by setup.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$VPY" -m pip freeze | grep -iE 'torch|ultralytics|numpy|opencv|open._?clip|timm' || true
} > environment.lock.txt
echo "Recorded the resolved versions to environment.lock.txt"

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
