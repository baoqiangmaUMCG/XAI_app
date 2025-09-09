#!/usr/bin/env bash
set -e  # stop on error

# === Create and activate conda env ===
conda create -n xai_macos python=3.11.13 -y
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate xai_macos

# === Install runtime deps (CPU/MPS build of PyTorch) ===
pip install -r requirements_runtime_macos.txt

# === Install build deps ===
pip install -r requirements_build_macos.txt

# === Build with your custom spec ===
export BUILD_NAME=app_macos_mps
pyinstaller app.spec

# === Exit env ===
conda deactivate
