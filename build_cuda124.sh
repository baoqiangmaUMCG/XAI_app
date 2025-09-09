#!/usr/bin/env bash
set -e  # stop on error

# === Create and activate conda env ===
conda create -n xai_cuda124 python=3.11.13 -y
# source ~/miniconda3/etc/profile.d/conda.sh   # adjust if conda is in a different path

conda activate xai_cuda124

# === Install runtime deps (PyTorch CUDA 12.4 build) ===
pip install -r requirements_runtime_cuda124.txt

# === Install build deps ===
pip install -r requirements_build_linux.txt

# === Build with your custom spec ===
export BUILD_NAME=app_cuda124_linux
pyinstaller app.spec

# === Exit env ===
conda deactivate
