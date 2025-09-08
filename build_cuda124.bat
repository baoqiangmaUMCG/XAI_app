@echo off
REM === Create a clean conda env ===
call conda create -n xai_cuda124 python=3.11.13 -y
call conda activate xai_cuda124

REM === Install runtime deps ===
pip install -r requirements_runtime_cuda124.txt

REM === Install build deps ===
pip install -r requirements_build_windows.txt

REM === Build with your custom spec ===
set BUILD_NAME=app_cuda124_windows
pyinstaller app.spec

REM === Exit env ===
conda deactivate

pause
