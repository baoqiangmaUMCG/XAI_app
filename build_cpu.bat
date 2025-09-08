@echo off
REM === Create a clean conda env ===
call conda create -n xai_cpu python=3.11.13 -y
call conda activate xai_cpu

REM === Install runtime deps ===
pip install -r requirements_runtime_cpu.txt

REM === Install build deps ===
pip install -r requirements_build_windows.txt

REM === Build with your custom spec ===
set BUILD_NAME=app_cpu_windows
pyinstaller app.spec

REM === Exit env ===
conda deactivate

pause
