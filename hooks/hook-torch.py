from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs
import sys

hiddenimports = collect_submodules("torch")

# Include all torch .py files
datas = collect_data_files("torch", include_py_files=True)

# On Linux, also include the .so libraries
if sys.platform.startswith("linux"):
    binaries = collect_dynamic_libs("torch")
else:
    binaries = []
