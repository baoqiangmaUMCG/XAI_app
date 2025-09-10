import os, sys, glob
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

hiddenimports = collect_submodules("torch")
datas = collect_data_files("torch", include_py_files=True)
binaries = collect_dynamic_libs("torch")

if sys.platform.startswith("linux"):
    cudnn_libs = []

    # Get current conda/venv prefix
    env_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)

    #cudnn_path = os.path.join(env_prefix, "lib", "python3.11", "site-packages", "nvidia", "cudnn", "lib")
    
    #import sys
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cudnn_path = os.path.join(env_prefix, "lib", pyver, "site-packages", "nvidia", "cudnn", "lib")
    
    if os.path.isdir(cudnn_path):
        cudnn_libs.extend(glob.glob(os.path.join(cudnn_path, "libcudnn*.so*")))

    for lib in cudnn_libs:
        binaries.append((lib, "torch/lib"))

        # Ensure symlink chain (.so ? .so.9 ? .so.9.1.0) is available
        if ".so." in lib:
            base = lib.split(".so.")[0] + ".so"
            linkname = os.path.join(os.path.dirname(lib), os.path.basename(base))
            if not os.path.exists(linkname):
                try:
                    os.symlink(os.path.basename(lib), linkname)
                except FileExistsError:
                    pass
            binaries.append((linkname, "torch/lib"))

print (binaries)
