from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules("torch")

# Add all .py files in torch (so inspect.getsource() works)
datas = collect_data_files("torch", include_py_files=True)