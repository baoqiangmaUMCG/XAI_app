from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# sklearn loads Cython extensions dynamically
hiddenimports = collect_submodules("sklearn")

# Include sklearn's data files (like joblib templates, etc.)
datas = collect_data_files("sklearn")
