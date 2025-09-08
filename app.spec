# -*- mode: python ; coding: utf-8 -*-
import importlib.util, os

block_cipher = None

# Absolute path to your script
#script_path = r"C:\Users\bma2\OneDrive - UMC Utrecht\projects\XAI_app\app.py"
script_path = os.path.abspath("app.py")

# Get build name from environment variable, fallback to default
build_name = os.environ.get("BUILD_NAME", "app_default")


def package_file(pkg, filename):
    """Collect one specific file inside a package."""
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.origin:
        raise ImportError(f"Could not find package {pkg}")
    base = os.path.dirname(spec.origin)
    return os.path.join(base, filename), pkg

def package_files(pkg):
    """Collect all non-.py files inside a package."""
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.origin:
        raise ImportError(f"Could not find package {pkg}")
    base = os.path.dirname(spec.origin)
    collected = []
    for root, _, files in os.walk(base):
        for f in files:
            if not f.endswith(('.py', '.pyc', '.pyo')):
                fullpath = os.path.join(root, f)
                relpath = os.path.relpath(root, os.path.dirname(base))
                collected.append((fullpath, os.path.join(pkg, relpath)))
    return collected

# Explicit critical files
added_files = [
    package_file("gradio_client", "types.json"),
    package_file("safehttpx", "version.txt"),
    package_file("groovy", "version.txt"),
]

# Plus all other data files from those packages
for pkg in ["gradio_client", "safehttpx", "groovy"]:
    added_files.extend(package_files(pkg))

a = Analysis(
    [script_path],
    pathex=[os.path.dirname(script_path)],
    binaries=[],
    datas=added_files,
    hiddenimports=[],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=build_name, #name='app_cuda118_windows'
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # True if you want a console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
