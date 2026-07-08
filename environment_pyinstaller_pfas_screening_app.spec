# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules


hiddenimports = collect_submodules("sklearn")
model_data = [
    (
        "models/rf_sklearn_bundle_POS.joblib",
        "models",
    ),
    (
        "models/rf_sklearn_bundle_NEG.joblib",
        "models",
    ),
]

a = Analysis(
    ["launch_pfas_app.py"],
    pathex=[],
    binaries=[],
    datas=model_data,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "PIL"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PFAS_CCS_Screening",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PFAS_CCS_Screening",
)
