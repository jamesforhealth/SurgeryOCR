# -*- mode: python ; coding: utf-8 -*-
import os

# ===== 1️⃣ 讀取環境變數決定模式 =====
BUILD_MODE = os.getenv('BUILD_MODE', 'DEV')  # 預設 DEV
IS_RELEASE = BUILD_MODE == 'RELEASE'

print(f"\n{'='*60}")
print(f"🔨 Building: {BUILD_MODE} Mode")
print(f"{'='*60}\n")

# ===== 2️⃣ 根據模式設定不同的排除項 =====
if IS_RELEASE:
    # Release 版本：排除所有重量級和非必要的套件
    excludes_list = [
        # --- AI/ML 相關 ---
        'torch',
        'torchvision',
        'torchaudio',
        'easyocr',
        'sklearn',

        # --- 科學計算/數據分析 ---
        'scipy',
        'pandas',
        'matplotlib',

        # --- 互動式環境/筆記本 ---
        'IPython',
        'jupyter',
        'notebook',

        # --- 其他 GUI 框架 (因為我們只用 Tkinter) ---
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',

        # --- 測試與文檔工具 ---
        'doctest',
        'unittest',
        'pydoc',
        'pytest',

        # --- 其他較少用到的大型套件 ---
        'bokeh',
        'numba',
        'dask',
    ]
    hidden_imports_list = []
    console_mode = False  # Release 不顯示控制台
    app_name = 'VideoAnnotator_Release'
else:
    # Dev 版本：包含所有功能
    excludes_list = []
    hidden_imports_list = [
        'torch',
        'easyocr',
    ]
    console_mode = True  # Dev 顯示控制台方便除錯
    app_name = 'VideoAnnotator_Dev'

# ===== 3️⃣ Analysis 配置 =====
a = Analysis(
    ['video_annotator_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('config', 'config')],
    hiddenimports=hidden_imports_list,  # ← 使用動態清單
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes_list,  # ← 使用動態清單
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=app_name,  # ← 動態命名
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=console_mode,  # ← 動態控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可選：添加 icon='app.ico'
)

# ===== 4️⃣ 打包完成提示 =====
print(f"\n{'='*60}")
print(f"✅ {BUILD_MODE} build configuration completed!")
print(f"📦 Output will be: dist/{app_name}.exe")
if IS_RELEASE:
    print(f"🎯 Excluded heavy packages: {', '.join(excludes_list)}")
print(f"{'='*60}\n")