@REM build_release.bat
@echo off
echo.
echo ========================================
echo 🚀 Building RELEASE Version
echo ========================================
echo.

set BUILD_MODE=RELEASE

REM 清理舊檔案
if exist dist\VideoAnnotator_Release.exe del dist\VideoAnnotator_Release.exe
if exist build rmdir /s /q build

REM 打包
pyinstaller VideoAnnotator.spec --log-level=DEBUG

echo.
echo ========================================
if exist dist\VideoAnnotator_Release.exe (
    echo ✅ SUCCESS!
    echo 📦 Output: dist\VideoAnnotator_Release.exe
    dir dist\VideoAnnotator_Release.exe | find "VideoAnnotator_Release.exe"
) else (
    echo ❌ FAILED!
)
echo ========================================
pause