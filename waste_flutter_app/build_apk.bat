@echo off
echo ========================================
echo Building Release APK for Android
echo ========================================
echo.

cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo Starting APK build...
echo This will take 3-5 minutes...
echo.

flutter build apk --release

echo.
echo ========================================
if exist build\app\outputs\flutter-apk\app-release.apk (
    echo ✅ SUCCESS! APK built successfully!
    echo.
    echo APK Location:
    echo %CD%\build\app\outputs\flutter-apk\app-release.apk
    echo.
    echo File size:
    dir build\app\outputs\flutter-apk\app-release.apk | find "app-release.apk"
    echo.
    echo Next steps:
    echo 1. Copy this APK to your phone
    echo 2. Install it (enable "Install from unknown sources" if needed)
    echo 3. Open the app and start classifying waste!
    echo.
    echo See INSTALL_APK_GUIDE.md for detailed instructions
) else (
    echo ❌ Build failed. Check the output above for errors.
)
echo ========================================
echo.
pause
