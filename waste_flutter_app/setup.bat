@echo off
echo ============================================
echo    EcoWaste AI - Flutter App Setup
echo ============================================
echo.

REM Check if Flutter is installed
echo [1/5] Checking Flutter installation...
flutter --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Flutter is not installed!
    echo.
    echo Please install Flutter from: https://flutter.dev/docs/get-started/install/windows
    echo After installation, add Flutter to your PATH and restart this script.
    pause
    exit /b 1
)
echo [OK] Flutter is installed
echo.

REM Check Flutter doctor
echo [2/5] Running Flutter doctor...
flutter doctor
echo.

REM Install dependencies
echo [3/5] Installing Flutter dependencies...
flutter pub get
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Check for connected devices
echo [4/5] Checking for connected devices...
flutter devices
echo.

REM Ask if user wants to run the app
echo [5/5] Setup complete!
echo.
echo ============================================
echo    Setup Complete!
echo ============================================
echo.
echo Your Flutter app is ready to run.
echo.
echo To run the app:
echo   1. Connect your phone via USB (with USB debugging enabled)
echo   2. Or start an Android/iOS emulator
echo   3. Run: flutter run
echo.
echo To build APK:
echo   - Debug:   flutter build apk --debug
echo   - Release: flutter build apk --release
echo.
echo Documentation:
echo   - README.md       - Complete documentation
echo   - QUICKSTART.md   - Quick setup guide
echo   - PROJECT_SUMMARY.md - Feature overview
echo.
echo ============================================

set /p run="Do you want to run the app now? (y/n): "
if /i "%run%"=="y" (
    echo.
    echo Starting app...
    flutter run
) else (
    echo.
    echo You can run the app later with: flutter run
)

echo.
echo Press any key to exit...
pause >nul
