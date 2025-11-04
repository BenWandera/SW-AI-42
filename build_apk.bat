@echo off
echo ========================================
echo Building Release APK
echo ========================================
echo.

cd waste_flutter_app

echo Building APK...
flutter build apk --release

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo APK Location:
echo %cd%\build\app\outputs\flutter-apk\app-release.apk
echo.
echo Instructions:
echo 1. Connect your Android phone via USB
echo 2. Enable USB Debugging on your phone
echo 3. Run: adb install build\app\outputs\flutter-apk\app-release.apk
echo.
echo Or copy the APK to your phone and install manually!
echo.
pause
