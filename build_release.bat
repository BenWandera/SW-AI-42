@echo off
echo ========================================
echo Building Release APK
echo ========================================
echo.

cd waste_flutter_app

echo Building APK... This may take 2-5 minutes...
flutter build apk --release

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo APK Location:
dir build\app\outputs\flutter-apk\app-release.apk
echo.
