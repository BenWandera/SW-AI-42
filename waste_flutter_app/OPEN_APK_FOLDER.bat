@echo off
echo ================================================
echo    Opening APK Folder
echo ================================================
echo.
echo Your APK file is located at:
echo %~dp0EcoWaste-AI.apk
echo.
echo File Size: ~30 MB
echo.
echo ================================================
echo Opening folder in Windows Explorer...
echo ================================================
start explorer "%~dp0"
echo.
echo Look for: EcoWaste-AI.apk
echo.
pause
