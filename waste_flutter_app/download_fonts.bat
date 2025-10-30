@echo off
echo ================================================
echo    SF Pro Font Downloader for EcoWaste AI
echo ================================================
echo.

cd /d "%~dp0"

echo Downloading SF Pro fonts from GitHub...
echo.

if not exist "assets\fonts" mkdir "assets\fonts"

powershell -Command "Invoke-WebRequest -Uri 'https://github.com/sahibjotsaggu/San-Francisco-Pro-Fonts/raw/master/SF-Pro-Display-Regular.otf' -OutFile 'assets/fonts/SF-Pro-Display-Regular.otf'"
echo Downloaded: SF-Pro-Display-Regular.otf

powershell -Command "Invoke-WebRequest -Uri 'https://github.com/sahibjotsaggu/San-Francisco-Pro-Fonts/raw/master/SF-Pro-Display-Medium.otf' -OutFile 'assets/fonts/SF-Pro-Display-Medium.otf'"
echo Downloaded: SF-Pro-Display-Medium.otf

powershell -Command "Invoke-WebRequest -Uri 'https://github.com/sahibjotsaggu/San-Francisco-Pro-Fonts/raw/master/SF-Pro-Display-Semibold.otf' -OutFile 'assets/fonts/SF-Pro-Display-Semibold.otf'"
echo Downloaded: SF-Pro-Display-Semibold.otf

powershell -Command "Invoke-WebRequest -Uri 'https://github.com/sahibjotsaggu/San-Francisco-Pro-Fonts/raw/master/SF-Pro-Display-Bold.otf' -OutFile 'assets/fonts/SF-Pro-Display-Bold.otf'"
echo Downloaded: SF-Pro-Display-Bold.otf

echo.
echo ================================================
echo Font download complete!
echo.
echo Cleaning Flutter project...
call flutter clean

echo.
echo Getting dependencies...
call flutter pub get

echo.
echo ================================================
echo All done! Fonts are ready to use.
echo Now building APK...
echo ================================================
echo.

call flutter build apk --release

echo.
echo ================================================
echo APK built successfully with SF Pro font!
echo ================================================
pause
