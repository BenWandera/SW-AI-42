@echo off
echo ============================================================
echo Starting REAL Waste Management API with MobileViT Model
echo ============================================================
echo.

cd /d %~dp0

echo Stopping any existing Python API servers...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting API with trained MobileViT model...
echo.

python real_api.py

pause
