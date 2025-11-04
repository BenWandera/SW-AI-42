@echo off
echo ============================================================
echo Starting Waste Management API Server
echo ============================================================
echo.
echo Server will run at: http://192.168.100.152:8000
echo.
echo IMPORTANT: Keep this window open while using the app!
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

cd /d "C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\api"
python real_api.py

pause
