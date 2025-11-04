@echo off
echo =========================================
echo  Active Learning System - Quick Test
echo =========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

echo [1/3] Checking API status...
echo.

REM Check if API is running
curl -s http://localhost:8000/ >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: API is not running!
    echo.
    echo Starting API server...
    echo.
    start "Waste API Server" cmd /k "cd api && python real_api.py"
    echo.
    echo Waiting for API to start (10 seconds)...
    timeout /t 10 /nobreak >nul
)

echo.
echo [2/3] API is ready!
echo.

echo [3/3] Running Active Learning Demo...
echo.

python test_active_learning.py --auto

echo.
echo =========================================
echo  Test Complete!
echo =========================================
echo.
echo Next Steps:
echo  1. Review the output above
echo  2. Check feedback_data/ folder for stored data
echo  3. Check model_backups/ for model versions
echo  4. See ACTIVE_LEARNING_GUIDE.md for details
echo.

pause
