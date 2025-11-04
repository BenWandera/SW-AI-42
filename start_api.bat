@echo off
echo ========================================
echo Waste Management API Server
echo ========================================
echo.

cd api

echo Checking Python...
python --version
echo.

echo Installing dependencies...
pip install -r requirements.txt
echo.

echo Starting API server on http://localhost:8000
echo Press Ctrl+C to stop
echo.

python main.py
