@echo off
REM GitHub Push Script for Windows
REM Push to main branch

echo ========================================
echo   GitHub Push to Main Branch
echo ========================================
echo.

REM Check if git is initialized
if not exist ".git" (
    echo [ERROR] Git repository not initialized!
    echo Please run: git init
    pause
    exit /b 1
)

REM Create .gitignore if it doesn't exist
if not exist ".gitignore" (
    echo Creating .gitignore...
    (
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo .venv/
        echo venv/
        echo.
        echo # Model files
        echo *.pth
        echo *.pt
        echo.
        echo # Data directories
        echo realwaste/
        echo hf_cache/
        echo.
        echo # Output directories
        echo mobilevit_metrics/
        echo system_metrics/
        echo gnn_visualizations/
        echo waste_gan_output/
        echo synthetic_outputs/
        echo.
        echo # Logs
        echo *.log
        echo deit_training_log.txt
    ) > .gitignore
    echo   [OK] .gitignore created
)

echo.
echo Staging all changes...
git add .

echo.
echo Current status:
git status --short

echo.
set /p COMMIT_MSG="Enter commit message (or press Enter for default): "

if "%COMMIT_MSG%"=="" (
    set COMMIT_MSG=Update: Complete waste management system with MobileViT, DeiT-Tiny, GNN, and GAN
)

echo.
echo Committing changes...
git commit -m "%COMMIT_MSG%"

echo.
echo Checking remote repository...
git remote get-url origin >nul 2>&1

if errorlevel 1 (
    echo.
    echo [INFO] No remote repository found!
    set /p REPO_URL="Enter your GitHub repository URL: "
    
    if "!REPO_URL!"=="" (
        echo [ERROR] No repository URL provided. Exiting.
        pause
        exit /b 1
    )
    
    git remote add origin "!REPO_URL!"
    echo   [OK] Remote added
)

echo.
echo Checking current branch...
for /f "tokens=*" %%i in ('git branch --show-current') do set CURRENT_BRANCH=%%i

echo Current branch: %CURRENT_BRANCH%

if not "%CURRENT_BRANCH%"=="main" (
    echo.
    echo Switching to main branch...
    git checkout -b main 2>nul || git checkout main
)

echo.
echo ========================================
echo   Pushing to GitHub (main branch)...
echo ========================================
git push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Push failed!
    echo.
    echo Common solutions:
    echo   1. Authenticate with GitHub
    echo   2. Check repository permissions
    echo   3. Try: git pull origin main --rebase
    echo   4. Run this script again
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Successfully pushed to GitHub!
echo ========================================
echo.
echo Visit your repository on GitHub to see the changes.
echo.
pause
