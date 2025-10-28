@echo off
REM GitHub Repository Setup Script for Windows

echo ============================================================
echo   GitHub Repository Setup for Waste Management AI
echo ============================================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed. Please install Git first.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [OK] Git is installed
echo.

REM Initialize git repository if not already initialized
if not exist ".git" (
    echo Initializing Git repository...
    git init
    echo [OK] Git repository initialized
) else (
    echo [OK] Git repository already exists
)
echo.

REM Create .gitattributes for Git LFS
echo Creating .gitattributes for large files...
(
echo # Git LFS configuration for large files
echo *.pth filter=lfs diff=lfs merge=lfs -text
echo *.pt filter=lfs diff=lfs merge=lfs -text
echo *.bin filter=lfs diff=lfs merge=lfs -text
echo *.h5 filter=lfs diff=lfs merge=lfs -text
echo *.ckpt filter=lfs diff=lfs merge=lfs -text
) > .gitattributes

echo [OK] .gitattributes created
echo.

REM Add files to git
echo Adding files to git...
git add .gitignore
git add .gitattributes
git add LICENSE
git add README_GITHUB.md
git add requirements.txt
git add *.py
git add "GNN model"
git add *.json
git add *.md

echo [OK] Files staged for commit
echo.

REM Show status
echo ============================================================
echo   Git Status:
echo ============================================================
git status
echo.

REM Instructions
echo ============================================================
echo   NEXT STEPS TO PUSH TO GITHUB
echo ============================================================
echo.
echo 1. Review the staged files above
echo.
echo 2. Commit your changes:
echo    git commit -m "Initial commit: AI Waste Management System"
echo.
echo 3. Create a new repository on GitHub:
echo    - Go to https://github.com/new
echo    - Name: waste-management-ai
echo    - Description: AI-powered waste classification with Vision Transformers and GNN
echo    - Make it Public or Private
echo    - Do NOT initialize with README (we already have one)
echo.
echo 4. Add GitHub as remote:
echo    git remote add origin https://github.com/YOUR_USERNAME/waste-management-ai.git
echo.
echo 5. Rename branch to main:
echo    git branch -M main
echo.
echo 6. Push to GitHub:
echo    git push -u origin main
echo.
echo ============================================================
echo   OPTIONAL: Git LFS for Large Model Files
echo ============================================================
echo.
echo If you want to include model files (*.pth):
echo    1. Download Git LFS: https://git-lfs.github.com/
echo    2. Install and run: git lfs install
echo    3. Track model files: git lfs track "*.pth"
echo    4. Commit and push as normal
echo.
echo WARNING: GitHub has a 100MB file size limit without LFS
echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo Your repository is ready to be pushed to GitHub.
echo Follow the steps above to complete the process.
echo.
pause
