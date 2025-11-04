@echo off
REM Setup Git LFS and Upload Everything to GitHub
REM This will upload ALL files including large models

echo ===============================================================
echo      Git LFS Setup - Upload Everything to GitHub
echo ===============================================================

REM Step 1: Check if Git LFS is installed
echo.
echo 1. Checking Git LFS installation...
where git-lfs >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [X] Git LFS is not installed!
    echo.
    echo Install Git LFS:
    echo    Download from: https://git-lfs.github.com/
    echo    Or use: winget install -e --id GitHub.GitLFS
    echo.
    echo After installation, run this script again.
    pause
    exit /b 1
) else (
    echo    [OK] Git LFS is installed
    git lfs version
)

REM Step 2: Initialize Git LFS
echo.
echo 2. Initializing Git LFS in repository...
git lfs install
echo    [OK] Git LFS initialized

REM Step 3: Track large file types with Git LFS
echo.
echo 3. Configuring Git LFS to track large files...

git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.ckpt"
git lfs track "*.bin"
git lfs track "*.h5"
git lfs track "*.pkl"
git lfs track "*.zip"
git lfs track "*.tar.gz"
git lfs track "*.rar"

echo    [OK] Git LFS tracking configured
echo.
echo    Tracking:
echo      * *.pth (PyTorch models)
echo      * *.pt (PyTorch tensors)
echo      * *.ckpt (Checkpoints)
echo      * *.bin (Binary models)
echo      * *.h5 (Keras/HDF5 models)
echo      * *.zip, *.tar.gz (Archives)

REM Step 4: Update .gitignore to allow large files
echo.
echo 4. Updating .gitignore to include everything...

(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo build/
echo develop-eggs/
echo dist/
echo downloads/
echo eggs/
echo .eggs/
echo lib/
echo lib64/
echo parts/
echo sdist/
echo var/
echo wheels/
echo *.egg-info/
echo .installed.cfg
echo *.egg
echo.
echo # Virtual Environment (still excluded^)
echo .venv/
echo venv/
echo ENV/
echo env/
echo.
echo # IDE files
echo .vscode/
echo .idea/
echo *.swp
echo *.swo
echo *~
echo.
echo # OS files
echo .DS_Store
echo Thumbs.db
echo desktop.ini
echo.
echo # Temporary files
echo *.tmp
echo *.temp
echo .pytest_cache/
echo .coverage
echo htmlcov/
echo.
echo # Git LFS objects
echo .git/lfs/
echo.
echo # Only exclude organize_files.py
echo organize_files.py
) > .gitignore

echo    [OK] .gitignore updated

REM Step 5: Add .gitattributes for LFS
echo.
echo 5. Creating .gitattributes for Git LFS...
git add .gitattributes
echo    [OK] .gitattributes added

REM Step 6: Stage all changes
echo.
echo 6. Staging all files...
git add .
echo    [OK] All files staged

REM Step 7: Show status
echo.
echo 7. Git Status (first 20 files):
git status --short | more

REM Step 8: Commit
echo.
echo 8. Committing changes...
set /p COMMIT_MSG="Enter commit message (or press Enter for default): "

if "%COMMIT_MSG%"=="" (
    set COMMIT_MSG=Add all files including models, datasets, and outputs using Git LFS
)

git commit -m "%COMMIT_MSG%"

REM Step 9: Push everything
echo.
echo 9. Pushing everything to GitHub...
echo    WARNING: This may take a while for large files...
echo.

git push origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===============================================================
    echo             SUCCESS! Everything uploaded to GitHub!
    echo ===============================================================
    echo.
    echo Upload Summary:
    echo    * All code files
    echo    * All model checkpoints (*.pth, *.pt^)
    echo    * All images and visualizations
    echo    * All datasets (if included^)
    echo    * All logs and outputs
    echo.
    echo Large files are stored in Git LFS
    echo Repository size on GitHub: ~1.5 GB
    echo.
    echo View at: https://github.com/BenWandera/SW-AI-42
    echo.
    echo Note: Users will need Git LFS to clone:
    echo    git lfs install
    echo    git clone https://github.com/BenWandera/SW-AI-42.git
) else (
    echo.
    echo [X] Push failed!
    echo.
    echo Common issues:
    echo    1. Git LFS quota exceeded (GitHub free: 1GB/month^)
    echo    2. File too large (LFS limit: 2GB per file^)
    echo    3. Network issues
    echo.
    echo Solutions:
    echo    * Upgrade to GitHub Pro for more LFS bandwidth
    echo    * Split very large files
    echo    * Use alternative storage (HuggingFace Hub, S3^)
)

echo.
echo ===============================================================
pause
