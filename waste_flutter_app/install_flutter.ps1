# Flutter Auto-Installer for Windows
# Run this script in PowerShell as Administrator

Write-Host "============================================" -ForegroundColor Green
Write-Host "   Flutter SDK Auto-Installer for Windows" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[WARNING] This script should be run as Administrator for best results." -ForegroundColor Yellow
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne 'y') {
        exit
    }
}

# Configuration
$flutterVersion = "3.16.0-stable"
$flutterUrl = "https://storage.googleapis.com/flutter_infra_release/releases/stable/windows/flutter_windows_$flutterVersion.zip"
$flutterZip = "$env:TEMP\flutter_sdk.zip"
$installPath = "C:\src"
$flutterPath = "$installPath\flutter"

Write-Host "[1/6] Checking existing Flutter installation..." -ForegroundColor Cyan
$existingFlutter = Get-Command flutter -ErrorAction SilentlyContinue
if ($existingFlutter) {
    Write-Host "[OK] Flutter is already installed at: $($existingFlutter.Source)" -ForegroundColor Green
    flutter --version
    Write-Host ""
    $reinstall = Read-Host "Do you want to reinstall/update Flutter? (y/n)"
    if ($reinstall -ne 'y') {
        Write-Host "Skipping installation. Proceeding to setup..." -ForegroundColor Yellow
        exit
    }
}

Write-Host "[2/6] Creating installation directory..." -ForegroundColor Cyan
try {
    if (-not (Test-Path $installPath)) {
        New-Item -ItemType Directory -Path $installPath -Force | Out-Null
        Write-Host "[OK] Created directory: $installPath" -ForegroundColor Green
    } else {
        Write-Host "[OK] Directory already exists: $installPath" -ForegroundColor Green
    }
} catch {
    Write-Host "[ERROR] Failed to create directory: $_" -ForegroundColor Red
    exit 1
}

Write-Host "[3/6] Downloading Flutter SDK (this may take a few minutes)..." -ForegroundColor Cyan
Write-Host "URL: $flutterUrl" -ForegroundColor Gray
try {
    # Download with progress
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $flutterUrl -OutFile $flutterZip -UseBasicParsing
    $ProgressPreference = 'Continue'
    
    $fileSize = (Get-Item $flutterZip).Length / 1MB
    Write-Host "[OK] Downloaded $([math]::Round($fileSize, 2)) MB" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Download failed: $_" -ForegroundColor Red
    Write-Host "Please check your internet connection and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host "[4/6] Extracting Flutter SDK..." -ForegroundColor Cyan
try {
    # Remove old installation if exists
    if (Test-Path $flutterPath) {
        Write-Host "Removing old Flutter installation..." -ForegroundColor Yellow
        Remove-Item -Path $flutterPath -Recurse -Force
    }
    
    # Extract
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($flutterZip, $installPath)
    Write-Host "[OK] Extracted to: $flutterPath" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Extraction failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "[5/6] Adding Flutter to PATH..." -ForegroundColor Cyan
try {
    $flutterBin = "$flutterPath\bin"
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    
    if ($currentPath -like "*$flutterBin*") {
        Write-Host "[OK] Flutter is already in PATH" -ForegroundColor Green
    } else {
        $newPath = "$currentPath;$flutterBin"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "[OK] Added Flutter to PATH" -ForegroundColor Green
        Write-Host "Path: $flutterBin" -ForegroundColor Gray
    }
    
    # Update current session PATH
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
} catch {
    Write-Host "[ERROR] Failed to add to PATH: $_" -ForegroundColor Red
    Write-Host "You may need to add it manually: $flutterBin" -ForegroundColor Yellow
}

Write-Host "[6/6] Cleaning up..." -ForegroundColor Cyan
try {
    Remove-Item -Path $flutterZip -Force
    Write-Host "[OK] Removed temporary files" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Could not remove temp file: $flutterZip" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "   Flutter Installation Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

# Verify installation
Write-Host "Verifying Flutter installation..." -ForegroundColor Cyan
Write-Host ""

try {
    & "$flutterBin\flutter" --version
    Write-Host ""
    Write-Host "[SUCCESS] Flutter is working!" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Could not verify Flutter in current session" -ForegroundColor Yellow
    Write-Host "Please close and reopen PowerShell/Terminal" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Next Steps" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Close and reopen your terminal" -ForegroundColor White
Write-Host ""
Write-Host "2. Run Flutter Doctor:" -ForegroundColor White
Write-Host "   flutter doctor" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Accept Android licenses:" -ForegroundColor White
Write-Host "   flutter doctor --android-licenses" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Navigate to your app and install dependencies:" -ForegroundColor White
Write-Host "   cd c:\Users\Z-BOOK\OneDrive\Documents\DATASETS\waste_flutter_app" -ForegroundColor Gray
Write-Host "   flutter pub get" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Run your app:" -ForegroundColor White
Write-Host "   flutter run" -ForegroundColor Gray
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to run flutter doctor
$runDoctor = Read-Host "Do you want to run 'flutter doctor' now? (y/n)"
if ($runDoctor -eq 'y') {
    Write-Host ""
    Write-Host "Running flutter doctor..." -ForegroundColor Cyan
    Write-Host ""
    & "$flutterBin\flutter" doctor -v
}

Write-Host ""
Write-Host "Installation complete! Press any key to exit..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
