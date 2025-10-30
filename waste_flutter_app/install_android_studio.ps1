# Android Studio Installer for Flutter Development
# Run this script in PowerShell as Administrator

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Android Studio Installation Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if already installed
$androidStudioPath = "C:\Program Files\Android\Android Studio\bin\studio64.exe"
if (Test-Path $androidStudioPath) {
    Write-Host "Android Studio is already installed at: $androidStudioPath" -ForegroundColor Green
    Write-Host "Please open Android Studio and install SDK components." -ForegroundColor Yellow
    exit 0
}

Write-Host "Android Studio is required for Android app development." -ForegroundColor Yellow
Write-Host ""
Write-Host "Installation Steps:" -ForegroundColor Cyan
Write-Host "1. Download Android Studio (Size: ~1.1 GB)" -ForegroundColor White
Write-Host "2. Install with default settings" -ForegroundColor White
Write-Host "3. Install Android SDK and tools" -ForegroundColor White
Write-Host "4. Accept Android licenses" -ForegroundColor White
Write-Host ""

$download = Read-Host "Do you want to download Android Studio now? (Y/N)"

if ($download -eq "Y" -or $download -eq "y") {
    $downloadUrl = "https://redirector.gvt1.com/edgedl/android/studio/install/2023.1.1.28/android-studio-2023.1.1.28-windows.exe"
    $installerPath = "$env:TEMP\android-studio-installer.exe"
    
    Write-Host ""
    Write-Host "Downloading Android Studio..." -ForegroundColor Cyan
    Write-Host "This may take 10-20 minutes depending on your connection..." -ForegroundColor Yellow
    
    try {
        # Download with progress
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing
        $ProgressPreference = 'Continue'
        
        Write-Host "✓ Download complete!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Starting installer..." -ForegroundColor Cyan
        Write-Host ""
        Write-Host "IMPORTANT: During installation:" -ForegroundColor Yellow
        Write-Host "  1. Accept all default settings" -ForegroundColor White
        Write-Host "  2. Check 'Android Virtual Device' option" -ForegroundColor White
        Write-Host "  3. Wait for SDK components to download" -ForegroundColor White
        Write-Host ""
        
        # Start installer
        Start-Process -FilePath $installerPath -Wait
        
        Write-Host ""
        Write-Host "✓ Installation started!" -ForegroundColor Green
        Write-Host ""
        Write-Host "After installation completes:" -ForegroundColor Cyan
        Write-Host "1. Open Android Studio" -ForegroundColor White
        Write-Host "2. Complete the setup wizard" -ForegroundColor White
        Write-Host "3. Install SDK components when prompted" -ForegroundColor White
        Write-Host "4. Run: flutter doctor --android-licenses" -ForegroundColor White
        Write-Host ""
        
    } catch {
        Write-Host "✗ Download failed: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please download manually from:" -ForegroundColor Yellow
        Write-Host "https://developer.android.com/studio" -ForegroundColor Cyan
    }
} else {
    Write-Host ""
    Write-Host "Manual Installation:" -ForegroundColor Cyan
    Write-Host "1. Visit: https://developer.android.com/studio" -ForegroundColor White
    Write-Host "2. Download Android Studio" -ForegroundColor White
    Write-Host "3. Run the installer" -ForegroundColor White
    Write-Host "4. Complete setup wizard" -ForegroundColor White
    Write-Host ""
}

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Next Steps After Installation:" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "1. flutter doctor --android-licenses" -ForegroundColor White
Write-Host "2. flutter doctor -v" -ForegroundColor White
Write-Host "3. flutter run" -ForegroundColor White
Write-Host ""
