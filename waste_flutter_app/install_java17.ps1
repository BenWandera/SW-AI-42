# PowerShell script to download and install Java 17
Write-Host "Downloading Java 17 (Eclipse Temurin)..." -ForegroundColor Green

$downloadUrl = "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.9%2B9/OpenJDK17U-jdk_x64_windows_hotspot_17.0.9_9.msi"
$installerPath = "$env:TEMP\jdk17_installer.msi"

# Download Java 17
Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath

Write-Host "Installing Java 17..." -ForegroundColor Green
Write-Host "Please follow the installation wizard." -ForegroundColor Yellow
Write-Host "IMPORTANT: Note the installation path (usually C:\Program Files\Eclipse Adoptium\jdk-17.0.9.9-hotspot\)" -ForegroundColor Yellow

# Run the installer
Start-Process -FilePath "msiexec.exe" -ArgumentList "/i `"$installerPath`"" -Wait

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Note the Java 17 installation path from the installer"
Write-Host "2. Open android/gradle.properties in your Flutter project"
Write-Host "3. Add this line (replace with your actual path):"
Write-Host "   org.gradle.java.home=C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.9.9-hotspot" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then run: flutter clean && flutter run" -ForegroundColor Cyan
