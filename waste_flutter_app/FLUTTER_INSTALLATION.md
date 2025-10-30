# 🚀 Flutter Installation Guide for Windows

## Prerequisites Installation

Before installing Flutter, you need these tools:

### 1. Install Git (Already Installed ✅)
You're using Git Bash, so Git is already installed!

### 2. Install Visual Studio Code (Recommended)
Download from: https://code.visualstudio.com/

**Required Extensions:**
- Flutter (by Dart Code)
- Dart (by Dart Code)

### 3. Install Android Studio
Download from: https://developer.android.com/studio

**During installation, make sure to install:**
- Android SDK
- Android SDK Platform
- Android Virtual Device (AVD)

## Flutter Installation Steps

### Option 1: Manual Installation (Recommended)

1. **Download Flutter SDK**
   - Visit: https://docs.flutter.dev/get-started/install/windows
   - Download Flutter SDK (stable channel)
   - Or direct link: https://storage.googleapis.com/flutter_infra_release/releases/stable/windows/flutter_windows_3.16.0-stable.zip

2. **Extract Flutter**
   ```bash
   # Extract to C:\src\flutter
   # Or any location WITHOUT spaces or special characters
   ```

3. **Add Flutter to PATH**
   
   **Method A: Using Windows Settings**
   - Press `Win + X` and select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "User variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\src\flutter\bin`
   - Click "OK" on all dialogs
   - **Restart Git Bash/Terminal**

   **Method B: Using Command Line (Admin PowerShell)**
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", "$env:Path;C:\src\flutter\bin", "User")
   ```

4. **Verify Installation**
   ```bash
   # Close and reopen Git Bash, then run:
   flutter --version
   flutter doctor
   ```

### Option 2: Quick Install Script

Run this in **PowerShell as Administrator**:

```powershell
# Download Flutter
$flutterUrl = "https://storage.googleapis.com/flutter_infra_release/releases/stable/windows/flutter_windows_3.16.0-stable.zip"
$flutterZip = "$env:TEMP\flutter.zip"
$flutterPath = "C:\src"

# Create directory
New-Item -ItemType Directory -Path $flutterPath -Force

# Download
Write-Host "Downloading Flutter SDK..."
Invoke-WebRequest -Uri $flutterUrl -OutFile $flutterZip

# Extract
Write-Host "Extracting Flutter..."
Expand-Archive -Path $flutterZip -DestinationPath $flutterPath -Force

# Add to PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*flutter\bin*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;C:\src\flutter\bin", "User")
    Write-Host "Flutter added to PATH"
}

# Cleanup
Remove-Item $flutterZip

Write-Host "Flutter installation complete!"
Write-Host "Please restart your terminal and run: flutter doctor"
```

## Post-Installation Setup

### 1. Run Flutter Doctor
```bash
flutter doctor
```

This will check for:
- ✅ Flutter SDK
- ✅ Android toolchain
- ✅ Chrome (for web development)
- ✅ Visual Studio Code
- ✅ Connected devices

### 2. Accept Android Licenses
```bash
flutter doctor --android-licenses
```
Press 'y' to accept all licenses.

### 3. Install Flutter Extensions in VS Code
1. Open VS Code
2. Press `Ctrl+Shift+X` (Extensions)
3. Search and install:
   - **Flutter**
   - **Dart**

### 4. Configure Android Studio
1. Open Android Studio
2. Go to: Tools → SDK Manager
3. Install:
   - Android SDK Platform (API 33 or higher)
   - Android SDK Build-Tools
   - Android Emulator
4. Create an AVD (Android Virtual Device):
   - Tools → Device Manager → Create Device
   - Choose Pixel 5 or similar
   - Select API level 33
   - Finish

## Verification Checklist

Run these commands to verify everything is set up:

```bash
# Check Flutter version
flutter --version

# Check system requirements
flutter doctor -v

# List available devices
flutter devices

# Create a test project
flutter create test_app
cd test_app
flutter run
```

## Expected Output of `flutter doctor`

```
Doctor summary (to see all details, run flutter doctor -v):
[✓] Flutter (Channel stable, 3.16.0, on Microsoft Windows)
[✓] Windows Version (Installed version of Windows is 10 or higher)
[✓] Android toolchain - develop for Android devices
[✓] Chrome - develop for the web
[✓] Visual Studio Code (version 1.85)
[✓] Connected device (2 available)
[✓] Network resources

• No issues found!
```

## Troubleshooting

### Issue: "flutter: command not found"
**Solution:**
1. Verify Flutter is extracted to `C:\src\flutter`
2. Check PATH includes `C:\src\flutter\bin`
3. **Restart your terminal/computer**
4. Run `echo $PATH` to verify

### Issue: "Android licenses not accepted"
**Solution:**
```bash
flutter doctor --android-licenses
# Press 'y' for each license
```

### Issue: "No devices found"
**Solution:**
1. **For Android Emulator:**
   ```bash
   # Start Android Studio → Device Manager → Start Emulator
   ```
2. **For Physical Device:**
   - Enable USB Debugging on phone
   - Connect via USB
   - Allow debugging on phone
   - Run `adb devices` to verify

### Issue: "cmdline-tools component is missing"
**Solution:**
1. Open Android Studio
2. Tools → SDK Manager
3. SDK Tools tab
4. Check "Android SDK Command-line Tools"
5. Click Apply

## Quick Alternative: Use Chocolatey

If you have Chocolatey package manager:

```bash
# In PowerShell (Admin)
choco install flutter
```

## After Flutter Installation

Once Flutter is installed, return to the waste_flutter_app directory:

```bash
cd c:/Users/Z-BOOK/OneDrive/Documents/DATASETS/waste_flutter_app
flutter pub get
flutter run
```

## Need Help?

- **Flutter Documentation**: https://docs.flutter.dev/
- **Flutter Discord**: https://discord.gg/flutter
- **Stack Overflow**: https://stackoverflow.com/questions/tagged/flutter

---

**Time Required**: 15-30 minutes (depending on download speed)

**Disk Space Required**: ~2.5 GB (Flutter SDK + Android Studio + Dependencies)

---

After installation, come back and we'll start coding! 🚀
