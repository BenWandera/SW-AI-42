# 🚀 Flutter App Quick Start Guide

## ⚡ Quick Setup (5 minutes)

### 1. Install Flutter

**Windows:**
```bash
# Download Flutter SDK from https://flutter.dev/docs/get-started/install/windows
# Extract to C:\src\flutter
# Add to PATH: C:\src\flutter\bin

# Verify installation
flutter doctor
```

**macOS:**
```bash
# Install using Homebrew
brew install flutter

# Or download from https://flutter.dev
# Verify installation
flutter doctor
```

**Linux:**
```bash
# Download and extract Flutter
wget https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_3.16.0-stable.tar.xz
tar xf flutter_linux_3.16.0-stable.tar.xz

# Add to PATH
export PATH="$PATH:`pwd`/flutter/bin"

# Verify
flutter doctor
```

### 2. Install Dependencies

```bash
cd waste_flutter_app
flutter pub get
```

### 3. Configure Backend URL

Edit `lib/services/api_service.dart`:

```dart
// Change this line:
static const String baseUrl = 'http://localhost:8000/api';

// To your backend URL:
static const String baseUrl = 'http://YOUR_IP:8000/api';
// Example: http://192.168.1.100:8000/api
```

### 4. Run the App

```bash
# Connect your phone or start an emulator
flutter devices

# Run the app
flutter run
```

## 📱 Running on Physical Device

### Android Phone

1. **Enable Developer Options**
   - Settings → About Phone → Tap "Build Number" 7 times
   - Settings → Developer Options → Enable "USB Debugging"

2. **Connect via USB**
   ```bash
   # Check device
   flutter devices
   
   # Run app
   flutter run
   ```

3. **Or Connect via WiFi** (ADB Wireless)
   ```bash
   # Connect via USB first
   adb tcpip 5555
   adb connect YOUR_PHONE_IP:5555
   
   # Now disconnect USB and run
   flutter run
   ```

### iPhone

1. **Requirements**
   - macOS computer
   - Xcode installed
   - Apple Developer account (free tier works)

2. **Setup**
   ```bash
   # Open iOS project
   open ios/Runner.xcworkspace
   
   # In Xcode:
   # - Select your team
   # - Connect iPhone
   # - Trust developer on phone
   # - Run
   ```

## 🖥️ Running on Emulator/Simulator

### Android Emulator

```bash
# List available emulators
flutter emulators

# Start emulator
flutter emulators --launch Pixel_5_API_33

# Run app
flutter run
```

### iOS Simulator (macOS only)

```bash
# Start simulator
open -a Simulator

# Run app
flutter run
```

## 🐛 Common Issues & Fixes

### Issue: "No devices found"

**Solution:**
```bash
# Check USB debugging (Android)
adb devices

# Restart ADB
adb kill-server
adb start-server

# For iOS, check cable and trust computer on phone
```

### Issue: "Gradle build failed" (Android)

**Solution:**
```bash
# Clean project
cd android
./gradlew clean

# Or
flutter clean
flutter pub get
```

### Issue: "CocoaPods not installed" (iOS)

**Solution:**
```bash
sudo gem install cocoapods
cd ios
pod install
```

### Issue: "Camera permission denied"

**Solution:**
- Uninstall app
- Reinstall with: `flutter run`
- Grant permissions when prompted

## 🔧 Development Tips

### Hot Reload
```bash
# While app is running, press:
r  # Hot reload (fast)
R  # Hot restart (slower, resets state)
p  # Toggle performance overlay
q  # Quit
```

### Debug Tools
```bash
# Open DevTools
flutter pub global activate devtools
flutter pub global run devtools

# Run with DevTools
flutter run --devtools
```

### Check Performance
```bash
# Profile mode (for performance testing)
flutter run --profile

# Check for UI jank
flutter run --trace-skia
```

## 📦 Building APK

### Debug APK (for testing)
```bash
flutter build apk --debug
# Output: build/app/outputs/flutter-apk/app-debug.apk
```

### Release APK (for distribution)
```bash
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk

# Install on phone
adb install build/app/outputs/flutter-apk/app-release.apk
```

### App Bundle (for Google Play)
```bash
flutter build appbundle --release
# Output: build/app/outputs/bundle/release/app-release.aab
```

## 🎯 Testing Backend Connection

### Method 1: Mock API (No Backend Required)

If you don't have the backend ready yet, modify `api_service.dart`:

```dart
// In classifyImage method, return mock data:
Future<ClassificationResult> classifyImage(File imageFile) async {
  await Future.delayed(const Duration(seconds: 2)); // Simulate API call
  
  return ClassificationResult(
    categoryId: 'plastic',
    categoryName: 'Plastic',
    confidence: 0.88,
    mobilevitConfidence: 0.85,
    gnnConfidence: 0.88,
    isCorrected: false,
    imagePath: imageFile.path,
    timestamp: DateTime.now(),
    allPredictions: {
      'plastic': 0.88,
      'glass': 0.06,
      'metal': 0.04,
    },
  );
}
```

### Method 2: Test with Real Backend

```bash
# 1. Start your Python backend
cd /path/to/backend
python api_server.py  # Or your backend start command

# 2. Find your IP address
# Windows: ipconfig
# macOS/Linux: ifconfig or ip addr

# 3. Update baseUrl in api_service.dart
static const String baseUrl = 'http://192.168.1.XXX:8000/api';

# 4. Test connection
curl http://192.168.1.XXX:8000/api/health
```

## 🔍 Debugging

### View Logs

**In Terminal:**
```bash
flutter run --verbose
```

**In Code:**
```dart
import 'package:logger/logger.dart';

final logger = Logger();
logger.d('Debug message');
logger.i('Info message');
logger.w('Warning message');
logger.e('Error message');
```

### Inspect Widget Tree

```bash
# While app is running, press:
w  # Dump widget hierarchy
t  # Dump rendering tree
```

## 🎨 Customization

### Change App Name

**Android:** `android/app/src/main/AndroidManifest.xml`
```xml
<application android:label="EcoWaste AI">
```

**iOS:** `ios/Runner/Info.plist`
```xml
<key>CFBundleName</key>
<string>EcoWaste AI</string>
```

### Change App Icon

1. Put your icon in `assets/images/app_icon.png` (1024x1024)
2. Run:
```bash
flutter pub run flutter_launcher_icons
```

### Change Primary Color

Edit `lib/main.dart`:
```dart
colorScheme: ColorScheme.fromSeed(
  seedColor: const Color(0xFF4CAF50), // Change this color
),
```

## 📊 Performance Monitoring

```bash
# Check app size
flutter build apk --analyze-size

# Profile build times
flutter build apk --profile

# Check for unnecessary rebuilds
flutter run --trace-widget-builds
```

## 🎓 Learning Resources

- [Flutter Documentation](https://docs.flutter.dev/)
- [Dart Language Tour](https://dart.dev/guides/language/language-tour)
- [Flutter Widget Catalog](https://flutter.dev/docs/development/ui/widgets)
- [Flutter Cookbook](https://docs.flutter.dev/cookbook)
- [Material Design 3](https://m3.material.io/)

## 💬 Need Help?

1. Check Flutter doctor: `flutter doctor -v`
2. Search Flutter issues: https://github.com/flutter/flutter/issues
3. Ask on Stack Overflow: https://stackoverflow.com/questions/tagged/flutter
4. Join Flutter Discord: https://discord.gg/N7Yshp4

## ✅ Checklist Before First Run

- [ ] Flutter installed (`flutter doctor` shows all green)
- [ ] Android Studio or Xcode installed
- [ ] Device connected or emulator running
- [ ] `flutter pub get` completed successfully
- [ ] Backend URL configured (or using mock data)
- [ ] Camera permissions configured
- [ ] Internet permission added

## 🚀 Next Steps After Setup

1. ✅ Test camera functionality
2. ✅ Test image picker from gallery
3. ✅ Verify classification flow (even with mock data)
4. ✅ Check result screen display
5. ✅ Navigate through all bottom nav tabs
6. ✅ Test on different screen sizes
7. ✅ Try dark mode

---

**Happy Coding! 🎉** Let me know if you encounter any issues!
