
## ✅ APK Built Successfully!

**APK Location (Choose One):**

📱 **Easy Access (Recommended):**
```
C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\waste_flutter_app\EcoWaste-AI.apk
```

📁 **Original Build Location:**
```
C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\waste_flutter_app\build\app\outputs\flutter-apk\app-release.apk
```

**APK Details:**
- ✅ Size: **30 MB** (includes SF Pro fonts)
- ✅ Font: San Francisco Pro Display
- ✅ Features: Real stats, Leaderboard, Camera, AI Classification

**To find the file:**
1. Open File Explorer (Windows Key + E)
2. Copy and paste this path in the address bar:
   ```
   C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\waste_flutter_app
   ```
3. Look for **EcoWaste-AI.apk** (30 MB file)

## Installation Steps

### Method 1: USB Cable (Recommended)

1. **Enable Developer Options on your phone:**
   - Go to **Settings** → **About Phone**
   - Tap **Build Number** 7 times
   - You'll see "You are now a developer!"

2. **Enable USB Debugging:**
   - Go to **Settings** → **Developer Options**
   - Enable **USB Debugging**
   - Enable **Install via USB** (if available)

3. **Connect phone to computer:**
   - Use a USB cable
   - On your phone, select **File Transfer** or **MTP** mode
   - Allow USB debugging when prompted

4. **Install APK using ADB:**
   ```bash
   adb install build/app/outputs/flutter-apk/app-release.apk
   ```

### Method 2: File Transfer (Easy)

1. **Copy APK to your phone:**
   - Connect phone via USB
   - Copy `app-release.apk` to your phone's **Downloads** folder
   - OR email it to yourself and download on phone
   - OR use Google Drive/Dropbox to transfer

2. **Install on phone:**
   - Open **Files** or **My Files** app on your phone
   - Navigate to **Downloads**
   - Tap on `app-release.apk`
   - Tap **Install** (you may need to allow installing from unknown sources)
   - Tap **Open** when installation completes

### Method 3: Cloud Transfer

1. **Upload APK to Google Drive:**
   - Upload `app-release.apk` to Google Drive
   - Open Google Drive on your phone
   - Download the APK
   - Tap to install

## ⚠️ Important: API Connection

Since you'll be using your phone, you need to update the API URL:

1. **Find your computer's IP address:**
   - Windows: Open Command Prompt and run `ipconfig`
   - Look for **IPv4 Address** (e.g., 192.168.1.100)

2. **Make sure API is running:**
   ```bash
   cd api
   python test_api.py
   ```
   API should be at `http://localhost:8000`

3. **Update Flutter app** (before building APK):
   - Edit `lib/services/api_service.dart`
   - Change the baseUrl to use your computer's IP:
   ```dart
   static String get baseUrl {
     return 'http://YOUR_IP_HERE:8000/api';  // e.g., http://192.168.1.100:8000/api
   }
   ```

4. **Rebuild APK** with the new IP address

## 🔥 Using the App

Once installed:

1. Open **EcoWaste AI** app
2. Tap the **green camera button (➕)**
3. Select **"Take Photo"** or **"Choose from Gallery"**
4. Take a picture of waste
5. Watch the AI classify it!
6. See your points earned!

## Testing Connection

Make sure:
- ✅ Your phone and computer are on the **same WiFi network**
- ✅ API server is running on your computer
- ✅ Windows Firewall allows port 8000 (or temporarily disable)
- ✅ You used your computer's IP address in the app

Test by opening browser on your phone and visiting:
```
http://YOUR_IP:8000
```

You should see: `{"status":"online","service":"Waste Management API",...}`

## 🎉 Ready!

Your phone now has a fully functional waste classification app with:
- ✅ Real camera support
- ✅ AI classification (MobileViT + GNN)
- ✅ Points and incentives
- ✅ Profile, History, Leaderboard, Rewards

Enjoy! 🌱
