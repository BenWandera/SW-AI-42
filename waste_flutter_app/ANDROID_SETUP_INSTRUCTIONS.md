# Android Studio Installation Guide

## Step 1: Run the Installer Script

### Open PowerShell as Administrator:
1. Press `Windows + X`
2. Click **"Windows PowerShell (Admin)"** or **"Terminal (Admin)"**
3. Click **"Yes"** when prompted

### Run these commands:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
cd "c:\Users\Z-BOOK\OneDrive\Documents\DATASETS\waste_flutter_app"
.\install_android_studio.ps1
```

---

## Step 2: Complete Android Studio Setup Wizard

After the installer downloads and starts:

### 1. Welcome Screen
- Click **"Next"**

### 2. Choose Components
- ✅ **Android Studio**
- ✅ **Android Virtual Device** (IMPORTANT!)
- Click **"Next"**

### 3. Installation Location
- Keep default: `C:\Program Files\Android\Android Studio`
- Click **"Next"**

### 4. Choose Start Menu Folder
- Keep default
- Click **"Install"**

### 5. Wait for Installation
- Takes 5-10 minutes
- Click **"Next"** when done
- Click **"Finish"**

---

## Step 3: First-Time Setup Wizard

Android Studio will now open:

### 1. Import Settings
- Select **"Do not import settings"**
- Click **"OK"**

### 2. Welcome
- Click **"Next"**

### 3. Install Type
- Select **"Standard"**
- Click **"Next"**

### 4. Select UI Theme
- Choose **Dark** or **Light** (your preference)
- Click **"Next"**

### 5. Verify Settings
- You should see:
  - ✅ Android SDK
  - ✅ Android SDK Platform
  - ✅ Performance (Intel HAXM or Hypervisor)
  - ✅ Android Virtual Device
- Click **"Next"**

### 6. License Agreement
- Click **"Accept"** for all licenses
- Click **"Finish"**

### 7. Downloading Components
- Takes 10-20 minutes (downloads ~2 GB)
- Wait for "Finish" button to appear
- Click **"Finish"**

---

## Step 4: Accept Android Licenses

Back in your Git Bash terminal, run:

```bash
flutter doctor --android-licenses
```

- Type **"y"** for each license prompt
- Press Enter

---

## Step 5: Verify Installation

```bash
flutter doctor -v
```

You should now see:
- ✅ Android toolchain
- ✅ Android Studio

---

## Step 6: Create Virtual Device (Emulator)

### Option A: Through Android Studio
1. Open Android Studio
2. Click **"More Actions"** → **"Virtual Device Manager"**
3. Click **"Create Device"**
4. Select **"Pixel 5"** or **"Pixel 6"**
5. Click **"Next"**
6. Select **"Tiramisu"** (API 33) or latest
7. Click **"Next"** → **"Finish"**
8. Click **▶️ Play** button to start emulator

### Option B: Through Command Line
```bash
flutter emulators --launch <emulator_id>
```

---

## Step 7: Run Your App!

Once emulator is running:

```bash
cd waste_flutter_app
flutter run
```

---

## Troubleshooting

### Issue: "Android SDK not found"
```bash
flutter config --android-sdk "C:\Users\Z-BOOK\AppData\Local\Android\Sdk"
```

### Issue: "cmdline-tools not found"
1. Open Android Studio
2. **Tools** → **SDK Manager**
3. Click **"SDK Tools"** tab
4. ✅ Check **"Android SDK Command-line Tools"**
5. Click **"Apply"** → **"OK"**

### Issue: Emulator won't start
- Check BIOS virtualization is enabled
- Or use physical phone instead

---

## Need Help?

Type in chat:
- **"stuck"** - If installation hangs
- **"error"** - If you see error messages
- **"done"** - When setup completes successfully
