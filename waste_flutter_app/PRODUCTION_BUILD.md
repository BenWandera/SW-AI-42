# 📱 Flutter App Production Build Guide

Quick guide to build and release your EcoWaste AI app with production backend.

---

## 🔧 Step 1: Update Production URL

After deploying to Crane Cloud, you'll get a URL like:
```
https://ecowaste-ai-xxxxxxxx.cranecloud.io
```

**Update this file:** `lib/services/api_service.dart`

```dart
// Line ~13: Update with your Crane Cloud URL
static const String PRODUCTION_URL = 'https://ecowaste-ai-xxxxxxxx.cranecloud.io';

// Line ~22: Enable production mode
static const bool USE_PRODUCTION = true; // 👈 Change to true!
```

---

## 📦 Step 2: Build Production APK

```bash
cd C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\waste_flutter_app

# Build release APK
flutter build apk --release

# APK will be at:
# build\app\outputs\flutter-apk\app-release.apk
```

---

## 🎯 Step 3: Test Production Build

1. Install APK on your phone
2. Open the app
3. Try classifying waste
4. Check if it connects to Crane Cloud (not local server)

### How to verify it's using production:
- Turn OFF your local API server
- App should still work (connecting to Crane Cloud!)
- If it fails, check URL in `api_service.dart`

---

## 🚀 Step 4: Distribute to Users

### Option A: Direct Distribution
```bash
# Copy APK to shareable location
cp build\app\outputs\flutter-apk\app-release.apk C:\Users\Z-BOOK\Downloads\EcoWaste-AI-Production.apk
```

Share via:
- Google Drive link
- WhatsApp
- Email
- USB transfer

### Option B: Google Play Store
1. Create Google Play Developer account ($25)
2. Create app listing
3. Upload APK
4. Fill app details
5. Submit for review
6. Wait 1-3 days for approval

---

## 🔄 Switching Between Development and Production

### For Local Testing:
```dart
static const bool USE_PRODUCTION = false; // Use local server
```
Then rebuild:
```bash
flutter build apk --release
```

### For Production Release:
```dart
static const bool USE_PRODUCTION = true; // Use Crane Cloud
```
Then rebuild:
```bash
flutter build apk --release
```

---

## ✅ Production Checklist

Before releasing to users:

- [ ] Backend deployed to Crane Cloud successfully
- [ ] Production URL updated in `api_service.dart`
- [ ] `USE_PRODUCTION = true` enabled
- [ ] APK built with `flutter build apk --release`
- [ ] Tested on real device
- [ ] Classification works (connects to Crane Cloud)
- [ ] Challenges load correctly
- [ ] Leaderboard shows data
- [ ] Rewards system works
- [ ] App tested without local server running

---

## 🐛 Troubleshooting

### App can't connect to server
**Check:**
1. Is `USE_PRODUCTION = true`?
2. Is production URL correct?
3. Is Crane Cloud API running? (check dashboard)
4. Internet connection on phone?

### App crashes on classification
**Check:**
1. Crane Cloud logs for errors
2. Model files downloaded correctly?
3. Enough memory allocated on Crane Cloud?

### Slow response times
**Possible causes:**
1. First request downloads models (60s)
2. Crane Cloud free tier may be slow
3. Consider upgrading plan if many users

---

## 📊 Monitoring Production

### Check Crane Cloud Dashboard:
- Request count
- Error rate
- Response times
- Memory usage

### User Feedback:
- Ask users to report issues
- Monitor classification accuracy
- Check which features are used most

---

## 🎉 You're Ready!

Your app is now production-ready with:
- ✅ Cloud-hosted backend (Crane Cloud)
- ✅ Real-time AI classification
- ✅ Challenges & rewards system
- ✅ Leaderboard with real users
- ✅ Scalable infrastructure

Share it with the world! 🌍🇺🇬
