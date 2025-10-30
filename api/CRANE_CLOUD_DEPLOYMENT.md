# 🚀 EcoWaste AI - Crane Cloud Deployment Guide

Complete step-by-step guide to deploy your EcoWaste AI backend to Crane Cloud.

---

## 📋 Prerequisites

1. ✅ GitHub account
2. ✅ Crane Cloud account (sign up at https://cranecloud.io)
3. ✅ Your API code ready (you have this!)

---

## 🎯 Part 1: Prepare Your GitHub Repository

### Step 1: Push API Code to GitHub

```bash
cd C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\api

# Initialize git (if not already done)
git init

# Add files
git add Dockerfile .dockerignore requirements.txt crane.yml real_api.py model_loader.py gnn_loader.py models.py .env.production

# Commit
git commit -m "Prepare for Crane Cloud deployment"

# Add your GitHub remote (replace with your repo URL)
git remote add origin https://github.com/BenWandera/ecowaste-ai-backend.git

# Push to GitHub
git push -u origin main
```

### Step 2: Verify Files on GitHub

Make sure these files are in your repository:
- ✅ `Dockerfile`
- ✅ `.dockerignore`
- ✅ `requirements.txt`
- ✅ `crane.yml`
- ✅ `real_api.py`
- ✅ `model_loader.py`
- ✅ `gnn_loader.py`
- ✅ `models.py`

---

## ☁️ Part 2: Deploy to Crane Cloud

### Step 1: Sign Up / Log In to Crane Cloud

1. Go to **https://cranecloud.io**
2. Click **"Sign Up"** or **"Log In"**
3. Complete registration (verify email if needed)

### Step 2: Create a New Project

1. Click **"Create Project"**
2. Project Name: `ecowaste-ai`
3. Description: `AI-powered waste classification app`
4. Click **"Create"**

### Step 3: Create a New App

1. Inside your project, click **"Create App"**
2. Select **"Deploy from GitHub"**
3. Connect your GitHub account (authorize Crane Cloud)
4. Select repository: `ecowaste-ai-backend`
5. Branch: `main`

### Step 4: Configure Deployment Settings

**Build Settings:**
- Build Method: **Docker**
- Dockerfile Path: `Dockerfile`

**Resource Allocation:**
- Memory: **2 GB** (required for ML models)
- CPU: **1 Core**
- Instances: **1**

**Port:**
- Port: **8000**

**Environment Variables:**
Add these in the Crane Cloud dashboard:

| Variable | Value |
|----------|-------|
| `ENVIRONMENT` | `production` |
| `PYTHONUNBUFFERED` | `1` |
| `HF_HOME` | `/app/hf_cache` |
| `TRANSFORMERS_CACHE` | `/app/hf_cache` |

### Step 5: Add Persistent Storage (Important!)

Crane Cloud will provide persistent volumes for:
1. **Model Cache**: `/app/hf_cache` (5 GB)
   - Stores downloaded AI models
   - Prevents re-downloading on restart

2. **User Data**: `/app/data` (1 GB)
   - Stores user stats (user_stats.json)
   - Persists across deployments

### Step 6: Deploy!

1. Click **"Deploy"**
2. Wait for build (first time takes 5-10 minutes)
   - Downloads Python dependencies
   - Downloads ML models
   - Builds Docker image

3. Monitor logs during deployment:
   - Green checkmark = Success ✅
   - Red X = Check logs for errors ❌

### Step 7: Get Your Production URL

After successful deployment, Crane Cloud gives you a URL like:
```
https://ecowaste-ai-xxxxxxxx.cranecloud.io
```

**Save this URL!** You'll need it for the Flutter app.

---

## 🧪 Part 3: Test Your Deployment

### Test 1: Health Check
```bash
curl https://ecowaste-ai-xxxxxxxx.cranecloud.io/
```

Expected response:
```json
{
  "service": "Waste Management API",
  "version": "2.0.0-production",
  "status": "online",
  "model_loaded": true
}
```

### Test 2: Classification API
```bash
# Upload an image (use Postman or curl)
curl -X POST "https://ecowaste-ai-xxxxxxxx.cranecloud.io/api/classify" \
  -F "file=@your_waste_image.jpg" \
  -F "user_id=test_user"
```

---

## 📱 Part 4: Update Flutter App

Update your Flutter app to use the production URL:

**File: `lib/services/api_service.dart`**

```dart
class ApiService {
  // Change from local IP to production URL
  static const String PRODUCTION_URL = 'https://ecowaste-ai-xxxxxxxx.cranecloud.io';
  static const String LOCAL_URL = 'http://192.168.100.152:8000';
  
  // Use production URL for real deployments
  static const bool USE_PRODUCTION = true; // Set to true for release
  
  static String get baseUrl {
    if (USE_PRODUCTION) {
      return '$PRODUCTION_URL/api';
    }
    
    if (Platform.isAndroid) {
      return 'http://$LOCAL_URL:8000/api';
    } else {
      return 'http://localhost:8000/api';
    }
  }
}
```

**Rebuild your APK:**
```bash
cd waste_flutter_app
flutter build apk --release
```

---

## 💰 Crane Cloud Pricing (as of 2024)

### Free Tier
- ✅ Good for testing
- ✅ Limited resources
- ✅ May sleep after inactivity

### Student/Startup Plan
- ✅ Free or heavily discounted
- ✅ 2GB RAM + 1 CPU (perfect for your app!)
- ✅ Recommended for EcoWaste AI

### Apply for Student Discount:
Email Crane Cloud support mentioning you're a Ugandan student/developer.

---

## 🔧 Troubleshooting

### Build Failed
**Problem:** Docker build errors
**Solution:** Check Dockerfile syntax, verify requirements.txt

### Model Download Timeout
**Problem:** First deployment takes too long
**Solution:** 
- Increase timeout in Crane Cloud settings
- Models download once, then cached

### Out of Memory
**Problem:** App crashes with OOM error
**Solution:** 
- Increase memory to 3-4GB in Crane Cloud
- Reduce model batch size in code

### App Not Responding
**Problem:** Health check fails
**Solution:**
- Check logs in Crane Cloud dashboard
- Verify PORT environment variable
- Check firewall settings

---

## 📊 Monitoring & Logs

### View Logs
1. Go to Crane Cloud dashboard
2. Select your app
3. Click **"Logs"** tab
4. See real-time server logs

### Monitor Resources
- CPU usage
- Memory usage
- Request count
- Response times

---

## 🔄 Updating Your App

When you make code changes:

```bash
# Commit and push to GitHub
git add .
git commit -m "Update API features"
git push origin main
```

Crane Cloud auto-deploys when you push to GitHub!

---

## 🎉 Success Checklist

- [ ] Code pushed to GitHub
- [ ] Crane Cloud account created
- [ ] App deployed successfully
- [ ] Production URL obtained
- [ ] Health check passes
- [ ] Classification endpoint works
- [ ] Flutter app updated with production URL
- [ ] New APK built and tested
- [ ] App shared with users!

---

## 📞 Support

**Crane Cloud Support:**
- Email: support@cranecloud.io
- Docs: https://docs.cranecloud.io
- Slack: Crane Cloud Community

**Your App Issues:**
- Check logs first
- Verify environment variables
- Test locally before deploying

---

## 🚀 Next Steps

1. **Deploy to Crane Cloud** (follow steps above)
2. **Test with real users** in Uganda
3. **Monitor performance** and errors
4. **Scale up** if you get more users!
5. **Deploy to Google Play Store** for wider distribution

---

Good luck with your deployment! 🎯🇺🇬
