"""
Download model file from GitHub using Git LFS if not present
This runs during container startup on Render
"""
import os
import sys
import urllib.request

MODEL_FILE = "best_mobilevit_waste_model.pth"
# Direct download URL for your model from GitHub LFS
GITHUB_LFS_URL = "https://media.githubusercontent.com/media/BenWandera/SW-AI-42/main/api/best_mobilevit_waste_model.pth"

def download_model():
    """Download model if it doesn't exist or is just a Git LFS pointer"""
    
    if os.path.exists(MODEL_FILE):
        # Check if it's a Git LFS pointer file (very small, ~130 bytes)
        file_size = os.path.getsize(MODEL_FILE)
        if file_size > 1000000:  # If > 1MB, it's the real file
            print(f"✅ Model file already exists ({file_size:,} bytes)")
            return True
        else:
            print(f"⚠️  Model file is a Git LFS pointer ({file_size} bytes), downloading actual file...")
            os.remove(MODEL_FILE)
    
    print(f"📥 Downloading model from GitHub LFS...")
    print(f"   URL: {GITHUB_LFS_URL}")
    
    try:
        urllib.request.urlretrieve(GITHUB_LFS_URL, MODEL_FILE)
        file_size = os.path.getsize(MODEL_FILE)
        print(f"✅ Model downloaded successfully ({file_size:,} bytes)")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
