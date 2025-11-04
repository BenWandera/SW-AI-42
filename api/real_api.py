"""
Real FastAPI Backend with MobileViT Model Integration
Loads trained MobileViT model for actual waste classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import MobileViTImageProcessor
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom model loader
from model_loader import load_trained_model

# Import GNN verifier
from gnn_loader import load_gnn_verifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
IS_PRODUCTION = ENVIRONMENT == 'production'

# Path to user stats file
USER_STATS_FILE = os.getenv('USER_STATS_FILE', 'data/user_stats.json') if IS_PRODUCTION else 'user_stats.json'

# Initialize FastAPI app
app = FastAPI(
    title="Waste Management API (Real MobileViT)",
    description="AI-powered waste classification with MobileViT + GNN",
    version="2.0.0-production"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Waste categories (matching your training data)
CLASS_NAMES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal',
    'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]

# Map class names to API category IDs
CLASS_TO_CATEGORY = {
    'Cardboard': {'id': 'paper', 'name': 'Paper & Cardboard', 'points': 12},
    'Food Organics': {'id': 'organic', 'name': 'Organic/Food Waste', 'points': 10},
    'Glass': {'id': 'glass', 'name': 'Glass', 'points': 18},
    'Metal': {'id': 'metal', 'name': 'Metal', 'points': 20},
    'Miscellaneous Trash': {'id': 'miscellaneous', 'name': 'Miscellaneous', 'points': 5},
    'Paper': {'id': 'paper', 'name': 'Paper & Cardboard', 'points': 12},
    'Plastic': {'id': 'plastic', 'name': 'Plastic', 'points': 15},
    'Textile Trash': {'id': 'miscellaneous', 'name': 'Miscellaneous', 'points': 5},
    'Vegetation': {'id': 'vegetation', 'name': 'Vegetation', 'points': 8}
}

# Global model variables
mobilevit_model = None
mobilevit_processor = None
gnn_verifier = None
device = None

# User stats storage with persistence
def load_user_stats():
    """Load user stats from JSON file"""
    if os.path.exists(USER_STATS_FILE):
        try:
            with open(USER_STATS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user stats: {e}")
    
    # Return default user if file doesn't exist
    return {
        "default_user": {
            "user_id": "default_user",
            "name": "Ben Wandera",
            "total_points": 0,
            "current_streak": 0,
            "items_classified": 0,
            "last_classification_date": None,
            "classification_history": [],
            "category_stats": {},
            "achievements": []
        }
    }

def save_user_stats():
    """Save user stats to JSON file"""
    try:
        with open(USER_STATS_FILE, 'w') as f:
            json.dump(user_stats, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving user stats: {e}")

# Load user stats from file on startup
user_stats = load_user_stats()
logger.info(f"📊 Loaded user stats: {user_stats.get('default_user', {}).get('total_points', 0)} points")

def calculate_tier(points: int) -> str:
    """Calculate user tier based on total points"""
    if points >= 5000:
        return "Platinum"
    elif points >= 2000:
        return "Gold"
    elif points >= 500:
        return "Silver"
    else:
        return "Bronze"

def update_user_streak(user_id: str):
    """Update user streak based on daily classifications"""
    from datetime import date
    
    user = user_stats.get(user_id)
    if not user:
        return
    
    today = date.today()
    last_date = user.get("last_classification_date")
    
    if last_date is None:
        # First classification
        user["current_streak"] = 1
        user["last_classification_date"] = today.isoformat()
    elif last_date == today.isoformat():
        # Already classified today, no change
        pass
    elif last_date == (today - timedelta(days=1)).isoformat():
        # Consecutive day, increment streak
        user["current_streak"] += 1
        user["last_classification_date"] = today.isoformat()
    else:
        # Streak broken, reset to 1
        user["current_streak"] = 1
        user["last_classification_date"] = today.isoformat()

def load_mobilevit_model(model_path: str = None):
    """Load trained MobileViT model"""
    global mobilevit_model, mobilevit_processor, device, CLASS_NAMES, gnn_verifier
    
    # Auto-detect model path (works both locally and on Render)
    if model_path is None:
        # Try multiple possible locations
        possible_paths = [
            "best_mobilevit_waste_model.pth",  # Docker/Render location
            "../best_mobilevit_waste_model.pth",  # Local dev location
            "/app/best_mobilevit_waste_model.pth"  # Absolute Docker path
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"✅ Found model at: {model_path}")
                break
        
        if model_path is None:
            logger.error("❌ Model file not found in any expected location!")
            logger.error(f"   Searched: {possible_paths}")
            return False
    
    try:
        logger.info(f"🔄 Loading MobileViT model from: {model_path}")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"📱 Using device: {device}")
        
        # Load model using custom loader
        mobilevit_model, saved_classes = load_trained_model(model_path, device=device)
        
        # Update CLASS_NAMES from saved model
        if saved_classes:
            CLASS_NAMES.clear()
            CLASS_NAMES.extend(saved_classes)
            logger.info(f"📊 Loaded {len(CLASS_NAMES)} classes")
        
        # Load processor
        mobilevit_processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        
        # Load GNN verifier
        logger.info("🧠 Loading GNN verification system...")
        gnn_verifier = load_gnn_verifier(device=str(device))
        
        logger.info("✅ MobileViT model loaded successfully!")
        return True
        
    except FileNotFoundError:
        logger.error(f"❌ Model file not found: {model_path}")
        logger.warning("⚠️ Running in MOCK mode - using random classification!")
        return False
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        logger.warning("⚠️ Running in MOCK mode - using random classification!")
        return False


def classify_image_mobilevit(image: Image.Image):
    """Classify image using MobileViT model"""
    global mobilevit_model, mobilevit_processor, device
    
    if mobilevit_model is None:
        # Fallback to mock classification
        logger.warning("⚠️ Using mock classification (model not loaded)")
        return mock_classify()
    
    try:
        # Preprocess image
        inputs = mobilevit_processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        # Run inference
        with torch.no_grad():
            logits = mobilevit_model(pixel_values)
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            probs_numpy = probs.cpu().numpy()[0]
            
            # Get prediction
            predicted_idx = torch.argmax(logits, dim=-1).item()
            confidence = float(probs_numpy[predicted_idx])
            predicted_class = CLASS_NAMES[predicted_idx]
        
        logger.info(f"🔍 MobileViT: {predicted_class} ({confidence:.2%})")
        
        # Get all predictions
        all_predictions = {
            CLASS_NAMES[i]: float(probs_numpy[i])
            for i in range(len(CLASS_NAMES))
        }
        
        return {
            'class_name': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'predicted_idx': predicted_idx
        }
        
    except Exception as e:
        logger.error(f"❌ Classification error: {e}")
        return mock_classify()


def mock_classify():
    """Fallback mock classification"""
    import random
    predicted_class = random.choice(CLASS_NAMES)
    confidence = random.uniform(0.75, 0.95)
    
    all_predictions = {
        name: random.uniform(0.01, 0.15) if name != predicted_class else confidence
        for name in CLASS_NAMES
    }
    
    return {
        'class_name': predicted_class,
        'confidence': confidence,
        'all_predictions': all_predictions,
        'predicted_idx': CLASS_NAMES.index(predicted_class)
    }


def apply_gnn_verification(mobilevit_result: dict):
    """
    Apply GNN verification to MobileViT prediction
    Uses real GNN reasoning model for safety-critical validation
    """
    global gnn_verifier
    
    if gnn_verifier is None:
        # Fallback to simple mock
        import random
        gnn_agrees = random.random() > 0.15
        gnn_confidence = mobilevit_result['confidence'] * (1.05 if gnn_agrees else 0.85)
        
        return {
            'agrees': gnn_agrees,
            'recommendation': mobilevit_result['class_name'],
            'confidence': gnn_confidence,
            'adjustment': gnn_confidence - mobilevit_result['confidence']
        }
    
    try:
        # Use real GNN verification
        verification = gnn_verifier.verify_classification(
            mobilevit_class=mobilevit_result['class_name'],
            mobilevit_confidence=mobilevit_result['confidence'],
            mobilevit_probs=mobilevit_result['all_predictions']
        )
        
        logger.info(f"🧠 GNN: {verification['recommendation']} ({verification['adjusted_confidence']:.2%}) - "
                   f"{'✓ Agrees' if verification['agrees'] else '✗ Caution'}")
        
        return {
            'agrees': verification['agrees'],
            'recommendation': verification['recommendation'],
            'confidence': verification['adjusted_confidence'],
            'adjustment': verification['confidence_delta'],
            'risk_level': verification['risk_level'],
            'is_safety_critical': verification.get('is_safety_critical', False),
            'reasoning': verification.get('reasoning', ''),
            'status': verification.get('status', 'VERIFIED')
        }
        
    except Exception as e:
        logger.error(f"❌ GNN verification error: {e}")
        # Fallback to simple verification
        return {
            'agrees': True,
            'recommendation': mobilevit_result['class_name'],
            'confidence': mobilevit_result['confidence'],
            'adjustment': 0.0,
            'reasoning': 'GNN verification unavailable, using MobileViT prediction'
        }


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("\n" + "="*60)
    print("🚀 Starting Waste Management API (Real MobileViT)")
    print("="*60)
    print("📱 Flutter app connection: http://192.168.100.152:8000")
    print("🔍 Loading MobileViT model...")
    
    model_loaded = load_mobilevit_model()  # Auto-detect model path
    
    if model_loaded:
        print("✅ MobileViT model ready!")
        print("🧠 GNN verification: " + ("Active (Rule-based)" if gnn_verifier else "Mock mode"))
    else:
        print("⚠️  Running in MOCK mode")
    
    print("📡 API ready!")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Waste Management API",
        "version": "2.0.0-REAL-MODEL",
        "model": "MobileViT + GNN",
        "status": "online",
        "model_loaded": mobilevit_model is not None,
        "classes": len(CLASS_NAMES),
        "mode": "REAL CLASSIFICATION" if mobilevit_model is not None else "MOCK MODE"
    }


@app.get("/api/debug/files")
async def debug_files():
    """Debug endpoint to check what files exist in the container"""
    import os
    files_in_app = os.listdir("/app") if os.path.exists("/app") else []
    files_in_current = os.listdir(".")
    
    model_checks = {
        "best_mobilevit_waste_model.pth": os.path.exists("best_mobilevit_waste_model.pth"),
        "/app/best_mobilevit_waste_model.pth": os.path.exists("/app/best_mobilevit_waste_model.pth"),
        "../best_mobilevit_waste_model.pth": os.path.exists("../best_mobilevit_waste_model.pth"),
    }
    
    return {
        "current_directory": os.getcwd(),
        "files_in_app_dir": files_in_app,
        "files_in_current_dir": files_in_current,
        "model_file_checks": model_checks,
        "model_loaded": mobilevit_model is not None
    }


@app.get("/api/debug/users")
async def debug_users():
    """Debug endpoint to see all users"""
    return {
        "total_users": len(user_stats),
        "users": [
            {
                "user_id": uid,
                "name": udata.get("name", "Unknown"),
                "points": udata.get("total_points", 0),
                "items": udata.get("items_classified", 0)
            }
            for uid, udata in user_stats.items()
        ]
    }


@app.post("/api/classify")
async def classify_waste(image: UploadFile = File(...), user_id: str = Form("default_user")):
    """
    Classify waste image using MobileViT + GNN
    """
    try:
        logger.info(f"📸 Classification request from user: {user_id}")
        
        # Validate image
        if image.content_type and not (image.content_type.startswith('image/') or 
                                       image.content_type == 'application/octet-stream'):
            logger.warning(f"⚠️ Received content type: {image.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")
        
        logger.info(f"📸 Receiving image: {image.filename}, type: {image.content_type}")
        
        # Read image
        image_data = await image.read()
        logger.info(f"✅ Image size: {len(image_data)} bytes")
        
        # Open image with PIL
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        logger.info(f"📐 Image dimensions: {pil_image.size}")
        
        # Step 1: MobileViT Classification
        mobilevit_result = classify_image_mobilevit(pil_image)
        
        # Step 2: GNN Verification
        gnn_result = apply_gnn_verification(mobilevit_result)
        
        # Determine final classification
        final_class_name = gnn_result['recommendation']
        final_confidence = gnn_result['confidence']
        
        # Map to API category
        category_info = CLASS_TO_CATEGORY.get(final_class_name, {
            'id': 'miscellaneous',
            'name': 'Miscellaneous',
            'points': 5
        })
        
        # Prepare all predictions (mapped to API categories)
        all_predictions_mapped = {}
        for class_name, prob in mobilevit_result['all_predictions'].items():
            cat = CLASS_TO_CATEGORY.get(class_name, {'name': class_name})
            all_predictions_mapped[cat['name']] = prob
        
        logger.info(f"✅ Final: {final_class_name} ({final_confidence:.2%})")
        
        # Award points to the actual user
        if user_id not in user_stats:
            user_stats[user_id] = {
                "user_id": user_id,
                "name": "Ben Wandera",
                "total_points": 0,
                "current_streak": 0,
                "items_classified": 0,
                "last_classification_date": None,
                "classification_history": [],
                "category_stats": {},
                "achievements": []
            }
        
        # Calculate points based on confidence
        base_points = category_info['points']
        confidence_bonus = int(base_points * 0.5) if final_confidence > 0.85 else 0
        total_earned = base_points + confidence_bonus
        
        # Update user stats
        user_stats[user_id]["total_points"] += total_earned
        user_stats[user_id]["items_classified"] += 1
        update_user_streak(user_id)
        
        # Update category stats
        if "category_stats" not in user_stats[user_id]:
            user_stats[user_id]["category_stats"] = {}
        
        category_key = final_class_name.lower()
        if category_key not in user_stats[user_id]["category_stats"]:
            user_stats[user_id]["category_stats"][category_key] = 0
        user_stats[user_id]["category_stats"][category_key] += 1
        
        # Check and award achievements
        if "achievements" not in user_stats[user_id]:
            user_stats[user_id]["achievements"] = []
        
        achievements = user_stats[user_id]["achievements"]
        
        # First Scan achievement
        if "first_scan" not in achievements and user_stats[user_id]["items_classified"] == 1:
            achievements.append("first_scan")
            logger.info("🏆 Achievement unlocked: First Scan!")
        
        # Century Club achievement (100 items)
        if "century_club" not in achievements and user_stats[user_id]["items_classified"] >= 100:
            achievements.append("century_club")
            logger.info("🏆 Achievement unlocked: Century Club!")
        
        # Week Warrior achievement (7 day streak)
        if "week_warrior" not in achievements and user_stats[user_id]["current_streak"] >= 7:
            achievements.append("week_warrior")
            logger.info("🏆 Achievement unlocked: Week Warrior!")
        
        # Accuracy Master achievement (95% accuracy on 50+ items)
        if "accuracy_master" not in achievements and user_stats[user_id]["items_classified"] >= 50:
            # Calculate accuracy from history
            correct = sum(1 for c in user_stats[user_id]["classification_history"] if c.get("confidence", 0) > 0.95)
            accuracy = (correct / len(user_stats[user_id]["classification_history"])) * 100
            if accuracy >= 95:
                achievements.append("accuracy_master")
                logger.info("🏆 Achievement unlocked: Accuracy Master!")
        
        # Add to history
        classification_record = {
            "category": final_class_name,
            "confidence": final_confidence,
            "points_earned": total_earned,
            "timestamp": datetime.now().isoformat()
        }
        user_stats[user_id]["classification_history"].append(classification_record)
        
        # Keep only last 100 classifications
        if len(user_stats[user_id]["classification_history"]) > 100:
            user_stats[user_id]["classification_history"] = user_stats[user_id]["classification_history"][-100:]
        
        # Save stats to file
        save_user_stats()
        
        logger.info(f"💰 Awarded {total_earned} points to user (Total: {user_stats[user_id]['total_points']})")
        
        return {
            "category_id": category_info['id'],
            "category_name": category_info['name'],
            "confidence": round(final_confidence, 4),
            "mobilevit_confidence": round(mobilevit_result['confidence'], 4),
            "gnn_confidence": round(gnn_result['confidence'], 4),
            "gnn_recommendation": category_info['name'],
            "base_points": category_info['points'],
            "timestamp": datetime.now().isoformat(),
            "image_path": image.filename or "uploaded_image.jpg",
            "is_corrected": not gnn_result['agrees'],
            "all_predictions": {k: round(v, 4) for k, v in all_predictions_mapped.items()},
            "points_awarded": total_earned,  # NEW: Points awarded
            "user_total_points": user_stats[user_id]["total_points"],  # NEW: Total points
            "gnn_reasoning": {
                "agrees_with_mobilevit": gnn_result['agrees'],
                "confidence_adjustment": round(gnn_result['adjustment'], 4),
                "recommendation": gnn_result['recommendation'],
                "original_mobilevit": mobilevit_result['class_name']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


# Pydantic models for other endpoints
class IncentiveRequest(BaseModel):
    user_id: str
    category_id: str
    confidence: float
    is_corrected: bool = False


class IncentiveResult(BaseModel):
    total_points: int
    base_points: int
    confidence_bonus: int
    accuracy_multiplier: float
    streak_bonus: int
    tier_multiplier: float
    message: str
    achievements_unlocked: list = []
    has_new_achievements: bool = False


@app.post("/api/incentive/calculate", response_model=IncentiveResult)
async def calculate_incentive(request: IncentiveRequest):
    """Calculate incentive points (mock for now)"""
    
    # Base points from category
    CATEGORY_POINTS = {
        'plastic': 15, 'paper': 12, 'organic': 10, 'vegetation': 8,
        'glass': 18, 'metal': 20, 'electronic': 30, 'medical': 50,
        'miscellaneous': 5
    }
    
    base_points = CATEGORY_POINTS.get(request.category_id, 10)
    confidence_bonus = int(base_points * request.confidence * 0.2)
    
    total = base_points + confidence_bonus
    
    return IncentiveResult(
        total_points=total,
        base_points=base_points,
        confidence_bonus=confidence_bonus,
        accuracy_multiplier=1.0,
        streak_bonus=0,
        tier_multiplier=1.0,
        message=f"Great job! You earned {total} points!",
        achievements_unlocked=[],
        has_new_achievements=False
    )


@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    """Get user profile with real stats"""
    
    # If user doesn't exist, create a new user profile
    if user_id not in user_stats:
        logger.info(f"Creating new user profile for: {user_id}")
        user_stats[user_id] = {
            "user_id": user_id,
            "name": f"User {user_id[:8]}",  # Temporary name, will be updated by app
            "total_points": 0,
            "current_streak": 0,
            "items_classified": 0,
            "last_classification_date": None,
            "classification_history": [],
            "category_stats": {},
            "achievements": [],
            "challenges": {},
            "joined_date": datetime.now().isoformat()
        }
        save_user_stats()
    
    user = user_stats[user_id]
    
    # Calculate tier
    tier = calculate_tier(user["total_points"])
    
    # Calculate level (500 points per level)
    level = max(1, (user["total_points"] // 500) + 1)
    
    # Get category stats from user data (already tracked)
    category_stats = user.get("category_stats", {})
    
    # Get achievements from user data (already tracked)
    achievements = user.get("achievements", [])
    
    # Calculate accuracy
    correct_classifications = user["items_classified"]  # Assume all are correct for now
    accuracy = (correct_classifications / user["items_classified"] * 100) if user["items_classified"] > 0 else 0
    
    return {
        "user_id": user["user_id"],
        "name": user["name"],
        "email": f"{user['name'].lower().replace(' ', '.')}@ecowaste.com",
        "neighborhood": "Kampala Central",
        "division": "Central Division",
        "total_points": user["total_points"],
        "current_streak": user["current_streak"],
        "tier": tier,
        "level": level,
        "items_classified": user["items_classified"],
        "correct_classifications": correct_classifications,
        "category_stats": category_stats,
        "achievements": achievements,
        "last_classification_date": user.get("last_classification_date"),
        "joined_date": "2024-07-01T00:00:00Z",
        "recent_classifications": user["classification_history"][-10:] if user["classification_history"] else []
    }


@app.post("/api/users/{user_id}/update")
async def update_user_profile(
    user_id: str, 
    name: str = None,
    email: str = None,
    neighborhood: str = None,
    division: str = None
):
    """Update user profile information"""
    
    # Create user if doesn't exist
    if user_id not in user_stats:
        logger.info(f"Creating new user during update: {user_id}")
        user_stats[user_id] = {
            "user_id": user_id,
            "name": name or f"User {user_id[:8]}",
            "total_points": 0,
            "current_streak": 0,
            "items_classified": 0,
            "last_classification_date": None,
            "classification_history": [],
            "category_stats": {},
            "achievements": [],
            "challenges": {},
            "joined_date": datetime.now().isoformat()
        }
    
    # Update user fields if provided
    if name:
        user_stats[user_id]["name"] = name
    if email:
        user_stats[user_id]["email"] = email
    if neighborhood:
        user_stats[user_id]["neighborhood"] = neighborhood
    if division:
        user_stats[user_id]["division"] = division
    
    save_user_stats()
    logger.info(f"✏️ Updated user profile: {user_id} -> {user_stats[user_id]['name']}")
    
    return {
        "success": True,
        "message": "Profile updated successfully",
        "user_id": user_id,
        "name": user_stats[user_id]["name"]
    }


@app.get("/api/leaderboard/{period}")
async def get_leaderboard(period: str):
    """Get leaderboard with real user data"""
    # Get all users and sort by points
    all_users = list(user_stats.values())
    sorted_users = sorted(all_users, key=lambda x: x['total_points'], reverse=True)
    
    # Add mock users if we don't have enough real users
    mock_users = [
        {"user_id": "mock1", "name": "Eco Champion", "total_points": 450, "items_classified": 45},
        {"user_id": "mock2", "name": "Green Warrior", "total_points": 380, "items_classified": 38},
        {"user_id": "mock3", "name": "Recycle Pro", "total_points": 320, "items_classified": 32},
        {"user_id": "mock4", "name": "Earth Saver", "total_points": 285, "items_classified": 29},
        {"user_id": "mock5", "name": "Eco Warrior", "total_points": 250, "items_classified": 25},
        {"user_id": "mock6", "name": "Nature Lover", "total_points": 215, "items_classified": 22},
        {"user_id": "mock7", "name": "Clean Hero", "total_points": 180, "items_classified": 18},
        {"user_id": "mock8", "name": "Waste Master", "total_points": 165, "items_classified": 17},
    ]
    
    # Combine real and mock users
    if len(sorted_users) < 8:
        # Add mock users that have fewer points than real users
        min_real_points = sorted_users[-1]['total_points'] if sorted_users else 0
        for mock in mock_users:
            if mock['total_points'] < min_real_points or not sorted_users:
                sorted_users.append(mock)
    
    # Re-sort after adding mock users
    sorted_users = sorted(sorted_users, key=lambda x: x['total_points'], reverse=True)
    
    # Limit to top 20
    sorted_users = sorted_users[:20]
    
    # Build leaderboard response
    leaderboard = []
    for rank, user in enumerate(sorted_users, start=1):
        leaderboard.append({
            "rank": rank,
            "user_id": user['user_id'],
            "name": user['name'],
            "points": user['total_points'],
            "items": user['items_classified']
        })
    
    return {
        "period": period,
        "users": leaderboard
    }


@app.get("/api/categories")
async def get_categories():
    """Get all waste categories"""
    categories = list(set(CLASS_TO_CATEGORY.values()))
    return {"categories": categories}


# ==================== REWARDS SYSTEM ====================

# Available rewards catalog
REWARDS_CATALOG = [
    {
        'id': 'coffee_voucher',
        'name': 'Coffee Voucher',
        'description': 'Enjoy a free coffee at partner cafes',
        'points': 500,
        'category': 'Food & Drink',
        'available': 15,
        'icon': 'local_cafe',
        'color': 'brown',
    },
    {
        'id': 'eco_tote_bag',
        'name': 'Eco Tote Bag',
        'description': 'Reusable shopping bag made from recycled materials',
        'points': 800,
        'category': 'Merchandise',
        'available': 8,
        'icon': 'shopping_bag',
        'color': 'green',
    },
    {
        'id': 'plant_tree',
        'name': 'Plant a Tree',
        'description': 'We\'ll plant a tree in your name',
        'points': 1000,
        'category': 'Environmental',
        'available': 100,
        'icon': 'park',
        'color': 'green_700',
    },
    {
        'id': 'gift_card_5',
        'name': '$5 Gift Card',
        'description': 'Amazon or local store gift card',
        'points': 1200,
        'category': 'Gift Cards',
        'available': 20,
        'icon': 'card_giftcard',
        'color': 'orange',
    },
    {
        'id': 'water_bottle',
        'name': 'Eco Water Bottle',
        'description': 'Stainless steel reusable water bottle',
        'points': 1500,
        'category': 'Merchandise',
        'available': 5,
        'icon': 'water_drop',
        'color': 'blue',
    },
    {
        'id': 'donation_10',
        'name': '$10 Donation',
        'description': 'Donate to environmental charity',
        'points': 2000,
        'category': 'Charity',
        'available': 50,
        'icon': 'favorite',
        'color': 'red',
    },
    {
        'id': 'premium_membership',
        'name': 'Premium Membership',
        'description': '3 months of premium features',
        'points': 3000,
        'category': 'Subscription',
        'available': 10,
        'icon': 'workspace_premium',
        'color': 'purple',
    },
]


@app.get("/api/rewards")
async def get_rewards():
    """Get available rewards catalog"""
    return {
        "rewards": REWARDS_CATALOG,
        "total_count": len(REWARDS_CATALOG)
    }


@app.post("/api/rewards/redeem")
async def redeem_reward(user_id: str, reward_id: str):
    """Redeem a reward with user points"""
    # Get user stats
    if user_id not in user_stats:
        user_id = "default_user"
    
    if user_id not in user_stats:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = user_stats[user_id]
    
    # Find the reward
    reward = next((r for r in REWARDS_CATALOG if r['id'] == reward_id), None)
    if not reward:
        raise HTTPException(status_code=404, detail="Reward not found")
    
    # Check if user has enough points
    if user["total_points"] < reward["points"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient points. Need {reward['points']}, have {user['total_points']}"
        )
    
    # Check if reward is available
    if reward["available"] <= 0:
        raise HTTPException(status_code=400, detail="Reward is out of stock")
    
    # Deduct points
    user["total_points"] -= reward["points"]
    
    # Add to redemption history
    if "redemption_history" not in user:
        user["redemption_history"] = []
    
    redemption = {
        "reward_id": reward_id,
        "reward_name": reward["name"],
        "points_spent": reward["points"],
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "redemption_code": f"ECO-{user_id[:4].upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    }
    
    user["redemption_history"].append(redemption)
    
    # Decrease reward availability (in production, this would be in a database)
    reward["available"] -= 1
    
    # Save user stats
    save_user_stats()
    
    logger.info(f"🎁 Reward redeemed: {user_id} -> {reward['name']} ({reward['points']} points)")
    
    return {
        "success": True,
        "message": f"Successfully redeemed {reward['name']}",
        "redemption": redemption,
        "remaining_points": user["total_points"]
    }


@app.get("/api/rewards/history/{user_id}")
async def get_redemption_history(user_id: str):
    """Get user's redemption history"""
    # Use default_user if not found
    if user_id not in user_stats:
        user_id = "default_user"
    
    if user_id not in user_stats:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = user_stats[user_id]
    history = user.get("redemption_history", [])
    
    # Sort by timestamp, most recent first
    history_sorted = sorted(
        history, 
        key=lambda x: x.get("timestamp", ""), 
        reverse=True
    )
    
    return {
        "user_id": user_id,
        "redemptions": history_sorted,
        "total_redemptions": len(history_sorted),
        "total_points_spent": sum(r.get("points_spent", 0) for r in history_sorted)
    }


# Challenges System
CHALLENGES = [
    {
        "id": "weekly_warrior",
        "title": "Weekly Warrior",
        "description": "Classify 20 items this week",
        "target": 20,
        "reward": 100,
        "type": "weekly",
        "icon": "calendar_today",
        "color": "#4CAF50",
    },
    {
        "id": "perfect_classifier",
        "title": "Perfect Classifier",
        "description": "Get 10 correct classifications in a row",
        "target": 10,
        "reward": 150,
        "type": "achievement",
        "icon": "stars",
        "color": "#FFD700",
    },
    {
        "id": "early_bird",
        "title": "Early Bird",
        "description": "Classify before 9 AM for 5 days",
        "target": 5,
        "reward": 75,
        "type": "habit",
        "icon": "wb_sunny",
        "color": "#FF9800",
    },
    {
        "id": "eco_champion",
        "title": "Eco Champion",
        "description": "Recycle 50 items this month",
        "target": 50,
        "reward": 250,
        "type": "monthly",
        "icon": "emoji_events",
        "color": "#9C27B0",
    },
    {
        "id": "community_leader",
        "title": "Community Leader",
        "description": "Help 5 friends start recycling",
        "target": 5,
        "reward": 200,
        "type": "social",
        "icon": "people",
        "color": "#2196F3",
    },
    {
        "id": "streak_master",
        "title": "Streak Master",
        "description": "Maintain a 30-day streak",
        "target": 30,
        "reward": 500,
        "type": "streak",
        "icon": "local_fire_department",
        "color": "#FF5722",
    },
]

# Track user challenge progress (in-memory, would be DB in production)
user_challenges = {}

@app.get("/api/challenges")
async def get_challenges(user_id: str = "default_user"):
    """Get all available challenges with user progress"""
    
    # Initialize user challenges if not exists
    if user_id not in user_challenges:
        user_challenges[user_id] = {}
        for challenge in CHALLENGES:
            user_challenges[user_id][challenge["id"]] = {
                "progress": 0,
                "completed": False,
                "claimed": False
            }
    
    # Get user stats for calculating progress
    user = user_stats.get(user_id, user_stats.get("default_user", {}))
    
    # Calculate real progress based on user stats
    challenges_with_progress = []
    for challenge in CHALLENGES:
        challenge_data = challenge.copy()
        user_progress = user_challenges[user_id].get(challenge["id"], {
            "progress": 0,
            "completed": False,
            "claimed": False
        })
        
        # Auto-calculate progress based on challenge type
        if challenge["id"] == "weekly_warrior":
            # Count classifications this week (simplified - use recent classifications)
            progress = min(user.get("items_classified", 0), challenge["target"])
        elif challenge["id"] == "eco_champion":
            # Monthly recycling count - use total items classified
            progress = min(user.get("items_classified", 0), challenge["target"])
        elif challenge["id"] == "streak_master":
            # Current streak
            progress = user.get("current_streak", 0)
        elif challenge["id"] == "perfect_classifier":
            # Count high-confidence classifications (>95% confidence)
            recent = user.get("recent_classifications", [])
            consecutive = 0
            for item in reversed(recent):  # Check most recent first
                if item.get("confidence", 0) > 0.95:
                    consecutive += 1
                else:
                    break
            progress = min(consecutive, challenge["target"])
        elif challenge["id"] == "early_bird":
            # Count morning classifications (simplified - use items/10 as proxy)
            progress = min(user.get("items_classified", 0) // 4, challenge["target"])
        elif challenge["id"] == "community_leader":
            # Use stored progress (requires social feature)
            progress = user_progress.get("progress", 0)
        else:
            # Use stored progress
            progress = user_progress.get("progress", 0)
        
        # Update progress
        user_challenges[user_id][challenge["id"]]["progress"] = progress
        if progress >= challenge["target"]:
            user_challenges[user_id][challenge["id"]]["completed"] = True
        
        challenge_data["progress"] = progress
        challenge_data["completed"] = user_challenges[user_id][challenge["id"]]["completed"]
        challenge_data["claimed"] = user_challenges[user_id][challenge["id"]].get("claimed", False)
        
        # Count actual participants (users who have started this challenge)
        participants = sum(1 for uid in user_challenges if challenge["id"] in user_challenges[uid])
        challenge_data["participants"] = participants
        
        challenges_with_progress.append(challenge_data)
    
    return {
        "challenges": challenges_with_progress,
        "total_challenges": len(challenges_with_progress),
        "completed_count": sum(1 for c in challenges_with_progress if c["completed"])
    }


@app.post("/api/challenges/{challenge_id}/claim")
async def claim_challenge_reward(challenge_id: str, user_id: str = "default_user"):
    """Claim reward for completed challenge"""
    
    # Find challenge
    challenge = next((c for c in CHALLENGES if c["id"] == challenge_id), None)
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    # Check if user has completed and not claimed
    if user_id not in user_challenges or challenge_id not in user_challenges[user_id]:
        raise HTTPException(status_code=400, detail="Challenge not started")
    
    user_progress = user_challenges[user_id][challenge_id]
    
    if not user_progress.get("completed", False):
        raise HTTPException(status_code=400, detail="Challenge not completed")
    
    if user_progress.get("claimed", False):
        raise HTTPException(status_code=400, detail="Reward already claimed")
    
    # Award points
    if user_id not in user_stats:
        user_id = "default_user"
    
    user_stats[user_id]["total_points"] += challenge["reward"]
    user_challenges[user_id][challenge_id]["claimed"] = True
    
    return {
        "success": True,
        "challenge_id": challenge_id,
        "reward": challenge["reward"],
        "new_total_points": user_stats[user_id]["total_points"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
