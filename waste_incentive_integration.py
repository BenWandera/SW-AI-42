"""
Integration Demo: Waste Classification + Incentivization System
Demonstrates how to combine MobileViT LVM classification with the incentive engine
"""

import torch
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json
import os
from PIL import Image

# Import your existing models
from models import Base, User, SortingEvent, WasteType, UserType, MembershipTier
from incentive_engine import IncentiveEngine, RewardManager


class WasteClassificationIncentiveSystem:
    """
    Integrated system combining waste classification with incentivization
    """
    
    def __init__(self, 
                 mobilevit_model_path: str,
                 anthropic_api_key: str,
                 database_url: str = "sqlite:///waste_management.db"):
        """
        Initialize the integrated system
        
        Args:
            mobilevit_model_path: Path to trained MobileViT model
            anthropic_api_key: Anthropic API key for incentive calculations
            database_url: Database connection string
        """
        
        # Setup database
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db = Session()
        
        # Load MobileViT model (placeholder - use your actual model loading)
        print(f"📱 Loading MobileViT model from {mobilevit_model_path}")
        self.mobilevit_model = self._load_mobilevit_model(mobilevit_model_path)
        
        # Initialize incentive system
        print(f"🎯 Initializing incentive engine with Anthropic API")
        self.incentive_engine = IncentiveEngine(anthropic_api_key, self.db)
        self.reward_manager = RewardManager(self.db)
        
        # Waste type mapping
        self.class_names = [
            "Cardboard", "Food Organics", "Glass", "Metal",
            "Miscellaneous Trash", "Paper", "Plastic", 
            "Textile Trash", "Vegetation"
        ]
        
        self.waste_type_mapping = {
            "Cardboard": WasteType.PAPER_CARDBOARD,
            "Food Organics": WasteType.ORGANIC,
            "Glass": WasteType.GLASS,
            "Metal": WasteType.METAL,
            "Miscellaneous Trash": WasteType.MIXED,
            "Paper": WasteType.PAPER_CARDBOARD,
            "Plastic": WasteType.PLASTIC,
            "Textile Trash": WasteType.MIXED,
            "Vegetation": WasteType.ORGANIC
        }
        
        print(f"✅ Waste Classification + Incentive System initialized")
    
    def _load_mobilevit_model(self, model_path: str):
        """Load the trained MobileViT model"""
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                print(f"✅ Model loaded successfully")
                # Return a placeholder - implement actual model loading
                return {"status": "loaded", "path": model_path}
            else:
                print(f"⚠️  Model file not found: {model_path}")
                return {"status": "not_found"}
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return {"status": "error", "message": str(e)}
    
    def classify_and_incentivize(self, 
                                image_path: str, 
                                user_id: int,
                                quantity_kg: float = 1.0) -> dict:
        """
        Complete workflow: classify waste and calculate incentives
        
        Args:
            image_path: Path to waste image
            user_id: ID of user sorting the waste
            quantity_kg: Weight of waste in kg
            
        Returns:
            Dictionary with classification results and incentive details
        """
        
        print(f"\n🔍 Processing waste classification and incentivization")
        print(f"   Image: {image_path}")
        print(f"   User ID: {user_id}")
        print(f"   Quantity: {quantity_kg} kg")
        
        # Get user from database
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"error": "User not found"}
        
        # Step 1: Classify waste using MobileViT
        classification_result = self._classify_waste(image_path)
        
        # Step 2: Validate classification (simulated for demo)
        validation_result = self._validate_classification(classification_result)
        
        # Step 3: Calculate incentives using Anthropic API
        incentive_result = self._calculate_incentives(
            user, classification_result, validation_result, quantity_kg
        )
        
        # Step 4: Update user data and create sorting event
        sorting_event = self._create_sorting_event(
            user, classification_result, validation_result, 
            incentive_result, quantity_kg, image_path
        )
        
        # Step 5: Update achievements and tier
        achievements = self.incentive_engine.check_achievements(user)
        tier_upgraded = self.incentive_engine.update_user_tier(user)
        streak_bonus = self.incentive_engine.update_user_streak(user)
        
        # Commit changes
        self.db.commit()
        
        # Step 6: Prepare response
        response = self._prepare_response(
            user, classification_result, validation_result,
            incentive_result, sorting_event, achievements, 
            tier_upgraded, streak_bonus
        )
        
        print(f"✅ Processing complete")
        return response
    
    def _classify_waste(self, image_path: str) -> dict:
        """
        Classify waste using MobileViT model
        """
        print(f"🤖 Classifying waste image...")
        
        # Simulated classification result (replace with actual MobileViT inference)
        # This would normally load the image, preprocess it, and run inference
        
        # Simulate realistic results based on your model's performance
        np.random.seed(42)  # For reproducible demo
        
        predicted_class_idx = np.random.randint(0, len(self.class_names))
        predicted_class_name = self.class_names[predicted_class_idx]
        confidence = np.random.uniform(0.75, 0.95)  # Simulate high confidence
        
        # Simulate secondary predictions
        probs = np.random.dirichlet(np.ones(len(self.class_names)))
        probs[predicted_class_idx] = confidence
        probs = probs / probs.sum()  # Normalize
        
        top_3_indices = np.argsort(probs)[::-1][:3]
        secondary_predictions = [
            {
                "class_name": self.class_names[idx],
                "confidence": float(probs[idx])
            }
            for idx in top_3_indices[1:]  # Skip the primary prediction
        ]
        
        result = {
            "predicted_class": predicted_class_name,
            "confidence": float(confidence),
            "waste_type": self.waste_type_mapping.get(predicted_class_name, WasteType.MIXED),
            "secondary_predictions": secondary_predictions,
            "all_probabilities": {
                self.class_names[i]: float(probs[i]) 
                for i in range(len(self.class_names))
            }
        }
        
        print(f"   Predicted: {predicted_class_name} ({confidence:.2%} confidence)")
        return result
    
    def _validate_classification(self, classification_result: dict) -> dict:
        """
        Validate the classification result (simulated)
        """
        print(f"✅ Validating classification...")
        
        # Simulate validation logic
        confidence = classification_result["confidence"]
        
        # Determine if correctly sorted based on confidence
        is_correctly_sorted = confidence > 0.8
        
        # Simulate contamination and safety checks
        is_contaminated = np.random.random() < 0.1  # 10% chance of contamination
        is_hazardous_unsafe = np.random.random() < 0.05  # 5% chance of unsafe disposal
        
        # Graph reasoning simulation (normally from your GNN model)
        graph_reasoning = {
            "spatial_correlation": np.random.uniform(0.7, 0.9),
            "temporal_pattern": np.random.uniform(0.6, 0.8),
            "neighborhood_consistency": np.random.uniform(0.8, 0.95),
            "volume_prediction": np.random.uniform(0.5, 2.0)
        }
        
        result = {
            "is_correctly_sorted": is_correctly_sorted,
            "is_contaminated": is_contaminated,
            "is_hazardous_unsafe": is_hazardous_unsafe,
            "graph_reasoning": graph_reasoning,
            "validation_score": confidence * 0.9 + 0.1  # Slight adjustment
        }
        
        print(f"   Correctly sorted: {is_correctly_sorted}")
        print(f"   Contaminated: {is_contaminated}")
        print(f"   Unsafe: {is_hazardous_unsafe}")
        
        return result
    
    def _calculate_incentives(self, user: User, classification: dict, 
                            validation: dict, quantity_kg: float) -> dict:
        """
        Calculate incentives using the IncentiveEngine
        """
        print(f"💰 Calculating incentives...")
        
        # Use the IncentiveEngine to calculate points
        points, multiplier, bonus_reason, feedback = self.incentive_engine.calculate_points(
            user=user,
            waste_type=classification["waste_type"],
            quantity_kg=quantity_kg,
            is_correctly_sorted=validation["is_correctly_sorted"],
            is_contaminated=validation["is_contaminated"],
            is_hazardous_unsafe=validation["is_hazardous_unsafe"],
            confidence_score=classification["confidence"],
            graph_reasoning_result=validation["graph_reasoning"],
            secondary_waste_types=[
                pred["class_name"] for pred in classification["secondary_predictions"]
            ]
        )
        
        result = {
            "points_earned": points,
            "multiplier": multiplier,
            "bonus_reason": bonus_reason,
            "feedback_message": feedback
        }
        
        print(f"   Points earned: {points}")
        print(f"   Multiplier: {multiplier}x")
        print(f"   Reason: {bonus_reason}")
        
        return result
    
    def _create_sorting_event(self, user: User, classification: dict,
                            validation: dict, incentive: dict,
                            quantity_kg: float, image_path: str) -> SortingEvent:
        """
        Create a sorting event record in the database
        """
        print(f"💾 Creating sorting event record...")
        
        # Create sorting event
        sorting_event = SortingEvent(
            user_id=user.id,
            waste_type=classification["waste_type"],
            secondary_waste_types=[
                pred["class_name"] for pred in classification["secondary_predictions"]
            ],
            quantity_kg=quantity_kg,
            is_correctly_sorted=validation["is_correctly_sorted"],
            is_contaminated=validation["is_contaminated"],
            is_hazardous_unsafe=validation["is_hazardous_unsafe"],
            confidence_score=classification["confidence"],
            points_earned=incentive["points_earned"],
            bonus_multiplier=incentive["multiplier"],
            bonus_reason=incentive["bonus_reason"],
            image_path=image_path,
            sorting_date=datetime.utcnow()
        )
        
        # Update user points
        user.total_points += incentive["points_earned"]
        user.available_points += max(0, incentive["points_earned"])  # Only positive points are available for redemption
        
        self.db.add(sorting_event)
        
        print(f"   Event ID: {sorting_event.id}")
        return sorting_event
    
    def _prepare_response(self, user: User, classification: dict,
                        validation: dict, incentive: dict,
                        sorting_event: SortingEvent, achievements: list,
                        tier_upgraded: bool, streak_bonus: int) -> dict:
        """
        Prepare the final response with all information
        """
        response = {
            "success": True,
            "classification": {
                "predicted_class": classification["predicted_class"],
                "confidence": classification["confidence"],
                "waste_type": classification["waste_type"].value,
                "secondary_predictions": classification["secondary_predictions"]
            },
            "validation": {
                "correctly_sorted": validation["is_correctly_sorted"],
                "contaminated": validation["is_contaminated"],
                "unsafe_disposal": validation["is_hazardous_unsafe"],
                "validation_score": validation["validation_score"]
            },
            "incentives": {
                "points_earned": incentive["points_earned"],
                "multiplier": incentive["multiplier"],
                "bonus_reason": incentive["bonus_reason"],
                "feedback_message": incentive["feedback_message"]
            },
            "user_status": {
                "total_points": user.total_points,
                "available_points": user.available_points,
                "membership_tier": user.membership_tier.value,
                "current_streak": user.current_streak_days,
                "tier_upgraded": tier_upgraded
            },
            "achievements": [
                {
                    "id": achievement.id,
                    "name": achievement.name,
                    "description": achievement.description,
                    "bonus_points": achievement.bonus_points
                }
                for achievement in achievements
            ],
            "streak_bonus": streak_bonus,
            "sorting_event_id": sorting_event.id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return response
    
    def get_user_dashboard(self, user_id: int) -> dict:
        """
        Get comprehensive user dashboard information
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"error": "User not found"}
        
        # Get recent sorting events
        recent_events = self.db.query(SortingEvent).filter(
            SortingEvent.user_id == user_id
        ).order_by(SortingEvent.sorting_date.desc()).limit(10).all()
        
        # Get available rewards
        available_rewards = self.reward_manager.get_available_rewards(user)
        
        # Get achievements
        user_achievements = [
            {
                "name": ua.achievement.name,
                "description": ua.achievement.description,
                "earned_at": ua.earned_at.isoformat()
            }
            for ua in user.achievements
        ]
        
        dashboard = {
            "user": {
                "id": user.id,
                "name": user.name,
                "total_points": user.total_points,
                "available_points": user.available_points,
                "membership_tier": user.membership_tier.value,
                "current_streak": user.current_streak_days,
                "longest_streak": user.longest_streak_days
            },
            "recent_events": [
                {
                    "date": event.sorting_date.isoformat(),
                    "waste_type": event.waste_type.value,
                    "points_earned": event.points_earned,
                    "correctly_sorted": event.is_correctly_sorted
                }
                for event in recent_events
            ],
            "available_rewards": available_rewards,
            "achievements": user_achievements,
            "statistics": {
                "total_events": len(user.sorting_events),
                "accuracy_rate": sum(1 for e in user.sorting_events if e.is_correctly_sorted) / max(1, len(user.sorting_events))
            }
        }
        
        return dashboard


def create_demo_users(system: WasteClassificationIncentiveSystem):
    """Create demo users for testing"""
    
    demo_users = [
        {
            "name": "Alice Nakato",
            "email": "alice@kampala.ug",
            "phone": "+256701234567",
            "user_type": UserType.INDIVIDUAL,
            "neighborhood": "Nakawa"
        },
        {
            "name": "Bob Ssemwogerere", 
            "email": "bob@kampala.ug",
            "phone": "+256702345678",
            "user_type": UserType.HOUSEHOLD,
            "neighborhood": "Central"
        },
        {
            "name": "Central Market",
            "email": "market@central.ug",
            "phone": "+256703456789",
            "user_type": UserType.BUSINESS,
            "neighborhood": "Central"
        }
    ]
    
    created_users = []
    for user_data in demo_users:
        # Check if user already exists
        existing = system.db.query(User).filter(User.email == user_data["email"]).first()
        if not existing:
            user = User(**user_data)
            system.db.add(user)
            system.db.commit()
            created_users.append(user)
            print(f"✅ Created user: {user.name} (ID: {user.id})")
        else:
            created_users.append(existing)
            print(f"👤 User exists: {existing.name} (ID: {existing.id})")
    
    return created_users


def main():
    """
    Demonstration of the integrated waste classification and incentivization system
    """
    print("🎯 Waste Classification + Incentivization System Demo")
    print("=" * 60)
    
    # Initialize system (replace with your actual paths and API key)
    system = WasteClassificationIncentiveSystem(
        mobilevit_model_path="best_mobilevit_waste_model.pth",
        anthropic_api_key="your-anthropic-api-key-here",  # Replace with actual key
        database_url="sqlite:///waste_management_demo.db"
    )
    
    # Create demo users
    print(f"\n👥 Creating demo users...")
    users = create_demo_users(system)
    
    # Simulate waste sorting events
    print(f"\n🗂️  Simulating waste sorting events...")
    
    demo_images = [
        "plastic_bottle.jpg",
        "food_waste.jpg", 
        "cardboard_box.jpg",
        "glass_jar.jpg",
        "metal_can.jpg"
    ]
    
    results = []
    for i, image in enumerate(demo_images):
        user = users[i % len(users)]  # Rotate through users
        
        print(f"\n--- Sorting Event {i+1} ---")
        result = system.classify_and_incentivize(
            image_path=image,
            user_id=user.id,
            quantity_kg=np.random.uniform(0.1, 2.0)
        )
        results.append(result)
        
        # Print summary
        if result.get("success"):
            print(f"✅ SUCCESS")
            print(f"   Classified as: {result['classification']['predicted_class']}")
            print(f"   Points earned: {result['incentives']['points_earned']}")
            print(f"   User total: {result['user_status']['total_points']} points")
            print(f"   Tier: {result['user_status']['membership_tier']}")
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
    
    # Show user dashboards
    print(f"\n📊 User Dashboards:")
    print("=" * 40)
    
    for user in users:
        dashboard = system.get_user_dashboard(user.id)
        print(f"\n👤 {dashboard['user']['name']}:")
        print(f"   Total Points: {dashboard['user']['total_points']}")
        print(f"   Available Points: {dashboard['user']['available_points']}")
        print(f"   Tier: {dashboard['user']['membership_tier']}")
        print(f"   Streak: {dashboard['user']['current_streak']} days")
        print(f"   Total Events: {dashboard['statistics']['total_events']}")
        print(f"   Accuracy: {dashboard['statistics']['accuracy_rate']:.1%}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📊 Summary:")
    print(f"   • {len(users)} users created")
    print(f"   • {len(results)} sorting events processed")
    print(f"   • Integrated MobileViT + Anthropic API + Database")
    print(f"   • Real-time incentive calculation with AI reasoning")


if __name__ == "__main__":
    main()