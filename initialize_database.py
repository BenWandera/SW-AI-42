"""
Database Initialization Script for Waste Management Incentivization System
Sets up the database with initial achievements, rewards, and incentive rules
"""

import os
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from models import (
    Base, User, Achievement, IncentiveRule, RewardCatalog, 
    WasteType, UserType, MembershipTier, AchievementType, RewardType
)


def initialize_database(database_url: str = "sqlite:///waste_management.db"):
    """
    Initialize the database with initial data
    
    Args:
        database_url: Database connection string
    """
    print(f"🗄️  Initializing waste management database...")
    print(f"   Database: {database_url}")
    
    # Create engine and tables
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        # Clear existing data (for demo purposes)
        print(f"🧹 Clearing existing data...")
        db.execute(text("DELETE FROM user_achievements"))
        db.execute(text("DELETE FROM redemptions"))
        db.execute(text("DELETE FROM sorting_events"))
        db.execute(text("DELETE FROM notifications"))
        db.execute(text("DELETE FROM leaderboard_entries"))
        db.execute(text("DELETE FROM waste_streams"))
        db.execute(text("DELETE FROM reward_catalog"))
        db.execute(text("DELETE FROM incentive_rules"))
        db.execute(text("DELETE FROM achievements"))
        db.execute(text("DELETE FROM users"))
        
        # Initialize achievements
        print(f"🏆 Creating achievements...")
        achievements = create_achievements(db)
        
        # Initialize incentive rules
        print(f"📋 Creating incentive rules...")
        rules = create_incentive_rules(db)
        
        # Initialize reward catalog
        print(f"🎁 Creating reward catalog...")
        rewards = create_reward_catalog(db)
        
        # Commit all changes
        db.commit()
        
        print(f"✅ Database initialization complete!")
        print(f"   • {len(achievements)} achievements created")
        print(f"   • {len(rules)} incentive rules created")
        print(f"   • {len(rewards)} rewards created")
        
        return {
            "success": True,
            "achievements": len(achievements),
            "rules": len(rules),
            "rewards": len(rewards),
            "database_url": database_url
        }
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error initializing database: {e}")
        return {"success": False, "error": str(e)}
    
    finally:
        db.close()


def create_achievements(db) -> list:
    """Create initial achievements"""
    
    achievements_data = [
        # Beginner achievements
        {
            "name": "First Sort",
            "description": "Complete your first waste sorting action",
            "achievement_type": AchievementType.MILESTONE,
            "criteria": {"sorting_events": 1},
            "bonus_points": 50,
            "icon": "🎯"
        },
        {
            "name": "Quick Learner", 
            "description": "Achieve 80% sorting accuracy in your first 10 sorts",
            "achievement_type": AchievementType.ACCURACY,
            "criteria": {"accuracy_threshold": 0.8, "events_count": 10},
            "bonus_points": 100,
            "icon": "🎓"
        },
        {
            "name": "Waste Warrior",
            "description": "Sort 100 different waste items",
            "achievement_type": AchievementType.MILESTONE,
            "criteria": {"sorting_events": 100},
            "bonus_points": 500,
            "icon": "⚔️"
        },
        
        # Accuracy achievements
        {
            "name": "Perfect Sorter",
            "description": "Achieve 95% accuracy over 50 consecutive sorts",
            "achievement_type": AchievementType.ACCURACY,
            "criteria": {"accuracy_threshold": 0.95, "consecutive_events": 50},
            "bonus_points": 300,
            "icon": "🎯"
        },
        {
            "name": "Contamination Finder",
            "description": "Successfully identify 20 contaminated items",
            "achievement_type": AchievementType.SPECIAL,
            "criteria": {"contamination_found": 20},
            "bonus_points": 200,
            "icon": "🔍"
        },
        {
            "name": "Safety Expert",
            "description": "Properly handle 10 hazardous waste items",
            "achievement_type": AchievementType.SPECIAL,
            "criteria": {"hazardous_handled": 10},
            "bonus_points": 400,
            "icon": "⚠️"
        },
        
        # Streak achievements
        {
            "name": "Consistency Champion",
            "description": "Maintain a 7-day sorting streak",
            "achievement_type": AchievementType.STREAK,
            "criteria": {"streak_days": 7},
            "bonus_points": 150,
            "icon": "🔥"
        },
        {
            "name": "Dedication Master",
            "description": "Maintain a 30-day sorting streak",
            "achievement_type": AchievementType.STREAK,
            "criteria": {"streak_days": 30},
            "bonus_points": 1000,
            "icon": "🌟"
        },
        {
            "name": "Unstoppable Force",
            "description": "Maintain a 100-day sorting streak",
            "achievement_type": AchievementType.STREAK,
            "criteria": {"streak_days": 100},
            "bonus_points": 5000,
            "icon": "💫"
        },
        
        # Volume achievements
        {
            "name": "Lightweight Champion",
            "description": "Sort 10kg of waste total",
            "achievement_type": AchievementType.VOLUME,
            "criteria": {"total_weight_kg": 10},
            "bonus_points": 100,
            "icon": "📦"
        },
        {
            "name": "Heavy Lifter",
            "description": "Sort 100kg of waste total",
            "achievement_type": AchievementType.VOLUME,
            "criteria": {"total_weight_kg": 100},
            "bonus_points": 500,
            "icon": "🏋️"
        },
        {
            "name": "Environmental Hero",
            "description": "Sort 1000kg of waste total",
            "achievement_type": AchievementType.VOLUME,
            "criteria": {"total_weight_kg": 1000},
            "bonus_points": 2000,
            "icon": "🌍"
        },
        
        # Specialty achievements
        {
            "name": "Plastic Specialist",
            "description": "Sort 50 plastic items with 90% accuracy",
            "achievement_type": AchievementType.CATEGORY,
            "criteria": {"waste_type": "PLASTIC", "count": 50, "accuracy": 0.9},
            "bonus_points": 300,
            "icon": "♻️"
        },
        {
            "name": "Organic Expert",
            "description": "Sort 50 organic waste items with 90% accuracy",
            "achievement_type": AchievementType.CATEGORY,
            "criteria": {"waste_type": "ORGANIC", "count": 50, "accuracy": 0.9},
            "bonus_points": 300,
            "icon": "🍎"
        },
        {
            "name": "Metal Master",
            "description": "Sort 25 metal items with 95% accuracy",
            "achievement_type": AchievementType.CATEGORY,
            "criteria": {"waste_type": "METAL", "count": 25, "accuracy": 0.95},
            "bonus_points": 400,
            "icon": "🔧"
        },
        
        # Community achievements
        {
            "name": "Community Helper",
            "description": "Help 5 neighbors with waste sorting",
            "achievement_type": AchievementType.SOCIAL,
            "criteria": {"neighbors_helped": 5},
            "bonus_points": 250,
            "icon": "🤝"
        },
        {
            "name": "Neighborhood Leader",
            "description": "Be the top sorter in your neighborhood for a month",
            "achievement_type": AchievementType.SOCIAL,
            "criteria": {"neighborhood_rank": 1, "duration_days": 30},
            "bonus_points": 800,
            "icon": "👑"
        }
    ]
    
    achievements = []
    for data in achievements_data:
        achievement = Achievement(
            name=data["name"],
            description=data["description"],
            achievement_type=data["achievement_type"],
            criteria=data["criteria"],
            bonus_points=data["bonus_points"],
            icon=data["icon"]
        )
        db.add(achievement)
        achievements.append(achievement)
    
    db.flush()  # Get IDs assigned
    return achievements


def create_incentive_rules(db) -> list:
    """Create initial incentive rules"""
    
    rules_data = [
        # Base sorting rules
        {
            "name": "Correct Sorting Bonus",
            "description": "Bonus for correctly sorting any waste type",
            "rule_type": "accuracy_bonus",
            "conditions": {"is_correctly_sorted": True},
            "point_multiplier": 1.0,
            "bonus_points": 10,
            "is_active": True
        },
        {
            "name": "High Confidence Bonus",
            "description": "Bonus for high-confidence classifications (>90%)",
            "rule_type": "confidence_bonus",
            "conditions": {"confidence_threshold": 0.9},
            "point_multiplier": 1.2,
            "bonus_points": 5,
            "is_active": True
        },
        {
            "name": "Perfect Accuracy Bonus",
            "description": "Bonus for perfect classification confidence (>95%)",
            "rule_type": "confidence_bonus",
            "conditions": {"confidence_threshold": 0.95},
            "point_multiplier": 1.5,
            "bonus_points": 15,
            "is_active": True
        },
        
        # Volume-based rules
        {
            "name": "Large Item Bonus",
            "description": "Bonus for sorting larger waste items (>2kg)",
            "rule_type": "volume_bonus",
            "conditions": {"min_weight_kg": 2.0},
            "point_multiplier": 1.3,
            "bonus_points": 20,
            "is_active": True
        },
        {
            "name": "Bulk Processing Bonus",
            "description": "Bonus for sorting very large items (>5kg)",
            "rule_type": "volume_bonus",
            "conditions": {"min_weight_kg": 5.0},
            "point_multiplier": 1.8,
            "bonus_points": 50,
            "is_active": True
        },
        
        # Difficulty-based rules
        {
            "name": "Contamination Detection Bonus",
            "description": "Extra points for identifying contaminated items",
            "rule_type": "difficulty_bonus",
            "conditions": {"contamination_detected": True},
            "point_multiplier": 1.0,
            "bonus_points": 25,
            "is_active": True
        },
        {
            "name": "Hazardous Material Bonus",
            "description": "High bonus for safely handling hazardous materials",
            "rule_type": "safety_bonus",
            "conditions": {"hazardous_material": True},
            "point_multiplier": 2.0,
            "bonus_points": 100,
            "is_active": True
        },
        
        # Time-based rules
        {
            "name": "Early Bird Bonus",
            "description": "Bonus for sorting before 8 AM",
            "rule_type": "time_bonus",
            "conditions": {"hour_range": [5, 8]},
            "point_multiplier": 1.1,
            "bonus_points": 5,
            "is_active": True
        },
        {
            "name": "Weekend Warrior Bonus",
            "description": "Bonus for weekend sorting activities",
            "rule_type": "time_bonus",
            "conditions": {"weekend_only": True},
            "point_multiplier": 1.2,
            "bonus_points": 10,
            "is_active": True
        },
        
        # Streak-based rules
        {
            "name": "Daily Streak Bonus",
            "description": "Progressive bonus for maintaining daily streaks",
            "rule_type": "streak_bonus",
            "conditions": {"min_streak_days": 3},
            "point_multiplier": 1.1,
            "bonus_points": 15,
            "is_active": True
        },
        {
            "name": "Weekly Streak Bonus",
            "description": "Significant bonus for week-long streaks",
            "rule_type": "streak_bonus",
            "conditions": {"min_streak_days": 7},
            "point_multiplier": 1.5,
            "bonus_points": 50,
            "is_active": True
        },
        
        # Membership tier rules
        {
            "name": "Bronze Member Bonus",
            "description": "Small bonus for Bronze tier members",
            "rule_type": "tier_bonus",
            "conditions": {"min_tier": "BRONZE"},
            "point_multiplier": 1.05,
            "bonus_points": 2,
            "is_active": True
        },
        {
            "name": "Silver Member Bonus",
            "description": "Moderate bonus for Silver tier members",
            "rule_type": "tier_bonus",
            "conditions": {"min_tier": "SILVER"},
            "point_multiplier": 1.15,
            "bonus_points": 5,
            "is_active": True
        },
        {
            "name": "Gold Member Bonus",
            "description": "Significant bonus for Gold tier members",
            "rule_type": "tier_bonus",
            "conditions": {"min_tier": "GOLD"},
            "point_multiplier": 1.25,
            "bonus_points": 10,
            "is_active": True
        }
    ]
    
    rules = []
    for data in rules_data:
        rule = IncentiveRule(
            name=data["name"],
            description=data["description"],
            rule_type=data["rule_type"],
            conditions=data["conditions"],
            point_multiplier=data["point_multiplier"],
            bonus_points=data["bonus_points"],
            is_active=data["is_active"],
            priority=len(rules) + 1,  # Set priority based on order
            valid_from=datetime.utcnow(),
            valid_until=datetime.utcnow() + timedelta(days=365)  # 1 year validity
        )
        db.add(rule)
        rules.append(rule)
    
    db.flush()
    return rules


def create_reward_catalog(db) -> list:
    """Create initial reward catalog"""
    
    rewards_data = [
        # Digital rewards
        {
            "name": "Environmental Badge",
            "description": "Digital badge showing your commitment to the environment",
            "reward_type": RewardType.DIGITAL,
            "point_cost": 100,
            "stock_quantity": None,  # Unlimited
            "is_available": True,
            "icon": "🌱",
            "metadata": {"badge_type": "environmental_steward", "rarity": "common"}
        },
        {
            "name": "Waste Expert Certificate",
            "description": "Digital certificate recognizing your waste sorting expertise",
            "reward_type": RewardType.DIGITAL,
            "point_cost": 500,
            "stock_quantity": None,
            "is_available": True,
            "icon": "📜",
            "metadata": {"certificate_type": "waste_expert", "downloadable": True}
        },
        {
            "name": "Kampala Hero Avatar",
            "description": "Exclusive avatar frame for your profile",
            "reward_type": RewardType.DIGITAL,
            "point_cost": 750,
            "stock_quantity": None,
            "is_available": True,
            "icon": "🦸",
            "metadata": {"avatar_type": "kampala_hero", "rarity": "rare"}
        },
        
        # Physical rewards
        {
            "name": "Reusable Water Bottle",
            "description": "Eco-friendly stainless steel water bottle",
            "reward_type": RewardType.PHYSICAL,
            "point_cost": 2000,
            "stock_quantity": 50,
            "is_available": True,
            "icon": "🍼",
            "metadata": {"material": "stainless_steel", "capacity_ml": 750, "brand": "EcoLife"}
        },
        {
            "name": "Organic Cotton Tote Bag",
            "description": "Sustainable shopping bag made from organic cotton",
            "reward_type": RewardType.PHYSICAL,
            "point_cost": 1500,
            "stock_quantity": 75,
            "is_available": True,
            "icon": "👜",
            "metadata": {"material": "organic_cotton", "size": "large", "brand": "GreenCarry"}
        },
        {
            "name": "Solar-Powered Phone Charger",
            "description": "Portable solar charger for sustainable energy",
            "reward_type": RewardType.PHYSICAL,
            "point_cost": 5000,
            "stock_quantity": 20,
            "is_available": True,
            "icon": "🔋",
            "metadata": {"power_output": "10W", "brand": "SolarTech", "weather_resistant": True}
        },
        {
            "name": "Premium Compost Bin",
            "description": "High-quality compost bin for organic waste",
            "reward_type": RewardType.PHYSICAL,
            "point_cost": 8000,
            "stock_quantity": 10,
            "is_available": True,
            "icon": "🗑️",
            "metadata": {"capacity_liters": 50, "material": "recycled_plastic", "brand": "CompostMaster"}
        },
        
        # Experience rewards
        {
            "name": "Waste Management Facility Tour",
            "description": "Guided tour of KCCA's waste management facilities",
            "reward_type": RewardType.EXPERIENCE,
            "point_cost": 3000,
            "stock_quantity": 4,  # 4 tours per month
            "is_available": True,
            "icon": "🏭",
            "metadata": {"duration_hours": 3, "includes_lunch": True, "group_size": 15}
        },
        {
            "name": "Environmental Workshop",
            "description": "Educational workshop on sustainable living practices",
            "reward_type": RewardType.EXPERIENCE,
            "point_cost": 2500,
            "stock_quantity": 8,  # 8 workshops per month
            "is_available": True,
            "icon": "🎓",
            "metadata": {"duration_hours": 4, "certificate_included": True, "expert_led": True}
        },
        {
            "name": "Tree Planting Event",
            "description": "Participate in community tree planting initiative",
            "reward_type": RewardType.EXPERIENCE,
            "point_cost": 1000,
            "stock_quantity": 20,  # 20 slots per event
            "is_available": True,
            "icon": "🌳",
            "metadata": {"duration_hours": 4, "trees_planted": 3, "location": "various"}
        },
        
        # Service credits
        {
            "name": "Free Waste Collection",
            "description": "One free household waste collection service",
            "reward_type": RewardType.SERVICE,
            "point_cost": 4000,
            "stock_quantity": 100,
            "is_available": True,
            "icon": "🚛",
            "metadata": {"service_type": "collection", "max_bags": 5, "valid_days": 30}
        },
        {
            "name": "Recycling Center Credit",
            "description": "Credit for premium recycling services",
            "reward_type": RewardType.SERVICE,
            "point_cost": 1500,
            "stock_quantity": 200,
            "is_available": True,
            "icon": "♻️",
            "metadata": {"credit_amount": "UGX 20,000", "valid_days": 90, "transfer_allowed": False}
        },
        {
            "name": "Bulk Waste Disposal",
            "description": "Free disposal service for large items",
            "reward_type": RewardType.SERVICE,
            "point_cost": 6000,
            "stock_quantity": 25,
            "is_available": True,
            "icon": "📦",
            "metadata": {"max_items": 3, "pickup_included": True, "valid_days": 60}
        },
        
        # Discount vouchers
        {
            "name": "Eco-Store Discount",
            "description": "20% discount at partner eco-friendly stores",
            "reward_type": RewardType.VOUCHER,
            "point_cost": 800,
            "stock_quantity": 500,
            "is_available": True,
            "icon": "🎫",
            "metadata": {"discount_percent": 20, "partner_stores": ["EcoMart", "GreenLife", "SustainShop"], "valid_days": 45}
        },
        {
            "name": "Public Transport Credit",
            "description": "Credit for Kampala public transportation",
            "reward_type": RewardType.VOUCHER,
            "point_cost": 1200,
            "stock_quantity": 300,
            "is_available": True,
            "icon": "🚌",
            "metadata": {"credit_amount": "UGX 15,000", "valid_transport": ["bus", "taxi"], "valid_days": 30}
        },
        {
            "name": "Farmers Market Voucher",
            "description": "Voucher for local organic farmers market",
            "reward_type": RewardType.VOUCHER,
            "point_cost": 1000,
            "stock_quantity": 150,
            "is_available": True,
            "icon": "🥕",
            "metadata": {"voucher_amount": "UGX 25,000", "market_locations": ["Nakasero", "Owino"], "valid_days": 21}
        }
    ]
    
    rewards = []
    for data in rewards_data:
        reward = RewardCatalog(
            name=data["name"],
            description=data["description"],
            reward_type=data["reward_type"],
            point_cost=data["point_cost"],
            stock_quantity=data["stock_quantity"],
            current_stock=data["stock_quantity"],  # Start with full stock
            is_available=data["is_available"],
            icon=data["icon"],
            reward_metadata=data["metadata"]
        )
        db.add(reward)
        rewards.append(reward)
    
    db.flush()
    return rewards


def create_configuration_file(database_url: str = "sqlite:///waste_management.db"):
    """Create a configuration file for the system"""
    
    config = {
        "database": {
            "url": database_url,
            "echo": False,
            "pool_size": 10,
            "max_overflow": 20
        },
        "anthropic": {
            "api_key": "your-anthropic-api-key-here",
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1000,
            "temperature": 0.3
        },
        "mobilevit": {
            "model_path": "best_mobilevit_waste_model.pth",
            "confidence_threshold": 0.8,
            "uncertainty_samples": 10
        },
        "points": {
            "base_points": 10,
            "max_multiplier": 3.0,
            "streak_bonus_cap": 100,
            "achievement_bonus_cap": 500
        },
        "tiers": {
            "bronze_threshold": 1000,
            "silver_threshold": 5000,
            "gold_threshold": 15000,
            "platinum_threshold": 50000
        },
        "rewards": {
            "restock_frequency_days": 30,
            "voucher_validity_days": 45,
            "experience_booking_days": 7
        },
        "notifications": {
            "achievement_enabled": True,
            "tier_upgrade_enabled": True,
            "streak_reminder_enabled": True,
            "reward_availability_enabled": True
        }
    }
    
    config_path = "waste_management_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📄 Configuration file created: {config_path}")
    return config_path


def main():
    """Main initialization function"""
    
    print("🚀 Waste Management Incentivization System Initialization")
    print("=" * 65)
    
    # Initialize database
    result = initialize_database()
    
    if result["success"]:
        # Create configuration file
        config_path = create_configuration_file()
        
        print(f"\n📊 Database Initialization Summary:")
        print(f"   ✅ Database: Initialized successfully")
        print(f"   ✅ Achievements: {result['achievements']} created")
        print(f"   ✅ Incentive Rules: {result['rules']} created")
        print(f"   ✅ Rewards: {result['rewards']} created")
        print(f"   ✅ Configuration: {config_path}")
        
        print(f"\n🔧 Next Steps:")
        print(f"   1. Update Anthropic API key in {config_path}")
        print(f"   2. Verify MobileViT model path in configuration")
        print(f"   3. Run waste_incentive_integration.py for demo")
        print(f"   4. Start the web API server")
        
        print(f"\n💡 Usage Example:")
        print(f"   python waste_incentive_integration.py")
        
    else:
        print(f"❌ Initialization failed: {result['error']}")
        
    print(f"\n✅ Initialization complete!")


if __name__ == "__main__":
    main()