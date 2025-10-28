"""
Database models for the waste management incentivization system
"""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, Enum, Text, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class WasteType(enum.Enum):
    """Enumeration of waste types"""
    PLASTIC = "plastic"
    ORGANIC = "organic"
    PAPER_CARDBOARD = "paper_cardboard"
    GLASS = "glass"
    METAL = "metal"
    ELECTRONIC = "electronic"
    MEDICAL = "medical"
    MIXED = "mixed"
    HAZARDOUS = "hazardous"


class UserType(enum.Enum):
    """User types in the system"""
    INDIVIDUAL = "individual"
    HOUSEHOLD = "household"
    BUSINESS = "business"
    INSTITUTION = "institution"


class MembershipTier(enum.Enum):
    """Membership tiers for users"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


class AchievementType(enum.Enum):
    """Types of achievements"""
    MILESTONE = "milestone"
    ACCURACY = "accuracy"
    STREAK = "streak"
    VOLUME = "volume"
    CATEGORY = "category"
    SPECIAL = "special"
    SOCIAL = "social"


class RewardType(enum.Enum):
    """Types of rewards"""
    DIGITAL = "digital"
    PHYSICAL = "physical"
    EXPERIENCE = "experience"
    SERVICE = "service"
    VOUCHER = "voucher"


class User(Base):
    """User model for waste sorters"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(20))
    user_type = Column(Enum(UserType), default=UserType.INDIVIDUAL)
    neighborhood = Column(String(100))
    
    # Points and gamification
    total_points = Column(Integer, default=0)
    available_points = Column(Integer, default=0)  # Points not yet redeemed
    membership_tier = Column(Enum(MembershipTier), default=MembershipTier.BRONZE)
    
    # Streak tracking
    current_streak_days = Column(Integer, default=0)
    longest_streak_days = Column(Integer, default=0)
    last_sorting_date = Column(DateTime)
    
    # Profile
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    sorting_events = relationship("SortingEvent", back_populates="user")
    achievements = relationship("UserAchievement", back_populates="user")
    redemptions = relationship("Redemption", back_populates="user")


class SortingEvent(Base):
    """Record of a waste sorting event"""
    __tablename__ = "sorting_events"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Waste details
    waste_type = Column(Enum(WasteType), nullable=False)
    secondary_waste_types = Column(JSON)  # List of additional waste types
    quantity_kg = Column(Float, nullable=False)
    
    # Classification results
    is_correctly_sorted = Column(Boolean, default=True)
    is_contaminated = Column(Boolean, default=False)
    is_hazardous_unsafe = Column(Boolean, default=False)
    confidence_score = Column(Float)  # LVM confidence
    
    # Points and rewards
    points_earned = Column(Integer, default=0)
    bonus_multiplier = Column(Float, default=1.0)
    bonus_reason = Column(Text)
    
    # Metadata
    sorting_date = Column(DateTime, default=datetime.utcnow)
    location_lat = Column(Float)
    location_lng = Column(Float)
    image_path = Column(String(255))  # Path to sorted waste image
    
    # Relationships
    user = relationship("User", back_populates="sorting_events")


class Achievement(Base):
    """Achievement definitions"""
    __tablename__ = "achievements"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    achievement_type = Column(Enum(AchievementType), nullable=False)
    criteria = Column(JSON)  # Achievement criteria as JSON
    icon = Column(String(10))  # Emoji icon
    
    # Rewards
    bonus_points = Column(Integer, default=0)
    badge_color = Column(String(20))  # For UI display
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user_achievements = relationship("UserAchievement", back_populates="achievement")


class UserAchievement(Base):
    """Junction table for user achievements"""
    __tablename__ = "user_achievements"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    achievement_id = Column(Integer, ForeignKey("achievements.id"), nullable=False)
    earned_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="achievements")
    achievement = relationship("Achievement", back_populates="user_achievements")


class IncentiveRule(Base):
    """Dynamic incentive rules"""
    __tablename__ = "incentive_rules"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Rule conditions
    waste_type = Column(Enum(WasteType))  # Null means applies to all
    applies_to_neighborhoods = Column(JSON)  # List of neighborhoods
    min_tier_required = Column(Enum(MembershipTier))
    
    # Rule effects
    points_modifier = Column(Integer, default=0)  # Add/subtract points
    multiplier = Column(Float, default=1.0)  # Multiply final points
    
    # Validity
    valid_from = Column(DateTime, default=datetime.utcnow)
    valid_until = Column(DateTime)  # Null means no expiry
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=1)  # Higher priority rules apply first
    
    created_at = Column(DateTime, default=datetime.utcnow)


class RewardCatalog(Base):
    """Available rewards for point redemption"""
    __tablename__ = "reward_catalog"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    reward_type = Column(Enum(RewardType), nullable=False)
    
    # Cost and availability
    point_cost = Column(Integer, nullable=False)
    stock_quantity = Column(Integer)  # Null means unlimited
    current_stock = Column(Integer)
    is_available = Column(Boolean, default=True)
    
    # Display
    icon = Column(String(10))  # Emoji icon
    reward_metadata = Column(JSON)  # Additional reward metadata
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    redemptions = relationship("Redemption", back_populates="reward")


class Redemption(Base):
    """Record of reward redemptions"""
    __tablename__ = "redemptions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reward_id = Column(Integer, ForeignKey("reward_catalog.id"), nullable=False)
    
    # Redemption details
    points_spent = Column(Integer, nullable=False)
    status = Column(String(20), default="pending")  # pending, approved, used, expired
    voucher_code = Column(String(50), unique=True)
    redemption_instructions = Column(Text)
    
    # Timestamps
    redeemed_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime)
    used_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="redemptions")
    reward = relationship("RewardCatalog", back_populates="redemptions")


class WasteStream(Base):
    """Tracking of waste streams and quantities"""
    __tablename__ = "waste_streams"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    waste_type = Column(Enum(WasteType), nullable=False)
    
    # Quantities
    total_collected_kg = Column(Float, default=0.0)
    total_sorted_kg = Column(Float, default=0.0)
    contamination_rate = Column(Float, default=0.0)  # Percentage
    
    # Location
    neighborhood = Column(String(100))
    collection_point = Column(String(100))
    
    # Derived metrics
    sorting_accuracy = Column(Float, default=0.0)  # Percentage
    participant_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Notification(Base):
    """User notifications"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Message details
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50))  # "achievement", "reward", "reminder", etc.
    
    # Status
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")


class LeaderboardEntry(Base):
    """Leaderboard rankings"""
    __tablename__ = "leaderboard"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Rankings
    rank_position = Column(Integer)
    points_total = Column(Integer)
    ranking_period = Column(String(20))  # "weekly", "monthly", "all_time"
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    
    # Additional metrics
    sorting_events_count = Column(Integer, default=0)
    accuracy_percentage = Column(Float, default=0.0)
    streak_days = Column(Integer, default=0)
    
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")