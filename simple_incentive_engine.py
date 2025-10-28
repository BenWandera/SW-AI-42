"""
Simplified Incentivization Logic (No Database)
Pure calculation engine for waste sorting incentives using Anthropic API
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import anthropic
from dataclasses import dataclass


@dataclass
class UserProfile:
    """Simple user profile without database dependency"""
    name: str
    total_points: int = 0
    current_streak: int = 0
    membership_tier: str = "BRONZE"
    total_sorts: int = 0
    accuracy_history: List[float] = None
    
    def __post_init__(self):
        if self.accuracy_history is None:
            self.accuracy_history = []


@dataclass
class SortingResult:
    """Result of a waste sorting event"""
    waste_type: str
    predicted_class: str
    confidence: float
    is_correctly_sorted: bool
    is_contaminated: bool = False
    is_hazardous: bool = False
    quantity_kg: float = 1.0
    
    
@dataclass
class IncentiveResult:
    """Result of incentive calculation"""
    points_earned: int
    multiplier: float
    bonus_reason: str
    feedback_message: str
    achievements_unlocked: List[str]
    tier_upgrade: Optional[str] = None


class SimpleIncentiveEngine:
    """
    Simplified incentive calculation engine without database dependencies
    """
    
    def __init__(self, anthropic_api_key: str):
        """
        Initialize the incentive engine
        
        Args:
            anthropic_api_key: Anthropic API key for AI-powered calculations
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Base configuration
        self.base_points = 10
        self.max_multiplier = 3.0
        
        # Tier thresholds
        self.tier_thresholds = {
            "BRONZE": 0,
            "SILVER": 1000,
            "GOLD": 5000,
            "PLATINUM": 15000
        }
        
        # Achievement definitions
        self.achievements = {
            "first_sort": {"name": "First Sort", "points": 50, "criteria": "total_sorts >= 1"},
            "accuracy_master": {"name": "Accuracy Master", "points": 100, "criteria": "accuracy > 0.9 and total_sorts >= 10"},
            "streak_warrior": {"name": "Streak Warrior", "points": 150, "criteria": "current_streak >= 7"},
            "waste_expert": {"name": "Waste Expert", "points": 500, "criteria": "total_sorts >= 100"},
            "contamination_detective": {"name": "Contamination Detective", "points": 200, "criteria": "contamination_found >= 5"},
            "safety_champion": {"name": "Safety Champion", "points": 300, "criteria": "hazardous_handled >= 3"}
        }
        
        print(f"🎯 Simple Incentive Engine initialized with Anthropic API")
    
    def calculate_incentives(self, 
                           user: UserProfile, 
                           sorting_result: SortingResult,
                           use_ai: bool = True) -> IncentiveResult:
        """
        Calculate incentives for a sorting event
        
        Args:
            user: User profile
            sorting_result: Details of the sorting event
            use_ai: Whether to use AI for enhanced calculation
            
        Returns:
            IncentiveResult with points, bonuses, and feedback
        """
        
        print(f"\n💰 Calculating incentives for {user.name}")
        print(f"   Waste: {sorting_result.waste_type}")
        print(f"   Accuracy: {sorting_result.confidence:.1%}")
        print(f"   Correctly sorted: {sorting_result.is_correctly_sorted}")
        
        if use_ai:
            try:
                return self._ai_enhanced_calculation(user, sorting_result)
            except Exception as e:
                print(f"⚠️  AI calculation failed, using fallback: {e}")
                return self._fallback_calculation(user, sorting_result)
        else:
            return self._fallback_calculation(user, sorting_result)
    
    def _ai_enhanced_calculation(self, user: UserProfile, sorting_result: SortingResult) -> IncentiveResult:
        """AI-enhanced incentive calculation using Anthropic Claude"""
        
        # Prepare context for AI
        context = self._build_user_context(user, sorting_result)
        
        prompt = f"""
You are an expert in gamification and environmental incentive systems. Calculate appropriate incentives for a waste sorting event.

User Context:
{json.dumps(context, indent=2)}

Base Rules:
- Base points: {self.base_points}
- Maximum multiplier: {self.max_multiplier}x
- Correct sorting: +10 bonus points
- High confidence (>90%): +5 bonus points, 1.2x multiplier
- Perfect confidence (>95%): +15 bonus points, 1.5x multiplier
- Large item (>2kg): +20 bonus points, 1.3x multiplier
- Contamination detection: +25 bonus points
- Hazardous material: +100 bonus points, 2.0x multiplier
- Streak bonuses: 3+ days = 1.1x, 7+ days = 1.5x

Calculate:
1. Total points (base + bonuses)
2. Final multiplier (combined from all sources, max {self.max_multiplier}x)
3. Bonus reason (why this multiplier/bonus was given)
4. Personalized feedback message (encouraging, specific, actionable)

Respond in JSON format:
{{
  "points_earned": <integer>,
  "multiplier": <float>,
  "bonus_reason": "<string>",
  "feedback_message": "<string>"
}}
"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse AI response
            ai_result = json.loads(response.content[0].text)
            
            # Calculate achievements
            achievements = self._check_achievements(user, sorting_result)
            
            # Check for tier upgrade
            new_total = user.total_points + ai_result["points_earned"]
            tier_upgrade = self._check_tier_upgrade(user.membership_tier, new_total)
            
            return IncentiveResult(
                points_earned=ai_result["points_earned"],
                multiplier=ai_result["multiplier"],
                bonus_reason=ai_result["bonus_reason"],
                feedback_message=ai_result["feedback_message"],
                achievements_unlocked=achievements,
                tier_upgrade=tier_upgrade
            )
            
        except Exception as e:
            print(f"❌ AI calculation error: {e}")
            return self._fallback_calculation(user, sorting_result)
    
    def _fallback_calculation(self, user: UserProfile, sorting_result: SortingResult) -> IncentiveResult:
        """Fallback calculation without AI"""
        
        points = self.base_points
        multiplier = 1.0
        bonus_reasons = []
        
        # Accuracy bonuses
        if sorting_result.is_correctly_sorted:
            points += 10
            bonus_reasons.append("correct sorting")
        
        if sorting_result.confidence > 0.95:
            points += 15
            multiplier *= 1.5
            bonus_reasons.append("perfect confidence")
        elif sorting_result.confidence > 0.9:
            points += 5
            multiplier *= 1.2
            bonus_reasons.append("high confidence")
        
        # Volume bonuses
        if sorting_result.quantity_kg > 5.0:
            points += 50
            multiplier *= 1.8
            bonus_reasons.append("bulk processing")
        elif sorting_result.quantity_kg > 2.0:
            points += 20
            multiplier *= 1.3
            bonus_reasons.append("large item")
        
        # Special conditions
        if sorting_result.is_contaminated:
            points += 25
            bonus_reasons.append("contamination detection")
        
        if sorting_result.is_hazardous:
            points += 100
            multiplier *= 2.0
            bonus_reasons.append("hazardous material handling")
        
        # Streak bonuses
        if user.current_streak >= 7:
            multiplier *= 1.5
            bonus_reasons.append("weekly streak")
        elif user.current_streak >= 3:
            multiplier *= 1.1
            bonus_reasons.append("streak bonus")
        
        # Apply multiplier cap
        multiplier = min(multiplier, self.max_multiplier)
        final_points = int(points * multiplier)
        
        # Generate feedback
        feedback = self._generate_simple_feedback(user, sorting_result, final_points, bonus_reasons)
        
        # Calculate achievements
        achievements = self._check_achievements(user, sorting_result)
        
        # Check for tier upgrade
        new_total = user.total_points + final_points
        tier_upgrade = self._check_tier_upgrade(user.membership_tier, new_total)
        
        return IncentiveResult(
            points_earned=final_points,
            multiplier=multiplier,
            bonus_reason=", ".join(bonus_reasons) if bonus_reasons else "base points",
            feedback_message=feedback,
            achievements_unlocked=achievements,
            tier_upgrade=tier_upgrade
        )
    
    def _build_user_context(self, user: UserProfile, sorting_result: SortingResult) -> dict:
        """Build context for AI calculation"""
        
        # Calculate average accuracy
        avg_accuracy = sum(user.accuracy_history) / len(user.accuracy_history) if user.accuracy_history else 0.0
        
        return {
            "user": {
                "name": user.name,
                "total_points": user.total_points,
                "membership_tier": user.membership_tier,
                "current_streak": user.current_streak,
                "total_sorts": user.total_sorts,
                "average_accuracy": avg_accuracy
            },
            "current_sort": {
                "waste_type": sorting_result.waste_type,
                "predicted_class": sorting_result.predicted_class,
                "confidence": sorting_result.confidence,
                "correctly_sorted": sorting_result.is_correctly_sorted,
                "contaminated": sorting_result.is_contaminated,
                "hazardous": sorting_result.is_hazardous,
                "quantity_kg": sorting_result.quantity_kg
            },
            "context": {
                "time_of_day": datetime.now().hour,
                "day_of_week": datetime.now().strftime("%A"),
                "is_weekend": datetime.now().weekday() >= 5
            }
        }
    
    def _check_achievements(self, user: UserProfile, sorting_result: SortingResult) -> List[str]:
        """Check what achievements were unlocked"""
        
        achievements_unlocked = []
        
        # Update temporary stats for checking
        temp_total_sorts = user.total_sorts + 1
        temp_accuracy = (sum(user.accuracy_history) + sorting_result.confidence) / temp_total_sorts if temp_total_sorts > 0 else sorting_result.confidence
        
        # Check each achievement
        for achievement_id, achievement in self.achievements.items():
            criteria = achievement["criteria"]
            
            # Simple evaluation (in production, use safer evaluation)
            context = {
                "total_sorts": temp_total_sorts,
                "accuracy": temp_accuracy,
                "current_streak": user.current_streak,
                "contamination_found": 1 if sorting_result.is_contaminated else 0,
                "hazardous_handled": 1 if sorting_result.is_hazardous else 0
            }
            
            try:
                if eval(criteria, {"__builtins__": {}}, context):
                    achievements_unlocked.append(achievement["name"])
            except:
                continue
        
        return achievements_unlocked
    
    def _check_tier_upgrade(self, current_tier: str, new_total_points: int) -> Optional[str]:
        """Check if user qualifies for tier upgrade"""
        
        for tier, threshold in self.tier_thresholds.items():
            if new_total_points >= threshold and tier != current_tier:
                # Find current tier index
                tiers = list(self.tier_thresholds.keys())
                current_idx = tiers.index(current_tier)
                new_idx = tiers.index(tier)
                
                if new_idx > current_idx:
                    return tier
        
        return None
    
    def _generate_simple_feedback(self, user: UserProfile, sorting_result: SortingResult, 
                                points: int, bonus_reasons: List[str]) -> str:
        """Generate simple feedback message"""
        
        messages = [
            f"Great job, {user.name}! You earned {points} points for sorting {sorting_result.waste_type}.",
        ]
        
        if sorting_result.confidence > 0.95:
            messages.append("Perfect classification confidence! You're becoming an expert.")
        elif sorting_result.confidence > 0.9:
            messages.append("Excellent accuracy! Keep up the great work.")
        
        if user.current_streak >= 7:
            messages.append(f"Amazing {user.current_streak}-day streak! You're on fire! 🔥")
        elif user.current_streak >= 3:
            messages.append(f"Nice {user.current_streak}-day streak going!")
        
        if sorting_result.is_contaminated:
            messages.append("Thanks for identifying contamination - that helps keep our recycling clean!")
        
        if sorting_result.is_hazardous:
            messages.append("Excellent safety awareness with hazardous materials!")
        
        if bonus_reasons:
            messages.append(f"Bonuses: {', '.join(bonus_reasons)}")
        
        return " ".join(messages)
    
    def update_user_profile(self, user: UserProfile, sorting_result: SortingResult, 
                          incentive_result: IncentiveResult) -> UserProfile:
        """Update user profile with new sorting event results"""
        
        # Update points
        user.total_points += incentive_result.points_earned
        
        # Update sorting count
        user.total_sorts += 1
        
        # Update accuracy history (keep last 20 for efficiency)
        user.accuracy_history.append(sorting_result.confidence)
        if len(user.accuracy_history) > 20:
            user.accuracy_history = user.accuracy_history[-20:]
        
        # Update streak (simplified - assume daily sorting)
        if sorting_result.is_correctly_sorted:
            user.current_streak += 1
        else:
            user.current_streak = 0
        
        # Update tier if upgraded
        if incentive_result.tier_upgrade:
            user.membership_tier = incentive_result.tier_upgrade
        
        return user


class WasteClassificationSimulator:
    """
    Simulates waste classification results for demo purposes
    """
    
    def __init__(self):
        self.waste_types = [
            "PLASTIC", "ORGANIC", "PAPER_CARDBOARD", "GLASS", 
            "METAL", "ELECTRONIC", "MIXED", "HAZARDOUS"
        ]
        
        self.class_names = [
            "Plastic Bottle", "Food Waste", "Cardboard Box", "Glass Jar",
            "Metal Can", "Electronic Device", "Mixed Waste", "Battery"
        ]
    
    def simulate_classification(self, waste_type: str = None) -> SortingResult:
        """Simulate a waste classification result"""
        
        import random
        
        if waste_type is None:
            waste_type = random.choice(self.waste_types)
        
        predicted_class = random.choice(self.class_names)
        confidence = random.uniform(0.75, 0.98)
        is_correctly_sorted = confidence > 0.8
        is_contaminated = random.random() < 0.1
        is_hazardous = waste_type == "HAZARDOUS" or random.random() < 0.05
        quantity_kg = random.uniform(0.1, 3.0)
        
        return SortingResult(
            waste_type=waste_type,
            predicted_class=predicted_class,
            confidence=confidence,
            is_correctly_sorted=is_correctly_sorted,
            is_contaminated=is_contaminated,
            is_hazardous=is_hazardous,
            quantity_kg=quantity_kg
        )


def demo_incentive_system():
    """
    Demonstration of the simplified incentive system
    """
    print("🎯 Simple Waste Incentivization System Demo")
    print("=" * 50)
    
    # Initialize system (replace with your API key)
    api_key = "your-anthropic-api-key-here"  # Replace with actual key
    engine = SimpleIncentiveEngine(api_key)
    simulator = WasteClassificationSimulator()
    
    # Create demo users
    users = [
        UserProfile("Alice Nakato", total_points=250, current_streak=3, total_sorts=15),
        UserProfile("Bob Ssemwogerere", total_points=1200, current_streak=8, membership_tier="SILVER", total_sorts=50),
        UserProfile("Sarah Namukasa", total_points=0, total_sorts=0)  # New user
    ]
    
    # Simulate sorting events
    for i in range(5):
        user = users[i % len(users)]
        sorting_result = simulator.simulate_classification()
        
        print(f"\n--- Sorting Event {i+1}: {user.name} ---")
        print(f"Waste Type: {sorting_result.waste_type}")
        print(f"Confidence: {sorting_result.confidence:.1%}")
        print(f"Quantity: {sorting_result.quantity_kg:.1f}kg")
        
        # Calculate incentives (try AI first, fallback if no API key)
        use_ai = api_key != "your-anthropic-api-key-here"
        
        try:
            incentive_result = engine.calculate_incentives(user, sorting_result, use_ai=use_ai)
            
            # Update user profile
            updated_user = engine.update_user_profile(user, sorting_result, incentive_result)
            
            # Display results
            print(f"\n🎉 Results:")
            print(f"   Points Earned: {incentive_result.points_earned}")
            print(f"   Multiplier: {incentive_result.multiplier:.1f}x")
            print(f"   Bonus Reason: {incentive_result.bonus_reason}")
            print(f"   Total Points: {updated_user.total_points}")
            print(f"   Tier: {updated_user.membership_tier}")
            print(f"   Streak: {updated_user.current_streak} days")
            
            if incentive_result.achievements_unlocked:
                print(f"   🏆 Achievements: {', '.join(incentive_result.achievements_unlocked)}")
            
            if incentive_result.tier_upgrade:
                print(f"   ⬆️  Tier Upgrade: {incentive_result.tier_upgrade}")
            
            print(f"\n💬 Feedback: {incentive_result.feedback_message}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Demo completed!")
    print(f"\n📊 Final User Stats:")
    for user in users:
        print(f"   {user.name}: {user.total_points} points, {user.membership_tier} tier, {user.current_streak} day streak")


if __name__ == "__main__":
    demo_incentive_system()