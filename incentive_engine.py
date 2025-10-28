"""
Incentive Calculation Engine with Anthropic API Integration
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import anthropic
from models import (
    WasteType, User, SortingEvent, MembershipTier, 
    IncentiveRule, Achievement, UserAchievement
)
from sqlalchemy.orm import Session


class IncentiveEngine:
    """
    Core engine for calculating points and managing incentives
    """
    
    # Base points for each waste type
    BASE_POINTS = {
        WasteType.PLASTIC: 10,
        WasteType.ORGANIC: 5,
        WasteType.PAPER_CARDBOARD: 15,
        WasteType.GLASS: 15,
        WasteType.METAL: 15,
        WasteType.ELECTRONIC: 25,
        WasteType.MEDICAL: 30,
        WasteType.MIXED: -5,
        WasteType.HAZARDOUS: 20,  # If properly contained
    }
    
    # Penalties
    CONTAMINATED_PENALTY = -10
    UNSAFE_DISPOSAL_PENALTY = -20
    
    # Tier thresholds
    TIER_THRESHOLDS = {
        MembershipTier.BRONZE: 0,
        MembershipTier.SILVER: 500,
        MembershipTier.GOLD: 2000,
        MembershipTier.PLATINUM: 5000,
    }
    
    # Streak bonus
    WEEKLY_STREAK_BONUS = 50
    
    def __init__(self, anthropic_api_key: str, db: Session):
        """
        Initialize the incentive engine
        
        Args:
            anthropic_api_key: Your Anthropic API key
            db: SQLAlchemy database session
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.db = db
    
    def calculate_points(
        self,
        user: User,
        waste_type: WasteType,
        quantity_kg: float,
        is_correctly_sorted: bool,
        is_contaminated: bool,
        is_hazardous_unsafe: bool,
        confidence_score: float,
        graph_reasoning_result: Dict,
        secondary_waste_types: Optional[List[str]] = None
    ) -> Tuple[int, float, str, str]:
        """
        Calculate points for a sorting event using Anthropic API for intelligent reasoning
        
        Returns:
            Tuple of (points_earned, multiplier, bonus_reason, feedback_message)
        """
        
        # Build context for Claude
        user_context = self._build_user_context(user)
        
        prompt = f"""You are an AI assistant for a waste management incentivization system in Uganda. 
Analyze this waste sorting event and calculate appropriate points.

USER CONTEXT:
{json.dumps(user_context, indent=2)}

SORTING EVENT:
- Primary waste type: {waste_type.value}
- Secondary types: {secondary_waste_types or 'None'}
- Quantity: {quantity_kg} kg
- Correctly sorted: {is_correctly_sorted}
- Contaminated: {is_contaminated}
- Hazardous/unsafe disposal: {is_hazardous_unsafe}
- LVM confidence: {confidence_score}
- Graph reasoning: {json.dumps(graph_reasoning_result, indent=2)}

BASE POINT SYSTEM:
{json.dumps({k.value: v for k, v in self.BASE_POINTS.items()}, indent=2)}

RULES:
1. Start with base points for the waste type
2. Apply penalties: -10 for contamination, -20 for unsafe disposal
3. Consider user's learning progress (new users get more encouragement)
4. Check if user maintains sorting streak (bonus if applicable)
5. Consider seasonal or scarcity bonuses if relevant
6. Apply tier-based multipliers (Bronze: 1.0x, Silver: 1.1x, Gold: 1.2x, Platinum: 1.3x)

Calculate:
1. Total points to award (can be negative)
2. Any multiplier applied
3. Brief reason for bonuses/penalties
4. Educational feedback message (2-3 sentences) to help user improve

Respond in JSON format:
{{
    "points": <integer>,
    "multiplier": <float>,
    "bonus_reason": "<string>",
    "feedback_message": "<string>",
    "improvement_tips": "<string or null>"
}}
"""
        
        # Call Anthropic API
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse response
            result = json.loads(response.content[0].text)
            
            return (
                result["points"],
                result["multiplier"],
                result["bonus_reason"],
                result["feedback_message"]
            )
            
        except Exception as e:
            # Fallback to basic calculation if API fails
            print(f"Anthropic API error: {e}. Using fallback calculation.")
            return self._fallback_calculation(
                waste_type, is_correctly_sorted, is_contaminated, 
                is_hazardous_unsafe, user
            )
    
    def _build_user_context(self, user: User) -> Dict:
        """Build user context for Claude"""
        return {
            "user_id": user.id,
            "name": user.name,
            "user_type": user.user_type.value,
            "total_points": user.total_points,
            "membership_tier": user.membership_tier.value,
            "current_streak": user.current_streak_days,
            "last_sorting_date": user.last_sorting_date.isoformat() if user.last_sorting_date else None,
            "neighborhood": user.neighborhood,
        }
    
    def _fallback_calculation(
        self, 
        waste_type: WasteType, 
        is_correctly_sorted: bool,
        is_contaminated: bool,
        is_hazardous_unsafe: bool,
        user: User
    ) -> Tuple[int, float, str, str]:
        """Fallback calculation without AI"""
        
        points = self.BASE_POINTS.get(waste_type, 0)
        multiplier = self._get_tier_multiplier(user.membership_tier)
        
        if not is_correctly_sorted:
            points = -5
        if is_contaminated:
            points += self.CONTAMINATED_PENALTY
        if is_hazardous_unsafe:
            points += self.UNSAFE_DISPOSAL_PENALTY
        
        points = int(points * multiplier)
        
        bonus_reason = f"Tier multiplier: {multiplier}x"
        feedback = "Thank you for sorting your waste!"
        
        return points, multiplier, bonus_reason, feedback
    
    def _get_tier_multiplier(self, tier: MembershipTier) -> float:
        """Get point multiplier based on membership tier"""
        multipliers = {
            MembershipTier.BRONZE: 1.0,
            MembershipTier.SILVER: 1.1,
            MembershipTier.GOLD: 1.2,
            MembershipTier.PLATINUM: 1.3,
        }
        return multipliers.get(tier, 1.0)
    
    def update_user_streak(self, user: User) -> Optional[int]:
        """
        Update user's sorting streak and award bonus if applicable
        
        Returns:
            Bonus points awarded for streak, or None
        """
        today = datetime.utcnow().date()
        
        if user.last_sorting_date:
            last_date = user.last_sorting_date.date()
            days_diff = (today - last_date).days
            
            if days_diff == 1:
                # Consecutive day
                user.current_streak_days += 1
                if user.current_streak_days > user.longest_streak_days:
                    user.longest_streak_days = user.current_streak_days
            elif days_diff == 0:
                # Same day, no change
                pass
            else:
                # Streak broken
                user.current_streak_days = 1
        else:
            # First sorting event
            user.current_streak_days = 1
        
        user.last_sorting_date = datetime.utcnow()
        
        # Award weekly streak bonus
        if user.current_streak_days % 7 == 0:
            return self.WEEKLY_STREAK_BONUS
        
        return None
    
    def update_user_tier(self, user: User) -> bool:
        """
        Check and update user's membership tier
        
        Returns:
            True if tier was upgraded, False otherwise
        """
        current_tier = user.membership_tier
        new_tier = current_tier
        
        for tier, threshold in sorted(
            self.TIER_THRESHOLDS.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if user.total_points >= threshold:
                new_tier = tier
                break
        
        if new_tier != current_tier:
            user.membership_tier = new_tier
            return True
        
        return False
    
    def check_achievements(self, user: User) -> List[Achievement]:
        """
        Check if user has earned any new achievements
        
        Returns:
            List of newly earned achievements
        """
        new_achievements = []
        
        # Get all active achievements
        all_achievements = self.db.query(Achievement).filter(
            Achievement.is_active == True
        ).all()
        
        # Get user's existing achievements
        existing_achievement_ids = {
            ua.achievement_id 
            for ua in user.achievements
        }
        
        for achievement in all_achievements:
            if achievement.id in existing_achievement_ids:
                continue
            
            # Check criteria
            earned = False
            
            if achievement.criteria_type == "total_points":
                if user.total_points >= achievement.criteria_value:
                    earned = True
            
            elif achievement.criteria_type == "streak":
                if user.longest_streak_days >= achievement.criteria_value:
                    earned = True
            
            elif achievement.criteria_type == "tier":
                tier_order = [
                    MembershipTier.BRONZE, 
                    MembershipTier.SILVER,
                    MembershipTier.GOLD, 
                    MembershipTier.PLATINUM
                ]
                if tier_order.index(user.membership_tier) >= achievement.criteria_value:
                    earned = True
            
            if earned:
                # Award achievement
                user_achievement = UserAchievement(
                    user_id=user.id,
                    achievement_id=achievement.id
                )
                self.db.add(user_achievement)
                
                # Award bonus points
                if achievement.bonus_points > 0:
                    user.total_points += achievement.bonus_points
                    user.available_points += achievement.bonus_points
                
                new_achievements.append(achievement)
        
        return new_achievements
    
    def apply_dynamic_rules(
        self, 
        user: User, 
        waste_type: WasteType, 
        base_points: int
    ) -> Tuple[int, List[str]]:
        """
        Apply any active dynamic incentive rules
        
        Returns:
            Tuple of (modified_points, list_of_applied_rules)
        """
        active_rules = self.db.query(IncentiveRule).filter(
            IncentiveRule.is_active == True,
            IncentiveRule.valid_from <= datetime.utcnow(),
            (IncentiveRule.valid_until.is_(None)) | 
            (IncentiveRule.valid_until >= datetime.utcnow())
        ).order_by(IncentiveRule.priority.desc()).all()
        
        modified_points = base_points
        applied_rules = []
        
        for rule in active_rules:
            # Check if rule applies to this waste type
            if rule.waste_type and rule.waste_type != waste_type:
                continue
            
            # Check neighborhood restriction
            if rule.applies_to_neighborhoods:
                neighborhoods = json.loads(rule.applies_to_neighborhoods)
                if user.neighborhood not in neighborhoods:
                    continue
            
            # Apply rule
            modified_points += rule.points_modifier
            modified_points = int(modified_points * rule.multiplier)
            applied_rules.append(rule.name)
        
        return modified_points, applied_rules


class RewardManager:
    """
    Manages reward redemptions
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_available_rewards(
        self, 
        user: User, 
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        Get rewards available to the user
        """
        from models import RewardCatalog
        
        query = self.db.query(RewardCatalog).filter(
            RewardCatalog.is_active == True,
            RewardCatalog.points_required <= user.available_points
        )
        
        if category:
            query = query.filter(RewardCatalog.category == category)
        
        # Filter by tier
        tier_order = {
            MembershipTier.BRONZE: 0,
            MembershipTier.SILVER: 1,
            MembershipTier.GOLD: 2,
            MembershipTier.PLATINUM: 3,
        }
        
        user_tier_level = tier_order[user.membership_tier]
        
        rewards = []
        for reward in query.all():
            reward_tier_level = tier_order[reward.min_tier_required]
            if user_tier_level >= reward_tier_level:
                rewards.append({
                    "id": reward.id,
                    "name": reward.name,
                    "description": reward.description,
                    "category": reward.category,
                    "points_required": reward.points_required,
                    "partner_name": reward.partner_name,
                })
        
        return rewards
    
    def redeem_reward(
        self, 
        user: User, 
        reward_id: int
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Process reward redemption
        
        Returns:
            Tuple of (success, message, voucher_code)
        """
        from models import RewardCatalog, Redemption
        import uuid
        
        reward = self.db.query(RewardCatalog).filter(
            RewardCatalog.id == reward_id,
            RewardCatalog.is_active == True
        ).first()
        
        if not reward:
            return False, "Reward not found or inactive", None
        
        # Check if user has enough points
        if user.available_points < reward.points_required:
            return False, "Insufficient points", None
        
        # Check tier requirement
        tier_order = {
            MembershipTier.BRONZE: 0,
            MembershipTier.SILVER: 1,
            MembershipTier.GOLD: 2,
            MembershipTier.PLATINUM: 3,
        }
        
        if tier_order[user.membership_tier] < tier_order[reward.min_tier_required]:
            return False, f"Requires {reward.min_tier_required.value} tier or higher", None
        
        # Check quantity
        if reward.quantity_available is not None and reward.quantity_available <= 0:
            return False, "Reward out of stock", None
        
        # Create redemption
        voucher_code = f"WM{uuid.uuid4().hex[:8].upper()}"
        
        redemption = Redemption(
            user_id=user.id,
            reward_id=reward.id,
            points_spent=reward.points_required,
            status="approved",
            voucher_code=voucher_code,
            redemption_instructions=reward.description,
            approved_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=90)
        )
        
        # Deduct points
        user.available_points -= reward.points_required
        
        # Update quantity
        if reward.quantity_available is not None:
            reward.quantity_available -= 1
        
        self.db.add(redemption)
        
        return True, "Reward redeemed successfully", voucher_code