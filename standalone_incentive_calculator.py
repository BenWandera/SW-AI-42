"""
Standalone Incentive Calculator
Simple, lightweight incentive calculation for waste sorting
No database dependencies - pure calculation logic
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import anthropic


@dataclass
class IncentiveConfig:
    """Configuration for incentive calculations"""
    base_points: int = 10
    max_multiplier: float = 3.0
    
    # Accuracy bonuses
    high_confidence_threshold: float = 0.9
    perfect_confidence_threshold: float = 0.95
    high_confidence_bonus: int = 5
    perfect_confidence_bonus: int = 15
    high_confidence_multiplier: float = 1.2
    perfect_confidence_multiplier: float = 1.5
    
    # Volume bonuses
    large_item_threshold: float = 2.0
    bulk_item_threshold: float = 5.0
    large_item_bonus: int = 20
    bulk_item_bonus: int = 50
    large_item_multiplier: float = 1.3
    bulk_item_multiplier: float = 1.8
    
    # Special condition bonuses
    correct_sorting_bonus: int = 10
    contamination_bonus: int = 25
    hazardous_bonus: int = 100
    hazardous_multiplier: float = 2.0
    
    # Streak multipliers
    short_streak_threshold: int = 3
    long_streak_threshold: int = 7
    short_streak_multiplier: float = 1.1
    long_streak_multiplier: float = 1.5


@dataclass
class WasteSortingEvent:
    """Details of a waste sorting event"""
    waste_type: str
    confidence: float
    is_correctly_sorted: bool
    quantity_kg: float = 1.0
    is_contaminated: bool = False
    is_hazardous: bool = False
    user_streak_days: int = 0
    user_total_sorts: int = 0


@dataclass
class IncentiveCalculation:
    """Result of incentive calculation"""
    base_points: int
    bonus_points: int
    total_points: int
    multiplier: float
    final_points: int
    calculation_details: Dict
    feedback_message: str


class StandaloneIncentiveCalculator:
    """
    Standalone incentive calculator with no external dependencies
    """
    
    def __init__(self, config: IncentiveConfig = None, anthropic_api_key: str = None):
        """
        Initialize calculator
        
        Args:
            config: Configuration object (uses defaults if None)
            anthropic_api_key: Optional API key for AI-enhanced feedback
        """
        self.config = config or IncentiveConfig()
        
        # Setup Anthropic client if API key provided
        self.anthropic_client = None
        if anthropic_api_key and anthropic_api_key != "your-api-key-here":
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                print("🤖 AI-enhanced feedback enabled")
            except Exception as e:
                print(f"⚠️  AI client setup failed: {e}")
                self.anthropic_client = None
        
        print(f"🎯 Standalone Incentive Calculator ready")
    
    def calculate_incentives(self, event: WasteSortingEvent) -> IncentiveCalculation:
        """
        Calculate incentives for a waste sorting event
        
        Args:
            event: Waste sorting event details
            
        Returns:
            IncentiveCalculation with complete breakdown
        """
        
        # Start with base points
        base_points = self.config.base_points
        bonus_points = 0
        multiplier = 1.0
        details = {
            "bonuses_applied": [],
            "multipliers_applied": []
        }
        
        # Correct sorting bonus
        if event.is_correctly_sorted:
            bonus_points += self.config.correct_sorting_bonus
            details["bonuses_applied"].append(f"Correct sorting: +{self.config.correct_sorting_bonus}")
        
        # Confidence-based bonuses
        if event.confidence >= self.config.perfect_confidence_threshold:
            bonus_points += self.config.perfect_confidence_bonus
            multiplier *= self.config.perfect_confidence_multiplier
            details["bonuses_applied"].append(f"Perfect confidence: +{self.config.perfect_confidence_bonus}")
            details["multipliers_applied"].append(f"Perfect confidence: {self.config.perfect_confidence_multiplier}x")
            
        elif event.confidence >= self.config.high_confidence_threshold:
            bonus_points += self.config.high_confidence_bonus
            multiplier *= self.config.high_confidence_multiplier
            details["bonuses_applied"].append(f"High confidence: +{self.config.high_confidence_bonus}")
            details["multipliers_applied"].append(f"High confidence: {self.config.high_confidence_multiplier}x")
        
        # Volume-based bonuses
        if event.quantity_kg >= self.config.bulk_item_threshold:
            bonus_points += self.config.bulk_item_bonus
            multiplier *= self.config.bulk_item_multiplier
            details["bonuses_applied"].append(f"Bulk item ({event.quantity_kg:.1f}kg): +{self.config.bulk_item_bonus}")
            details["multipliers_applied"].append(f"Bulk processing: {self.config.bulk_item_multiplier}x")
            
        elif event.quantity_kg >= self.config.large_item_threshold:
            bonus_points += self.config.large_item_bonus
            multiplier *= self.config.large_item_multiplier
            details["bonuses_applied"].append(f"Large item ({event.quantity_kg:.1f}kg): +{self.config.large_item_bonus}")
            details["multipliers_applied"].append(f"Large item: {self.config.large_item_multiplier}x")
        
        # Special condition bonuses
        if event.is_contaminated:
            bonus_points += self.config.contamination_bonus
            details["bonuses_applied"].append(f"Contamination detection: +{self.config.contamination_bonus}")
        
        if event.is_hazardous:
            bonus_points += self.config.hazardous_bonus
            multiplier *= self.config.hazardous_multiplier
            details["bonuses_applied"].append(f"Hazardous material: +{self.config.hazardous_bonus}")
            details["multipliers_applied"].append(f"Hazardous handling: {self.config.hazardous_multiplier}x")
        
        # Streak bonuses
        if event.user_streak_days >= self.config.long_streak_threshold:
            multiplier *= self.config.long_streak_multiplier
            details["multipliers_applied"].append(f"Long streak ({event.user_streak_days} days): {self.config.long_streak_multiplier}x")
            
        elif event.user_streak_days >= self.config.short_streak_threshold:
            multiplier *= self.config.short_streak_multiplier
            details["multipliers_applied"].append(f"Streak ({event.user_streak_days} days): {self.config.short_streak_multiplier}x")
        
        # Apply multiplier cap
        multiplier = min(multiplier, self.config.max_multiplier)
        if multiplier == self.config.max_multiplier:
            details["multipliers_applied"].append(f"Capped at maximum: {self.config.max_multiplier}x")
        
        # Calculate final points
        total_points = base_points + bonus_points
        final_points = int(total_points * multiplier)
        
        # Generate feedback message
        feedback = self._generate_feedback(event, final_points, details)
        
        return IncentiveCalculation(
            base_points=base_points,
            bonus_points=bonus_points,
            total_points=total_points,
            multiplier=multiplier,
            final_points=final_points,
            calculation_details=details,
            feedback_message=feedback
        )
    
    def _generate_feedback(self, event: WasteSortingEvent, points: int, details: Dict) -> str:
        """Generate feedback message"""
        
        # Try AI-enhanced feedback first
        if self.anthropic_client:
            try:
                return self._generate_ai_feedback(event, points, details)
            except Exception as e:
                print(f"⚠️  AI feedback failed: {e}")
        
        # Fallback to template-based feedback
        return self._generate_template_feedback(event, points, details)
    
    def _generate_ai_feedback(self, event: WasteSortingEvent, points: int, details: Dict) -> str:
        """Generate AI-enhanced feedback using Anthropic"""
        
        prompt = f"""
Generate encouraging, personalized feedback for a waste sorting event.

Event Details:
- Waste Type: {event.waste_type}
- Confidence: {event.confidence:.1%}
- Correctly Sorted: {event.is_correctly_sorted}
- Quantity: {event.quantity_kg:.1f}kg
- Contaminated: {event.is_contaminated}
- Hazardous: {event.is_hazardous}
- User Streak: {event.user_streak_days} days
- Total Sorts: {event.user_total_sorts}

Points Earned: {points}
Bonuses: {', '.join(details['bonuses_applied']) if details['bonuses_applied'] else 'None'}
Multipliers: {', '.join(details['multipliers_applied']) if details['multipliers_applied'] else 'None'}

Generate a brief (2-3 sentences), encouraging message that:
1. Congratulates the user on their points
2. Highlights their best achievement in this sort
3. Provides motivation for improvement or continued excellence

Be specific, positive, and actionable. Use emojis sparingly.
"""
        
        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    
    def _generate_template_feedback(self, event: WasteSortingEvent, points: int, details: Dict) -> str:
        """Generate template-based feedback"""
        
        messages = [f"Great job! You earned {points} points for sorting {event.waste_type}."]
        
        # Confidence feedback
        if event.confidence >= 0.95:
            messages.append("Perfect classification confidence! 🎯")
        elif event.confidence >= 0.9:
            messages.append("Excellent accuracy!")
        elif event.confidence >= 0.8:
            messages.append("Good sorting skills!")
        
        # Special achievements
        if event.is_hazardous:
            messages.append("Thanks for safely handling hazardous materials! ⚠️")
        elif event.is_contaminated:
            messages.append("Great contamination detection! 🔍")
        
        # Streak encouragement
        if event.user_streak_days >= 7:
            messages.append(f"Amazing {event.user_streak_days}-day streak! 🔥")
        elif event.user_streak_days >= 3:
            messages.append(f"Nice {event.user_streak_days}-day streak going!")
        
        # Volume recognition
        if event.quantity_kg >= 5.0:
            messages.append("Impressive bulk processing! 📦")
        elif event.quantity_kg >= 2.0:
            messages.append("Thanks for handling the large items!")
        
        return " ".join(messages)
    
    def calculate_batch(self, events: List[WasteSortingEvent]) -> List[IncentiveCalculation]:
        """Calculate incentives for multiple events"""
        
        results = []
        for i, event in enumerate(events):
            print(f"Processing event {i+1}/{len(events)}: {event.waste_type}")
            result = self.calculate_incentives(event)
            results.append(result)
        
        return results
    
    def get_summary_statistics(self, calculations: List[IncentiveCalculation]) -> Dict:
        """Get summary statistics for a batch of calculations"""
        
        if not calculations:
            return {}
        
        total_points = sum(calc.final_points for calc in calculations)
        avg_points = total_points / len(calculations)
        max_points = max(calc.final_points for calc in calculations)
        min_points = min(calc.final_points for calc in calculations)
        
        avg_multiplier = sum(calc.multiplier for calc in calculations) / len(calculations)
        
        return {
            "total_events": len(calculations),
            "total_points": total_points,
            "average_points": avg_points,
            "max_points": max_points,
            "min_points": min_points,
            "average_multiplier": avg_multiplier
        }


def demo_standalone_calculator():
    """Demo of standalone incentive calculator"""
    
    print("🎯 Standalone Incentive Calculator Demo")
    print("=" * 45)
    
    # Initialize calculator
    calculator = StandaloneIncentiveCalculator(
        anthropic_api_key="your-api-key-here"  # Replace with real key for AI feedback
    )
    
    # Demo events
    events = [
        WasteSortingEvent(
            waste_type="PLASTIC",
            confidence=0.95,
            is_correctly_sorted=True,
            quantity_kg=0.5,
            user_streak_days=5,
            user_total_sorts=23
        ),
        WasteSortingEvent(
            waste_type="HAZARDOUS",
            confidence=0.87,
            is_correctly_sorted=True,
            quantity_kg=1.2,
            is_hazardous=True,
            user_streak_days=5,
            user_total_sorts=24
        ),
        WasteSortingEvent(
            waste_type="ORGANIC",
            confidence=0.82,
            is_correctly_sorted=True,
            quantity_kg=3.5,
            is_contaminated=True,
            user_streak_days=6,
            user_total_sorts=25
        )
    ]
    
    # Process events
    results = []
    for i, event in enumerate(events):
        print(f"\n--- Event {i+1}: {event.waste_type} ---")
        print(f"Confidence: {event.confidence:.1%}")
        print(f"Quantity: {event.quantity_kg}kg")
        print(f"Streak: {event.user_streak_days} days")
        
        calculation = calculator.calculate_incentives(event)
        results.append(calculation)
        
        print(f"\nResults:")
        print(f"  Base Points: {calculation.base_points}")
        print(f"  Bonus Points: {calculation.bonus_points}")
        print(f"  Multiplier: {calculation.multiplier:.1f}x")
        print(f"  Final Points: {calculation.final_points}")
        print(f"  Bonuses: {', '.join(calculation.calculation_details['bonuses_applied'])}")
        print(f"  Feedback: {calculation.feedback_message}")
    
    # Summary statistics
    summary = calculator.get_summary_statistics(results)
    print(f"\n📊 Summary Statistics:")
    print(f"  Total Events: {summary['total_events']}")
    print(f"  Total Points: {summary['total_points']}")
    print(f"  Average Points: {summary['average_points']:.1f}")
    print(f"  Average Multiplier: {summary['average_multiplier']:.1f}x")
    
    print(f"\n✅ Demo completed!")
    
    # Export example
    print(f"\n💾 Example JSON Export:")
    example_export = {
        "events": [asdict(event) for event in events],
        "calculations": [asdict(calc) for calc in results],
        "summary": summary
    }
    print(json.dumps(example_export, indent=2))


if __name__ == "__main__":
    demo_standalone_calculator()