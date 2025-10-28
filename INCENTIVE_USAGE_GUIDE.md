# 🎯 Simple Incentive System Usage Guide

## Overview
Your incentivization system is now simplified and ready to use! No database required - just pure calculation logic with optional AI-enhanced feedback.

## 📁 Files Created

### 1. **`simple_incentive_engine.py`** 
Complete incentive system with user profiles and achievements (no database)

### 2. **`mobilevit_incentive_demo.py`**
Integration demo showing how to use with your MobileViT model

### 3. **`standalone_incentive_calculator.py`** ⭐ **RECOMMENDED**
Lightweight, focused calculator - perfect for integration

## 🚀 Quick Start (Recommended)

Use the **standalone calculator** for easy integration:

```python
from standalone_incentive_calculator import (
    StandaloneIncentiveCalculator, 
    WasteSortingEvent, 
    IncentiveConfig
)

# Initialize calculator
calculator = StandaloneIncentiveCalculator()

# Create a sorting event
event = WasteSortingEvent(
    waste_type="PLASTIC",
    confidence=0.92,           # Your MobileViT confidence
    is_correctly_sorted=True,  # Based on confidence threshold
    quantity_kg=1.5,          # Estimated weight
    user_streak_days=3        # User's current streak
)

# Calculate incentives
result = calculator.calculate_incentives(event)

print(f"Points earned: {result.final_points}")
print(f"Feedback: {result.feedback_message}")
```

## 🔧 Integration with Your MobileViT Model

```python
# Your existing MobileViT inference
def classify_waste(image_path):
    # Your MobileViT code here
    predicted_class = "Plastic"  # From your model
    confidence = 0.89           # From your model
    return predicted_class, confidence

# Add incentive calculation
def process_with_incentives(image_path, user_streak=0):
    # Classify waste
    predicted_class, confidence = classify_waste(image_path)
    
    # Create sorting event
    event = WasteSortingEvent(
        waste_type=map_to_waste_type(predicted_class),
        confidence=confidence,
        is_correctly_sorted=confidence > 0.8,  # Your threshold
        quantity_kg=estimate_weight(image_path),  # Your estimation
        user_streak_days=user_streak
    )
    
    # Calculate incentives
    calculator = StandaloneIncentiveCalculator()
    result = calculator.calculate_incentives(event)
    
    return {
        "classification": predicted_class,
        "confidence": confidence,
        "points": result.final_points,
        "feedback": result.feedback_message
    }
```

## 🎮 Incentive Rules

### **Base Points**: 10 points per sort

### **Accuracy Bonuses**:
- ✅ Correct sorting: +10 points
- 🎯 High confidence (>90%): +5 points, 1.2x multiplier
- 🎯 Perfect confidence (>95%): +15 points, 1.5x multiplier

### **Volume Bonuses**:
- 📦 Large item (>2kg): +20 points, 1.3x multiplier
- 📦 Bulk item (>5kg): +50 points, 1.8x multiplier

### **Special Bonuses**:
- 🔍 Contamination detection: +25 points
- ⚠️ Hazardous material: +100 points, 2.0x multiplier

### **Streak Multipliers**:
- 🔥 3+ day streak: 1.1x multiplier
- 🔥 7+ day streak: 1.5x multiplier

### **Maximum multiplier**: 3.0x (combined)

## 📊 Example Results

```
Event: PLASTIC, 95% confidence, 0.5kg, 5-day streak
→ Result: 57 points (Base:10 + Bonuses:25) × 1.65x
→ Feedback: "Perfect classification confidence! Nice 5-day streak!"

Event: HAZARDOUS, 87% confidence, 1.2kg, 5-day streak  
→ Result: 264 points (Base:10 + Bonuses:110) × 2.2x
→ Feedback: "Thanks for safely handling hazardous materials!"

Event: ORGANIC, 82% confidence, 3.5kg, contaminated, 6-day streak
→ Result: 92 points (Base:10 + Bonuses:55) × 1.43x
→ Feedback: "Great contamination detection! Thanks for large items!"
```

## 🤖 AI-Enhanced Feedback (Optional)

To enable AI-powered personalized feedback:

1. Get Anthropic API key from [console.anthropic.com](https://console.anthropic.com/)
2. Initialize with your API key:

```python
calculator = StandaloneIncentiveCalculator(
    anthropic_api_key="your-actual-api-key"
)
```

The system automatically falls back to template feedback if AI is unavailable.

## ⚙️ Customization

Adjust the incentive rules by modifying the config:

```python
from standalone_incentive_calculator import IncentiveConfig

custom_config = IncentiveConfig(
    base_points=15,                    # Increase base points
    perfect_confidence_threshold=0.98, # Stricter perfect threshold
    large_item_threshold=1.5,         # Lower threshold for large items
    max_multiplier=4.0                # Higher max multiplier
)

calculator = StandaloneIncentiveCalculator(config=custom_config)
```

## 🔄 Batch Processing

Process multiple events efficiently:

```python
events = [
    WasteSortingEvent(...),
    WasteSortingEvent(...),
    WasteSortingEvent(...)
]

results = calculator.calculate_batch(events)
summary = calculator.get_summary_statistics(results)

print(f"Total points: {summary['total_points']}")
print(f"Average points: {summary['average_points']:.1f}")
```

## 📤 Export/Import

The system works with standard Python data structures and JSON:

```python
import json

# Export calculation
calculation_data = asdict(result)
json_export = json.dumps(calculation_data, indent=2)

# Save to file
with open("incentive_results.json", "w") as f:
    json.dump(calculation_data, f, indent=2)
```

## 🎯 Next Steps

1. **Basic Integration**: Start with `standalone_incentive_calculator.py`
2. **Add to Pipeline**: Integrate after your MobileViT classification
3. **Customize Rules**: Adjust `IncentiveConfig` for your needs
4. **Add AI Feedback**: Get Anthropic API key for enhanced messages
5. **Scale Up**: Use batch processing for multiple images

## ✅ Benefits

- ✅ **No Database**: Pure calculation logic, no setup required
- ✅ **Lightweight**: Minimal dependencies (just `anthropic` for AI)
- ✅ **Flexible**: Easy to customize rules and thresholds
- ✅ **Reliable**: Automatic fallback if AI service unavailable
- ✅ **Scalable**: Efficient batch processing
- ✅ **Portable**: Works anywhere Python runs

---

**Your incentive system is ready to motivate waste sorters in Kampala!** 🌍♻️

Start with the standalone calculator and integrate it step by step into your existing workflow.