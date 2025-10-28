# 🎯 Waste Management Incentivization System

## Overview
This system integrates your MobileViT waste classification model with an AI-powered incentivization engine using Anthropic's Claude API. Users earn points for correctly sorting waste, with intelligent bonus calculations based on accuracy, difficulty, and behavioral patterns.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Anthropic API Key
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create an account and get your API key
3. Update `waste_management_config.json` with your API key

### 3. Initialize Database
```bash
python initialize_database.py
```

### 4. Run Demo
```bash
python waste_incentive_integration.py
```

## 📁 Project Structure

```
DATASETS/
├── models.py                           # Database schema
├── incentive_engine.py                 # Core incentivization logic
├── waste_incentive_integration.py      # Complete integration demo
├── initialize_database.py              # Database setup script
├── requirements.txt                    # Python dependencies
├── waste_management_config.json        # System configuration
└── best_mobilevit_waste_model.pth      # Your trained model
```

## 🔧 System Components

### 1. **Database Models** (`models.py`)
- User management with gamification features
- Sorting event tracking
- Achievement system
- Reward catalog and redemptions
- Leaderboards and notifications

### 2. **Incentive Engine** (`incentive_engine.py`)
- **IncentiveEngine**: AI-powered point calculation using Claude
- **RewardManager**: Handles reward redemption and voucher generation
- Dynamic rule application and tier management
- Streak tracking and achievement detection

### 3. **Integration System** (`waste_incentive_integration.py`)
- **WasteClassificationIncentiveSystem**: Complete workflow integration
- MobileViT classification + validation
- Real-time incentive calculation
- User dashboard and analytics

## 🎮 Key Features

### **AI-Powered Point Calculation**
- Uses Anthropic Claude for intelligent bonus reasoning
- Considers multiple factors: accuracy, difficulty, user history
- Fallback calculation system for API failures
- Dynamic rule application based on user context

### **Gamification Elements**
- **Points System**: Base points + AI-calculated bonuses
- **Tier System**: Bronze → Silver → Gold → Platinum
- **Achievements**: 16 different achievement types
- **Streaks**: Daily sorting streak bonuses
- **Rewards**: Digital badges, physical items, experiences, services

### **Comprehensive Tracking**
- Sorting accuracy and confidence scores
- Contamination and hazardous material detection
- Weight-based volume tracking
- Temporal pattern analysis
- Neighborhood comparisons

## 📊 Usage Examples

### **Basic Sorting Event**
```python
# Initialize system
system = WasteClassificationIncentiveSystem(
    mobilevit_model_path="best_mobilevit_waste_model.pth",
    anthropic_api_key="your-api-key-here"
)

# Process waste sorting
result = system.classify_and_incentivize(
    image_path="plastic_bottle.jpg",
    user_id=1,
    quantity_kg=0.5
)

print(f"Points earned: {result['incentives']['points_earned']}")
print(f"Classification: {result['classification']['predicted_class']}")
print(f"AI Feedback: {result['incentives']['feedback_message']}")
```

### **User Dashboard**
```python
dashboard = system.get_user_dashboard(user_id=1)
print(f"Total Points: {dashboard['user']['total_points']}")
print(f"Current Tier: {dashboard['user']['membership_tier']}")
print(f"Achievements: {len(dashboard['achievements'])}")
```

## 🎁 Reward System

### **Digital Rewards**
- Environmental badges and certificates
- Profile avatars and frames
- Achievement showcases

### **Physical Rewards**
- Eco-friendly water bottles and tote bags
- Solar chargers and compost bins
- Sustainable living products

### **Experience Rewards**
- Waste management facility tours
- Environmental workshops
- Tree planting events

### **Service Credits**
- Free waste collection services
- Recycling center credits
- Bulk waste disposal

### **Vouchers**
- Eco-store discounts
- Public transport credits
- Farmers market vouchers

## 🏆 Achievement System

### **Categories**
- **Milestone**: First sort, 100 sorts, 1000 sorts
- **Accuracy**: 95% accuracy over 50 sorts
- **Streak**: 7-day, 30-day, 100-day streaks
- **Volume**: 10kg, 100kg, 1000kg total
- **Category**: Plastic specialist, Organic expert
- **Special**: Contamination finder, Safety expert
- **Social**: Community helper, Neighborhood leader

## 🔄 Integration with Existing Models

### **GNN Model Integration**
- Uses graph reasoning results for bonus calculations
- Spatial correlation and temporal pattern analysis
- Neighborhood consistency scoring
- Volume prediction integration

### **MobileViT Model Integration**
- Real-time waste classification
- Confidence score analysis
- Secondary prediction handling
- Uncertainty quantification

## 📈 Performance Features

### **AI Reasoning** (via Claude)
- Contextual bonus calculation
- Personalized feedback generation
- Dynamic rule interpretation
- Behavioral pattern analysis

### **Fallback Systems**
- Local calculation when API unavailable
- Rule-based backup logic
- Error handling and recovery
- Performance monitoring

## 🔧 Configuration

### **Database Settings**
```json
{
  "database": {
    "url": "sqlite:///waste_management.db",
    "pool_size": 10
  }
}
```

### **Point System**
```json
{
  "points": {
    "base_points": 10,
    "max_multiplier": 3.0,
    "achievement_bonus_cap": 500
  }
}
```

### **Tier Thresholds**
```json
{
  "tiers": {
    "bronze_threshold": 1000,
    "silver_threshold": 5000,
    "gold_threshold": 15000,
    "platinum_threshold": 50000
  }
}
```

## 🚀 Next Steps

1. **API Development**: Create REST API for mobile/web apps
2. **Real-time Dashboard**: Build user interface for monitoring
3. **Mobile Integration**: Connect with mobile app for image capture
4. **Analytics**: Advanced reporting and insights
5. **Social Features**: Community challenges and leaderboards

## 📞 Integration with KCCA

This system is designed to work with:
- **Kampala's 5 divisions**: Central, Kawempe, Rubaga, Makindye, Nakawa
- **KCCA data**: Real waste volume predictions and patterns
- **Local context**: Uganda-specific rewards and services
- **Community engagement**: Neighborhood-based competitions

## 🔐 Security & Privacy

- User data encryption
- Secure API communication
- GDPR-compliant data handling
- Anonymous analytics options

---

**Ready to revolutionize waste management in Kampala with AI-powered incentivization!** 🌍♻️