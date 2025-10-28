# Waste Classification Relational Graph Network (RGN)

## Project Overview

This repository implements a **Graph-Reasoning Vision Model** for real-time waste classification, valuation, and incentive-based sorting in Uganda. The system uses a Relational Graph Network (RGN) to handle hierarchical classification, safety-critical rules, and conflict resolution.

## 🎯 Key Features

### 1. **Hierarchical Classification**
- **Material Level**: Identifies specific material types (plastic types, organic subtypes, etc.)
- **Category Level**: Classifies into main waste categories (plastic, organic, paper, glass, metal, electronic, medical)
- **Disposal Level**: Recommends appropriate disposal methods (recyclable, compostable, hazardous, etc.)

### 2. **Safety-Critical Classification**
- Prioritizes medical and hazardous waste identification
- Applies strict rules for high-risk waste handling
- Heavy penalty system for misclassification of safety-critical items
- Automatic override mechanisms for critical safety scenarios

### 3. **Conflict Resolution**
- Detects when multiple incompatible waste types are mixed
- Resolves conflicts using safety-priority rules
- Provides actionable recommendations for proper separation
- Negative point penalties for mixed waste

### 4. **Graph-Based Reasoning**
- Knowledge graph encodes relationships between materials, categories, and disposal methods
- Relational Graph Convolution (RGC) layers propagate information across the graph
- Attention mechanism weights the importance of different graph nodes
- Hierarchical consistency ensures predictions align across levels

### 5. **Incentive System**
- Point-based rewards for proper waste sorting
- Higher points for safety-critical waste handled correctly
- Penalties for low confidence or mixed waste
- Confidence-weighted multipliers

## 📁 Project Structure

```
├── waste_reasoning_rgn.py          # Core RGN model implementation
├── train_reasoning_model.py        # Training script with safety-aware loss
├── inference_engine.py             # Real-time classification engine
├── visualization_tools.py          # Graph and training visualizations
└── README.md                       # This file
```

## 🏗️ Architecture Components

### 1. **Knowledge Graph Structure**

```
Material Nodes (16 types)
    ├── plastic_pet, plastic_pvc, plastic_other
    ├── organic_food, organic_yard
    ├── paper_cardboard, paper_mixed
    ├── glass_clear, glass_colored
    ├── metal_aluminum, metal_steel
    ├── electronic_battery, electronic_circuit
    └── medical_sharp, medical_infectious, medical_pharmaceutical

Category Nodes (7 types)
    ├── plastic, organic, paper
    ├── glass, metal
    ├── electronic, medical

Disposal Nodes (5 types)
    ├── recyclable, compostable
    ├── landfill
    ├── hazardous_disposal
    └── specialized_facility
```

### 2. **Edge Relationships**

- **derives_from**: Material → Category (e.g., plastic_pet → plastic)
- **requires**: Category → Disposal (e.g., medical → hazardous_disposal)
- **conflicts_with**: Material ↔ Material (e.g., medical_sharp ↔ organic_food)

### 3. **Model Architecture**

```python
Vision Embedding (2048-dim)
    ↓
Vision Projection Layer (256-dim)
    ↓
Graph Node Embedding
    ↓
Relational Graph Convolution Layers (3 layers)
    ├── Material Classifier Head
    ├── Category Classifier Head
    ├── Disposal Classifier Head
    ├── Risk Predictor Head
    └── Confidence Estimator Head
```

## 🚀 Usage

### 1. **Training the Model**

```python
from train_reasoning_model import WasteReasoningTrainer, generate_synthetic_data
from waste_reasoning_rgn import create_waste_reasoning_model

# Generate training data (replace with real annotated waste images)
train_data = generate_synthetic_data(num_samples=800)
val_data = generate_synthetic_data(num_samples=200)

# Create datasets and loaders
train_dataset = WasteDataset(*train_data)
val_dataset = WasteDataset(*val_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Create model and trainer
model = create_waste_reasoning_model(vision_embedding_dim=2048)
trainer = WasteReasoningTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    safety_weight=10.0  # Higher weight for safety-critical errors
)

# Train
trainer.train(num_epochs=20, save_path='waste_reasoning_model.pt')
```

### 2. **Inference (Single Item)**

```python
from inference_engine import WasteClassificationAPI
import numpy as np

# Create API
api = WasteClassificationAPI('waste_reasoning_model.pt')

# Get vision embedding from your LVM (e.g., EfficientNet, ViT)
vision_embedding = extract_vision_features(image)  # Shape: (2048,)

# Classify
result = api.classify_from_camera(
    vision_embedding=vision_embedding,
    user_id="user_001",
    location="Kampala Central"
)

# Access results
print(f"Category: {result['category']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Points Earned: {result['points']}")
print(f"Message: {result['user_message']}")
```

### 3. **Inference (Multiple Items with Conflict Detection)**

```python
# Classify multiple items in a bin
vision_embeddings = [
    extract_vision_features(item1),
    extract_vision_features(item2),
    extract_vision_features(item3)
]

result = api.classify_bin_contents(
    vision_embeddings=vision_embeddings,
    user_id="user_001"
)

if result['conflict_detected']:
    print(f"Conflict: {result['conflict_reason']}")
    print(f"Action: {result['recommended_action']}")
    print(f"Points: {result['points']} (penalty)")
else:
    print(f"Great! Earned {result['combined_points']} points")
```

### 4. **Visualization**

```python
from visualization_tools import GraphVisualizer, create_visualizations

# Generate all visualizations
create_visualizations()

# This creates:
# - knowledge_graph.png: Full graph structure
# - conflict_matrix.png: Material conflicts
# - reasoning_path.png: Attention-based reasoning
# - training_curves.png: Training metrics
```

## 🔒 Safety Rules Implementation

### Rule 1: Medical Waste Priority
```python
# Medical waste MUST be classified as medical
# Cannot be mixed with any other waste type
# Requires hazardous disposal
```

### Rule 2: Electronic Battery Handling
```python
# Batteries MUST NOT go to landfill
# Fire hazard if damaged
# Requires specialized recycling facility
```

### Rule 3: Conflict Resolution Priority
```python
Priority Order:
1. CRITICAL risk (medical_sharp, medical_infectious)
2. HIGH_RISK (electronic_battery)
3. MEDIUM_RISK
4. LOW_RISK
5. SAFE
```

## 📊 Loss Functions

### 1. **Safety-Aware Loss**
```python
Total Loss = Material Loss + Category Loss + Disposal Loss + Risk Loss
           + Safety Penalty + Hierarchical Consistency Loss

Safety Penalty = 10.0 × (misclassified high-risk items)
```

### 2. **Hierarchical Consistency**
- Ensures material predictions align with category predictions
- Uses KL-divergence between expected and predicted distributions

## 🎯 Evaluation Metrics

- **Overall Accuracy**: Classification accuracy across all categories
- **Safety-Critical Accuracy**: Accuracy for high-risk and critical items (most important)
- **Hierarchical Consistency**: Alignment between hierarchy levels
- **Conflict Detection Rate**: Percentage of conflicts correctly identified
- **Safety Violation Count**: Number of safety-critical misclassifications

## 📱 Mobile Integration

### Requirements
- Vision embedding extraction: EfficientNetV2, MobileNetV3, or ViT
- Model compression: TensorFlow Lite or ONNX Runtime Mobile
- Target model size: <50MB for offline deployment
- Inference time: <500ms on mid-range smartphones

### Deployment Pipeline
```
1. Extract vision features from LVM (on-device or cloud)
2. Run RGN reasoning (on-device)
3. Apply safety rules
4. Calculate points
5. Store in local database
6. Sync with server (federated learning)
```

## 🔄 Federated Learning

The system supports continuous improvement through federated learning:

```python
# Client-side (on device)
1. User provides feedback on classification
2. Model fine-tunes locally
3. Compute weight updates

# Server-side
1. Aggregate updates from multiple devices
2. Apply differential privacy
3. Update global model
4. Distribute new weights
```

## 📈 Point System

| Action | Points |
|--------|--------|
| Recyclable waste (proper) | +5 |
| Compostable waste | +4 |
| Hazardous disposal | +10 |
| Safety-critical (proper) | +15 |
| Low confidence | -2 |
| Mixed waste | -10 |

**Confidence Multiplier**: Points × min(confidence × 1.5, 1.5)

## 🌍 Uganda Context

### Targeted Waste Categories
- **Plastics**: PET bottles, HDPE containers, shopping bags
- **Organic**: Food waste, yard waste
- **Paper/Cardboard**: Packaging materials
- **Medical**: Syringes, gloves, masks (healthcare facilities)
- **Electronic**: Batteries, small devices
- **Glass/Metal**: Bottles, cans

### Use Cases
1. **Households**: Incentivize proper sorting at home
2. **Healthcare Facilities**: Ensure safe medical waste handling
3. **Local Businesses**: Track and reward waste management
4. **KCCA Integration**: Optimize collection routes and resources

## 🛠️ Dependencies

```bash
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
networkx>=3.1
tqdm>=4.65.0

# For mobile deployment
tensorflow-lite>=2.13.0  # or
onnxruntime>=1.15.0

# For vision models
torchvision>=0.15.0
timm>=0.9.0  # For EfficientNet, ViT
```

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{waste_rgn_2025,
  title={Graph-Reasoning Vision Models for Real-Time Waste Valuation and Incentive-Based Sorting},
  author={Your Name},
  journal={Waste Management AI},
  year={2025}
}
```

## 📄 License

This project is intended for research and development purposes for waste management in Uganda.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional waste categories specific to Uganda
- Enhanced conflict detection rules
- Mobile app integration examples
- Real-world dataset annotations
- Performance optimizations

## 📧 Contact

For questions or collaboration opportunities, please contact [your contact information].

---

**Note**: This system is designed specifically for Uganda's waste management context but can be adapted for other regions by modifying the knowledge graph structure and safety rules.