"""
MobileViT-Small Architecture Explanation
A detailed breakdown of the model architecture used for waste classification
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def explain_mobilevit_architecture():
    """
    Comprehensive explanation of MobileViT-Small architecture for waste classification
    """
    
    print("🏗️  MobileViT-Small Architecture for Waste Classification")
    print("=" * 70)
    
    print("""
📱 MobileViT (Mobile Vision Transformer) Overview:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MobileViT combines the best of CNNs and Vision Transformers:
• 🔄 CNNs for local spatial processing (early layers)
• 🎯 Transformers for global context understanding (later layers)
• ⚡ Mobile-optimized design for efficiency
• 🎨 Hierarchical feature extraction

Key Innovation: MobileViT blocks that seamlessly integrate convolutions 
with self-attention for both local and global feature learning.
""")

    # Architecture Components
    print("\n🧱 Architecture Components:")
    print("━" * 40)
    
    components = [
        ("1️⃣ Input Layer", "256×256×3 RGB images → Preprocessed tensors"),
        ("2️⃣ Stem Block", "Initial convolution + normalization"),
        ("3️⃣ MV2 Blocks", "MobileNetV2-style inverted residual blocks"),
        ("4️⃣ MobileViT Blocks", "Hybrid CNN-Transformer blocks"),
        ("5️⃣ Global Pool", "Spatial feature aggregation"),
        ("6️⃣ Custom Classifier", "Waste-specific classification head")
    ]
    
    for component, description in components:
        print(f"   {component:<20} {description}")
    
    print(f"\n📊 Model Statistics:")
    print(f"   • Total Parameters: 5,401,001 (~5.4M)")
    print(f"   • Trainable Parameters: 463,369 (~463K)")
    print(f"   • Model Size: ~21 MB")
    print(f"   • Input Resolution: 256×256")
    print(f"   • Output Classes: 9 waste categories")
    
    return components

def detailed_architecture_breakdown():
    """
    Detailed layer-by-layer breakdown
    """
    
    print("\n🔍 Detailed Architecture Breakdown:")
    print("━" * 50)
    
    layers = [
        {
            "name": "Input Processing",
            "details": [
                "Input: 256×256×3 RGB images",
                "Preprocessing: Normalization (ImageNet stats)",
                "Data augmentation: Random flips, rotations, color jitter"
            ]
        },
        {
            "name": "Stem Block",
            "details": [
                "Conv2d: 3→16 channels, 3×3 kernel, stride=2",
                "BatchNorm + SiLU activation",
                "Output: 128×128×16"
            ]
        },
        {
            "name": "MobileNetV2 Blocks (Stage 1)",
            "details": [
                "Block 1: 16→32 channels (depthwise separable)",
                "Block 2: 32→64 channels", 
                "Inverted residual structure with expansion",
                "Output: 32×32×64"
            ]
        },
        {
            "name": "MobileViT Block 1",
            "details": [
                "Input: 32×32×64",
                "Local processing: 3×3 convolutions",
                "Global processing: Multi-head self-attention",
                "Patch size: 2×2, Transformer dim: 144",
                "Output: 16×16×96"
            ]
        },
        {
            "name": "MobileViT Block 2", 
            "details": [
                "Input: 16×16×96",
                "Enhanced transformer processing",
                "Patch size: 2×2, Transformer dim: 192",
                "Output: 8×8×128"
            ]
        },
        {
            "name": "MobileViT Block 3",
            "details": [
                "Input: 8×8×128",
                "Final feature extraction",
                "Patch size: 2×2, Transformer dim: 240", 
                "Output: 4×4×160"
            ]
        },
        {
            "name": "Global Average Pooling",
            "details": [
                "Input: 4×4×160",
                "Spatial pooling: (4×4) → (1×1)",
                "Output: 640-dimensional feature vector"
            ]
        },
        {
            "name": "Custom Classifier Head",
            "details": [
                "Linear 1: 640 → 512 (+ BatchNorm + ReLU + Dropout)",
                "Linear 2: 512 → 256 (+ BatchNorm + ReLU + Dropout)", 
                "Linear 3: 256 → 9 (waste classes)",
                "Activation: Softmax for probabilities"
            ]
        }
    ]
    
    for i, layer in enumerate(layers, 1):
        print(f"\n{i}. {layer['name']}")
        print("   " + "─" * 30)
        for detail in layer['details']:
            print(f"   • {detail}")

def mobilevit_block_explanation():
    """
    Detailed explanation of the key MobileViT block
    """
    
    print("\n🎯 MobileViT Block - The Core Innovation:")
    print("━" * 50)
    
    print("""
The MobileViT block is where the magic happens! It combines:

🔄 Local Processing (CNN part):
   ┌─────────────────────────────────────┐
   │  Input Feature Map                  │
   │         ↓                           │
   │  3×3 Convolution (local features)   │
   │         ↓                           │
   │  1×1 Convolution (channel mixing)   │
   │         ↓                           │
   └─────────────────────────────────────┘

🌍 Global Processing (Transformer part):  
   ┌─────────────────────────────────────┐
   │  Unfold to patches (e.g., 2×2)      │
   │         ↓                           │
   │  Multi-Head Self-Attention          │
   │         ↓                           │
   │  Feed-Forward Network               │
   │         ↓                           │
   │  Fold back to feature map           │
   └─────────────────────────────────────┘

🔗 Fusion:
   ┌─────────────────────────────────────┐
   │  Concatenate local + global         │
   │         ↓                           │
   │  1×1 Convolution (feature fusion)   │
   │         ↓                           │
   │  Output Feature Map                 │
   └─────────────────────────────────────┘
""")

def waste_classification_adaptation():
    """
    Explain how we adapted MobileViT for waste classification
    """
    
    print("\n🗂️  Adaptation for Waste Classification:")
    print("━" * 50)
    
    print("""
Original MobileViT → Waste Classification MobileViT:

1️⃣ Backbone Freezing:
   • Freeze pre-trained MobileViT weights (5M parameters)
   • Only train the classification head (463K parameters)
   • Reduces training time and prevents overfitting

2️⃣ Custom Classification Head:
   Original: 640 → 1000 (ImageNet classes)
   Our Model: 640 → 512 → 256 → 9 (waste classes)
   
   Why 3 layers?
   • 640→512: Initial dimensionality reduction
   • 512→256: Further feature refinement  
   • 256→9: Final waste type classification

3️⃣ Waste-Specific Optimizations:
   • Dropout (0.3, 0.4, 0.5): Prevent overfitting on waste data
   • BatchNorm: Stable training with different waste textures
   • ReLU: Non-linear feature learning
   • Label Smoothing: Handle similar waste categories

4️⃣ Training Strategy:
   • Learning Rate: 0.001 (backbone) vs 0.0001 (classifier)
   • Data Augmentation: Rotations, flips, color changes
   • Early Stopping: Prevent overfitting
   • Cosine Annealing: Smooth learning rate decay
""")

def training_process_explanation():
    """
    Explain the training process
    """
    
    print("\n🚀 Training Process:")
    print("━" * 30)
    
    print("""
Dataset Preparation (70/20/10 split):
• Training: 6,652 images → Learn waste patterns
• Validation: 1,902 images → Monitor performance  
• Testing: 950 images → Final evaluation

Training Loop:
1. Load batch of waste images (8 images per batch)
2. Apply data augmentation (random transforms)
3. Forward pass through MobileViT:
   Input → Stem → MV2 Blocks → MobileViT Blocks → Pool → Classifier
4. Calculate loss (CrossEntropy with label smoothing)
5. Backward pass (only update classifier weights)
6. Update weights with AdamW optimizer
7. Adjust learning rate with cosine scheduler

Monitoring:
• Track training/validation loss and accuracy
• Save best model based on validation accuracy
• Early stopping if no improvement for 8 epochs
• Generate confusion matrix and classification report
""")

def efficiency_benefits():
    """
    Explain efficiency benefits
    """
    
    print("\n⚡ Efficiency Benefits:")
    print("━" * 30)
    
    efficiency_points = [
        ("Model Size", "5.4M parameters vs 30M+ in larger models"),
        ("Training Speed", "Only 8.6% parameters trainable → 10x faster"),
        ("Memory Usage", "~21MB model vs 100MB+ alternatives"),
        ("Inference Speed", "Mobile-optimized architecture"), 
        ("Data Efficiency", "Pre-trained backbone needs less waste data"),
        ("Accuracy", "Transformer attention captures global waste patterns")
    ]
    
    for metric, benefit in efficiency_points:
        print(f"   • {metric:<15}: {benefit}")

def create_architecture_diagram():
    """
    Create a visual diagram of the architecture
    """
    
    print("\n🎨 Generating Architecture Diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define components with positions and sizes
    components = [
        {"name": "Input\n256×256×3", "pos": (1, 8), "size": (1.5, 1), "color": "lightblue"},
        {"name": "Stem Block\n3×3 Conv", "pos": (3, 8), "size": (1.5, 1), "color": "lightgreen"},
        {"name": "MV2 Block 1\n16→32", "pos": (5, 8), "size": (1.5, 1), "color": "lightcoral"},
        {"name": "MV2 Block 2\n32→64", "pos": (7, 8), "size": (1.5, 1), "color": "lightcoral"},
        {"name": "MobileViT 1\n64→96", "pos": (9, 8), "size": (1.5, 1), "color": "gold"},
        {"name": "MobileViT 2\n96→128", "pos": (11, 8), "size": (1.5, 1), "color": "gold"},
        {"name": "MobileViT 3\n128→160", "pos": (9, 6), "size": (1.5, 1), "color": "gold"},
        {"name": "Global Pool\n4×4→1×1", "pos": (7, 6), "size": (1.5, 1), "color": "plum"},
        {"name": "Linear 1\n640→512", "pos": (5, 6), "size": (1.5, 1), "color": "wheat"},
        {"name": "Linear 2\n512→256", "pos": (3, 6), "size": (1.5, 1), "color": "wheat"},
        {"name": "Output\n9 classes", "pos": (1, 6), "size": (1.5, 1), "color": "lightpink"},
    ]
    
    # Draw components
    for comp in components:
        rect = FancyBboxPatch(
            comp["pos"], comp["size"][0], comp["size"][1],
            boxstyle="round,pad=0.1",
            facecolor=comp["color"],
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(
            comp["pos"][0] + comp["size"][0]/2,
            comp["pos"][1] + comp["size"][1]/2,
            comp["name"],
            ha="center", va="center",
            fontsize=9, fontweight="bold"
        )
    
    # Draw arrows
    arrows = [
        ((2.5, 8.5), (3, 8.5)),    # Input → Stem
        ((4.5, 8.5), (5, 8.5)),    # Stem → MV2-1
        ((6.5, 8.5), (7, 8.5)),    # MV2-1 → MV2-2
        ((8.5, 8.5), (9, 8.5)),    # MV2-2 → MViT-1
        ((10.5, 8.5), (11, 8.5)),  # MViT-1 → MViT-2
        ((11.75, 8), (10.25, 7)),  # MViT-2 → MViT-3
        ((9.25, 7), (8.25, 6.5)),  # MViT-3 → Pool
        ((7.25, 6.5), (6.25, 6.5)), # Pool → Linear-1
        ((5.25, 6.5), (4.25, 6.5)), # Linear-1 → Linear-2
        ((3.25, 6.5), (2.25, 6.5)), # Linear-2 → Output
    ]
    
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                   arrowprops=dict(arrowstyle="->", lw=2, color="darkblue"))
    
    # Add title and labels
    ax.set_title("MobileViT-Small Waste Classification Architecture", 
                fontsize=16, fontweight="bold", pad=20)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightblue', label='Input'),
        patches.Patch(color='lightgreen', label='Stem'),
        patches.Patch(color='lightcoral', label='MobileNetV2 Blocks'),
        patches.Patch(color='gold', label='MobileViT Blocks'),
        patches.Patch(color='plum', label='Pooling'),
        patches.Patch(color='wheat', label='Classifier'),
        patches.Patch(color='lightpink', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # Set limits and remove axes
    ax.set_xlim(0, 13)
    ax.set_ylim(5, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mobilevit_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Architecture diagram saved as 'mobilevit_architecture_diagram.png'")

def main():
    """
    Run complete architecture explanation
    """
    
    # Basic overview
    explain_mobilevit_architecture()
    
    # Detailed breakdown
    detailed_architecture_breakdown()
    
    # Core innovation
    mobilevit_block_explanation()
    
    # Waste classification adaptation
    waste_classification_adaptation()
    
    # Training process
    training_process_explanation()
    
    # Efficiency benefits
    efficiency_benefits()
    
    # Visual diagram
    create_architecture_diagram()
    
    print(f"\n🎉 Architecture Explanation Complete!")
    print(f"━" * 50)
    print(f"""
Key Takeaways:
• MobileViT combines CNN efficiency with Transformer power
• Only 8.6% of parameters are trainable (efficient fine-tuning)
• Hierarchical feature extraction from local to global
• Optimized for mobile/edge deployment
• Perfect for waste classification with 9 categories
• Real-time inference capability with high accuracy
""")

if __name__ == "__main__":
    main()