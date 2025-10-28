# AI-Powered Waste Management System 🗑️♻️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive AI system for automated waste classification, featuring Vision Transformers, Graph Neural Networks, GAN-based data augmentation, and a gamified incentive engine.

## 🌟 Features

### 1. **Vision Transformer Classification**
- **MobileViT-Small**: 88.42% accuracy, 5.6M parameters
- **DeiT-Tiny**: 89.26% accuracy, 5.7M parameters
- Multi-class waste classification (9 categories)
- Real-time inference capability

### 2. **GNN Reasoning System**
- Knowledge graph with 21 nodes, 23 edges
- Hierarchical waste relationships (Material → Category → Disposal)
- Misclassification correction: +4.84% accuracy improvement
- Risk-aware decision making

### 3. **GAN Data Augmentation**
- Synthetic waste image generation
- Quality score: 82%
- 1000+ synthetic samples generated
- Dataset balancing capability

### 4. **Incentive Engine**
- Gamified reward system
- Multi-tier progression (Bronze → Silver → Gold → Platinum → Diamond)
- Real-time feedback
- User engagement tracking

## 📊 System Performance

| Component | Accuracy | Parameters | Speed |
|-----------|----------|------------|-------|
| MobileViT | 88.42% | 5.6M | ~30ms |
| DeiT-Tiny | 89.26% | 5.7M | ~28ms |
| GNN Standalone | 75.5% | - | - |
| GNN-Corrected | 93.26% | - | - |
| **Integrated System** | **85.39%** | - | - |

## 🗂️ Project Structure

```
DATASETS/
├── training_scripts/          # Model training scripts
│   ├── enhanced_mobilevit_trainer.py
│   ├── deit_tiny_trainer.py
│   └── realwaste_model_training.py
├── evaluation/               # Evaluation and metrics
│   ├── mobilevit_accuracy_metrics.py
│   ├── compare_models.py
│   └── complete_system_metrics.py
├── visualization/            # Data visualization
│   ├── visualize_knowledge_graph.py
│   ├── realwaste_comprehensive_eda.py
│   └── gan_comparison_visualizer.py
├── GNN model/               # Graph Neural Network
│   ├── new GNN/
│   │   └── waste_reasoning_rgn.py
│   └── create_gnn_visualizations.py
├── incentive_system/        # Gamification engine
│   ├── simple_incentive_engine.py
│   └── standalone_incentive_calculator.py
├── synthetic_generation/    # GAN generators
│   ├── synthetic_waste_generator.py
│   └── lightweight_gan_creator.py
├── system_integration/      # Complete system
│   ├── complete_waste_system.py
│   └── simple_complete_demo.py
├── documentation/           # Reports and guides
│   ├── MOBILEVIT_ACCURACY_REPORT.md
│   ├── INCENTIVE_USAGE_GUIDE.md
│   └── PROJECT_SUMMARY.md
├── config/                 # Configuration files
│   └── requirements.txt
├── results/               # Metrics (JSON)
│   ├── system_accuracy_metrics.json
│   └── gnn_classification_accuracy.json
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/waste-management-ai.git
cd waste-management-ai
```

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r config/requirements.txt
```

### 3. Download Dataset
Download the RealWaste dataset and place it in:
```
realwaste/RealWaste/
├── Cardboard/
├── Food Organics/
├── Glass/
├── Metal/
├── Miscellaneous Trash/
├── Paper/
├── Plastic/
├── Textile Trash/
└── Vegetation/
```

### 4. Train Models

**Train MobileViT:**
```bash
python training_scripts/enhanced_mobilevit_trainer.py
```

**Train DeiT-Tiny:**
```bash
python training_scripts/deit_tiny_trainer.py
```

### 5. Run Evaluation
```bash
# Display MobileViT metrics
python evaluation/display_mobilevit_metrics.py

# Compare models
python evaluation/compare_models.py

# System-wide metrics
python evaluation/complete_system_metrics.py
```

### 6. Demo the System
```bash
python system_integration/complete_waste_system.py
```

## 📈 Model Training Results

### MobileViT-Small
- **Test Accuracy**: 88.42%
- **Training Time**: 13.1 hours (50 epochs)
- **Best Classes**: Glass (F1: 0.925), Vegetation (F1: 0.918)
- **Worst Class**: Miscellaneous Trash (F1: 0.798)

### DeiT-Tiny
- **Test Accuracy**: 89.26%
- **Training Time**: 11.8 hours (50 epochs)
- **Improvement**: +0.84% over MobileViT
- **Best Classes**: Glass (F1: 0.936), Vegetation (F1: 0.925)

### GNN Enhancement
- **Baseline (MobileViT)**: 88.42%
- **GNN Corrected**: 93.26%
- **Improvement**: +4.84%
- **Corrections Made**: 46/950 samples (4.8%)

## 🧠 Knowledge Graph Structure

The GNN uses a hierarchical knowledge graph:

**Level 1: Materials (9 nodes)**
- plastic_pet, plastic_pvc, organic_food, paper_cardboard, glass_clear, metal_aluminum, electronic_battery, medical_sharp, medical_infectious

**Level 2: Categories (7 nodes)**
- plastic, organic, paper, glass, metal, electronic, medical

**Level 3: Disposal Methods (5 nodes)**
- recyclable, compostable, landfill, hazardous_disposal, specialized_facility

**Relationships:**
- `derives_from`: Material → Category (9 edges)
- `requires`: Category → Disposal (11 edges)
- `conflicts_with`: Safety conflicts (3 edges)

## 🎮 Incentive System

The gamification engine includes:
- **5 Tiers**: Bronze → Silver → Gold → Platinum → Diamond
- **Points System**: Based on accuracy, difficulty, and waste type
- **Streak Bonuses**: Encourage consistent participation
- **Real-time Feedback**: Immediate validation and rewards

## 📊 Visualizations

The system generates comprehensive visualizations:
- Confusion matrices
- Per-class performance metrics
- Training curves
- Knowledge graph structure
- GAN quality comparison
- System architecture diagrams

## 🔬 Technologies Used

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Vision Models**: MobileViT, DeiT, ResNet50
- **Graph Neural Networks**: Custom RGN implementation
- **GANs**: Lightweight GAN for data augmentation
- **Visualization**: Matplotlib, Seaborn, NetworkX
- **Data Processing**: NumPy, Pandas, PIL

## 📝 Key Files

| File | Purpose |
|------|---------|
| `enhanced_mobilevit_trainer.py` | MobileViT training pipeline |
| `deit_tiny_trainer.py` | DeiT-Tiny training pipeline |
| `waste_reasoning_rgn.py` | GNN reasoning model |
| `simple_incentive_engine.py` | Gamification engine |
| `synthetic_waste_generator.py` | GAN-based data generation |
| `complete_waste_system.py` | Integrated system demo |
| `compare_models.py` | Model comparison utility |

## 🎯 Future Improvements

- [ ] Real-time video classification
- [ ] Mobile app integration
- [ ] Multi-language support
- [ ] Cloud deployment (AWS/Azure)
- [ ] Edge device optimization (Raspberry Pi, Jetson)
- [ ] Explainable AI (Grad-CAM visualizations)
- [ ] Active learning pipeline
- [ ] Blockchain-based incentive tracking

## 📄 Documentation

Detailed documentation available in the `documentation/` folder:
- **MOBILEVIT_ACCURACY_REPORT.md**: Complete model evaluation
- **INCENTIVE_USAGE_GUIDE.md**: Incentive system usage
- **PROJECT_SUMMARY.md**: Project overview and architecture

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **RealWaste Dataset**: For providing comprehensive waste classification data
- **Hugging Face**: For pre-trained transformer models
- **PyTorch Team**: For the deep learning framework
- **Open Source Community**: For valuable tools and libraries

## 📊 Citation

If you use this work in your research, please cite:

```bibtex
@software{waste_management_ai_2025,
  title={AI-Powered Waste Management System},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/waste-management-ai}
}
```

---

**⭐ Star this repository if you find it useful!**

**🔄 Watch for updates and improvements**

**🍴 Fork to create your own version**
