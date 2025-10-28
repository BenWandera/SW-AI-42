"""
RealWaste Dataset Integration with GNN Reasoning Model
Connects RealWaste images to the Graph Neural Network for waste classification
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import GNN components
import sys
import os
gnn_path = os.path.join(os.path.dirname(__file__), 'new GNN')
if gnn_path not in sys.path:
    sys.path.insert(0, gnn_path)
from waste_reasoning_rgn import create_waste_reasoning_model, WasteCategory, RiskLevel  # type: ignore


class RealWasteDataset(Dataset):
    """Dataset loader for RealWaste images with category mapping"""
    
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: Path to RealWaste dataset
            transform: Image transformations
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.category_names = []
        
        # Map RealWaste categories to GNN waste categories
        self.category_mapping = {
            'plastic': 'PLASTIC',
            'organic': 'ORGANIC',
            'food organics': 'ORGANIC',
            'paper': 'PAPER',
            'glass': 'GLASS',
            'metal': 'METAL',
            'cardboard': 'PAPER',
            'textile trash': 'MIXED',
            'miscellaneous trash': 'MIXED',
            'vegetation': 'ORGANIC'
        }
        
        print(f"📂 Loading RealWaste dataset from: {root_dir}")
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all images from RealWaste directory structure"""
        
        if not os.path.exists(self.root_dir):
            print(f"❌ Dataset path not found: {self.root_dir}")
            return
        
        category_counts = defaultdict(int)
        
        # Walk through directory structure
        for root, dirs, files in os.walk(self.root_dir):
            # Get category name from directory
            category = os.path.basename(root).lower()
            
            # Skip non-category directories
            if category in ['realwaste', 'images', 'testing', 'eda_visualizations', 'image_analysis']:
                continue
            
            # Map to GNN category
            if category not in self.category_mapping:
                continue
            
            gnn_category = self.category_mapping[category]
            
            # Load images from this category
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(root, img_file)
                
                try:
                    # Verify image can be opened
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    self.images.append(img_path)
                    self.labels.append(gnn_category)
                    self.category_names.append(category)
                    category_counts[gnn_category] += 1
                    
                except Exception as e:
                    continue
        
        print(f"✅ Loaded {len(self.images)} images")
        print(f"📊 Category distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"   {cat}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        category_name = self.category_names[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, category_name, img_path


class VisionFeatureExtractor:
    """Extract vision embeddings using pre-trained models"""
    
    def __init__(self, model_name: str = 'resnet50', device: str = 'cpu'):
        """
        Args:
            model_name: Pre-trained model to use (resnet50, efficientnet_b0)
            device: Device to run on
        """
        self.device = device
        self.model_name = model_name
        
        print(f"🤖 Loading {model_name} feature extractor...")
        
        # Load pre-trained model
        if model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2')
            # Remove final classification layer
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.embedding_dim = 2048
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            # Remove final classification layer
            self.model = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d(1)
            )
            self.embedding_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✅ Feature extractor ready (embedding_dim: {self.embedding_dim})")
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from batch of images"""
        with torch.no_grad():
            images = images.to(self.device)
            features = self.model(images)
            features = features.view(features.size(0), -1)
        return features


class RealWasteGNNIntegration:
    """Integration between RealWaste dataset and GNN model"""
    
    def __init__(self, 
                 realwaste_path: str = r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\realwaste\RealWaste",
                 device: str = None):
        """
        Initialize integration system
        
        Args:
            realwaste_path: Path to RealWaste dataset
            device: Device to use (auto-detect if None)
        """
        self.realwaste_path = realwaste_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🌟 RealWaste + GNN Integration System")
        print(f"=" * 50)
        print(f"Device: {self.device}")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        self.dataset = None
        self.feature_extractor = None
        self.gnn_model = None
        self.features_cache = None
        
    def load_dataset(self, max_images: int = None):
        """Load RealWaste dataset"""
        print(f"\n📂 Loading RealWaste dataset...")
        self.dataset = RealWasteDataset(self.realwaste_path, transform=self.transform)
        
        if len(self.dataset) == 0:
            print("❌ No images loaded. Please check dataset path.")
            return False
        
        return True
    
    def initialize_feature_extractor(self, model_name: str = 'resnet50'):
        """Initialize vision feature extractor"""
        print(f"\n🔧 Initializing feature extractor...")
        self.feature_extractor = VisionFeatureExtractor(model_name, self.device)
        return True
    
    def initialize_gnn_model(self):
        """Initialize GNN reasoning model"""
        print(f"\n🧠 Initializing GNN reasoning model...")
        
        if self.feature_extractor is None:
            print("❌ Feature extractor must be initialized first")
            return False
        
        # Create GNN model with proper embedding dimension
        self.gnn_model = create_waste_reasoning_model(
            vision_embedding_dim=self.feature_extractor.embedding_dim
        )
        self.gnn_model = self.gnn_model.to(self.device)
        self.gnn_model.eval()
        
        print(f"✅ GNN model ready")
        return True
    
    def extract_all_features(self, batch_size: int = 32, max_samples: int = None):
        """Extract vision features for all images"""
        
        if self.dataset is None or self.feature_extractor is None:
            print("❌ Dataset and feature extractor must be initialized")
            return None
        
        print(f"\n🔍 Extracting vision features...")
        
        # Create dataloader
        num_samples = min(len(self.dataset), max_samples) if max_samples else len(self.dataset)
        subset_indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        subset = torch.utils.data.Subset(self.dataset, subset_indices)
        
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_features = []
        all_labels = []
        all_categories = []
        all_paths = []
        
        with torch.no_grad():
            for images, labels, categories, paths in tqdm(dataloader, desc="Extracting features"):
                # Extract features
                features = self.feature_extractor.extract_features(images)
                
                all_features.append(features.cpu())
                all_labels.extend(labels)
                all_categories.extend(categories)
                all_paths.extend(paths)
        
        # Concatenate all features
        self.features_cache = {
            'features': torch.cat(all_features, dim=0),
            'labels': all_labels,
            'categories': all_categories,
            'paths': all_paths
        }
        
        print(f"✅ Extracted features for {len(all_labels)} images")
        print(f"   Feature dimension: {self.features_cache['features'].shape}")
        
        return self.features_cache
    
    def run_gnn_inference(self, num_samples: int = 100):
        """Run GNN inference on sample images"""
        
        if self.features_cache is None:
            print("❌ Features must be extracted first")
            return None
        
        print(f"\n🔮 Running GNN inference on {num_samples} samples...")
        
        # Sample random images
        num_samples = min(num_samples, len(self.features_cache['features']))
        indices = np.random.choice(len(self.features_cache['features']), num_samples, replace=False)
        
        results = []
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="GNN inference"):
                # Get features
                features = self.features_cache['features'][idx:idx+1].to(self.device)
                
                # Run GNN
                outputs = self.gnn_model(features)
                
                # Get predictions
                material_pred = outputs['material_logits'].argmax(dim=1).item()
                category_pred = outputs['category_logits'].argmax(dim=1).item()
                disposal_pred = outputs['disposal_logits'].argmax(dim=1).item()
                risk_pred = outputs['risk_scores'].argmax(dim=1).item() if 'risk_scores' in outputs else 0
                confidence = outputs['confidence'].squeeze().item() if 'confidence' in outputs else 0.5
                
                results.append({
                    'idx': int(idx),
                    'true_label': self.features_cache['labels'][idx],
                    'true_category': self.features_cache['categories'][idx],
                    'image_path': self.features_cache['paths'][idx],
                    'predicted_material': material_pred,
                    'predicted_category': category_pred,
                    'predicted_disposal': disposal_pred,
                    'predicted_risk': risk_pred,
                    'confidence': confidence
                })
        
        print(f"✅ GNN inference complete")
        return results
    
    def create_visualizations(self, results: list, save_dir: str = "gnn_visualizations"):
        """Create comprehensive visualizations"""
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n📊 Creating visualizations...")
        
        # 1. Confidence distribution
        self._plot_confidence_distribution(results, save_dir)
        
        # 2. Category confusion matrix
        self._plot_category_matrix(results, save_dir)
        
        # 3. Risk level distribution
        self._plot_risk_distribution(results, save_dir)
        
        # 4. Sample predictions
        self._plot_sample_predictions(results, save_dir, num_samples=12)
        
        print(f"✅ All visualizations saved to: {save_dir}")
    
    def _plot_confidence_distribution(self, results: list, save_dir: str):
        """Plot confidence score distribution"""
        confidences = [r['confidence'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('GNN Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_conf = np.mean(confidences)
        plt.axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Confidence distribution plot saved")
    
    def _plot_category_matrix(self, results: list, save_dir: str):
        """Plot category prediction matrix"""
        
        # Get unique categories
        true_cats = [r['true_label'] for r in results]
        pred_cats = [r['predicted_category'] for r in results]
        
        unique_cats = sorted(list(set(true_cats)))
        
        # Create confusion matrix
        matrix = np.zeros((len(unique_cats), len(unique_cats)))
        
        for true, pred in zip(true_cats, pred_cats):
            if true in unique_cats:
                true_idx = unique_cats.index(true)
                # Map pred to closest category
                pred_idx = pred % len(unique_cats)
                matrix[true_idx, pred_idx] += 1
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=unique_cats, yticklabels=unique_cats,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Category', fontsize=12)
        plt.ylabel('True Category', fontsize=12)
        plt.title('Category Prediction Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'category_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Category matrix plot saved")
    
    def _plot_risk_distribution(self, results: list, save_dir: str):
        """Plot risk level distribution"""
        
        risk_labels = ['SAFE', 'LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK', 'CRITICAL']
        risk_counts = defaultdict(int)
        
        for r in results:
            risk_level = r['predicted_risk']
            if risk_level < len(risk_labels):
                risk_counts[risk_labels[risk_level]] += 1
        
        # Plot
        plt.figure(figsize=(10, 6))
        categories = list(risk_counts.keys())
        counts = [risk_counts[cat] for cat in categories]
        colors = ['green', 'lightblue', 'yellow', 'orange', 'red'][:len(categories)]
        
        bars = plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Risk Level', fontsize=12)
        plt.ylabel('Number of Items', fontsize=12)
        plt.title('Predicted Risk Level Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'risk_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Risk distribution plot saved")
    
    def _plot_sample_predictions(self, results: list, save_dir: str, num_samples: int = 12):
        """Plot sample images with predictions"""
        
        # Select random samples
        sample_indices = np.random.choice(len(results), min(num_samples, len(results)), replace=False)
        samples = [results[i] for i in sample_indices]
        
        # Create grid
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (ax, sample) in enumerate(zip(axes, samples)):
            # Load image
            try:
                img = Image.open(sample['image_path']).convert('RGB')
                img = img.resize((224, 224))
            except:
                img = Image.new('RGB', (224, 224), color='gray')
            
            ax.imshow(img)
            ax.axis('off')
            
            # Add prediction info
            true_label = sample['true_label']
            confidence = sample['confidence']
            risk = sample['predicted_risk']
            risk_labels = ['SAFE', 'LOW', 'MED', 'HIGH', 'CRIT']
            risk_text = risk_labels[risk] if risk < len(risk_labels) else 'UNK'
            
            title = f"True: {true_label}\nConf: {confidence:.2f}\nRisk: {risk_text}"
            ax.set_title(title, fontsize=9, fontweight='bold')
        
        plt.suptitle('GNN Sample Predictions on RealWaste Dataset', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Sample predictions plot saved")
    
    def generate_report(self, results: list, save_path: str = "gnn_analysis_report.json"):
        """Generate comprehensive analysis report"""
        
        print(f"\n📝 Generating analysis report...")
        
        # Calculate statistics
        confidences = [r['confidence'] for r in results]
        risks = [r['predicted_risk'] for r in results]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'path': self.realwaste_path,
                'total_images': len(self.dataset) if self.dataset else 0,
                'analyzed_samples': len(results)
            },
            'model': {
                'feature_extractor': self.feature_extractor.model_name if self.feature_extractor else 'N/A',
                'embedding_dim': self.feature_extractor.embedding_dim if self.feature_extractor else 0,
                'device': self.device
            },
            'statistics': {
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'min_confidence': float(np.min(confidences)),
                'max_confidence': float(np.max(confidences)),
                'median_confidence': float(np.median(confidences))
            },
            'risk_distribution': {
                'safe': int(sum(1 for r in risks if r == 0)),
                'low_risk': int(sum(1 for r in risks if r == 1)),
                'medium_risk': int(sum(1 for r in risks if r == 2)),
                'high_risk': int(sum(1 for r in risks if r == 3)),
                'critical': int(sum(1 for r in risks if r == 4))
            },
            'category_distribution': {}
        }
        
        # Category distribution
        for r in results:
            cat = r['true_label']
            if cat not in report['category_distribution']:
                report['category_distribution'][cat] = {
                    'count': 0,
                    'avg_confidence': []
                }
            report['category_distribution'][cat]['count'] += 1
            report['category_distribution'][cat]['avg_confidence'].append(r['confidence'])
        
        # Calculate average confidence per category
        for cat in report['category_distribution']:
            conf_list = report['category_distribution'][cat]['avg_confidence']
            report['category_distribution'][cat]['avg_confidence'] = float(np.mean(conf_list))
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved to: {save_path}")
        
        # Print summary
        print(f"\n📊 Analysis Summary:")
        print(f"   Samples analyzed: {report['dataset']['analyzed_samples']}")
        print(f"   Mean confidence: {report['statistics']['mean_confidence']:.3f}")
        print(f"   High-risk items: {report['risk_distribution']['high_risk']}")
        print(f"   Critical items: {report['risk_distribution']['critical']}")
        
        return report


def main():
    """Main execution function"""
    
    print("🚀 RealWaste + GNN Integration Pipeline")
    print("=" * 60)
    
    # Initialize integration
    integration = RealWasteGNNIntegration()
    
    # Step 1: Load dataset
    if not integration.load_dataset():
        print("❌ Failed to load dataset")
        return
    
    # Step 2: Initialize feature extractor
    if not integration.initialize_feature_extractor(model_name='resnet50'):
        print("❌ Failed to initialize feature extractor")
        return
    
    # Step 3: Initialize GNN model
    if not integration.initialize_gnn_model():
        print("❌ Failed to initialize GNN model")
        return
    
    # Step 4: Extract features (use subset for demo)
    features = integration.extract_all_features(batch_size=32, max_samples=500)
    
    if features is None:
        print("❌ Failed to extract features")
        return
    
    # Step 5: Run GNN inference
    results = integration.run_gnn_inference(num_samples=200)
    
    if results is None:
        print("❌ Failed to run GNN inference")
        return
    
    # Step 6: Create visualizations
    integration.create_visualizations(results, save_dir="gnn_visualizations")
    
    # Step 7: Generate report
    report = integration.generate_report(results, save_path="gnn_analysis_report.json")
    
    print(f"\n✅ Integration pipeline completed successfully!")
    print(f"📁 Check 'gnn_visualizations' folder for visual outputs")
    print(f"📄 Check 'gnn_analysis_report.json' for detailed statistics")


if __name__ == "__main__":
    main()
