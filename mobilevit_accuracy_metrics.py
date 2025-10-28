"""
MobileViT Model Accuracy Metrics and Evaluation
Comprehensive evaluation with confusion matrix, classification report, and visualizations
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import os
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class WasteDataset(Dataset):
    """Dataset for waste classification"""
    
    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset(max_samples_per_class)
    
    def _load_dataset(self, max_samples_per_class):
        """Load images and create class mappings"""
        
        print(f"📂 Loading dataset from: {self.root_dir}")
        
        class_counts = defaultdict(int)
        class_images = defaultdict(list)
        
        # Walk through directory
        for root, dirs, files in os.walk(self.root_dir):
            class_name = os.path.basename(root).lower().replace(' ', '_')
            
            # Skip non-class directories
            if class_name in ['realwaste', 'images', 'testing', 'eda_visualizations', 'image_analysis']:
                continue
            
            # Get image files
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                continue
            
            # Collect images for this class
            for img_file in image_files:
                img_path = os.path.join(root, img_file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    class_images[class_name].append(img_path)
                except:
                    continue
        
        # Create class mappings
        unique_classes = sorted(class_images.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Sample images if needed
        for class_name, img_paths in class_images.items():
            if max_samples_per_class and len(img_paths) > max_samples_per_class:
                img_paths = np.random.choice(img_paths, max_samples_per_class, replace=False)
            
            for img_path in img_paths:
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
                self.label_names.append(class_name)
                class_counts[class_name] += 1
        
        print(f"✅ Loaded {len(self.images)} images from {len(unique_classes)} classes")
        print(f"📊 Class distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"   {cls}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MobileViTEvaluator:
    """Comprehensive MobileViT model evaluation"""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model checkpoint (if None, uses pre-trained)
            device: Device to use
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.model_path = model_path
        
        print(f"🤖 MobileViT Evaluator")
        print(f"   Device: {self.device}")
    
    def load_model(self, num_classes=None):
        """Load MobileViT model"""
        
        print(f"📥 Loading MobileViT model...")
        
        # Initialize processor
        self.processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        
        if self.model_path and os.path.exists(self.model_path):
            # Load trained model
            print(f"   Loading from checkpoint: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            num_classes = checkpoint.get('num_classes', num_classes)
            
            self.model = MobileViTForImageClassification.from_pretrained(
                "apple/mobilevit-small",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Loaded trained model with {num_classes} classes")
        else:
            # Use pre-trained model
            print(f"   Using pre-trained model (ImageNet)")
            if num_classes:
                self.model = MobileViTForImageClassification.from_pretrained(
                    "apple/mobilevit-small",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            else:
                self.model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
            print(f"   ⚠️ Note: Using pre-trained weights, not fine-tuned on waste data")
        
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_model(self, dataloader, class_names):
        """Evaluate model and collect predictions"""
        
        print(f"\n🔍 Evaluating model...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                logits = outputs.logits
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        print(f"✅ Evaluation complete")
        
        return all_preds, all_labels, all_probs
    
    def calculate_metrics(self, y_true, y_pred, y_probs, class_names):
        """Calculate comprehensive metrics"""
        
        print(f"\n📊 Calculating metrics...")
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics (with zero_division handling)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class detailed metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted)
            },
            'per_class': {},
            'confusion_matrix': cm.tolist()
        }
        
        # Add per-class metrics
        for idx, class_name in enumerate(class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[idx]) if idx < len(precision_per_class) else 0.0,
                'recall': float(recall_per_class[idx]) if idx < len(recall_per_class) else 0.0,
                'f1_score': float(f1_per_class[idx]) if idx < len(f1_per_class) else 0.0,
                'support': int(np.sum(y_true == idx))
            }
        
        print(f"✅ Metrics calculated")
        
        return metrics
    
    def create_visualizations(self, y_true, y_pred, y_probs, class_names, metrics, save_dir='mobilevit_metrics'):
        """Create comprehensive visualizations"""
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n📊 Creating visualizations...")
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(y_true, y_pred, class_names, save_dir)
        
        # 2. Accuracy Metrics Bar Chart
        self._plot_accuracy_metrics(metrics, save_dir)
        
        # 3. Per-Class Performance
        self._plot_per_class_performance(metrics, class_names, save_dir)
        
        # 4. ROC Curves (if binary or multi-class)
        if len(class_names) <= 10:  # Only for reasonable number of classes
            self._plot_roc_curves(y_true, y_probs, class_names, save_dir)
        
        # 5. Prediction Confidence Distribution
        self._plot_confidence_distribution(y_probs, y_true, y_pred, save_dir)
        
        print(f"✅ All visualizations saved to: {save_dir}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, class_names, save_dir):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title('MobileViT Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Confusion matrix saved")
    
    def _plot_accuracy_metrics(self, metrics, save_dir):
        """Plot overall accuracy metrics"""
        
        overall = metrics['overall']
        
        metric_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)']
        metric_values = [
            overall['accuracy'],
            overall['precision_macro'],
            overall['recall_macro'],
            overall['f1_macro']
        ]
        
        plt.figure(figsize=(10, 6))
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}\n({height*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('MobileViT Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal reference lines
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (80%)')
        plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair (60%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accuracy_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Accuracy metrics plot saved")
    
    def _plot_per_class_performance(self, metrics, class_names, save_dir):
        """Plot per-class performance"""
        
        per_class = metrics['per_class']
        
        # Prepare data
        classes = list(per_class.keys())
        precisions = [per_class[c]['precision'] for c in classes]
        recalls = [per_class[c]['recall'] for c in classes]
        f1_scores = [per_class[c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(14, 8))
        
        plt.bar(x - width, precisions, width, label='Precision', color='#3498db', alpha=0.8, edgecolor='black')
        plt.bar(x, recalls, width, label='Recall', color='#e74c3c', alpha=0.8, edgecolor='black')
        plt.bar(x + width, f1_scores, width, label='F1-Score', color='#2ecc71', alpha=0.8, edgecolor='black')
        
        plt.xlabel('Waste Category', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Per-class performance plot saved")
    
    def _plot_roc_curves(self, y_true, y_probs, class_names, save_dir):
        """Plot ROC curves for each class"""
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        plt.figure(figsize=(12, 10))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(class_names):
            if i < y_probs.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Multi-Class Classification', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ ROC curves saved")
    
    def _plot_confidence_distribution(self, y_probs, y_true, y_pred, save_dir):
        """Plot prediction confidence distribution"""
        
        # Get max confidence for each prediction
        max_probs = np.max(y_probs, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = (y_true == y_pred)
        correct_confidences = max_probs[correct_mask]
        incorrect_confidences = max_probs[~correct_mask]
        
        plt.figure(figsize=(12, 6))
        
        plt.hist(correct_confidences, bins=50, alpha=0.7, color='green', 
                label=f'Correct ({len(correct_confidences)} samples)', edgecolor='black')
        plt.hist(incorrect_confidences, bins=50, alpha=0.7, color='red',
                label=f'Incorrect ({len(incorrect_confidences)} samples)', edgecolor='black')
        
        # Add mean lines
        plt.axvline(np.mean(correct_confidences), color='darkgreen', linestyle='--', 
                   linewidth=2, label=f'Mean Correct: {np.mean(correct_confidences):.3f}')
        if len(incorrect_confidences) > 0:
            plt.axvline(np.mean(incorrect_confidences), color='darkred', linestyle='--',
                       linewidth=2, label=f'Mean Incorrect: {np.mean(incorrect_confidences):.3f}')
        
        plt.xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Prediction Confidence Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Confidence distribution plot saved")
    
    def generate_report(self, metrics, save_path='mobilevit_evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': 'MobileViT-small',
            'model_path': self.model_path or 'Pre-trained (ImageNet)',
            'device': str(self.device),
            'metrics': metrics
        }
        
        # Save JSON report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved: {save_path}")
        
        # Print summary
        print(f"\n" + "="*60)
        print(f"MOBILEVIT MODEL EVALUATION SUMMARY")
        print(f"="*60)
        print(f"Overall Accuracy: {metrics['overall']['accuracy']:.3f} ({metrics['overall']['accuracy']*100:.1f}%)")
        print(f"Precision (Macro): {metrics['overall']['precision_macro']:.3f}")
        print(f"Recall (Macro): {metrics['overall']['recall_macro']:.3f}")
        print(f"F1-Score (Macro): {metrics['overall']['f1_macro']:.3f}")
        print(f"\nPer-Class Performance:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.3f}")
            print(f"    Recall: {class_metrics['recall']:.3f}")
            print(f"    F1-Score: {class_metrics['f1_score']:.3f}")
            print(f"    Support: {class_metrics['support']} samples")
        print(f"="*60)
        
        return report


def main():
    """Main evaluation function"""
    
    print("🚀 MobileViT Model Accuracy Evaluation")
    print("="*60)
    
    # Configuration
    dataset_path = r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\realwaste\RealWaste"
    model_path = r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\best_mobilevit_waste_model.pth"
    
    # Check if trained model exists
    if not os.path.exists(model_path):
        print(f"⚠️ Trained model not found: {model_path}")
        print(f"   Using pre-trained MobileViT (not fine-tuned on waste data)")
        print(f"   Metrics will reflect pre-trained model performance")
        model_path = None
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"\n1️⃣ Loading dataset...")
    dataset = WasteDataset(dataset_path, transform=transform, max_samples_per_class=100)
    
    if len(dataset) == 0:
        print("❌ No images loaded. Please check dataset path.")
        return
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize evaluator
    print(f"\n2️⃣ Initializing evaluator...")
    evaluator = MobileViTEvaluator(model_path=model_path)
    evaluator.load_model(num_classes=len(dataset.class_to_idx))
    
    # Evaluate model
    print(f"\n3️⃣ Running evaluation...")
    class_names = [dataset.idx_to_class[i] for i in range(len(dataset.idx_to_class))]
    y_pred, y_true, y_probs = evaluator.evaluate_model(dataloader, class_names)
    
    # Calculate metrics
    print(f"\n4️⃣ Calculating metrics...")
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_probs, class_names)
    
    # Create visualizations
    print(f"\n5️⃣ Creating visualizations...")
    evaluator.create_visualizations(y_true, y_pred, y_probs, class_names, metrics)
    
    # Generate report
    print(f"\n6️⃣ Generating report...")
    report = evaluator.generate_report(metrics)
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"\n📁 Check 'mobilevit_metrics' folder for all visualizations")
    print(f"📄 Check 'mobilevit_evaluation_report.json' for detailed metrics")


if __name__ == "__main__":
    main()
