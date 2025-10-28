"""
Enhanced GNN Misclassification Handler with Classification Accuracy Testing
Uses GNN reasoning to correct MobileViT misclassifications and evaluates performance
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import os
import json
from datetime import datetime
from tqdm import tqdm
import sys
gnn_path = os.path.join(os.path.dirname(__file__), 'GNN model', 'new GNN')
if gnn_path not in sys.path:
    sys.path.insert(0, gnn_path)
from waste_reasoning_rgn import WasteReasoningRGN, WasteKnowledgeGraph  # type: ignore
import warnings
warnings.filterwarnings('ignore')


class MisclassificationCorrector:
    """GNN-based misclassification correction system"""
    
    def __init__(self, mobilevit_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mobilevit_path = mobilevit_path
        
        # Models
        self.mobilevit_model = None
        self.mobilevit_processor = None
        self.feature_extractor = None
        self.gnn_model = None
        self.knowledge_graph = None
        
        # Class mappings
        self.mobilevit_classes = [
            "Cardboard", "Food Organics", "Glass", "Metal", 
            "Miscellaneous Trash", "Paper", "Plastic", 
            "Textile Trash", "Vegetation"
        ]
        
        self.gnn_class_mapping = {
            'Cardboard': 'PAPER',
            'Food Organics': 'ORGANIC',
            'Glass': 'GLASS',
            'Metal': 'METAL',
            'Miscellaneous Trash': 'MIXED',
            'Paper': 'PAPER',
            'Plastic': 'PLASTIC',
            'Textile Trash': 'MIXED',
            'Vegetation': 'ORGANIC'
        }
        
        self.gnn_to_mobilevit = {
            'PAPER': ['Cardboard', 'Paper'],
            'ORGANIC': ['Food Organics', 'Vegetation'],
            'GLASS': ['Glass'],
            'METAL': ['Metal'],
            'PLASTIC': ['Plastic'],
            'MIXED': ['Miscellaneous Trash', 'Textile Trash']
        }
        
        print(f"🤖 Misclassification Corrector initialized")
        print(f"   Device: {self.device}")
    
    def load_models(self):
        """Load MobileViT and GNN models"""
        
        print(f"\n📥 Loading models...")
        
        # Load MobileViT
        self.mobilevit_processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        
        if os.path.exists(self.mobilevit_path):
            checkpoint = torch.load(self.mobilevit_path, map_location=self.device, weights_only=False)
            num_classes = len(self.mobilevit_classes)
            
            self.mobilevit_model = MobileViTForImageClassification.from_pretrained(
                "apple/mobilevit-small",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            
            # Fix state dict keys
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('mobilevit.mobilevit.'):
                    new_key = k.replace('mobilevit.mobilevit.', 'mobilevit.')
                elif k == 'classifier.1.weight':
                    new_key = 'classifier.weight'
                elif k == 'classifier.1.bias':
                    new_key = 'classifier.bias'
                else:
                    continue
                new_state_dict[new_key] = v
            
            self.mobilevit_model.load_state_dict(new_state_dict, strict=False)
            self.mobilevit_model.to(self.device)
            self.mobilevit_model.eval()
            print(f"   ✓ MobileViT model loaded")
        
        # Load ResNet50 feature extractor
        self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor.fc = nn.Identity()
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        print(f"   ✓ ResNet50 feature extractor loaded")
        
        # Initialize knowledge graph and GNN
        self.knowledge_graph = WasteKnowledgeGraph()
        self.gnn_model = WasteReasoningRGN(
            input_dim=2048,
            hidden_dim=256,
            num_node_types=len(self.knowledge_graph.node_types),
            num_edge_types=len(self.knowledge_graph.edge_types),
            num_categories=len(self.knowledge_graph.categories)
        )
        self.gnn_model.to(self.device)
        self.gnn_model.eval()
        print(f"   ✓ GNN reasoning model initialized")
    
    def classify_with_correction(self, image_path):
        """
        Classify image with MobileViT and apply GNN correction if needed
        Returns: (mobilevit_pred, gnn_corrected_pred, mobilevit_conf, gnn_conf, corrected)
        """
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # MobileViT classification
        mobilevit_inputs = self.mobilevit_processor(images=image, return_tensors="pt")
        mobilevit_inputs = {k: v.to(self.device) for k, v in mobilevit_inputs.items()}
        
        with torch.no_grad():
            outputs = self.mobilevit_model(**mobilevit_inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            mobilevit_pred_idx = torch.argmax(probs).item()
            mobilevit_conf = probs[mobilevit_pred_idx].item()
        
        mobilevit_pred = self.mobilevit_classes[mobilevit_pred_idx]
        
        # Extract features for GNN
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        
        # GNN reasoning
        graph_data = self.knowledge_graph.to_pyg_data(self.device)
        gnn_outputs = self.gnn_model(features, graph_data)
        
        gnn_category_logits = gnn_outputs['category_logits']
        gnn_category_probs = torch.softmax(gnn_category_logits, dim=1)[0]
        gnn_pred_idx = torch.argmax(gnn_category_probs).item()
        gnn_conf = gnn_category_probs[gnn_pred_idx].item()
        
        gnn_category = self.knowledge_graph.categories[gnn_pred_idx]
        
        # Check if correction is needed
        mobilevit_gnn_category = self.gnn_class_mapping[mobilevit_pred]
        
        corrected = False
        final_pred = mobilevit_pred
        
        # Apply correction if:
        # 1. MobileViT confidence is low (<0.7) AND
        # 2. GNN confidence is high (>0.6) AND
        # 3. GNN disagrees with MobileViT
        if mobilevit_conf < 0.7 and gnn_conf > 0.6 and gnn_category != mobilevit_gnn_category:
            # GNN suggests correction - pick most likely class from GNN category
            possible_classes = self.gnn_to_mobilevit[gnn_category]
            # Re-check MobileViT probs for these classes
            class_probs = {cls: probs[self.mobilevit_classes.index(cls)].item() 
                          for cls in possible_classes}
            final_pred = max(class_probs, key=class_probs.get)
            corrected = True
        
        return mobilevit_pred, final_pred, mobilevit_conf, gnn_conf, corrected
    
    def evaluate_correction_accuracy(self, dataset_path, max_samples=500):
        """Evaluate GNN correction accuracy on test set"""
        
        print(f"\n🔍 Evaluating misclassification correction...")
        
        # Load test images
        test_images = []
        test_labels = []
        
        for root, dirs, files in os.walk(dataset_path):
            class_name = os.path.basename(root).lower().replace(' ', '_')
            
            if class_name not in [c.lower().replace(' ', '_') for c in self.mobilevit_classes]:
                continue
            
            # Get actual class name
            actual_class = [c for c in self.mobilevit_classes 
                          if c.lower().replace(' ', '_') == class_name][0]
            
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files[:max_samples // len(self.mobilevit_classes)]:
                img_path = os.path.join(root, img_file)
                test_images.append(img_path)
                test_labels.append(actual_class)
        
        print(f"   Loaded {len(test_images)} test images")
        
        # Evaluate
        mobilevit_predictions = []
        corrected_predictions = []
        corrections_made = 0
        
        for img_path, true_label in tqdm(zip(test_images, test_labels), 
                                         total=len(test_images), desc="Evaluating"):
            try:
                mobilevit_pred, final_pred, mobilevit_conf, gnn_conf, corrected = \
                    self.classify_with_correction(img_path)
                
                mobilevit_predictions.append(mobilevit_pred)
                corrected_predictions.append(final_pred)
                
                if corrected:
                    corrections_made += 1
            except Exception as e:
                # If error, use mobilevit prediction
                mobilevit_predictions.append(mobilevit_predictions[-1] if mobilevit_predictions else self.mobilevit_classes[0])
                corrected_predictions.append(mobilevit_predictions[-1] if mobilevit_predictions else self.mobilevit_classes[0])
        
        # Calculate metrics
        mobilevit_accuracy = accuracy_score(test_labels, mobilevit_predictions)
        corrected_accuracy = accuracy_score(test_labels, corrected_predictions)
        
        improvement = corrected_accuracy - mobilevit_accuracy
        
        results = {
            'mobilevit_only': {
                'accuracy': mobilevit_accuracy,
                'precision': precision_score(test_labels, mobilevit_predictions, average='weighted', zero_division=0),
                'recall': recall_score(test_labels, mobilevit_predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(test_labels, mobilevit_predictions, average='weighted', zero_division=0)
            },
            'gnn_corrected': {
                'accuracy': corrected_accuracy,
                'precision': precision_score(test_labels, corrected_predictions, average='weighted', zero_division=0),
                'recall': recall_score(test_labels, corrected_predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(test_labels, corrected_predictions, average='weighted', zero_division=0)
            },
            'correction_stats': {
                'corrections_made': corrections_made,
                'total_samples': len(test_images),
                'correction_rate': corrections_made / len(test_images) * 100,
                'accuracy_improvement': improvement * 100
            },
            'confusion_matrices': {
                'mobilevit': confusion_matrix(test_labels, mobilevit_predictions).tolist(),
                'corrected': confusion_matrix(test_labels, corrected_predictions).tolist()
            },
            'test_labels': test_labels,
            'mobilevit_predictions': mobilevit_predictions,
            'corrected_predictions': corrected_predictions
        }
        
        print(f"\n   ✓ Evaluation complete")
        print(f"   • MobileViT-only Accuracy: {mobilevit_accuracy*100:.2f}%")
        print(f"   • GNN-corrected Accuracy: {corrected_accuracy*100:.2f}%")
        print(f"   • Improvement: {improvement*100:.2f}%")
        print(f"   • Corrections Made: {corrections_made}/{len(test_images)} ({corrections_made/len(test_images)*100:.1f}%)")
        
        return results
    
    def visualize_correction_results(self, results, save_dir='gnn_correction_metrics'):
        """Create comprehensive visualizations of correction results"""
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n📊 Creating visualizations...")
        
        # 1. Accuracy Comparison
        self._plot_accuracy_comparison(results, save_dir)
        
        # 2. Confusion Matrices
        self._plot_confusion_matrices(results, save_dir)
        
        # 3. Correction Impact Analysis
        self._plot_correction_impact(results, save_dir)
        
        # 4. Per-Class Performance
        self._plot_per_class_performance(results, save_dir)
        
        print(f"   ✓ All visualizations saved to: {save_dir}")
    
    def _plot_accuracy_comparison(self, results, save_dir):
        """Plot accuracy comparison"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        mobilevit_values = [
            results['mobilevit_only']['accuracy'] * 100,
            results['mobilevit_only']['precision'] * 100,
            results['mobilevit_only']['recall'] * 100,
            results['mobilevit_only']['f1_score'] * 100
        ]
        corrected_values = [
            results['gnn_corrected']['accuracy'] * 100,
            results['gnn_corrected']['precision'] * 100,
            results['gnn_corrected']['recall'] * 100,
            results['gnn_corrected']['f1_score'] * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, mobilevit_values, width, label='MobileViT Only',
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
        bars2 = ax1.bar(x + width/2, corrected_values, width, label='GNN Corrected',
                       color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{height:.1f}%', ha='center', fontsize=9, fontweight='bold')
        
        ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax1.set_title('MobileViT vs GNN-Corrected Performance', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 105)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Correction stats
        ax2.axis('off')
        
        correction_text = f"""
        GNN CORRECTION STATISTICS
        ═══════════════════════════════════
        
        📊 Correction Rate:
           {results['correction_stats']['corrections_made']} / {results['correction_stats']['total_samples']} samples
           ({results['correction_stats']['correction_rate']:.1f}%)
        
        📈 Accuracy Improvement:
           {results['correction_stats']['accuracy_improvement']:.2f}%
        
        🎯 Final Accuracy:
           {results['gnn_corrected']['accuracy']*100:.2f}%
        
        ✅ Status: {'Improvement Achieved' if results['correction_stats']['accuracy_improvement'] > 0 else 'No Improvement'}
        """
        
        bg_color = 'lightgreen' if results['correction_stats']['accuracy_improvement'] > 0 else 'lightyellow'
        
        ax2.text(0.5, 0.5, correction_text, fontsize=11, verticalalignment='center',
                horizontalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5, 
                         edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gnn_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ Accuracy comparison saved")
    
    def _plot_confusion_matrices(self, results, save_dir):
        """Plot confusion matrices"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        cm_mobilevit = np.array(results['confusion_matrices']['mobilevit'])
        cm_corrected = np.array(results['confusion_matrices']['corrected'])
        
        # Normalize
        cm_mobilevit_norm = cm_mobilevit.astype('float') / cm_mobilevit.sum(axis=1)[:, np.newaxis]
        cm_corrected_norm = cm_corrected.astype('float') / cm_corrected.sum(axis=1)[:, np.newaxis]
        
        # Plot MobileViT
        sns.heatmap(cm_mobilevit_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.mobilevit_classes, yticklabels=self.mobilevit_classes,
                   ax=ax1, cbar_kws={'label': 'Normalized Count'})
        ax1.set_title('MobileViT Only - Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax1.set_ylabel('True', fontsize=11, fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot Corrected
        sns.heatmap(cm_corrected_norm, annot=True, fmt='.2f', cmap='Greens',
                   xticklabels=self.mobilevit_classes, yticklabels=self.mobilevit_classes,
                   ax=ax2, cbar_kws={'label': 'Normalized Count'})
        ax2.set_title('GNN Corrected - Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax2.set_ylabel('True', fontsize=11, fontweight='bold')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gnn_confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ Confusion matrices saved")
    
    def _plot_correction_impact(self, results, save_dir):
        """Plot correction impact analysis"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        fig.suptitle('GNN Misclassification Correction Impact Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Correction distribution
        correction_rate = results['correction_stats']['correction_rate']
        no_correction_rate = 100 - correction_rate
        
        sizes = [correction_rate, no_correction_rate]
        labels = [f'Corrected\n({results["correction_stats"]["corrections_made"]})', 
                 f'Unchanged\n({results["correction_stats"]["total_samples"] - results["correction_stats"]["corrections_made"]})']
        colors = ['#e74c3c', '#3498db']
        explode = (0.05, 0)
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
               explode=explode, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Correction Distribution', fontsize=12, fontweight='bold', pad=15)
        
        # 2. Accuracy improvement
        improvement = results['correction_stats']['accuracy_improvement']
        mobilevit_acc = results['mobilevit_only']['accuracy'] * 100
        corrected_acc = results['gnn_corrected']['accuracy'] * 100
        
        categories = ['MobileViT\nOnly', 'GNN\nCorrected']
        accuracies = [mobilevit_acc, corrected_acc]
        colors_bars = ['#3498db', '#2ecc71']
        
        bars = ax2.bar(categories, accuracies, color=colors_bars, alpha=0.8,
                      edgecolor='black', linewidth=2)
        
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold')
        
        # Add improvement arrow
        if improvement > 0:
            ax2.annotate('', xy=(1, corrected_acc), xytext=(0, mobilevit_acc),
                        arrowprops=dict(arrowstyle='->', lw=3, color='green'))
            ax2.text(0.5, (mobilevit_acc + corrected_acc)/2, 
                    f'+{improvement:.2f}%', ha='center', fontsize=11, 
                    fontweight='bold', color='green')
        
        ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Accuracy Improvement', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Metric improvements
        metrics = ['Precision', 'Recall', 'F1-Score']
        improvements = [
            (results['gnn_corrected']['precision'] - results['mobilevit_only']['precision']) * 100,
            (results['gnn_corrected']['recall'] - results['mobilevit_only']['recall']) * 100,
            (results['gnn_corrected']['f1_score'] - results['mobilevit_only']['f1_score']) * 100
        ]
        
        colors_impr = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvements]
        bars = ax3.barh(metrics, improvements, color=colors_impr, alpha=0.8, edgecolor='black', linewidth=2)
        
        for bar, val in zip(bars, improvements):
            ax3.text(val + 0.05 if val > 0 else val - 0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:+.2f}%', va='center', fontsize=10, fontweight='bold')
        
        ax3.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Metric-wise Improvement', fontsize=12, fontweight='bold', pad=15)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.invert_yaxis()
        
        # 4. Summary stats
        ax4.axis('off')
        
        summary_text = f"""
        GNN CORRECTION SUMMARY
        ═══════════════════════════════════════
        
        📊 Test Samples: {results['correction_stats']['total_samples']}
        
        🔄 Corrections Applied: {results['correction_stats']['corrections_made']}
        
        📈 Improvement Metrics:
           • Accuracy: {improvement:+.2f}%
           • Precision: {improvements[0]:+.2f}%
           • Recall: {improvements[1]:+.2f}%
           • F1-Score: {improvements[2]:+.2f}%
        
        🎯 Final Performance:
           • Accuracy: {corrected_acc:.2f}%
           • Precision: {results['gnn_corrected']['precision']*100:.2f}%
           • Recall: {results['gnn_corrected']['recall']*100:.2f}%
           • F1-Score: {results['gnn_corrected']['f1_score']*100:.2f}%
        
        ✅ Status: GNN Correction {'Effective' if improvement > 0 else 'Neutral'}
        """
        
        ax4.text(0.5, 0.5, summary_text, fontsize=10, verticalalignment='center',
                horizontalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5,
                         edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gnn_correction_impact.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ Correction impact analysis saved")
    
    def _plot_per_class_performance(self, results, save_dir):
        """Plot per-class performance comparison"""
        
        # Calculate per-class metrics
        mobilevit_report = classification_report(
            results['test_labels'], results['mobilevit_predictions'],
            output_dict=True, zero_division=0
        )
        
        corrected_report = classification_report(
            results['test_labels'], results['corrected_predictions'],
            output_dict=True, zero_division=0
        )
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        fig.suptitle('Per-Class Performance Comparison', fontsize=16, fontweight='bold', y=0.98)
        
        # F1-Score comparison
        classes = [c for c in self.mobilevit_classes if c in mobilevit_report]
        mobilevit_f1 = [mobilevit_report[c]['f1-score'] for c in classes]
        corrected_f1 = [corrected_report[c]['f1-score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, mobilevit_f1, width, label='MobileViT Only',
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, corrected_f1, width, label='GNN Corrected',
                       color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax1.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
        ax1.set_title('F1-Score per Class', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Improvement heatmap
        improvements = [(c - m) for m, c in zip(mobilevit_f1, corrected_f1)]
        
        colors_heat = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvements]
        bars = ax2.barh(classes, improvements, color=colors_heat, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, improvements):
            ax2.text(val + 0.005 if val > 0 else val - 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', va='center', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('F1-Score Improvement', fontsize=11, fontweight='bold')
        ax2.set_title('Per-Class Improvement from GNN Correction', fontsize=13, fontweight='bold', pad=15)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gnn_per_class_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✓ Per-class performance comparison saved")
    
    def save_results(self, results, save_path='gnn_correction_results.json'):
        """Save results to JSON"""
        
        # Convert numpy arrays to lists
        results_serializable = {
            'timestamp': datetime.now().isoformat(),
            'mobilevit_only': results['mobilevit_only'],
            'gnn_corrected': results['gnn_corrected'],
            'correction_stats': results['correction_stats'],
            'confusion_matrices': results['confusion_matrices']
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n📄 Results saved: {save_path}")


def main():
    """Main execution"""
    
    print("🚀 GNN MISCLASSIFICATION CORRECTION & ACCURACY TESTING")
    print("="*70)
    
    # Configuration
    mobilevit_path = "best_mobilevit_waste_model.pth"
    dataset_path = r"realwaste\RealWaste"
    
    # Initialize corrector
    print("\n1️⃣ Initializing correction system...")
    corrector = MisclassificationCorrector(mobilevit_path)
    corrector.load_models()
    
    # Evaluate correction accuracy
    print("\n2️⃣ Evaluating correction accuracy...")
    results = corrector.evaluate_correction_accuracy(dataset_path, max_samples=450)
    
    # Create visualizations
    print("\n3️⃣ Creating visualizations...")
    corrector.visualize_correction_results(results)
    
    # Save results
    print("\n4️⃣ Saving results...")
    corrector.save_results(results)
    
    print("\n✅ GNN correction evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
