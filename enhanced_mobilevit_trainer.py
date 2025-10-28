"""
MobileViT Enhanced Training with Synthetic Waste Images
Combines GAN-generated synthetic images with real data for improved classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
import os
import json
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class EnhancedWasteDataset(Dataset):
    """Enhanced waste dataset combining real and synthetic images"""
    
    def __init__(self, 
                 real_data_path: str,
                 synthetic_data_path: str = None,
                 synthetic_ratio: float = 0.3,
                 transform=None,
                 include_synthetic: bool = True):
        """
        Initialize enhanced dataset
        
        Args:
            real_data_path: Path to real waste images
            synthetic_data_path: Path to synthetic waste images
            synthetic_ratio: Ratio of synthetic to real images (0.0 - 1.0)
            transform: Image transformations
            include_synthetic: Whether to include synthetic data
        """
        
        self.transform = transform
        self.include_synthetic = include_synthetic
        self.synthetic_ratio = synthetic_ratio
        
        # Load real data
        self.real_images, self.real_labels = self._load_real_data(real_data_path)
        print(f"📸 Loaded {len(self.real_images)} real images")
        
        # Load synthetic data if available
        self.synthetic_images, self.synthetic_labels = [], []
        if include_synthetic and synthetic_data_path and os.path.exists(synthetic_data_path):
            self.synthetic_images, self.synthetic_labels = self._load_synthetic_data(synthetic_data_path)
            print(f"🤖 Loaded {len(self.synthetic_images)} synthetic images")
        
        # Combine datasets
        self.images, self.labels = self._combine_datasets()
        
        # Create class mapping
        unique_labels = sorted(list(set(self.labels)))
        self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}
        self.num_classes = len(unique_labels)
        
        print(f"📊 Dataset summary:")
        print(f"   Total images: {len(self.images)}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Synthetic ratio: {len(self.synthetic_images) / len(self.images) * 100:.1f}%")
        
    def _load_real_data(self, data_path: str):
        """Load real waste images"""
        images, labels = [], []
        
        if not os.path.exists(data_path):
            print(f"⚠️ Real data path not found: {data_path}")
            return images, labels
        
        # Walk through directory structure
        for root, dirs, files in os.walk(data_path):
            # Skip if no image files
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                continue
                
            # Get class name from directory
            class_name = os.path.basename(root).lower()
            
            # Skip certain directories
            if class_name in ['realwaste', 'images', 'testing', 'train', 'val', 'test']:
                continue
                
            # Clean class name
            class_name = class_name.replace(' ', '_').replace('-', '_')
            
            for img_file in image_files:
                img_path = os.path.join(root, img_file)
                try:
                    # Verify image can be opened
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    images.append(img_path)
                    labels.append(class_name)
                    
                except Exception as e:
                    print(f"⚠️ Skipping corrupted image: {img_path}")
                    continue
        
        return images, labels
    
    def _load_synthetic_data(self, synthetic_path: str):
        """Load synthetic waste images"""
        images, labels = [], []
        
        # Load synthetic images organized by class
        for class_dir in os.listdir(synthetic_path):
            class_path = os.path.join(synthetic_path, class_dir)
            
            if not os.path.isdir(class_path):
                continue
                
            class_name = class_dir.lower()
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    images.append(img_path)
                    labels.append(class_name)
        
        return images, labels
    
    def _combine_datasets(self):
        """Combine real and synthetic data with specified ratio"""
        all_images = self.real_images[:]
        all_labels = self.real_labels[:]
        
        if self.include_synthetic and self.synthetic_images:
            # Calculate how many synthetic images to include
            real_count = len(self.real_images)
            synthetic_count = int(real_count * self.synthetic_ratio)
            
            # Randomly sample synthetic images
            if synthetic_count < len(self.synthetic_images):
                indices = np.random.choice(len(self.synthetic_images), synthetic_count, replace=False)
                selected_synthetic = [self.synthetic_images[i] for i in indices]
                selected_labels = [self.synthetic_labels[i] for i in indices]
            else:
                selected_synthetic = self.synthetic_images
                selected_labels = self.synthetic_labels
            
            # Add synthetic data
            all_images.extend(selected_synthetic)
            all_labels.extend(selected_labels)
        
        return all_images, all_labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_name = self.labels[idx]
        label_idx = self.class_to_idx[label_name]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
    
    def get_class_distribution(self):
        """Get distribution of classes in dataset"""
        distribution = defaultdict(int)
        for label in self.labels:
            distribution[label] += 1
        return dict(distribution)


class MobileViTWasteClassifier:
    """Enhanced MobileViT classifier with synthetic data support"""
    
    def __init__(self, model_name: str = "apple/mobilevit-small", num_classes: int = None):
        """
        Initialize MobileViT classifier
        
        Args:
            model_name: HuggingFace model name
            num_classes: Number of waste classes
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Initialize processor
        self.processor = MobileViTImageProcessor.from_pretrained(model_name)
        
        # Initialize model (will be configured when we know num_classes)
        self.model = None
        self.num_classes = num_classes
        
        print(f"🤖 MobileViT Classifier initialized")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
    
    def _setup_model(self, num_classes: int):
        """Setup model with correct number of classes"""
        self.num_classes = num_classes
        
        # Load pre-trained model
        self.model = MobileViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        self.model.to(self.device)
        print(f"📱 Model configured for {num_classes} classes")
    
    def create_transforms(self, is_training: bool = True):
        """Create image transforms for training/validation"""
        
        if is_training:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
            ])
    
    def train_enhanced_model(self,
                           real_data_path: str,
                           synthetic_data_path: str = None,
                           synthetic_ratios: list = [0.0, 0.1, 0.3, 0.5],
                           epochs: int = 10,
                           batch_size: int = 16,
                           learning_rate: float = 1e-4,
                           save_dir: str = "enhanced_mobilevit_results"):
        """
        Train MobileViT with different synthetic data ratios
        
        Args:
            real_data_path: Path to real waste images
            synthetic_data_path: Path to synthetic waste images
            synthetic_ratios: List of synthetic ratios to test
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_dir: Directory to save results
        """
        
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        print(f"🚀 Enhanced MobileViT Training")
        print(f"   Synthetic ratios: {synthetic_ratios}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        
        for ratio in synthetic_ratios:
            print(f"\
📊 Training with synthetic ratio: {ratio}")
            print("=" * 50)
            
            # Create dataset
            train_transform = self.create_transforms(is_training=True)
            
            dataset = EnhancedWasteDataset(
                real_data_path=real_data_path,
                synthetic_data_path=synthetic_data_path,
                synthetic_ratio=ratio,
                transform=train_transform,
                include_synthetic=(ratio > 0)
            )
            
            # Setup model if not done
            if self.model is None:
                self._setup_model(dataset.num_classes)
            
            # Split dataset (80% train, 20% validation)
            total_size = len(dataset)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Train model
            metrics = self._train_epoch_loop(
                train_loader, val_loader, epochs, learning_rate
            )
            
            # Store results
            results[f"ratio_{ratio}"] = {
                "synthetic_ratio": ratio,
                "dataset_size": total_size,
                "train_accuracy": metrics["train_accuracy"],
                "val_accuracy": metrics["val_accuracy"],
                "final_loss": metrics["final_loss"]
            }
            
            # Save model
            model_path = os.path.join(save_dir, f"mobilevit_synthetic_{ratio}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'synthetic_ratio': ratio,
                'class_to_idx': dataset.class_to_idx,
                'metrics': metrics
            }, model_path)
            
            print(f"💾 Model saved: {model_path}")
        
        # Save comparison results
        results_path = os.path.join(save_dir, "synthetic_comparison.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create comparison visualization
        self._visualize_synthetic_comparison(results, save_dir)
        
        print(f"\
✅ Enhanced training completed!")
        print(f"📁 Results saved: {save_dir}")
        
        return results
    
    def _train_epoch_loop(self, train_loader, val_loader, epochs, learning_rate):
        """Training loop for one configuration"""
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        train_accuracies = []
        val_accuracies = []
        losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images).logits
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_loss = train_loss / len(train_loader)
            
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            losses.append(avg_loss)
            
            if epoch % 2 == 0:  # Print every 2 epochs
                print(f"   Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Loss: {avg_loss:.4f}")
        
        return {
            "train_accuracy": train_accuracies[-1],
            "val_accuracy": val_accuracies[-1],
            "final_loss": losses[-1],
            "train_history": train_accuracies,
            "val_history": val_accuracies,
            "loss_history": losses
        }
    
    def _visualize_synthetic_comparison(self, results, save_dir):
        """Create visualization comparing synthetic ratios"""
        
        ratios = []
        train_accs = []
        val_accs = []
        
        for key, metrics in results.items():
            ratios.append(metrics["synthetic_ratio"])
            train_accs.append(metrics["train_accuracy"])
            val_accs.append(metrics["val_accuracy"])
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(ratios, train_accs, 'o-', label='Train Accuracy', linewidth=2, markersize=8)
        plt.plot(ratios, val_accs, 's-', label='Validation Accuracy', linewidth=2, markersize=8)
        plt.xlabel('Synthetic Data Ratio')
        plt.ylabel('Accuracy (%)')
        plt.title('MobileViT Performance vs Synthetic Data Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        dataset_sizes = [results[key]["dataset_size"] for key in results.keys()]
        plt.bar(range(len(ratios)), dataset_sizes, alpha=0.7)
        plt.xlabel('Synthetic Ratio')
        plt.ylabel('Total Dataset Size')
        plt.title('Dataset Size vs Synthetic Ratio')
        plt.xticks(range(len(ratios)), [f"{r:.1f}" for r in ratios])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'synthetic_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plot saved")


def demo_enhanced_training():
    """Demo of enhanced MobileViT training with synthetic data"""
    
    print("🚀 Enhanced MobileViT Training Demo")
    print("=" * 50)
    
    # Paths
    real_data_path = "realwaste/RealWaste"
    synthetic_data_path = "synthetic_outputs/augmentation_dataset"
    
    # Check if paths exist
    if not os.path.exists(real_data_path):
        print(f"❌ Real data path not found: {real_data_path}")
        print("   Please ensure you have the RealWaste dataset")
        return
    
    if not os.path.exists(synthetic_data_path):
        print(f"❌ Synthetic data path not found: {synthetic_data_path}")
        print("   Please run synthetic_waste_generator.py first")
        return
    
    # Initialize classifier
    classifier = MobileViTWasteClassifier()
    
    # Train with different synthetic ratios
    results = classifier.train_enhanced_model(
        real_data_path=real_data_path,
        synthetic_data_path=synthetic_data_path,
        synthetic_ratios=[0.0, 0.2, 0.4],  # Smaller ratios for demo
        epochs=3,  # Fewer epochs for demo
        batch_size=8,  # Smaller batch for memory
        learning_rate=1e-4,
        save_dir="enhanced_mobilevit_demo"
    )
    
    print(f"\
📊 Results Summary:")
    print(f"=" * 30)
    
    for ratio_key, metrics in results.items():
        ratio = metrics["synthetic_ratio"]
        train_acc = metrics["train_accuracy"]
        val_acc = metrics["val_accuracy"]
        print(f"Synthetic Ratio {ratio:.1f}: Train {train_acc:.1f}%, Val {val_acc:.1f}%")
    
    print(f"\
💡 Key Insights:")
    print(f"   • Synthetic data can help when real data is limited")
    print(f"   • Optimal ratio depends on data quality and domain")
    print(f"   • Monitor validation accuracy to avoid overfitting")
    print(f"   • Combine with traditional augmentation techniques")


if __name__ == "__main__":
    demo_enhanced_training()