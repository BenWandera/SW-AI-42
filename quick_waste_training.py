"""
Quick RealWaste Model Training - Lightweight Version
Fast training script for immediate results and visualization
"""

import os
import json
import random
import numpy as np
from datetime import datetime
import glob
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SimpleWasteDataset(Dataset):
    """Lightweight dataset for quick training"""
    
    def __init__(self, data_path, transform=None, max_samples_per_class=100):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset(max_samples_per_class)
    
    def _load_dataset(self, max_samples_per_class):
        # Get categories
        categories = [d for d in os.listdir(self.data_path) 
                     if os.path.isdir(os.path.join(self.data_path, d)) 
                     and not d.startswith('.') 
                     and d not in ['eda_visualizations', 'image_analysis']]
        
        # Create mappings
        for idx, category in enumerate(sorted(categories)):
            self.class_to_idx[category] = idx
            self.idx_to_class[idx] = category
        
        print(f"Categories: {categories}")
        
        # Load limited samples
        for category in categories:
            category_path = os.path.join(self.data_path, category)
            
            image_files = []
            for ext in ['*.jpg', '*.jpeg']:
                image_files.extend(glob.glob(os.path.join(category_path, ext)))
            
            # Limit samples
            if len(image_files) > max_samples_per_class:
                image_files = random.sample(image_files, max_samples_per_class)
            
            for img_path in image_files:
                self.samples.append({
                    'image_path': img_path,
                    'label': self.class_to_idx[category],
                    'category': category
                })
        
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image, sample['label']
        except:
            # Return dummy if error
            dummy = torch.zeros(3, 224, 224)
            return dummy, sample['label']

class QuickWasteNet(nn.Module):
    """Lightweight CNN for quick training"""
    
    def __init__(self, num_classes=9):
        super(QuickWasteNet, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class QuickTrainer:
    """Fast trainer for demonstration"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        
        # Quick config
        self.config = {
            'batch_size': 32,
            'learning_rate': 0.01,
            'epochs': 10,
            'image_size': 224,
            'samples_per_class': 50  # Limited for speed
        }
        
        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_acc': [], 'val_acc': [], 'test_acc': []
        }
    
    def create_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def prepare_data(self):
        print("\\nPreparing quick dataset...")
        
        train_transform, val_transform = self.create_transforms()
        
        # Load dataset
        dataset = SimpleWasteDataset(
            self.data_path, 
            transform=val_transform,
            max_samples_per_class=self.config['samples_per_class']
        )
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        print(f"Splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Create loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        self.class_names = [dataset.idx_to_class[i] for i in range(len(dataset.idx_to_class))]
        return True
    
    def create_model(self):
        print("\\nCreating quick model...")
        
        self.model = QuickWasteNet(num_classes=len(self.class_names))
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        
        # Count parameters
        params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {params:,}")
        
        return True
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return running_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, loader, desc="Validation"):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=desc):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return running_loss / len(loader), 100. * correct / total, all_preds, all_labels
    
    def train_model(self):
        print("\\nStarting quick training...")
        
        for epoch in range(self.config['epochs']):
            print(f"\\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(self.val_loader, "Validation")
            
            # Test every few epochs
            if (epoch + 1) % 3 == 0:
                test_loss, test_acc, _, _ = self.validate(self.test_loader, "Testing")
            else:
                test_loss, test_acc = 0, 0
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['test_loss'].append(test_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['test_acc'].append(test_acc)
            
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            if test_acc > 0:
                print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
    
    def final_test(self):
        print("\\nFinal evaluation...")
        
        test_loss, test_acc, test_preds, test_labels = self.validate(self.test_loader, "Final Test")
        
        # Classification report
        report = classification_report(test_labels, test_preds, target_names=self.class_names)
        print("\\nClassification Report:")
        print(report)
        
        return test_loss, test_acc, test_preds, test_labels
    
    def create_visualizations(self, test_preds=None, test_labels=None):
        print("\\nCreating visualizations...")
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RealWaste Quick Training Results', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. Training Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        test_epochs = [i for i, loss in enumerate(self.history['test_loss'], 1) if loss > 0]
        test_losses = [loss for loss in self.history['test_loss'] if loss > 0]
        if test_losses:
            axes[0, 0].plot(test_epochs, test_losses, 'g-', label='Test', linewidth=2, marker='o')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val', linewidth=2)
        test_accs = [acc for acc in self.history['test_acc'] if acc > 0]
        if test_accs:
            axes[0, 1].plot(test_epochs, test_accs, 'g-', label='Test', linewidth=2, marker='o')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Performance Comparison
        final_metrics = {
            'Train': self.history['train_acc'][-1],
            'Val': self.history['val_acc'][-1],
            'Test': max([acc for acc in self.history['test_acc'] if acc > 0], default=0)
        }
        
        bars = axes[0, 2].bar(final_metrics.keys(), final_metrics.values(), 
                             color=['blue', 'red', 'green'], alpha=0.7)
        axes[0, 2].set_title('Final Performance Comparison')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].set_ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Loss vs Accuracy
        axes[1, 0].plot(self.history['train_loss'], self.history['train_acc'], 'bo-', label='Train')
        axes[1, 0].plot(self.history['val_loss'], self.history['val_acc'], 'ro-', label='Val')
        axes[1, 0].set_title('Loss vs Accuracy')
        axes[1, 0].set_xlabel('Loss')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Training Progress
        val_improvement = np.diff(self.history['val_acc']) if len(self.history['val_acc']) > 1 else [0]
        axes[1, 1].plot(range(2, len(epochs) + 1), val_improvement, 'r-', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Validation Accuracy Improvement')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Change (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Training Statistics
        stats_text = f"""
Quick Training Statistics

Total Epochs: {len(epochs)}
Best Val Acc: {max(self.history['val_acc']):.2f}%
Final Test Acc: {final_metrics['Test']:.2f}%

Dataset Split:
• Train: 70%
• Validation: 20% 
• Test: 10%

Classes: {len(self.class_names)}
Samples/Class: {self.config['samples_per_class']}
Device: {self.device}
        """
        axes[1, 2].text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
                        transform=axes[1, 2].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('quick_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion Matrix (if test data available)
        if test_preds is not None and test_labels is not None:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(test_labels, test_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Quick Training - Confusion Matrix', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('quick_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("Visualizations saved:")
        print("• quick_training_results.png")
        print("• quick_confusion_matrix.png")
    
    def run_quick_training(self):
        print("🚀 Quick RealWaste Training Demo")
        print("=" * 50)
        
        try:
            # Prepare data
            if not self.prepare_data():
                return False
            
            # Create model
            if not self.create_model():
                return False
            
            # Train
            self.train_model()
            
            # Final test
            test_loss, test_acc, test_preds, test_labels = self.final_test()
            
            # Visualizations
            self.create_visualizations(test_preds, test_labels)
            
            print(f"\\n🎉 Quick training completed!")
            print(f"📊 Final test accuracy: {test_acc:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

def main():
    data_path = r"C:\\Users\\Z-BOOK\\OneDrive\\Documents\\DATASETS\\realwaste\\RealWaste"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return
    
    trainer = QuickTrainer(data_path)
    trainer.run_quick_training()

if __name__ == "__main__":
    main()