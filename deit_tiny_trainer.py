"""
DeiT-Tiny Training and Comparison with MobileViT
Trains DeiT-Tiny on RealWaste dataset and compares performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DeiTForImageClassification, DeiTImageProcessor
from PIL import Image
import os
import json
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class RealWasteDataset(Dataset):
    """RealWaste dataset loader"""
    
    def __init__(self, image_paths, labels, transform=None, processor=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.processor:
                # Use DeiT processor
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0)
                return pixel_values, label
            elif self.transform:
                # Use custom transform
                image = self.transform(image)
                return image, label
            else:
                return image, label
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.processor:
                black_img = Image.new('RGB', (224, 224), color='black')
                inputs = self.processor(images=black_img, return_tensors="pt")
                return inputs['pixel_values'].squeeze(0), label
            else:
                return torch.zeros(3, 224, 224), label


class DeiTTinyTrainer:
    """Train and evaluate DeiT-Tiny model"""
    
    def __init__(self, num_classes=9, device='cuda'):
        self.num_classes = num_classes
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Class names
        self.class_names = [
            'Cardboard', 'Food Organics', 'Glass', 'Metal',
            'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
        ]
        
        print(f"🔧 Using device: {self.device}")
    
    def load_dataset(self, data_dir='realwaste/RealWaste', split_ratio=(0.7, 0.2, 0.1)):
        """Load and split RealWaste dataset"""
        
        print("\n📂 Loading RealWaste dataset...")
        print(f"   Split ratio: {split_ratio[0]*100}% train, {split_ratio[1]*100}% val, {split_ratio[2]*100}% test")
        
        all_images = []
        all_labels = []
        
        # Load images from each category
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ Warning: Directory not found: {class_dir}")
                continue
            
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                     if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            all_images.extend(images)
            all_labels.extend([class_idx] * len(images))
            
            print(f"   • {class_name}: {len(images)} images")
        
        print(f"\n   Total images: {len(all_images)}")
        
        # Shuffle dataset
        indices = np.random.permutation(len(all_images))
        all_images = [all_images[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        
        # Split dataset
        n_total = len(all_images)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        
        train_images = all_images[:n_train]
        train_labels = all_labels[:n_train]
        
        val_images = all_images[n_train:n_train+n_val]
        val_labels = all_labels[n_train:n_train+n_val]
        
        test_images = all_images[n_train+n_val:]
        test_labels = all_labels[n_train+n_val:]
        
        print(f"\n   📊 Split sizes:")
        print(f"      • Train: {len(train_images)} ({len(train_images)/n_total*100:.1f}%)")
        print(f"      • Validation: {len(val_images)} ({len(val_images)/n_total*100:.1f}%)")
        print(f"      • Test: {len(test_images)} ({len(test_images)/n_total*100:.1f}%)")
        
        return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
    
    def setup_model(self):
        """Initialize DeiT-Tiny model"""
        
        print("\n🤖 Loading DeiT-Tiny model...")
        
        # Load processor
        self.processor = DeiTImageProcessor.from_pretrained('facebook/deit-tiny-patch16-224')
        print("   ✓ Processor loaded")
        
        # Load model
        self.model = DeiTForImageClassification.from_pretrained(
            'facebook/deit-tiny-patch16-224',
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   ✓ Model loaded")
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Trainable parameters: {trainable_params:,}")
        print(f"   • Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(pixel_values=images)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation', leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(pixel_values=images)
                logits = outputs.logits
                
                loss = criterion(logits, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_data, val_data, epochs=50, batch_size=32, lr=1e-4):
        """Full training loop"""
        
        print(f"\n🚀 Starting DeiT-Tiny Training")
        print("="*70)
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {lr}")
        print("="*70)
        
        # Create datasets
        train_images, train_labels = train_data
        val_images, val_labels = val_data
        
        train_dataset = RealWasteDataset(train_images, train_labels, processor=self.processor)
        val_dataset = RealWasteDataset(val_images, val_labels, processor=self.processor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"\nEpoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, 'best_deit_tiny_waste_model.pth')
                print(f"   ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✅ Training completed in {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
        print(f"{'='*70}")
        
        return best_val_acc
    
    def evaluate(self, test_data, batch_size=32):
        """Evaluate on test set"""
        
        print("\n📊 Evaluating on test set...")
        
        test_images, test_labels = test_data
        test_dataset = RealWasteDataset(test_images, test_labels, processor=self.processor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images = images.to(self.device)
                
                outputs = self.model(pixel_values=images)
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        class_report = classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            digits=3,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'test_accuracy': accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'per_class_f1': {
                self.class_names[i]: class_report[self.class_names[i]]['f1-score']
                for i in range(self.num_classes)
            }
        }
        
        print(f"\n{'='*70}")
        print("📈 TEST RESULTS")
        print(f"{'='*70}")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        print(f"{'='*70}")
        
        return results
    
    def save_results(self, results, filename='deit_tiny_waste_results.json'):
        """Save results to JSON"""
        
        # Add training history
        results['training_history'] = self.history
        results['model_info'] = {
            'name': 'DeiT-Tiny',
            'architecture': 'Data-efficient Image Transformer',
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'image_size': 224,
            'patch_size': 16
        }
        results['timestamp'] = datetime.now().isoformat()
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main execution"""
    
    print("🎯 DeiT-Tiny Waste Classification Training")
    print("="*70)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize trainer
    trainer = DeiTTinyTrainer(num_classes=9)
    
    # Load dataset
    train_data, val_data, test_data = trainer.load_dataset(
        data_dir='realwaste/RealWaste',
        split_ratio=(0.7, 0.2, 0.1)
    )
    
    # Setup model
    trainer.setup_model()
    
    # Train model
    best_val_acc = trainer.train(
        train_data, val_data,
        epochs=50,
        batch_size=32,
        lr=1e-4
    )
    
    # Load best model
    print("\n📥 Loading best model for evaluation...")
    checkpoint = torch.load('best_deit_tiny_waste_model.pth')
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    print("   ✓ Best model loaded")
    
    # Evaluate
    results = trainer.evaluate(test_data)
    
    # Save results
    trainer.save_results(results)
    
    print("\n✅ DeiT-Tiny training and evaluation complete!")
    print(f"🎯 Final Test Accuracy: {results['test_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
