"""
MobileViT-Small Waste Classification Training
Using the downloaded apple/mobilevit-small model for waste type classification
70% Train | 20% Validation | 10% Test split
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import time
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class WasteDataset(Dataset):
    """Dataset class for waste classification with MobileViT preprocessing"""
    
    def __init__(self, image_paths, labels, processor, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        
        # Additional augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        else:
            self.augment_transform = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]
            
            # Apply augmentation if training
            if self.augment and self.augment_transform:
                image = self.augment_transform(image)
            
            # Process with MobileViT processor
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            
            return pixel_values, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return dummy data in case of error
            dummy_image = torch.zeros(3, 256, 256)
            return dummy_image, torch.tensor(0, dtype=torch.long)

class MobileViTWasteClassifier(nn.Module):
    """MobileViT-based waste classifier"""
    
    def __init__(self, model_path, num_classes=9, freeze_backbone=True):
        super().__init__()
        
        print(f"🤖 Loading MobileViT model from: {model_path}")
        
        # Load pre-trained MobileViT model
        try:
            self.mobilevit = MobileViTForImageClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                ignore_mismatched_sizes=True,
                cache_dir=r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\hf_cache"
            )
        except Exception as e:
            print(f"⚠️  Local model loading failed: {e}")
            print("🔄 Trying to load from HuggingFace...")
            self.mobilevit = MobileViTForImageClassification.from_pretrained(
                "apple/mobilevit-small",
                torch_dtype=torch.float32,
                ignore_mismatched_sizes=True,
                cache_dir=r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\hf_cache"
            )
        
        # Get the hidden size from the model config
        # MobileViT uses neck_hidden_sizes for the final features
        if hasattr(self.mobilevit.config, 'neck_hidden_sizes'):
            hidden_size = self.mobilevit.config.neck_hidden_sizes[-1]  # Last neck layer
        elif hasattr(self.mobilevit.config, 'hidden_sizes'):
            hidden_size = self.mobilevit.config.hidden_sizes[-1]  # Last hidden size
        else:
            hidden_size = 640  # Default for MobileViT-small
        
        print(f"🔍 Using hidden size: {hidden_size}")
        
        # Remove the original classifier
        self.mobilevit.classifier = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.mobilevit.mobilevit.parameters():
                param.requires_grad = False
            print("🔒 Backbone frozen - only training classifier")
        else:
            print("🔓 Full model training enabled")
        
        # Custom classifier for waste classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, pixel_values):
        # Extract features from MobileViT
        outputs = self.mobilevit.mobilevit(pixel_values)
        
        # Get the pooled representation
        # MobileViT outputs different structure than typical transformers
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # For MobileViT, we need to pool the feature maps
            last_hidden_state = outputs.last_hidden_state
            if len(last_hidden_state.shape) == 4:  # [B, C, H, W]
                features = last_hidden_state.mean(dim=[2, 3])  # Global average pooling
            elif len(last_hidden_state.shape) == 3:  # [B, S, C]
                features = last_hidden_state.mean(dim=1)
            else:
                features = last_hidden_state
        else:
            # Fallback: use the outputs directly and try to pool
            if isinstance(outputs, tuple):
                features = outputs[0]
            else:
                features = outputs
            
            # Ensure proper pooling for feature maps
            if len(features.shape) == 4:  # [B, C, H, W]
                features = features.mean(dim=[2, 3])
            elif len(features.shape) == 3:  # [B, S, C]
                features = features.mean(dim=1)
        
        # Ensure features are 2D [batch_size, feature_dim]
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        # Classify
        logits = self.classifier(features)
        return logits

class WasteTrainer:
    """Trainer class for waste classification"""
    
    def __init__(self, model, device, class_names):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                accuracy = 100. * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.2f}%',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                try:
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    running_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                    # Update progress bar
                    accuracy = 100. * correct / total
                    avg_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0
                    pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'Acc': f'{accuracy:.2f}%'
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        epoch_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0
        epoch_acc = 100. * correct / total if total > 0 else 0
        return epoch_loss, epoch_acc, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs=20, lr=0.001, patience=7):
        """Full training loop"""
        
        # Setup optimizer with different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': lr}       # Higher LR for classifier
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"🚀 Starting MobileViT waste classification training")
        print(f"📊 Device: {self.device}")
        print(f"🎯 Classes: {len(self.class_names)}")
        print(f"📅 Epochs: {epochs}")
        print(f"📈 Learning rate: {lr}")
        print(f"⏰ Patience: {patience}")
        
        best_val_acc = 0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validation phase
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch(val_loader, criterion, epoch)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch results
            print(f"\n📊 Epoch {epoch+1}/{epochs} Results ({epoch_time:.1f}s):")
            print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'class_names': self.class_names
                }, 'best_mobilevit_waste_model.pth')
                print(f"   💾 New best model saved! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   ⏰ Early stopping triggered after {patience} epochs without improvement")
                    break
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
        
        return best_val_acc
    
    def test_model(self, test_loader):
        """Test the best model"""
        print("\n🧪 Testing best model...")
        
        # Load best model
        checkpoint = torch.load('best_mobilevit_waste_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_preds = []
        all_targets = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                
                try:
                    output = self.model(data)
                    _, predicted = output.max(1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                except Exception as e:
                    print(f"Error in test batch: {e}")
                    continue
        
        test_acc = 100. * correct / total if total > 0 else 0
        print(f"🎯 Final Test Accuracy: {test_acc:.2f}%")
        
        return all_preds, all_targets, test_acc
    
    def plot_results(self, test_preds, test_targets):
        """Plot training history and confusion matrix"""
        
        # Plot training history
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss plot
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Confusion Matrix
        cm = confusion_matrix(test_targets, test_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=self.class_names, yticklabels=self.class_names)
        ax3.set_title('Confusion Matrix - Test Set')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Accuracy per class
        accuracies = []
        for i in range(len(self.class_names)):
            if i < len(cm):
                class_correct = cm[i, i] if cm[i, i:].sum() > 0 else 0
                class_total = cm[i, :].sum() if cm[i, :].sum() > 0 else 1
                acc = 100 * class_correct / class_total
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        bars = ax4.bar(range(len(self.class_names)), accuracies, color='skyblue')
        ax4.set_title('Accuracy per Class')
        ax4.set_xlabel('Waste Type')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_xticks(range(len(self.class_names)))
        ax4.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('mobilevit_waste_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Results visualization saved as 'mobilevit_waste_results.png'")

def load_realwaste_dataset(data_dir):
    """Load RealWaste dataset with 70/20/10 split"""
    
    print(f"📁 Loading RealWaste dataset from: {data_dir}")
    
    # Get waste categories
    categories = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d)) 
                 and not d.startswith('.') 
                 and d not in ['eda_visualizations', 'image_analysis']]
    
    categories.sort()  # Ensure consistent ordering
    print(f"📊 Found {len(categories)} waste categories:")
    for i, cat in enumerate(categories):
        print(f"   {i}: {cat}")
    
    # Load all images and labels
    all_images = []
    all_labels = []
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(category_path, ext)))
        
        all_images.extend(images)
        all_labels.extend([idx] * len(images))
        
        print(f"   {category}: {len(images)} images")
    
    print(f"\n📈 Total images loaded: {len(all_images)}")
    
    # Create 70/20/10 split
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, all_labels, 
        test_size=0.3, 
        random_state=42, 
        stratify=all_labels
    )
    
    # Second split: 20% val, 10% test from the 30% temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.333,  # 10/30 = 0.333 to get 10% of total
        random_state=42, 
        stratify=y_temp
    )
    
    print(f"\n📊 Dataset splits:")
    print(f"   Training:   {len(X_train)} images ({len(X_train)/len(all_images)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} images ({len(X_val)/len(all_images)*100:.1f}%)")
    print(f"   Testing:    {len(X_test)} images ({len(X_test)/len(all_images)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, categories

def main():
    """Main training function"""
    
    print("🗂️  MobileViT-Small Waste Classification Training")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💾 Device: {device}")
    
    # Model and data paths
    mobilevit_path = r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\hf_cache\models--apple--mobilevit-small\snapshots\6c8612545dc98676d27bc518fa521d9e9af23e34"
    data_path = r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\realwaste\RealWaste"
    
    # Check if paths exist
    if not os.path.exists(mobilevit_path):
        print(f"❌ MobileViT model not found at: {mobilevit_path}")
        # Try alternative path with model name
        mobilevit_path = "apple/mobilevit-small"
        print(f"🔄 Trying to load from HuggingFace: {mobilevit_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ RealWaste dataset not found at: {data_path}")
        return
    
    print(f"✅ MobileViT model path: {mobilevit_path}")
    print(f"✅ RealWaste dataset found at: {data_path}")
    
    # Load dataset
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_realwaste_dataset(data_path)
    
    # Load MobileViT processor
    print(f"\n🔧 Loading MobileViT processor...")
    try:
        processor = MobileViTImageProcessor.from_pretrained(
            mobilevit_path,
            cache_dir=r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\hf_cache"
        )
        print("✅ MobileViT processor loaded successfully")
    except Exception as e:
        print(f"⚠️  Local processor failed: {e}")
        print("🔄 Trying to load from HuggingFace...")
        try:
            processor = MobileViTImageProcessor.from_pretrained(
                "apple/mobilevit-small",
                cache_dir=r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\hf_cache"
            )
            print("✅ MobileViT processor loaded from HuggingFace")
        except Exception as e2:
            print(f"❌ Error loading processor from HuggingFace: {e2}")
            return
    
    # Create datasets
    print(f"\n📊 Creating datasets...")
    train_dataset = WasteDataset(X_train, y_train, processor, augment=True)
    val_dataset = WasteDataset(X_val, y_val, processor, augment=False)
    test_dataset = WasteDataset(X_test, y_test, processor, augment=False)
    
    # Create data loaders
    batch_size = 8 if device.type == 'cpu' else 16
    num_workers = 0 if device.type == 'cpu' else 2
    
    print(f"📦 Batch size: {batch_size}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    
    # Create model
    print(f"\n🤖 Creating MobileViT waste classifier...")
    try:
        model = MobileViTWasteClassifier(
            model_path=mobilevit_path,
            num_classes=len(class_names),
            freeze_backbone=True  # Start with frozen backbone for faster training
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully!")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        print(f"⚡ Trainable ratio: {trainable_params/total_params*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Create trainer
    trainer = WasteTrainer(model, device, class_names)
    
    # Training parameters
    epochs = 25
    learning_rate = 0.001
    patience = 8
    
    print(f"\n🚀 Training configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Patience: {patience}")
    print(f"   Classes: {class_names}")
    
    # Start training
    start_time = time.time()
    best_val_acc = trainer.train(
        train_loader, val_loader, 
        epochs=epochs, lr=learning_rate, patience=patience
    )
    training_time = time.time() - start_time
    
    # Test the model
    test_preds, test_targets, test_acc = trainer.test_model(test_loader)
    
    # Generate detailed classification report
    print(f"\n📊 Detailed Classification Results:")
    print("=" * 60)
    report = classification_report(
        test_targets, test_preds, 
        target_names=class_names, 
        digits=3, zero_division=0
    )
    print(report)
    
    # Save results to file
    results = {
        'model': 'MobileViT-Small',
        'dataset': 'RealWaste',
        'data_split': {'train': 70, 'val': 20, 'test': 10},
        'total_images': len(X_train) + len(X_val) + len(X_test),
        'training_time_minutes': training_time / 60,
        'best_validation_accuracy': best_val_acc,
        'final_test_accuracy': test_acc,
        'class_names': class_names,
        'classification_report': report
    }
    
    with open('mobilevit_waste_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    trainer.plot_results(test_preds, test_targets)
    
    # Final summary
    print(f"\n🎉 MobileViT Waste Classification Training Complete!")
    print("=" * 60)
    print(f"📊 Final Results:")
    print(f"   • Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   • Final Test Accuracy: {test_acc:.2f}%")
    print(f"   • Training Time: {training_time/60:.1f} minutes")
    print(f"   • Total Parameters: {total_params:,}")
    print(f"   • Trainable Parameters: {trainable_params:,}")
    
    print(f"\n💾 Files Generated:")
    print(f"   • best_mobilevit_waste_model.pth - Best trained model")
    print(f"   • mobilevit_waste_results.png - Visualization results")
    print(f"   • mobilevit_waste_results.json - Detailed results")
    
    print(f"\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()