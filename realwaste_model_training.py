"""
RealWaste Model Training Script
Train Malaysian TinyLlama-SigLIP on waste classification with 70/20/10 split
"""

import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob
from collections import defaultdict
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealWasteDataset(Dataset):
    """Custom dataset for RealWaste classification"""
    
    def __init__(self, data_path, transform=None, max_samples_per_class=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset(max_samples_per_class)
    
    def _load_dataset(self, max_samples_per_class):
        """Load the dataset with proper class mapping"""
        
        # Get all waste categories
        categories = [d for d in os.listdir(self.data_path) 
                     if os.path.isdir(os.path.join(self.data_path, d)) 
                     and not d.startswith('.') 
                     and d not in ['eda_visualizations', 'image_analysis']]
        
        # Create class mappings
        for idx, category in enumerate(sorted(categories)):
            self.class_to_idx[category] = idx
            self.idx_to_class[idx] = category
        
        logger.info(f"Found {len(categories)} categories: {categories}")
        
        # Load samples
        total_samples = 0
        for category in categories:
            category_path = os.path.join(self.data_path, category)
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(category_path, ext)))
                image_files.extend(glob.glob(os.path.join(category_path, ext.upper())))
            
            # Limit samples if specified
            if max_samples_per_class and len(image_files) > max_samples_per_class:
                image_files = random.sample(image_files, max_samples_per_class)
            
            # Add to dataset
            for img_path in image_files:
                self.samples.append({
                    'image_path': img_path,
                    'category': category,
                    'label': self.class_to_idx[category]
                })
                total_samples += 1
        
        logger.info(f"Loaded {total_samples} total samples")
        for category in categories:
            count = len([s for s in self.samples if s['category'] == category])
            logger.info(f"  {category}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'label': sample['label'],
                'category': sample['category'],
                'path': sample['image_path']
            }
        except Exception as e:
            logger.error(f"Error loading image {sample['image_path']}: {e}")
            # Return a dummy sample if image loading fails
            dummy_image = Image.new('RGB', (384, 384), color=(128, 128, 128))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            
            return {
                'image': dummy_image,
                'label': sample['label'],
                'category': sample['category'],
                'path': sample['image_path']
            }

class SimpleWasteClassifier(nn.Module):
    """Simple CNN classifier for waste classification"""
    
    def __init__(self, num_classes=9):
        super(SimpleWasteClassifier, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class WasteModelTrainer:
    """Training manager for waste classification"""
    
    def __init__(self, data_path, device='auto'):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        
        # Training configuration
        self.config = {
            'batch_size': 16,
            'learning_rate': 0.001,
            'epochs': 20,
            'train_split': 0.7,
            'val_split': 0.2,
            'test_split': 0.1,
            'image_size': 384,
            'num_workers': 2
        }
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Training configuration: {self.config}")
        
        # Initialize components
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def create_transforms(self):
        """Create image transforms"""
        try:
            import torchvision.transforms as transforms
            
            # Training transforms (with augmentation)
            train_transform = transforms.Compose([
                transforms.Resize((self.config['image_size'], self.config['image_size'])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Validation/test transforms (no augmentation)
            val_transform = transforms.Compose([
                transforms.Resize((self.config['image_size'], self.config['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            return train_transform, val_transform
            
        except ImportError:
            logger.warning("torchvision not available, using simple transforms")
            # Simple transforms without torchvision
            return self._simple_transform(), self._simple_transform()
    
    def _simple_transform(self):
        """Simple transform function without torchvision"""
        def transform(image):
            # Resize image
            image = image.resize((self.config['image_size'], self.config['image_size']))
            # Convert to tensor
            image_array = np.array(image).astype(np.float32) / 255.0
            # Convert to CHW format
            image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
            return image_tensor
        
        return transform
    
    def prepare_data(self):
        """Prepare the dataset with 70/20/10 split"""
        logger.info("Preparing dataset...")
        
        # Create transforms
        train_transform, val_transform = self.create_transforms()
        
        # Load full dataset
        full_dataset = RealWasteDataset(self.data_path, transform=val_transform)
        self.dataset = full_dataset
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(self.config['train_split'] * total_size)
        val_size = int(self.config['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        logger.info(f"Dataset split: Train({train_size}), Val({val_size}), Test({test_size})")
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        logger.info("Dataset preparation completed")
        return True
    
    def create_model(self):
        """Create and initialize the model"""
        num_classes = len(self.dataset.class_to_idx)
        
        logger.info(f"Creating model with {num_classes} classes")
        
        self.model = SimpleWasteClassifier(num_classes=num_classes)
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )
        
        logger.info("Model created successfully")
        return True
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self):
        """Full training loop"""
        logger.info("Starting training...")
        
        best_val_acc = 0.0
        
        for epoch in range(self.config['epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            logger.info("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_to_idx': self.dataset.class_to_idx,
                }, 'best_waste_model.pth')
                logger.info(f"New best model saved with Val Acc: {val_acc:.2f}%")
        
        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    def test_model(self):
        """Test the trained model"""
        logger.info("Testing model...")
        
        # Load best model
        checkpoint = torch.load('best_waste_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        test_acc = 100. * correct / total
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        
        # Generate classification report
        class_names = [self.dataset.idx_to_class[i] for i in range(len(self.dataset.idx_to_class))]
        report = classification_report(true_labels, predictions, target_names=class_names)
        
        logger.info("Classification Report:")
        logger.info(f"\n{report}")
        
        # Save results
        results = {
            'test_accuracy': test_acc,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels,
            'class_names': class_names
        }
        
        with open('test_results.json', 'w') as f:
            json.dump({
                'test_accuracy': test_acc,
                'classification_report': report,
                'class_names': class_names
            }, f, indent=2)
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Training history plot saved as training_history.png")
    
    def run_full_training(self):
        """Run the complete training pipeline"""
        logger.info("🚀 Starting RealWaste Model Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Prepare data
            if not self.prepare_data():
                return False
            
            # Create model
            if not self.create_model():
                return False
            
            # Train model
            self.train_model()
            
            # Test model
            test_results = self.test_model()
            
            # Plot results
            self.plot_training_history()
            
            logger.info("🎉 Training pipeline completed successfully!")
            logger.info(f"📊 Final Test Accuracy: {test_results['test_accuracy']:.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

def main():
    """Main training function"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Dataset path
    data_path = r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\realwaste\RealWaste"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return
    
    print("🗂️ RealWaste Dataset Model Training")
    print("=" * 50)
    print(f"📁 Dataset: {data_path}")
    print("🎯 Data Split: 70% Train, 20% Validation, 10% Test")
    print("🤖 Model: Simple CNN Classifier")
    
    # Initialize trainer
    trainer = WasteModelTrainer(data_path)
    
    # Run training
    success = trainer.run_full_training()
    
    if success:
        print("\n🎉 Model training completed successfully!")
        print("📁 Generated files:")
        print("   - best_waste_model.pth (trained model)")
        print("   - test_results.json (evaluation results)")
        print("   - training_history.png (training plots)")
    else:
        print("\n❌ Model training failed. Check logs for details.")

if __name__ == "__main__":
    main()