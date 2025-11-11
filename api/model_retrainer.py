"""
Model Retraining System for Active Learning
Incrementally improves MobileViT model with user feedback
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image
import logging
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class FeedbackDataset(Dataset):
    """Dataset from user feedback for retraining"""
    
    def __init__(self, feedback_samples: List[Dict], processor, class_to_idx: Dict):
        self.samples = feedback_samples
        self.processor = processor
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = sample["image_path"]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224))
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Get label
        correct_class = sample["correct_class"]
        label = self.class_to_idx.get(correct_class, 0)
        
        return pixel_values, label


class ModelRetrainer:
    """Handle incremental model retraining"""
    
    def __init__(
        self,
        model_path: str = "best_mobilevit_waste_model.pth",
        backup_dir: str = "model_backups",
        learning_rate: float = 1e-5,  # Lower LR for fine-tuning
        device: str = None
    ):
        self.model_path = Path(model_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        self.learning_rate = learning_rate
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        
        logger.info(f"🔧 ModelRetrainer initialized on {self.device}")
    
    def backup_current_model(self) -> str:
        """Create backup of current model"""
        if not self.model_path.exists():
            logger.warning(f"⚠️ Model file not found: {self.model_path}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"model_backup_{timestamp}.pth"
        
        try:
            shutil.copy2(self.model_path, backup_path)
            logger.info(f"✅ Model backed up to: {backup_path}")
            
            # Also save metadata
            metadata = {
                "original_path": str(self.model_path),
                "backup_timestamp": timestamp,
                "backup_date": datetime.now().isoformat()
            }
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return str(backup_path)
        except Exception as e:
            logger.error(f"❌ Error backing up model: {e}")
            return None
    
    def load_model(self, class_names: List[str]):
        """Load model for retraining"""
        try:
            # Load model state
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model
            num_classes = len(class_names)
            model = MobileViTForImageClassification.from_pretrained(
                "apple/mobilevit-small",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            
            # Load weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Loaded model from checkpoint")
            else:
                model.load_state_dict(checkpoint)
                logger.info("✅ Loaded model from state dict")
            
            model.to(self.device)
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return None
    
    def retrain(
        self,
        feedback_samples: List[Dict],
        class_names: List[str],
        epochs: int = 3,
        batch_size: int = 8,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Retrain model with feedback samples
        
        Args:
            feedback_samples: List of feedback samples
            class_names: List of class names
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction for validation
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"🚀 Starting model retraining...")
        logger.info(f"   📊 Samples: {len(feedback_samples)}")
        logger.info(f"   🏷️ Classes: {len(class_names)}")
        logger.info(f"   📈 Epochs: {epochs}")
        
        # Backup current model
        backup_path = self.backup_current_model()
        
        try:
            # Create class mapping
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            
            # Split data
            split_idx = int(len(feedback_samples) * (1 - validation_split))
            train_samples = feedback_samples[:split_idx]
            val_samples = feedback_samples[split_idx:]
            
            logger.info(f"   📂 Train: {len(train_samples)}, Val: {len(val_samples)}")
            
            # Create datasets
            train_dataset = FeedbackDataset(train_samples, self.processor, class_to_idx)
            val_dataset = FeedbackDataset(val_samples, self.processor, class_to_idx) if val_samples else None
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0  # Use 0 for Windows compatibility
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            ) if val_dataset else None
            
            # Load model
            model = self.load_model(class_names)
            if model is None:
                raise Exception("Failed to load model")
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
            
            # Training loop
            best_val_acc = 0.0
            training_history = []
            
            for epoch in range(epochs):
                logger.info(f"\n📚 Epoch {epoch + 1}/{epochs}")
                
                # Train
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs).logits
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                    
                    if (batch_idx + 1) % 5 == 0:
                        logger.info(f"   Batch {batch_idx + 1}/{len(train_loader)}: Loss={loss.item():.4f}")
                
                train_acc = 100.0 * train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)
                
                logger.info(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                
                # Validation
                val_acc = 0.0
                val_loss = 0.0
                
                if val_loader:
                    model.eval()
                    val_loss_sum = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.device)
                            
                            outputs = model(inputs).logits
                            loss = criterion(outputs, labels)
                            
                            val_loss_sum += loss.item()
                            _, predicted = outputs.max(1)
                            val_total += labels.size(0)
                            val_correct += predicted.eq(labels).sum().item()
                    
                    val_acc = 100.0 * val_correct / val_total
                    val_loss = val_loss_sum / len(val_loader)
                    
                    logger.info(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                    
                    # Save best model
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        logger.info(f"   ✨ New best validation accuracy!")
                
                # Record history
                training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
            
            # Save retrained model
            logger.info(f"\n💾 Saving retrained model...")
            save_dict = {
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'num_classes': len(class_names),
                'retrain_timestamp': datetime.now().isoformat(),
                'training_samples': len(feedback_samples),
                'backup_path': backup_path
            }
            
            torch.save(save_dict, self.model_path)
            logger.info(f"✅ Model saved to: {self.model_path}")
            
            # Return results
            results = {
                "success": True,
                "backup_path": backup_path,
                "epochs": epochs,
                "training_samples": len(train_samples),
                "validation_samples": len(val_samples),
                "final_train_acc": training_history[-1]["train_acc"],
                "final_val_acc": training_history[-1]["val_acc"],
                "best_val_acc": best_val_acc,
                "training_history": training_history,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"\n✅ Retraining complete!")
            logger.info(f"   📊 Final Train Acc: {results['final_train_acc']:.2f}%")
            logger.info(f"   📊 Final Val Acc: {results['final_val_acc']:.2f}%")
            logger.info(f"   ⭐ Best Val Acc: {results['best_val_acc']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during retraining: {e}", exc_info=True)
            
            # Restore backup if training failed
            if backup_path and Path(backup_path).exists():
                logger.info(f"🔄 Restoring model from backup...")
                shutil.copy2(backup_path, self.model_path)
                logger.info(f"✅ Model restored")
            
            return {
                "success": False,
                "error": str(e),
                "backup_path": backup_path,
                "timestamp": datetime.now().isoformat()
            }
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore model from a specific backup"""
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            logger.error(f"❌ Backup not found: {backup_path}")
            return False
        
        try:
            shutil.copy2(backup_path, self.model_path)
            logger.info(f"✅ Model restored from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error restoring model: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List all model backups"""
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("model_backup_*.pth"), reverse=True):
            metadata_file = backup_file.with_suffix('.json')
            metadata = {}
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            backups.append({
                "path": str(backup_file),
                "filename": backup_file.name,
                "size_mb": backup_file.stat().st_size / (1024 * 1024),
                "created": metadata.get("backup_date", "Unknown"),
                "metadata": metadata
            })
        
        return backups
