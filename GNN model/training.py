"""
Training script for Waste GNN
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

from waste_gnn import WasteGNN, compute_loss


class WasteGNNTrainer:
    """
    Trainer class for Waste GNN
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_volume_loss': [],
            'val_volume_loss': [],
            'train_type_loss': [],
            'val_type_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(self, data, train_mask, optimizer, volume_weight=1.0, type_weight=1.0):
        """
        Train for one epoch
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        
        # Forward pass
        volume_pred, type_pred = self.model(data.x, data.edge_index)
        
        # Apply mask for training nodes
        volume_pred = volume_pred[train_mask]
        type_pred = type_pred[train_mask]
        volume_target = data.y_volume[train_mask]
        type_target = data.y_type[train_mask]
        
        # Compute loss
        loss, volume_loss, type_loss = compute_loss(
            volume_pred, type_pred, volume_target, type_target,
            volume_weight, type_weight
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item(), volume_loss.item(), type_loss.item()
    
    @torch.no_grad()
    def evaluate(self, data, mask):
        """
        Evaluate model on validation/test set
        """
        self.model.eval()
        data = data.to(self.device)
        
        # Forward pass
        volume_pred, type_pred = self.model(data.x, data.edge_index)
        
        # Apply mask
        volume_pred = volume_pred[mask]
        type_pred = type_pred[mask]
        volume_target = data.y_volume[mask]
        type_target = data.y_type[mask]
        
        # Compute loss
        loss, volume_loss, type_loss = compute_loss(
            volume_pred, type_pred, volume_target, type_target
        )
        
        # Compute metrics
        metrics = self._compute_metrics(volume_pred, type_pred, volume_target, type_target)
        
        return loss.item(), volume_loss.item(), type_loss.item(), metrics
    
    def _compute_metrics(self, volume_pred, type_pred, volume_target, type_target):
        """
        Compute evaluation metrics
        """
        # Convert to numpy
        volume_pred_np = volume_pred.cpu().numpy().flatten()
        volume_target_np = volume_target.cpu().numpy()
        type_pred_np = torch.sigmoid(type_pred).cpu().numpy()
        type_target_np = type_target.cpu().numpy()
        
        # Volume metrics (regression)
        volume_mae = mean_absolute_error(volume_target_np, volume_pred_np)
        volume_rmse = np.sqrt(mean_squared_error(volume_target_np, volume_pred_np))
        volume_r2 = r2_score(volume_target_np, volume_pred_np)
        
        # Type metrics (multi-label classification)
        type_pred_binary = (type_pred_np > 0.5).astype(int)
        
        # Convert continuous targets to binary using same threshold
        type_target_binary = (type_target_np > 0.5).astype(int)
        
        # For multi-label classification, use subset accuracy or macro/micro averages
        type_accuracy = 1 - hamming_loss(type_target_binary, type_pred_binary)  # Hamming accuracy
        type_precision = precision_score(type_target_binary, type_pred_binary, average='macro', zero_division=0)
        type_recall = recall_score(type_target_binary, type_pred_binary, average='macro', zero_division=0)
        type_f1 = f1_score(type_target_binary, type_pred_binary, average='macro', zero_division=0)
        
        metrics = {
            'volume_mae': volume_mae,
            'volume_rmse': volume_rmse,
            'volume_r2': volume_r2,
            'type_accuracy': type_accuracy,
            'type_precision': type_precision,
            'type_recall': type_recall,
            'type_f1': type_f1
        }
        
        return metrics
    
    def train(self, data, train_mask, val_mask, 
              epochs=200, 
              lr=0.001,
              weight_decay=5e-4,
              volume_weight=1.0,
              type_weight=1.0,
              patience=20,
              save_path='models'):
        """
        Full training loop
        """
        print(f"Training on device: {self.device}")
        print(f"Number of training nodes: {train_mask.sum()}")
        
        # Check if we have validation data
        has_validation = val_mask.sum() > 0
        if has_validation:
            print(f"Number of validation nodes: {val_mask.sum()}")
        else:
            print("No validation set - using training loss for monitoring")
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="Training"):
            # Train
            train_loss, train_vol_loss, train_type_loss = self.train_epoch(
                data, train_mask, optimizer, volume_weight, type_weight
            )
            
            # Validate (only if validation set exists)
            if has_validation:
                val_loss, val_vol_loss, val_type_loss, val_metrics = self.evaluate(data, val_mask)
                monitoring_loss = val_loss
            else:
                # Use training loss for monitoring when no validation set
                val_loss, val_vol_loss, val_type_loss, val_metrics = train_loss, train_vol_loss, train_type_loss, {}
                monitoring_loss = train_loss
            
            # Scheduler step
            scheduler.step(monitoring_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_volume_loss'].append(train_vol_loss)
            self.history['train_type_loss'].append(train_type_loss)
            
            if has_validation:
                self.history['val_loss'].append(val_loss)
                self.history['val_volume_loss'].append(val_vol_loss)
                self.history['val_type_loss'].append(val_type_loss)
                self.history['val_metrics'].append(val_metrics)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                if has_validation:
                    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                    print(f"Volume RMSE: {val_metrics['volume_rmse']:.2f} | Type F1: {val_metrics['type_f1']:.4f}")
                else:
                    print(f"Train Loss: {train_loss:.4f}")
            
            # Early stopping and model saving
            if monitoring_loss < best_val_loss:
                best_val_loss = monitoring_loss
                patience_counter = 0
                
                # Save best model
                model_save_data = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': monitoring_loss,
                }
                if has_validation:
                    model_save_data['val_loss'] = val_loss
                    model_save_data['val_metrics'] = val_metrics
                
                torch.save(model_save_data, os.path.join(save_path, 'best_model.pt'))
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        checkpoint = torch.load(os.path.join(save_path, 'best_model.pt'), weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if has_validation:
            print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
        else:
            print(f"\nTraining completed. Best training loss: {best_val_loss:.4f}")
        
        return self.history
    
    def plot_training_history(self, save_path='plots'):
        """
        Plot training history
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Check if we have validation data
        has_validation = len(self.history.get('val_loss', [])) > 0
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        if has_validation:
            axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Volume loss
        axes[0, 1].plot(self.history['train_volume_loss'], label='Train')
        if has_validation:
            axes[0, 1].plot(self.history['val_volume_loss'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Volume Loss (MSE)')
        axes[0, 1].set_title('Volume Prediction Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Type loss
        axes[1, 0].plot(self.history['train_type_loss'], label='Train')
        if has_validation:
            axes[1, 0].plot(self.history['val_type_loss'], label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Type Loss (BCE)')
        axes[1, 0].set_title('Waste Type Prediction Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Metrics
        if has_validation and self.history.get('val_metrics'):
            epochs = range(len(self.history['val_metrics']))
            volume_rmse = [m['volume_rmse'] for m in self.history['val_metrics']]
            type_f1 = [m['type_f1'] for m in self.history['val_metrics']]
            
            ax2 = axes[1, 1].twinx()
            axes[1, 1].plot(epochs, volume_rmse, 'b-', label='Volume RMSE')
            ax2.plot(epochs, type_f1, 'r-', label='Type F1 Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Volume RMSE (kg)', color='b')
            ax2.set_ylabel('Type F1 Score', color='r')
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].grid(True)
            
            # Combine legends
            lines1, labels1 = axes[1, 1].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            # Show training loss progression if no validation
            axes[1, 1].plot(self.history['train_loss'], 'g-', label='Training Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Training Loss')
            axes[1, 1].set_title('Training Progress (No Validation)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {save_path}/training_history.png")
    
    def save_results(self, test_metrics, save_path='results'):
        """
        Save final results
        """
        os.makedirs(save_path, exist_ok=True)
        
        results = {
            'test_metrics': test_metrics,
            'training_history': {
                'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
                'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else None
            }
        }
        
        with open(os.path.join(save_path, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {save_path}/results.json")