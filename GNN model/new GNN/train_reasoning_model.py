"""
Training Script for Waste Reasoning RGN
Includes safety-aware loss functions and hierarchical training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
import json
from waste_reasoning_rgn import (
    WasteReasoningRGN, create_waste_reasoning_model,
    WasteCategory, RiskLevel
)


class WasteDataset(Dataset):
    """Dataset for waste classification with hierarchical labels"""
    
    def __init__(self, 
                 vision_embeddings: np.ndarray,
                 material_labels: np.ndarray,
                 category_labels: np.ndarray,
                 disposal_labels: np.ndarray,
                 risk_labels: np.ndarray):
        """
        Args:
            vision_embeddings: Pre-extracted vision features [N, embedding_dim]
            material_labels: Material type labels [N]
            category_labels: Category labels [N]
            disposal_labels: Disposal method labels [N]
            risk_labels: Risk level labels [N]
        """
        self.vision_embeddings = torch.FloatTensor(vision_embeddings)
        self.material_labels = torch.LongTensor(material_labels)
        self.category_labels = torch.LongTensor(category_labels)
        self.disposal_labels = torch.LongTensor(disposal_labels)
        self.risk_labels = torch.LongTensor(risk_labels)
    
    def __len__(self):
        return len(self.vision_embeddings)
    
    def __getitem__(self, idx):
        return {
            'vision_embedding': self.vision_embeddings[idx],
            'material_label': self.material_labels[idx],
            'category_label': self.category_labels[idx],
            'disposal_label': self.disposal_labels[idx],
            'risk_label': self.risk_labels[idx]
        }


class SafetyAwareLoss(nn.Module):
    """
    Custom loss function that heavily penalizes safety-critical misclassifications
    """
    
    def __init__(self, 
                 safety_weight: float = 10.0,
                 hierarchical_weight: float = 0.5):
        super(SafetyAwareLoss, self).__init__()
        self.safety_weight = safety_weight
        self.hierarchical_weight = hierarchical_weight
        
        # Standard cross-entropy for each classification head
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with safety prioritization
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth labels dictionary
        
        Returns:
            Dictionary of loss components
        """
        # Material classification loss
        material_loss = self.ce_loss(
            outputs['material_logits'],
            targets['material_label']
        )
        
        # Category classification loss
        category_loss = self.ce_loss(
            outputs['category_logits'],
            targets['category_label']
        )
        
        # Disposal classification loss
        disposal_loss = self.ce_loss(
            outputs['disposal_logits'],
            targets['disposal_label']
        )
        
        # Risk prediction loss
        risk_loss = self.ce_loss(
            outputs['risk_scores'],
            targets['risk_label']
        )
        
        # Safety-critical penalty
        # Identify high-risk samples
        high_risk_mask = targets['risk_label'] >= RiskLevel.HIGH_RISK.value
        
        # Apply extra penalty for misclassifying high-risk items
        safety_penalty = torch.zeros_like(category_loss)
        if high_risk_mask.any():
            category_predictions = torch.argmax(outputs['category_logits'], dim=1)
            category_correct = (category_predictions == targets['category_label'])
            
            # Heavy penalty for wrong classification of high-risk items
            safety_penalty[high_risk_mask & ~category_correct] = self.safety_weight
        
        # Hierarchical consistency loss
        # Materials should be consistent with categories
        hierarchy_loss = self._compute_hierarchy_consistency(outputs, targets)
        
        # Combine all losses
        total_loss = (
            material_loss.mean() +
            category_loss.mean() +
            disposal_loss.mean() +
            risk_loss.mean() +
            safety_penalty.mean() +
            self.hierarchical_weight * hierarchy_loss
        )
        
        return {
            'total_loss': total_loss,
            'material_loss': material_loss.mean(),
            'category_loss': category_loss.mean(),
            'disposal_loss': disposal_loss.mean(),
            'risk_loss': risk_loss.mean(),
            'safety_penalty': safety_penalty.mean(),
            'hierarchy_loss': hierarchy_loss
        }
    
    def _compute_hierarchy_consistency(self,
                                      outputs: Dict[str, torch.Tensor],
                                      targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Ensure predictions are consistent across hierarchy levels
        Material → Category → Disposal
        """
        # Get predicted probabilities
        material_probs = torch.softmax(outputs['material_logits'], dim=1)
        category_probs = torch.softmax(outputs['category_logits'], dim=1)
        
        # Define material-category compatibility matrix
        # This encodes which materials should map to which categories
        compatibility_matrix = self._get_compatibility_matrix(
            material_probs.device
        )
        
        # Compute consistency: materials should align with categories
        expected_category_dist = torch.matmul(material_probs, compatibility_matrix)
        
        # KL divergence between expected and predicted category distributions
        consistency_loss = nn.functional.kl_div(
            torch.log(category_probs + 1e-8),
            expected_category_dist,
            reduction='batchmean'
        )
        
        return consistency_loss
    
    def _get_compatibility_matrix(self, device) -> torch.Tensor:
        """
        Create material-to-category compatibility matrix
        Rows: materials (16), Columns: categories (9)
        """
        # Simplified example - in practice, load from knowledge graph
        matrix = torch.zeros((16, 9), device=device)
        
        # Plastic materials → Plastic category (index 0)
        matrix[0:4, 0] = 1.0
        
        # Organic materials → Organic category (index 1)
        matrix[4:6, 1] = 1.0
        
        # Paper materials → Paper category (index 2)
        matrix[6:8, 2] = 1.0
        
        # Glass materials → Glass category (index 3)
        matrix[8:10, 3] = 1.0
        
        # Metal materials → Metal category (index 4)
        matrix[10:12, 4] = 1.0
        
        # Electronic materials → Electronic category (index 5)
        matrix[12:14, 5] = 1.0
        
        # Medical materials → Medical category (index 6)
        matrix[14:16, 6] = 1.0
        
        # Normalize rows
        row_sums = matrix.sum(dim=1, keepdim=True)
        matrix = matrix / (row_sums + 1e-8)
        
        return matrix


class ConflictResolutionLoss(nn.Module):
    """
    Additional loss for training the model to detect conflicting waste items
    """
    
    def __init__(self):
        super(ConflictResolutionLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self,
                conflict_scores: torch.Tensor,
                has_conflicts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conflict_scores: Predicted conflict probabilities [batch_size]
            has_conflicts: Ground truth conflict labels [batch_size]
        
        Returns:
            Conflict detection loss
        """
        return self.bce_loss(conflict_scores, has_conflicts.float())


class WasteReasoningTrainer:
    """
    Trainer for the Waste Reasoning RGN with safety-aware training
    """
    
    def __init__(self,
                 model: WasteReasoningRGN,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 safety_weight: float = 10.0):
        
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss functions
        self.safety_loss = SafetyAwareLoss(safety_weight=safety_weight)
        
        # Optimizer with different learning rates for different components
        self.optimizer = optim.AdamW([
            {'params': model.vision_projection.parameters(), 'lr': learning_rate},
            {'params': model.rgc_layers.parameters(), 'lr': learning_rate * 0.5},
            {'params': model.material_classifier.parameters(), 'lr': learning_rate},
            {'params': model.category_classifier.parameters(), 'lr': learning_rate},
            {'params': model.disposal_classifier.parameters(), 'lr': learning_rate},
            {'params': model.risk_predictor.parameters(), 'lr': learning_rate * 2},  # Higher LR for risk
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Metrics tracking
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        self.safety_violations = []
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'material': 0.0,
            'category': 0.0,
            'disposal': 0.0,
            'risk': 0.0,
            'safety_penalty': 0.0,
            'hierarchy': 0.0
        }
        
        safety_critical_correct = 0
        safety_critical_total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            vision_embedding = batch['vision_embedding'].to(self.device)
            targets = {
                'material_label': batch['material_label'].to(self.device),
                'category_label': batch['category_label'].to(self.device),
                'disposal_label': batch['disposal_label'].to(self.device),
                'risk_label': batch['risk_label'].to(self.device)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(vision_embedding)
            
            # Apply safety rules
            outputs = self.model.apply_safety_rules(outputs)
            
            # Compute loss
            losses = self.safety_loss(outputs, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            for key in epoch_losses.keys():
                if key == 'total':
                    epoch_losses[key] += losses['total_loss'].item()
                elif f'{key}_loss' in losses:
                    epoch_losses[key] += losses[f'{key}_loss'].item()
                elif key == 'safety_penalty':
                    epoch_losses[key] += losses['safety_penalty'].item()
                elif key == 'hierarchy':
                    epoch_losses[key] += losses['hierarchy_loss'].item()
            
            # Track safety-critical accuracy
            high_risk_mask = targets['risk_label'] >= RiskLevel.HIGH_RISK.value
            if high_risk_mask.any():
                predictions = torch.argmax(outputs['category_logits'], dim=1)
                correct = (predictions == targets['category_label'])[high_risk_mask]
                safety_critical_correct += correct.sum().item()
                safety_critical_total += high_risk_mask.sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'safety_acc': f'{safety_critical_correct}/{safety_critical_total}'
            })
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Add safety metrics
        epoch_losses['safety_critical_accuracy'] = (
            safety_critical_correct / max(safety_critical_total, 1)
        )
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'material': 0.0,
            'category': 0.0,
            'disposal': 0.0,
            'risk': 0.0,
            'safety_penalty': 0.0,
            'hierarchy': 0.0
        }
        
        # Confusion matrices
        all_category_preds = []
        all_category_labels = []
        all_risk_labels = []
        
        safety_violations = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                vision_embedding = batch['vision_embedding'].to(self.device)
                targets = {
                    'material_label': batch['material_label'].to(self.device),
                    'category_label': batch['category_label'].to(self.device),
                    'disposal_label': batch['disposal_label'].to(self.device),
                    'risk_label': batch['risk_label'].to(self.device)
                }
                
                # Forward pass
                outputs = self.model(vision_embedding)
                outputs = self.model.apply_safety_rules(outputs)
                
                # Compute loss
                losses = self.safety_loss(outputs, targets)
                
                # Track losses
                for key in val_losses.keys():
                    if key == 'total':
                        val_losses[key] += losses['total_loss'].item()
                    elif f'{key}_loss' in losses:
                        val_losses[key] += losses[f'{key}_loss'].item()
                    elif key == 'safety_penalty':
                        val_losses[key] += losses['safety_penalty'].item()
                    elif key == 'hierarchy':
                        val_losses[key] += losses['hierarchy_loss'].item()
                
                # Collect predictions
                category_preds = torch.argmax(outputs['category_logits'], dim=1)
                all_category_preds.extend(category_preds.cpu().numpy())
                all_category_labels.extend(targets['category_label'].cpu().numpy())
                all_risk_labels.extend(targets['risk_label'].cpu().numpy())
                
                # Check for safety violations
                high_risk_mask = targets['risk_label'] >= RiskLevel.HIGH_RISK.value
                if high_risk_mask.any():
                    incorrect = category_preds != targets['category_label']
                    violations = high_risk_mask & incorrect
                    if violations.any():
                        violation_indices = torch.where(violations)[0]
                        for idx in violation_indices:
                            safety_violations.append({
                                'predicted': category_preds[idx].item(),
                                'actual': targets['category_label'][idx].item(),
                                'risk_level': targets['risk_label'][idx].item()
                            })
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Compute metrics
        all_category_preds = np.array(all_category_preds)
        all_category_labels = np.array(all_category_labels)
        all_risk_labels = np.array(all_risk_labels)
        
        # Overall accuracy
        val_losses['accuracy'] = np.mean(all_category_preds == all_category_labels)
        
        # Safety-critical accuracy
        high_risk_mask = all_risk_labels >= RiskLevel.HIGH_RISK.value
        if high_risk_mask.any():
            safety_acc = np.mean(
                all_category_preds[high_risk_mask] == all_category_labels[high_risk_mask]
            )
            val_losses['safety_critical_accuracy'] = safety_acc
        else:
            val_losses['safety_critical_accuracy'] = 1.0
        
        # Store violations
        self.safety_violations = safety_violations
        
        return val_losses
    
    def train(self, num_epochs: int, save_path: str = 'waste_reasoning_model.pt'):
        """
        Full training loop
        
        Args:
            num_epochs: Number of training epochs
            save_path: Path to save best model
        """
        print("=" * 60)
        print("Starting Waste Reasoning RGN Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_losses['total'])
            
            # Store history
            self.train_history.append(train_losses)
            self.val_history.append(val_losses)
            
            # Print metrics
            print(f"\nTraining Losses:")
            print(f"  Total: {train_losses['total']:.4f}")
            print(f"  Category: {train_losses['category']:.4f}")
            print(f"  Risk: {train_losses['risk']:.4f}")
            print(f"  Safety Penalty: {train_losses['safety_penalty']:.4f}")
            print(f"  Safety-Critical Accuracy: {train_losses['safety_critical_accuracy']:.2%}")
            
            print(f"\nValidation Losses:")
            print(f"  Total: {val_losses['total']:.4f}")
            print(f"  Accuracy: {val_losses['accuracy']:.2%}")
            print(f"  Safety-Critical Accuracy: {val_losses['safety_critical_accuracy']:.2%}")
            
            if self.safety_violations:
                print(f"  ⚠️  Safety Violations: {len(self.safety_violations)}")
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_history': self.train_history,
                    'val_history': self.val_history,
                    'best_val_loss': self.best_val_loss
                }, save_path)
                print(f"  ✓ Saved best model to {save_path}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        return self.train_history, self.val_history


def generate_synthetic_data(num_samples: int = 1000) -> Tuple:
    """
    Generate synthetic training data for demonstration
    In practice, this would be replaced with real annotated waste images
    """
    # Random vision embeddings
    vision_embeddings = np.random.randn(num_samples, 2048).astype(np.float32)
    
    # Generate labels with realistic distributions
    material_labels = np.random.randint(0, 16, num_samples)
    
    # Category labels (correlated with materials)
    category_labels = np.zeros(num_samples, dtype=np.int64)
    category_labels[material_labels < 4] = 0  # Plastic
    category_labels[(material_labels >= 4) & (material_labels < 6)] = 1  # Organic
    category_labels[(material_labels >= 6) & (material_labels < 8)] = 2  # Paper
    category_labels[(material_labels >= 8) & (material_labels < 10)] = 3  # Glass
    category_labels[(material_labels >= 10) & (material_labels < 12)] = 4  # Metal
    category_labels[(material_labels >= 12) & (material_labels < 14)] = 5  # Electronic
    category_labels[material_labels >= 14] = 6  # Medical
    
    # Disposal labels (based on category)
    disposal_labels = np.zeros(num_samples, dtype=np.int64)
    disposal_labels[category_labels <= 4] = 0  # Recyclable
    disposal_labels[category_labels == 1] = 1  # Compostable
    disposal_labels[category_labels >= 5] = 3  # Hazardous
    
    # Risk labels (higher for medical and electronic)
    risk_labels = np.ones(num_samples, dtype=np.int64)
    risk_labels[category_labels == 5] = 3  # Electronic: HIGH_RISK
    risk_labels[category_labels == 6] = 4  # Medical: CRITICAL
    
    return vision_embeddings, material_labels, category_labels, disposal_labels, risk_labels


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic training data...")
    train_data = generate_synthetic_data(num_samples=800)
    val_data = generate_synthetic_data(num_samples=200)
    
    # Create datasets
    train_dataset = WasteDataset(*train_data)
    val_dataset = WasteDataset(*val_data)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("Creating Waste Reasoning RGN...")
    model = create_waste_reasoning_model(vision_embedding_dim=2048)
    
    # Create trainer
    trainer = WasteReasoningTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        safety_weight=10.0
    )
    
    # Train
    train_history, val_history = trainer.train(num_epochs=10)
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump({
            'train': [{k: float(v) if isinstance(v, (int, float)) else v 
                      for k, v in epoch.items()} for epoch in train_history],
            'val': [{k: float(v) if isinstance(v, (int, float)) else v 
                    for k, v in epoch.items()} for epoch in val_history]
        }, f, indent=2)
    
    print("\n✓ Training history saved to training_history.json")