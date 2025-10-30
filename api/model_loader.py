"""
Model loader that properly handles your custom trained MobileViT model
"""
import torch
import torch.nn as nn
from transformers import MobileViTModel
import logging

logger = logging.getLogger(__name__)


class CustomMobileViTClassifier(nn.Module):
    """
    Custom MobileViT classifier matching your training setup
    """
    def __init__(self, num_classes=9):
        super().__init__()
        
        # Load base MobileViT model (without classification head)
        self.mobilevit = MobileViTModel.from_pretrained("apple/mobilevit-small")
        
        # Custom classifier head (matching your training code)
        hidden_size = self.mobilevit.config.neck_hidden_sizes[-1]  # 640 for mobilevit-small
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 0 - Pooling
            nn.Linear(hidden_size, 512),  # 1 - 640 -> 512
            nn.BatchNorm1d(512),  # 2
            nn.ReLU(),  # 3
            nn.Dropout(0.3),  # 4
            nn.Linear(512, 256),  # 5 - 512 -> 256
            nn.BatchNorm1d(256),  # 6
            nn.ReLU(),  # 7
            nn.Dropout(0.2),  # 8
            nn.Linear(256, num_classes)  # 9 - 256 -> num_classes
        )
    
    def forward(self, pixel_values):
        # Get features from MobileViT
        outputs = self.mobilevit(pixel_values=pixel_values)
        features = outputs.last_hidden_state  # Shape: (batch, channels, height, width) = (B, 640, 8, 8)
        
        # Apply pooling (part of classifier[0])
        features = self.classifier[0](features)  # AdaptiveAvgPool2d -> (batch, 640, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten -> (batch, 640)
        
        # Apply rest of classifier (starting from index 1: Linear layers, etc.)
        logits = features
        for layer in self.classifier[1:]:
            logits = layer(logits)
        
        return logits


def load_trained_model(model_path, device='cpu', num_classes=9):
    """
    Load your trained MobileViT model
    
    Args:
        model_path: Path to best_mobilevit_waste_model.pth
        device: 'cpu' or 'cuda'
        num_classes: Number of waste classes (9)
    
    Returns:
        Loaded model in eval mode
    """
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Get saved class names
        if 'class_names' in checkpoint:
            saved_classes = checkpoint['class_names']
            logger.info(f"Saved classes: {saved_classes}")
            num_classes = len(saved_classes)
        
        # Create model
        model = CustomMobileViTClassifier(num_classes=num_classes)
        
        # Load state dict
        state_dict = checkpoint['model_state_dict']
        
        # Load weights (strict=False to handle any mismatches)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Log training info
        if 'epoch' in checkpoint:
            logger.info(f"Model trained for {checkpoint['epoch']} epochs")
        if 'best_val_acc' in checkpoint:
            logger.info(f"Best validation accuracy: {checkpoint['best_val_acc']:.2%}")
        
        logger.info("✅ Model loaded successfully!")
        
        return model, saved_classes if 'class_names' in checkpoint else None
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)
    
    model, classes = load_trained_model("../best_mobilevit_waste_model.pth")
    print(f"\n✅ Model loaded successfully!")
    print(f"Classes: {classes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
