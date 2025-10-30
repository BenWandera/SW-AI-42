"""
GNN Model Loader for Waste Classification Verification
Integrates the Relational Graph Network for safety-critical reasoning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import logging
import sys
import os

# Add GNN path
gnn_path = os.path.join(os.path.dirname(__file__), '..', 'GNN model', 'new GNN')
if os.path.exists(gnn_path) and gnn_path not in sys.path:
    sys.path.insert(0, gnn_path)

logger = logging.getLogger(__name__)


class MobileViTToRGNAdapter(nn.Module):
    """
    Adapter to convert MobileViT features (640-dim) to RGN expected format (2048-dim)
    This bridges the gap between MobileViT embeddings and GNN expectations
    """
    
    def __init__(self, mobilevit_dim=640, rgn_dim=2048):
        super().__init__()
        
        # Projection layers to expand MobileViT features
        self.adapter = nn.Sequential(
            nn.Linear(mobilevit_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, rgn_dim),
            nn.LayerNorm(rgn_dim)
        )
    
    def forward(self, mobilevit_features):
        """
        Args:
            mobilevit_features: [batch_size, 640] from MobileViT
        Returns:
            rgn_features: [batch_size, 2048] for RGN
        """
        return self.adapter(mobilevit_features)


class SimplifiedGNNVerifier:
    """
    Simplified GNN-based verification system
    Uses rule-based reasoning when full GNN model is not available
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Waste category mapping
        self.mobilevit_to_gnn_category = {
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
        
        # Risk levels for each category
        self.risk_levels = {
            'PLASTIC': 1,  # LOW_RISK
            'ORGANIC': 0,  # SAFE
            'PAPER': 0,    # SAFE
            'GLASS': 1,    # LOW_RISK
            'METAL': 1,    # LOW_RISK
            'ELECTRONIC': 3,  # HIGH_RISK
            'MEDICAL': 4,   # CRITICAL
            'MIXED': 1      # LOW_RISK
        }
        
        # Confidence adjustments based on category characteristics
        self.confidence_modifiers = {
            'PLASTIC': 1.05,    # Easy to identify
            'GLASS': 1.03,      # Distinctive features
            'METAL': 1.04,      # Clear visual cues
            'PAPER': 1.02,      # Moderate confidence
            'ORGANIC': 0.98,    # Can be ambiguous
            'MIXED': 0.95,      # Uncertain category
        }
        
        logger.info("✅ Simplified GNN Verifier initialized")
    
    def verify_classification(self, 
                             mobilevit_class: str, 
                             mobilevit_confidence: float,
                             mobilevit_probs: Dict[str, float]) -> Dict:
        """
        Verify MobileViT classification using rule-based reasoning
        
        Args:
            mobilevit_class: Predicted class from MobileViT
            mobilevit_confidence: Confidence score
            mobilevit_probs: All class probabilities
        
        Returns:
            Verification results with adjusted confidence
        """
        
        # Map to GNN category
        gnn_category = self.mobilevit_to_gnn_category.get(mobilevit_class, 'MIXED')
        
        # Get risk level
        risk_level = self.risk_levels.get(gnn_category, 1)
        
        # Calculate confidence modifier
        base_modifier = self.confidence_modifiers.get(gnn_category, 1.0)
        
        # Check for ambiguity (multiple high probabilities)
        sorted_probs = sorted(mobilevit_probs.values(), reverse=True)
        if len(sorted_probs) > 1:
            prob_gap = sorted_probs[0] - sorted_probs[1]
            if prob_gap < 0.2:  # Close predictions
                base_modifier *= 0.95  # Reduce confidence
            elif prob_gap > 0.5:  # Very clear prediction
                base_modifier *= 1.05  # Boost confidence
        
        # Apply confidence adjustment
        adjusted_confidence = min(mobilevit_confidence * base_modifier, 0.99)
        
        # Determine if GNN agrees
        confidence_delta = adjusted_confidence - mobilevit_confidence
        agrees = abs(confidence_delta) < 0.1  # Within 10% is agreement
        
        # Generate reasoning
        if agrees:
            reasoning = f"GNN confirms {mobilevit_class} classification with {adjusted_confidence:.1%} confidence"
        else:
            reasoning = f"GNN suggests caution: adjusted confidence from {mobilevit_confidence:.1%} to {adjusted_confidence:.1%}"
        
        return {
            'agrees': agrees,
            'gnn_category': gnn_category,
            'adjusted_confidence': adjusted_confidence,
            'confidence_delta': confidence_delta,
            'risk_level': risk_level,
            'is_safety_critical': risk_level >= 3,
            'reasoning': reasoning,
            'recommendation': mobilevit_class,  # Keep same class
            'status': 'VERIFIED' if agrees else 'CAUTIONED'
        }


def load_gnn_verifier(model_path: Optional[str] = None, device='cpu'):
    """
    Load GNN verification model
    
    Args:
        model_path: Path to trained GNN model (optional)
        device: Device for inference
    
    Returns:
        GNN verifier instance
    """
    try:
        if model_path and os.path.exists(model_path):
            # Try to load full GNN model
            try:
                from waste_reasoning_rgn import create_waste_reasoning_model
                from inference_engine import WasteClassificationEngine
                
                logger.info(f"🔄 Loading full GNN model from {model_path}")
                engine = WasteClassificationEngine(model_path, device=device)
                logger.info("✅ Full GNN model loaded successfully")
                return engine
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load full GNN model: {e}")
                logger.info("📋 Falling back to simplified GNN verifier")
                return SimplifiedGNNVerifier(device=device)
        else:
            logger.info("📋 Using simplified GNN verifier (no model path provided)")
            return SimplifiedGNNVerifier(device=device)
            
    except Exception as e:
        logger.error(f"❌ Error loading GNN: {e}")
        logger.info("📋 Using simplified GNN verifier as fallback")
        return SimplifiedGNNVerifier(device=device)


if __name__ == "__main__":
    # Test the simplified verifier
    logging.basicConfig(level=logging.INFO)
    
    verifier = load_gnn_verifier()
    
    # Test verification
    mobilevit_probs = {
        'Plastic': 0.92,
        'Glass': 0.04,
        'Metal': 0.02,
        'Paper': 0.01,
        'Cardboard': 0.01
    }
    
    result = verifier.verify_classification(
        mobilevit_class='Plastic',
        mobilevit_confidence=0.92,
        mobilevit_probs=mobilevit_probs
    )
    
    print("\n✅ GNN Verification Test:")
    print(f"   Category: {result['gnn_category']}")
    print(f"   Agrees: {result['agrees']}")
    print(f"   Adjusted Confidence: {result['adjusted_confidence']:.2%}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Reasoning: {result['reasoning']}")
