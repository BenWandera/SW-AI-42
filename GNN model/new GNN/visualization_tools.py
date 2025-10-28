"""
Inference Engine for Waste Reasoning RGN
Handles real-time classification, conflict resolution, and safety validation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from waste_reasoning_rgn import (
    WasteReasoningRGN, create_waste_reasoning_model,
    WasteCategory, RiskLevel, DisposalMethod
)


class WasteClassificationEngine:
    """
    Real-time inference engine for waste classification
    Integrates safety rules and conflict resolution
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 confidence_threshold: float = 0.7):
        """
        Initialize the classification engine
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference
            confidence_threshold: Minimum confidence for classification
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = create_waste_reasoning_model(vision_embedding_dim=2048)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Using device: {device}")
    
    def classify_single_item(self, 
                            vision_embedding: np.ndarray) -> Dict[str, any]:
        """
        Classify a single waste item
        
        Args:
            vision_embedding: Vision features from LVM [embedding_dim]
        
        Returns:
            Classification results with safety validation
        """
        # Convert to tensor
        embedding = torch.FloatTensor(vision_embedding).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(embedding)
            
            # Apply safety rules
            outputs = self.model.apply_safety_rules(outputs)
            
            # Get predictions
            material_pred = torch.argmax(outputs['material_logits'][0]).item()
            category_pred = torch.argmax(outputs['category_logits'][0]).item()
            disposal_pred = torch.argmax(outputs['disposal_logits'][0]).item()
            risk_pred = torch.argmax(outputs['risk_scores'][0]).item()
            confidence = outputs['confidence'][0].item()
            
            # Get probabilities
            category_probs = torch.softmax(outputs['category_logits'][0], dim=0)
            risk_probs = torch.softmax(outputs['risk_scores'][0], dim=0)
            
            # Generate explanation
            explanation = self.model.get_explanation(outputs, sample_idx=0)
            
            # Build result
            result = {
                'category': list(WasteCategory)[category_pred].value,
                'category_confidence': category_probs[category_pred].item(),
                'risk_level': list(RiskLevel)[risk_pred].name,
                'risk_confidence': risk_probs[risk_pred].item(),
                'disposal_method': self._get_disposal_name(disposal_pred),
                'overall_confidence': confidence,
                'is_safety_critical': risk_pred >= RiskLevel.HIGH_RISK.value,
                'explanation': explanation,
                'status': 'SUCCESS'
            }
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                result['status'] = 'LOW_CONFIDENCE'
                result['warning'] = f"Confidence {confidence:.2%} below threshold {self.confidence_threshold:.2%}"
            
            # Add safety warnings
            if result['is_safety_critical']:
                result['safety_warning'] = self._generate_safety_warning(
                    category_pred, risk_pred
                )
            
            # Add point valuation
            result['points'] = self._calculate_points(result)
            
            return result
    
    def classify_multiple_items(self,
                               vision_embeddings: List[np.ndarray]) -> Dict[str, any]:
        """
        Classify multiple waste items and resolve conflicts
        
        Args:
            vision_embeddings: List of vision features for each detected item
        
        Returns:
            Combined classification with conflict resolution
        """
        # Classify each item individually
        individual_results = []
        for embedding in vision_embeddings:
            result = self.classify_single_item(embedding)
            individual_results.append(result)
        
        # Check for conflicts
        has_conflicts = self._detect_conflicts(individual_results)
        
        if has_conflicts:
            # Resolve conflicts
            resolved_result = self._resolve_conflicts(individual_results)
            resolved_result['conflict_detected'] = True
            resolved_result['individual_items'] = individual_results
            return resolved_result
        else:
            # No conflicts - return aggregated result
            return {
                'conflict_detected': False,
                'individual_items': individual_results,
                'total_items': len(individual_results),
                'combined_points': sum(r['points'] for r in individual_results),
                'status': 'SUCCESS'
            }
    
    def _detect_conflicts(self, results: List[Dict]) -> bool:
        """
        Detect if there are conflicting waste types
        
        Args:
            results: List of classification results
        
        Returns:
            True if conflicts detected
        """
        if len(results) <= 1:
            return False
        
        # Check for safety-critical conflicts
        has_medical = any(r['category'] == 'medical' for r in results)
        has_organic = any(r['category'] == 'organic' for r in results)
        has_electronic = any(r['category'] == 'electronic' for r in results)
        
        # Medical waste should never be mixed with anything
        if has_medical and len(results) > 1:
            return True
        
        # Electronic (batteries) should not be with organic
        if has_electronic and has_organic:
            return True
        
        # Check risk level conflicts
        risk_levels = [r['risk_level'] for r in results]
        has_critical = 'CRITICAL' in risk_levels
        has_safe = 'SAFE' in risk_levels or 'LOW_RISK' in risk_levels
        
        if has_critical and has_safe:
            return True
        
        return False
    
    def _resolve_conflicts(self, results: List[Dict]) -> Dict[str, any]:
        """
        Resolve conflicts using safety priority rules
        
        Priority order:
        1. CRITICAL risk items (medical)
        2. HIGH_RISK items (electronic with batteries)
        3. MEDIUM_RISK items
        4. LOW_RISK items
        5. SAFE items
        
        Args:
            results: List of conflicting classification results
        
        Returns:
            Resolved classification with safety priority
        """
        # Find highest risk item
        risk_priority = {
            'CRITICAL': 4,
            'HIGH_RISK': 3,
            'MEDIUM_RISK': 2,
            'LOW_RISK': 1,
            'SAFE': 0
        }
        
        highest_risk_result = max(results, 
                                 key=lambda r: risk_priority[r['risk_level']])
        
        # Build resolved result
        resolved = {
            'resolution_strategy': 'SAFETY_PRIORITY',
            'primary_category': highest_risk_result['category'],
            'risk_level': highest_risk_result['risk_level'],
            'disposal_method': highest_risk_result['disposal_method'],
            'conflict_reason': self._explain_conflict(results),
            'recommended_action': self._get_conflict_action(highest_risk_result),
            'points': -10,  # Negative points for mixed waste
            'status': 'CONFLICT_RESOLVED'
        }
        
        return resolved
    
    def _explain_conflict(self, results: List[Dict]) -> str:
        """Generate human-readable conflict explanation"""
        categories = [r['category'] for r in results]
        risk_levels = [r['risk_level'] for r in results]
        
        has_medical = 'medical' in categories
        has_critical = 'CRITICAL' in risk_levels
        
        if has_medical:
            return "Medical waste detected mixed with other waste types. Medical waste must be separated immediately."
        elif has_critical:
            return "Critical risk item detected. This waste requires specialized handling and cannot be mixed."
        else:
            return f"Multiple waste types detected: {', '.join(set(categories))}. Please separate for proper disposal."
    
    def _get_conflict_action(self, primary_result: Dict) -> str:
        """Get recommended action for conflict resolution"""
        if primary_result['is_safety_critical']:
            return (
                f"IMMEDIATE ACTION REQUIRED: Separate {primary_result['category']} waste. "
                f"Use {primary_result['disposal_method']}. "
                f"Follow safety protocols for handling."
            )
        else:
            return (
                f"Please separate waste items. "
                f"Dispose {primary_result['category']} waste using {primary_result['disposal_method']}."
            )
    
    def _get_disposal_name(self, disposal_idx: int) -> str:
        """Map disposal index to name"""
        disposal_map = {
            0: "recyclable",
            1: "compostable",
            2: "landfill",
            3: "hazardous_disposal",
            4: "specialized_facility"
        }
        return disposal_map.get(disposal_idx, "unknown")
    
    def _generate_safety_warning(self, category_idx: int, risk_idx: int) -> str:
        """Generate safety warning message"""
        category = list(WasteCategory)[category_idx].value
        risk = list(RiskLevel)[risk_idx].name
        
        warnings = {
            'medical': "⚠️ MEDICAL WASTE: Handle with protective equipment. Risk of infection or injury.",
            'electronic': "⚠️ ELECTRONIC WASTE: Contains hazardous materials. Do not break or incinerate."
        }
        
        return warnings.get(category, f"⚠️ {risk} WASTE: Follow proper safety protocols.")
    
    def _calculate_points(self, result: Dict) -> int:
        """
        Calculate incentive points for waste classification
        
        Point system:
        - Recyclable waste (properly sorted): +5 points
        - Compostable waste: +4 points
        - Proper disposal of hazardous: +10 points
        - Low confidence: -2 points
        - Safety-critical properly handled: +15 points
        """
        points = 0
        
        # Base points for disposal method
        disposal_points = {
            'recyclable': 5,
            'compostable': 4,
            'hazardous_disposal': 10,
            'specialized_facility': 10,
            'landfill': 1
        }
        
        points += disposal_points.get(result['disposal_method'], 0)
        
        # Bonus for safety-critical proper handling
        if result['is_safety_critical']:
            points += 15
        
        # Penalty for low confidence
        if result['overall_confidence'] < self.confidence_threshold:
            points -= 2
        
        # Confidence multiplier
        confidence_multiplier = min(result['overall_confidence'] * 1.5, 1.5)
        points = int(points * confidence_multiplier)
        
        return max(points, 0)  # Minimum 0 points


class WasteClassificationAPI:
    """
    High-level API for waste classification in mobile app
    """
    
    def __init__(self, model_path: str):
        self.engine = WasteClassificationEngine(model_path)
        self.session_history = []
    
    def classify_from_camera(self, 
                           vision_embedding: np.ndarray,
                           user_id: str,
                           location: Optional[str] = None) -> Dict[str, any]:
        """
        Classify waste from camera input
        
        Args:
            vision_embedding: Vision features from LVM
            user_id: User identifier
            location: Optional location information
        
        Returns:
            Classification result with user feedback
        """
        # Classify
        result = self.engine.classify_single_item(vision_embedding)
        
        # Add user context
        result['user_id'] = user_id
        result['location'] = location
        result['timestamp'] = self._get_timestamp()
        
        # Generate user-friendly message
        result['user_message'] = self._generate_user_message(result)
        
        # Add educational tip
        result['educational_tip'] = self._get_educational_tip(result)
        
        # Store in history
        self.session_history.append(result)
        
        return result
    
    def classify_bin_contents(self,
                             vision_embeddings: List[np.ndarray],
                             user_id: str) -> Dict[str, any]:
        """
        Classify multiple items in a bin
        
        Args:
            vision_embeddings: List of vision features for detected items
            user_id: User identifier
        
        Returns:
            Combined classification with conflict resolution
        """
        result = self.engine.classify_multiple_items(vision_embeddings)
        
        # Add user context
        result['user_id'] = user_id
        result['timestamp'] = self._get_timestamp()
        
        # Generate feedback
        if result['conflict_detected']:
            result['user_message'] = (
                f"⚠️ Mixed waste detected! {result['conflict_reason']} "
                f"{result['recommended_action']}"
            )
            result['feedback_type'] = 'CORRECTION_NEEDED'
        else:
            total_points = result['combined_points']
            result['user_message'] = (
                f"✓ Great job! {result['total_items']} items properly sorted. "
                f"You earned {total_points} points!"
            )
            result['feedback_type'] = 'POSITIVE'
        
        return result
    
    def get_user_statistics(self, user_id: str) -> Dict[str, any]:
        """Get user waste sorting statistics"""
        user_sessions = [s for s in self.session_history if s.get('user_id') == user_id]
        
        if not user_sessions:
            return {'message': 'No sorting history found'}
        
        total_points = sum(s.get('points', 0) for s in user_sessions)
        total_items = len(user_sessions)
        
        category_counts = {}
        for session in user_sessions:
            category = session.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'user_id': user_id,
            'total_items_sorted': total_items,
            'total_points': total_points,
            'category_breakdown': category_counts,
            'average_confidence': np.mean([s.get('overall_confidence', 0) 
                                          for s in user_sessions]),
            'safety_items_handled': sum(1 for s in user_sessions 
                                       if s.get('is_safety_critical', False))
        }
    
    def _generate_user_message(self, result: Dict) -> str:
        """Generate user-friendly feedback message"""
        category = result['category']
        points = result['points']
        confidence = result['overall_confidence']
        
        if result['is_safety_critical']:
            return (
                f"⚠️ {category.upper()} waste detected! "
                f"{result.get('safety_warning', '')} "
                f"Proper handling: +{points} points."
            )
        elif confidence >= 0.9:
            return (
                f"✓ Excellent! {category.title()} waste properly identified. "
                f"Earned {points} points!"
            )
        elif confidence >= 0.7:
            return (
                f"✓ {category.title()} waste detected. "
                f"Dispose in {result['disposal_method']} bin. +{points} points."
            )
        else:
            return (
                f"⚠️ Low confidence detection. "
                f"Please ensure good lighting and clear view of the item."
            )
    
    def _get_educational_tip(self, result: Dict) -> str:
        """Provide educational tip based on classification"""
        tips = {
            'plastic': "💡 Tip: Rinse plastic containers before recycling to prevent contamination.",
            'organic': "💡 Tip: Organic waste can be composted to create nutrient-rich soil!",
            'paper': "💡 Tip: Keep paper dry and clean for better recycling quality.",
            'glass': "💡 Tip: Glass can be recycled infinitely without losing quality.",
            'metal': "💡 Tip: Aluminum cans save 95% energy when recycled vs. new production.",
            'electronic': "💡 Tip: E-waste contains valuable materials that can be recovered safely.",
            'medical': "💡 Tip: Medical waste must be handled by trained professionals only."
        }
        
        return tips.get(result['category'], "💡 Tip: Proper waste sorting helps our environment!")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


def demo_classification():
    """
    Demonstration of the classification engine
    """
    print("=" * 70)
    print("WASTE CLASSIFICATION ENGINE - DEMONSTRATION")
    print("=" * 70)
    
    # Create API (using dummy model path for demo)
    api = WasteClassificationAPI('waste_reasoning_model.pt')
    
    print("\n📱 Simulating mobile app usage...\n")
    
    # Scenario 1: Single item classification (plastic bottle)
    print("Scenario 1: User scans a plastic bottle")
    print("-" * 70)
    plastic_embedding = np.random.randn(2048)  # Simulated vision embedding
    result1 = api.classify_from_camera(
        vision_embedding=plastic_embedding,
        user_id="user_001",
        location="Kampala Central"
    )
    
    print(f"Category: {result1['category']}")
    print(f"Confidence: {result1['overall_confidence']:.2%}")
    print(f"Risk Level: {result1['risk_level']}")
    print(f"Disposal: {result1['disposal_method']}")
    print(f"Points Earned: {result1['points']}")
    print(f"Message: {result1['user_message']}")
    print(f"{result1['educational_tip']}")
    
    # Scenario 2: Medical waste (safety-critical)
    print("\n\nScenario 2: Healthcare worker scans medical waste")
    print("-" * 70)
    medical_embedding = np.random.randn(2048)
    result2 = api.classify_from_camera(
        vision_embedding=medical_embedding,
        user_id="health_worker_01",
        location="Mulago Hospital"
    )
    
    print(f"Category: {result2['category']}")
    print(f"Risk Level: {result2['risk_level']}")
    print(f"Safety Critical: {result2['is_safety_critical']}")
    print(f"Points Earned: {result2['points']}")
    print(f"Message: {result2['user_message']}")
    if 'safety_warning' in result2:
        print(f"Warning: {result2['safety_warning']}")
    
    # Scenario 3: Mixed waste (conflict detection)
    print("\n\nScenario 3: User scans bin with mixed waste")
    print("-" * 70)
    mixed_embeddings = [
        np.random.randn(2048),  # Plastic
        np.random.randn(2048),  # Organic
        np.random.randn(2048)   # Medical (conflict!)
    ]
    
    result3 = api.classify_bin_contents(
        vision_embeddings=mixed_embeddings,
        user_id="user_001"
    )
    
    print(f"Conflict Detected: {result3['conflict_detected']}")
    if result3['conflict_detected']:
        print(f"Conflict Reason: {result3['conflict_reason']}")
        print(f"Primary Category: {result3['primary_category']}")
        print(f"Recommended Action: {result3['recommended_action']}")
        print(f"Points: {result3['points']} (penalty for mixing)")
    print(f"Feedback: {result3['user_message']}")
    
    # User statistics
    print("\n\nUser Statistics")
    print("-" * 70)
    stats = api.get_user_statistics("user_001")
    print(f"Total Items Sorted: {stats['total_items_sorted']}")
    print(f"Total Points: {stats['total_points']}")
    print(f"Category Breakdown: {stats['category_breakdown']}")
    print(f"Average Confidence: {stats['average_confidence']:.2%}")
    
    print("\n" + "=" * 70)
    print("✓ Demonstration completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    # Run demonstration
    demo_classification()