"""
Active Learning System for Waste Classification
Enables continuous model improvement from user feedback
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FeedbackStorage:
    """Store and manage user feedback for active learning"""
    
    def __init__(self, storage_dir: str = "feedback_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.feedback_file = self.storage_dir / "user_feedback.json"
        self.images_dir = self.storage_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.feedback_data = self._load_feedback()
        
    def _load_feedback(self) -> Dict:
        """Load existing feedback from disk"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading feedback: {e}")
                return {"feedbacks": [], "statistics": {}}
        return {"feedbacks": [], "statistics": {}}
    
    def _save_feedback(self):
        """Save feedback to disk"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def add_feedback(
        self,
        user_id: str,
        image_data: bytes,
        predicted_class: str,
        predicted_confidence: float,
        correct_class: str,
        is_correct: bool,
        timestamp: str = None
    ) -> str:
        """
        Add user feedback for a classification
        
        Args:
            user_id: User who provided feedback
            image_data: Raw image bytes
            predicted_class: Model's prediction
            predicted_confidence: Prediction confidence
            correct_class: User-corrected class
            is_correct: Whether prediction was correct
            timestamp: Optional timestamp
            
        Returns:
            feedback_id: Unique ID for this feedback
        """
        feedback_id = f"feedback_{len(self.feedback_data['feedbacks'])}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Save image
        image_path = self.images_dir / f"{feedback_id}.jpg"
        try:
            with open(image_path, 'wb') as f:
                f.write(image_data)
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            image_path = None
        
        # Create feedback record
        feedback = {
            "feedback_id": feedback_id,
            "user_id": user_id,
            "timestamp": timestamp or datetime.now().isoformat(),
            "predicted_class": predicted_class,
            "predicted_confidence": predicted_confidence,
            "correct_class": correct_class,
            "is_correct": is_correct,
            "image_path": str(image_path) if image_path else None,
            "used_for_training": False,
            "priority": self._calculate_priority(predicted_confidence, is_correct)
        }
        
        self.feedback_data["feedbacks"].append(feedback)
        self._update_statistics(feedback)
        self._save_feedback()
        
        logger.info(f"✅ Feedback added: {feedback_id} - {'✓ Correct' if is_correct else '✗ Incorrect'}")
        
        return feedback_id
    
    def _calculate_priority(self, confidence: float, is_correct: bool) -> float:
        """
        Calculate priority for active learning
        High priority = uncertain predictions or incorrect predictions
        """
        if not is_correct:
            return 1.0  # Highest priority for incorrect predictions
        elif confidence < 0.7:
            return 0.8  # High priority for low confidence
        elif confidence < 0.85:
            return 0.5  # Medium priority
        else:
            return 0.2  # Low priority for high confidence correct predictions
    
    def _update_statistics(self, feedback: Dict):
        """Update aggregate statistics"""
        stats = self.feedback_data.setdefault("statistics", {
            "total_feedback": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "class_accuracy": {},
            "confusion_matrix": {},
            "avg_confidence": 0.0,
            "last_updated": None
        })
        
        stats["total_feedback"] += 1
        
        if feedback["is_correct"]:
            stats["correct_predictions"] += 1
        else:
            stats["incorrect_predictions"] += 1
        
        # Update class accuracy
        pred_class = feedback["predicted_class"]
        if pred_class not in stats["class_accuracy"]:
            stats["class_accuracy"][pred_class] = {"correct": 0, "total": 0}
        
        stats["class_accuracy"][pred_class]["total"] += 1
        if feedback["is_correct"]:
            stats["class_accuracy"][pred_class]["correct"] += 1
        
        # Update confusion matrix
        if not feedback["is_correct"]:
            confusion_key = f"{pred_class}->{feedback['correct_class']}"
            stats["confusion_matrix"][confusion_key] = stats["confusion_matrix"].get(confusion_key, 0) + 1
        
        # Update average confidence
        total = stats["total_feedback"]
        old_avg = stats["avg_confidence"]
        new_conf = feedback["predicted_confidence"]
        stats["avg_confidence"] = (old_avg * (total - 1) + new_conf) / total
        
        stats["last_updated"] = datetime.now().isoformat()
    
    def get_training_samples(
        self, 
        min_samples: int = 50,
        priority_threshold: float = 0.5,
        unused_only: bool = True
    ) -> List[Dict]:
        """
        Get samples for retraining
        
        Args:
            min_samples: Minimum number of samples needed
            priority_threshold: Only include samples above this priority
            unused_only: Only return samples not yet used for training
            
        Returns:
            List of feedback samples suitable for training
        """
        feedbacks = self.feedback_data["feedbacks"]
        
        # Filter samples
        filtered = [
            f for f in feedbacks
            if (f["priority"] >= priority_threshold)
            and (not unused_only or not f.get("used_for_training", False))
            and f.get("image_path") is not None
        ]
        
        # Sort by priority (highest first)
        filtered.sort(key=lambda x: x["priority"], reverse=True)
        
        logger.info(f"📊 Training samples: {len(filtered)} available (min needed: {min_samples})")
        
        return filtered if len(filtered) >= min_samples else []
    
    def mark_as_used(self, feedback_ids: List[str]):
        """Mark feedback samples as used for training"""
        for feedback in self.feedback_data["feedbacks"]:
            if feedback["feedback_id"] in feedback_ids:
                feedback["used_for_training"] = True
        self._save_feedback()
        logger.info(f"✅ Marked {len(feedback_ids)} samples as used for training")
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        stats = self.feedback_data.get("statistics", {})
        
        # Calculate accuracy
        total = stats.get("total_feedback", 0)
        correct = stats.get("correct_predictions", 0)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Calculate per-class accuracy
        class_acc = {}
        for class_name, data in stats.get("class_accuracy", {}).items():
            acc = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
            class_acc[class_name] = {
                "accuracy": acc,
                "correct": data["correct"],
                "total": data["total"]
            }
        
        return {
            "total_feedback": total,
            "correct_predictions": correct,
            "incorrect_predictions": stats.get("incorrect_predictions", 0),
            "overall_accuracy": accuracy,
            "class_accuracy": class_acc,
            "confusion_matrix": stats.get("confusion_matrix", {}),
            "avg_confidence": stats.get("avg_confidence", 0),
            "last_updated": stats.get("last_updated"),
            "samples_ready_for_training": len(self.get_training_samples())
        }


class ActiveLearningManager:
    """Manage active learning pipeline"""
    
    def __init__(
        self,
        feedback_storage: FeedbackStorage,
        retrain_threshold: int = 100,
        retrain_interval_days: int = 7
    ):
        self.feedback_storage = feedback_storage
        self.retrain_threshold = retrain_threshold
        self.retrain_interval_days = retrain_interval_days
        
        self.metadata_file = Path("feedback_data") / "training_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load training metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        
        return {
            "last_retrain": None,
            "retrain_count": 0,
            "total_samples_used": 0,
            "performance_history": []
        }
    
    def _save_metadata(self):
        """Save training metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if model should be retrained
        
        Returns:
            (should_retrain, reason)
        """
        stats = self.feedback_storage.get_statistics()
        
        # Get unused feedback count
        unused_samples = len(self.feedback_storage.get_training_samples(
            min_samples=0, 
            unused_only=True
        ))
        
        # Check threshold
        if unused_samples >= self.retrain_threshold:
            return True, f"Reached threshold: {unused_samples} new samples"
        
        # Check time interval
        last_retrain = self.metadata.get("last_retrain")
        if last_retrain:
            last_date = datetime.fromisoformat(last_retrain)
            days_since = (datetime.now() - last_date).days
            
            if days_since >= self.retrain_interval_days and unused_samples >= 20:
                return True, f"Time-based: {days_since} days since last retrain"
        
        # Check accuracy degradation
        if stats["total_feedback"] >= 50:
            recent_accuracy = self._calculate_recent_accuracy()
            if recent_accuracy < 75.0:  # Less than 75% accuracy
                return True, f"Low accuracy: {recent_accuracy:.1f}%"
        
        return False, "Not ready for retraining"
    
    def _calculate_recent_accuracy(self, window: int = 50) -> float:
        """Calculate accuracy over recent predictions"""
        feedbacks = self.feedback_storage.feedback_data["feedbacks"][-window:]
        if not feedbacks:
            return 100.0
        
        correct = sum(1 for f in feedbacks if f["is_correct"])
        return (correct / len(feedbacks)) * 100
    
    def prepare_training_data(self) -> Optional[Dict]:
        """
        Prepare data for model retraining
        
        Returns:
            Dictionary with training data paths and metadata
        """
        samples = self.feedback_storage.get_training_samples(
            min_samples=20,
            priority_threshold=0.3,
            unused_only=True
        )
        
        if not samples:
            logger.warning("⚠️ Not enough samples for training")
            return None
        
        # Organize by class
        class_samples = {}
        for sample in samples:
            correct_class = sample["correct_class"]
            if correct_class not in class_samples:
                class_samples[correct_class] = []
            class_samples[correct_class].append(sample)
        
        logger.info(f"📊 Training data prepared:")
        for class_name, class_samples_list in class_samples.items():
            logger.info(f"   • {class_name}: {len(class_samples_list)} samples")
        
        return {
            "samples": samples,
            "class_distribution": {k: len(v) for k, v in class_samples.items()},
            "total_samples": len(samples),
            "timestamp": datetime.now().isoformat()
        }
    
    def record_training(self, training_data: Dict, performance: Dict):
        """Record successful training"""
        sample_ids = [s["feedback_id"] for s in training_data["samples"]]
        
        # Mark samples as used
        self.feedback_storage.mark_as_used(sample_ids)
        
        # Update metadata
        self.metadata["last_retrain"] = datetime.now().isoformat()
        self.metadata["retrain_count"] += 1
        self.metadata["total_samples_used"] += training_data["total_samples"]
        
        # Record performance
        self.metadata["performance_history"].append({
            "timestamp": datetime.now().isoformat(),
            "samples_used": training_data["total_samples"],
            "class_distribution": training_data["class_distribution"],
            "performance": performance
        })
        
        self._save_metadata()
        
        logger.info(f"✅ Training recorded: {training_data['total_samples']} samples used")
    
    def get_uncertain_predictions(
        self, 
        predictions: List[Dict],
        uncertainty_threshold: float = 0.85
    ) -> List[Dict]:
        """
        Identify predictions that would benefit from user feedback
        (Active Learning Query Strategy)
        
        Args:
            predictions: List of recent predictions
            uncertainty_threshold: Confidence below this is uncertain
            
        Returns:
            Predictions that should be shown to users for feedback
        """
        uncertain = [
            p for p in predictions
            if p.get("confidence", 1.0) < uncertainty_threshold
        ]
        
        # Sort by uncertainty (lowest confidence first)
        uncertain.sort(key=lambda x: x.get("confidence", 1.0))
        
        return uncertain
    
    def get_dashboard_data(self) -> Dict:
        """Get data for active learning dashboard"""
        stats = self.feedback_storage.get_statistics()
        should_retrain, reason = self.should_retrain()
        
        return {
            "feedback_statistics": stats,
            "training_metadata": self.metadata,
            "retraining": {
                "should_retrain": should_retrain,
                "reason": reason,
                "threshold": self.retrain_threshold,
                "interval_days": self.retrain_interval_days
            },
            "recommendations": self._get_recommendations(stats)
        }
    
    def _get_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations based on statistics"""
        recommendations = []
        
        # Check overall accuracy
        if stats["overall_accuracy"] < 80:
            recommendations.append(
                f"⚠️ Overall accuracy is {stats['overall_accuracy']:.1f}%. Consider retraining soon."
            )
        
        # Check class-specific issues
        for class_name, data in stats["class_accuracy"].items():
            if data["total"] >= 10 and data["accuracy"] < 70:
                recommendations.append(
                    f"⚠️ {class_name}: Low accuracy ({data['accuracy']:.1f}%) - needs more training data"
                )
        
        # Check confusion patterns
        confusion = stats.get("confusion_matrix", {})
        if confusion:
            most_confused = max(confusion.items(), key=lambda x: x[1])
            if most_confused[1] >= 5:
                recommendations.append(
                    f"🔄 Common confusion: {most_confused[0]} ({most_confused[1]} times)"
                )
        
        # Check readiness for training
        unused = len(self.feedback_storage.get_training_samples(min_samples=0, unused_only=True))
        if unused >= self.retrain_threshold:
            recommendations.append(
                f"✅ Ready to retrain! {unused} new samples available."
            )
        
        return recommendations if recommendations else ["✅ System performing well. Keep collecting feedback!"]


# Singleton instances
_feedback_storage = None
_active_learning_manager = None


def get_feedback_storage() -> FeedbackStorage:
    """Get global feedback storage instance"""
    global _feedback_storage
    if _feedback_storage is None:
        _feedback_storage = FeedbackStorage()
    return _feedback_storage


def get_active_learning_manager() -> ActiveLearningManager:
    """Get global active learning manager instance"""
    global _active_learning_manager
    if _active_learning_manager is None:
        _active_learning_manager = ActiveLearningManager(
            feedback_storage=get_feedback_storage(),
            retrain_threshold=100,  # Retrain after 100 new samples
            retrain_interval_days=7  # Or every 7 days
        )
    return _active_learning_manager
