"""
Monitor DeiT-Tiny Training Progress
Checks for training outputs and displays status
"""

import os
import json
import time
from datetime import datetime

def check_training_progress():
    """Check if training files exist and show progress"""
    
    print("🔍 Checking DeiT-Tiny Training Progress")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check for checkpoint
    if os.path.exists('best_deit_tiny_waste_model.pth'):
        size_mb = os.path.getsize('best_deit_tiny_waste_model.pth') / (1024**2)
        mod_time = datetime.fromtimestamp(os.path.getmtime('best_deit_tiny_waste_model.pth'))
        print(f"✓ Checkpoint found: best_deit_tiny_waste_model.pth")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Last modified: {mod_time.strftime('%H:%M:%S')}\n")
    else:
        print("⏳ Checkpoint not yet created (training in progress...)\n")
    
    # Check for results
    if os.path.exists('deit_tiny_waste_results.json'):
        with open('deit_tiny_waste_results.json', 'r') as f:
            results = json.load(f)
        
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1-Score: {results['f1_score']:.3f}")
        
        if 'training_history' in results:
            history = results['training_history']
            if history['val_acc']:
                print(f"\nBest Validation Accuracy: {max(history['val_acc']):.2f}%")
                print(f"Epochs trained: {len(history['val_acc'])}")
        
        print("\n📊 Per-Class F1-Scores:")
        if 'per_class_f1' in results:
            for class_name, f1 in results['per_class_f1'].items():
                print(f"  • {class_name}: {f1:.3f}")
    else:
        print("⏳ Training in progress...")
        print("   Results file will be created when training completes\n")
    
    print("="*70)

if __name__ == "__main__":
    check_training_progress()
