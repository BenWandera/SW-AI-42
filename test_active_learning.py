"""
Active Learning Demo and Test Script
Demonstrates the active learning system with simulated feedback
"""

import requests
import json
import random
from pathlib import Path
from PIL import Image
import io
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Waste classes
WASTE_CLASSES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal',
    'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_section(text):
    """Print formatted section"""
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")


def create_sample_image(class_name: str) -> bytes:
    """Create a simple colored image for testing"""
    # Different colors for different classes
    color_map = {
        'Plastic': (255, 100, 100),    # Red
        'Paper': (100, 100, 255),      # Blue
        'Glass': (100, 255, 100),      # Green
        'Metal': (200, 200, 200),      # Gray
        'Cardboard': (200, 150, 100),  # Brown
        'Food Organics': (150, 200, 50), # Light green
        'Vegetation': (50, 150, 50),   # Dark green
    }
    
    color = color_map.get(class_name, (128, 128, 128))
    
    # Create image
    img = Image.new('RGB', (224, 224), color)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


def check_api_status():
    """Check if API is running"""
    print_section("Checking API Status")
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is online")
            print(f"   Version: {data['version']}")
            print(f"   Model: {data['model']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Classes: {data['classes']}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not accessible: {e}")
        print(f"   Make sure the API is running on {API_BASE_URL}")
        return False


def simulate_feedback(num_samples: int = 20):
    """Simulate user feedback on classifications"""
    print_section(f"Simulating {num_samples} User Feedbacks")
    
    feedback_results = []
    
    for i in range(num_samples):
        # Pick a random class
        true_class = random.choice(WASTE_CLASSES)
        
        # Simulate prediction (80% accuracy)
        if random.random() < 0.8:
            predicted_class = true_class
            is_correct = True
            confidence = random.uniform(0.85, 0.98)
        else:
            predicted_class = random.choice([c for c in WASTE_CLASSES if c != true_class])
            is_correct = False
            confidence = random.uniform(0.60, 0.85)
        
        # Create sample image
        image_data = create_sample_image(true_class)
        
        # Submit feedback
        try:
            files = {'image': ('test_image.jpg', image_data, 'image/jpeg')}
            data = {
                'user_id': 'demo_user',
                'predicted_class': predicted_class,
                'predicted_confidence': confidence,
                'correct_class': true_class,
                'is_correct': is_correct
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/feedback/submit",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                feedback_results.append(result)
                
                status = "✓" if is_correct else "✗"
                print(f"{i+1:2d}. {status} Predicted: {predicted_class:20s} | True: {true_class:20s} | Conf: {confidence:.2f}")
                
                if (i + 1) % 5 == 0:
                    stats = result.get('statistics', {})
                    print(f"    📊 Total: {stats.get('total_feedback', 0)} | Accuracy: {stats.get('accuracy', 0):.1f}%")
            else:
                print(f"{i+1:2d}. ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"{i+1:2d}. ❌ Exception: {e}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    return feedback_results


def get_statistics():
    """Get current feedback statistics"""
    print_section("Feedback Statistics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/feedback/statistics")
        
        if response.status_code == 200:
            data = response.json()
            stats = data['statistics']
            
            print(f"📊 Overall Statistics:")
            print(f"   Total Feedback: {stats.get('total_feedback', 0)}")
            print(f"   Correct: {stats.get('correct_predictions', 0)}")
            print(f"   Incorrect: {stats.get('incorrect_predictions', 0)}")
            print(f"   Accuracy: {stats.get('overall_accuracy', 0):.2f}%")
            print(f"   Avg Confidence: {stats.get('avg_confidence', 0):.2f}")
            print(f"   Samples Ready: {stats.get('samples_ready_for_training', 0)}")
            
            print(f"\n📋 Class Accuracy:")
            class_acc = stats.get('class_accuracy', {})
            for class_name, acc_data in sorted(class_acc.items()):
                print(f"   {class_name:20s}: {acc_data['accuracy']:5.1f}% ({acc_data['correct']}/{acc_data['total']})")
            
            print(f"\n🔄 Confusion Matrix:")
            confusion = stats.get('confusion_matrix', {})
            if confusion:
                for conf_pair, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"   {conf_pair}: {count} times")
            else:
                print(f"   No confusion (all correct)")
            
            return stats
        else:
            print(f"❌ Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def get_dashboard():
    """Get active learning dashboard"""
    print_section("Active Learning Dashboard")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/learning/dashboard")
        
        if response.status_code == 200:
            data = response.json()
            dashboard = data['dashboard']
            
            # Retraining info
            retraining = dashboard.get('retraining', {})
            print(f"🔄 Retraining Status:")
            print(f"   Should Retrain: {retraining.get('should_retrain', False)}")
            print(f"   Reason: {retraining.get('reason', 'N/A')}")
            print(f"   Threshold: {retraining.get('threshold', 100)} samples")
            print(f"   Interval: {retraining.get('interval_days', 7)} days")
            
            # Training metadata
            metadata = dashboard.get('training_metadata', {})
            print(f"\n📚 Training History:")
            print(f"   Last Retrain: {metadata.get('last_retrain', 'Never')}")
            print(f"   Retrain Count: {metadata.get('retrain_count', 0)}")
            print(f"   Total Samples Used: {metadata.get('total_samples_used', 0)}")
            
            # Recommendations
            recommendations = dashboard.get('recommendations', [])
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
            
            return dashboard
        else:
            print(f"❌ Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def trigger_retraining(force=False):
    """Trigger model retraining"""
    print_section("Triggering Model Retraining")
    
    try:
        params = {'force': force, 'epochs': 2, 'batch_size': 4}
        response = requests.post(
            f"{API_BASE_URL}/api/model/retrain",
            params=params
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                results = data.get('results', {})
                print(f"✅ Retraining Successful!")
                print(f"   Backup: {results.get('backup_path', 'N/A')}")
                print(f"   Training Samples: {results.get('training_samples', 0)}")
                print(f"   Validation Samples: {results.get('validation_samples', 0)}")
                print(f"   Final Train Acc: {results.get('final_train_acc', 0):.2f}%")
                print(f"   Final Val Acc: {results.get('final_val_acc', 0):.2f}%")
                print(f"   Best Val Acc: {results.get('best_val_acc', 0):.2f}%")
                
                print(f"\n📈 Training History:")
                history = results.get('training_history', [])
                for epoch in history:
                    print(f"   Epoch {epoch['epoch']}: Train={epoch['train_acc']:.1f}% | Val={epoch['val_acc']:.1f}%")
            else:
                print(f"⚠️ {data.get('message', 'Retraining not recommended')}")
                if data.get('force_required'):
                    print(f"   Use force=True to retrain anyway")
            
            return data
        else:
            print(f"❌ Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def list_backups():
    """List model backups"""
    print_section("Model Backups")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/model/backups")
        
        if response.status_code == 200:
            data = response.json()
            backups = data.get('backups', [])
            
            print(f"📦 Total Backups: {len(backups)}")
            
            for i, backup in enumerate(backups[:5], 1):  # Show latest 5
                print(f"\n{i}. {backup['filename']}")
                print(f"   Size: {backup['size_mb']:.2f} MB")
                print(f"   Created: {backup.get('created', 'Unknown')}")
            
            return backups
        else:
            print(f"❌ Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def run_full_demo():
    """Run complete active learning demo"""
    print_header("Active Learning System Demo")
    
    # Check API
    if not check_api_status():
        print("\n❌ Cannot proceed without API. Please start the API server first.")
        print(f"   Run: python api/real_api.py")
        return
    
    # Simulate feedback
    print("\n" + "="*60)
    print("STEP 1: Collecting User Feedback")
    print("="*60)
    simulate_feedback(num_samples=20)
    
    # Get statistics
    print("\n" + "="*60)
    print("STEP 2: Analyzing Feedback")
    print("="*60)
    get_statistics()
    
    # Get dashboard
    print("\n" + "="*60)
    print("STEP 3: Checking Learning Status")
    print("="*60)
    dashboard = get_dashboard()
    
    # Check if retraining is recommended
    print("\n" + "="*60)
    print("STEP 4: Model Retraining (if needed)")
    print("="*60)
    
    if dashboard and dashboard.get('retraining', {}).get('should_retrain'):
        print("\n🔄 Retraining is recommended!")
        user_input = input("\nDo you want to trigger retraining? (yes/no): ")
        
        if user_input.lower() in ['yes', 'y']:
            trigger_retraining(force=False)
        else:
            print("⏭️ Skipping retraining")
    else:
        print("\n⏭️ Not enough samples for retraining yet")
        print("   Continue collecting feedback to improve the model!")
    
    # List backups
    print("\n" + "="*60)
    print("STEP 5: Model Backups")
    print("="*60)
    list_backups()
    
    # Final summary
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\n✅ Active Learning System is ready for production!")
    print("\n📖 Next Steps:")
    print("   1. Integrate feedback UI in Flutter app")
    print("   2. Monitor feedback statistics regularly")
    print("   3. Review and trigger retraining when ready")
    print("   4. Track model performance improvements")
    print("\n📚 See ACTIVE_LEARNING_GUIDE.md for details")


def interactive_menu():
    """Interactive menu for testing"""
    while True:
        print_header("Active Learning Test Menu")
        print("\n1. Check API Status")
        print("2. Simulate Feedback (20 samples)")
        print("3. Get Statistics")
        print("4. View Dashboard")
        print("5. Trigger Retraining")
        print("6. List Backups")
        print("7. Run Full Demo")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            check_api_status()
        elif choice == '2':
            simulate_feedback(20)
        elif choice == '3':
            get_statistics()
        elif choice == '4':
            get_dashboard()
        elif choice == '5':
            force = input("Force retraining? (yes/no): ").lower() in ['yes', 'y']
            trigger_retraining(force=force)
        elif choice == '6':
            list_backups()
        elif choice == '7':
            run_full_demo()
        elif choice == '8':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    import sys
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     Active Learning Demo & Test Script                  ║
    ║     Waste Classification System                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Run full demo automatically
        run_full_demo()
    else:
        # Interactive menu
        interactive_menu()
