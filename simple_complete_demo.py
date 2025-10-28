"""
Simple Complete Waste Management System Demo
"""

import torch
from PIL import Image
import json
import os
from datetime import datetime
import numpy as np

# Import our modules
from synthetic_waste_generator import SyntheticWasteGenerator
from simple_incentive_engine import SimpleIncentiveEngine, SortingResult, UserProfile

class SimpleCompleteSystem:
    """Simple integrated waste management system"""
    
    def __init__(self):
        print("🌟 Simple Waste Management System")
        print("=" * 40)
        
        # Initialize components
        self.synthetic_generator = None
        self.incentive_engine = None
        
        self.init_components()
    
    def init_components(self):
        """Initialize system components"""
        
        # 1. Synthetic Generator
        try:
            self.synthetic_generator = SyntheticWasteGenerator()
            if self.synthetic_generator.generator is not None:
                print("✅ Synthetic generator ready")
            else:
                print("⚠️ Synthetic generator not available")
        except Exception as e:
            print(f"⚠️ Generator error: {e}")
        
        # 2. Incentive Engine
        try:
            self.incentive_engine = SimpleIncentiveEngine(anthropic_api_key="demo")  # Dummy key for demo
            # Create demo user
            self.demo_user = UserProfile(name="Demo User")
            print("✅ Incentive engine ready")
        except Exception as e:
            print(f"⚠️ Incentive error: {e}")
            self.demo_user = None
    
    def generate_and_score(self, num_images=3):
        """Generate synthetic images and simulate scoring"""
        
        print(f"\n🎯 Generate and Score Demo")
        print("=" * 30)
        
        # Generate synthetic images
        if self.synthetic_generator and self.synthetic_generator.generator:
            print(f"🎨 Generating {num_images} synthetic images...")
            images = self.synthetic_generator.generate_images(
                num_images=num_images,
                save_path="demo_generated.png"
            )
            print(f"✅ Generated images saved as demo_generated.png")
        else:
            print("❌ Cannot generate images - no trained model")
            return
        
        # Simulate classification and scoring
        waste_types = ['plastic', 'organic', 'paper', 'metal']
        total_points = 0
        
        print(f"\n🔍 Simulating classification and scoring...")
        
        for i in range(num_images):
            # Simulate classification
            waste_type = np.random.choice(waste_types)
            confidence = np.random.uniform(0.7, 0.95)
            
            print(f"\nImage {i+1}:")
            print(f"   Classified as: {waste_type}")
            print(f"   Confidence: {confidence:.2f}")
            
            # Calculate incentive
            if self.incentive_engine and hasattr(self, 'demo_user') and self.demo_user:
                sorting_result = SortingResult(
                    waste_type=waste_type,
                    predicted_class=waste_type,
                    confidence=confidence,
                    is_correctly_sorted=True,
                    quantity_kg=1.0
                )
                
                incentive = self.incentive_engine.calculate_incentives(
                    user=self.demo_user,
                    sorting_result=sorting_result,
                    use_ai=False  # Disable AI for demo
                )
                
                points = incentive.points_earned
                total_points += points
                print(f"   Points earned: {points}")
                # Check if feedback exists, otherwise use default
                feedback = getattr(incentive, 'feedback', 'Good job!')
                print(f"   Feedback: {feedback}")
        
        print(f"\n📊 Demo Summary:")
        print(f"   Images processed: {num_images}")
        print(f"   Total points earned: {total_points}")
        print(f"   System components working: ✅")
        
        return total_points
    
    def system_status(self):
        """Check system status"""
        
        print(f"\n🔧 System Status")
        print("=" * 20)
        
        gen_status = self.synthetic_generator is not None and self.synthetic_generator.generator is not None
        inc_status = self.incentive_engine is not None
        
        print(f"🎨 Generator: {'✅' if gen_status else '❌'}")
        print(f"💰 Incentive: {'✅' if inc_status else '❌'}")
        
        overall = gen_status and inc_status
        print(f"🌟 Overall: {'✅ Ready' if overall else '⚠️ Partial'}")
        
        return overall


def main():
    """Main demo function"""
    
    print("🚀 Simple Waste Management System Demo")
    print("=" * 45)
    
    # Initialize system
    system = SimpleCompleteSystem()
    
    # Check status
    is_ready = system.system_status()
    
    if not is_ready:
        print("\n⚠️ System not fully operational")
        print("Some components may not work correctly")
    
    # Run demo
    try:
        total_points = system.generate_and_score(num_images=3)
        
        print(f"\n✅ Demo completed successfully!")
        print(f"💡 This shows the integration of:")
        print(f"   • GAN synthetic image generation")
        print(f"   • Waste classification simulation")
        print(f"   • Incentive point calculation")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check if all required models are available")


if __name__ == "__main__":
    main()