"""
Synthetic Waste Image Generator
Use the trained GAN to generate synthetic waste images for data augmentation
"""

import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime
import json


class Generator(nn.Module):
    """Generator network (must match training architecture)"""
    
    def __init__(self, latent_dim=100, img_channels=3, features=64):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            
            # features*8 x 4 x 4
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            
            # features*4 x 8 x 8
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            
            # features*2 x 16 x 16
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            
            # features x 32 x 32
            nn.ConvTranspose2d(features, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: img_channels x 64 x 64
        )
    
    def forward(self, input):
        return self.main(input)


class SyntheticWasteGenerator:
    """Generate synthetic waste images using trained GAN"""
    
    def __init__(self, model_path: str = "waste_gan_output/models/wastegan_latest.pth"):
        """
        Initialize generator
        
        Args:
            model_path: Path to trained GAN model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        if os.path.exists(model_path):
            self.generator, self.latent_dim = self._load_model(model_path)
            print(f"✅ Loaded GAN model from: {model_path}")
            print(f"   Latent dimension: {self.latent_dim}")
            print(f"   Device: {self.device}")
        else:
            print(f"❌ Model not found: {model_path}")
            print("   Please train a GAN model first using waste_gan_trainer.py")
            self.generator = None
    
    def _load_model(self, model_path: str):
        """Load trained generator model"""
        # Load with weights_only=False for compatibility with older models
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        latent_dim = checkpoint.get('latent_dim', 100)
        generator = Generator(latent_dim=latent_dim).to(self.device)
        generator.load_state_dict(checkpoint['generator'])
        generator.eval()
        
        return generator, latent_dim
    
    def generate_images(self, 
                       num_images: int = 16,
                       seed: int = None,
                       save_path: str = None,
                       save_individual: bool = False) -> torch.Tensor:
        """
        Generate synthetic waste images
        
        Args:
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            save_path: Path to save image grid
            save_individual: Whether to save individual images
            
        Returns:
            Generated images tensor
        """
        
        if self.generator is None:
            print("❌ No generator loaded!")
            return None
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"🎨 Generating {num_images} synthetic waste images...")
        
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(num_images, self.latent_dim, 1, 1, device=self.device)
            
            # Generate images
            generated_images = self.generator(noise)
            
            # Denormalize from [-1, 1] to [0, 1]
            generated_images = (generated_images + 1) / 2.0
            
            # Clamp values to ensure valid range
            generated_images = torch.clamp(generated_images, 0, 1)
        
        # Save images
        if save_path:
            # Save as grid
            nrow = int(np.sqrt(num_images))
            vutils.save_image(generated_images, save_path, nrow=nrow, normalize=False)
            print(f"💾 Grid saved: {save_path}")
            
            # Save individual images if requested
            if save_individual:
                base_dir = os.path.dirname(save_path)
                name_base = os.path.splitext(os.path.basename(save_path))[0]
                individual_dir = os.path.join(base_dir, f"{name_base}_individual")
                os.makedirs(individual_dir, exist_ok=True)
                
                for i, img in enumerate(generated_images):
                    individual_path = os.path.join(individual_dir, f"synthetic_waste_{i:03d}.png")
                    vutils.save_image(img, individual_path, normalize=False)
                
                print(f"💾 Individual images saved: {individual_dir}")
        
        print(f"✅ Generated {num_images} synthetic waste images")
        return generated_images
    
    def generate_class_specific(self, 
                               waste_class: str,
                               num_images: int = 8,
                               attempts: int = 50,
                               save_path: str = None) -> torch.Tensor:
        """
        Generate images and attempt to filter by waste class
        (Note: This is a simplified approach - proper class conditioning would require a conditional GAN)
        
        Args:
            waste_class: Target waste class (for naming/organization)
            num_images: Number of images to keep
            attempts: Number of generation attempts
            save_path: Path to save filtered images
            
        Returns:
            Selected images tensor
        """
        
        if self.generator is None:
            print("❌ No generator loaded!")
            return None
        
        print(f"🎯 Generating {waste_class} waste images...")
        print(f"   Target: {num_images} images from {attempts} attempts")
        
        # Generate many images
        all_images = self.generate_images(attempts, save_path=None)
        
        # For now, just randomly select images
        # In a real implementation, you'd use a classifier to filter
        selected_indices = np.random.choice(attempts, num_images, replace=False)
        selected_images = all_images[selected_indices]
        
        if save_path:
            nrow = int(np.sqrt(num_images))
            vutils.save_image(selected_images, save_path, nrow=nrow, normalize=False)
            print(f"💾 {waste_class} images saved: {save_path}")
        
        return selected_images
    
    def create_augmentation_dataset(self, 
                                  output_dir: str = "synthetic_waste_dataset",
                                  images_per_class: int = 50,
                                  waste_classes: list = None):
        """
        Create a synthetic dataset for data augmentation
        
        Args:
            output_dir: Directory to save synthetic dataset
            images_per_class: Number of images per waste class
            waste_classes: List of waste classes to generate
        """
        
        if waste_classes is None:
            waste_classes = [
                "plastic", "organic", "paper", "glass", 
                "metal", "cardboard", "mixed", "electronic"
            ]
        
        print(f"📦 Creating synthetic waste dataset...")
        print(f"   Output: {output_dir}")
        print(f"   Classes: {len(waste_classes)}")
        print(f"   Images per class: {images_per_class}")
        print(f"   Total images: {len(waste_classes) * images_per_class}")
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        
        dataset_info = {
            "generated_at": datetime.now().isoformat(),
            "total_images": len(waste_classes) * images_per_class,
            "classes": waste_classes,
            "images_per_class": images_per_class,
            "image_size": "64x64",
            "format": "PNG"
        }
        
        for class_name in waste_classes:
            print(f"\n🗂️  Generating {class_name} images...")
            
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Generate images in batches
            batch_size = 16
            generated_count = 0
            
            while generated_count < images_per_class:
                remaining = min(batch_size, images_per_class - generated_count)
                
                # Generate batch
                images = self.generate_images(remaining, save_path=None)
                
                # Save individual images
                for i, img in enumerate(images):
                    img_path = os.path.join(class_dir, f"synthetic_{class_name}_{generated_count + i:03d}.png")
                    vutils.save_image(img, img_path, normalize=False)
                
                generated_count += remaining
                print(f"   Generated {generated_count}/{images_per_class} {class_name} images")
            
            dataset_info[f"{class_name}_count"] = generated_count
        
        # Save dataset info
        info_path = os.path.join(output_dir, "dataset_info.json")
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n✅ Synthetic dataset created!")
        print(f"📁 Location: {output_dir}")
        print(f"📊 Info: {info_path}")
        
        return dataset_info
    
    def create_samples_showcase(self, save_path: str = "synthetic_waste_showcase.png"):
        """Create a showcase of diverse synthetic waste images"""
        
        print(f"🎨 Creating synthetic waste showcase...")
        
        # Generate diverse samples with different seeds
        all_samples = []
        seeds = [42, 123, 456, 789, 999, 1337, 2020, 2023]
        
        for seed in seeds:
            samples = self.generate_images(8, seed=seed, save_path=None)
            all_samples.append(samples)
        
        # Combine all samples
        showcase = torch.cat(all_samples, dim=0)
        
        # Save showcase
        vutils.save_image(showcase, save_path, nrow=8, normalize=False)
        print(f"💾 Showcase saved: {save_path}")
        
        return showcase


def demo_synthetic_generation():
    """Demo of synthetic waste image generation"""
    
    print("🎯 Synthetic Waste Image Generation Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = SyntheticWasteGenerator()
    
    if generator.generator is None:
        print("❌ No trained model found!")
        print("   Please run waste_gan_trainer.py first to train a GAN model")
        return
    
    # Create output directory
    output_dir = "synthetic_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate basic samples
    print(f"\n1️⃣ Generating basic samples...")
    basic_samples = generator.generate_images(
        num_images=16,
        seed=42,
        save_path=f"{output_dir}/basic_samples.png"
    )
    
    # 2. Generate larger batch
    print(f"\n2️⃣ Generating larger batch...")
    large_batch = generator.generate_images(
        num_images=64,
        save_path=f"{output_dir}/large_batch.png",
        save_individual=True
    )
    
    # 3. Generate class-specific samples (simulated)
    print(f"\n3️⃣ Generating class-specific samples...")
    waste_classes = ["plastic", "organic", "paper", "metal"]
    
    for class_name in waste_classes:
        class_samples = generator.generate_class_specific(
            waste_class=class_name,
            num_images=9,
            attempts=30,
            save_path=f"{output_dir}/{class_name}_samples.png"
        )
    
    # 4. Create showcase
    print(f"\n4️⃣ Creating diverse showcase...")
    showcase = generator.create_samples_showcase(
        save_path=f"{output_dir}/diverse_showcase.png"
    )
    
    # 5. Create small augmentation dataset
    print(f"\n5️⃣ Creating augmentation dataset...")
    dataset_info = generator.create_augmentation_dataset(
        output_dir=f"{output_dir}/augmentation_dataset",
        images_per_class=20,  # Small demo dataset
        waste_classes=["plastic", "organic", "paper", "metal"]
    )
    
    print(f"\n✅ Demo completed!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"\n📊 Generated:")
    print(f"   • Basic samples: 16 images")
    print(f"   • Large batch: 64 images")
    print(f"   • Class samples: 4 classes × 9 images")
    print(f"   • Diverse showcase: 64 images")
    print(f"   • Augmentation dataset: {dataset_info['total_images']} images")
    
    print(f"\n💡 Usage for your MobileViT:")
    print(f"   1. Use synthetic images for data augmentation")
    print(f"   2. Test model robustness with synthetic data")
    print(f"   3. Expand training dataset with generated images")
    print(f"   4. Create balanced datasets across waste types")


if __name__ == "__main__":
    demo_synthetic_generation()