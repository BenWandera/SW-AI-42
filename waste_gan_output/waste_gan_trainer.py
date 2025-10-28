"""
Waste GAN Training Script
Train a GAN to generate synthetic waste images for data augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import glob
from pathlib import Path


class WasteImageDataset(Dataset):
    """Dataset for loading waste images"""
    
    def __init__(self, image_folder: str, transform=None, image_size: int = 64):
        self.image_folder = image_folder
        self.image_size = image_size
        
        # Find all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))
        
        print(f"Found {len(self.image_paths)} images in {image_folder}")
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            return torch.randn(3, self.image_size, self.image_size)


class Generator(nn.Module):
    """Lightweight Generator for waste images"""
    
    def __init__(self, latent_dim=100, img_channels=3, features=64):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Generator layers
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


class Discriminator(nn.Module):
    """Lightweight Discriminator for waste images"""
    
    def __init__(self, img_channels=3, features=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: img_channels x 64 x 64
            nn.Conv2d(img_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # features x 32 x 32
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # features*2 x 16 x 16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # features*4 x 8 x 8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # features*8 x 4 x 4
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class WasteGAN:
    """Complete GAN system for waste image generation"""
    
    def __init__(self, latent_dim=100, lr=0.0002, beta1=0.5, device=None):
        self.latent_dim = latent_dim
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🎯 Using device: {self.device}")
        
        # Initialize networks
        self.generator = Generator(latent_dim=latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Initialize weights
        self._weights_init(self.generator)
        self._weights_init(self.discriminator)
        
        # Loss and optimizers
        self.criterion = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Training history
        self.history = {'g_loss': [], 'd_loss': [], 'real_score': [], 'fake_score': []}
        
        print(f"✅ WasteGAN initialized")
        print(f"   Generator: {sum(p.numel() for p in self.generator.parameters()):,} parameters")
        print(f"   Discriminator: {sum(p.numel() for p in self.discriminator.parameters()):,} parameters")
    
    def _weights_init(self, model):
        """Initialize model weights"""
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def train(self, dataset_path, num_epochs=50, batch_size=64, save_interval=10, output_dir="waste_gan_output"):
        """Train the GAN"""
        
        print(f"\n🚀 Starting GAN Training")
        print(f"   Dataset: {dataset_path}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch Size: {batch_size}")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        
        # Create dataset
        dataset = WasteImageDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        if len(dataset) == 0:
            print("❌ No images found! Creating demo data...")
            self._create_demo_data(dataset_path)
            dataset = WasteImageDataset(dataset_path)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Fixed noise for tracking progress
        fixed_noise = torch.randn(64, self.latent_dim, 1, 1, device=self.device)
        
        print(f"\n📊 Training Progress:")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            g_losses = []
            d_losses = []
            real_scores = []
            fake_scores = []
            
            for i, real_data in enumerate(dataloader):
                batch_size_current = real_data.size(0)
                real_data = real_data.to(self.device)
                
                # Train Discriminator
                self.discriminator.zero_grad()
                
                # Real data
                real_labels = torch.ones(batch_size_current, device=self.device)
                real_output = self.discriminator(real_data)
                d_loss_real = self.criterion(real_output, real_labels)
                d_loss_real.backward()
                
                # Fake data
                noise = torch.randn(batch_size_current, self.latent_dim, 1, 1, device=self.device)
                fake_data = self.generator(noise)
                fake_labels = torch.zeros(batch_size_current, device=self.device)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                d_loss_fake.backward()
                
                d_loss = d_loss_real + d_loss_fake
                self.optimizer_d.step()
                
                # Train Generator
                self.generator.zero_grad()
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, real_labels)  # Trick discriminator
                g_loss.backward()
                self.optimizer_g.step()
                
                # Store metrics
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                real_scores.append(real_output.mean().item())
                fake_scores.append(fake_output.mean().item())
                
                if i % 50 == 0:
                    print(f"[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] "
                          f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                          f"D(x): {real_output.mean().item():.4f} D(G(z)): {fake_output.mean().item():.4f}")
            
            # Epoch summary
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            avg_real_score = np.mean(real_scores)
            avg_fake_score = np.mean(fake_scores)
            
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['real_score'].append(avg_real_score)
            self.history['fake_score'].append(avg_fake_score)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
            print(f"  Real Score: {avg_real_score:.4f}, Fake Score: {avg_fake_score:.4f}")
            
            # Save samples and model
            if epoch % save_interval == 0 or epoch == num_epochs - 1:
                self._save_samples(fixed_noise, epoch, output_dir)
                self._save_model(epoch, output_dir)
                self._save_plots(output_dir)
        
        print(f"\n✅ Training completed! Output saved to: {output_dir}")
    
    def _create_demo_data(self, folder):
        """Create demo data if no real images found"""
        os.makedirs(folder, exist_ok=True)
        
        print(f"Creating demo images in {folder}...")
        for i in range(20):
            # Create colorful demo images
            img = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(os.path.join(folder, f"demo_{i:03d}.png"))
        
        print(f"Created 20 demo images")
    
    def _save_samples(self, fixed_noise, epoch, output_dir):
        """Save generated samples"""
        with torch.no_grad():
            fake_images = self.generator(fixed_noise)
            fake_images = (fake_images + 1) / 2.0  # Denormalize
            
            sample_path = f"{output_dir}/samples/epoch_{epoch:03d}.png"
            vutils.save_image(fake_images, sample_path, normalize=False, nrow=8)
            print(f"💾 Saved samples: {sample_path}")
    
    def _save_model(self, epoch, output_dir):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'history': self.history,
            'latent_dim': self.latent_dim
        }
        
        model_path = f"{output_dir}/models/wastegan_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, model_path)
        
        # Save as latest
        latest_path = f"{output_dir}/models/wastegan_latest.pth"
        torch.save(checkpoint, latest_path)
        
        print(f"💾 Saved model: {model_path}")
    
    def _save_plots(self, output_dir):
        """Save training plots"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history['g_loss'], label='Generator', alpha=0.7)
        axes[0].plot(self.history['d_loss'], label='Discriminator', alpha=0.7)
        axes[0].set_title('Training Losses')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Score plot
        axes[1].plot(self.history['real_score'], label='Real Score D(x)', alpha=0.7)
        axes[1].plot(self.history['fake_score'], label='Fake Score D(G(z))', alpha=0.7)
        axes[1].set_title('Discriminator Scores')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = f"{output_dir}/training_progress.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved plots: {plot_path}")
    
    def generate_images(self, num_images=64, save_path=None):
        """Generate synthetic images"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_images, self.latent_dim, 1, 1, device=self.device)
            generated = self.generator(noise)
            generated = (generated + 1) / 2.0  # Denormalize
            
            if save_path:
                vutils.save_image(generated, save_path, normalize=False, nrow=8)
                print(f"💾 Generated images saved: {save_path}")
        
        self.generator.train()
        return generated
    
    def load_model(self, checkpoint_path):
        """Load trained model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.history = checkpoint.get('history', {})
        
        print(f"✅ Model loaded from: {checkpoint_path}")


def main():
    """Main training function"""
    print("🎯 Waste GAN Training System")
    print("=" * 40)
    
    # Initialize GAN
    gan = WasteGAN(
        latent_dim=100,
        lr=0.0002,
        beta1=0.5
    )
    
    # Training configuration
    dataset_path = "realwaste/RealWaste"  # Your waste images
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"⚠️  Dataset not found: {dataset_path}")
        print("   Using demo data instead...")
        dataset_path = "demo_waste_images"
    
    # Train the GAN
    gan.train(
        dataset_path=dataset_path,
        num_epochs=30,  # Adjust based on your needs
        batch_size=32,  # Smaller for less memory usage
        save_interval=5,
        output_dir="waste_gan_output"
    )
    
    # Generate final samples
    print(f"\n🎨 Generating final synthetic waste images...")
    gan.generate_images(
        num_images=32,
        save_path="waste_gan_output/final_generated_waste.png"
    )
    
    print(f"\n✅ GAN training completed!")
    print(f"📁 Check 'waste_gan_output' folder for:")
    print(f"   • Generated samples in 'samples/' folder")
    print(f"   • Trained models in 'models/' folder")
    print(f"   • Training plots: 'training_progress.png'")
    print(f"   • Final samples: 'final_generated_waste.png'")


if __name__ == "__main__":
    main()