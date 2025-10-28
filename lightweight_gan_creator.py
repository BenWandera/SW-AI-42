"""
Lightweight GAN for Synthetic Waste Image Generation
Implements a simple but effective GAN architecture for generating waste images
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
from typing import Tuple, List


class WasteImageDataset(Dataset):
    """Dataset for loading waste images"""
    
    def __init__(self, image_folder: str, transform=None, image_size: int = 64):
        """
        Initialize dataset
        
        Args:
            image_folder: Path to folder containing waste images
            transform: Image transformations
            image_size: Size to resize images to
        """
        self.image_folder = image_folder
        self.image_size = image_size
        
        # Find all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))
        
        print(f"Found {len(self.image_paths)} images in {image_folder}")
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
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
            # Return a random tensor if image fails to load
            return torch.randn(3, self.image_size, self.image_size)


class Generator(nn.Module):
    """Generator network for creating synthetic waste images"""
    
    def __init__(self, latent_dim: int = 100, image_channels: int = 3, feature_maps: int = 64):
        """
        Initialize generator
        
        Args:
            latent_dim: Dimension of random noise vector
            image_channels: Number of image channels (3 for RGB)
            feature_maps: Base number of feature maps
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Generator architecture (4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64)
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # State: (feature_maps*8) x 4 x 4
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (feature_maps*4) x 8 x 8
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (feature_maps*2) x 16 x 16
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State: (feature_maps) x 32 x 32
            
            nn.ConvTranspose2d(feature_maps, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: image_channels x 64 x 64
        )
    
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """Discriminator network for distinguishing real from fake images"""
    
    def __init__(self, image_channels: int = 3, feature_maps: int = 64):
        """
        Initialize discriminator
        
        Args:
            image_channels: Number of image channels
            feature_maps: Base number of feature maps
        """
        super(Discriminator, self).__init__()
        
        # Discriminator architecture (64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1)
        self.main = nn.Sequential(
            # Input: image_channels x 64 x 64
            nn.Conv2d(image_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: feature_maps x 32 x 32
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*2) x 16 x 16
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*4) x 8 x 8
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*8) x 4 x 4
            
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class WasteGAN:
    """Complete GAN system for waste image generation"""
    
    def __init__(self, 
                 latent_dim: int = 100,
                 image_size: int = 64,
                 lr_g: float = 0.0002,
                 lr_d: float = 0.0002,
                 beta1: float = 0.5,
                 device: str = None):
        """
        Initialize WasteGAN
        
        Args:
            latent_dim: Dimension of noise vector
            image_size: Size of generated images
            lr_g: Generator learning rate
            lr_d: Discriminator learning rate
            beta1: Beta1 parameter for Adam optimizer
            device: Device to use ('cuda' or 'cpu')
        """
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
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
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(beta1, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))
        
        # Training metrics
        self.training_history = {
            'generator_losses': [],
            'discriminator_losses': [],
            'real_scores': [],
            'fake_scores': []
        }
        
        print(f"✅ WasteGAN initialized")
        print(f"   Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"   Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def _weights_init(self, m):
        """Initialize network weights"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def train(self, 
              dataset_path: str,
              num_epochs: int = 50,
              batch_size: int = 64,
              save_interval: int = 10,
              output_dir: str = "gan_outputs"):
        """
        Train the GAN
        
        Args:
            dataset_path: Path to training images
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            save_interval: Save model and samples every N epochs
            output_dir: Directory to save outputs
        """
        
        print(f"\n🚀 Starting GAN training")
        print(f"   Dataset: {dataset_path}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Output dir: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        
        # Create dataset and dataloader
        dataset = WasteImageDataset(dataset_path, image_size=self.image_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        if len(dataset) == 0:
            print("❌ No images found in dataset!")
            return
        
        # Fixed noise for consistent sample generation
        fixed_noise = torch.randn(64, self.latent_dim, 1, 1, device=self.device)
        
        # Labels for real and fake
        real_label = 1.0
        fake_label = 0.0
        
        print(f"\n📊 Training Progress:")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_real_score = 0.0
            epoch_fake_score = 0.0
            num_batches = 0
            
            for i, data in enumerate(dataloader):
                ############################
                # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                ############################
                
                # Train with real images
                self.discriminator.zero_grad()
                real_data = data.to(self.device)
                batch_size_actual = real_data.size(0)
                
                label = torch.full((batch_size_actual,), real_label, dtype=torch.float, device=self.device)
                output = self.discriminator(real_data)
                loss_d_real = self.criterion(output, label)
                loss_d_real.backward()
                d_x = output.mean().item()
                
                # Train with fake images
                noise = torch.randn(batch_size_actual, self.latent_dim, 1, 1, device=self.device)
                fake_data = self.generator(noise)
                label.fill_(fake_label)
                output = self.discriminator(fake_data.detach())
                loss_d_fake = self.criterion(output, label)
                loss_d_fake.backward()
                d_g_z1 = output.mean().item()
                
                loss_d = loss_d_real + loss_d_fake
                self.optimizer_d.step()
                
                ############################
                # (2) Update Generator: maximize log(D(G(z)))
                ############################
                
                self.generator.zero_grad()
                label.fill_(real_label)  # Fake labels are real for generator cost
                output = self.discriminator(fake_data)
                loss_g = self.criterion(output, label)
                loss_g.backward()
                d_g_z2 = output.mean().item()
                self.optimizer_g.step()
                
                # Accumulate metrics
                epoch_g_loss += loss_g.item()
                epoch_d_loss += loss_d.item()
                epoch_real_score += d_x
                epoch_fake_score += d_g_z2
                num_batches += 1
                
                # Print progress
                if i % 50 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                          f"D_loss: {loss_d.item():.4f} G_loss: {loss_g.item():.4f} "
                          f"D(x): {d_x:.4f} D(G(z)): {d_g_z1:.4f}/{d_g_z2:.4f}")
            
            # Calculate epoch averages
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            avg_real_score = epoch_real_score / num_batches
            avg_fake_score = epoch_fake_score / num_batches
            
            # Store training history
            self.training_history['generator_losses'].append(avg_g_loss)
            self.training_history['discriminator_losses'].append(avg_d_loss)
            self.training_history['real_scores'].append(avg_real_score)
            self.training_history['fake_scores'].append(avg_fake_score)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Generator Loss: {avg_g_loss:.4f}")
            print(f"  Discriminator Loss: {avg_d_loss:.4f}")
            print(f"  Real Score: {avg_real_score:.4f}")
            print(f"  Fake Score: {avg_fake_score:.4f}")
            
            # Save samples and model
            if epoch % save_interval == 0 or epoch == num_epochs - 1:
                self._save_samples(fixed_noise, epoch, output_dir)
                self._save_model(epoch, output_dir)
                self._save_training_plots(output_dir)
        
        print(f"\n✅ Training completed!")
        print(f"📁 Outputs saved to: {output_dir}")
    
    def _save_samples(self, fixed_noise, epoch, output_dir):
        """Save generated samples"""
        with torch.no_grad():
            fake_images = self.generator(fixed_noise)
            
            # Denormalize images
            fake_images = (fake_images + 1) / 2.0  # Convert from [-1,1] to [0,1]
            
            # Save sample grid
            sample_path = os.path.join(output_dir, "samples", f"epoch_{epoch:03d}.png")
            vutils.save_image(fake_images, sample_path, normalize=False, nrow=8)
            
            print(f"💾 Saved samples: {sample_path}")
    
    def _save_model(self, epoch, output_dir):
        """Save model checkpoints"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'training_history': self.training_history,
            'config': {
                'latent_dim': self.latent_dim,
                'image_size': self.image_size
            }
        }
        
        model_path = os.path.join(output_dir, "models", f"wastegan_epoch_{epoch:03d}.pth")
        torch.save(checkpoint, model_path)
        
        # Also save as latest
        latest_path = os.path.join(output_dir, "models", "wastegan_latest.pth")
        torch.save(checkpoint, latest_path)
        
        print(f"💾 Saved model: {model_path}")
    
    def _save_training_plots(self, output_dir):
        """Save training progress plots"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['generator_losses'], label='Generator')
        plt.plot(self.training_history['discriminator_losses'], label='Discriminator')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Score plot
        plt.subplot(1, 3, 2)
        plt.plot(self.training_history['real_scores'], label='Real Score D(x)')
        plt.plot(self.training_history['fake_scores'], label='Fake Score D(G(z))')
        plt.title('Discriminator Scores')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        # Combined plot
        plt.subplot(1, 3, 3)
        plt.plot(self.training_history['generator_losses'], label='G Loss', alpha=0.7)
        plt.plot(self.training_history['discriminator_losses'], label='D Loss', alpha=0.7)
        plt.title('Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_progress.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved training plots: {plot_path}")
    
    def generate_images(self, num_images: int = 64, save_path: str = None) -> torch.Tensor:
        """
        Generate synthetic waste images
        
        Args:
            num_images: Number of images to generate
            save_path: Optional path to save generated images
            
        Returns:
            Generated images tensor
        """
        
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_images, self.latent_dim, 1, 1, device=self.device)
            generated_images = self.generator(noise)
            
            # Denormalize
            generated_images = (generated_images + 1) / 2.0
            
            if save_path:
                vutils.save_image(generated_images, save_path, normalize=False, nrow=8)
                print(f"💾 Generated images saved: {save_path}")
        
        self.generator.train()
        return generated_images
    
    def load_model(self, checkpoint_path: str):
        """Load trained model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        print(f"✅ Model loaded from: {checkpoint_path}")
        print(f"   Trained for: {checkpoint['epoch']} epochs")


def create_lightweight_dcgan(latent_dim=100, img_size=64, img_channels=3, features=32):
    """
    Create a lightweight DCGAN implementation
    Much smaller than standard DCGAN for faster training
    """
    
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            
            # Calculate initial size
            init_size = img_size // 8  # 8 for 64x64
            
            self.init_conv = nn.Sequential(
                nn.Linear(latent_dim, features * 4 * init_size * init_size),
                nn.BatchNorm1d(features * 4 * init_size * init_size),
                nn.ReLU(inplace=True)
            )
            
            self.conv_blocks = nn.Sequential(
                # 8x8 -> 16x16
                nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 2),
                nn.ReLU(inplace=True),
                
                # 16x16 -> 32x32
                nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                
                # 32x32 -> 64x64
                nn.ConvTranspose2d(features, img_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )
            
        def forward(self, z):
            init_size = img_size // 8
            out = self.init_conv(z)
            out = out.view(out.shape[0], features * 4, init_size, init_size)
            img = self.conv_blocks(out)
            return img
    
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            
            def discriminator_block(in_feat, out_feat, normalize=True):
                layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1)]
                if normalize:
                    layers.append(nn.BatchNorm2d(out_feat))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers
            
            self.model = nn.Sequential(
                # 64x64 -> 32x32
                *discriminator_block(img_channels, features, normalize=False),
                # 32x32 -> 16x16
                *discriminator_block(features, features * 2),
                # 16x16 -> 8x8
                *discriminator_block(features * 2, features * 4),
                # 8x8 -> 4x4
                nn.Conv2d(features * 4, 1, 4, 1, 0),
                nn.Sigmoid()
            )
            
        def forward(self, img):
            validity = self.model(img)
            return validity.view(-1).squeeze()
    
    generator = Generator()
    discriminator = Discriminator()
    
    return generator, discriminator

def create_mobilegan(latent_dim=64, img_size=64, img_channels=3):
    """
    Create an ultra-lightweight MobileGAN for mobile/edge deployment
    """
    
    class MobileGenerator(nn.Module):
        def __init__(self):
            super(MobileGenerator, self).__init__()
            
            # Depthwise separable convolutions for efficiency
            def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
                return nn.Sequential(
                    # Depthwise
                    nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU6(inplace=True),
                    # Pointwise
                    nn.ConvTranspose2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU6(inplace=True)
                )
            
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, 32 * 8 * 8),
                nn.ReLU6(inplace=True)
            )
            
            self.conv_layers = nn.Sequential(
                # 8x8 -> 16x16
                depthwise_separable_conv(32, 16, 4, 2, 1),
                # 16x16 -> 32x32
                depthwise_separable_conv(16, 8, 4, 2, 1),
                # 32x32 -> 64x64
                nn.ConvTranspose2d(8, img_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )
            
        def forward(self, z):
            out = self.fc(z)
            out = out.view(out.shape[0], 32, 8, 8)
            img = self.conv_layers(out)
            return img
    
    class MobileDiscriminator(nn.Module):
        def __init__(self):
            super(MobileDiscriminator, self).__init__()
            
            def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
                return nn.Sequential(
                    # Depthwise
                    nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU6(inplace=True),
                    # Pointwise
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU6(inplace=True)
                )
            
            self.conv_layers = nn.Sequential(
                # 64x64 -> 32x32
                depthwise_separable_conv(img_channels, 8, 4, 2, 1),
                # 32x32 -> 16x16
                depthwise_separable_conv(8, 16, 4, 2, 1),
                # 16x16 -> 8x8
                depthwise_separable_conv(16, 32, 4, 2, 1),
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
        def forward(self, img):
            features = self.conv_layers(img)
            validity = self.classifier(features)
            return validity.squeeze()
    
    generator = MobileGenerator()
    discriminator = MobileDiscriminator()
    
    return generator, discriminator

def create_training_script():
    """Create a lightweight training script for the GAN models"""
    
    training_code = '''
"""
Lightweight GAN Training Script
Fast training for waste image generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class WasteDataset(Dataset):
    """Custom dataset for waste images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        
        # Collect all image paths
        for category_dir in self.root_dir.iterdir():
            if category_dir.is_dir():
                for img_path in category_dir.glob("*.jpg"):
                    self.images.append(img_path)
        
        print(f"Found {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def train_lightweight_gan(data_path, model_type="dcgan", epochs=50, batch_size=32, lr=0.0002):
    """Train a lightweight GAN on waste dataset"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = WasteDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Load models (import from the main script)
    if model_type == "dcgan":
        from lightweight_gan_creator import create_lightweight_dcgan
        generator, discriminator = create_lightweight_dcgan()
    else:  # mobilegan
        from lightweight_gan_creator import create_mobilegan
        generator, discriminator = create_mobilegan()
    
    generator.to(device)
    discriminator.to(device)
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Training loop
    for epoch in range(epochs):
        for i, real_imgs in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real images
            real_output = discriminator(real_imgs)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, 100 if model_type == "dcgan" else 64).to(device)
            fake_imgs = generator(z)
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_imgs)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            # Print progress
            if i % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        
        # Save sample images
        if epoch % 10 == 0:
            with torch.no_grad():
                z = torch.randn(16, 100 if model_type == "dcgan" else 64).to(device)
                fake_imgs = generator(z)
                save_image(fake_imgs, f"generated_samples_epoch_{epoch}.png", 
                          nrow=4, normalize=True)
    
    # Save models
    torch.save(generator.state_dict(), f"{model_type}_generator.pth")
    torch.save(discriminator.state_dict(), f"{model_type}_discriminator.pth")
    print(f"Models saved as {model_type}_generator.pth and {model_type}_discriminator.pth")

if __name__ == "__main__":
    # Example usage
    data_path = "C:/Users/Z-BOOK/OneDrive/Documents/DATASETS/realwaste/RealWaste"
    
    print("Starting lightweight GAN training...")
    print("Model options: 'dcgan' or 'mobilegan'")
    
    # Train lightweight DCGAN
    train_lightweight_gan(data_path, model_type="dcgan", epochs=20, batch_size=16)
'''
    
    script_path = Path("lightweight_gan_models/train_lightweight_gan.py")
    script_path.parent.mkdir(exist_ok=True)
    with open(script_path, "w") as f:
        f.write(training_code)
    
    print(f"✓ Training script created: {script_path}")
    return script_path

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("LIGHTWEIGHT GAN MODEL CREATOR")
    print("=" * 70)
    
    # Create models directory
    models_dir = Path("lightweight_gan_models")
    models_dir.mkdir(exist_ok=True)
    
    print("\n1. Creating lightweight DCGAN models...")
    gen_dcgan, disc_dcgan = create_lightweight_dcgan()
    
    # Count parameters
    gen_params = sum(p.numel() for p in gen_dcgan.parameters())
    disc_params = sum(p.numel() for p in disc_dcgan.parameters())
    total_dcgan = gen_params + disc_params
    
    print(f"✓ DCGAN created:")
    print(f"  Generator: {gen_params:,} parameters ({gen_params/1e6:.2f}M)")
    print(f"  Discriminator: {disc_params:,} parameters ({disc_params/1e6:.2f}M)")
    print(f"  Total: {total_dcgan:,} parameters ({total_dcgan/1e6:.2f}M)")
    
    print("\n2. Creating MobileGAN models...")
    gen_mobile, disc_mobile = create_mobilegan()
    
    # Count parameters
    gen_mobile_params = sum(p.numel() for p in gen_mobile.parameters())
    disc_mobile_params = sum(p.numel() for p in disc_mobile.parameters())
    total_mobile = gen_mobile_params + disc_mobile_params
    
    print(f"✓ MobileGAN created:")
    print(f"  Generator: {gen_mobile_params:,} parameters ({gen_mobile_params/1e6:.2f}M)")
    print(f"  Discriminator: {disc_mobile_params:,} parameters ({disc_mobile_params/1e6:.2f}M)")
    print(f"  Total: {total_mobile:,} parameters ({total_mobile/1e6:.2f}M)")
    
    print("\n3. Creating training script...")
    training_script = create_training_script()
    
    print("\n4. Testing model generation...")
    
    # Test DCGAN
    with torch.no_grad():
        z = torch.randn(4, 100)  # DCGAN latent dim = 100
        fake_images = gen_dcgan(z)
        print(f"✓ DCGAN test generation: {fake_images.shape}")
    
    # Test MobileGAN
    with torch.no_grad():
        z = torch.randn(4, 64)  # MobileGAN latent dim = 64
        fake_images = gen_mobile(z)
        print(f"✓ MobileGAN test generation: {fake_images.shape}")
    
    # Save model architectures
    model_info = {
        "dcgan": {
            "generator_params": int(gen_params),
            "discriminator_params": int(disc_params),
            "total_params": int(total_dcgan),
            "latent_dim": 100,
            "img_size": 64,
            "memory_efficient": True
        },
        "mobilegan": {
            "generator_params": int(gen_mobile_params),
            "discriminator_params": int(disc_mobile_params),
            "total_params": int(total_mobile),
            "latent_dim": 64,
            "img_size": 64,
            "mobile_optimized": True,
            "ultra_lightweight": True
        }
    }
    
    with open(models_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY & NEXT STEPS")
    print("=" * 70)
    
    print(f"\n📁 Models saved in: {models_dir.absolute()}")
    print(f"📋 Model info: {models_dir / 'model_info.json'}")
    print(f"🚀 Training script: {training_script}")
    
    print("\n💡 Available lightweight models:")
    print(f"  1. DCGAN Lite: {total_dcgan/1e6:.2f}M parameters (balanced)")
    print(f"  2. MobileGAN: {total_mobile/1e6:.2f}M parameters (ultra-light)")
    
    print("\n🎯 Quick start commands:")
    print(f"  # Train DCGAN on your waste dataset")
    print(f"  python {training_script}")
    
    print("\n  # Or import models directly:")
    print("  from lightweight_gan_creator import create_lightweight_dcgan, create_mobilegan")
    print("  gen, disc = create_lightweight_dcgan()  # or create_mobilegan()")
    
    print("\n🔧 Integration with your GAN.py:")
    print("  1. Replace the build_generator() and build_discriminator() functions")
    print("  2. Use the imported models from this script")
    print("  3. Adjust latent_dim (100 for DCGAN, 64 for MobileGAN)")
    print("  4. Start with smaller batch sizes and learning rates")
    
    print(f"\n✅ Ready to train! Models are {total_mobile/1e6:.1f}x lighter than standard GANs.")

if __name__ == "__main__":
    main()