"""
Display Sample Images from Each RealWaste Category
Creates a grid showing one representative image from each waste category
"""

import os
import glob
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np

def get_sample_image_from_category(category_path, category_name):
    """Get a random sample image from a category"""
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(category_path, ext)))
        image_files.extend(glob.glob(os.path.join(category_path, ext.upper())))
    
    if not image_files:
        print(f"No images found in {category_name}")
        return None, None
    
    # Select a random image
    sample_image = random.choice(image_files)
    return sample_image, os.path.basename(sample_image)

def create_category_grid(dataset_path, output_filename="category_samples.png"):
    """Create a grid showing one sample from each category"""
    
    # Get all category directories
    categories = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d)) 
                 and not d.startswith('.') 
                 and d not in ['eda_visualizations', 'image_analysis']]
    
    categories = sorted(categories)
    print(f"Found {len(categories)} categories: {', '.join(categories)}")
    
    # Calculate grid dimensions
    n_categories = len(categories)
    cols = 3  # 3 columns
    rows = (n_categories + cols - 1) // cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    
    # If only one row, make axes a list
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    sample_info = []
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        sample_path, sample_filename = get_sample_image_from_category(category_path, category)
        
        if sample_path:
            try:
                # Load and display image
                img = Image.open(sample_path)
                axes[idx].imshow(img)
                axes[idx].set_title(f"{category}\n{sample_filename}", fontsize=12, fontweight='bold')
                axes[idx].axis('off')
                
                sample_info.append({
                    'category': category,
                    'filename': sample_filename,
                    'path': sample_path
                })
                
                print(f"✓ {category}: {sample_filename}")
                
            except Exception as e:
                print(f"✗ Error loading image from {category}: {e}")
                axes[idx].text(0.5, 0.5, f"Error loading\n{category}", 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(category, fontsize=12, fontweight='bold')
                axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f"No images\nin {category}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(category, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
    
    # Hide any unused subplots
    for idx in range(len(categories), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('RealWaste Dataset - Sample Images by Category', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure instead of showing
    
    print(f"\nSample grid saved as: {output_filename}")
    return sample_info

def create_detailed_montage(dataset_path, output_filename="detailed_category_montage.png"):
    """Create a more detailed montage with category information"""
    
    # Get all category directories
    categories = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d)) 
                 and not d.startswith('.') 
                 and d not in ['eda_visualizations', 'image_analysis']]
    
    categories = sorted(categories)
    
    # Create a large image to hold all samples
    img_size = 524  # Standard size for this dataset
    padding = 50
    text_height = 100
    
    # Calculate dimensions
    cols = 3
    rows = (len(categories) + cols - 1) // cols
    
    total_width = cols * img_size + (cols + 1) * padding
    total_height = rows * (img_size + text_height) + (rows + 1) * padding
    
    # Create the montage image
    montage = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(montage)
    
    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    sample_info = []
    
    for idx, category in enumerate(categories):
        row = idx // cols
        col = idx % cols
        
        # Calculate position
        x = col * (img_size + padding) + padding
        y = row * (img_size + text_height + padding) + padding
        
        category_path = os.path.join(dataset_path, category)
        sample_path, sample_filename = get_sample_image_from_category(category_path, category)
        
        if sample_path:
            try:
                # Load and resize image
                img = Image.open(sample_path)
                img = img.resize((img_size, img_size), Image.Lanczos)
                
                # Paste image
                montage.paste(img, (x, y))
                
                # Add category name
                text_y = y + img_size + 10
                draw.text((x, text_y), category, fill='black', font=font)
                draw.text((x, text_y + 30), sample_filename, fill='gray', font=small_font)
                
                # Count images in category
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    image_files.extend(glob.glob(os.path.join(category_path, ext)))
                    image_files.extend(glob.glob(os.path.join(category_path, ext.upper())))
                
                draw.text((x, text_y + 50), f"{len(image_files)} images", fill='blue', font=small_font)
                
                sample_info.append({
                    'category': category,
                    'filename': sample_filename,
                    'path': sample_path,
                    'image_count': len(image_files)
                })
                
                print(f"✓ Added {category}: {sample_filename} ({len(image_files)} images)")
                
            except Exception as e:
                print(f"✗ Error processing {category}: {e}")
                # Draw error placeholder
                draw.rectangle([x, y, x + img_size, y + img_size], outline='red', width=3)
                draw.text((x + 10, y + img_size//2), f"Error loading\n{category}", fill='red', font=font)
    
    # Add title
    title = "RealWaste Dataset - Category Samples"
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text((total_width//2 - title_width//2, 10), title, fill='black', font=font)
    
    # Save the montage
    montage.save(output_filename, quality=95)
    print(f"\nDetailed montage saved as: {output_filename}")
    
    return sample_info

def main():
    """Main function to create sample displays"""
    dataset_path = r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\realwaste\RealWaste"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found!")
        return
    
    print("Creating RealWaste Dataset Sample Displays...")
    print("=" * 50)
    
    # Set random seed for reproducible samples
    random.seed(42)
    
    # Create matplotlib grid
    print("\n1. Creating matplotlib grid...")
    sample_info_grid = create_category_grid(dataset_path, "realwaste_category_grid.png")
    
    print("\n2. Creating detailed montage...")
    sample_info_montage = create_detailed_montage(dataset_path, "realwaste_category_montage.png")
    
    print("\n" + "=" * 50)
    print("Sample Display Creation Complete!")
    print(f"Generated files:")
    print(f"  - realwaste_category_grid.png (matplotlib grid)")
    print(f"  - realwaste_category_montage.png (detailed montage)")
    
    # Print sample summary
    print(f"\nSample Images Selected:")
    for info in sample_info_montage:
        print(f"  {info['category']:<20}: {info['filename']} ({info['image_count']} total images)")

if __name__ == "__main__":
    main()