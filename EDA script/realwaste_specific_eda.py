"""
RealWaste Dataset EDA Script
Generates specific visualizations:
1. Class Distribution Graph
2. Qualitative Visualization Graph (sample images per category)
3. Pixel and Color Distribution Graph
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from collections import Counter
import random
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

class RealWasteEDA:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.categories = self._get_categories()
        self.image_data = []
        self.category_counts = {}
        
    def _get_categories(self):
        """Get all waste categories from the dataset"""
        categories = [d for d in os.listdir(self.dataset_path) 
                     if os.path.isdir(os.path.join(self.dataset_path, d)) 
                     and not d.startswith('.') 
                     and d not in ['eda_visualizations', 'image_analysis']]
        return sorted(categories)
    
    def collect_dataset_info(self):
        """Collect basic information about the dataset"""
        print("Collecting dataset information...")
        
        for category in self.categories:
            category_path = os.path.join(self.dataset_path, category)
            
            # Get all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(category_path, ext)))
                image_files.extend(glob.glob(os.path.join(category_path, ext.upper())))
            
            self.category_counts[category] = len(image_files)
            
            # Store image paths for later analysis
            for img_path in image_files:
                self.image_data.append({
                    'category': category,
                    'path': img_path,
                    'filename': os.path.basename(img_path)
                })
        
        print(f"Found {len(self.categories)} categories with {len(self.image_data)} total images")
    
    def create_class_distribution_graph(self):
        """Create class distribution visualization"""
        print("Creating class distribution graph...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        categories = list(self.category_counts.keys())
        counts = list(self.category_counts.values())
        
        bars = ax1.bar(range(len(categories)), counts, color=sns.color_palette("husl", len(categories)))
        ax1.set_title('Class Distribution - Number of Images per Category', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Waste Categories', fontsize=12)
        ax1.set_ylabel('Number of Images', fontsize=12)
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = sns.color_palette("husl", len(categories))
        wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('Class Distribution - Percentage Distribution', fontsize=14, fontweight='bold')
        
        # Improve pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        plt.tight_layout()
        plt.savefig('realwaste_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        total_images = sum(counts)
        print(f"\nClass Distribution Statistics:")
        print(f"Total Images: {total_images}")
        print(f"Number of Classes: {len(categories)}")
        print(f"Average Images per Class: {total_images/len(categories):.1f}")
        print(f"Most Common Class: {max(self.category_counts, key=self.category_counts.get)} ({max(counts)} images)")
        print(f"Least Common Class: {min(self.category_counts, key=self.category_counts.get)} ({min(counts)} images)")
        print("✅ Class distribution graph saved as 'realwaste_class_distribution.png'")
    
    def create_qualitative_visualization(self, samples_per_category=6):
        """Create qualitative visualization showing sample images from each category"""
        print("Creating qualitative visualization...")
        
        # Calculate grid dimensions
        n_categories = len(self.categories)
        n_cols = samples_per_category
        n_rows = n_categories
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_categories))
        fig.suptitle('Qualitative Visualization - Sample Images per Category', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        for row, category in enumerate(self.categories):
            # Get random sample images from this category
            category_images = [item for item in self.image_data if item['category'] == category]
            sample_images = random.sample(category_images, min(samples_per_category, len(category_images)))
            
            for col in range(n_cols):
                if n_rows == 1:
                    ax = axes[col]
                else:
                    ax = axes[row, col]
                
                if col < len(sample_images):
                    # Load and display image
                    img_path = sample_images[col]['path']
                    try:
                        img = Image.open(img_path)
                        ax.imshow(img)
                        ax.set_title(f"{category}\n{sample_images[col]['filename']}", 
                                   fontsize=8, pad=5)
                    except Exception as e:
                        ax.text(0.5, 0.5, 'Image\nLoad Error', ha='center', va='center',
                               transform=ax.transAxes, fontsize=12)
                        ax.set_title(f"{category}\nError", fontsize=8)
                else:
                    # Empty subplot
                    ax.set_title(f"{category}\nNo more images", fontsize=8)
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                           transform=ax.transAxes, fontsize=14, alpha=0.5)
                
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('realwaste_qualitative_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Qualitative visualization saved as 'realwaste_qualitative_visualization.png'")
    
    def analyze_pixel_color_distribution(self, sample_size=200):
        """Analyze pixel intensity and color distribution"""
        print("Analyzing pixel and color distribution...")
        
        # Sample images for analysis (more samples for better statistics)
        sampled_data = random.sample(self.image_data, min(sample_size, len(self.image_data)))
        
        pixel_intensities = []
        color_histograms = {'red': [], 'green': [], 'blue': []}
        brightness_values = []
        category_brightness = {cat: [] for cat in self.categories}
        category_colors = {cat: {'red': [], 'green': [], 'blue': []} for cat in self.categories}
        
        print(f"Processing {len(sampled_data)} sample images...")
        
        for i, item in enumerate(sampled_data):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(sampled_data)} images...")
            
            try:
                # Load image
                img = cv2.imread(item['path'])
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize for faster processing
                img_resized = cv2.resize(img, (128, 128))
                
                # Pixel intensity analysis
                gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
                pixel_intensities.extend(gray.flatten())
                
                # Color channel analysis
                red_pixels = img_resized[:,:,0].flatten()
                green_pixels = img_resized[:,:,1].flatten()
                blue_pixels = img_resized[:,:,2].flatten()
                
                color_histograms['red'].extend(red_pixels)
                color_histograms['green'].extend(green_pixels)
                color_histograms['blue'].extend(blue_pixels)
                
                # Category-specific color analysis
                category = item['category']
                category_colors[category]['red'].extend(red_pixels)
                category_colors[category]['green'].extend(green_pixels)
                category_colors[category]['blue'].extend(blue_pixels)
                
                # Brightness analysis
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                category_brightness[category].append(brightness)
                
            except Exception as e:
                print(f"Error processing {item['path']}: {e}")
                continue
        
        print("Creating pixel and color distribution visualizations...")
        
        # Create visualizations
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Pixel and Color Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall Pixel Intensity Distribution
        axes[0, 0].hist(pixel_intensities, bins=50, alpha=0.7, color='gray', edgecolor='black')
        axes[0, 0].set_title('Overall Pixel Intensity Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Pixel Intensity (0-255)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RGB Channel Distribution
        axes[0, 1].hist(color_histograms['red'], bins=50, alpha=0.6, color='red', 
                       label='Red', density=True)
        axes[0, 1].hist(color_histograms['green'], bins=50, alpha=0.6, color='green', 
                       label='Green', density=True)
        axes[0, 1].hist(color_histograms['blue'], bins=50, alpha=0.6, color='blue', 
                       label='Blue', density=True)
        axes[0, 1].set_title('RGB Channel Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Color Intensity (0-255)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Brightness Distribution by Category
        brightness_by_category = []
        category_labels = []
        for category in self.categories:
            if category_brightness[category]:
                brightness_by_category.append(category_brightness[category])
                category_labels.append(category[:10])  # Truncate long names
        
        if brightness_by_category:
            axes[0, 2].boxplot(brightness_by_category, labels=category_labels)
            axes[0, 2].set_title('Brightness Distribution by Category', fontweight='bold')
            axes[0, 2].set_xlabel('Category')
            axes[0, 2].set_ylabel('Average Brightness')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Average RGB Values by Category
        avg_colors_by_category = {}
        for category in self.categories:
            if category_colors[category]['red']:
                avg_colors_by_category[category] = {
                    'red': np.mean(category_colors[category]['red']),
                    'green': np.mean(category_colors[category]['green']),
                    'blue': np.mean(category_colors[category]['blue'])
                }
        
        if avg_colors_by_category:
            categories_short = [cat[:10] for cat in avg_colors_by_category.keys()]
            red_avgs = [avg_colors_by_category[cat]['red'] for cat in avg_colors_by_category.keys()]
            green_avgs = [avg_colors_by_category[cat]['green'] for cat in avg_colors_by_category.keys()]
            blue_avgs = [avg_colors_by_category[cat]['blue'] for cat in avg_colors_by_category.keys()]
            
            x = np.arange(len(categories_short))
            width = 0.25
            
            axes[1, 0].bar(x - width, red_avgs, width, label='Red', color='red', alpha=0.7)
            axes[1, 0].bar(x, green_avgs, width, label='Green', color='green', alpha=0.7)
            axes[1, 0].bar(x + width, blue_avgs, width, label='Blue', color='blue', alpha=0.7)
            
            axes[1, 0].set_title('Average RGB Values by Category', fontweight='bold')
            axes[1, 0].set_xlabel('Category')
            axes[1, 0].set_ylabel('Average Intensity')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(categories_short, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Overall Color Statistics
        color_stats = {
            'Red': np.mean(color_histograms['red']),
            'Green': np.mean(color_histograms['green']),
            'Blue': np.mean(color_histograms['blue'])
        }
        
        axes[1, 1].bar(color_stats.keys(), color_stats.values(), 
                      color=['red', 'green', 'blue'], alpha=0.7)
        axes[1, 1].set_title('Overall Average RGB Values', fontweight='bold')
        axes[1, 1].set_ylabel('Average Intensity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Color Variance Analysis
        color_variances = {
            'Red': np.var(color_histograms['red']),
            'Green': np.var(color_histograms['green']),
            'Blue': np.var(color_histograms['blue'])
        }
        
        axes[1, 2].bar(color_variances.keys(), color_variances.values(), 
                      color=['red', 'green', 'blue'], alpha=0.7)
        axes[1, 2].set_title('Color Channel Variance', fontweight='bold')
        axes[1, 2].set_ylabel('Variance')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Pixel Intensity Statistics
        intensity_stats = {
            'Mean': np.mean(pixel_intensities),
            'Median': np.median(pixel_intensities),
            'Std Dev': np.std(pixel_intensities)
        }
        
        axes[2, 0].bar(intensity_stats.keys(), intensity_stats.values(), 
                      color='gray', alpha=0.7)
        axes[2, 0].set_title('Pixel Intensity Statistics', fontweight='bold')
        axes[2, 0].set_ylabel('Value')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Brightness vs Category (bar chart)
        if category_brightness:
            cat_brightness_means = []
            cat_names = []
            for category in self.categories:
                if category_brightness[category]:
                    cat_brightness_means.append(np.mean(category_brightness[category]))
                    cat_names.append(category[:10])
            
            if cat_brightness_means:
                axes[2, 1].bar(range(len(cat_names)), cat_brightness_means, 
                              color=sns.color_palette("husl", len(cat_names)))
                axes[2, 1].set_title('Average Brightness by Category', fontweight='bold')
                axes[2, 1].set_xlabel('Category')
                axes[2, 1].set_ylabel('Average Brightness')
                axes[2, 1].set_xticks(range(len(cat_names)))
                axes[2, 1].set_xticklabels(cat_names, rotation=45, ha='right')
                axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Color Distribution Summary
        axes[2, 2].text(0.1, 0.8, f"Color Analysis Summary:", fontsize=14, fontweight='bold',
                       transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.7, f"• Total pixels analyzed: {len(pixel_intensities):,}", 
                       fontsize=12, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.6, f"• Avg Red: {np.mean(color_histograms['red']):.1f}", 
                       fontsize=12, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.5, f"• Avg Green: {np.mean(color_histograms['green']):.1f}", 
                       fontsize=12, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.4, f"• Avg Blue: {np.mean(color_histograms['blue']):.1f}", 
                       fontsize=12, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.3, f"• Avg Brightness: {np.mean(brightness_values):.1f}", 
                       fontsize=12, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.2, f"• Intensity Range: {min(pixel_intensities)}-{max(pixel_intensities)}", 
                       fontsize=12, transform=axes[2, 2].transAxes)
        axes[2, 2].set_xlim(0, 1)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('realwaste_pixel_color_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print color statistics
        print(f"\nPixel and Color Distribution Statistics:")
        print(f"Average Red Intensity: {np.mean(color_histograms['red']):.2f}")
        print(f"Average Green Intensity: {np.mean(color_histograms['green']):.2f}")
        print(f"Average Blue Intensity: {np.mean(color_histograms['blue']):.2f}")
        print(f"Overall Average Brightness: {np.mean(brightness_values):.2f}")
        print(f"Pixel Intensity Range: {min(pixel_intensities)} - {max(pixel_intensities)}")
        print("✅ Pixel and color distribution graph saved as 'realwaste_pixel_color_distribution.png'")
    
    def generate_summary_report(self):
        """Generate a summary report of the EDA"""
        print("\n" + "="*60)
        print("REALWASTE DATASET EDA SUMMARY")
        print("="*60)
        
        total_images = sum(self.category_counts.values())
        
        print(f"\nDataset Overview:")
        print(f"  Total Images: {total_images:,}")
        print(f"  Number of Categories: {len(self.categories)}")
        print(f"  Categories: {', '.join(self.categories)}")
        
        print(f"\nClass Distribution:")
        for category, count in sorted(self.category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_images) * 100
            print(f"  {category:<20}: {count:>6} images ({percentage:>5.1f}%)")
        
        # Class balance analysis
        max_count = max(self.category_counts.values())
        min_count = min(self.category_counts.values())
        imbalance_ratio = max_count / min_count
        
        print(f"\nClass Balance Analysis:")
        print(f"  Most Common Class: {max_count} images")
        print(f"  Least Common Class: {min_count} images")
        print(f"  Imbalance Ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2:
            print("  ⚠️ Dataset shows class imbalance - consider balancing techniques")
        else:
            print("  ✅ Dataset is reasonably balanced")
        
        print(f"\nGenerated Visualizations:")
        print(f"  📊 realwaste_class_distribution.png - Class distribution charts")
        print(f"  🖼️ realwaste_qualitative_visualization.png - Sample images per category")
        print(f"  🎨 realwaste_pixel_color_distribution.png - Pixel and color analysis")
    
    def run_complete_eda(self):
        """Run the complete EDA analysis"""
        print("Starting Comprehensive EDA for RealWaste Dataset")
        print("="*50)
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Collect dataset information
        self.collect_dataset_info()
        
        # Generate all visualizations
        self.create_class_distribution_graph()
        self.create_qualitative_visualization()
        self.analyze_pixel_color_distribution()
        
        # Generate summary report
        self.generate_summary_report()
        
        print(f"\n✅ EDA Complete! All visualizations saved to current directory.")

def main():
    """Main function to run the EDA"""
    dataset_path = r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\realwaste\RealWaste"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path '{dataset_path}' not found!")
        return
    
    # Create EDA instance and run analysis
    eda = RealWasteEDA(dataset_path)
    eda.run_complete_eda()

if __name__ == "__main__":
    main()