"""
Comprehensive EDA for RealWaste Dataset
This script provides a thorough exploratory data analysis of the RealWaste image classification dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageStat
import warnings
from collections import defaultdict, Counter
from pathlib import Path
import json
from datetime import datetime
import cv2

warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class RealWasteEDA:
    """
    Comprehensive EDA class for RealWaste dataset
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.categories = []
        self.dataset_info = {}
        self.image_stats = []
        self.results_dir = Path("realwaste_eda_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self._discover_categories()
        
    def _discover_categories(self):
        """Discover all categories in the dataset"""
        self.categories = []
        for item in self.dataset_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if item.name not in ['eda_visualizations', 'image_analysis']:
                    self.categories.append(item.name)
        
        print(f"Found {len(self.categories)} categories:")
        for i, cat in enumerate(sorted(self.categories), 1):
            print(f"  {i}. {cat}")
    
    def collect_basic_stats(self):
        """Collect basic statistics about the dataset"""
        print("\n" + "="*60)
        print("COLLECTING BASIC DATASET STATISTICS")
        print("="*60)
        
        total_images = 0
        category_counts = {}
        
        for category in self.categories:
            category_path = self.dataset_path / category
            if category_path.exists():
                image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png")) + list(category_path.glob("*.jpeg"))
                count = len(image_files)
                category_counts[category] = count
                total_images += count
                print(f"  {category}: {count} images")
        
        self.dataset_info = {
            'total_images': total_images,
            'total_categories': len(self.categories),
            'category_counts': category_counts,
            'categories': self.categories
        }
        
        print(f"\nTotal Images: {total_images}")
        print(f"Total Categories: {len(self.categories)}")
        
        return category_counts
    
    def analyze_class_distribution(self, category_counts):
        """Analyze and visualize class distribution"""
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
        df = df.sort_values('Count', ascending=False)
        
        print(f"Mean images per class: {df['Count'].mean():.2f}")
        print(f"Median images per class: {df['Count'].median():.2f}")
        print(f"Standard deviation: {df['Count'].std():.2f}")
        print(f"Min images: {df['Count'].min()} ({df.loc[df['Count'].idxmin(), 'Category']})")
        print(f"Max images: {df['Count'].max()} ({df.loc[df['Count'].idxmax(), 'Category']})")
        
        imbalance_ratio = df['Count'].max() / df['Count'].min()
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 2:
            print("⚠️  Dataset shows class imbalance (ratio > 2)")
        else:
            print("✅ Dataset is relatively balanced")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')
        
        axes[0, 0].bar(range(len(df)), df['Count'], color=plt.cm.Set3(np.arange(len(df))))
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].set_title('Images per Category')
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels(df['Category'], rotation=45, ha='right')
        
        for i, (category, count) in enumerate(zip(df['Category'], df['Count'])):
            axes[0, 0].text(i, count + 5, str(count), ha='center', va='bottom')
        
        axes[0, 1].pie(df['Count'], labels=df['Category'], autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Percentage Distribution')
        
        axes[1, 0].barh(df['Category'], df['Count'], color=plt.cm.Set3(np.arange(len(df))))
        axes[1, 0].set_xlabel('Number of Images')
        axes[1, 0].set_title('Images per Category (Sorted)')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        axes[1, 1].boxplot(df['Count'], labels=['All Categories'])
        axes[1, 1].set_ylabel('Number of Images')
        axes[1, 1].set_title('Distribution Statistics')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        with open(self.results_dir / 'class_statistics.json', 'w') as f:
            json.dump({
                'category_counts': category_counts,
                'statistics': {
                    'mean': float(df['Count'].mean()),
                    'median': float(df['Count'].median()),
                    'std': float(df['Count'].std()),
                    'min': int(df['Count'].min()),
                    'max': int(df['Count'].max()),
                    'imbalance_ratio': float(imbalance_ratio)
                }
            }, f, indent=2)
    
    def analyze_image_properties(self, sample_size_per_class=20):
        """Analyze image properties like dimensions, file sizes, and color statistics"""
        print("\n" + "="*60)
        print("IMAGE PROPERTIES ANALYSIS")
        print("="*60)
        
        image_data = []
        
        for category in self.categories:
            category_path = self.dataset_path / category
            if not category_path.exists():
                continue
                
            image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
            
            if len(image_files) > sample_size_per_class:
                sample_files = np.random.choice(image_files, sample_size_per_class, replace=False)
            else:
                sample_files = image_files
            
            print(f"Analyzing {len(sample_files)} images from {category}...")
            
            for img_path in sample_files:
                try:
                    file_size = img_path.stat().st_size / (1024 * 1024)
                    
                    with Image.open(img_path) as img:
                        width, height = img.size
                        format_type = img.format
                        mode = img.mode
                        
                        stat = ImageStat.Stat(img)
                        mean_colors = stat.mean
                        std_colors = stat.stddev
                    
                    aspect_ratio = width / height
                    
                    image_data.append({
                        'category': category,
                        'filename': img_path.name,
                        'width': width,
                        'height': height,
                        'aspect_ratio': aspect_ratio,
                        'file_size_mb': file_size,
                        'format': format_type,
                        'mode': mode,
                        'mean_brightness': np.mean(mean_colors) if mean_colors else 0,
                        'color_std': np.mean(std_colors) if std_colors else 0
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        df_images = pd.DataFrame(image_data)
        
        if df_images.empty:
            print("No images could be processed!")
            return
        
        print(f"\nAnalyzed {len(df_images)} images total")
        print(f"Image dimensions range:")
        print(f"  Width: {df_images['width'].min()} - {df_images['width'].max()} pixels")
        print(f"  Height: {df_images['height'].min()} - {df_images['height'].max()} pixels")
        print(f"File size range: {df_images['file_size_mb'].min():.3f} - {df_images['file_size_mb'].max():.3f} MB")
        print(f"Aspect ratio range: {df_images['aspect_ratio'].min():.3f} - {df_images['aspect_ratio'].max():.3f}")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Image Properties Analysis', fontsize=16, fontweight='bold')
        
        scatter = axes[0, 0].scatter(df_images['width'], df_images['height'], 
                                   c=pd.Categorical(df_images['category']).codes, 
                                   alpha=0.6, cmap='tab10')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Height (pixels)')
        axes[0, 0].set_title('Image Dimensions Distribution')
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].hist(df_images['aspect_ratio'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(1.0, color='red', linestyle='--', label='Square (1:1)')
        axes[0, 1].axvline(4/3, color='orange', linestyle='--', label='4:3')
        axes[0, 1].axvline(16/9, color='green', linestyle='--', label='16:9')
        axes[0, 1].set_xlabel('Aspect Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Aspect Ratio Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].hist(df_images['file_size_mb'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('File Size (MB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('File Size Distribution')
        axes[1, 0].grid(alpha=0.3)
        
        categories = df_images['category'].unique()
        for i, cat in enumerate(categories):
            cat_data = df_images[df_images['category'] == cat]['mean_brightness']
            axes[1, 1].hist(cat_data, alpha=0.6, label=cat, bins=20)
        axes[1, 1].set_xlabel('Mean Brightness')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Brightness Distribution by Category')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(alpha=0.3)
        
        df_melted = pd.melt(df_images, id_vars=['category'], 
                          value_vars=['width', 'height'],
                          var_name='dimension', value_name='pixels')
        sns.boxplot(data=df_melted, x='category', y='pixels', hue='dimension', ax=axes[2, 0])
        axes[2, 0].set_xlabel('Category')
        axes[2, 0].set_ylabel('Pixels')
        axes[2, 0].set_title('Image Dimensions by Category')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        format_counts = df_images['format'].value_counts()
        axes[2, 1].pie(format_counts.values, labels=format_counts.index, autopct='%1.1f%%')
        axes[2, 1].set_title('File Format Distribution')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'image_properties.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        summary_stats = {
            'dimension_stats': {
                'width': {
                    'mean': float(df_images['width'].mean()),
                    'std': float(df_images['width'].std()),
                    'min': int(df_images['width'].min()),
                    'max': int(df_images['width'].max())
                },
                'height': {
                    'mean': float(df_images['height'].mean()),
                    'std': float(df_images['height'].std()),
                    'min': int(df_images['height'].min()),
                    'max': int(df_images['height'].max())
                }
            },
            'file_size_stats': {
                'mean_mb': float(df_images['file_size_mb'].mean()),
                'std_mb': float(df_images['file_size_mb'].std()),
                'min_mb': float(df_images['file_size_mb'].min()),
                'max_mb': float(df_images['file_size_mb'].max())
            },
            'aspect_ratio_stats': {
                'mean': float(df_images['aspect_ratio'].mean()),
                'std': float(df_images['aspect_ratio'].std()),
                'min': float(df_images['aspect_ratio'].min()),
                'max': float(df_images['aspect_ratio'].max())
            },
            'format_distribution': format_counts.to_dict()
        }
        
        with open(self.results_dir / 'image_properties_stats.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        df_images.to_csv(self.results_dir / 'detailed_image_analysis.csv', index=False)
        
        return df_images
    
    def analyze_color_properties(self, sample_size_per_class=10):
        """Analyze color properties of images"""
        print("\n" + "="*60)
        print("COLOR PROPERTIES ANALYSIS")
        print("="*60)
        
        color_data = []
        
        for category in self.categories:
            category_path = self.dataset_path / category
            if not category_path.exists():
                continue
                
            image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
            
            if len(image_files) > sample_size_per_class:
                sample_files = np.random.choice(image_files, sample_size_per_class, replace=False)
            else:
                sample_files = image_files
            
            print(f"Analyzing colors in {len(sample_files)} images from {category}...")
            
            for img_path in sample_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    mean_rgb = np.mean(img_rgb, axis=(0, 1))
                    std_rgb = np.std(img_rgb, axis=(0, 1))
                    
                    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mean_hsv = np.mean(img_hsv, axis=(0, 1))
                    
                    unique_colors = len(np.unique(img_rgb.reshape(-1, 3), axis=0))
                    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
                    color_diversity = unique_colors / total_pixels
                    
                    color_data.append({
                        'category': category,
                        'filename': img_path.name,
                        'mean_r': mean_rgb[0],
                        'mean_g': mean_rgb[1],
                        'mean_b': mean_rgb[2],
                        'std_r': std_rgb[0],
                        'std_g': std_rgb[1],
                        'std_b': std_rgb[2],
                        'mean_h': mean_hsv[0],
                        'mean_s': mean_hsv[1],
                        'mean_v': mean_hsv[2],
                        'color_diversity': color_diversity,
                        'brightness': np.mean(mean_rgb)
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        if not color_data:
            print("No color data could be extracted!")
            return
        
        df_colors = pd.DataFrame(color_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Color Properties Analysis', fontsize=16, fontweight='bold')
        
        categories = df_colors['category'].unique()
        x_pos = np.arange(len(categories))
        
        mean_r = [df_colors[df_colors['category'] == cat]['mean_r'].mean() for cat in categories]
        mean_g = [df_colors[df_colors['category'] == cat]['mean_g'].mean() for cat in categories]
        mean_b = [df_colors[df_colors['category'] == cat]['mean_b'].mean() for cat in categories]
        
        width = 0.25
        axes[0, 0].bar(x_pos - width, mean_r, width, label='Red', color='red', alpha=0.7)
        axes[0, 0].bar(x_pos, mean_g, width, label='Green', color='green', alpha=0.7)
        axes[0, 0].bar(x_pos + width, mean_b, width, label='Blue', color='blue', alpha=0.7)
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Mean RGB Value')
        axes[0, 0].set_title('Average RGB Values by Category')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(categories, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        for cat in categories:
            cat_brightness = df_colors[df_colors['category'] == cat]['brightness']
            axes[0, 1].hist(cat_brightness, alpha=0.6, label=cat, bins=15)
        axes[0, 1].set_xlabel('Brightness')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Brightness Distribution by Category')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(alpha=0.3)
        
        sns.boxplot(data=df_colors, x='category', y='color_diversity', ax=axes[0, 2])
        axes[0, 2].set_xlabel('Category')
        axes[0, 2].set_ylabel('Color Diversity')
        axes[0, 2].set_title('Color Diversity by Category')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        sns.scatterplot(data=df_colors, x='mean_h', y='mean_s', hue='category', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Mean Hue')
        axes[1, 0].set_ylabel('Mean Saturation')
        axes[1, 0].set_title('Hue vs Saturation by Category')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        df_colors['color_variance'] = (df_colors['std_r'] + df_colors['std_g'] + df_colors['std_b']) / 3
        sns.boxplot(data=df_colors, x='category', y='color_variance', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Color Variance')
        axes[1, 1].set_title('Color Variance by Category')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 2].scatter(df_colors['mean_r'], df_colors['mean_g'], 
                          c=df_colors['mean_b'], s=50, alpha=0.6, cmap='viridis')
        axes[1, 2].set_xlabel('Mean Red')
        axes[1, 2].set_ylabel('Mean Green')
        axes[1, 2].set_title('RGB Color Space Distribution\n(Blue intensity as color)')
        colorbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
        colorbar.set_label('Mean Blue')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'color_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        color_stats = {}
        for cat in categories:
            cat_data = df_colors[df_colors['category'] == cat]
            color_stats[cat] = {
                'mean_rgb': [float(cat_data['mean_r'].mean()), 
                           float(cat_data['mean_g'].mean()), 
                           float(cat_data['mean_b'].mean())],
                'mean_brightness': float(cat_data['brightness'].mean()),
                'mean_color_diversity': float(cat_data['color_diversity'].mean()),
                'mean_color_variance': float(cat_data['color_variance'].mean())
            }
        
        with open(self.results_dir / 'color_analysis_stats.json', 'w') as f:
            json.dump(color_stats, f, indent=2)
        
        return df_colors
    
    def create_sample_images_grid(self, samples_per_class=4):
        """Create a grid showing sample images from each category"""
        print("\n" + "="*60)
        print("CREATING SAMPLE IMAGES GRID")
        print("="*60)
        
        n_categories = len(self.categories)
        fig, axes = plt.subplots(n_categories, samples_per_class, 
                               figsize=(samples_per_class * 3, n_categories * 3))
        fig.suptitle('Sample Images from Each Category', fontsize=16, fontweight='bold')
        
        if n_categories == 1:
            axes = axes.reshape(1, -1)
        
        for i, category in enumerate(self.categories):
            category_path = self.dataset_path / category
            if not category_path.exists():
                continue
                
            image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
            
            if len(image_files) >= samples_per_class:
                sample_files = np.random.choice(image_files, samples_per_class, replace=False)
            else:
                sample_files = image_files
                while len(sample_files) < samples_per_class:
                    sample_files = list(sample_files) + list(image_files)
                sample_files = sample_files[:samples_per_class]
            
            for j, img_path in enumerate(sample_files):
                try:
                    img = Image.open(img_path)
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    if j == 0:
                        axes[i, j].set_title(f"{category}\n{img_path.name}", fontsize=10)
                    else:
                        axes[i, j].set_title(img_path.name, fontsize=8)
                except Exception as e:
                    axes[i, j].text(0.5, 0.5, f'Error loading\n{img_path.name}', 
                                  ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_images_grid.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORT")
        print("="*60)
        
        report = []
        report.append("# RealWaste Dataset - Exploratory Data Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset Path: {self.dataset_path}")
        report.append("")
        
        report.append("## Dataset Overview")
        report.append(f"- **Total Categories**: {self.dataset_info['total_categories']}")
        report.append(f"- **Total Images**: {self.dataset_info['total_images']}")
        report.append(f"- **Average Images per Category**: {self.dataset_info['total_images'] / self.dataset_info['total_categories']:.2f}")
        report.append("")
        
        report.append("## Category Distribution")
        for category, count in sorted(self.dataset_info['category_counts'].items(), 
                                    key=lambda x: x[1], reverse=True):
            percentage = (count / self.dataset_info['total_images']) * 100
            report.append(f"- **{category}**: {count} images ({percentage:.1f}%)")
        report.append("")
        
        try:
            with open(self.results_dir / 'class_statistics.json', 'r') as f:
                class_stats = json.load(f)
            
            stats = class_stats['statistics']
            report.append("## Statistical Analysis")
            report.append(f"- **Mean images per class**: {stats['mean']:.2f}")
            report.append(f"- **Standard deviation**: {stats['std']:.2f}")
            report.append(f"- **Class imbalance ratio**: {stats['imbalance_ratio']:.2f}")
            
            if stats['imbalance_ratio'] > 2:
                report.append("- ⚠️ **Dataset shows class imbalance** (consider data augmentation or resampling)")
            else:
                report.append("- ✅ **Dataset is relatively balanced**")
            report.append("")
        except:
            pass
        
        report.append("## Key Findings")
        report.append("- Dataset contains waste classification images across multiple categories")
        report.append("- Images are in JPG format with varying dimensions")
        report.append("- Suitable for computer vision and machine learning tasks")
        report.append("")
        
        report.append("## Recommendations")
        report.append("- Consider data augmentation for smaller classes to improve balance")
        report.append("- Standardize image sizes for consistent model training")
        report.append("- Implement train/validation/test splits maintaining class proportions")
        report.append("- Consider using transfer learning with pre-trained models")
        report.append("")
        
        report.append("## Generated Files")
        report.append("- `class_distribution.png`: Class distribution visualizations")
        report.append("- `image_properties.png`: Image properties analysis")
        report.append("- `color_analysis.png`: Color properties analysis")
        report.append("- `sample_images_grid.png`: Sample images from each category")
        report.append("- `detailed_image_analysis.csv`: Detailed image metadata")
        report.append("- Various JSON files with statistical summaries")
        
        report_text = "\n".join(report)
        with open(self.results_dir / 'EDA_Summary_Report.md', 'w') as f:
            f.write(report_text)
        
        print("Summary report saved to: EDA_Summary_Report.md")
        print(f"All results saved to: {self.results_dir}")
        
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete EDA analysis"""
        print("🚀 Starting Comprehensive EDA for RealWaste Dataset")
        print("=" * 80)
        
        try:
            category_counts = self.collect_basic_stats()
            
            self.analyze_class_distribution(category_counts)
            
            self.analyze_image_properties(sample_size_per_class=30)
            
            self.analyze_color_properties(sample_size_per_class=15)
            
            self.create_sample_images_grid(samples_per_class=4)
            
            self.generate_summary_report()
            
            print("\n" + "="*80)
            print("✅ EDA ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"📁 All results saved to: {self.results_dir}")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the EDA"""
    dataset_path = Path(r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\realwaste\RealWaste")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    eda = RealWasteEDA(dataset_path)
    eda.run_complete_analysis()

if __name__ == "__main__":
    main()
