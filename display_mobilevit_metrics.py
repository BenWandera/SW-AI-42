"""
MobileViT Model Metrics Visualization
Display comprehensive metrics from trained model results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
import pandas as pd
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(json_path):
    """Load results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_classification_report(report_str):
    """Parse scikit-learn classification report string"""
    lines = report_str.strip().split('\n')
    
    # Parse class-wise metrics
    class_data = []
    for line in lines[2:-4]:  # Skip header and summary lines
        parts = line.split()
        if len(parts) >= 5:
            class_name = ' '.join(parts[:-4])
            precision = float(parts[-4])
            recall = float(parts[-3])
            f1_score = float(parts[-2])
            support = int(parts[-1])
            
            class_data.append({
                'class': class_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': support
            })
    
    # Parse overall metrics
    overall_accuracy = None
    macro_avg = None
    weighted_avg = None
    
    for line in lines[-4:]:
        if 'accuracy' in line:
            parts = line.split()
            overall_accuracy = float(parts[1])
        elif 'macro avg' in line:
            parts = line.split()
            macro_avg = {
                'precision': float(parts[2]),
                'recall': float(parts[3]),
                'f1_score': float(parts[4])
            }
        elif 'weighted avg' in line:
            parts = line.split()
            weighted_avg = {
                'precision': float(parts[2]),
                'recall': float(parts[3]),
                'f1_score': float(parts[4])
            }
    
    return class_data, overall_accuracy, macro_avg, weighted_avg


def create_accuracy_overview(results, save_dir):
    """Create overall accuracy metrics visualization"""
    
    val_acc = results['best_validation_accuracy']
    test_acc = results['final_test_accuracy']
    training_time = results['training_time_minutes']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Validation vs Test Accuracy
    ax1 = axes[0]
    metrics = ['Validation\nAccuracy', 'Test\nAccuracy']
    values = [val_acc, test_acc]
    colors = ['#3498db', '#2ecc71']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('MobileViT Model Accuracy', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Good (80%)')
    ax1.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Fair (70%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Training Info
    ax2 = axes[1]
    ax2.axis('off')
    
    info_text = f"""
    📊 MODEL PERFORMANCE SUMMARY
    ═══════════════════════════════════
    
    🎯 Best Validation Accuracy: {val_acc:.2f}%
    ✓ Final Test Accuracy: {test_acc:.2f}%
    
    📈 Dataset Statistics:
       • Total Images: {results['total_images']:,}
       • Number of Classes: {len(results['class_names'])}
       • Train/Val/Test Split: {results['data_split']['train']}/{results['data_split']['val']}/{results['data_split']['test']}%
    
    ⏱️  Training Time: {training_time:.1f} minutes
                        ({training_time/60:.1f} hours)
    
    🤖 Model: {results['model']}
    📁 Dataset: {results['dataset']}
    
    ✅ Model Status: TRAINED & VALIDATED
    """
    
    ax2.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mobilevit_accuracy_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Accuracy overview saved")


def create_per_class_metrics(class_data, save_dir):
    """Create per-class performance visualization"""
    
    df = pd.DataFrame(class_data)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Precision, Recall, F1-Score
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax1.bar(x - width, df['precision'], width, label='Precision', 
                    color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x, df['recall'], width, label='Recall',
                    color='#e74c3c', alpha=0.8, edgecolor='black')
    bars3 = ax1.bar(x + width, df['f1_score'], width, label='F1-Score',
                    color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Waste Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['class'], rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.3)
    
    # Plot 2: Support (Sample Count)
    ax2 = axes[1]
    bars = ax2.bar(df['class'], df['support'], color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Waste Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Test Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Test Set Distribution', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticklabels(df['class'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mobilevit_per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Per-class metrics saved")


def create_metrics_heatmap(class_data, save_dir):
    """Create heatmap of all metrics"""
    
    df = pd.DataFrame(class_data)
    
    # Create metrics matrix
    metrics_matrix = df[['precision', 'recall', 'f1_score']].values.T
    
    plt.figure(figsize=(12, 6))
    
    sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
               xticklabels=df['class'], yticklabels=['Precision', 'Recall', 'F1-Score'],
               cbar_kws={'label': 'Score'}, vmin=0.7, vmax=1.0, center=0.85,
               linewidths=1, linecolor='gray')
    
    plt.title('MobileViT Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Waste Category', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mobilevit_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Metrics heatmap saved")


def create_macro_weighted_comparison(macro_avg, weighted_avg, save_dir):
    """Compare macro vs weighted averages"""
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    macro_values = [macro_avg['precision'], macro_avg['recall'], macro_avg['f1_score']]
    weighted_values = [weighted_avg['precision'], weighted_avg['recall'], weighted_avg['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, macro_values, width, label='Macro Average',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, weighted_values, width, label='Weighted Average',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Macro vs Weighted Average Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.88, color='green', linestyle='--', alpha=0.5, label='Target')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mobilevit_macro_weighted_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Macro/weighted comparison saved")


def create_summary_dashboard(results, class_data, overall_accuracy, macro_avg, save_dir):
    """Create comprehensive summary dashboard"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Top: Model Info
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis('off')
    
    info_text = f"""
    MOBILEVIT WASTE CLASSIFICATION MODEL - COMPREHENSIVE METRICS REPORT
    ═══════════════════════════════════════════════════════════════════════════════════════════════
    
    MODEL: {results['model']}  |  DATASET: {results['dataset']}  |  TOTAL IMAGES: {results['total_images']:,}  |  CLASSES: {len(results['class_names'])}
    
    ✅ FINAL TEST ACCURACY: {results['final_test_accuracy']:.2f}%  |  🎯 BEST VALIDATION ACCURACY: {results['best_validation_accuracy']:.2f}%
    
    Training Time: {results['training_time_minutes']:.1f} minutes ({results['training_time_minutes']/60:.1f} hours)
    """
    
    ax_info.text(0.5, 0.5, info_text, fontsize=11, verticalalignment='center', 
                horizontalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Middle Left: Overall Accuracy Bar
    ax1 = fig.add_subplot(gs[1, 0])
    accuracies = ['Test\nAccuracy', 'Validation\nAccuracy']
    acc_values = [results['final_test_accuracy'], results['best_validation_accuracy']]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax1.barh(accuracies, acc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, value in zip(bars, acc_values):
        ax1.text(value, bar.get_y() + bar.get_height()/2,
                f' {value:.2f}%', va='center', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Overall Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Middle Center: Macro Averages
    ax2 = fig.add_subplot(gs[1, 1])
    macro_metrics = ['Precision', 'Recall', 'F1-Score']
    macro_vals = [macro_avg['precision'], macro_avg['recall'], macro_avg['f1_score']]
    colors2 = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax2.bar(macro_metrics, macro_vals, color=colors2, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, value in zip(bars, macro_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax2.set_title('Macro Average Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Middle Right: Top 5 & Bottom 5 Classes
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    
    df = pd.DataFrame(class_data)
    df_sorted = df.sort_values('f1_score', ascending=False)
    
    top5 = df_sorted.head(5)
    bottom5 = df_sorted.tail(5)
    
    summary_text = "TOP 5 CLASSES (F1-Score):\n"
    for idx, row in top5.iterrows():
        summary_text += f"  {row['class'][:15]:15s}: {row['f1_score']:.3f}\n"
    
    summary_text += "\nBOTTOM 5 CLASSES:\n"
    for idx, row in bottom5.iterrows():
        summary_text += f"  {row['class'][:15]:15s}: {row['f1_score']:.3f}\n"
    
    ax3.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Bottom: Per-class F1-Score Comparison
    ax4 = fig.add_subplot(gs[2, :])
    
    df_sorted = df.sort_values('f1_score', ascending=True)
    colors_gradient = plt.cm.RdYlGn(df_sorted['f1_score'])
    
    bars = ax4.barh(df_sorted['class'], df_sorted['f1_score'], 
                    color=colors_gradient, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars, df_sorted['f1_score']):
        ax4.text(value, bar.get_y() + bar.get_height()/2,
                f' {value:.3f}', va='center', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('F1-Score', fontsize=10, fontweight='bold')
    ax4.set_title('Per-Class F1-Score Ranking', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 1.1)
    ax4.axvline(x=0.88, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Overall Average')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.savefig(os.path.join(save_dir, 'mobilevit_summary_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Summary dashboard saved")


def main():
    """Main function to generate all metrics visualizations"""
    
    print("🚀 MobileViT Metrics Visualization")
    print("="*60)
    
    # Load results
    results_path = r"C:\Users\Z-BOOK\OneDrive\Documents\DATASETS\mobilevit_waste_results.json"
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return
    
    print(f"📂 Loading results from: {results_path}")
    results = load_results(results_path)
    
    # Parse classification report
    print(f"📊 Parsing classification report...")
    class_data, overall_accuracy, macro_avg, weighted_avg = parse_classification_report(
        results['classification_report']
    )
    
    # Create output directory
    save_dir = 'mobilevit_metrics'
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Creating visualizations in: {save_dir}")
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    
    create_accuracy_overview(results, save_dir)
    create_per_class_metrics(class_data, save_dir)
    create_metrics_heatmap(class_data, save_dir)
    create_macro_weighted_comparison(macro_avg, weighted_avg, save_dir)
    create_summary_dashboard(results, class_data, overall_accuracy, macro_avg, save_dir)
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"MOBILEVIT MODEL PERFORMANCE SUMMARY")
    print(f"="*60)
    print(f"✅ Test Accuracy: {results['final_test_accuracy']:.2f}%")
    print(f"🎯 Validation Accuracy: {results['best_validation_accuracy']:.2f}%")
    print(f"📈 Macro Precision: {macro_avg['precision']:.3f}")
    print(f"📈 Macro Recall: {macro_avg['recall']:.3f}")
    print(f"📈 Macro F1-Score: {macro_avg['f1_score']:.3f}")
    print(f"\n⏱️  Training Time: {results['training_time_minutes']:.1f} minutes")
    print(f"📁 Total Images: {results['total_images']:,}")
    print(f"🏷️  Number of Classes: {len(results['class_names'])}")
    print(f"\n📊 Per-Class Performance:")
    
    df = pd.DataFrame(class_data)
    df_sorted = df.sort_values('f1_score', ascending=False)
    for idx, row in df_sorted.iterrows():
        print(f"   {row['class']:20s} - F1: {row['f1_score']:.3f} (P: {row['precision']:.3f}, R: {row['recall']:.3f})")
    
    print(f"\n" + "="*60)
    print(f"✅ All visualizations saved to: {save_dir}/")
    print(f"   • mobilevit_accuracy_overview.png")
    print(f"   • mobilevit_per_class_metrics.png")
    print(f"   • mobilevit_metrics_heatmap.png")
    print(f"   • mobilevit_macro_weighted_comparison.png")
    print(f"   • mobilevit_summary_dashboard.png")
    print(f"="*60)


if __name__ == "__main__":
    main()
