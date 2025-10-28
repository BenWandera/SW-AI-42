"""
Division-based Waste Volume Visualization for Kampala
Creates comprehensive visual graphs for waste volume anticipation across the 5 divisions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class KampalaDivisionVisualizer:
    def __init__(self):
        """
        Initialize visualizer for Kampala's 5 divisions
        """
        self.divisions = ['Central', 'Kawempe', 'Rubaga', 'Makindye', 'Nakawa']
        self.division_colors = {
            'Central': '#FF6B6B',      # Red - highest waste
            'Kawempe': '#4ECDC4',      # Teal
            'Rubaga': '#45B7D1',       # Blue
            'Makindye': '#96CEB4',     # Green
            'Nakawa': '#FFEAA7'        # Yellow
        }
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_division_waste_graphs(self, location_data, volume_predictions, save_path='plots'):
        """
        Create comprehensive waste volume visualization for all 5 divisions
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Prepare division-based data
        division_data = self._prepare_division_data(location_data, volume_predictions)
        
        # Create main visualization dashboard
        self._create_division_dashboard(division_data, save_path)
        
        # Create individual division analysis
        self._create_individual_division_plots(division_data, save_path)
        
        # Create temporal projections
        self._create_temporal_projections(division_data, save_path)
        
        # Create comparative analysis
        self._create_comparative_analysis(division_data, save_path)
        
        print(f"Division waste volume visualizations saved to {save_path}/")
        return division_data
    
    def _prepare_division_data(self, location_data, volume_predictions):
        """
        Prepare division-aggregated data for visualization
        """
        # Combine location data with predictions
        df = location_data.copy()
        df['predicted_volume'] = volume_predictions.flatten()
        
        # Group by division
        division_stats = {}
        
        for division in self.divisions:
            div_data = df[df['division'] == division].copy()
            
            if len(div_data) > 0:
                division_stats[division] = {
                    'locations': div_data['location_id'].tolist(),
                    'total_volume': div_data['predicted_volume'].sum(),
                    'avg_volume': div_data['predicted_volume'].mean(),
                    'max_volume': div_data['predicted_volume'].max(),
                    'min_volume': div_data['predicted_volume'].min(),
                    'std_volume': div_data['predicted_volume'].std(),
                    'location_count': len(div_data),
                    'commercial_count': div_data['is_commercial'].sum(),
                    'residential_count': (div_data['is_commercial'] == 0).sum(),
                    'volume_per_location': div_data['predicted_volume'].values,
                    'population_density': div_data['population_density'].mean(),
                    'coordinates': div_data[['latitude', 'longitude']].values
                }
            else:
                # Handle case where division has no data
                division_stats[division] = {
                    'locations': [],
                    'total_volume': 0,
                    'avg_volume': 0,
                    'max_volume': 0,
                    'min_volume': 0,
                    'std_volume': 0,
                    'location_count': 0,
                    'commercial_count': 0,
                    'residential_count': 0,
                    'volume_per_location': np.array([]),
                    'population_density': 0,
                    'coordinates': np.array([]).reshape(0, 2)
                }
        
        return division_stats
    
    def _create_division_dashboard(self, division_data, save_path):
        """
        Create main dashboard with overview of all 5 divisions
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Kampala Waste Volume Anticipation Dashboard - 5 Divisions Analysis', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Total Volume by Division (Bar Chart)
        divisions = list(division_data.keys())
        total_volumes = [division_data[div]['total_volume'] for div in divisions]
        colors = [self.division_colors[div] for div in divisions]
        
        bars = axes[0, 0].bar(divisions, total_volumes, color=colors, alpha=0.8, edgecolor='black')
        axes[0, 0].set_title('Total Anticipated Waste Volume by Division', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Volume (kg/day)', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, volume in zip(bars, total_volumes):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{volume:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Average Volume per Location (Horizontal Bar Chart)
        avg_volumes = [division_data[div]['avg_volume'] for div in divisions]
        bars = axes[0, 1].barh(divisions, avg_volumes, color=colors, alpha=0.8, edgecolor='black')
        axes[0, 1].set_title('Average Volume per Location by Division', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Volume (kg/day/location)', fontsize=12)
        
        # Add value labels
        for bar, volume in zip(bars, avg_volumes):
            axes[0, 1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{volume:.2f}', ha='left', va='center', fontweight='bold')
        
        # 3. Location Count by Division (Pie Chart)
        location_counts = [division_data[div]['location_count'] for div in divisions]
        wedges, texts, autotexts = axes[0, 2].pie(location_counts, labels=divisions, colors=colors,
                                                 autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('Distribution of Monitoring Locations', fontsize=14, fontweight='bold')
        
        # 4. Volume Distribution (Box Plot)
        volume_data = []
        division_labels = []
        for div in divisions:
            if len(division_data[div]['volume_per_location']) > 0:
                volume_data.extend(division_data[div]['volume_per_location'])
                division_labels.extend([div] * len(division_data[div]['volume_per_location']))
        
        if volume_data:
            volume_df = pd.DataFrame({'Division': division_labels, 'Volume': volume_data})
            sns.boxplot(data=volume_df, x='Division', y='Volume', ax=axes[1, 0], palette=colors)
            axes[1, 0].set_title('Volume Distribution by Division', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Volume (kg/day)', fontsize=12)
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Commercial vs Residential Distribution
        commercial_counts = [division_data[div]['commercial_count'] for div in divisions]
        residential_counts = [division_data[div]['residential_count'] for div in divisions]
        
        x = np.arange(len(divisions))
        width = 0.35
        
        bars1 = axes[1, 1].bar(x - width/2, commercial_counts, width, label='Commercial', 
                              color='#FF9999', alpha=0.8, edgecolor='black')
        bars2 = axes[1, 1].bar(x + width/2, residential_counts, width, label='Residential', 
                              color='#66B2FF', alpha=0.8, edgecolor='black')
        
        axes[1, 1].set_title('Location Types by Division', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Locations', fontsize=12)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(divisions, rotation=45)
        axes[1, 1].legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + 0.05,
                               f'{int(height)}', ha='center', va='bottom')
        
        # 6. Priority Ranking
        priority_scores = []
        for div in divisions:
            # Calculate priority score based on total volume and location density
            total_vol = division_data[div]['total_volume']
            location_count = max(division_data[div]['location_count'], 1)
            score = total_vol + (total_vol / location_count) * 0.5
            priority_scores.append(score)
        
        # Sort divisions by priority
        div_priority = sorted(zip(divisions, priority_scores), key=lambda x: x[1], reverse=True)
        priority_divs, priority_vals = zip(*div_priority)
        priority_colors = [self.division_colors[div] for div in priority_divs]
        
        bars = axes[1, 2].barh(range(len(priority_divs)), priority_vals, color=priority_colors, 
                              alpha=0.8, edgecolor='black')
        axes[1, 2].set_title('Division Priority Ranking\n(Based on Volume & Density)', 
                           fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Priority Score', fontsize=12)
        axes[1, 2].set_yticks(range(len(priority_divs)))
        axes[1, 2].set_yticklabels([f"{i+1}. {div}" for i, div in enumerate(priority_divs)])
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, priority_vals)):
            axes[1, 2].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                           f'{score:.1f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'kampala_division_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_individual_division_plots(self, division_data, save_path):
        """
        Create detailed plots for each division
        """
        for division in self.divisions:
            data = division_data[division]
            
            if data['location_count'] == 0:
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{division} Division - Detailed Waste Volume Analysis', 
                        fontsize=16, fontweight='bold')
            
            # 1. Volume by Location
            locations = data['locations']
            volumes = data['volume_per_location']
            
            if len(volumes) > 0:
                bars = axes[0, 0].bar(range(len(locations)), volumes, 
                                     color=self.division_colors[division], alpha=0.7)
                axes[0, 0].set_title(f'Volume by Location in {division}', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel('Location Index')
                axes[0, 0].set_ylabel('Volume (kg/day)')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Highlight top 3 locations
                if len(volumes) >= 3:
                    top_indices = np.argsort(volumes)[-3:]
                    for idx in top_indices:
                        bars[idx].set_color('red')
                        bars[idx].set_alpha(0.9)
            
            # 2. Volume Distribution Histogram
            if len(volumes) > 1:
                axes[0, 1].hist(volumes, bins=min(10, len(volumes)), 
                              color=self.division_colors[division], alpha=0.7, edgecolor='black')
                axes[0, 1].axvline(data['avg_volume'], color='red', linestyle='--', 
                                 label=f'Mean: {data["avg_volume"]:.2f}')
                axes[0, 1].set_title(f'Volume Distribution in {division}', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('Volume (kg/day)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Statistics Summary
            axes[1, 0].axis('off')
            stats_text = f"""
{division} Division Statistics:

Total Locations: {data['location_count']}
Commercial: {data['commercial_count']}
Residential: {data['residential_count']}

Volume Metrics:
• Total: {data['total_volume']:.2f} kg/day
• Average: {data['avg_volume']:.2f} kg/day
• Maximum: {data['max_volume']:.2f} kg/day
• Minimum: {data['min_volume']:.2f} kg/day
• Std Dev: {data['std_volume']:.2f} kg/day

Population Density: {data['population_density']:.0f}
            """
            axes[1, 0].text(0.1, 0.9, stats_text, transform=axes[1, 0].transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=self.division_colors[division], alpha=0.3))
            
            # 4. Spatial Distribution (if coordinates available)
            if len(data['coordinates']) > 0:
                coords = data['coordinates']
                scatter = axes[1, 1].scatter(coords[:, 1], coords[:, 0], 
                                           c=volumes, s=100, alpha=0.7,
                                           cmap='Reds', edgecolors='black')
                axes[1, 1].set_title(f'Spatial Distribution in {division}', fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('Longitude')
                axes[1, 1].set_ylabel('Latitude')
                plt.colorbar(scatter, ax=axes[1, 1], label='Volume (kg/day)')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{division.lower()}_division_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_temporal_projections(self, division_data, save_path):
        """
        Create temporal projections for waste volume anticipation
        """
        # Create weekly and monthly projections
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Waste Volume Temporal Projections by Division', 
                     fontsize=16, fontweight='bold')
        
        # 1. Daily Volume Projection (7 days)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for division in self.divisions:
            if division_data[division]['location_count'] > 0:
                base_volume = division_data[division]['total_volume']
                # Simulate weekly variation (higher on weekends for residential, weekdays for commercial)
                daily_volumes = []
                for i in range(7):
                    # Weekend boost for residential areas, weekday boost for commercial
                    weekend_factor = 1.2 if i >= 5 else 0.9
                    weekday_factor = 1.1 if i < 5 else 0.8
                    # Mix based on commercial/residential ratio
                    commercial_ratio = division_data[division]['commercial_count'] / max(division_data[division]['location_count'], 1)
                    factor = weekend_factor * (1 - commercial_ratio) + weekday_factor * commercial_ratio
                    daily_volumes.append(base_volume * factor)
                
                axes[0, 0].plot(days, daily_volumes, marker='o', linewidth=2, 
                              label=division, color=self.division_colors[division])
        
        axes[0, 0].set_title('Weekly Volume Projection', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Volume (kg/day)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Monthly Volume Projection (12 months)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for division in self.divisions:
            if division_data[division]['location_count'] > 0:
                base_volume = division_data[division]['total_volume']
                # Simulate seasonal variation
                monthly_volumes = []
                for i in range(12):
                    # Seasonal factors (higher in dry season, lower in rainy season)
                    seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * i / 12)
                    monthly_volumes.append(base_volume * seasonal_factor)
                
                axes[0, 1].plot(months, monthly_volumes, marker='s', linewidth=2,
                              label=division, color=self.division_colors[division])
        
        axes[0, 1].set_title('Monthly Volume Projection', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Volume (kg/day)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Growth Projection (5 years)
        years = list(range(2025, 2030))
        
        for division in self.divisions:
            if division_data[division]['location_count'] > 0:
                base_volume = division_data[division]['total_volume']
                # Simulate population growth effect (3-5% annually)
                growth_rate = 0.04  # 4% annual growth
                yearly_volumes = []
                for i, year in enumerate(years):
                    projected_volume = base_volume * (1 + growth_rate) ** i
                    yearly_volumes.append(projected_volume)
                
                axes[1, 0].plot(years, yearly_volumes, marker='^', linewidth=2,
                              label=division, color=self.division_colors[division])
        
        axes[1, 0].set_title('5-Year Growth Projection', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Volume (kg/day)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Collection Efficiency Analysis
        divisions = list(division_data.keys())
        current_volumes = [division_data[div]['total_volume'] for div in divisions]
        
        # Simulate different collection efficiencies
        efficiencies = ['50%', '70%', '85%', '95%']
        efficiency_values = [0.5, 0.7, 0.85, 0.95]
        
        x = np.arange(len(divisions))
        width = 0.2
        
        for i, (eff, val) in enumerate(zip(efficiencies, efficiency_values)):
            collected_volumes = [vol * val for vol in current_volumes]
            bars = axes[1, 1].bar(x + i * width, collected_volumes, width, 
                                label=f'{eff} Efficiency', alpha=0.8)
        
        axes[1, 1].set_title('Collection Efficiency Scenarios', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Collected Volume (kg/day)')
        axes[1, 1].set_xticks(x + width * 1.5)
        axes[1, 1].set_xticklabels(divisions, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'temporal_projections.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comparative_analysis(self, division_data, save_path):
        """
        Create comparative analysis between divisions
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Kampala Divisions - Comparative Waste Analysis', 
                     fontsize=16, fontweight='bold')
        
        divisions = list(division_data.keys())
        
        # 1. Efficiency Metrics Radar Chart
        # Prepare metrics for radar chart
        metrics = ['Total Volume', 'Avg Volume', 'Location Count', 'Commercial Ratio', 'Pop Density']
        
        # Normalize metrics for radar chart
        total_vols = [division_data[div]['total_volume'] for div in divisions]
        avg_vols = [division_data[div]['avg_volume'] for div in divisions]
        loc_counts = [division_data[div]['location_count'] for div in divisions]
        comm_ratios = [division_data[div]['commercial_count'] / max(division_data[div]['location_count'], 1) for div in divisions]
        pop_densities = [division_data[div]['population_density'] for div in divisions]
        
        # Normalize to 0-1 scale
        max_total = max(total_vols) if max(total_vols) > 0 else 1
        max_avg = max(avg_vols) if max(avg_vols) > 0 else 1
        max_count = max(loc_counts) if max(loc_counts) > 0 else 1
        max_density = max(pop_densities) if max(pop_densities) > 0 else 1
        
        # Create comparison matrix
        comparison_data = []
        for i, div in enumerate(divisions):
            normalized_values = [
                total_vols[i] / max_total,
                avg_vols[i] / max_avg,
                loc_counts[i] / max_count,
                comm_ratios[i],  # Already 0-1
                pop_densities[i] / max_density
            ]
            comparison_data.append(normalized_values)
        
        # Heatmap
        im = axes[0, 0].imshow(comparison_data, cmap='RdYlBu_r', aspect='auto')
        axes[0, 0].set_xticks(range(len(metrics)))
        axes[0, 0].set_xticklabels(metrics, rotation=45, ha='right')
        axes[0, 0].set_yticks(range(len(divisions)))
        axes[0, 0].set_yticklabels(divisions)
        axes[0, 0].set_title('Division Performance Heatmap\n(Normalized Values)', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(divisions)):
            for j in range(len(metrics)):
                axes[0, 0].text(j, i, f'{comparison_data[i][j]:.2f}', 
                               ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
        
        # 2. Volume vs Efficiency Scatter
        total_volumes = [division_data[div]['total_volume'] for div in divisions]
        avg_volumes = [division_data[div]['avg_volume'] for div in divisions]
        colors = [self.division_colors[div] for div in divisions]
        
        scatter = axes[0, 1].scatter(total_volumes, avg_volumes, s=200, c=colors, alpha=0.7, edgecolors='black')
        axes[0, 1].set_xlabel('Total Volume (kg/day)')
        axes[0, 1].set_ylabel('Average Volume per Location (kg/day)')
        axes[0, 1].set_title('Volume vs Efficiency Analysis', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add division labels
        for i, div in enumerate(divisions):
            axes[0, 1].annotate(div, (total_volumes[i], avg_volumes[i]), 
                              xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        # 3. Resource Allocation Recommendation
        total_volume_sum = sum(total_volumes)
        if total_volume_sum > 0:
            percentages = [(vol / total_volume_sum) * 100 for vol in total_volumes]
        else:
            percentages = [20] * 5  # Equal distribution if no data
        
        wedges, texts, autotexts = axes[1, 0].pie(percentages, labels=divisions, 
                                                 colors=[self.division_colors[div] for div in divisions],
                                                 autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Recommended Resource Allocation\n(Based on Volume)', fontsize=12, fontweight='bold')
        
        # 4. Cost-Benefit Analysis
        # Simulate collection costs and benefits
        collection_costs = []
        environmental_benefits = []
        
        for div in divisions:
            # Cost based on volume and location count
            volume = division_data[div]['total_volume']
            locations = division_data[div]['location_count']
            
            # Estimated cost: $2 per kg + $50 per location per day
            cost = volume * 2 + locations * 50
            collection_costs.append(cost)
            
            # Environmental benefit score (arbitrary scale)
            benefit = volume * 1.5 + locations * 30
            environmental_benefits.append(benefit)
        
        x = np.arange(len(divisions))
        width = 0.35
        
        bars1 = axes[1, 1].bar(x - width/2, collection_costs, width, 
                              label='Collection Cost ($)', color='lightcoral', alpha=0.8)
        
        ax2 = axes[1, 1].twinx()
        bars2 = ax2.bar(x + width/2, environmental_benefits, width, 
                       label='Environmental Benefit', color='lightgreen', alpha=0.8)
        
        axes[1, 1].set_xlabel('Division')
        axes[1, 1].set_ylabel('Collection Cost ($)', color='red')
        ax2.set_ylabel('Environmental Benefit Score', color='green')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(divisions, rotation=45)
        axes[1, 1].set_title('Cost-Benefit Analysis by Division', fontsize=12, fontweight='bold')
        
        # Add legends
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'comparative_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_division_report(self, division_data, save_path='results'):
        """
        Generate comprehensive text report for division analysis
        """
        os.makedirs(save_path, exist_ok=True)
        
        report_path = os.path.join(save_path, 'kampala_divisions_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KAMPALA WASTE VOLUME ANTICIPATION - DIVISION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            total_volume = sum(division_data[div]['total_volume'] for div in self.divisions)
            total_locations = sum(division_data[div]['location_count'] for div in self.divisions)
            
            f.write(f"Total Anticipated Waste Volume: {total_volume:.2f} kg/day\n")
            f.write(f"Total Monitoring Locations: {total_locations}\n")
            f.write(f"Average Volume per Division: {total_volume/5:.2f} kg/day\n\n")
            
            # Ranking
            division_volumes = [(div, division_data[div]['total_volume']) for div in self.divisions]
            division_volumes.sort(key=lambda x: x[1], reverse=True)
            
            f.write("DIVISION RANKING BY WASTE VOLUME\n")
            f.write("-" * 40 + "\n")
            for i, (div, vol) in enumerate(division_volumes, 1):
                percentage = (vol / total_volume * 100) if total_volume > 0 else 0
                f.write(f"{i}. {div}: {vol:.2f} kg/day ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Detailed division analysis
            for division in self.divisions:
                data = division_data[division]
                
                f.write(f"{division.upper()} DIVISION ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Volume: {data['total_volume']:.2f} kg/day\n")
                f.write(f"Average per Location: {data['avg_volume']:.2f} kg/day\n")
                f.write(f"Location Count: {data['location_count']}\n")
                f.write(f"Commercial Locations: {data['commercial_count']}\n")
                f.write(f"Residential Locations: {data['residential_count']}\n")
                f.write(f"Population Density: {data['population_density']:.0f}\n")
                f.write(f"Volume Range: {data['min_volume']:.2f} - {data['max_volume']:.2f} kg/day\n")
                f.write(f"Standard Deviation: {data['std_volume']:.2f}\n\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS:\n")
                if data['total_volume'] > total_volume / 5 * 1.2:  # Above average
                    f.write("• Priority division - increase collection frequency\n")
                    f.write("• Deploy additional collection vehicles\n")
                    f.write("• Consider waste reduction programs\n")
                elif data['total_volume'] < total_volume / 5 * 0.8:  # Below average
                    f.write("• Standard collection schedule sufficient\n")
                    f.write("• Focus on efficiency optimization\n")
                    f.write("• Monitor for seasonal variations\n")
                else:
                    f.write("• Maintain current collection schedule\n")
                    f.write("• Regular monitoring recommended\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"Division analysis report saved to {report_path}")
        return report_path