"""
Prediction and Visualization for Waste GNN
Generate waste volume and type predictions for Kampala locations
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import folium
from folium import plugins
import json
import os


class WastePredictor:
    """
    Make predictions and visualize results for waste management
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.waste_types = [
            'plastic', 'paper', 'metal', 'glass', 'organic',
            'cardboard', 'textile', 'electronic', 'hazardous', 'other'
        ]
    
    @torch.no_grad()
    def predict(self, data):
        """
        Make predictions for all nodes in the graph
        
        Returns:
            volume_pred: Predicted waste volumes
            type_pred: Predicted waste type probabilities
        """
        self.model.eval()
        data = data.to(self.device)
        
        volume_pred, type_pred = self.model(data.x, data.edge_index)
        
        # Convert to probabilities
        type_pred = torch.sigmoid(type_pred)
        
        return volume_pred.cpu().numpy(), type_pred.cpu().numpy()
    
    def identify_high_waste_locations(self, volume_pred, location_data, top_n=20):
        """
        Identify locations with highest predicted waste volumes
        
        Returns:
            DataFrame with top locations
        """
        location_data = location_data.copy()
        location_data['predicted_volume'] = volume_pred.flatten()
        
        # Sort by predicted volume
        top_locations = location_data.nlargest(top_n, 'predicted_volume')
        
        return top_locations
    
    def analyze_waste_types_by_location(self, type_pred, location_data):
        """
        Analyze predominant waste types for each location
        """
        location_data = location_data.copy()
        
        # Add waste type predictions
        for i, waste_type in enumerate(self.waste_types):
            location_data[f'{waste_type}_prob'] = type_pred[:, i]
        
        # Find dominant waste type for each location
        location_data['dominant_waste_type'] = [
            self.waste_types[np.argmax(type_pred[i])]
            for i in range(len(type_pred))
        ]
        
        location_data['dominant_waste_prob'] = [
            np.max(type_pred[i])
            for i in range(len(type_pred))
        ]
        
        return location_data
    
    def create_kampala_waste_map(self, location_data, volume_pred, type_pred, save_path='maps'):
        """
        Create interactive map of waste predictions in Kampala
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Convert predictions to native Python types for JSON serialization
        if hasattr(volume_pred, 'detach'):
            volume_pred = volume_pred.detach().cpu().numpy()
        if hasattr(type_pred, 'detach'):
            type_pred = type_pred.detach().cpu().numpy()
        
        # Ensure native Python floats
        volume_pred = volume_pred.astype(float)
        type_pred = type_pred.astype(float)
        
        # Center on Kampala
        kampala_center = [0.3476, 32.5825]
        
        # Create base map
        m = folium.Map(
            location=kampala_center,
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add waste type predictions
        location_data = location_data.copy()
        location_data['predicted_volume'] = volume_pred.flatten()
        
        # Color scheme for waste volumes
        for idx, row in location_data.iterrows():
            # Volume-based marker size and color
            volume = float(row['predicted_volume'])
            volume_normalized = min(volume / float(location_data['predicted_volume'].max()), 1.0)
            
            # Determine dominant waste type
            dominant_idx = np.argmax(type_pred[idx])
            dominant_type = self.waste_types[dominant_idx]
            dominant_prob = float(type_pred[idx, dominant_idx])
            
            # Color based on volume intensity
            color = self._get_color_for_volume(volume_normalized)
            
            # Marker size based on volume
            radius = 5 + volume_normalized * 15
            
            # Create popup with information
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4>Location {row['location_id']}</h4>
                <b>Predicted Volume:</b> {volume:.1f} kg/day<br>
                <b>Dominant Waste Type:</b> {dominant_type.capitalize()} ({dominant_prob*100:.1f}%)<br>
                <b>Type:</b> {'Commercial' if row['is_commercial'] else 'Residential'}<br>
                <b>Population Density:</b> {float(row['population_density']):.0f}<br>
                <hr>
                <b>Waste Type Probabilities:</b><br>
                {'<br>'.join([f'{wt.capitalize()}: {float(type_pred[idx, i])*100:.1f}%' 
                             for i, wt in enumerate(self.waste_types) if float(type_pred[idx, i]) > 0.1])}
            </div>
            """
            
            folium.CircleMarker(
                location=[float(row['latitude']), float(row['longitude'])],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; 
                    background-color: white; z-index:9999; 
                    border:2px solid grey; padding: 10px;
                    font-family: Arial;">
            <h4 style="margin-top:0;">Waste Volume Legend</h4>
            <p><span style="color: green;">●</span> Low Volume</p>
            <p><span style="color: yellow;">●</span> Medium Volume</p>
            <p><span style="color: orange;">●</span> High Volume</p>
            <p><span style="color: red;">●</span> Very High Volume</p>
            <hr>
            <p style="font-size: 10px;">Marker size indicates relative volume</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add heatmap layer
        heat_data = [[float(row['latitude']), float(row['longitude']), float(row['predicted_volume'])] 
                     for idx, row in location_data.iterrows()]
        
        plugins.HeatMap(heat_data, name='Waste Volume Heatmap', 
                       radius=15, blur=20, max_zoom=13).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        map_file = os.path.join(save_path, 'kampala_waste_map.html')
        m.save(map_file)
        print(f"Interactive map saved to {map_file}")
        
        return m
    
    def _get_color_for_volume(self, volume_normalized):
        """
        Get color based on volume intensity
        """
        if volume_normalized < 0.25:
            return 'green'
        elif volume_normalized < 0.5:
            return 'yellow'
        elif volume_normalized < 0.75:
            return 'orange'
        else:
            return 'red'
    
    def plot_waste_distribution(self, volume_pred, type_pred, location_data, save_path='plots'):
        """
        Create visualization plots for waste distribution
        """
        os.makedirs(save_path, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Volume distribution histogram
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(volume_pred, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Predicted Volume (kg/day)', fontsize=12)
        plt.ylabel('Number of Locations', fontsize=12)
        plt.title('Distribution of Predicted Waste Volumes', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 2. Volume by location type
        ax2 = plt.subplot(2, 3, 2)
        commercial_vol = volume_pred[location_data['is_commercial'].values].flatten()
        residential_vol = volume_pred[location_data['is_residential'].values].flatten()
        
        plt.boxplot([commercial_vol, residential_vol], labels=['Commercial', 'Residential'])
        plt.ylabel('Predicted Volume (kg/day)', fontsize=12)
        plt.title('Waste Volume by Location Type', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 3. Waste type frequencies
        ax3 = plt.subplot(2, 3, 3)
        type_avg = np.mean(type_pred, axis=0)
        plt.barh(self.waste_types, type_avg, color='steelblue')
        plt.xlabel('Average Probability', fontsize=12)
        plt.title('Average Waste Type Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # 4. Spatial distribution scatter
        ax4 = plt.subplot(2, 3, 4)
        scatter = plt.scatter(location_data['longitude'], location_data['latitude'],
                            c=volume_pred.flatten(), s=100, cmap='YlOrRd',
                            alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Predicted Volume (kg/day)')
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title('Spatial Distribution of Waste Volumes', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 5. Waste type heatmap
        ax5 = plt.subplot(2, 3, 5)
        # Show top 20 locations
        top_indices = np.argsort(volume_pred.flatten())[-20:]
        type_subset = type_pred[top_indices]
        
        sns.heatmap(type_subset.T, cmap='YlOrRd', cbar_kws={'label': 'Probability'},
                   yticklabels=self.waste_types, xticklabels=False)
        plt.xlabel('Top 20 Locations', fontsize=12)
        plt.ylabel('Waste Type', fontsize=12)
        plt.title('Waste Type Distribution (Top 20 Locations)', fontsize=14, fontweight='bold')
        
        # 6. Volume vs Population Density
        ax6 = plt.subplot(2, 3, 6)
        plt.scatter(location_data['population_density'], volume_pred.flatten(),
                   alpha=0.5, edgecolors='black', linewidth=0.5)
        plt.xlabel('Population Density', fontsize=12)
        plt.ylabel('Predicted Volume (kg/day)', fontsize=12)
        plt.title('Volume vs Population Density', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(location_data['population_density'], volume_pred.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(location_data['population_density'], 
                p(location_data['population_density']), 
                "r--", alpha=0.8, linewidth=2, label='Trend')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'waste_distribution_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution plots saved to {save_path}/waste_distribution_analysis.png")
    
    def generate_recommendations(self, volume_pred, type_pred, location_data, save_path='results'):
        """
        Generate actionable recommendations for waste management
        """
        os.makedirs(save_path, exist_ok=True)
        
        location_data = location_data.copy()
        location_data['predicted_volume'] = volume_pred.flatten()
        
        # Identify high-priority locations
        top_20 = location_data.nlargest(20, 'predicted_volume')
        
        recommendations = {
            'high_priority_locations': [],
            'waste_type_insights': {},
            'collection_optimization': {}
        }
        
        # High priority locations
        for idx, row in top_20.iterrows():
            dominant_type_idx = np.argmax(type_pred[idx])
            dominant_type = self.waste_types[dominant_type_idx]
            
            recommendations['high_priority_locations'].append({
                'location_id': int(row['location_id']),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'predicted_volume': float(row['predicted_volume']),
                'location_type': 'Commercial' if row['is_commercial'] else 'Residential',
                'dominant_waste_type': dominant_type,
                'recommendation': self._get_recommendation(row['predicted_volume'], dominant_type)
            })
        
        # Waste type insights
        for i, waste_type in enumerate(self.waste_types):
            type_prevalence = np.mean(type_pred[:, i])
            recommendations['waste_type_insights'][waste_type] = {
                'average_probability': float(type_prevalence),
                'high_prevalence_locations': int(np.sum(type_pred[:, i] > 0.5)),
                'collection_strategy': self._get_collection_strategy(waste_type, type_prevalence)
            }
        
        # Collection optimization
        total_volume = np.sum(volume_pred)
        recommendations['collection_optimization'] = {
            'total_daily_volume_kg': float(total_volume),
            'average_volume_per_location': float(np.mean(volume_pred)),
            'recommended_collection_frequency': self._get_collection_frequency(volume_pred),
            'estimated_trucks_needed': int(np.ceil(total_volume / 5000))  # Assuming 5000kg per truck
        }
        
        # Save recommendations
        with open(os.path.join(save_path, 'recommendations.json'), 'w') as f:
            json.dump(recommendations, f, indent=4)
        
        # Create summary report
        self._create_summary_report(recommendations, save_path)
        
        print(f"Recommendations saved to {save_path}/recommendations.json")
        
        return recommendations
    
    def _get_recommendation(self, volume, waste_type):
        """
        Generate specific recommendation based on volume and type
        """
        if volume > 400:
            freq = "daily"
        elif volume > 200:
            freq = "every 2 days"
        else:
            freq = "twice weekly"
        
        return f"Implement {freq} collection with focus on {waste_type} waste management"
    
    def _get_collection_strategy(self, waste_type, prevalence):
        """
        Get collection strategy for specific waste type
        """
        strategies = {
            'plastic': 'Implement recycling bins and partner with plastic recyclers',
            'paper': 'Set up paper collection points for recycling',
            'organic': 'Consider composting programs and organic waste processing',
            'metal': 'Establish metal scrap collection centers',
            'glass': 'Set up safe glass collection and recycling stations',
            'cardboard': 'Partner with cardboard recyclers for regular pickup',
            'textile': 'Organize textile donation and recycling drives',
            'electronic': 'Create e-waste collection centers with proper disposal',
            'hazardous': 'Implement specialized hazardous waste handling protocols',
            'other': 'Assess and categorize for appropriate disposal methods'
        }
        
        return strategies.get(waste_type, 'Standard collection procedures')
    
    def _get_collection_frequency(self, volume_pred):
        """
        Recommend collection frequency based on volumes
        """
        avg_volume = np.mean(volume_pred)
        
        if avg_volume > 300:
            return "Daily collection for all high-volume areas"
        elif avg_volume > 150:
            return "Daily for high-volume, every 2 days for medium-volume areas"
        else:
            return "Every 2-3 days with flexible scheduling"
    
    def _create_summary_report(self, recommendations, save_path):
        """
        Create a text summary report
        """
        report_file = os.path.join(save_path, 'summary_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KAMPALA WASTE MANAGEMENT - PREDICTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            opt = recommendations['collection_optimization']
            f.write(f"Total Daily Waste Volume: {opt['total_daily_volume_kg']:.1f} kg\n")
            f.write(f"Average Volume per Location: {opt['average_volume_per_location']:.1f} kg\n")
            f.write(f"Estimated Trucks Needed: {opt['estimated_trucks_needed']}\n")
            f.write(f"Recommended Collection Frequency: {opt['recommended_collection_frequency']}\n\n")
            
            f.write("TOP 10 HIGH-PRIORITY LOCATIONS\n")
            f.write("-" * 80 + "\n")
            for i, loc in enumerate(recommendations['high_priority_locations'][:10], 1):
                f.write(f"{i}. Location {loc['location_id']} ({loc['location_type']})\n")
                f.write(f"   Volume: {loc['predicted_volume']:.1f} kg/day\n")
                f.write(f"   Primary Waste: {loc['dominant_waste_type']}\n")
                f.write(f"   Action: {loc['recommendation']}\n\n")
            
            f.write("\nWASTE TYPE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for waste_type, info in recommendations['waste_type_insights'].items():
                f.write(f"{waste_type.upper()}:\n")
                f.write(f"  Prevalence: {info['average_probability']*100:.1f}%\n")
                f.write(f"  Strategy: {info['collection_strategy']}\n\n")
        
        print(f"Summary report saved to {report_file}")