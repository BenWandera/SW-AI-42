"""
Data preprocessing for Waste GNN
Handles Kiteezi dataset and generates synthetic waste classification data
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import json
import os
import requests
from pathlib import Path


class WasteDataProcessor:
    """
    Process waste datasets and create graph structures
    """
    
    def __init__(self, kiteezi_path=None):
        """
        Args:
            kiteezi_path: Path to Kiteezi volume dataset
        """
        self.kiteezi_path = kiteezi_path
        
        self.scaler = StandardScaler()
        
        self.waste_types = [
            'plastic', 'paper', 'metal', 'glass', 'organic',
            'cardboard', 'textile', 'electronic', 'hazardous', 'other'
        ]
        
        # Kampala's 5 divisions with realistic characteristics
        self.kampala_divisions = {
            'Central': {
                'center': (0.3163, 32.5822),
                'radius': 0.02,
                'commercial_ratio': 0.7,
                'population_density': 8500,
                'waste_multiplier': 1.5,
                'income_level': 'mixed'
            },
            'Kawempe': {
                'center': (0.3776, 32.5681),
                'radius': 0.04,
                'commercial_ratio': 0.3,
                'population_density': 6200,
                'waste_multiplier': 1.2,
                'income_level': 'low-middle'
            },
            'Makindye': {
                'center': (0.2847, 32.5947),
                'radius': 0.035,
                'commercial_ratio': 0.4,
                'population_density': 5800,
                'waste_multiplier': 1.1,
                'income_level': 'middle'
            },
            'Nakawa': {
                'center': (0.3363, 32.6178),
                'radius': 0.03,
                'commercial_ratio': 0.5,
                'population_density': 7000,
                'waste_multiplier': 1.3,
                'income_level': 'middle-high'
            },
            'Rubaga': {
                'center': (0.3001, 32.5503),
                'radius': 0.03,
                'commercial_ratio': 0.45,
                'population_density': 6500,
                'waste_multiplier': 1.15,
                'income_level': 'middle'
            }
        }
        
    
    def load_kiteezi_data(self):
        """
        Load Kiteezi waste volume dataset from Western Sydney University research data
        
        Returns:
            DataFrame with volume measurements
        """
        print("Loading Kiteezi dataset...")
        
        # Try to load local file first
        if self.kiteezi_path and os.path.exists(self.kiteezi_path):
            try:
                if self.kiteezi_path.endswith('.csv'):
                    df = pd.read_csv(self.kiteezi_path)
                elif self.kiteezi_path.endswith('.xlsx'):
                    df = pd.read_excel(self.kiteezi_path)
                else:
                    df = pd.read_csv(self.kiteezi_path)
                print(f"Loaded local Kiteezi data: {len(df)} records")
                return self._process_real_kiteezi_data(df)
            except Exception as e:
                print(f"Error loading local Kiteezi data: {e}")
        
        # Check for manually downloaded processed data
        processed_file = "processed_kiteezi_data.csv"
        if os.path.exists(processed_file):
            try:
                df = pd.read_csv(processed_file)
                print(f"Loaded processed Kiteezi data: {len(df)} records")
                return self._process_real_kiteezi_data(df)
            except Exception as e:
                print(f"Error loading processed Kiteezi data: {e}")
        
        # First, try to load the real KCCA dataset from kiteezi_data folder
        real_data_path = os.path.join(os.path.dirname(__file__), 'kiteezi_data', 'Kiteezi_waste_quantities_Apr2011Sep2017.csv')
        
        if os.path.exists(real_data_path):
            print(f"Found real KCCA Kiteezi dataset: {real_data_path}")
            return self._load_real_kiteezi_data(real_data_path)
        
        # Try to download the real Kiteezi dataset from Western Sydney University
        print("Attempting to download real Kiteezi landfill data from Western Sydney University...")
        try:
            # These URLs might require authentication or be different
            # This is a fallback attempt
            csv_urls = [
                "https://research-data.westernsydney.edu.au/published/ff921990519311ecb15399911543e199/Kiteezi_waste_quantities_Apr2011Sep2017.csv",
                "https://research-data.westernsydney.edu.au/published/ff921990519311ecb15399911543e199/Kiteezi_waste_quantities_Oct2017Dec2020.csv"
            ]
            
            dfs = []
            for url in csv_urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        # Save temporarily and read
                        temp_file = f"temp_kiteezi_{len(dfs)}.csv"
                        with open(temp_file, 'wb') as f:
                            f.write(response.content)
                        df_temp = pd.read_csv(temp_file)
                        dfs.append(df_temp)
                        os.remove(temp_file)  # Clean up
                        print(f"Downloaded {len(df_temp)} records from {url.split('/')[-1]}")
                    else:
                        print(f"Failed to download {url}, status code: {response.status_code}")
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
            
            if dfs:
                # Combine the datasets
                combined_df = pd.concat(dfs, ignore_index=True)
                print(f"Combined real Kiteezi data: {len(combined_df)} total records")
                
                # Process the real data into the expected format
                processed_df = self._process_real_kiteezi_data(combined_df)
                return processed_df
                
        except Exception as e:
            print(f"Error downloading real Kiteezi data: {e}")
        
        # Fallback to synthetic data
        print("\n" + "="*60)
        print("REAL KITEEZI DATA NOT AVAILABLE")
        print("="*60)
        print("To use the real Kiteezi landfill dataset:")
        print("1. Visit: https://research-data.westernsydney.edu.au/published/ff921990519311ecb15399911543e199")
        print("2. Download the CSV/Excel files manually")
        print("3. Place them in a 'kiteezi_data' folder")
        print("4. Run download_kiteezi_data.py to process them")
        print("5. Re-run the GNN model")
        print("\nUsing synthetic data as fallback for now...")
        print("="*60)
        return self._generate_synthetic_kiteezi_data()
    
    def _process_real_kiteezi_data(self, raw_df):
        """
        Process the real Kiteezi dataset into the format expected by the GNN model
        
        Args:
            raw_df: Raw dataframe from Western Sydney University dataset
            
        Returns:
            Processed dataframe with location_id, coordinates, volume_kg, etc.
        """
        print("Processing real Kiteezi data for GNN model...")
        
        # The real dataset likely has columns like: Date, Division, Waste_Quantity, etc.
        # We need to map this to our expected format with synthetic location coordinates
        
        # First, let's examine the structure
        print("Raw data columns:", raw_df.columns.tolist())
        print("Raw data shape:", raw_df.shape)
        if len(raw_df) > 0:
            print("Sample data:")
            print(raw_df.head())
        
        processed_data = []
        location_id = 0
        
        # Map divisions to our coordinate system
        division_coords = {
            'Central': {'lat_center': 0.3163, 'lon_center': 32.5822},
            'Kawempe': {'lat_center': 0.3774, 'lon_center': 32.5747}, 
            'Makindye': {'lat_center': 0.2682, 'lon_center': 32.6140},
            'Nakawa': {'lat_center': 0.3319, 'lon_center': 32.6165},
            'Rubaga': {'lat_center': 0.3064, 'lon_center': 32.5378}
        }
        
        # Group by division and create synthetic locations within each division
        if 'Division' in raw_df.columns or 'division' in raw_df.columns:
            div_col = 'Division' if 'Division' in raw_df.columns else 'division'
            
            for division in raw_df[div_col].unique():
                if pd.isna(division) or division not in division_coords:
                    continue
                    
                division_data = raw_df[raw_df[div_col] == division]
                coords = division_coords[division]
                
                # Create 20 synthetic locations per division
                for i in range(20):
                    # Add some randomness around the division center
                    lat = coords['lat_center'] + np.random.normal(0, 0.01)
                    lon = coords['lon_center'] + np.random.normal(0, 0.01)
                    
                    # Calculate average volume for this division
                    if 'Waste_Quantity' in division_data.columns:
                        avg_volume = division_data['Waste_Quantity'].mean()
                    elif 'waste_quantity' in division_data.columns:
                        avg_volume = division_data['waste_quantity'].mean()
                    else:
                        # Try to find any numeric column that might be volume
                        numeric_cols = division_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            avg_volume = division_data[numeric_cols[0]].mean()
                        else:
                            avg_volume = 300  # Fallback
                    
                    # Create daily records for this location
                    for day in range(365):
                        # Add some daily variation
                        daily_volume = max(0, avg_volume + np.random.normal(0, avg_volume * 0.2))
                        
                        processed_data.append({
                            'location_id': location_id,
                            'division': division,
                            'latitude': lat,
                            'longitude': lon,
                            'day': day,
                            'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day),
                            'volume_kg': daily_volume,
                            'is_commercial': np.random.choice([0, 1], p=[0.7, 0.3]),
                            'is_residential': np.random.choice([0, 1], p=[0.3, 0.7]),
                            'population_density': self.kampala_divisions[division]['population_density'],
                            'distance_to_market': np.random.uniform(0.5, 5.0),
                            'income_level': self.kampala_divisions[division]['income_level']
                        })
                    
                    location_id += 1
        else:
            print("Division column not found, using synthetic data structure")
            return self._generate_synthetic_kiteezi_data()
        
        df = pd.DataFrame(processed_data)
        print(f"Processed real Kiteezi data: {len(df)} records for {location_id} locations")
        print("Division distribution from real data:")
        for div in df['division'].unique():
            count = df[df['division'] == div]['location_id'].nunique()
            avg_vol = df[df['division'] == div]['volume_kg'].mean()
            print(f"  {div}: {count} locations, avg {avg_vol:.1f} kg/day")
        
        return df
    
    def _load_real_kiteezi_data(self, file_path):
        """
        Load and process the real KCCA monthly tonnage data
        """
        print("Processing real KCCA waste tonnage data (Apr 2011 - Sep 2017)...")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # The data structure: 
        # Row 0: Headers (months from Apr-11 to Sep-17)
        # Rows 1-5: Division data (Central, Kawempe, Lubaga, Makindye, Nakawa)
        # Row 6: KCCA Grand Total
        
        # Get division names (excluding the total row)
        divisions = df.iloc[0:5, 0].values  # Only Central, Kawempe, Lubaga, Makindye, Nakawa (exclude KCCA Grand Total)
        
        print(f"Processing divisions: {divisions}")
        
        # Map CSV division names to our kampala_divisions keys
        division_name_mapping = {
            'Central': 'Central',
            'Kawempe': 'Kawempe', 
            'Lubaga': 'Rubaga',  # CSV uses "Lubaga" but our dict uses "Rubaga"
            'Makindye': 'Makindye',
            'Nakawa': 'Nakawa'
        }
        
        # Get month columns (all except first column which contains division names)
        month_columns = df.columns[1:]
        
        # Process the data into long format
        processed_data = []
        location_id = 0
        
        for div_idx, division in enumerate(divisions):
            division_name_csv = division.strip()
            division_name = division_name_mapping.get(division_name_csv, division_name_csv)
            print(f"Processing {division_name_csv} -> {division_name} division...")
            
            # Get tonnage data for this division
            tonnage_data = df.iloc[div_idx, 1:].values  # Skip division name column
            
            # Create synthetic locations within each division
            n_locations_per_division = 20  # 20 locations per division = 100 total
            
            for loc in range(n_locations_per_division):
                # Generate realistic coordinates for each division
                lat_base, lon_base = self.kampala_divisions[division_name]['center']
                radius = self.kampala_divisions[division_name]['radius']
                
                latitude = np.random.uniform(lat_base - radius, lat_base + radius)
                longitude = np.random.uniform(lon_base - radius, lon_base + radius)
                
                # Process each month of data
                for month_idx, month_col in enumerate(month_columns):
                    try:
                        total_monthly_tonnage = float(tonnage_data[month_idx])
                        
                        # Distribute monthly tonnage across locations in this division
                        # Add some variation between locations
                        location_factor = np.random.uniform(0.5, 1.5)  # Variation factor
                        daily_tonnage = (total_monthly_tonnage / n_locations_per_division) * location_factor / 30  # Approximate daily from monthly
                        daily_kg = daily_tonnage * 1000  # Convert tonnes to kg
                        
                        # Parse month-year from column name (e.g., "Apr-11")
                        month_year = month_col.strip()
                        month_str, year_str = month_year.split('-')
                        year = 2000 + int(year_str)  # Convert "11" to 2011
                        month_num = {
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }[month_str]
                        
                        # Create multiple days for this month
                        import calendar
                        days_in_month = calendar.monthrange(year, month_num)[1]  # Get actual days in month
                        for day in range(1, days_in_month + 1):
                            day_variation = np.random.uniform(0.8, 1.2)  # Daily variation
                            final_volume = daily_kg * day_variation
                            
                            processed_data.append({
                                'location_id': location_id,
                                'division': division_name,  # Use mapped name
                                'latitude': latitude,
                                'longitude': longitude,
                                'year': year,
                                'month': month_num,
                                'day': day,
                                'date': pd.Timestamp(year, month_num, day),
                                'volume_kg': max(0, final_volume),  # Ensure non-negative
                                'is_commercial': np.random.choice([True, False], p=[0.3, 0.7]),
                                'is_residential': np.random.choice([True, False], p=[0.7, 0.3]),
                                'population_density': self.kampala_divisions[division_name]['population_density'],
                                'distance_to_market': np.random.uniform(0.5, 5.0),
                                'income_level': self.kampala_divisions[division_name]['income_level']
                            })
                    
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not process {month_col} data for {division_name}: {e}")
                        continue
                
                location_id += 1
        
        result_df = pd.DataFrame(processed_data)
        
        print(f"Processed real KCCA data:")
        print(f"  - {len(result_df)} total records")
        print(f"  - {result_df['location_id'].nunique()} unique locations")
        print(f"  - Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        print(f"  - Average daily volume: {result_df['volume_kg'].mean():.1f} kg/day")
        print(f"  - Division distribution:")
        for div in divisions:
            div_name_csv = div.strip()
            div_name = division_name_mapping.get(div_name_csv, div_name_csv)
            div_data = result_df[result_df['division'] == div_name]
            avg_vol = div_data['volume_kg'].mean()
            print(f"    {div_name_csv} -> {div_name}: {div_data['location_id'].nunique()} locations, avg {avg_vol:.1f} kg/day")
        
        return result_df

    def _generate_synthetic_kiteezi_data(self, n_locations=100, n_days=365):
        """
        Generate synthetic volume data for Kampala locations with realistic patterns
        Based on division characteristics and real-world factors
        """
        print("Generating synthetic Kiteezi volume data with division-based patterns...")
        
        data = []
        location_id = 0
        
        # Generate locations for each division
        for division_name, division_info in self.kampala_divisions.items():
            # Number of locations proportional to area and density
            n_div_locations = int(n_locations * 0.2)  # 20% per division
            
            center_lat, center_lon = division_info['center']
            radius = division_info['radius']
            
            for _ in range(n_div_locations):
                # Random location within division radius
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, radius)
                lat = center_lat + r * np.cos(angle)
                lon = center_lon + r * np.sin(angle)
                
                # Location characteristics based on division
                is_commercial = np.random.rand() < division_info['commercial_ratio']
                is_residential = not is_commercial
                
                # Population density with variation
                base_density = division_info['population_density']
                population_density = base_density * np.random.uniform(0.7, 1.3)
                
                # Income level affects waste generation
                income_multiplier = {
                    'low': 0.7,
                    'low-middle': 0.85,
                    'middle': 1.0,
                    'middle-high': 1.15,
                    'high': 1.3,
                    'mixed': 1.0
                }.get(division_info['income_level'], 1.0)
                
                # Market proximity (affects commercial waste)
                distance_to_market = np.random.uniform(0.1, 2.0)  # km
                market_effect = 1.2 if distance_to_market < 0.5 else 1.0
                
                for day in range(n_days):
                    # Base volume calculation
                    if is_commercial:
                        base_volume = np.random.uniform(150, 600) * division_info['waste_multiplier']
                    else:
                        base_volume = np.random.uniform(40, 200) * division_info['waste_multiplier']
                    
                    # Apply income multiplier
                    base_volume *= income_multiplier
                    
                    # Weekly patterns
                    day_of_week = day % 7
                    if is_residential:
                        # More waste on weekends in residential
                        if day_of_week in [5, 6]:  # Saturday, Sunday
                            weekly_multiplier = 1.3
                        else:
                            weekly_multiplier = 1.0
                    else:
                        # More waste on weekdays in commercial
                        if day_of_week in [0, 1, 2, 3, 4]:  # Monday-Friday
                            weekly_multiplier = 1.2 * market_effect
                        else:
                            weekly_multiplier = 0.7
                    
                    # Monthly patterns (end of month more waste)
                    day_of_month = (day % 30) + 1
                    if day_of_month > 25:
                        monthly_multiplier = 1.15
                    else:
                        monthly_multiplier = 1.0
                    
                    # Seasonal patterns (rainy season affects organic waste)
                    # Assuming March-May and Sept-Nov are rainy seasons
                    month = (day // 30) % 12
                    if month in [2, 3, 4, 8, 9, 10]:  # Rainy months (0-indexed)
                        seasonal_multiplier = 1.1
                    else:
                        seasonal_multiplier = 1.0
                    
                    # Holiday effects (more waste during holidays)
                    # Simplified: assume some holidays
                    is_holiday = (day % 90) in [0, 1]  # Quarterly holidays
                    holiday_multiplier = 1.25 if is_holiday else 1.0
                    
                    # Calculate final volume with all factors
                    volume = (base_volume * 
                             weekly_multiplier * 
                             monthly_multiplier * 
                             seasonal_multiplier * 
                             holiday_multiplier * 
                             np.random.uniform(0.85, 1.15))  # Random variation
                    
                    data.append({
                        'location_id': location_id,
                        'division': division_name,
                        'latitude': lat,
                        'longitude': lon,
                        'day': day,
                        'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day),
                        'volume_kg': volume,
                        'is_commercial': is_commercial,
                        'is_residential': is_residential,
                        'population_density': population_density,
                        'distance_to_market': distance_to_market,
                        'income_level': division_info['income_level']
                    })
                
                location_id += 1
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} records for {location_id} locations across 5 divisions")
        print(f"Division distribution:")
        for div in self.kampala_divisions.keys():
            count = df[df['division'] == div]['location_id'].nunique()
            avg_vol = df[df['division'] == div]['volume_kg'].mean()
            print(f"  {div}: {count} locations, avg {avg_vol:.1f} kg/day")
        
        return df
    
    def create_kampala_graph_data(self, include_temporal=False):
        """
        Create graph data structure for Kampala waste prediction
        
        Args:
            include_temporal: Whether to include temporal features
            
        Returns:
            Data object for PyTorch Geometric and location information
        """
        # Load Kiteezi dataset only
        kiteezi_df = self.load_kiteezi_data()
        
        # Get unique locations with division information
        agg_dict = {
            'latitude': 'first',
            'longitude': 'first',
            'is_commercial': 'first',
            'is_residential': 'first',
            'population_density': 'first',
            'volume_kg': ['mean', 'std', 'max']
        }
        
        # Add division and other columns if they exist
        if 'division' in kiteezi_df.columns:
            agg_dict['division'] = 'first'
        if 'distance_to_market' in kiteezi_df.columns:
            agg_dict['distance_to_market'] = 'first'
        if 'income_level' in kiteezi_df.columns:
            agg_dict['income_level'] = 'first'
        
        location_data = kiteezi_df.groupby('location_id').agg(agg_dict).reset_index()
        
        # Flatten multi-level columns dynamically
        new_columns = ['location_id']
        for col in location_data.columns[1:]:  # Skip location_id
            if isinstance(col, tuple):
                if col[1] == '':  # Single level column
                    new_columns.append(col[0])
                else:  # Multi-level column (aggregated)
                    if col[1] in ['mean', 'std', 'max']:
                        new_columns.append(f"{col[0]}_{col[1]}")
                    else:
                        new_columns.append(col[0])
            else:
                new_columns.append(col)
        
        location_data.columns = new_columns
        
        # Add back division and other info
        if 'division' in kiteezi_df.columns:
            division_map = kiteezi_df.groupby('location_id')['division'].first()
            location_data['division'] = location_data['location_id'].map(division_map)
        
        if 'distance_to_market' in kiteezi_df.columns:
            dist_map = kiteezi_df.groupby('location_id')['distance_to_market'].first()
            location_data['distance_to_market'] = location_data['location_id'].map(dist_map)
        
        if 'income_level' in kiteezi_df.columns:
            income_map = kiteezi_df.groupby('location_id')['income_level'].first()
            location_data['income_level'] = location_data['location_id'].map(income_map)
        
        # Create node features
        node_features = self._create_node_features(location_data)
        
        # Create spatial graph edges
        locations = location_data[['latitude', 'longitude']].values
        edge_index = self._create_spatial_edges(locations, k_neighbors=8)  # Increased connectivity
        
        # Create targets
        volume_targets = location_data['volume_kg_mean'].values
        type_targets = self._create_type_targets(location_data)
        
        # Normalize features
        node_features_normalized = self.scaler.fit_transform(node_features)
        
        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(node_features_normalized),
            edge_index=torch.LongTensor(edge_index),
            y_volume=torch.FloatTensor(volume_targets),
            y_type=torch.FloatTensor(type_targets)
        )
        
        return data, location_data
    
    def _create_node_features(self, location_data):
        """
        Create feature matrix for location nodes with division information
        """
        features = []
        
        # Encode divisions
        division_map = {name: idx for idx, name in enumerate(self.kampala_divisions.keys())}
        
        for idx, row in location_data.iterrows():
            # Get division encoding (one-hot)
            division_encoded = [0] * len(self.kampala_divisions)
            if 'division' in row:
                div_idx = division_map.get(row['division'], 0)
                division_encoded[div_idx] = 1
            
            # Income level encoding
            income_encoding = {
                'low': 0.2,
                'low-middle': 0.4,
                'middle': 0.6,
                'middle-high': 0.8,
                'high': 1.0,
                'mixed': 0.5
            }.get(row.get('income_level', 'middle'), 0.6)
            
            feat = [
                row['latitude'],
                row['longitude'],
                row['is_commercial'],
                row['is_residential'],
                row['population_density'] / 10000,  # Normalize
                row['volume_kg_mean'] / 1000,  # Normalize to tons
                row['volume_kg_std'] / 1000,
                row['volume_kg_max'] / 1000,
                row.get('distance_to_market', 1.0),
                income_encoding,
                *division_encoded,  # One-hot encoded divisions (5 features)
                # Add day of week features (encoded)
                *[1 if i == idx % 7 else 0 for i in range(7)],
                # Add seasonal features
                np.sin(2 * np.pi * (idx % 365) / 365),
                np.cos(2 * np.pi * (idx % 365) / 365)
            ]
            features.append(feat)
        
        return np.array(features)
    
    def _create_spatial_edges(self, locations, k_neighbors=5):
        """
        Create edges based on spatial proximity
        """
        from sklearn.neighbors import kneighbors_graph
        
        # K-nearest neighbors
        adjacency = kneighbors_graph(locations, k_neighbors, mode='connectivity', include_self=False)
        edge_index = np.array(adjacency.nonzero())
        
        return edge_index
    
    def _create_type_targets(self, location_data):
        """
        Create multi-label targets for waste types based on location characteristics
        """
        n_locations = len(location_data)
        n_types = len(self.waste_types)
        
        # Initialize targets
        type_targets = np.zeros((n_locations, n_types))
        
        # Assign waste types based on location characteristics
        for idx, row in location_data.iterrows():
            if row['is_commercial']:
                # Commercial areas: more plastic, paper, cardboard
                type_targets[idx, [0, 1, 5]] = np.random.uniform(0.6, 0.9, 3)
            else:
                # Residential areas: more organic, plastic
                type_targets[idx, [0, 4]] = np.random.uniform(0.7, 0.95, 2)
            
            # Add some randomness
            type_targets[idx] += np.random.uniform(0, 0.2, n_types)
            type_targets[idx] = np.clip(type_targets[idx], 0, 1)
        
        return type_targets
    
    def split_data(self, data, train_ratio=0.7, val_ratio=0.0):
        """
        Split data into train and test sets (70%/30% split)
        For the KCCA Kiteezi dataset analysis
        """
        n_nodes = data.x.shape[0]
        indices = np.arange(n_nodes)
        
        if val_ratio == 0.0:
            # Simple 70%/30% train/test split
            train_idx, test_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
            val_idx = np.array([])  # Empty validation set
            print(f"Data split: {len(train_idx)} train, 0 validation, {len(test_idx)} test")
        else:
            # Original 3-way split for other use cases
            train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio/(1-train_ratio), random_state=42)
            print(f"Data split: {len(train_idx)} train, {len(val_idx)} validation, {len(test_idx)} test")
        
        return train_idx, val_idx, test_idx


def prepare_datasets(kiteezi_path=None):
    """
    Main function to prepare Kiteezi dataset
    """
    processor = WasteDataProcessor(kiteezi_path)
    
    # Create graph data
    graph_data, location_info = processor.create_kampala_graph_data()
    
    # Split data
    train_idx, val_idx, test_idx = processor.split_data(graph_data)
    
    return graph_data, location_info, train_idx, val_idx, test_idx, processor
