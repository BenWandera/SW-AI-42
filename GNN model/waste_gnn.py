"""
Graph Neural Network for Waste Volume and Type Prediction in Kampala
Uses KCCA Kiteezi dataset for waste volume anticipation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np


class WasteGNN(nn.Module):
    """
    Graph Neural Network for predicting waste volumes and types at different locations.
    
    Architecture:
    - Graph Convolutional layers to capture spatial relationships
    - Attention mechanism to weight important location features
    - Multi-task learning: waste volume regression + waste type classification
    """
    
    def __init__(self, 
                 num_node_features,
                 num_waste_types,
                 hidden_channels=64,
                 num_layers=3,
                 dropout=0.3):
        """
        Args:
            num_node_features: Number of features per location node
            num_waste_types: Number of waste categories to predict
            hidden_channels: Size of hidden layers
            num_layers: Number of GNN layers
            dropout: Dropout rate for regularization
        """
        super(WasteGNN, self).__init__()
        
        self.num_waste_types = num_waste_types
        self.dropout = dropout
        
        # Graph Attention layers for capturing spatial relationships
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels * 4)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 4)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Volume prediction head (regression)
        self.volume_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # Waste type prediction head (multi-label classification)
        self.type_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_waste_types)
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GNN
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector for batched graphs
            
        Returns:
            volume_pred: Predicted waste volumes
            type_pred: Predicted waste type probabilities
        """
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third GNN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Predictions
        volume_pred = self.volume_head(x)
        type_pred = self.type_head(x)
        
        return volume_pred, type_pred


class TemporalWasteGNN(nn.Module):
    """
    Extended GNN with temporal modeling using LSTM for time-series prediction
    """
    
    def __init__(self,
                 num_node_features,
                 num_waste_types,
                 hidden_channels=64,
                 lstm_hidden=32,
                 num_gnn_layers=3,
                 dropout=0.3):
        super(TemporalWasteGNN, self).__init__()
        
        self.num_waste_types = num_waste_types
        self.dropout = dropout
        
        # Spatial GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Temporal LSTM layer
        self.lstm = nn.LSTM(hidden_channels, lstm_hidden, batch_first=True, num_layers=2)
        
        # Prediction heads
        self.volume_head = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        self.type_head = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_waste_types)
        )
    
    def forward(self, x, edge_index, temporal_seq=None):
        """
        Forward pass with temporal modeling
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            temporal_seq: Optional temporal sequence for LSTM
        """
        # Spatial feature extraction
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # Temporal modeling (if sequence provided)
        if temporal_seq is not None:
            x = x.unsqueeze(1)  # Add time dimension
            x, _ = self.lstm(x)
            x = x.squeeze(1)
        
        # Predictions
        volume_pred = self.volume_head(x)
        type_pred = self.type_head(x)
        
        return volume_pred, type_pred


def create_spatial_graph(locations, k_neighbors=5):
    """
    Create a spatial graph connecting nearby locations
    
    Args:
        locations: Array of location coordinates [num_locations, 2] (lat, lon)
        k_neighbors: Number of nearest neighbors to connect
        
    Returns:
        edge_index: Graph connectivity tensor
    """
    from sklearn.neighbors import kneighbors_graph
    
    # Create k-nearest neighbors graph
    adjacency = kneighbors_graph(locations, k_neighbors, mode='connectivity', include_self=False)
    
    # Convert to edge index format
    edge_index = torch.tensor(np.array(adjacency.nonzero()), dtype=torch.long)
    
    return edge_index


def compute_loss(volume_pred, type_pred, volume_target, type_target, 
                 volume_weight=1.0, type_weight=1.0):
    """
    Multi-task loss function
    
    Args:
        volume_pred: Predicted volumes
        type_pred: Predicted waste types (logits)
        volume_target: Target volumes
        type_target: Target waste types (multi-hot encoded)
        volume_weight: Weight for volume loss
        type_weight: Weight for type loss
    """
    # Volume loss (MSE)
    volume_loss = F.mse_loss(volume_pred.squeeze(), volume_target)
    
    # Type loss (Binary Cross Entropy for multi-label)
    type_loss = F.binary_cross_entropy_with_logits(type_pred, type_target)
    
    # Combined loss
    total_loss = volume_weight * volume_loss + type_weight * type_loss
    
    return total_loss, volume_loss, type_loss