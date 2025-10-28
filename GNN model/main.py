"""
Main execution script for Kampala Waste GNN
This script orchestrates the entire pipeline: data loading, training, and prediction
"""

import torch
import numpy as np
import os
import argparse
from pathlib import Path

from data_preprocessing import prepare_datasets
from waste_gnn import WasteGNN
from training import WasteGNNTrainer
from prediction import WastePredictor


def main(args):
    """
    Main execution function
    """
    print("=" * 80)
    print("KAMPALA WASTE PREDICTION USING GRAPH NEURAL NETWORKS")
    print("=" * 80)
    print()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # ==================== STEP 1: DATA PREPARATION ====================
    print("STEP 1: Loading and preparing datasets...")
    print("-" * 80)
    
    graph_data, location_info, train_idx, val_idx, test_idx, processor = prepare_datasets(
        kiteezi_path=args.kiteezi_path
    )
    
    print(f"Number of locations (nodes): {graph_data.x.shape[0]}")
    print(f"Number of features per node: {graph_data.x.shape[1]}")
    print(f"Number of edges: {graph_data.edge_index.shape[1]}")
    print(f"Training set size: {len(train_idx)}")
    print(f"Validation set size: {len(val_idx)}")
    print(f"Test set size: {len(test_idx)}")
    print()
    
    # Create masks
    train_mask = torch.zeros(graph_data.x.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True
    
    val_mask = torch.zeros(graph_data.x.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True
    
    test_mask = torch.zeros(graph_data.x.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True
    
    # ==================== STEP 2: MODEL INITIALIZATION ====================
    print("STEP 2: Initializing GNN model...")
    print("-" * 80)
    
    num_node_features = graph_data.x.shape[1]
    num_waste_types = graph_data.y_type.shape[1]
    
    model = WasteGNN(
        num_node_features=num_node_features,
        num_waste_types=num_waste_types,
        hidden_channels=args.hidden_channels,
        num_layers=3,
        dropout=args.dropout
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    
    # ==================== STEP 3: TRAINING ====================
    if not args.skip_training:
        print("STEP 3: Training model...")
        print("-" * 80)
        
        trainer = WasteGNNTrainer(model, device=device)
        
        history = trainer.train(
            data=graph_data,
            train_mask=train_mask,
            val_mask=val_mask,
            epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            volume_weight=args.volume_weight,
            type_weight=args.type_weight,
            patience=args.patience,
            save_path=args.model_save_path
        )
        
        # Plot training history
        trainer.plot_training_history(save_path=args.plot_save_path)
        print()
        
        # ==================== STEP 4: EVALUATION ====================
        print("STEP 4: Evaluating on test set...")
        print("-" * 80)
        
        test_loss, test_vol_loss, test_type_loss, test_metrics = trainer.evaluate(
            graph_data, test_mask
        )
        
        print(f"Test Results:")
        print(f"  Total Loss: {test_loss:.4f}")
        print(f"  Volume Loss (MSE): {test_vol_loss:.4f}")
        print(f"  Type Loss (BCE): {test_type_loss:.4f}")
        print(f"\nVolume Prediction Metrics:")
        print(f"  MAE: {test_metrics['volume_mae']:.2f} kg/day")
        print(f"  RMSE: {test_metrics['volume_rmse']:.2f} kg/day")
        print(f"  R² Score: {test_metrics['volume_r2']:.4f}")
        print(f"\nWaste Type Prediction Metrics:")
        print(f"  Accuracy: {test_metrics['type_accuracy']:.4f}")
        print(f"  Precision: {test_metrics['type_precision']:.4f}")
        print(f"  Recall: {test_metrics['type_recall']:.4f}")
        print(f"  F1 Score: {test_metrics['type_f1']:.4f}")
        print()
        
        # Save results
        trainer.save_results(test_metrics, save_path=args.results_save_path)
    else:
        print("STEP 3-4: Skipping training (loading pretrained model)...")
        print("-" * 80)
        
        # Load pretrained model
        checkpoint = torch.load(os.path.join(args.model_save_path, 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_save_path}")
        print()
    
    # ==================== STEP 5: PREDICTION & VISUALIZATION ====================
    print("STEP 5: Generating predictions and visualizations...")
    print("-" * 80)
    
    predictor = WastePredictor(model, device=device)
    
    # Make predictions for all locations
    volume_pred, type_pred = predictor.predict(graph_data)
    
    print(f"Predictions generated for {len(volume_pred)} locations")
    print(f"Average predicted volume: {np.mean(volume_pred):.2f} kg/day")
    print(f"Total daily volume: {np.sum(volume_pred):.2f} kg")
    print()
    
    # Identify high-priority locations
    print("Identifying high-priority locations...")
    top_locations = predictor.identify_high_waste_locations(
        volume_pred, location_info, top_n=20
    )
    print(f"Top 5 highest volume locations:")
    for idx, row in top_locations.head(5).iterrows():
        print(f"  Location {row['location_id']}: {row['predicted_volume']:.1f} kg/day")
    print()
    
    # Analyze waste types
    print("Analyzing waste type distribution...")
    location_analysis = predictor.analyze_waste_types_by_location(type_pred, location_info)
    print("Dominant waste types by area:")
    waste_type_counts = location_analysis['dominant_waste_type'].value_counts()
    for waste_type, count in waste_type_counts.head(5).items():
        print(f"  {waste_type}: {count} locations")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    predictor.plot_waste_distribution(
        volume_pred, type_pred, location_info,
        save_path=args.plot_save_path
    )
    
    # Create division-specific visualizations
    print("Creating division-specific waste volume visualizations...")
    from division_visualizer import KampalaDivisionVisualizer
    
    visualizer = KampalaDivisionVisualizer()
    division_data = visualizer.create_division_waste_graphs(
        location_info, volume_pred, save_path=args.plot_save_path
    )
    
    # Generate division analysis report
    division_report_path = visualizer.generate_division_report(
        division_data, save_path=args.results_save_path
    )
    
    # Create interactive map
    if args.create_map:
        print("Creating interactive Kampala waste map...")
        predictor.create_kampala_waste_map(
            location_info, volume_pred, type_pred,
            save_path=args.map_save_path
        )
    
    # Generate recommendations
    print("Generating actionable recommendations...")
    recommendations = predictor.generate_recommendations(
        volume_pred, type_pred, location_info,
        save_path=args.results_save_path
    )
    print()
    
    # ==================== COMPLETION ====================
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  - Model: {args.model_save_path}/best_model.pt")
    print(f"  - Training plots: {args.plot_save_path}/training_history.png")
    print(f"  - Analysis plots: {args.plot_save_path}/waste_distribution_analysis.png")
    print(f"  - Division dashboard: {args.plot_save_path}/kampala_division_dashboard.png")
    print(f"  - Division analysis: {args.plot_save_path}/[division]_division_analysis.png")
    print(f"  - Temporal projections: {args.plot_save_path}/temporal_projections.png")
    print(f"  - Comparative analysis: {args.plot_save_path}/comparative_analysis.png")
    if args.create_map:
        print(f"  - Interactive map: {args.map_save_path}/kampala_waste_map.html")
    print(f"  - Recommendations: {args.results_save_path}/recommendations.json")
    print(f"  - Summary report: {args.results_save_path}/summary_report.txt")
    print(f"  - Division report: {args.results_save_path}/kampala_divisions_report.txt")
    print(f"  - Test results: {args.results_save_path}/results.json")
    print()
    print("Next steps:")
    print("  1. Review the division dashboard for comprehensive waste volume overview")
    print("  2. Analyze individual division reports for targeted strategies")
    print("  3. Use temporal projections for collection planning")
    print("  4. Implement resource allocation based on comparative analysis")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kampala Waste Prediction using Graph Neural Networks"
    )
    
    # Data paths
    parser.add_argument('--kiteezi-path', type=str, default=None,
                       help='Path to Kiteezi volume dataset')
    
    # Model parameters
    parser.add_argument('--hidden-channels', type=int, default=64,
                       help='Number of hidden channels in GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for regularization')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--volume-weight', type=float, default=1.0,
                       help='Weight for volume loss in multi-task learning')
    parser.add_argument('--type-weight', type=float, default=1.0,
                       help='Weight for type loss in multi-task learning')
    parser.add_argument('--patience', type=int, default=20,
                       help='Patience for early stopping')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and use pretrained model')
    
    # Output paths
    parser.add_argument('--model-save-path', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--plot-save-path', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--map-save-path', type=str, default='maps',
                       help='Directory to save interactive maps')
    parser.add_argument('--results-save-path', type=str, default='results',
                       help='Directory to save results and recommendations')
    
    # Other options
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--create-map', action='store_true', default=True,
                       help='Create interactive Kampala waste map')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.plot_save_path, exist_ok=True)
    os.makedirs(args.map_save_path, exist_ok=True)
    os.makedirs(args.results_save_path, exist_ok=True)
    
    # Run main pipeline
    main(args)