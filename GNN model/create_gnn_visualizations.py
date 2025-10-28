"""
GNN Knowledge Graph Visualization
Creates visual representations of the graph structure and relationships
"""

import sys
import os

# Add GNN model path for imports
gnn_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new GNN')
if gnn_path not in sys.path:
    sys.path.insert(0, gnn_path)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from waste_reasoning_rgn import create_waste_reasoning_model  # type: ignore

def visualize_knowledge_graph(model, save_path='gnn_knowledge_graph.png'):
    """Visualize the complete knowledge graph structure"""
    
    print("📊 Creating knowledge graph visualization...")
    
    kg = model.knowledge_graph
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    color_map = {
        'MATERIAL': '#FF6B6B',  # Red
        'CATEGORY': '#4ECDC4',  # Teal
        'DISPOSAL': '#95E1D3'   # Light teal
    }
    
    size_map = {
        'MATERIAL': 1500,
        'CATEGORY': 2500,
        'DISPOSAL': 2000
    }
    
    for node in kg.nodes:
        G.add_node(node.node_id)
        node_colors.append(color_map.get(node.node_type, '#CCCCCC'))
        node_sizes.append(size_map.get(node.node_type, 1000))
        
        # Simplified label
        label = node.label.replace('_', ' ').title()
        if len(label) > 15:
            label = label[:12] + '...'
        node_labels[node.node_id] = label
    
    # Add edges with relation types
    edge_colors = []
    edge_styles = []
    
    relation_color_map = {
        'derives_from': 'blue',
        'requires': 'green',
        'conflicts_with': 'red'
    }
    
    relation_style_map = {
        'derives_from': 'solid',
        'requires': 'dashed',
        'conflicts_with': 'dotted'
    }
    
    for edge in kg.edges:
        G.add_edge(edge.source, edge.target, 
                  relation=edge.relation_type,
                  weight=edge.weight)
        edge_colors.append(relation_color_map.get(edge.relation_type, 'gray'))
        edge_styles.append(relation_style_map.get(edge.relation_type, 'solid'))
    
    # Create figure
    plt.figure(figsize=(20, 14))
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    # Draw edges with different styles
    for i, edge in enumerate(G.edges()):
        nx.draw_networkx_edges(G, pos,
                              edgelist=[edge],
                              edge_color=[edge_colors[i]],
                              style=edge_styles[i],
                              width=2,
                              alpha=0.6,
                              arrowsize=20,
                              arrowstyle='->')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, node_labels, 
                           font_size=8,
                           font_weight='bold',
                           font_color='black')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Material Nodes',
              markerfacecolor='#FF6B6B', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Category Nodes',
              markerfacecolor='#4ECDC4', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Disposal Nodes',
              markerfacecolor='#95E1D3', markersize=15),
        Line2D([0], [0], color='blue', linewidth=2, label='derives_from'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='requires'),
        Line2D([0], [0], color='red', linewidth=2, linestyle=':', label='conflicts_with')
    ]
    
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.title('Waste Classification Knowledge Graph', fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Knowledge graph saved to: {save_path}")
    
    return G


def visualize_graph_statistics(model, save_path='gnn_graph_statistics.png'):
    """Create statistical visualizations of the knowledge graph"""
    
    print("📊 Creating graph statistics visualization...")
    
    kg = model.knowledge_graph
    
    # Collect statistics
    node_type_counts = {}
    relation_type_counts = {}
    risk_level_counts = {}
    
    for node in kg.nodes:
        node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1
        if hasattr(node, 'risk_level') and node.risk_level is not None:
            risk_name = node.risk_level.name if hasattr(node.risk_level, 'name') else str(node.risk_level)
            risk_level_counts[risk_name] = risk_level_counts.get(risk_name, 0) + 1
    
    for edge in kg.edges:
        relation_type_counts[edge.relation_type] = relation_type_counts.get(edge.relation_type, 0) + 1
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Node type distribution
    ax = axes[0, 0]
    node_types = list(node_type_counts.keys())
    node_counts = list(node_type_counts.values())
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    bars = ax.bar(node_types, node_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Node Type Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Relation type distribution
    ax = axes[0, 1]
    relation_types = list(relation_type_counts.keys())
    relation_counts = list(relation_type_counts.values())
    colors = ['#5DADE2', '#58D68D', '#EC7063']
    
    bars = ax.bar(relation_types, relation_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Edge Relation Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Risk level distribution
    ax = axes[1, 0]
    if risk_level_counts:
        risk_levels = list(risk_level_counts.keys())
        risk_counts = list(risk_level_counts.values())
        colors_risk = ['green', 'lightblue', 'yellow', 'orange', 'red'][:len(risk_levels)]
        
        bars = ax.bar(risk_levels, risk_counts, color=colors_risk, alpha=0.7, edgecolor='black', linewidth=2)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No Risk Data Available', 
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
    
    # 4. Graph connectivity metrics
    ax = axes[1, 1]
    
    # Calculate metrics
    total_nodes = len(kg.nodes)
    total_edges = len(kg.edges)
    avg_degree = (2 * total_edges) / total_nodes if total_nodes > 0 else 0
    safety_critical = sum(1 for node in kg.nodes if node.is_safety_critical)
    
    metrics = {
        'Total Nodes': total_nodes,
        'Total Edges': total_edges,
        'Avg Degree': round(avg_degree, 2),
        'Safety Critical': safety_critical
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.barh(metric_names, metric_values, color='#6C5CE7', alpha=0.7, edgecolor='black', linewidth=2)
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{width}',
               ha='left', va='center', fontsize=12, fontweight='bold', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Graph Connectivity Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Knowledge Graph Statistics', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Graph statistics saved to: {save_path}")


def visualize_model_architecture(model, save_path='gnn_architecture.png'):
    """Visualize the GNN model architecture"""
    
    print("📊 Creating model architecture visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Define architecture layers
    layers = [
        {'name': 'Vision Embedding\n(2048-dim)', 'type': 'input', 'y': 0.9},
        {'name': 'Vision Projection\n(256-dim)', 'type': 'projection', 'y': 0.75},
        {'name': 'Knowledge Graph\nEmbedding', 'type': 'graph', 'y': 0.6},
        {'name': 'RGC Layer 1\n(Relational Conv)', 'type': 'rgc', 'y': 0.45},
        {'name': 'RGC Layer 2\n(Relational Conv)', 'type': 'rgc', 'y': 0.30},
        {'name': 'RGC Layer 3\n(Relational Conv)', 'type': 'rgc', 'y': 0.15}
    ]
    
    outputs = [
        {'name': 'Material\nClassifier', 'y': 0.0, 'x': 0.15},
        {'name': 'Category\nClassifier', 'y': 0.0, 'x': 0.35},
        {'name': 'Disposal\nClassifier', 'y': 0.0, 'x': 0.55},
        {'name': 'Risk\nPredictor', 'y': 0.0, 'x': 0.75}
    ]
    
    # Color scheme
    color_map = {
        'input': '#FF6B6B',
        'projection': '#4ECDC4',
        'graph': '#95E1D3',
        'rgc': '#A8E6CF',
        'output': '#FFD93D'
    }
    
    # Draw main layers
    for layer in layers:
        box = FancyBboxPatch((0.25, layer['y']), 0.5, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor=color_map[layer['type']],
                            edgecolor='black',
                            linewidth=2,
                            alpha=0.8)
        ax.add_patch(box)
        ax.text(0.5, layer['y'] + 0.04, layer['name'],
               ha='center', va='center',
               fontsize=11, fontweight='bold')
        
        # Draw arrows between layers
        if layer['y'] > 0.15:
            ax.annotate('', xy=(0.5, layer['y']),
                       xytext=(0.5, layer['y'] + 0.13),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    
    # Draw output heads
    for out in outputs:
        box = FancyBboxPatch((out['x'] - 0.08, out['y']), 0.16, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor=color_map['output'],
                            edgecolor='black',
                            linewidth=2,
                            alpha=0.8)
        ax.add_patch(box)
        ax.text(out['x'], out['y'] + 0.04, out['name'],
               ha='center', va='center',
               fontsize=9, fontweight='bold')
        
        # Draw arrow from last layer to output
        ax.annotate('', xy=(out['x'], out['y'] + 0.08),
                   xytext=(0.5, 0.15),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.7))
    
    # Add title and info
    ax.text(0.5, 0.97, 'Waste Reasoning GNN Architecture',
           ha='center', va='top',
           fontsize=16, fontweight='bold')
    
    # Add legend
    legend_y = 0.85
    legend_items = [
        ('Input Layer', color_map['input']),
        ('Projection', color_map['projection']),
        ('Graph Embedding', color_map['graph']),
        ('RGC Layers', color_map['rgc']),
        ('Output Heads', color_map['output'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y = legend_y - i * 0.05
        box = FancyBboxPatch((0.02, y), 0.03, 0.03,
                            boxstyle="round,pad=0.003",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=1)
        ax.add_patch(box)
        ax.text(0.06, y + 0.015, label,
               ha='left', va='center',
               fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Model architecture saved to: {save_path}")


def create_all_visualizations(output_dir='gnn_visualizations'):
    """Create all GNN visualizations"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎨 Creating GNN Visualizations")
    print("=" * 50)
    
    # Initialize model
    print("🧠 Initializing GNN model...")
    model = create_waste_reasoning_model(vision_embedding_dim=2048)
    print("✅ Model initialized")
    
    # Create visualizations
    visualize_knowledge_graph(model, 
                             save_path=os.path.join(output_dir, 'knowledge_graph.png'))
    
    visualize_graph_statistics(model,
                              save_path=os.path.join(output_dir, 'graph_statistics.png'))
    
    visualize_model_architecture(model,
                                save_path=os.path.join(output_dir, 'model_architecture.png'))
    
    print(f"\n✅ All visualizations created in: {output_dir}")
    print("📊 Generated files:")
    print("   • knowledge_graph.png - Full graph structure")
    print("   • graph_statistics.png - Statistical analysis")
    print("   • model_architecture.png - Model layer structure")


if __name__ == "__main__":
    create_all_visualizations()