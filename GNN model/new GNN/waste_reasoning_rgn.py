"""
Relational Graph Network (RGN) for Waste Classification Reasoning
Handles safety-critical classifications, hierarchical reasoning, and conflict resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum


class WasteCategory(Enum):
    """Primary waste categories"""
    PLASTIC = "plastic"
    ORGANIC = "organic"
    PAPER = "paper"
    GLASS = "glass"
    METAL = "metal"
    ELECTRONIC = "electronic"
    MEDICAL = "medical"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class MaterialType(Enum):
    """Material composition types"""
    PLASTIC_PET = "plastic_pet"
    PLASTIC_HDPE = "plastic_hdpe"
    PLASTIC_PVC = "plastic_pvc"
    PLASTIC_OTHER = "plastic_other"
    ORGANIC_FOOD = "organic_food"
    ORGANIC_YARD = "organic_yard"
    PAPER_CARDBOARD = "paper_cardboard"
    PAPER_MIXED = "paper_mixed"
    GLASS_CLEAR = "glass_clear"
    GLASS_COLORED = "glass_colored"
    METAL_ALUMINUM = "metal_aluminum"
    METAL_STEEL = "metal_steel"
    ELECTRONIC_CIRCUIT = "electronic_circuit"
    ELECTRONIC_BATTERY = "electronic_battery"
    MEDICAL_SHARP = "medical_sharp"
    MEDICAL_INFECTIOUS = "medical_infectious"
    MEDICAL_PHARMACEUTICAL = "medical_pharmaceutical"


class RiskLevel(Enum):
    """Safety risk levels"""
    SAFE = 0
    LOW_RISK = 1
    MEDIUM_RISK = 2
    HIGH_RISK = 3
    CRITICAL = 4


class DisposalMethod(Enum):
    """Disposal recommendations"""
    RECYCLABLE = "recyclable"
    COMPOSTABLE = "compostable"
    LANDFILL = "landfill"
    HAZARDOUS_DISPOSAL = "hazardous_disposal"
    SPECIALIZED_FACILITY = "specialized_facility"
    INCINERATION = "incineration"


@dataclass
class WasteNode:
    """Node representation in the waste knowledge graph"""
    node_id: int
    node_type: str  # 'category', 'material', 'risk', 'disposal'
    label: str
    features: torch.Tensor
    risk_level: int = 0
    is_safety_critical: bool = False


@dataclass
class GraphEdge:
    """Edge representation with relation type"""
    source: int
    target: int
    relation_type: str  # 'contains', 'derives_from', 'requires', 'conflicts_with'
    weight: float = 1.0


class WasteKnowledgeGraph:
    """
    Constructs the hierarchical knowledge graph for waste classification
    Structure: Material → Category → Disposal Method
    """
    
    def __init__(self):
        self.nodes: List[WasteNode] = []
        self.edges: List[GraphEdge] = []
        self.node_map: Dict[str, int] = {}
        self.safety_rules: Dict[str, List[str]] = {}
        self.conflict_matrix: torch.Tensor = None
        
        self._build_graph()
        self._define_safety_rules()
        self._build_conflict_matrix()
    
    def _build_graph(self):
        """Build the hierarchical waste classification graph"""
        node_id = 0
        
        # Level 1: Material Nodes
        materials = [
            ("plastic_pet", "material", RiskLevel.LOW_RISK, False),
            ("plastic_pvc", "material", RiskLevel.MEDIUM_RISK, False),
            ("organic_food", "material", RiskLevel.SAFE, False),
            ("paper_cardboard", "material", RiskLevel.SAFE, False),
            ("glass_clear", "material", RiskLevel.LOW_RISK, False),
            ("metal_aluminum", "material", RiskLevel.LOW_RISK, False),
            ("electronic_battery", "material", RiskLevel.HIGH_RISK, True),
            ("medical_sharp", "material", RiskLevel.CRITICAL, True),
            ("medical_infectious", "material", RiskLevel.CRITICAL, True),
        ]
        
        for label, node_type, risk, is_critical in materials:
            features = self._create_material_features(label, risk.value)
            node = WasteNode(
                node_id=node_id,
                node_type=node_type,
                label=label,
                features=features,
                risk_level=risk.value,
                is_safety_critical=is_critical
            )
            self.nodes.append(node)
            self.node_map[label] = node_id
            node_id += 1
        
        # Level 2: Category Nodes
        categories = [
            ("plastic", "category", RiskLevel.LOW_RISK),
            ("organic", "category", RiskLevel.SAFE),
            ("paper", "category", RiskLevel.SAFE),
            ("glass", "category", RiskLevel.LOW_RISK),
            ("metal", "category", RiskLevel.LOW_RISK),
            ("electronic", "category", RiskLevel.HIGH_RISK),
            ("medical", "category", RiskLevel.CRITICAL),
        ]
        
        for label, node_type, risk in categories:
            features = self._create_category_features(label, risk.value)
            node = WasteNode(
                node_id=node_id,
                node_type=node_type,
                label=label,
                features=features,
                risk_level=risk.value,
                is_safety_critical=(risk.value >= RiskLevel.HIGH_RISK.value)
            )
            self.nodes.append(node)
            self.node_map[label] = node_id
            node_id += 1
        
        # Level 3: Disposal Method Nodes
        disposals = [
            ("recyclable", "disposal"),
            ("compostable", "disposal"),
            ("landfill", "disposal"),
            ("hazardous_disposal", "disposal"),
            ("specialized_facility", "disposal"),
        ]
        
        for label, node_type in disposals:
            features = self._create_disposal_features(label)
            node = WasteNode(
                node_id=node_id,
                node_type=node_type,
                label=label,
                features=features
            )
            self.nodes.append(node)
            self.node_map[label] = node_id
            node_id += 1
        
        # Build hierarchical edges: Material → Category → Disposal
        self._build_hierarchical_edges()
    
    def _create_material_features(self, label: str, risk: int) -> torch.Tensor:
        """Create feature vector for material nodes"""
        # Feature vector: [recyclability, biodegradability, toxicity, density, risk_score]
        feature_map = {
            "plastic_pet": [0.9, 0.1, 0.2, 0.4, risk],
            "plastic_pvc": [0.5, 0.0, 0.7, 0.5, risk],
            "organic_food": [0.0, 1.0, 0.0, 0.3, risk],
            "paper_cardboard": [0.8, 0.6, 0.1, 0.2, risk],
            "glass_clear": [1.0, 0.0, 0.0, 0.8, risk],
            "metal_aluminum": [1.0, 0.0, 0.1, 0.7, risk],
            "electronic_battery": [0.3, 0.0, 0.9, 0.6, risk],
            "medical_sharp": [0.0, 0.0, 1.0, 0.5, risk],
            "medical_infectious": [0.0, 0.0, 1.0, 0.4, risk],
        }
        return torch.tensor(feature_map.get(label, [0.0] * 5), dtype=torch.float32)
    
    def _create_category_features(self, label: str, risk: int) -> torch.Tensor:
        """Create feature vector for category nodes"""
        # Feature vector: [volume_frequency, sorting_difficulty, contamination_risk, risk_score]
        feature_map = {
            "plastic": [0.9, 0.6, 0.5, risk],
            "organic": [0.8, 0.3, 0.4, risk],
            "paper": [0.7, 0.4, 0.3, risk],
            "glass": [0.5, 0.5, 0.2, risk],
            "metal": [0.4, 0.6, 0.2, risk],
            "electronic": [0.3, 0.8, 0.7, risk],
            "medical": [0.2, 0.9, 1.0, risk],
        }
        return torch.tensor(feature_map.get(label, [0.0] * 4), dtype=torch.float32)
    
    def _create_disposal_features(self, label: str) -> torch.Tensor:
        """Create feature vector for disposal nodes"""
        # Feature vector: [environmental_impact, cost, accessibility, safety_requirement]
        feature_map = {
            "recyclable": [0.2, 0.4, 0.8, 0.3],
            "compostable": [0.1, 0.3, 0.7, 0.2],
            "landfill": [0.7, 0.5, 0.9, 0.4],
            "hazardous_disposal": [0.3, 0.9, 0.3, 1.0],
            "specialized_facility": [0.4, 0.8, 0.4, 0.9],
        }
        return torch.tensor(feature_map.get(label, [0.0] * 4), dtype=torch.float32)
    
    def _build_hierarchical_edges(self):
        """Build edges representing hierarchical relationships"""
        # Material → Category edges
        material_to_category = {
            "plastic_pet": "plastic",
            "plastic_pvc": "plastic",
            "organic_food": "organic",
            "paper_cardboard": "paper",
            "glass_clear": "glass",
            "metal_aluminum": "metal",
            "electronic_battery": "electronic",
            "medical_sharp": "medical",
            "medical_infectious": "medical",
        }
        
        for material, category in material_to_category.items():
            edge = GraphEdge(
                source=self.node_map[material],
                target=self.node_map[category],
                relation_type="derives_from",
                weight=1.0
            )
            self.edges.append(edge)
        
        # Category → Disposal edges
        category_to_disposal = {
            "plastic": ["recyclable", "landfill"],
            "organic": ["compostable"],
            "paper": ["recyclable", "compostable"],
            "glass": ["recyclable"],
            "metal": ["recyclable"],
            "electronic": ["specialized_facility", "hazardous_disposal"],
            "medical": ["hazardous_disposal", "specialized_facility"],
        }
        
        for category, disposals in category_to_disposal.items():
            for disposal in disposals:
                edge = GraphEdge(
                    source=self.node_map[category],
                    target=self.node_map[disposal],
                    relation_type="requires",
                    weight=1.0
                )
                self.edges.append(edge)
        
        # Add conflict edges (materials that shouldn't be mixed)
        conflicts = [
            ("medical_sharp", "plastic_pet"),
            ("medical_infectious", "organic_food"),
            ("electronic_battery", "organic_food"),
        ]
        
        for mat1, mat2 in conflicts:
            edge = GraphEdge(
                source=self.node_map[mat1],
                target=self.node_map[mat2],
                relation_type="conflicts_with",
                weight=1.0
            )
            self.edges.append(edge)
    
    def _define_safety_rules(self):
        """Define safety-critical classification rules"""
        self.safety_rules = {
            "medical_sharp": [
                "MUST be classified as medical waste",
                "MUST use hazardous disposal",
                "CANNOT be mixed with any other waste",
                "Requires immediate specialized handling"
            ],
            "medical_infectious": [
                "MUST be classified as medical waste",
                "MUST use specialized facility",
                "CANNOT be in contact with food waste",
                "Requires sterilization before disposal"
            ],
            "electronic_battery": [
                "MUST be classified as electronic waste",
                "CANNOT go to landfill",
                "MUST use specialized recycling facility",
                "Fire hazard if damaged"
            ],
            "plastic_pvc": [
                "SHOULD NOT be incinerated",
                "May release toxic chemicals when burned",
                "Prefer specialized recycling"
            ]
        }
    
    def _build_conflict_matrix(self):
        """Build matrix representing conflicts between waste types"""
        n_nodes = len(self.nodes)
        self.conflict_matrix = torch.zeros((n_nodes, n_nodes))
        
        for edge in self.edges:
            if edge.relation_type == "conflicts_with":
                self.conflict_matrix[edge.source, edge.target] = 1.0
                self.conflict_matrix[edge.target, edge.source] = 1.0
    
    def get_node_features(self) -> torch.Tensor:
        """Get all node features as a tensor"""
        max_feature_dim = max(node.features.size(0) for node in self.nodes)
        features = torch.zeros((len(self.nodes), max_feature_dim))
        
        for i, node in enumerate(self.nodes):
            feat_len = node.features.size(0)
            features[i, :feat_len] = node.features
        
        return features
    
    def get_edge_index(self) -> torch.Tensor:
        """Get edge indices in PyTorch Geometric format"""
        edge_index = torch.tensor(
            [[edge.source, edge.target] for edge in self.edges],
            dtype=torch.long
        ).t()
        return edge_index
    
    def get_edge_types(self) -> torch.Tensor:
        """Get edge type indices for relational reasoning"""
        relation_map = {
            "derives_from": 0,
            "requires": 1,
            "conflicts_with": 2
        }
        edge_types = torch.tensor(
            [relation_map[edge.relation_type] for edge in self.edges],
            dtype=torch.long
        )
        return edge_types


class RelationalGraphConvolution(nn.Module):
    """
    Relational Graph Convolution layer for multi-relational reasoning
    """
    
    def __init__(self, in_features: int, out_features: int, num_relations: int):
        super(RelationalGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        
        # Separate weight matrices for each relation type
        self.relation_weights = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_relations)
        ])
        
        # Self-loop weight
        self.self_weight = nn.Linear(in_features, out_features)
        
        # Attention mechanism for relation importance
        self.attention = nn.Linear(in_features * 2, 1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_types: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with relational aggregation
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_types: Edge type indices [num_edges]
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        
        # Self-loop transformation
        out = self.self_weight(x)
        
        # Aggregate messages from each relation type
        for relation_id in range(self.num_relations):
            # Get edges of this relation type
            mask = edge_types == relation_id
            if mask.sum() == 0:
                continue
            
            rel_edge_index = edge_index[:, mask]
            
            # Source and target nodes
            source_nodes = rel_edge_index[0]
            target_nodes = rel_edge_index[1]
            
            # Transform source features
            transformed = self.relation_weights[relation_id](x[source_nodes])
            
            # Compute attention weights
            source_features = x[source_nodes]
            target_features = x[target_nodes]
            attention_input = torch.cat([source_features, target_features], dim=1)
            attention_weights = torch.sigmoid(self.attention(attention_input))
            
            # Apply attention
            transformed = transformed * attention_weights
            
            # Aggregate to target nodes
            aggregated = torch.zeros((num_nodes, self.out_features), 
                                    dtype=x.dtype, device=x.device)
            aggregated.index_add_(0, target_nodes, transformed)
            
            out = out + aggregated
        
        return out


class WasteReasoningRGN(nn.Module):
    """
    Complete Relational Graph Network for waste classification reasoning
    Handles hierarchical classification, safety rules, and conflict resolution
    """
    
    def __init__(self, 
                 input_dim: int = 2048,  # Vision model embedding size
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_relations: int = 3,
                 num_categories: int = 9):
        super(WasteReasoningRGN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initialize knowledge graph
        self.knowledge_graph = WasteKnowledgeGraph()
        self.num_relations = num_relations
        
        # Vision embedding projection
        self.vision_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph node feature embedding
        graph_feature_dim = self.knowledge_graph.get_node_features().size(1)
        self.graph_embedding = nn.Linear(graph_feature_dim, hidden_dim)
        
        # Relational Graph Convolution layers
        self.rgc_layers = nn.ModuleList([
            RelationalGraphConvolution(hidden_dim, hidden_dim, num_relations)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Hierarchical reasoning layers
        self.material_classifier = nn.Linear(hidden_dim, 16)  # Material types
        self.category_classifier = nn.Linear(hidden_dim, num_categories)
        self.disposal_classifier = nn.Linear(hidden_dim, 5)  # Disposal methods
        
        # Risk assessment head
        self.risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 risk levels
        )
        
        # Conflict detection head
        self.conflict_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vision_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the reasoning network
        
        Args:
            vision_embedding: Visual features from LVM [batch_size, input_dim]
        
        Returns:
            Dictionary containing:
                - material_logits: Material classification
                - category_logits: Category classification
                - disposal_logits: Disposal method classification
                - risk_scores: Safety risk assessment
                - confidence: Classification confidence
                - graph_embeddings: Final graph node embeddings
        """
        batch_size = vision_embedding.size(0)
        
        # Project vision features
        vision_features = self.vision_projection(vision_embedding)  # [batch, hidden_dim]
        
        # Get graph structure
        graph_features = self.knowledge_graph.get_node_features().to(vision_embedding.device)
        edge_index = self.knowledge_graph.get_edge_index().to(vision_embedding.device)
        edge_types = self.knowledge_graph.get_edge_types().to(vision_embedding.device)
        
        # Embed graph node features
        graph_embed = self.graph_embedding(graph_features)  # [num_nodes, hidden_dim]
        
        # Graph reasoning through RGC layers
        h = graph_embed
        for i, (rgc, norm) in enumerate(zip(self.rgc_layers, self.layer_norms)):
            h_new = rgc(h, edge_index, edge_types)
            h_new = norm(h_new)
            h = F.relu(h_new) + h  # Residual connection
        
        # Aggregate graph reasoning with vision features
        # Use attention pooling to get relevant graph nodes
        attention_scores = torch.matmul(vision_features, h.t())  # [batch, num_nodes]
        attention_weights = F.softmax(attention_scores, dim=1)
        graph_context = torch.matmul(attention_weights, h)  # [batch, hidden_dim]
        
        # Combine vision and graph reasoning
        combined_features = vision_features + graph_context
        
        # Hierarchical classification
        material_logits = self.material_classifier(combined_features)
        category_logits = self.category_classifier(combined_features)
        disposal_logits = self.disposal_classifier(combined_features)
        
        # Risk assessment
        risk_scores = self.risk_predictor(combined_features)
        
        # Confidence estimation
        confidence = self.confidence_estimator(combined_features)
        
        return {
            "material_logits": material_logits,
            "category_logits": category_logits,
            "disposal_logits": disposal_logits,
            "risk_scores": risk_scores,
            "confidence": confidence,
            "graph_embeddings": h,
            "attention_weights": attention_weights
        }
    
    def apply_safety_rules(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply safety-critical rules to override predictions when necessary
        
        Args:
            outputs: Raw model outputs
        
        Returns:
            Corrected outputs with safety rules applied
        """
        risk_scores = outputs["risk_scores"]
        category_logits = outputs["category_logits"]
        
        # Rule 1: If critical risk detected, force medical/hazardous classification
        critical_risk_mask = torch.argmax(risk_scores, dim=1) >= RiskLevel.HIGH_RISK.value
        
        if critical_risk_mask.any():
            # Override category to medical or electronic
            medical_idx = list(WasteCategory).index(WasteCategory.MEDICAL)
            category_logits[critical_risk_mask, medical_idx] += 10.0  # Strong boost
        
        # Rule 2: Electronic waste with battery signature → specialized disposal
        # This would be implemented with additional sensor/context data
        
        outputs["category_logits"] = category_logits
        return outputs
    
    def resolve_conflicts(self, 
                         predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Resolve conflicts when multiple waste items are detected
        
        Args:
            predictions: List of predictions for detected items
        
        Returns:
            Resolved final classification
        """
        if len(predictions) == 1:
            return predictions[0]
        
        # Check for conflicts using conflict matrix
        conflict_matrix = self.knowledge_graph.conflict_matrix
        
        resolved = predictions[0].copy()
        max_risk = torch.argmax(predictions[0]["risk_scores"])
        
        for pred in predictions[1:]:
            current_risk = torch.argmax(pred["risk_scores"])
            
            # Priority rule: highest risk takes precedence
            if current_risk > max_risk:
                resolved = pred
                max_risk = current_risk
        
        # Add conflict warning flag
        resolved["has_conflict"] = torch.tensor([len(predictions) > 1])
        resolved["num_items"] = torch.tensor([len(predictions)])
        
        return resolved
    
    def get_explanation(self, outputs: Dict[str, torch.Tensor], 
                       sample_idx: int = 0) -> Dict[str, any]:
        """
        Generate human-readable explanation for the classification
        
        Args:
            outputs: Model outputs
            sample_idx: Index of sample in batch
        
        Returns:
            Explanation dictionary
        """
        category_idx = torch.argmax(outputs["category_logits"][sample_idx]).item()
        category = list(WasteCategory)[category_idx]
        
        risk_idx = torch.argmax(outputs["risk_scores"][sample_idx]).item()
        risk_level = list(RiskLevel)[risk_idx]
        
        confidence = outputs["confidence"][sample_idx].item()
        
        # Get top contributing graph nodes
        attention = outputs["attention_weights"][sample_idx]
        top_nodes_idx = torch.topk(attention, k=3).indices
        top_nodes = [self.knowledge_graph.nodes[idx].label for idx in top_nodes_idx]
        
        explanation = {
            "category": category.value,
            "risk_level": risk_level.name,
            "confidence": f"{confidence:.2%}",
            "reasoning_path": " → ".join(top_nodes),
            "safety_critical": risk_idx >= RiskLevel.HIGH_RISK.value,
        }
        
        # Add safety rules if applicable
        for node_label in top_nodes:
            if node_label in self.knowledge_graph.safety_rules:
                explanation["safety_rules"] = self.knowledge_graph.safety_rules[node_label]
                break
        
        return explanation


def create_waste_reasoning_model(vision_embedding_dim: int = 2048) -> WasteReasoningRGN:
    """
    Factory function to create the waste reasoning model
    
    Args:
        vision_embedding_dim: Dimension of vision model embeddings
    
    Returns:
        Initialized WasteReasoningRGN model
    """
    model = WasteReasoningRGN(
        input_dim=vision_embedding_dim,
        hidden_dim=256,
        num_layers=3,
        num_relations=3,
        num_categories=len(WasteCategory)
    )
    return model


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Waste Classification Relational Graph Network (RGN)")
    print("=" * 60)
    
    # Create model
    model = create_waste_reasoning_model(vision_embedding_dim=2048)
    print(f"\n✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Simulate vision embedding from LVM
    batch_size = 4
    vision_embedding = torch.randn(batch_size, 2048)
    
    print(f"\n✓ Processing batch of {batch_size} waste images...")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(vision_embedding)
        outputs = model.apply_safety_rules(outputs)
    
    print("\n" + "=" * 60)
    print("Model Outputs:")
    print("=" * 60)
    
    for i in range(batch_size):
        print(f"\nSample {i + 1}:")
        explanation = model.get_explanation(outputs, sample_idx=i)
        
        print(f"  Category: {explanation['category']}")
        print(f"  Risk Level: {explanation['risk_level']}")
        print(f"  Confidence: {explanation['confidence']}")
        print(f"  Reasoning Path: {explanation['reasoning_path']}")
        print(f"  Safety Critical: {explanation['safety_critical']}")
        
        if "safety_rules" in explanation:
            print("  Safety Rules:")
            for rule in explanation['safety_rules']:
                print(f"    - {rule}")
    
    print("\n" + "=" * 60)
    print("Knowledge Graph Statistics:")
    print("=" * 60)
    kg = model.knowledge_graph
    print(f"  Total Nodes: {len(kg.nodes)}")
    print(f"  Total Edges: {len(kg.edges)}")
    print(f"  Node Types: {set(node.node_type for node in kg.nodes)}")
    print(f"  Relation Types: {set(edge.relation_type for edge in kg.edges)}")
    
    safety_critical_nodes = [node for node in kg.nodes if node.is_safety_critical]
    print(f"  Safety-Critical Nodes: {len(safety_critical_nodes)}")
    for node in safety_critical_nodes:
        print(f"    - {node.label} (Risk Level: {node.risk_level})")
    
    print("\n✓ Reasoning model test completed successfully!")