#!/usr/bin/env python3
"""
GNN Model Architecture for Corner Kick Outcome Prediction

Implements Phase 3.1: Core GNN architecture using PyTorch Geometric.
Based on Bekkers & Sahasrabudhe (2024) "A Graph Neural Network Deep-Dive into Successful Counterattacks"

Architecture:
- 3 GraphConv layers with increasing/decreasing channels
- Global pooling (mean + max concatenation)
- 2 fully connected layers
- Binary classification output (goal prediction)

Author: mseo
Date: October 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, NNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional, Dict


class CornerGNN(nn.Module):
    """
    Graph Neural Network for corner kick outcome prediction.

    Processes player positions and features to predict goal probability.

    Architecture:
        GraphConv1: 14 → 64 (ReLU, Dropout=0.3)
        GraphConv2: 64 → 128 (ReLU, Dropout=0.3)
        GraphConv3: 128 → 64 (ReLU)
        Global Pooling: Mean + Max concatenation → 128
        Dense1: 128 → 64 (ReLU, Dropout=0.2)
        Dense2: 64 → 32 (ReLU)
        Output: 32 → 1 (Sigmoid for binary)
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim1: int = 64,
        hidden_dim2: int = 128,
        hidden_dim3: int = 64,
        fc_dim1: int = 64,
        fc_dim2: int = 32,
        output_dim: int = 1,
        dropout_rate: float = 0.3,
        fc_dropout_rate: float = 0.2,
        use_edge_features: bool = False
    ):
        """
        Initialize the Corner GNN model.

        Args:
            input_dim: Number of node features (default: 14)
            hidden_dim1: Hidden dimension for first GCN layer
            hidden_dim2: Hidden dimension for second GCN layer
            hidden_dim3: Hidden dimension for third GCN layer
            fc_dim1: Hidden dimension for first FC layer
            fc_dim2: Hidden dimension for second FC layer
            output_dim: Output dimension (1 for binary classification)
            dropout_rate: Dropout rate for GCN layers
            fc_dropout_rate: Dropout rate for FC layers
            use_edge_features: Whether to use edge features (future enhancement)
        """
        super(CornerGNN, self).__init__()

        self.use_edge_features = use_edge_features

        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, hidden_dim3)

        # Dropout layers for GCN
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        # After global pooling: mean + max concatenation doubles the dimension
        self.fc1 = nn.Linear(hidden_dim3 * 2, fc_dim1)
        self.fc2 = nn.Linear(fc_dim1, fc_dim2)
        self.fc3 = nn.Linear(fc_dim2, output_dim)

        # FC dropout
        self.fc_dropout = nn.Dropout(fc_dropout_rate)

        # Batch normalization layers (optional, for stability)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes] (for batched graphs)
            edge_attr: Edge features [num_edges, edge_dim] (optional, not used yet)

        Returns:
            Output predictions [batch_size, output_dim]
        """
        # Graph convolution layers with ReLU and dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        # No dropout after last conv layer (following paper)

        # Global pooling: concatenate mean and max pooling
        if batch is None:
            # Single graph case
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_dropout(x)

        x = self.fc2(x)
        x = F.relu(x)

        # Output layer (no activation - will use BCEWithLogitsLoss)
        x = self.fc3(x)

        return x

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get probability predictions (with sigmoid activation).

        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch assignment vector

        Returns:
            Probability predictions [batch_size, 1]
        """
        logits = self.forward(x, edge_index, batch)
        return torch.sigmoid(logits)


class CornerGNNWithAttention(CornerGNN):
    """
    Enhanced GNN with attention mechanism for corner prediction.

    Uses GATConv instead of GCNConv for attention-based aggregation.
    """

    def __init__(self, *args, num_heads: int = 4, **kwargs):
        """
        Initialize GNN with attention.

        Args:
            num_heads: Number of attention heads for GAT layers
            *args, **kwargs: Arguments passed to parent class
        """
        super().__init__(*args, **kwargs)

        from torch_geometric.nn import GATConv

        # Replace GCN layers with GAT layers
        self.conv1 = GATConv(self.conv1.in_channels,
                             self.conv1.out_channels // num_heads,
                             heads=num_heads, dropout=0.3)
        self.conv2 = GATConv(self.conv1.out_channels,
                             self.conv2.out_channels // num_heads,
                             heads=num_heads, dropout=0.3)
        self.conv3 = GATConv(self.conv2.out_channels,
                             self.conv3.out_channels,
                             heads=1, concat=False, dropout=0.3)


class CornerGNNWithEdgeFeatures(CornerGNN):
    """
    Enhanced GNN that uses 6-dimensional edge features for improved tactical understanding.

    Edge features include:
    - Distance between players (normalized)
    - Relative velocity (vx, vy, magnitude)
    - Angle between players (sin, cos)

    Uses NNConv (Neural Network Convolution) which learns edge-specific transformations.
    """

    def __init__(self, *args, edge_dim: int = 6, **kwargs):
        """
        Initialize GNN with edge feature support.

        Args:
            edge_dim: Dimension of edge features (default: 6)
            *args, **kwargs: Arguments passed to parent class
        """
        # Initialize parent WITHOUT edge features flag
        # We need to override the layers anyway
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('use_edge_features', None)
        super().__init__(*args, **kwargs_copy)

        self.edge_dim = edge_dim
        self.use_edge_features = True

        # Get dimensions from parent initialization
        input_dim = args[0] if args else kwargs.get('input_dim', 14)
        hidden_dim1 = args[1] if len(args) > 1 else kwargs.get('hidden_dim1', 64)
        hidden_dim2 = args[2] if len(args) > 2 else kwargs.get('hidden_dim2', 128)
        hidden_dim3 = args[3] if len(args) > 3 else kwargs.get('hidden_dim3', 64)

        # Replace GCN layers with NNConv that supports edge features
        # NNConv: learns a neural network for each edge type

        # Edge network for first layer: maps edge features to weight matrix
        edge_nn1 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim1 * input_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim1 * input_dim, hidden_dim1 * input_dim)
        )
        self.conv1 = NNConv(input_dim, hidden_dim1, edge_nn1, aggr='mean')

        # Edge network for second layer
        edge_nn2 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim2 * hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim2 * hidden_dim1, hidden_dim2 * hidden_dim1)
        )
        self.conv2 = NNConv(hidden_dim1, hidden_dim2, edge_nn2, aggr='mean')

        # Edge network for third layer
        edge_nn3 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim3 * hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim3 * hidden_dim2, hidden_dim3 * hidden_dim2)
        )
        self.conv3 = NNConv(hidden_dim2, hidden_dim3, edge_nn3, aggr='mean')

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with edge features.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]
            edge_attr: Edge features [num_edges, edge_dim] (REQUIRED for this model)

        Returns:
            Output predictions [batch_size, output_dim]
        """
        if edge_attr is None:
            raise ValueError("CornerGNNWithEdgeFeatures requires edge_attr to be provided")

        # Graph convolution layers with edge features
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        # Global pooling: concatenate mean and max pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_dropout(x)

        x = self.fc2(x)
        x = F.relu(x)

        # Output layer
        x = self.fc3(x)

        return x


def create_model(model_type: str = "gcn", use_edge_features: bool = False, **kwargs) -> CornerGNN:
    """
    Factory function to create different GNN model variants.

    Args:
        model_type: Type of model ("gcn", "gat", or "gcn_edge")
        use_edge_features: If True and model_type="gcn", use CornerGNNWithEdgeFeatures
        **kwargs: Model hyperparameters

    Returns:
        Instantiated model
    """
    if model_type == "gcn":
        if use_edge_features:
            return CornerGNNWithEdgeFeatures(**kwargs)
        else:
            return CornerGNN(**kwargs)
    elif model_type == "gcn_edge":
        # Explicit edge-aware GCN model
        return CornerGNNWithEdgeFeatures(**kwargs)
    elif model_type == "gat":
        return CornerGNNWithAttention(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: gcn, gcn_edge, gat")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model with dummy data
    print("Testing CornerGNN model...")

    # Create dummy data
    num_nodes = 22  # 11 vs 11 players
    num_edges = 50  # Some connections
    batch_size = 4

    # Node features (14-dimensional)
    x = torch.randn(num_nodes * batch_size, 14)

    # Edge indices (random connections)
    edge_index = torch.randint(0, num_nodes * batch_size, (2, num_edges * batch_size))

    # Batch assignment
    batch = torch.cat([torch.full((num_nodes,), i) for i in range(batch_size)])

    # Create and test model
    model = create_model("gcn")
    print(f"Model has {count_parameters(model):,} trainable parameters")

    # Forward pass
    output = model(x, edge_index, batch)
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")

    # Get probabilities
    probs = model.predict_proba(x, edge_index, batch)
    print(f"Probabilities: {probs.squeeze()}")

    # Test GAT variant
    print("\nTesting CornerGNNWithAttention model...")
    model_gat = create_model("gat")
    print(f"GAT model has {count_parameters(model_gat):,} trainable parameters")

    output_gat = model_gat(x, edge_index, batch)
    print(f"GAT output shape: {output_gat.shape}")

    print("\n✅ Model tests passed!")