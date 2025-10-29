#!/usr/bin/env python3
"""
GATv2 Encoder for TacticAI-Style Corner Prediction

Implements Day 10-11: GATv2 Encoder Implementation
- GATv2Encoder: 3-layer Graph Attention Network v2 with batch normalization
- D2GATv2: GATv2 with D2 frame averaging for geometric invariance

Architecture (based on TacticAI Implementation Plan):
- TacticAI: 4 layers, 8 heads, 4-dim latent (~50k params)
- Ours: 3 layers, 4 heads, 16-dim latent (~25-30k params)
- Rationale: Reduced capacity for smaller dataset, wider features for missing velocities

Key Features:
- ELU activations (not ReLU, as per TacticAI paper)
- Batch normalization after each layer
- Dropout 0.4 for regularization
- D2 frame averaging: Generate 4 D2 views, encode each, average node embeddings

Author: mseo
Date: October 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from typing import Tuple
import sys
from pathlib import Path

# Import D2Augmentation
sys.path.append(str(Path(__file__).parent.parent))
from src.data.augmentation import D2Augmentation


class GATv2Encoder(nn.Module):
    """
    3-layer GATv2 encoder for corner kick graphs.

    Architecture:
    - Layer 1: GATv2Conv(14, hidden_dim, heads=num_heads, dropout=0.4)
    - Layer 2: GATv2Conv(hidden_dim*heads, hidden_dim, heads=num_heads, dropout=0.4)
    - Layer 3: GATv2Conv(hidden_dim*heads, hidden_dim, heads=1, concat=False, dropout=0.4)
    - Batch normalization after each layer
    - ELU activations

    Output:
    - graph_emb: Global mean-pooled graph embeddings [batch_size, hidden_dim]
    - node_emb: Per-node embeddings [num_nodes, hidden_dim]
    """

    def __init__(
        self,
        in_channels: int = 14,
        hidden_dim: int = 24,  # 24-dim gives ~27k params (target: 25-35k)
        num_heads: int = 4,
        dropout: float = 0.4
    ):
        """
        Initialize GATv2 encoder.

        Args:
            in_channels: Input feature dimension (default 14)
            hidden_dim: Hidden dimension for node embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Layer 1: 14 → hidden_dim (with multi-head attention)
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate attention heads
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)

        # Layer 2: hidden_dim*heads → hidden_dim (with multi-head attention)
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)

        # Layer 3: hidden_dim*heads → hidden_dim (single head, no concat)
        self.conv3 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False  # Average attention heads (single head)
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Activation function (ELU, not ReLU)
        self.elu = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GATv2 encoder.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes] indicating graph membership

        Returns:
            graph_emb: Graph-level embeddings [batch_size, hidden_dim]
            node_emb: Node-level embeddings [num_nodes, hidden_dim]
        """
        # Layer 1
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = self.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = self.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 3
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = self.elu(h)

        # Node embeddings (final layer output)
        node_emb = h

        # Graph embeddings (global mean pooling)
        graph_emb = global_mean_pool(h, batch)

        return graph_emb, node_emb


class D2GATv2(nn.Module):
    """
    GATv2 encoder with D2 frame averaging.

    Implements geometric invariance by:
    1. Generating 4 D2 views (identity, h-flip, v-flip, both-flip)
    2. Encoding each view through GATv2Encoder
    3. Averaging node embeddings across views
    4. Global mean pooling for graph embeddings

    This provides robustness to spatial transformations and acts
    as data augmentation during inference.
    """

    def __init__(
        self,
        in_channels: int = 14,
        hidden_dim: int = 24,  # 24-dim gives ~27k params (target: 25-35k)
        num_heads: int = 4,
        dropout: float = 0.4
    ):
        """
        Initialize D2GATv2 encoder.

        Args:
            in_channels: Input feature dimension (default 14)
            hidden_dim: Hidden dimension for node embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Base GATv2 encoder (shared across all D2 views)
        self.encoder = GATv2Encoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # D2 augmentation module
        self.augmenter = D2Augmentation()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through D2GATv2 encoder with frame averaging.

        Args:
            x: Node features [num_nodes, in_channels]
               Expected format: [x_pos, y_pos, ..., vx, vy, ...]
               Positions at columns 0-1, velocities at columns 4-5
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes] indicating graph membership

        Returns:
            graph_emb: Graph-level embeddings [batch_size, hidden_dim]
            node_emb: Averaged node-level embeddings [num_nodes, hidden_dim]
        """
        batch_size = batch.max().item() + 1
        num_nodes = x.size(0)

        # Generate 4 D2 views of the full feature tensor
        views = self.augmenter.get_all_views(x, edge_index)

        # Encode each view and collect node embeddings
        node_embeddings_list = []

        for x_view, edge_view in views:
            # Encode this view
            _, node_emb_view = self.encoder(x_view, edge_view, batch)

            node_embeddings_list.append(node_emb_view)

        # Average node embeddings across all 4 D2 views
        node_emb_stacked = torch.stack(node_embeddings_list, dim=0)  # [4, num_nodes, hidden_dim]
        node_emb = node_emb_stacked.mean(dim=0)  # [num_nodes, hidden_dim]

        # Global mean pooling for graph embeddings
        graph_emb = global_mean_pool(node_emb, batch)  # [batch_size, hidden_dim]

        return graph_emb, node_emb


if __name__ == "__main__":
    # Test GATv2 encoder
    print("="*60)
    print("TESTING GATV2 ENCODER")
    print("="*60)

    # Create dummy batch
    batch_size = 4
    num_nodes_per_graph = 22
    num_nodes = batch_size * num_nodes_per_graph
    num_features = 14

    x = torch.randn(num_nodes, num_features)
    # Set valid positions
    x[:, 0] = torch.rand(num_nodes) * 120  # x in [0, 120]
    x[:, 1] = torch.rand(num_nodes) * 80   # y in [0, 80]

    # Edge index (fully connected within each graph)
    edges = []
    for i in range(batch_size):
        offset = i * num_nodes_per_graph
        for src in range(num_nodes_per_graph):
            for dst in range(num_nodes_per_graph):
                if src != dst:
                    edges.append([offset + src, offset + dst])

    edge_index = torch.tensor(edges).t()

    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes_per_graph)

    print(f"\nInput:")
    print(f"  x shape: {x.shape}")
    print(f"  edge_index shape: {edge_index.shape}")
    print(f"  batch size: {batch_size}")

    # Test GATv2Encoder
    print("\n1. Testing GATv2Encoder...")
    model = GATv2Encoder(in_channels=14, hidden_dim=24, num_heads=4, dropout=0.4)
    graph_emb, node_emb = model(x, edge_index, batch)

    print(f"  Graph embeddings: {graph_emb.shape}")
    print(f"  Node embeddings: {node_emb.shape}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameter count: {num_params:,}")

    # Test D2GATv2
    print("\n2. Testing D2GATv2...")
    model_d2 = D2GATv2(in_channels=14, hidden_dim=24, num_heads=4, dropout=0.4)
    graph_emb_d2, node_emb_d2 = model_d2(x, edge_index, batch)

    print(f"  Graph embeddings: {graph_emb_d2.shape}")
    print(f"  Node embeddings: {node_emb_d2.shape}")

    num_params_d2 = sum(p.numel() for p in model_d2.parameters())
    print(f"  Parameter count: {num_params_d2:,}")

    print("\n" + "="*60)
    print("✅ GATV2 ENCODER IMPLEMENTED!")
    print("="*60)
