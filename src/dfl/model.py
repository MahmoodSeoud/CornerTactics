"""Spatio-Temporal Graph Neural Network for corner kick outcome prediction.

This module implements the ST-GNN architecture following TacticAI design principles:
- SpatialGNN: GATv2Conv layers to process single frame's graph
- TemporalAggregator: GRU to aggregate frame-level representations over time
- CornerKickPredictor: Full model with multi-head outputs for different prediction tasks
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from typing import List, Dict

from torch_geometric.data import Data


class SpatialGNN(nn.Module):
    """Process a single frame's graph using GATv2 convolutions.

    Takes a graph representation of a single tracking frame (22 players + 1 ball)
    and outputs a fixed-size graph-level embedding.

    Attributes:
        in_channels: Number of input node features (default: 8)
        hidden_channels: Hidden layer dimension (default: 64)
        out_channels: Output embedding dimension (default: 32)
        heads: Number of attention heads in GATv2Conv (default: 4)
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        out_channels: int = 32,
        heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, heads=heads, concat=False)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through spatial GNN.

        Args:
            x: Node feature tensor (num_nodes, in_channels)
            edge_index: Edge connectivity tensor (2, num_edges)
            batch: Batch assignment tensor for batched graphs (num_nodes,)

        Returns:
            Graph-level embedding tensor:
                - (1, out_channels) for single graph
                - (batch_size, out_channels) for batched graphs
        """
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = torch.relu(x)

        # Pool to get graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # Single graph: mean over all nodes
            x = x.mean(dim=0, keepdim=True)

        return x


class TemporalAggregator(nn.Module):
    """Aggregate frame-level representations over time using GRU.

    Takes a sequence of frame embeddings and outputs a single temporal embedding.

    Attributes:
        input_dim: Dimension of frame embeddings (default: 32)
        hidden_dim: GRU hidden state dimension (default: 64)
        num_layers: Number of GRU layers (default: 2)
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal aggregator.

        Args:
            frame_embeddings: Tensor of shape (batch, seq_len, input_dim)

        Returns:
            Final hidden state tensor of shape (batch, hidden_dim)
        """
        output, hidden = self.gru(frame_embeddings)
        # Return last layer's final hidden state
        return hidden[-1]


class CornerKickPredictor(nn.Module):
    """Full ST-GNN for corner kick outcome prediction.

    Combines SpatialGNN for per-frame processing with TemporalAggregator for
    sequence modeling, then applies multi-head prediction layers.

    Multi-head outputs:
        - shot: Binary prediction (sigmoid) - will there be a shot?
        - goal: Binary prediction (sigmoid) - will there be a goal?
        - contact: 2-class logits - first contact team (attacking/defending)
        - outcome: N-class logits - detailed outcome category

    Attributes:
        node_features: Input node feature dimension (default: 8)
        gnn_hidden: SpatialGNN hidden dimension (default: 64)
        gnn_out: SpatialGNN output dimension (default: 32)
        temporal_hidden: TemporalAggregator hidden dimension (default: 64)
        num_classes_outcome: Number of outcome classes (default: 6)
    """

    def __init__(
        self,
        node_features: int = 8,
        gnn_hidden: int = 64,
        gnn_out: int = 32,
        temporal_hidden: int = 64,
        num_classes_outcome: int = 6,
    ):
        super().__init__()
        self.spatial_gnn = SpatialGNN(
            in_channels=node_features,
            hidden_channels=gnn_hidden,
            out_channels=gnn_out,
        )
        self.temporal = TemporalAggregator(
            input_dim=gnn_out,
            hidden_dim=temporal_hidden,
        )

        # Multi-head prediction layers
        self.head_shot = nn.Linear(temporal_hidden, 1)
        self.head_goal = nn.Linear(temporal_hidden, 1)
        self.head_contact = nn.Linear(temporal_hidden, 2)
        self.head_outcome = nn.Linear(temporal_hidden, num_classes_outcome)

    def forward(self, graph_sequences: List[List[Data]]) -> Dict[str, torch.Tensor]:
        """Forward pass through full ST-GNN.

        Args:
            graph_sequences: List of graph sequences, where each sequence is a list
                of PyTorch Geometric Data objects representing frames.
                graph_sequences[i] = [graph_t0, graph_t1, ..., graph_tN] for corner i

        Returns:
            Dict with predictions for each head:
                - 'shot': (batch_size, 1) sigmoid probabilities
                - 'goal': (batch_size, 1) sigmoid probabilities
                - 'contact': (batch_size, 2) logits for attacking/defending
                - 'outcome': (batch_size, num_classes) logits for outcome classes
        """
        batch_embeddings = []

        for seq in graph_sequences:
            frame_embs = []
            for graph in seq:
                # Process each frame through spatial GNN
                emb = self.spatial_gnn(graph.x, graph.edge_index)
                frame_embs.append(emb.squeeze(0))

            # Stack frames: (seq_len, gnn_out)
            frame_embs = torch.stack(frame_embs, dim=0)
            batch_embeddings.append(frame_embs)

        # Pad sequences to same length
        max_len = max(emb.shape[0] for emb in batch_embeddings)
        gnn_out_dim = batch_embeddings[0].shape[-1]

        # Get the device from input data
        device = batch_embeddings[0].device

        padded = torch.zeros(
            len(batch_embeddings), max_len, gnn_out_dim, device=device
        )
        for i, emb in enumerate(batch_embeddings):
            padded[i, : emb.shape[0]] = emb

        # Temporal aggregation
        temporal_out = self.temporal(padded)

        return {
            "shot": torch.sigmoid(self.head_shot(temporal_out)),
            "goal": torch.sigmoid(self.head_goal(temporal_out)),
            "contact": self.head_contact(temporal_out),
            "outcome": self.head_outcome(temporal_out),
        }
