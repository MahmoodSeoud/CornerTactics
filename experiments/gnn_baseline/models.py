"""GNN models for corner kick outcome prediction.

Implements three architectures:
- GAT (Graph Attention Network)
- GraphSAGE
- MPNN (Message Passing Neural Network)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    SAGEConv,
    NNConv,
    global_mean_pool,
    global_max_pool,
)


class GATModel(nn.Module):
    """Graph Attention Network for graph-level binary classification.

    Uses multi-head attention to aggregate neighbor information,
    followed by global pooling and a classification head.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize GAT model.

        Args:
            in_channels: Number of input node features
            hidden_channels: Hidden layer dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                )
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * heads * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            Predictions [batch_size, 1] in range [0, 1]
        """
        # Apply GAT layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling (concatenate mean and max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Classification
        x = self.classifier(x)
        return torch.sigmoid(x)


class GraphSAGEModel(nn.Module):
    """GraphSAGE model for graph-level binary classification.

    Uses sampling and aggregating neighbor features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize GraphSAGE model.

        Args:
            in_channels: Number of input node features
            hidden_channels: Hidden layer dimension
            num_layers: Number of SAGE layers
            dropout: Dropout probability
        """
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            Predictions [batch_size, 1] in range [0, 1]
        """
        # Apply SAGE layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Classification
        x = self.classifier(x)
        return torch.sigmoid(x)


class MPNNModel(nn.Module):
    """Message Passing Neural Network for graph-level binary classification.

    Uses edge features in message passing via NNConv.
    """

    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize MPNN model.

        Args:
            in_channels: Number of input node features
            edge_channels: Number of edge features
            hidden_channels: Hidden layer dimension
            num_layers: Number of message passing layers
            dropout: Dropout probability
        """
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.edge_nns = nn.ModuleList()

        # First layer
        edge_nn = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels * hidden_channels),
        )
        self.edge_nns.append(edge_nn)
        self.convs.append(NNConv(in_channels, hidden_channels, edge_nn, aggr='mean'))

        # Hidden layers
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(edge_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels * hidden_channels),
            )
            self.edge_nns.append(edge_nn)
            self.convs.append(
                NNConv(hidden_channels, hidden_channels, edge_nn, aggr='mean')
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_channels]
            batch: Batch assignment vector [num_nodes]

        Returns:
            Predictions [batch_size, 1] in range [0, 1]
        """
        # Apply MPNN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Classification
        x = self.classifier(x)
        return torch.sigmoid(x)


def create_model(
    name: str,
    in_channels: int,
    edge_channels: Optional[int] = None,
    hidden_channels: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """Factory function to create GNN models.

    Args:
        name: Model name ('gat', 'graphsage', 'mpnn')
        in_channels: Number of input node features
        edge_channels: Number of edge features (required for MPNN)
        hidden_channels: Hidden layer dimension
        num_layers: Number of layers
        dropout: Dropout probability
        **kwargs: Additional model-specific arguments

    Returns:
        GNN model instance

    Raises:
        ValueError: If model name is not recognized
    """
    name = name.lower()

    if name == 'gat':
        return GATModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )
    elif name == 'graphsage':
        return GraphSAGEModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif name == 'mpnn':
        if edge_channels is None:
            raise ValueError("edge_channels is required for MPNN")
        return MPNNModel(
            in_channels=in_channels,
            edge_channels=edge_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown model: {name}. Available: 'gat', 'graphsage', 'mpnn'"
        )
