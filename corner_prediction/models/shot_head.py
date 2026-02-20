"""Stage 2: Conditional shot prediction head.

Predicts whether a corner kick results in a shot, conditioned on
graph-level features (and optionally on receiver identity via the backbone).
"""

import torch
import torch.nn as nn


class ShotHead(nn.Module):
    """Graph-level binary classification head for shot prediction.

    Takes a graph-level embedding (from pooling node embeddings) and
    optional graph-level features (e.g. corner_side), and produces
    a shot logit.

    Args:
        input_dim: Backbone output dimension after pooling (e.g. 128).
        graph_feature_dim: Number of graph-level features (default 1 for corner_side).
            Set to 0 to skip graph features.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 128,
        graph_feature_dim: int = 1,
        hidden_dim: int = 32,
        dropout: float = 0.3,
        linear_only: bool = False,
    ):
        super().__init__()
        self.graph_feature_dim = graph_feature_dim
        total_input = input_dim + graph_feature_dim
        if linear_only:
            self.mlp = nn.Linear(total_input, 1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(total_input, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        graph_embedding: torch.Tensor,
        graph_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute shot logit.

        Args:
            graph_embedding: Pooled graph embedding [B, input_dim].
            graph_features: Optional graph-level features [B, graph_feature_dim].

        Returns:
            Shot logit [B, 1].
        """
        if graph_features is not None and self.graph_feature_dim > 0:
            x = torch.cat([graph_embedding, graph_features], dim=-1)
        else:
            x = graph_embedding
        return self.mlp(x)
