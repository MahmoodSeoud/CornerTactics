"""GNN backbone for corner kick prediction.

Supports two modes:
    - "pretrained": Loads USSF CrystalConv weights with feature projection layers.
      Backbone params frozen; projection layers trainable.
    - "scratch": Lightweight CrystalConv from scratch (smaller hidden dim).

The backbone produces per-node embeddings suitable for both Stage 1
(receiver prediction) and Stage 2 (shot prediction via pooling).
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv

logger = logging.getLogger(__name__)

# USSF pretrained backbone dimensions
_USSF_NODE_DIM = 12
_USSF_EDGE_DIM = 6
_USSF_HIDDEN = 128
_USSF_NUM_LAYERS = 3

# From-scratch defaults (smaller to prevent overfitting on ~80 samples)
_SCRATCH_HIDDEN = 64
_SCRATCH_NUM_LAYERS = 3


class CornerBackbone(nn.Module):
    """CrystalConv GNN backbone for corner kick graphs.

    Args:
        mode: "pretrained" (load USSF weights) or "scratch" (train from zero).
        node_features: Input node feature dimension (14 = 13 graph features + 1 receiver indicator).
        edge_features: Input edge feature dimension (4 from build_graphs).
        pretrained_path: Path to USSF backbone weights (.pt file).
        hidden_channels: Hidden dimension. 128 for pretrained, 64 for scratch.
        num_conv_layers: Number of CGConv layers (default 3).
        freeze: Whether to freeze backbone conv layers (only applies to pretrained).
    """

    def __init__(
        self,
        mode: str = "pretrained",
        node_features: int = 14,
        edge_features: int = 4,
        pretrained_path: Optional[str] = None,
        hidden_channels: Optional[int] = None,
        num_conv_layers: int = 3,
        freeze: bool = True,
    ):
        super().__init__()

        if mode not in ("pretrained", "scratch"):
            raise ValueError(f"Unknown mode: {mode!r}. Use 'pretrained' or 'scratch'.")

        self.mode = mode
        self.node_features = node_features
        self.edge_features = edge_features
        self.freeze = freeze

        if mode == "pretrained":
            self._hidden_channels = hidden_channels or _USSF_HIDDEN
            self._init_pretrained(node_features, edge_features, num_conv_layers)
            if pretrained_path is not None:
                self.load_pretrained(pretrained_path)
            if freeze:
                self._freeze_backbone()
        else:
            self._hidden_channels = hidden_channels or _SCRATCH_HIDDEN
            self._init_scratch(node_features, edge_features, num_conv_layers)

    @property
    def output_dim(self) -> int:
        """Output dimension of per-node embeddings."""
        return self._hidden_channels

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_pretrained(self, node_features: int, edge_features: int, num_conv_layers: int):
        """Set up pretrained-mode layers with feature projections."""
        # Projection layers: map corner features → USSF expected dims
        self.node_proj = nn.Linear(node_features, _USSF_NODE_DIM)
        self.edge_proj = nn.Linear(edge_features, _USSF_EDGE_DIM)

        # CGConv backbone (architecture matches USSF exactly)
        self.conv1 = CGConv(channels=_USSF_NODE_DIM, dim=_USSF_EDGE_DIM, aggr="add")
        self.lin_in = nn.Linear(_USSF_NODE_DIM, self._hidden_channels)
        self.convs = nn.ModuleList([
            CGConv(channels=self._hidden_channels, dim=_USSF_EDGE_DIM, aggr="add")
            for _ in range(num_conv_layers - 1)
        ])

    def _init_scratch(self, node_features: int, edge_features: int, num_conv_layers: int):
        """Set up from-scratch layers (no projection needed)."""
        self.node_proj = None
        self.edge_proj = None

        self.conv1 = CGConv(channels=node_features, dim=edge_features, aggr="add")
        self.lin_in = nn.Linear(node_features, self._hidden_channels)
        self.convs = nn.ModuleList([
            CGConv(channels=self._hidden_channels, dim=edge_features, aggr="add")
            for _ in range(num_conv_layers - 1)
        ])

    # ------------------------------------------------------------------
    # Weight loading / freezing
    # ------------------------------------------------------------------

    def load_pretrained(self, path: str):
        """Load USSF backbone weights from a saved state dict.

        Expected format (from CounterattackGNN.get_backbone_state()):
            {"conv1": dict, "lin_in": dict, "convs": [dict, ...], "config": dict}
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=True)
        self.conv1.load_state_dict(state["conv1"])
        self.lin_in.load_state_dict(state["lin_in"])
        for i, conv_state in enumerate(state["convs"]):
            self.convs[i].load_state_dict(conv_state)

        logger.info("Loaded pretrained backbone from %s", path)

    def _freeze_backbone(self):
        """Freeze conv and lin_in parameters (not projection layers)."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.lin_in.parameters():
            param.requires_grad = False
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute per-node embeddings.

        Args:
            x: Node features [N, node_features].
            edge_index: Edge indices [2, E].
            edge_attr: Edge features [E, edge_features].

        Returns:
            Node embeddings [N, hidden_channels].
        """
        # Project features if in pretrained mode
        if self.node_proj is not None:
            x = self.node_proj(x)
        if self.edge_proj is not None:
            edge_attr = self.edge_proj(edge_attr)

        # First conv layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.lin_in(x)

        # Remaining conv layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        return x
