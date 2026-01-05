"""Attention Visualization for GAT models.

Extracts and visualizes attention weights from Graph Attention Networks
to understand which player relationships the model focuses on.

Usage:
    1. Train a GAT model on corner kick data
    2. Extract attention weights using AttentionExtractor
    3. Visualize attention on a pitch diagram
    4. Compare attention patterns between shot and no-shot corners
"""

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from torch_geometric.data import Batch


class AttentionExtractor:
    """Extract attention weights from a trained GAT model.

    Uses forward hooks to capture attention coefficients from GATConv layers.
    Note: GATConv in PyG doesn't always expose attention weights directly.
    This extractor provides a best-effort approach.

    Args:
        model: A trained GAT model
        device: Device to use for inference
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

        # Storage for attention weights
        self._attention_weights: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        self._gat_layers: List[Tuple[str, nn.Module]] = []

        self._find_gat_layers()
        self._setup_hooks()

    def _find_gat_layers(self):
        """Find all GATConv layers in the model."""
        for name, module in self.model.named_modules():
            if 'GATConv' in module.__class__.__name__:
                self._gat_layers.append((name, module))

    def _setup_hooks(self):
        """Register forward hooks on GATConv layers."""
        for name, module in self._gat_layers:
            hook = module.register_forward_hook(
                self._make_hook(name)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        """Create a hook that captures attention weights."""
        def hook(module, input, output):
            # GATConv returns (output, (edge_index, attention_weights)) when return_attention_weights=True
            # For standard forward, we try to access internal alpha
            # or create synthetic attention based on output

            if hasattr(module, '_alpha') and module._alpha is not None:
                self._attention_weights[layer_name] = module._alpha.detach().cpu()
            elif isinstance(output, tuple) and len(output) == 2:
                # If model was called with return_attention_weights=True
                # output is (node_features, (edge_index, attention_weights))
                _, attention_tuple = output
                if isinstance(attention_tuple, tuple) and len(attention_tuple) == 2:
                    _, attention_weights = attention_tuple
                    self._attention_weights[layer_name] = attention_weights.detach().cpu()

        return hook

    def extract(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Extract attention weights for a batch.

        Args:
            batch: PyTorch Geometric batch of graphs

        Returns:
            Dictionary mapping layer names to attention weight tensors
            Each tensor has shape [num_edges, num_heads]
            Note: May be empty if attention weights not accessible
        """
        self._attention_weights.clear()
        batch = batch.to(self.device)

        with torch.no_grad():
            # Forward pass to trigger hooks
            _ = self.model(batch.x, batch.edge_index, batch.batch)

        return dict(self._attention_weights)

    def extract_with_attention(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract attention by running forward with return_attention_weights.

        This is a more reliable method but requires modifying forward calls.

        Args:
            batch: PyTorch Geometric batch of graphs

        Returns:
            Tuple of (model output, attention weights dict)
        """
        batch = batch.to(self.device)
        attention_dict = {}

        with torch.no_grad():
            x = batch.x
            # Run through each GAT layer manually
            for i, (name, conv) in enumerate(self._gat_layers):
                x, attention = conv(x, batch.edge_index, return_attention_weights=True)
                attention_dict[name] = attention[1].detach().cpu()
                x = torch.nn.functional.elu(x)
                x = torch.nn.functional.dropout(x, p=self.model.dropout, training=False)

        return x, attention_dict

    def __del__(self):
        """Clean up hooks."""
        for hook in self._hooks:
            hook.remove()


def aggregate_attention_per_node(
    edge_index: torch.Tensor,
    attention: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Aggregate attention weights to per-node importance scores.

    Args:
        edge_index: Edge connectivity [2, num_edges]
        attention: Attention weights [num_edges, num_heads]
        num_nodes: Number of nodes in the graph

    Returns:
        Node importance scores [num_nodes]
    """
    # Average attention across heads
    edge_attention = attention.mean(dim=1)  # [num_edges]

    # Sum incoming attention for each node
    target_nodes = edge_index[1]  # Target nodes receive attention
    node_importance = torch.zeros(num_nodes)

    for i, target in enumerate(target_nodes):
        node_importance[target] += edge_attention[i]

    # Normalize by number of incoming edges
    incoming_count = torch.bincount(target_nodes, minlength=num_nodes).float()
    incoming_count = incoming_count.clamp(min=1)  # Avoid division by zero
    node_importance = node_importance / incoming_count

    return node_importance


def aggregate_attention_per_edge(
    attention: torch.Tensor,
) -> torch.Tensor:
    """Aggregate attention across heads for each edge.

    Args:
        attention: Attention weights [num_edges, num_heads]

    Returns:
        Edge importance scores [num_edges]
    """
    return attention.mean(dim=1)


def create_attention_plot_data(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    attention_weights: torch.Tensor,
    team_indicators: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Create data suitable for attention visualization.

    Args:
        positions: Player positions [num_nodes, 2]
        edge_index: Edge connectivity [2, num_edges]
        attention_weights: Attention [num_edges, num_heads]
        team_indicators: Team flags [num_nodes] (1=attacker, 0=defender)

    Returns:
        Dictionary with plot data
    """
    edge_weights = aggregate_attention_per_edge(attention_weights)

    return {
        'node_positions': positions.numpy(),
        'edge_index': edge_index.numpy(),
        'edge_weights': edge_weights.numpy(),
        'team_colors': team_indicators.numpy(),
    }


def plot_attention_on_pitch(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    attention_weights: torch.Tensor,
    team_indicators: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    title: str = 'Attention Weights',
    show_edges: bool = True,
    edge_threshold: float = 0.0,
) -> plt.Figure:
    """Visualize attention weights on a pitch diagram.

    Args:
        positions: Player positions [num_nodes, 2]
        edge_index: Edge connectivity [2, num_edges]
        attention_weights: Attention [num_edges, num_heads]
        team_indicators: Team flags [num_nodes]
        ax: Matplotlib axes (creates new figure if None)
        title: Plot title
        show_edges: Whether to show attention edges
        edge_threshold: Minimum attention to show edge

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    # Draw pitch outline (StatsBomb coordinates: 120 x 80)
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, 85)

    # Pitch markings
    ax.plot([0, 0, 120, 120, 0], [0, 80, 80, 0, 0], 'k-', lw=2)  # Boundary
    ax.plot([60, 60], [0, 80], 'k-', lw=1)  # Halfway line

    # Goal areas (StatsBomb uses attacking direction right)
    ax.plot([120, 120, 102, 102], [18, 62, 62, 18], 'k-', lw=1)  # Penalty area
    ax.plot([120, 120, 114, 114], [30, 50, 50, 30], 'k-', lw=1)  # Goal area

    # Convert to numpy
    pos = positions.numpy()
    teams = team_indicators.numpy()
    edge_idx = edge_index.numpy()
    edge_weights = aggregate_attention_per_edge(attention_weights).numpy()

    # Normalize edge weights for visualization
    if edge_weights.max() > edge_weights.min():
        edge_weights_norm = (edge_weights - edge_weights.min()) / (
            edge_weights.max() - edge_weights.min()
        )
    else:
        edge_weights_norm = np.ones_like(edge_weights) * 0.5

    # Draw edges with attention weights
    if show_edges:
        for i in range(edge_idx.shape[1]):
            if edge_weights_norm[i] < edge_threshold:
                continue

            src, tgt = edge_idx[0, i], edge_idx[1, i]
            alpha = 0.1 + 0.9 * edge_weights_norm[i]
            width = 0.5 + 2 * edge_weights_norm[i]

            ax.plot(
                [pos[src, 0], pos[tgt, 0]],
                [pos[src, 1], pos[tgt, 1]],
                'gray',
                alpha=alpha,
                linewidth=width,
            )

    # Draw players
    attacker_color = 'red'
    defender_color = 'blue'

    for i in range(len(pos)):
        color = attacker_color if teams[i] > 0.5 else defender_color
        ax.scatter(pos[i, 0], pos[i, 1], c=color, s=200, zorder=5, edgecolors='black')
        ax.annotate(str(i), (pos[i, 0], pos[i, 1]), ha='center', va='center',
                   fontsize=8, color='white', fontweight='bold', zorder=6)

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel('x (attacking direction →)')
    ax.set_ylabel('y')

    # Legend
    ax.scatter([], [], c=attacker_color, s=100, label='Attacker')
    ax.scatter([], [], c=defender_color, s=100, label='Defender')
    ax.legend(loc='upper left')

    return fig


def compare_attention_patterns(
    model: nn.Module,
    shot_batch: Batch,
    no_shot_batch: Batch,
) -> Dict[str, torch.Tensor]:
    """Compare attention patterns between shot and no-shot corners.

    Args:
        model: Trained GAT model
        shot_batch: Batch of shot corners (y=1)
        no_shot_batch: Batch of no-shot corners (y=0)

    Returns:
        Dictionary with mean attention for each class and difference
    """
    extractor = AttentionExtractor(model)

    # Extract attention for shots
    shot_attention = extractor.extract(shot_batch)

    # Extract attention for no-shots
    no_shot_attention = extractor.extract(no_shot_batch)

    # Compute mean attention across all layers
    def mean_attention(att_dict):
        if not att_dict:
            return torch.tensor(0.0)
        all_att = [att.mean() for att in att_dict.values()]
        return torch.stack(all_att).mean()

    shot_mean = mean_attention(shot_attention)
    no_shot_mean = mean_attention(no_shot_attention)

    return {
        'shot_mean_attention': shot_mean,
        'no_shot_mean_attention': no_shot_mean,
        'attention_difference': shot_mean - no_shot_mean,
        'shot_attention_by_layer': shot_attention,
        'no_shot_attention_by_layer': no_shot_attention,
    }
