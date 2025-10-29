#!/usr/bin/env python3
"""
Visual Demo of D2 Augmentation

Displays all 4 D2 views of a corner kick graph on a soccer pitch.
Saves the visualization to data/results/d2_augmentation_demo.png
"""

import pickle
import torch
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.augmentation import D2Augmentation


# StatsBomb pitch dimensions
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def load_sample_graph(graph_path: str, sample_idx: int = 0):
    """
    Load a sample graph from the graph dataset.

    Args:
        graph_path: Path to pickled graph file
        sample_idx: Index of graph to load

    Returns:
        Tuple of (x, edge_index, metadata)
    """
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)

    # Get a single graph
    graph = graphs[sample_idx]

    # Convert to tensors if not already
    # CornerGraph dataclass has node_features, not x
    node_features = graph.node_features
    if not isinstance(node_features, torch.Tensor):
        x = torch.tensor(node_features, dtype=torch.float32)
    else:
        x = node_features

    edge_index = graph.edge_index
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Build metadata dict
    metadata = {
        'corner_id': graph.corner_id,
        'num_nodes': graph.num_nodes,
        'num_edges': graph.num_edges,
        'outcome_label': graph.outcome_label,
        'goal_scored': graph.goal_scored
    }

    return x, edge_index, metadata


def plot_corner_view(ax, x, edge_index, title, pitch_length=120, pitch_width=80):
    """
    Plot a single view of the corner kick on a pitch.

    Args:
        ax: Matplotlib axis
        x: Node features [num_nodes, 14]
        edge_index: Edge connectivity [2, num_edges]
        title: Title for the subplot
        pitch_length: Pitch length (default 120)
        pitch_width: Pitch width (default 80)
    """
    # Create pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22ab4d', line_color='white',
                  linewidth=2, stripe=False)
    pitch.draw(ax=ax)

    # Extract positions and team flags
    x_pos = x[:, 0].numpy()
    y_pos = x[:, 1].numpy()
    team_flags = x[:, 10].numpy()

    # Separate attacking and defending players
    attacking_mask = team_flags == 1.0
    defending_mask = team_flags == 0.0

    # Plot players
    if attacking_mask.any():
        ax.scatter(x_pos[attacking_mask], y_pos[attacking_mask],
                  c='red', s=200, edgecolors='white', linewidths=2,
                  alpha=0.8, zorder=3, label='Attacking')

    if defending_mask.any():
        ax.scatter(x_pos[defending_mask], y_pos[defending_mask],
                  c='blue', s=200, edgecolors='white', linewidths=2,
                  alpha=0.8, zorder=3, label='Defending')

    # Draw edges (connections between players)
    edge_index_np = edge_index.numpy()
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, i]
        ax.plot([x_pos[src], x_pos[dst]],
               [y_pos[src], y_pos[dst]],
               'gray', alpha=0.2, linewidth=1, zorder=1)

    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    return ax


def visualize_d2_augmentation(
    graph_path: str = "data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl",
    sample_idx: int = 0,
    output_path: str = "data/results/d2_augmentation_demo.png"
):
    """
    Create visualization of all 4 D2 views.

    Args:
        graph_path: Path to graph dataset
        sample_idx: Index of graph to visualize
        output_path: Where to save the visualization
    """
    # Load sample graph
    print(f"Loading graph from {graph_path}...")
    x, edge_index, metadata = load_sample_graph(graph_path, sample_idx)

    print(f"Graph: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
    print(f"Metadata: {metadata}")

    # Create D2 augmentation
    aug = D2Augmentation(pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH)

    # Generate all 4 views
    print("Generating D2 views...")
    views = aug.get_all_views(x, edge_index)

    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('D2 Symmetry Augmentation: 4 Views of Corner Kick',
                 fontsize=18, fontweight='bold', y=0.98)

    titles = [
        'Identity (Original)',
        'Horizontal Flip',
        'Vertical Flip',
        'Both Flips (180° Rotation)'
    ]

    # Plot each view
    for idx, (ax, (view_x, view_edge), title) in enumerate(zip(axes.flat, views, titles)):
        print(f"Plotting view {idx+1}: {title}")
        plot_corner_view(ax, view_x, view_edge, title)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")

    # Show figure
    plt.show()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize D2 augmentation for corner kick graphs'
    )
    parser.add_argument(
        '--graph-path',
        type=str,
        default='data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl',
        help='Path to graph dataset'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Index of graph to visualize'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/results/d2_augmentation_demo.png',
        help='Output path for visualization'
    )

    args = parser.parse_args()

    visualize_d2_augmentation(
        graph_path=args.graph_path,
        sample_idx=args.sample_idx,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
