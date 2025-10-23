#!/usr/bin/env python3
"""
Visualize Graph Structure and Adjacency Matrices

Debug tool for Phase 2.2 graph construction.

This script:
1. Loads node features for a sample corner
2. Builds graphs with different adjacency strategies
3. Visualizes adjacency patterns overlaid on soccer pitch
4. Compares graph properties across strategies

Usage:
    python scripts/visualize_graph_structure.py --corner-id <id> --strategy all
    python scripts/visualize_graph_structure.py --strategy team --num-samples 5

Output:
    - data/results/graphs/adjacency_visualization_<corner_id>.png
    - data/results/graphs/strategy_comparison_<corner_id>.png
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import argparse
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph_builder import GraphBuilder, CornerGraph, compare_adjacency_strategies

# Pitch dimensions
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def draw_pitch(ax, crop_to_attacking_half: bool = False):
    """
    Draw soccer pitch on matplotlib axes.

    Args:
        ax: Matplotlib axes
        crop_to_attacking_half: If True, only show right half
    """
    # Set pitch boundaries
    if crop_to_attacking_half:
        ax.set_xlim(60, 120)
        ax.set_ylim(0, 80)
    else:
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 80)

    ax.set_aspect('equal')
    ax.set_facecolor('#1a5f3a')  # Grass green

    # Pitch outline
    pitch_color = 'white'
    pitch_linewidth = 2

    if not crop_to_attacking_half:
        # Full pitch outline
        ax.plot([0, 0], [0, 80], color=pitch_color, linewidth=pitch_linewidth)
        ax.plot([0, 120], [80, 80], color=pitch_color, linewidth=pitch_linewidth)
        ax.plot([120, 120], [80, 0], color=pitch_color, linewidth=pitch_linewidth)
        ax.plot([120, 0], [0, 0], color=pitch_color, linewidth=pitch_linewidth)

        # Halfway line
        ax.plot([60, 60], [0, 80], color=pitch_color, linewidth=pitch_linewidth)

        # Center circle
        center_circle = plt.Circle((60, 40), 10, color=pitch_color, fill=False, linewidth=pitch_linewidth)
        ax.add_patch(center_circle)

    # Penalty area (right side)
    ax.plot([102, 102], [18, 62], color=pitch_color, linewidth=pitch_linewidth)
    ax.plot([102, 120], [18, 18], color=pitch_color, linewidth=pitch_linewidth)
    ax.plot([102, 120], [62, 62], color=pitch_color, linewidth=pitch_linewidth)

    # 6-yard box (right side)
    ax.plot([114, 114], [30, 50], color=pitch_color, linewidth=pitch_linewidth)
    ax.plot([114, 120], [30, 30], color=pitch_color, linewidth=pitch_linewidth)
    ax.plot([114, 120], [50, 50], color=pitch_color, linewidth=pitch_linewidth)

    # Goal (right side)
    ax.plot([120, 120], [36, 44], color=pitch_color, linewidth=pitch_linewidth + 1)

    ax.axis('off')


def visualize_graph_structure(
    graph: CornerGraph,
    output_path: str,
    crop_to_attacking_half: bool = True
):
    """
    Visualize graph structure on soccer pitch.

    Args:
        graph: CornerGraph object
        output_path: Path to save figure
        crop_to_attacking_half: If True, crop to attacking half
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw pitch
    draw_pitch(ax, crop_to_attacking_half=crop_to_attacking_half)

    # Extract positions and teams
    positions = graph.node_features[:, :2]  # x, y coordinates
    teams = graph.teams

    # Plot nodes (players)
    attacking_mask = np.array([t == 'attacking' for t in teams])
    defending_mask = ~attacking_mask

    ax.scatter(
        positions[attacking_mask, 0],
        positions[attacking_mask, 1],
        c='red',
        s=200,
        edgecolors='white',
        linewidths=2,
        alpha=0.8,
        label='Attacking',
        zorder=10
    )

    ax.scatter(
        positions[defending_mask, 0],
        positions[defending_mask, 1],
        c='blue',
        s=200,
        edgecolors='white',
        linewidths=2,
        alpha=0.8,
        label='Defending',
        zorder=10
    )

    # Plot edges (connections)
    edge_index = graph.edge_index
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        pos_src = positions[src]
        pos_dst = positions[dst]

        # Draw edge
        ax.plot(
            [pos_src[0], pos_dst[0]],
            [pos_src[1], pos_dst[1]],
            color='yellow',
            alpha=0.3,
            linewidth=1,
            zorder=1
        )

    # Title with graph statistics
    title = f"Corner: {graph.corner_id}\n"
    title += f"Nodes: {graph.num_nodes} | Edges: {graph.num_edges} | "
    title += f"Avg Degree: {graph.num_edges / graph.num_nodes:.1f}"
    if graph.outcome_label:
        title += f" | Outcome: {graph.outcome_label}"

    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a5f3a', bbox_inches='tight')
    plt.close()


def compare_strategies_visualization(
    graphs: Dict[str, CornerGraph],
    output_path: str
):
    """
    Create comparison visualization of all adjacency strategies.

    Args:
        graphs: Dictionary mapping strategy name to CornerGraph
        output_path: Path to save figure
    """
    strategies = ['team', 'distance', 'delaunay', 'ball_centric', 'zone']
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]

        if strategy not in graphs:
            ax.axis('off')
            continue

        graph = graphs[strategy]

        # Draw pitch (cropped)
        draw_pitch(ax, crop_to_attacking_half=True)

        # Extract positions and teams
        positions = graph.node_features[:, :2]
        teams = graph.teams

        # Plot nodes
        attacking_mask = np.array([t == 'attacking' for t in teams])
        defending_mask = ~attacking_mask

        ax.scatter(
            positions[attacking_mask, 0],
            positions[attacking_mask, 1],
            c='red',
            s=150,
            edgecolors='white',
            linewidths=1.5,
            alpha=0.8,
            zorder=10
        )

        ax.scatter(
            positions[defending_mask, 0],
            positions[defending_mask, 1],
            c='blue',
            s=150,
            edgecolors='white',
            linewidths=1.5,
            alpha=0.8,
            zorder=10
        )

        # Plot edges
        edge_index = graph.edge_index
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            pos_src = positions[src]
            pos_dst = positions[dst]

            ax.plot(
                [pos_src[0], pos_dst[0]],
                [pos_src[1], pos_dst[1]],
                color='yellow',
                alpha=0.25,
                linewidth=0.8,
                zorder=1
            )

        # Title with statistics
        title = f"{strategy.upper()}\n"
        title += f"Edges: {graph.num_edges} | Avg Degree: {graph.num_edges / graph.num_nodes:.1f}"
        ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=10)

    # Hide unused subplot
    axes[5].axis('off')

    # Overall title
    corner_id = list(graphs.values())[0].corner_id
    fig.suptitle(
        f"Adjacency Strategy Comparison - Corner: {corner_id}",
        fontsize=16,
        fontweight='bold',
        color='white',
        y=0.98
    )

    plt.tight_layout()
    fig.patch.set_facecolor('#1a5f3a')
    plt.savefig(output_path, dpi=150, facecolor='#1a5f3a', bbox_inches='tight')
    plt.close()


def print_strategy_statistics(graphs: Dict[str, CornerGraph]):
    """
    Print comparison statistics for all strategies.

    Args:
        graphs: Dictionary mapping strategy name to CornerGraph
    """
    print("\n" + "=" * 60)
    print("Adjacency Strategy Comparison")
    print("=" * 60)

    for strategy, graph in graphs.items():
        avg_degree = graph.num_edges / graph.num_nodes if graph.num_nodes > 0 else 0
        sparsity = 1 - (graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1))) \
                   if graph.num_nodes > 1 else 0

        print(f"\n{strategy.upper()}:")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.num_edges}")
        print(f"  Avg Degree: {avg_degree:.2f}")
        print(f"  Sparsity: {sparsity:.2%}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Visualize graph structures")
    parser.add_argument(
        '--corner-id',
        type=str,
        default=None,
        help='Specific corner ID to visualize (default: first corner)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='all',
        choices=['all', 'team', 'distance', 'delaunay', 'ball_centric', 'zone'],
        help='Adjacency strategy to visualize'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of sample corners to visualize'
    )
    parser.add_argument(
        '--features-path',
        type=str,
        default='data/features/node_features/statsbomb_player_features.parquet',
        help='Path to node features'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results/graphs',
        help='Output directory for visualizations'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Graph Structure Visualization")
    print("=" * 60)

    # Load node features
    print(f"Loading features from {args.features_path}")
    features_df = pd.read_parquet(args.features_path)
    print(f"Loaded {len(features_df)} player features")

    # Select corner(s)
    corner_ids = features_df['corner_id'].unique()
    if args.corner_id:
        selected_corners = [args.corner_id]
    else:
        selected_corners = corner_ids[:args.num_samples]

    print(f"Visualizing {len(selected_corners)} corner(s)")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each corner
    for corner_id in selected_corners:
        print(f"\nProcessing corner: {corner_id}")

        # Extract features for this corner
        corner_features = features_df[features_df['corner_id'] == corner_id].copy()

        if args.strategy == 'all':
            # Compare all strategies
            graphs = compare_adjacency_strategies(corner_features, corner_id)

            # Print statistics
            print_strategy_statistics(graphs)

            # Create comparison visualization
            output_path = output_dir / f"strategy_comparison_{corner_id}.png"
            compare_strategies_visualization(graphs, str(output_path))
            print(f"✅ Saved comparison to {output_path}")

        else:
            # Single strategy
            builder = GraphBuilder(adjacency_strategy=args.strategy)
            graph = builder.build_graph_from_features(corner_features, corner_id)

            # Visualize
            output_path = output_dir / f"graph_{args.strategy}_{corner_id}.png"
            visualize_graph_structure(graph, str(output_path))
            print(f"✅ Saved visualization to {output_path}")

            # Print statistics
            print(f"\nGraph Statistics ({args.strategy}):")
            print(f"  Nodes: {graph.num_nodes}")
            print(f"  Edges: {graph.num_edges}")
            print(f"  Avg Degree: {graph.num_edges / graph.num_nodes:.2f}")

    print("\n" + "=" * 60)
    print("Visualization Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
