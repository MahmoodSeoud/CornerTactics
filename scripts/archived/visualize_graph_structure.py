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
from mplsoccer import Pitch, VerticalPitch

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
    Uses mplsoccer with vertical orientation (goal at top).
    Professional broadcast style matching StatsBomb visualizations.

    Args:
        graphs: Dictionary mapping strategy name to CornerGraph
        output_path: Path to save figure
    """
    strategies = ['team', 'team_with_ball', 'distance', 'delaunay', 'ball_centric', 'zone']

    # Create 2x3 grid (6 strategies)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.patch.set_facecolor('#ffffff')
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]

        if strategy not in graphs:
            ax.axis('off')
            continue

        graph = graphs[strategy]

        # Create vertical pitch (goal at top)
        # Use VerticalPitch class for proper vertical orientation
        pitch = VerticalPitch(
            pitch_type='statsbomb',
            pitch_color='#195905',      # Rich grass green
            line_color='#ffffff',        # White lines
            linewidth=2.5,
            line_zorder=2,
            stripe=True,                 # Broadcast-style stripes
            stripe_color='#1a6d08',      # Alternating grass shade
            half=True,                   # Show only attacking half
            pad_top=2,
            pad_bottom=2,
            pad_left=2,
            pad_right=2
        )

        pitch.draw(ax=ax)

        # Draw custom goal box behind the end line (vertical pitch: goal at top)
        # StatsBomb goal is 8 yards wide centered at y=40 (y: 36-44)
        # For vertical pitch, goal at x: 36-44, behind y=120 line
        from matplotlib.patches import Rectangle
        goal_box = Rectangle(
            (36, 120),  # (x, y) bottom-left corner - starts AT the line
            8,          # width (8 yards)
            2.5,        # height (depth of goal extending behind)
            linewidth=2.5,
            edgecolor='#FFFFFF',
            facecolor='#4A90E2',  # Blue fill like reference image
            zorder=0,  # Behind everything
            alpha=0.4
        )
        ax.add_patch(goal_box)

        # Draw goal posts (thicker white lines at sides)
        from matplotlib.lines import Line2D
        # Left post
        left_post = Line2D([36, 36], [120, 122.5], linewidth=3.5, color='#FFFFFF', zorder=1)
        # Right post
        right_post = Line2D([44, 44], [120, 122.5], linewidth=3.5, color='#FFFFFF', zorder=1)
        # Crossbar
        crossbar = Line2D([36, 44], [122.5, 122.5], linewidth=3.5, color='#FFFFFF', zorder=1)
        ax.add_line(left_post)
        ax.add_line(right_post)
        ax.add_line(crossbar)

        # Extract positions and teams
        # mplsoccer handles coordinate transform with orientation='vertical'
        positions_orig = graph.node_features[:, :2]  # x, y in StatsBomb coords
        teams = graph.teams

        # Check if this graph has a ball node (team_with_ball strategy)
        has_ball_node = (strategy == 'team_with_ball')

        # No coordinate transformation needed - mplsoccer handles it!
        positions = positions_orig

        # Plot nodes (players)
        attacking_mask = np.array([t == 'attacking' for t in teams])
        defending_mask = ~attacking_mask

        # Attacking players
        if attacking_mask.any():
            pitch.scatter(
                positions[attacking_mask, 0],
                positions[attacking_mask, 1],
                s=150,
                color='#E31E24',
                edgecolors='#FFFFFF',
                linewidth=1,
                zorder=9,
                ax=ax,
                alpha=0.75
            )

        # Defending players
        if defending_mask.any():
            pitch.scatter(
                positions[defending_mask, 0],
                positions[defending_mask, 1],
                s=150,
                color='#0047AB',
                edgecolors='#FFFFFF',
                linewidth=1,
                zorder=9,
                ax=ax,
                alpha=0.75
            )

        # Plot edges (graph connections) - color by connection type
        edge_index = graph.edge_index
        ball_node_idx = graph.num_nodes - 1 if has_ball_node else None

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]

            # Check if edge involves ball node
            is_ball_edge = (ball_node_idx is not None and
                           (src == ball_node_idx or dst == ball_node_idx))

            if is_ball_edge:
                # Connection to ball: black
                edge_color = '#000000'
                edge_alpha = 0.8  # Very visible
                player_idx = src if dst == ball_node_idx else dst
                pos_src = positions[player_idx] if src == ball_node_idx else positions[src]
                pos_dst = positions[player_idx] if dst == ball_node_idx else positions[dst]
                # Get ball position from features (distance_to_ball_target columns)
                # For simplicity, use corner location from first player's data
                ball_x = graph.node_features[0, 0]  # Approximate ball location
                ball_y = graph.node_features[0, 1]  # Will be improved
                if src == ball_node_idx:
                    pos_src = np.array([ball_x, ball_y])
                else:
                    pos_dst = np.array([ball_x, ball_y])
            else:
                # Player-to-player connection
                pos_src = positions[src]
                pos_dst = positions[dst]
                team_src = teams[src]
                team_dst = teams[dst]

                if team_src == team_dst:
                    # Same team connection
                    if team_src == 'attacking':
                        edge_color = '#E31E24'  # Bright red for attacking-attacking
                        edge_alpha = 0.8  # Very visible
                    else:
                        edge_color = '#0047AB'  # Royal blue for defending-defending
                        edge_alpha = 0.8  # Very visible
                else:
                    # Cross-team connection: yellow
                    edge_color = '#FFD700'
                    edge_alpha = 0.7  # Visible

            # Draw edge line
            pitch.plot(
                [pos_src[0], pos_dst[0]],
                [pos_src[1], pos_dst[1]],
                color=edge_color,
                alpha=edge_alpha,
                linewidth=1.2,
                zorder=1,
                ax=ax
            )

        # Plot ball node if present
        if has_ball_node and ball_node_idx is not None:
            ball_x = graph.node_features[0, 0]
            ball_y = graph.node_features[0, 1]

            pitch.scatter(
                ball_x, ball_y,
                s=30,
                color='#808080',
                edgecolors='none',
                linewidth=0,
                marker='o',
                zorder=15,
                ax=ax,
                alpha=1.0
            )

        # Title with statistics
        title = f"{strategy.upper()}\n"
        title += f"Edges: {graph.num_edges} | Avg Degree: {graph.num_edges / graph.num_nodes:.1f}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Attacking Player',
               markerfacecolor='#E31E24', markeredgecolor='#FFFFFF', markersize=8,
               markeredgewidth=0.5, linestyle='None', alpha=0.75),
        Line2D([0], [0], marker='o', color='w', label='Defending Player',
               markerfacecolor='#0047AB', markeredgecolor='#FFFFFF', markersize=8,
               markeredgewidth=0.5, linestyle='None', alpha=0.75),
        Line2D([0], [0], marker='o', color='w', label='Ball Node',
               markerfacecolor='#808080', markeredgecolor='none', markersize=6,
               markeredgewidth=0, linestyle='None', alpha=1.0),
        Line2D([0], [0], color='#E31E24', lw=3, alpha=0.8, label='Intra-Team (Attack)'),
        Line2D([0], [0], color='#0047AB', lw=3, alpha=0.8, label='Intra-Team (Defense)'),
        Line2D([0], [0], color='#FFD700', lw=3, alpha=0.7, label='Inter-Team'),
        Line2D([0], [0], color='#000000', lw=3, alpha=0.8, label='Ball Connection')
    ]

    # Add legend to figure (not to subplot)
    fig.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        title='Legend',
        title_fontsize=14,
        bbox_to_anchor=(0.98, 0.02)
    )

    # Overall title
    corner_id = list(graphs.values())[0].corner_id
    fig.suptitle(
        f"Graph Adjacency Strategy Comparison\nCorner: {corner_id}",
        fontsize=18,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, facecolor='#ffffff', bbox_inches='tight')
    plt.close()

    print(f"✅ Saved strategy comparison: {output_path}")


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
