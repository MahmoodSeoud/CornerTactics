#!/usr/bin/env python3
"""
Temporal Data Augmentation for StatsBomb Corners

Creates synthetic temporal variations similar to US Soccer Federation approach:
- Multiple temporal offsets per corner (5 frames like SkillCorner)
- Position perturbations to simulate player movement
- Mirror augmentation for left/right corners

This matches the temporal expansion strategy used by US Soccer Fed
for their counterattack GNN.

Usage:
    python scripts/augment_statsbomb_temporal.py

Author: mseo
Date: October 2024
"""

import sys
import pickle
import warnings
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph_builder import CornerGraph

warnings.filterwarnings('ignore')


def augment_player_positions(node_features: np.ndarray, temporal_offset: float,
                             noise_scale: float = 0.5) -> np.ndarray:
    """
    Augment player positions with temporal offset and noise.

    Args:
        node_features: Original node features [N, 14]
        temporal_offset: Time offset in seconds (-2.0 to +2.0)
        noise_scale: Scale of gaussian noise for position perturbation

    Returns:
        Augmented node features [N, 14]
    """
    augmented = node_features.copy()

    # Add gaussian noise to positions (simulate movement uncertainty)
    # Noise magnitude increases with temporal distance from freeze-frame
    noise_magnitude = abs(temporal_offset) * noise_scale

    # Apply noise to spatial features (x, y)
    augmented[:, 0] += np.random.normal(0, noise_magnitude, len(augmented))  # x
    augmented[:, 1] += np.random.normal(0, noise_magnitude, len(augmented))  # y

    # Update distance-based features after position changes
    for i in range(len(augmented)):
        x, y = augmented[i, 0], augmented[i, 1]

        # Recalculate distance to goal (assuming goal at 120, 40)
        goal_x, goal_y = 120, 40
        augmented[i, 2] = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)

        # Recalculate angle to goal
        augmented[i, 8] = np.arctan2(goal_y - y, goal_x - x)

    # Add small noise to velocity features (simulate velocity estimation uncertainty)
    if abs(temporal_offset) > 0:
        velocity_noise = 0.1
        augmented[:, 4] += np.random.normal(0, velocity_noise, len(augmented))  # vx
        augmented[:, 5] += np.random.normal(0, velocity_noise, len(augmented))  # vy

        # Recalculate velocity magnitude and angle
        augmented[:, 6] = np.sqrt(augmented[:, 4]**2 + augmented[:, 5]**2)
        augmented[:, 7] = np.arctan2(augmented[:, 5], augmented[:, 4])

    return augmented


def mirror_corner(graph: CornerGraph) -> CornerGraph:
    """
    Mirror corner kick from one side to the other.

    Flips y-coordinates to create left/right symmetry.
    This doubles the dataset with geometrically valid variations.

    Args:
        graph: Original corner graph

    Returns:
        Mirrored corner graph
    """
    mirrored_features = graph.node_features.copy()

    # Flip y-coordinate (assuming pitch is 80 units wide)
    mirrored_features[:, 1] = 80 - mirrored_features[:, 1]

    # Update angle features (flip angles)
    mirrored_features[:, 7] = -mirrored_features[:, 7]  # velocity_angle
    mirrored_features[:, 8] = -mirrored_features[:, 8]  # angle_to_goal
    mirrored_features[:, 9] = -mirrored_features[:, 9]  # angle_to_ball

    # Recalculate distance to goal with mirrored position
    for i in range(len(mirrored_features)):
        x, y = mirrored_features[i, 0], mirrored_features[i, 1]
        goal_x, goal_y = 120, 40
        mirrored_features[i, 2] = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)

    # Create mirrored graph
    mirrored_graph = CornerGraph(
        corner_id=f"{graph.corner_id}_mirror",
        node_features=mirrored_features,
        edge_index=graph.edge_index.copy(),
        edge_features=graph.edge_features.copy() if graph.edge_features is not None else None,
        adjacency_matrix=graph.adjacency_matrix.copy() if graph.adjacency_matrix is not None else None,
        player_ids=graph.player_ids.copy(),
        teams=graph.teams.copy(),
        goal_scored=graph.goal_scored,
        outcome_label=graph.outcome_label
    )

    return mirrored_graph


def augment_statsbomb_temporal():
    """
    Main function to augment StatsBomb corners with temporal variations.

    Creates 5 temporal frames per corner like SkillCorner:
    - t = -2.0s, -1.0s, 0.0s, +1.0s, +2.0s

    Plus mirror augmentation for geometric diversity.
    """
    print("="*70)
    print("StatsBomb Temporal Augmentation")
    print("="*70)

    # Load original StatsBomb graphs
    graphs_file = Path("data/graphs/adjacency_team/statsbomb_graphs.pkl")
    if not graphs_file.exists():
        print(f"Error: {graphs_file} not found!")
        return

    with open(graphs_file, 'rb') as f:
        original_graphs = pickle.load(f)

    print(f"\nLoaded {len(original_graphs)} original StatsBomb corners")

    # Temporal offsets (matching SkillCorner)
    temporal_offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]

    augmented_graphs = []

    # Set random seed for reproducibility
    np.random.seed(42)

    print("\nGenerating temporal augmentations...")
    for graph in tqdm(original_graphs, desc="Augmenting corners"):

        # Create temporal variations
        for offset in temporal_offsets:
            # Get outcome label (shot or goal = dangerous)
            is_dangerous = graph.goal_scored or (graph.outcome_label in ['Shot', 'shot'])

            if offset == 0.0:
                # Use original freeze-frame for t=0
                aug_graph = CornerGraph(
                    corner_id=f"{graph.corner_id}_t{offset:+.1f}",
                    node_features=graph.node_features.copy(),
                    edge_index=graph.edge_index.copy(),
                    edge_features=graph.edge_features.copy() if graph.edge_features is not None else None,
                    adjacency_matrix=graph.adjacency_matrix.copy() if graph.adjacency_matrix is not None else None,
                    player_ids=graph.player_ids.copy(),
                    teams=graph.teams.copy(),
                    goal_scored=is_dangerous,  # Use dangerous situation instead of just goal
                    outcome_label=graph.outcome_label
                )
            else:
                # Apply position augmentation for other time offsets
                augmented_features = augment_player_positions(
                    graph.node_features, offset, noise_scale=0.5
                )

                aug_graph = CornerGraph(
                    corner_id=f"{graph.corner_id}_t{offset:+.1f}",
                    node_features=augmented_features,
                    edge_index=graph.edge_index.copy(),
                    edge_features=graph.edge_features.copy() if graph.edge_features is not None else None,
                    adjacency_matrix=graph.adjacency_matrix.copy() if graph.adjacency_matrix is not None else None,
                    player_ids=graph.player_ids.copy(),
                    teams=graph.teams.copy(),
                    goal_scored=is_dangerous,
                    outcome_label=graph.outcome_label
                )

            augmented_graphs.append(aug_graph)

    print(f"Created {len(augmented_graphs)} temporal graphs (5x original)")

    # Optional: Add mirror augmentation for additional geometric diversity
    print("\nApplying mirror augmentation...")
    mirrored_graphs = []
    for graph in tqdm(augmented_graphs[:len(original_graphs)], desc="Mirroring corners"):
        # Only mirror the t=0 frames to avoid excessive augmentation
        if "_t+0.0" in graph.corner_id:
            mirrored = mirror_corner(graph)
            mirrored_graphs.append(mirrored)

    # Combine all augmented graphs
    all_graphs = augmented_graphs + mirrored_graphs

    print(f"\n{'='*70}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*70}")
    print(f"Original corners: {len(original_graphs)}")
    print(f"Temporal frames (5x): {len(augmented_graphs)}")
    print(f"Mirror augmentation: {len(mirrored_graphs)}")
    print(f"Total augmented graphs: {len(all_graphs)}")

    # Count dangerous situations
    dangerous_count = sum(1 for g in all_graphs if g.goal_scored)
    print(f"Dangerous situations: {dangerous_count} ({dangerous_count/len(all_graphs)*100:.1f}%)")

    # Save augmented graphs
    output_dir = Path("data/graphs/adjacency_team")
    output_file = output_dir / "statsbomb_temporal_augmented.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(all_graphs, f)

    print(f"\nSaved to: {output_file}")

    # Statistics
    print(f"\n{'='*70}")
    print("Dataset Statistics")
    print(f"{'='*70}")
    print(f"StatsBomb augmented: {len(all_graphs):,}")
    print(f"  - Temporal frames: {len(augmented_graphs):,}")
    print(f"  - Mirror augmentation: {len(mirrored_graphs):,}")
    print(f"  - Dangerous situations: {dangerous_count} ({dangerous_count/len(all_graphs)*100:.1f}%)")
    print(f"\nAverage nodes per graph: {np.mean([g.num_nodes for g in all_graphs]):.1f}")
    print(f"Average edges per graph: {np.mean([g.num_edges for g in all_graphs]):.1f}")


if __name__ == "__main__":
    augment_statsbomb_temporal()
