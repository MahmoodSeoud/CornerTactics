#!/usr/bin/env python3
"""
Build Graph Dataset from SkillCorner Temporal Features

Converts SkillCorner temporal features into graph representations
and merges with StatsBomb augmented graphs.

Author: mseo
Date: October 2024
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph_builder import CornerGraph, GraphBuilder


def build_skillcorner_graphs(features_file: str, adjacency_strategy: str = 'team') -> list:
    """
    Build graphs from SkillCorner temporal features.

    Args:
        features_file: Path to SkillCorner features CSV/parquet
        adjacency_strategy: Strategy for building adjacency matrix

    Returns:
        List of CornerGraph objects
    """
    print("="*70)
    print("Building SkillCorner Temporal Graphs")
    print("="*70)

    # Load features
    print(f"\nLoading features from {features_file}")
    if features_file.endswith('.parquet'):
        df = pd.read_parquet(features_file)
    else:
        df = pd.read_csv(features_file)

    print(f"Loaded {len(df)} player features across {df['corner_id'].nunique()} corners")

    # Initialize graph builder
    builder = GraphBuilder(adjacency_strategy=adjacency_strategy)

    # Group by corner_id
    corner_groups = df.groupby('corner_id')

    graphs = []
    dangerous_count = 0

    print(f"\nBuilding graphs with '{adjacency_strategy}' adjacency strategy...")
    for corner_id, corner_df in tqdm(corner_groups, desc="Building graphs"):
        # Get outcome
        outcome = corner_df['outcome'].iloc[0] if 'outcome' in corner_df.columns else 'Unknown'
        goal_scored_val = corner_df['goal_scored'].iloc[0] if 'goal_scored' in corner_df.columns else False

        # Dangerous situation = shot OR goal
        is_dangerous = outcome in ['Shot', 'Goal'] or goal_scored_val
        if is_dangerous:
            dangerous_count += 1

        # Use GraphBuilder's built-in method
        graph = builder.build_graph_from_features(corner_df, str(corner_id))

        # Update the goal_scored flag to reflect dangerous situation
        graph.goal_scored = is_dangerous

        graphs.append(graph)

    print(f"\n{'='*70}")
    print("Graph Building Complete")
    print(f"{'='*70}")
    print(f"Total graphs: {len(graphs)}")
    print(f"Dangerous situations: {dangerous_count} ({dangerous_count/len(graphs)*100:.1f}%)")
    print(f"Average nodes per graph: {np.mean([g.num_nodes for g in graphs]):.1f}")
    print(f"Average edges per graph: {np.mean([g.num_edges for g in graphs]):.1f}")

    return graphs


def merge_datasets(statsbomb_file: str, skillcorner_graphs: list, output_file: str):
    """
    Merge StatsBomb and SkillCorner graph datasets.

    Args:
        statsbomb_file: Path to StatsBomb augmented graphs
        skillcorner_graphs: List of SkillCorner graphs
        output_file: Path to save merged dataset
    """
    print(f"\n{'='*70}")
    print("Merging Datasets")
    print(f"{'='*70}")

    # Load StatsBomb graphs
    print(f"\nLoading StatsBomb graphs from {statsbomb_file}")
    with open(statsbomb_file, 'rb') as f:
        statsbomb_graphs = pickle.load(f)
    print(f"Loaded {len(statsbomb_graphs)} StatsBomb graphs")

    # Combine datasets
    all_graphs = statsbomb_graphs + skillcorner_graphs

    # Statistics
    dangerous_count = sum(1 for g in all_graphs if g.goal_scored)

    print(f"\n{'='*70}")
    print("Merged Dataset Statistics")
    print(f"{'='*70}")
    print(f"StatsBomb graphs: {len(statsbomb_graphs):,}")
    print(f"SkillCorner graphs: {len(skillcorner_graphs):,}")
    print(f"Total graphs: {len(all_graphs):,}")
    print(f"Dangerous situations: {dangerous_count:,} ({dangerous_count/len(all_graphs)*100:.1f}%)")
    print(f"\nAverage nodes per graph: {np.mean([g.num_nodes for g in all_graphs]):.1f}")
    print(f"Average edges per graph: {np.mean([g.num_edges for g in all_graphs]):.1f}")

    # Save merged dataset
    print(f"\nSaving merged dataset to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(all_graphs, f)

    print(f"✓ Saved {len(all_graphs):,} graphs")


if __name__ == "__main__":
    # Paths
    features_file = "data/features/temporal/skillcorner_temporal_features.parquet"
    statsbomb_file = "data/graphs/adjacency_team/statsbomb_temporal_augmented.pkl"
    skillcorner_output = "data/graphs/adjacency_team/skillcorner_temporal_graphs.pkl"
    merged_output = "data/graphs/adjacency_team/combined_temporal_graphs.pkl"

    # Build SkillCorner graphs
    skillcorner_graphs = build_skillcorner_graphs(
        features_file=features_file,
        adjacency_strategy='team'
    )

    # Save SkillCorner graphs separately
    print(f"\nSaving SkillCorner graphs to {skillcorner_output}")
    with open(skillcorner_output, 'wb') as f:
        pickle.dump(skillcorner_graphs, f)
    print(f"✓ Saved {len(skillcorner_graphs):,} SkillCorner graphs")

    # Merge with StatsBomb
    merge_datasets(
        statsbomb_file=statsbomb_file,
        skillcorner_graphs=skillcorner_graphs,
        output_file=merged_output
    )

    print(f"\n{'='*70}")
    print("✓ Complete!")
    print(f"{'='*70}")
    print(f"SkillCorner graphs: {skillcorner_output}")
    print(f"Combined dataset: {merged_output}")
