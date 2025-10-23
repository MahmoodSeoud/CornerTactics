#!/usr/bin/env python3
"""
Build Graph Dataset from Node Features

Implements Phase 2.2: Batch graph construction from Phase 2.1 node features.

This script:
1. Loads player node features (StatsBomb/SkillCorner)
2. Builds graphs with selected adjacency strategy
3. Computes edge features
4. Saves graph dataset for GNN training

Usage:
    python scripts/build_graph_dataset.py --strategy team --dataset statsbomb
    python scripts/build_graph_dataset.py --strategy distance --dataset all

Output:
    - data/graphs/adjacency_<strategy>/statsbomb_graphs.pkl
    - data/graphs/adjacency_<strategy>/skillcorner_graphs.pkl
    - data/graphs/adjacency_<strategy>/graph_statistics.json
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
import argparse
from tqdm import tqdm
from typing import List, Dict
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph_builder import GraphBuilder, CornerGraph, AdjacencyStrategy


def build_graphs_from_features(
    features_path: str,
    strategy: AdjacencyStrategy,
    output_dir: str,
    dataset_name: str
) -> List[CornerGraph]:
    """
    Build graphs from node features dataset.

    Args:
        features_path: Path to parquet file with node features
        strategy: Adjacency matrix construction strategy
        output_dir: Output directory for graphs
        dataset_name: Name of dataset (statsbomb/skillcorner)

    Returns:
        List of CornerGraph objects
    """
    print("\n" + "=" * 60)
    print(f"Building {dataset_name.upper()} Graphs - Strategy: {strategy}")
    print("=" * 60)

    # Load node features
    print(f"Loading features from {features_path}")
    features_df = pd.read_parquet(features_path)
    print(f"Loaded {len(features_df)} player features")

    # Get unique corners
    corner_ids = features_df['corner_id'].unique()
    print(f"Found {len(corner_ids)} unique corners")

    # Initialize graph builder
    builder = GraphBuilder(adjacency_strategy=strategy)

    # Build graphs for each corner
    graphs = []
    failed_corners = []

    for corner_id in tqdm(corner_ids, desc="Building graphs"):
        try:
            # Extract features for this corner
            corner_features = features_df[features_df['corner_id'] == corner_id].copy()

            # Build graph
            graph = builder.build_graph_from_features(corner_features, corner_id)
            graphs.append(graph)

        except Exception as e:
            warnings.warn(f"Failed to build graph for corner {corner_id}: {e}")
            failed_corners.append(corner_id)
            continue

    print(f"\n✅ Successfully built {len(graphs)} graphs")
    if failed_corners:
        print(f"⚠️  Failed to build {len(failed_corners)} graphs")

    # Save graphs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{dataset_name}_graphs.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Saved graphs to {output_file}")

    return graphs


def compute_graph_statistics(graphs: List[CornerGraph], dataset_name: str) -> Dict:
    """
    Compute statistics about the graph dataset.

    Args:
        graphs: List of CornerGraph objects
        dataset_name: Name of dataset

    Returns:
        Dictionary with statistics
    """
    print(f"\n{dataset_name.upper()} Graph Statistics")
    print("-" * 60)

    num_nodes_list = [g.num_nodes for g in graphs]
    num_edges_list = [g.num_edges for g in graphs]

    stats = {
        'dataset': dataset_name,
        'num_graphs': len(graphs),
        'num_nodes': {
            'mean': float(np.mean(num_nodes_list)),
            'std': float(np.std(num_nodes_list)),
            'min': int(np.min(num_nodes_list)),
            'max': int(np.max(num_nodes_list))
        },
        'num_edges': {
            'mean': float(np.mean(num_edges_list)),
            'std': float(np.std(num_edges_list)),
            'min': int(np.min(num_edges_list)),
            'max': int(np.max(num_edges_list))
        },
        'avg_degree': float(np.mean([g.num_edges / g.num_nodes for g in graphs if g.num_nodes > 0])),
        'sparsity': float(np.mean([1 - (g.num_edges / (g.num_nodes * (g.num_nodes - 1)))
                                   for g in graphs if g.num_nodes > 1]))
    }

    # Outcome distribution
    outcomes = [g.outcome_label for g in graphs if g.outcome_label]
    outcome_counts = pd.Series(outcomes).value_counts().to_dict()
    stats['outcome_distribution'] = {k: int(v) for k, v in outcome_counts.items()}

    # Goal rate
    goals = sum([g.goal_scored for g in graphs])
    stats['goal_rate'] = float(goals / len(graphs)) if graphs else 0.0

    # Print statistics
    print(f"Total graphs: {stats['num_graphs']}")
    print(f"Nodes per graph: {stats['num_nodes']['mean']:.1f} ± {stats['num_nodes']['std']:.1f} "
          f"(min={stats['num_nodes']['min']}, max={stats['num_nodes']['max']})")
    print(f"Edges per graph: {stats['num_edges']['mean']:.1f} ± {stats['num_edges']['std']:.1f} "
          f"(min={stats['num_edges']['min']}, max={stats['num_edges']['max']})")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Graph sparsity: {stats['sparsity']:.2%}")
    print(f"Goal rate: {stats['goal_rate']:.2%}")

    if outcome_counts:
        print("\nOutcome distribution:")
        for outcome, count in outcome_counts.items():
            print(f"  {outcome}: {count} ({100*count/len(graphs):.1f}%)")

    return stats


def save_graph_statistics(
    stats_list: List[Dict],
    strategy: AdjacencyStrategy,
    output_dir: str
):
    """
    Save combined statistics to JSON file.

    Args:
        stats_list: List of statistics dictionaries
        strategy: Adjacency strategy used
        output_dir: Output directory
    """
    combined_stats = {
        'adjacency_strategy': strategy,
        'datasets': stats_list
    }

    output_path = Path(output_dir)
    stats_file = output_path / "graph_statistics.json"

    with open(stats_file, 'w') as f:
        json.dump(combined_stats, f, indent=2)

    print(f"\n✅ Saved statistics to {stats_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Build graph dataset from node features")
    parser.add_argument(
        '--strategy',
        type=str,
        default='team',
        choices=['team', 'distance', 'delaunay', 'ball_centric', 'zone'],
        help='Adjacency matrix construction strategy'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['statsbomb', 'skillcorner', 'all'],
        help='Dataset to process'
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default='data/features/node_features',
        help='Directory containing node features'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/graphs/adjacency_<strategy>)'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"data/graphs/adjacency_{args.strategy}"

    print("\n" + "=" * 60)
    print("Phase 2.2: Graph Dataset Construction")
    print("=" * 60)
    print(f"Adjacency strategy: {args.strategy}")
    print(f"Dataset(s): {args.dataset}")
    print(f"Output directory: {args.output_dir}")

    all_stats = []

    # Process StatsBomb
    if args.dataset in ['statsbomb', 'all']:
        statsbomb_features = Path(args.features_dir) / "statsbomb_player_features.parquet"
        if statsbomb_features.exists():
            statsbomb_graphs = build_graphs_from_features(
                features_path=str(statsbomb_features),
                strategy=args.strategy,
                output_dir=args.output_dir,
                dataset_name='statsbomb'
            )
            stats = compute_graph_statistics(statsbomb_graphs, 'statsbomb')
            all_stats.append(stats)
        else:
            print(f"⚠️  StatsBomb features not found at {statsbomb_features}")

    # Process SkillCorner
    if args.dataset in ['skillcorner', 'all']:
        skillcorner_features = Path(args.features_dir) / "skillcorner_player_features.parquet"
        if skillcorner_features.exists():
            skillcorner_graphs = build_graphs_from_features(
                features_path=str(skillcorner_features),
                strategy=args.strategy,
                output_dir=args.output_dir,
                dataset_name='skillcorner'
            )
            stats = compute_graph_statistics(skillcorner_graphs, 'skillcorner')
            all_stats.append(stats)
        else:
            print(f"⚠️  SkillCorner features not found at {skillcorner_features}")

    # Save combined statistics
    if all_stats:
        save_graph_statistics(all_stats, args.strategy, args.output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Graph Dataset Construction Complete")
    print("=" * 60)
    total_graphs = sum([s['num_graphs'] for s in all_stats])
    print(f"Total graphs created: {total_graphs}")
    print(f"Strategy: {args.strategy}")
    print(f"Output: {args.output_dir}")

    # Example usage for next phase
    print("\n" + "=" * 60)
    print("Next Steps (Phase 3: GNN Model Training)")
    print("=" * 60)
    print("1. Load graphs:")
    print(f"   import pickle")
    print(f"   with open('{args.output_dir}/statsbomb_graphs.pkl', 'rb') as f:")
    print(f"       graphs = pickle.load(f)")
    print()
    print("2. Convert to PyTorch Geometric format:")
    print("   from torch_geometric.data import Data")
    print("   data_list = [Data(x=g.node_features, edge_index=g.edge_index, ...")
    print("                for g in graphs]")
    print()
    print("3. Train GNN model (Phase 3.1)")


if __name__ == "__main__":
    main()
