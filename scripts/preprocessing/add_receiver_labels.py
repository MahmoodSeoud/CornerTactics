#!/usr/bin/env python3
"""
Add Receiver Labels to Corner Graphs (TacticAI Day 1-2) - Location-Based Matching

Maps receiver labels to graph node indices using Ball Receipt location matching.

Approach:
1. Load receiver name and Ball Receipt location from corners CSV
2. Match Ball Receipt location to closest freeze frame position (attacking team only)
3. Store receiver_player_name, receiver_location, receiver_node_index in graphs

Based on TacticAI Implementation Plan:
- Extract receiver (player who receives ball after corner)
- Use Ball Receipt location to find closest freeze frame position
- Add receiver_player_name, receiver_location, receiver_node_index to graphs
- Target: 85%+ coverage (expect ~570/668 corners with receiver locations)

Output: data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl
"""

import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.receiver_labeler import ReceiverLabeler
from src.graph_builder import CornerGraph

# StatsBomb data access
try:
    from statsbombpy import sb
except ImportError:
    logger.error("statsbombpy not installed. Run: pip install statsbombpy")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_statsbomb_corners() -> pd.DataFrame:
    """Load StatsBomb corners with receiver data."""
    # Use the newly downloaded corners with receiver info
    corners_path = Path("data/raw/statsbomb/corners_360.csv")

    if not corners_path.exists():
        logger.error(f"StatsBomb corners file not found: {corners_path}")
        raise FileNotFoundError(f"Expected file: {corners_path}")

    logger.info(f"Loading StatsBomb corners from {corners_path}")
    corners_df = pd.read_csv(corners_path)
    logger.info(f"Loaded {len(corners_df)} corners")

    # Check for receiver columns
    receiver_cols = ['receiver_name', 'receiver_location_x', 'receiver_location_y']
    missing_cols = [col for col in receiver_cols if col not in corners_df.columns]
    if missing_cols:
        logger.error(f"Missing receiver columns: {missing_cols}")
        logger.error("Please re-download StatsBomb data with receiver extraction")
        raise ValueError(f"Missing columns: {missing_cols}")

    return corners_df


def load_existing_graphs(graph_path: Path) -> List[CornerGraph]:
    """Load existing corner graphs."""
    logger.info(f"Loading graphs from {graph_path}")

    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)

    logger.info(f"Loaded {len(graphs)} graphs")
    return graphs


def add_receiver_labels_to_graphs(
    graphs: List[CornerGraph],
    corners_df: pd.DataFrame
) -> tuple[List[CornerGraph], Dict]:
    """
    Add receiver labels to corner graphs using Ball Receipt location matching.

    Args:
        graphs: List of CornerGraph objects
        corners_df: StatsBomb corners DataFrame with receiver_name and receiver_location_x/y

    Returns:
        Tuple of (updated_graphs, statistics)
    """
    # Create lookup dictionary: corner_id -> (receiver_name, receiver_location)
    corner_lookup = {}
    for _, row in corners_df.iterrows():
        corner_id = row['corner_id']
        receiver_name = row.get('receiver_name')
        receiver_x = row.get('receiver_location_x')
        receiver_y = row.get('receiver_location_y')

        # Only add if we have both name and location
        if pd.notna(receiver_name) and pd.notna(receiver_x) and pd.notna(receiver_y):
            corner_lookup[corner_id] = {
                'name': receiver_name,
                'location': np.array([receiver_x, receiver_y])
            }
        else:
            corner_lookup[corner_id] = None

    logger.info(f"Built lookup for {len(corner_lookup)} corners")
    logger.info(f"Corners with receiver location: {sum(1 for v in corner_lookup.values() if v is not None)}")

    updated_graphs = []
    stats = {
        'total_graphs': len(graphs),
        'with_receiver': 0,
        'without_receiver': 0,
        'mapped_to_node': 0,
        'failed_mapping': 0,
        'coverage_pct': 0.0,
        'mapping_pct': 0.0,
        'avg_distance': 0.0,
        'distances': []
    }

    logger.info("Adding receiver labels to graphs using location matching...")

    for graph in tqdm(graphs, desc="Processing graphs"):
        # Extract base corner_id (might have temporal suffix like "_t0")
        corner_id = graph.corner_id
        base_corner_id = corner_id.split('_t')[0]  # Remove temporal suffix if present

        # Also try mirror suffix (e.g., "_t0_mirror")
        base_corner_id = base_corner_id.split('_mirror')[0]

        # Look up receiver
        receiver_info = corner_lookup.get(base_corner_id, None)

        # Update graph with receiver info
        if receiver_info is not None:
            receiver_name = receiver_info['name']
            receiver_location = receiver_info['location']

            # Store receiver info in graph
            graph.receiver_player_name = receiver_name
            graph.receiver_location = receiver_location

            # Find closest attacking player to receiver location
            # Node features: [x, y, ...]
            positions = graph.node_features[:, :2]  # First 2 dims are x, y

            # Filter to attacking team only (assuming graph.teams exists)
            if hasattr(graph, 'teams') and graph.teams is not None:
                attacking_indices = [i for i, team in enumerate(graph.teams) if team == 'attacking']
            else:
                # Fallback: assume first half of players are attacking (not ideal)
                attacking_indices = list(range(len(positions) // 2))

            if len(attacking_indices) > 0:
                attacking_positions = positions[attacking_indices]

                # Calculate distances to receiver location
                distances = np.linalg.norm(attacking_positions - receiver_location, axis=1)
                closest_attacking_idx = attacking_indices[np.argmin(distances)]
                min_distance = np.min(distances)

                graph.receiver_node_index = closest_attacking_idx

                stats['mapped_to_node'] += 1
                stats['distances'].append(min_distance)
            else:
                # No attacking players found
                graph.receiver_node_index = None
                stats['failed_mapping'] += 1

            stats['with_receiver'] += 1
        else:
            graph.receiver_player_name = None
            graph.receiver_location = None
            graph.receiver_node_index = None

            stats['without_receiver'] += 1

        updated_graphs.append(graph)

    stats['coverage_pct'] = (stats['with_receiver'] / stats['total_graphs']) * 100
    stats['mapping_pct'] = (stats['mapped_to_node'] / stats['with_receiver']) * 100 if stats['with_receiver'] > 0 else 0.0
    stats['avg_distance'] = np.mean(stats['distances']) if stats['distances'] else 0.0

    return updated_graphs, stats


def main():
    """Main execution function."""
    logger.info("=== Adding Receiver Labels to Corner Graphs ===")
    logger.info("Using Ball Receipt location matching approach")

    # Paths - Use StatsBomb temporal augmented graphs
    input_graph_path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented.pkl")
    output_graph_path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl")

    # Check if input exists
    if not input_graph_path.exists():
        logger.error(f"Input graph file not found: {input_graph_path}")
        logger.error("Please run: sbatch scripts/slurm/phase2_4_statsbomb_augment.sh")
        return

    # Load data
    corners_df = load_statsbomb_corners()
    graphs = load_existing_graphs(input_graph_path)

    # Add receiver labels
    updated_graphs, stats = add_receiver_labels_to_graphs(graphs, corners_df)

    # Save updated graphs
    logger.info(f"\nSaving updated graphs to {output_graph_path}")
    with open(output_graph_path, 'wb') as f:
        pickle.dump(updated_graphs, f)

    # Print statistics
    logger.info("\n" + "="*70)
    logger.info("RECEIVER LABEL STATISTICS")
    logger.info("="*70)
    logger.info(f"Total graphs: {stats['total_graphs']}")
    logger.info(f"Graphs with receiver location: {stats['with_receiver']}")
    logger.info(f"Graphs without receiver location: {stats['without_receiver']}")
    logger.info(f"Successfully mapped to node: {stats['mapped_to_node']}")
    logger.info(f"Failed mapping (no attacking players): {stats['failed_mapping']}")
    logger.info(f"")
    logger.info(f"Coverage: {stats['coverage_pct']:.1f}%")
    logger.info(f"Mapping success rate: {stats['mapping_pct']:.1f}%")
    logger.info(f"Average distance to matched position: {stats['avg_distance']:.2f}m")
    logger.info("="*70)

    # Check success criteria (adjusted for 668 corners with receiver locations)
    # With 5 temporal frames + mirrors = 10x augmentation
    # Expected: 668 * 10 = 6,680 graphs with receivers
    expected_with_receiver = 668 * 10
    if stats['with_receiver'] >= expected_with_receiver * 0.85:
        logger.info("✅ SUCCESS: Achieved expected receiver coverage")
    else:
        logger.warning(f"⚠️ WARNING: Lower than expected receiver coverage")
        logger.warning(f"   Expected: ~{expected_with_receiver} graphs with receivers")
        logger.warning(f"   Actual: {stats['with_receiver']} graphs with receivers")

    if stats['mapping_pct'] >= 95.0:
        logger.info("✅ SUCCESS: High mapping success rate (>95%)")
    else:
        logger.warning(f"⚠️ WARNING: Mapping success rate below 95%")

    logger.info(f"\nOutput saved to: {output_graph_path}")


if __name__ == "__main__":
    main()
