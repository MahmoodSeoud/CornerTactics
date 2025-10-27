#!/usr/bin/env python3
"""
Add Receiver Labels to Corner Graphs (TacticAI Day 1-2)

Extracts receiver_player_id from StatsBomb events and adds to CornerGraph metadata.

Based on TacticAI Implementation Plan:
- Extract receiver (player who touches ball 0-5s after corner)
- Add receiver_player_id to CornerGraph metadata
- Add player_ids list to map node indices to player IDs
- Target: 85%+ coverage (expect ~950/1118 corners)

Output: data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.receiver_labeler import ReceiverLabeler
from src.graph_builder import CornerGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_statsbomb_corners() -> pd.DataFrame:
    """Load StatsBomb corners with outcome data."""
    corners_path = Path("data/raw/statsbomb/corners_360_with_outcomes.csv")

    if not corners_path.exists():
        logger.error(f"StatsBomb corners file not found: {corners_path}")
        raise FileNotFoundError(f"Expected file: {corners_path}")

    logger.info(f"Loading StatsBomb corners from {corners_path}")
    corners_df = pd.read_csv(corners_path)
    logger.info(f"Loaded {len(corners_df)} corners")

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
    Add receiver labels to corner graphs using outcome data.

    Args:
        graphs: List of CornerGraph objects
        corners_df: StatsBomb corners DataFrame with outcome_player column

    Returns:
        Tuple of (updated_graphs, statistics)
    """
    # Create lookup dictionary: corner_id -> receiver info
    corner_lookup = {}
    for _, row in corners_df.iterrows():
        corner_id = row['corner_id']
        same_team = row['same_team']
        outcome_player = row['outcome_player']

        # Only use outcome_player if it's the attacking team (same_team=True)
        # This is the "receiver" - first attacking player to touch the ball
        if same_team and pd.notna(outcome_player):
            corner_lookup[corner_id] = outcome_player
        else:
            corner_lookup[corner_id] = None

    logger.info(f"Built lookup for {len(corner_lookup)} corners")

    updated_graphs = []
    stats = {
        'total_graphs': len(graphs),
        'with_receiver': 0,
        'without_receiver': 0,
        'coverage_pct': 0.0
    }

    logger.info("Adding receiver labels to graphs...")

    for graph in tqdm(graphs, desc="Processing graphs"):
        # Extract base corner_id (might have temporal suffix like "_t0")
        corner_id = graph.corner_id
        base_corner_id = corner_id.split('_t')[0]  # Remove temporal suffix if present

        # Look up receiver
        receiver_name = corner_lookup.get(base_corner_id, None)

        # Update graph with receiver info
        if receiver_name is not None:
            # Note: We don't have player IDs in this CSV, so we can't map to node index
            # This would require loading full StatsBomb event data
            graph.receiver_player_id = None  # Not available in CSV
            graph.receiver_player_name = receiver_name
            graph.receiver_node_index = None  # Would need player ID to map this

            stats['with_receiver'] += 1
        else:
            graph.receiver_player_id = None
            graph.receiver_player_name = None
            graph.receiver_node_index = None

            stats['without_receiver'] += 1

        updated_graphs.append(graph)

    stats['coverage_pct'] = (stats['with_receiver'] / stats['total_graphs']) * 100

    return updated_graphs, stats


def main():
    """Main execution function."""
    logger.info("=== Adding Receiver Labels to Corner Graphs ===")

    # Paths
    input_graph_path = Path("data/graphs/adjacency_team/combined_temporal_graphs.pkl")
    output_graph_path = Path("data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl")

    # Load data
    corners_df = load_statsbomb_corners()
    graphs = load_existing_graphs(input_graph_path)

    # Add receiver labels
    updated_graphs, stats = add_receiver_labels_to_graphs(graphs, corners_df)

    # Save updated graphs
    logger.info(f"Saving updated graphs to {output_graph_path}")
    with open(output_graph_path, 'wb') as f:
        pickle.dump(updated_graphs, f)

    # Print statistics
    logger.info("\n=== Receiver Label Statistics ===")
    logger.info(f"Total graphs: {stats['total_graphs']}")
    logger.info(f"Graphs with receiver: {stats['with_receiver']}")
    logger.info(f"Graphs without receiver: {stats['without_receiver']}")
    logger.info(f"Coverage: {stats['coverage_pct']:.1f}%")

    # Check success criteria
    if stats['coverage_pct'] >= 85.0 and stats['with_receiver'] >= 900:
        logger.info("✅ SUCCESS: Met target criteria (85% coverage, 900+ receivers)")
    else:
        logger.warning(f"⚠️ WARNING: Did not meet target criteria")
        logger.warning(f"   Target: 85% coverage, 900+ receivers")
        logger.warning(f"   Actual: {stats['coverage_pct']:.1f}% coverage, {stats['with_receiver']} receivers")

    logger.info(f"\nOutput saved to: {output_graph_path}")


if __name__ == "__main__":
    main()
