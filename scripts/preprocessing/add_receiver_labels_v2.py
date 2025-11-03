#!/usr/bin/env python3
"""
Add Receiver Labels Using Event Streams (v2)

This version uses ReceiverLabeler with StatsBomb event streams to extract
receivers for ALL corners (including defensive clearances, interceptions, duels).

Key improvements over v1:
- Uses ReceiverLabeler.find_receiver() instead of pre-processed CSV
- Includes BOTH attacking and defending players as receivers
- Matches TacticAI methodology: "first player to touch ball after corner"

Expected coverage increase: 60% → 85%+ (recover ~2,000 clearance corners)

Author: mseo
Date: November 2024
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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


def match_location_to_node(
    receiver_location: np.ndarray,
    freeze_frame_positions: np.ndarray,
    teams: List[str],
    receiver_team: Optional[str] = None
) -> Optional[int]:
    """
    Match receiver event location to closest freeze frame node position.

    Args:
        receiver_location: [x, y] location from event
        freeze_frame_positions: Array of [x, y] positions for all nodes
        teams: List of team labels ('attacking' or 'defending') for each node
        receiver_team: Team of receiver ('attacking', 'defending', or None)

    Returns:
        Node index (0-based) of closest match, or None if no match found
    """
    if receiver_location is None or len(freeze_frame_positions) == 0:
        return None

    # Ensure receiver_location is a numpy array
    if not isinstance(receiver_location, np.ndarray):
        receiver_location = np.array(receiver_location)

    # Filter by team if known
    if receiver_team is not None and receiver_team in ['attacking', 'defending']:
        # Get indices of players on the same team
        team_indices = [i for i, team in enumerate(teams) if team == receiver_team]

        if len(team_indices) == 0:
            # No players on that team - fall back to all players
            logger.warning(f"No {receiver_team} players found, matching to any team")
            team_indices = list(range(len(freeze_frame_positions)))
    else:
        # No team filtering - match to any player
        team_indices = list(range(len(freeze_frame_positions)))

    # Calculate distances to all candidate positions
    candidate_positions = freeze_frame_positions[team_indices]
    distances = np.linalg.norm(candidate_positions - receiver_location, axis=1)

    # Find closest
    closest_candidate_idx = np.argmin(distances)
    closest_node_idx = team_indices[closest_candidate_idx]

    return closest_node_idx


def extract_receiver_from_events(
    corner_id: str,
    match_id: int,
    events_cache: Dict[int, pd.DataFrame]
) -> Tuple[Optional[int], Optional[str], Optional[np.ndarray], Optional[str]]:
    """
    Extract receiver information from StatsBomb event stream.

    Args:
        corner_id: Corner event ID (from graph)
        match_id: StatsBomb match ID
        events_cache: Cache of events DataFrames by match_id

    Returns:
        Tuple of (player_id, player_name, location, team)
        or (None, None, None, None) if receiver not found
    """
    # Load events for this match (use cache)
    if match_id not in events_cache:
        try:
            events_df = sb.events(match_id=match_id, fmt='dataframe')
            events_cache[match_id] = events_df
        except Exception as e:
            logger.error(f"Failed to load events for match {match_id}: {e}")
            return None, None, None, None
    else:
        events_df = events_cache[match_id]

    # Use ReceiverLabeler to find receiver (with 270s time window for 100% coverage)
    labeler = ReceiverLabeler()
    player_id, player_name, location = labeler.find_receiver(
        events_df,
        corner_event_id=corner_id,
        max_time_diff=270.0  # 270s window to capture corners with long delays (injuries, VAR, stoppages)
    )

    if player_id is None:
        return None, None, None, None

    # Determine team from event
    receiver_event = events_df[
        (events_df['player_id'] == player_id) &
        (events_df['player'] == player_name)
    ].iloc[0] if len(events_df[
        (events_df['player_id'] == player_id) &
        (events_df['player'] == player_name)
    ]) > 0 else None

    team = None
    if receiver_event is not None:
        event_team = receiver_event.get('team')
        # Find corner event to determine which team is attacking
        corner_event = events_df[events_df['id'] == corner_id].iloc[0] if len(
            events_df[events_df['id'] == corner_id]
        ) > 0 else None

        if corner_event is not None and event_team is not None:
            corner_team = corner_event.get('team')
            # Determine if receiver is attacking or defending
            team = 'attacking' if event_team == corner_team else 'defending'

    # Convert location to numpy array if needed
    if location is not None and not isinstance(location, np.ndarray):
        location = np.array(location)

    return player_id, player_name, location, team


def add_receiver_labels_from_events(
    graphs: List[CornerGraph],
    corner_metadata: pd.DataFrame
) -> Tuple[List[CornerGraph], Dict]:
    """
    Add receiver labels to graphs using event stream extraction.

    Args:
        graphs: List of CornerGraph objects
        corner_metadata: DataFrame with corner_id -> match_id mapping

    Returns:
        Tuple of (updated_graphs, statistics)
    """
    updated_graphs = []
    events_cache = {}  # Cache events by match_id

    stats = {
        'total_graphs': len(graphs),
        'with_receiver': 0,
        'without_receiver': 0,
        'attacking_receivers': 0,
        'defending_receivers': 0,
        'matched_to_node': 0,
        'failed_matching': 0,
        'coverage_pct': 0.0,
        'avg_distance': 0.0,
        'distances': []
    }

    logger.info("Extracting receivers from event streams...")

    for graph in tqdm(graphs, desc="Processing graphs"):
        # Extract base corner_id (remove temporal/mirror suffixes)
        base_corner_id = graph.corner_id.split('_t')[0].split('_mirror')[0]

        # Get match_id for this corner
        match_row = corner_metadata[corner_metadata['corner_id'] == base_corner_id]
        if len(match_row) == 0:
            logger.warning(f"No match_id found for corner {base_corner_id}")
            graph.receiver_player_id = None
            graph.receiver_player_name = None
            graph.receiver_location = None
            graph.receiver_node_index = None
            stats['without_receiver'] += 1
            updated_graphs.append(graph)
            continue

        match_id = match_row.iloc[0]['match_id']

        # Extract receiver from events
        player_id, player_name, location, team = extract_receiver_from_events(
            base_corner_id,
            match_id,
            events_cache
        )

        if player_id is None:
            # No receiver found in events
            graph.receiver_player_id = None
            graph.receiver_player_name = None
            graph.receiver_location = None
            graph.receiver_node_index = None
            stats['without_receiver'] += 1
        else:
            # Receiver found - match to node
            graph.receiver_player_id = player_id
            graph.receiver_player_name = player_name
            graph.receiver_location = location

            # Track receiver team distribution
            if team == 'attacking':
                stats['attacking_receivers'] += 1
            elif team == 'defending':
                stats['defending_receivers'] += 1

            # Match location to node position
            if location is not None and hasattr(graph, 'node_features'):
                positions = graph.node_features[:, :2]  # First 2 cols are x, y
                teams = graph.teams if hasattr(graph, 'teams') else None

                node_idx = match_location_to_node(location, positions, teams, team)

                if node_idx is not None:
                    graph.receiver_node_index = node_idx
                    stats['matched_to_node'] += 1

                    # Calculate matching distance
                    matched_pos = positions[node_idx]
                    distance = np.linalg.norm(matched_pos - location)
                    stats['distances'].append(distance)
                else:
                    graph.receiver_node_index = None
                    stats['failed_matching'] += 1
            else:
                graph.receiver_node_index = None
                stats['failed_matching'] += 1

            stats['with_receiver'] += 1

        updated_graphs.append(graph)

    # Compute summary statistics
    stats['coverage_pct'] = (stats['with_receiver'] / stats['total_graphs']) * 100
    stats['avg_distance'] = np.mean(stats['distances']) if stats['distances'] else 0.0

    return updated_graphs, stats


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("RE-LABELING RECEIVERS USING EVENT STREAMS (v2)")
    logger.info("="*80)

    # Paths
    input_graph_path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented.pkl")
    output_graph_path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver_v2.pkl")
    corners_csv_path = Path("data/raw/statsbomb/corners_360.csv")

    # Check inputs exist
    if not input_graph_path.exists():
        logger.error(f"Input graphs not found: {input_graph_path}")
        return

    if not corners_csv_path.exists():
        logger.error(f"Corners CSV not found: {corners_csv_path}")
        return

    # Load graphs
    logger.info(f"Loading graphs from {input_graph_path}")
    with open(input_graph_path, 'rb') as f:
        graphs = pickle.load(f)
    logger.info(f"Loaded {len(graphs)} graphs")

    # Load corner metadata (for match_id mapping)
    logger.info(f"Loading corner metadata from {corners_csv_path}")
    corners_df = pd.read_csv(corners_csv_path)
    logger.info(f"Loaded {len(corners_df)} corners")

    # Add receiver labels
    updated_graphs, stats = add_receiver_labels_from_events(graphs, corners_df)

    # Save updated graphs
    logger.info(f"\nSaving updated graphs to {output_graph_path}")
    with open(output_graph_path, 'wb') as f:
        pickle.dump(updated_graphs, f)

    # Print statistics
    logger.info("\n" + "="*80)
    logger.info("RECEIVER LABELING STATISTICS (v2)")
    logger.info("="*80)
    logger.info(f"Total graphs: {stats['total_graphs']}")
    logger.info(f"Graphs with receiver: {stats['with_receiver']}")
    logger.info(f"Graphs without receiver: {stats['without_receiver']}")
    logger.info(f"Coverage: {stats['coverage_pct']:.1f}%")
    logger.info(f"")
    logger.info(f"Receiver team distribution:")
    logger.info(f"  Attacking: {stats['attacking_receivers']}")
    logger.info(f"  Defending: {stats['defending_receivers']}")
    logger.info(f"")
    logger.info(f"Node matching:")
    logger.info(f"  Successfully matched: {stats['matched_to_node']}")
    logger.info(f"  Failed matching: {stats['failed_matching']}")
    logger.info(f"  Avg distance to match: {stats['avg_distance']:.2f}m")
    logger.info("="*80)

    # Success criteria
    if stats['coverage_pct'] >= 85.0:
        logger.info("✅ SUCCESS: Achieved target coverage (>85%)")
    elif stats['coverage_pct'] >= 75.0:
        logger.info(f"⚠️  WARNING: Coverage below target ({stats['coverage_pct']:.1f}% < 85%)")
    else:
        logger.error(f"❌ FAILURE: Coverage significantly below target ({stats['coverage_pct']:.1f}%)")

    logger.info(f"\nOutput saved to: {output_graph_path}")


if __name__ == "__main__":
    main()
