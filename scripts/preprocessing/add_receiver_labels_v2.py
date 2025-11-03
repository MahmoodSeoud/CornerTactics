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


if __name__ == "__main__":
    logger.info("Event-stream-based receiver labeling script loaded successfully")
