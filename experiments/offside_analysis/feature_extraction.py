"""Feature extraction for offside signal investigation.

Extracts spatial features from freeze-frame data that could predict
offside outcomes in corner kicks.
"""

from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


def get_defenders(freeze_frame: List[Dict], exclude_goalkeeper: bool = True) -> List[Dict]:
    """Get defender players from freeze frame.

    Args:
        freeze_frame: List of player dictionaries
        exclude_goalkeeper: Whether to exclude the goalkeeper

    Returns:
        List of defender player dictionaries
    """
    defenders = []
    for player in freeze_frame:
        if not player['teammate']:  # Defender
            if exclude_goalkeeper and player['keeper']:
                continue
            defenders.append(player)
    return defenders


def get_attackers(freeze_frame: List[Dict], exclude_corner_taker: bool = True) -> List[Dict]:
    """Get attacker players from freeze frame.

    Args:
        freeze_frame: List of player dictionaries
        exclude_corner_taker: Whether to exclude the corner taker

    Returns:
        List of attacker player dictionaries
    """
    attackers = []
    for player in freeze_frame:
        if player['teammate']:  # Attacker
            if exclude_corner_taker and player.get('actor', False):
                continue
            attackers.append(player)
    return attackers


def find_last_defender_x(freeze_frame: List[Dict]) -> Optional[float]:
    """Find x-position of last defender (excluding goalkeeper).

    The "last defender" is the defender furthest from goal (lowest x).
    In corners, this defines the offside line.

    Args:
        freeze_frame: List of player dictionaries

    Returns:
        X-position of last defender, or None if no outfield defenders
    """
    defenders = get_defenders(freeze_frame, exclude_goalkeeper=True)

    if not defenders:
        return None

    # Last defender = highest x (closest to goal, defines offside line)
    # Note: In StatsBomb coords, goal is at x=120
    x_positions = [d['location'][0] for d in defenders]
    return max(x_positions)


def compute_defensive_line_spread(freeze_frame: List[Dict]) -> float:
    """Compute y-axis spread of defensive line.

    Args:
        freeze_frame: List of player dictionaries

    Returns:
        Y-spread (max_y - min_y) of defenders
    """
    defenders = get_defenders(freeze_frame, exclude_goalkeeper=True)

    if len(defenders) < 2:
        return 0.0

    y_positions = [d['location'][1] for d in defenders]
    return max(y_positions) - min(y_positions)


def compute_defensive_compactness(freeze_frame: List[Dict]) -> float:
    """Compute compactness of defensive line (std dev of x-positions).

    Args:
        freeze_frame: List of player dictionaries

    Returns:
        Standard deviation of defender x-positions
    """
    defenders = get_defenders(freeze_frame, exclude_goalkeeper=True)

    if len(defenders) < 2:
        return 0.0

    x_positions = [d['location'][0] for d in defenders]
    return float(np.std(x_positions))


def count_attackers_beyond_defender(freeze_frame: List[Dict]) -> int:
    """Count attackers positioned beyond (higher x than) last defender.

    Args:
        freeze_frame: List of player dictionaries

    Returns:
        Number of attackers beyond the offside line
    """
    last_defender_x = find_last_defender_x(freeze_frame)

    if last_defender_x is None:
        return 0

    attackers = get_attackers(freeze_frame, exclude_corner_taker=True)

    count = 0
    for attacker in attackers:
        if attacker['location'][0] > last_defender_x:
            count += 1

    return count


def find_furthest_attacker_x(
    freeze_frame: List[Dict],
    exclude_corner_taker: bool = True,
) -> Optional[float]:
    """Find x-position of furthest forward attacker.

    Args:
        freeze_frame: List of player dictionaries
        exclude_corner_taker: Whether to exclude corner taker

    Returns:
        X-position of furthest forward attacker
    """
    attackers = get_attackers(freeze_frame, exclude_corner_taker=exclude_corner_taker)

    if not attackers:
        return None

    x_positions = [a['location'][0] for a in attackers]
    return max(x_positions)


def compute_attacker_defender_gap(freeze_frame: List[Dict]) -> float:
    """Compute gap between furthest attacker and last defender.

    Positive = attacker beyond defender (potential offside)
    Negative = attacker behind defender (onside)

    Args:
        freeze_frame: List of player dictionaries

    Returns:
        Distance in x-direction (attacker_x - defender_x)
    """
    last_defender_x = find_last_defender_x(freeze_frame)
    furthest_attacker_x = find_furthest_attacker_x(freeze_frame)

    if last_defender_x is None or furthest_attacker_x is None:
        return 0.0

    return furthest_attacker_x - last_defender_x


def count_attackers_in_offside_zone(freeze_frame: List[Dict]) -> int:
    """Count attackers in potential offside zone.

    Offside zone: x > last_defender_x AND x < 120 (goal line)

    Args:
        freeze_frame: List of player dictionaries

    Returns:
        Number of attackers in offside zone
    """
    last_defender_x = find_last_defender_x(freeze_frame)

    if last_defender_x is None:
        return 0

    attackers = get_attackers(freeze_frame, exclude_corner_taker=True)

    count = 0
    for attacker in attackers:
        x = attacker['location'][0]
        # In offside zone: beyond defender but not at goal line
        if x > last_defender_x and x < 120.0:
            count += 1

    return count


def extract_offside_features(corner: Dict[str, Any]) -> Dict[str, float]:
    """Extract all offside-related features from a corner.

    Args:
        corner: Corner dictionary with freeze_frame

    Returns:
        Dictionary of feature name -> value
    """
    freeze_frame = corner.get('freeze_frame', [])

    # Handle empty freeze frame
    if not freeze_frame:
        return {
            'last_defender_x': np.nan,
            'defensive_line_spread': np.nan,
            'defensive_compactness': np.nan,
            'attackers_beyond_defender': np.nan,
            'furthest_attacker_x': np.nan,
            'attacker_defender_gap': np.nan,
            'attackers_in_offside_zone': np.nan,
            'num_defenders': np.nan,
            'num_attackers': np.nan,
        }

    # Extract individual features
    last_def_x = find_last_defender_x(freeze_frame)
    furthest_att_x = find_furthest_attacker_x(freeze_frame)

    features = {
        'last_defender_x': last_def_x if last_def_x is not None else np.nan,
        'defensive_line_spread': compute_defensive_line_spread(freeze_frame),
        'defensive_compactness': compute_defensive_compactness(freeze_frame),
        'attackers_beyond_defender': count_attackers_beyond_defender(freeze_frame),
        'furthest_attacker_x': furthest_att_x if furthest_att_x is not None else np.nan,
        'attacker_defender_gap': compute_attacker_defender_gap(freeze_frame),
        'attackers_in_offside_zone': count_attackers_in_offside_zone(freeze_frame),
        'num_defenders': len(get_defenders(freeze_frame, exclude_goalkeeper=True)),
        'num_attackers': len(get_attackers(freeze_frame, exclude_corner_taker=True)),
    }

    return features


def extract_features_batch(corners: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract offside features from a list of corners.

    Args:
        corners: List of corner dictionaries

    Returns:
        DataFrame with one row per corner, columns for each feature
    """
    all_features = []

    for corner in corners:
        features = extract_offside_features(corner)
        # Add corner metadata
        features['match_id'] = corner.get('match_id', '')
        features['shot_outcome'] = corner.get('shot_outcome', np.nan)
        all_features.append(features)

    return pd.DataFrame(all_features)
