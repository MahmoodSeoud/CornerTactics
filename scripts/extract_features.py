"""
Task 3: Feature Engineering

Extracts 27 features from corner kick freeze frame data.
"""

import numpy as np


# StatsBomb pitch dimensions and zones
PITCH_LENGTH = 120
PITCH_WIDTH = 80
GOAL_CENTER = (120, 40)

# Penalty box: x > 102, 18 < y < 62
PENALTY_BOX = {'x_min': 102, 'y_min': 18, 'y_max': 62}

# Near goal area: x > 108, 30 < y < 50
NEAR_GOAL = {'x_min': 108, 'y_min': 30, 'y_max': 50}


def extract_basic_metadata(event):
    """
    Extract basic corner metadata (5 features)

    Args:
        event: Corner event dictionary

    Returns:
        dict: 5 features (corner_side, period, minute, corner_x, corner_y)
    """
    location = event['location']

    return {
        'corner_side': 0 if location[1] < 40 else 1,  # Left=0, Right=1
        'period': event['period'],
        'minute': event['minute'],
        'corner_x': location[0],
        'corner_y': location[1]
    }


def extract_player_counts(freeze_frame):
    """
    Extract player count features (6 features)

    Args:
        freeze_frame: List of player positions

    Returns:
        dict: 6 features (total counts, box counts, near goal counts)
    """
    attacking_players = [p for p in freeze_frame if p['teammate']]
    defending_players = [p for p in freeze_frame if not p['teammate']]

    def in_penalty_box(location):
        x, y = location
        return x > PENALTY_BOX['x_min'] and PENALTY_BOX['y_min'] < y < PENALTY_BOX['y_max']

    def near_goal(location):
        x, y = location
        return x > NEAR_GOAL['x_min'] and NEAR_GOAL['y_min'] < y < NEAR_GOAL['y_max']

    attacking_in_box = sum(1 for p in attacking_players if in_penalty_box(p['location']))
    defending_in_box = sum(1 for p in defending_players if in_penalty_box(p['location']))

    attacking_near_goal_count = sum(1 for p in attacking_players if near_goal(p['location']))
    defending_near_goal_count = sum(1 for p in defending_players if near_goal(p['location']))

    return {
        'total_attacking': len(attacking_players),
        'total_defending': len(defending_players),
        'attacking_in_box': attacking_in_box,
        'defending_in_box': defending_in_box,
        'attacking_near_goal': attacking_near_goal_count,
        'defending_near_goal': defending_near_goal_count
    }


def extract_spatial_density(freeze_frame):
    """
    Extract spatial density features (4 features)

    Args:
        freeze_frame: List of player positions

    Returns:
        dict: 4 features (densities, advantage, ratio)
    """
    # Penalty box area: (120 - 102) * (62 - 18) = 18 * 44 = 792
    box_area = (PITCH_LENGTH - PENALTY_BOX['x_min']) * (PENALTY_BOX['y_max'] - PENALTY_BOX['y_min'])

    attacking_players = [p for p in freeze_frame if p['teammate']]
    defending_players = [p for p in freeze_frame if not p['teammate']]

    def in_penalty_box(location):
        x, y = location
        return x > PENALTY_BOX['x_min'] and PENALTY_BOX['y_min'] < y < PENALTY_BOX['y_max']

    attacking_in_box = sum(1 for p in attacking_players if in_penalty_box(p['location']))
    defending_in_box = sum(1 for p in defending_players if in_penalty_box(p['location']))

    attacking_density = attacking_in_box / box_area
    defending_density = defending_in_box / box_area

    numerical_advantage = attacking_in_box - defending_in_box

    # Handle division by zero for ratio
    if defending_in_box == 0:
        ratio = float(attacking_in_box) if attacking_in_box > 0 else 0.0
        ratio = max(ratio, 100.0)  # Cap at high value instead of inf
    else:
        ratio = attacking_in_box / defending_in_box

    return {
        'attacking_density': attacking_density,
        'defending_density': defending_density,
        'numerical_advantage': numerical_advantage,
        'attacker_defender_ratio': ratio
    }


def extract_positional_features(freeze_frame):
    """
    Extract positional features (8 features)

    Args:
        freeze_frame: List of player positions

    Returns:
        dict: 8 features (centroids, compactness, depth, distances)
    """
    attacking_players = [p for p in freeze_frame if p['teammate']]
    defending_players = [p for p in freeze_frame if not p['teammate']]

    # Calculate centroids
    if attacking_players:
        attacking_positions = np.array([p['location'] for p in attacking_players])
        attacking_centroid = attacking_positions.mean(axis=0)
        attacking_centroid_x, attacking_centroid_y = attacking_centroid
    else:
        attacking_centroid_x, attacking_centroid_y = 0.0, 0.0

    if defending_players:
        defending_positions = np.array([p['location'] for p in defending_players])
        defending_centroid = defending_positions.mean(axis=0)
        defending_centroid_x, defending_centroid_y = defending_centroid
    else:
        defending_centroid_x, defending_centroid_y = 0.0, 0.0

    # Defending compactness (std of y positions)
    if len(defending_players) > 0:
        defending_y_positions = [p['location'][1] for p in defending_players]
        defending_compactness = np.std(defending_y_positions)
    else:
        defending_compactness = 0.0

    # Defending depth (max x - min x)
    if len(defending_players) > 0:
        defending_x_positions = [p['location'][0] for p in defending_players]
        defending_depth = max(defending_x_positions) - min(defending_x_positions)
    else:
        defending_depth = 0.0

    # Distance to goal center (120, 40)
    attacking_to_goal_dist = np.sqrt(
        (attacking_centroid_x - GOAL_CENTER[0])**2 +
        (attacking_centroid_y - GOAL_CENTER[1])**2
    )

    defending_to_goal_dist = np.sqrt(
        (defending_centroid_x - GOAL_CENTER[0])**2 +
        (defending_centroid_y - GOAL_CENTER[1])**2
    )

    return {
        'attacking_centroid_x': float(attacking_centroid_x),
        'attacking_centroid_y': float(attacking_centroid_y),
        'defending_centroid_x': float(defending_centroid_x),
        'defending_centroid_y': float(defending_centroid_y),
        'defending_compactness': float(defending_compactness),
        'defending_depth': float(defending_depth),
        'attacking_to_goal_dist': float(attacking_to_goal_dist),
        'defending_to_goal_dist': float(defending_to_goal_dist)
    }


def extract_pass_trajectory(event):
    """
    Extract pass trajectory features (4 features)

    Args:
        event: Corner event dictionary

    Returns:
        dict: 4 features (end_x, end_y, length, height)
    """
    pass_data = event['pass']

    # Pass height encoding: Ground=0, Low=1, High=2
    height_mapping = {
        'Ground Pass': 0,
        'Low Pass': 1,
        'High Pass': 2
    }

    height_name = pass_data['height']['name']
    pass_height = height_mapping.get(height_name, 1)  # Default to Low if unknown

    return {
        'pass_end_x': pass_data['end_location'][0],
        'pass_end_y': pass_data['end_location'][1],
        'pass_length': pass_data['length'],
        'pass_height': pass_height
    }


def extract_all_features(corner):
    """
    Extract all 27 features from a corner

    Args:
        corner: Dictionary with 'event' and 'freeze_frame' keys

    Returns:
        dict: All 27 features
    """
    features = {}

    # 5 basic metadata features
    features.update(extract_basic_metadata(corner['event']))

    # 6 player count features
    features.update(extract_player_counts(corner['freeze_frame']))

    # 4 spatial density features
    features.update(extract_spatial_density(corner['freeze_frame']))

    # 8 positional features
    features.update(extract_positional_features(corner['freeze_frame']))

    # 4 pass trajectory features
    features.update(extract_pass_trajectory(corner['event']))

    return features
