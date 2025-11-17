"""
Task 3: Feature Engineering

Extracts 49 features from corner kick freeze frame data and match context.

Feature Categories:
- Basic metadata (5)
- Temporal (3)
- Player counts (6)
- Spatial density (4)
- Positional (8)
- Pass trajectory (4)
- Pass technique & body part (5)
- Pass outcome & context (4)
- Goalkeeper features (3)
- Score state (4)
- Substitution patterns (3)
"""

import numpy as np
import json
from pathlib import Path


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


def extract_temporal_features(event):
    """
    Extract temporal features (3 features)

    Args:
        event: Corner event dictionary

    Returns:
        dict: 3 features (second, timestamp_seconds, duration)
    """
    # Convert timestamp HH:MM:SS.mmm to total seconds
    timestamp_str = event['timestamp']
    time_parts = timestamp_str.split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])
    timestamp_seconds = hours * 3600 + minutes * 60 + seconds

    return {
        'second': event['second'],
        'timestamp_seconds': float(timestamp_seconds),
        'duration': event['duration']
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


def extract_pass_technique_features(event):
    """
    Extract pass technique and body part features (5 features)

    Args:
        event: Corner event dictionary

    Returns:
        dict: 5 features (angle, body_part, technique, is_inswinging, is_outswinging)
    """
    pass_data = event['pass']

    # Pass angle (already in radians)
    pass_angle = pass_data['angle']

    # Body part encoding: Right Foot=0, Left Foot=1
    # Note: Only these 2 values observed in corners
    body_part_mapping = {
        'Right Foot': 0,
        'Left Foot': 1
    }
    body_part_name = pass_data['body_part']['name']
    pass_body_part = body_part_mapping.get(body_part_name, 0)

    # Technique encoding: Inswinging=0, Outswinging=1, Straight=2, null=3
    technique_mapping = {
        'Inswinging': 0,
        'Outswinging': 1,
        'Straight': 2
    }
    technique_obj = pass_data.get('technique', None)
    if technique_obj and 'name' in technique_obj:
        pass_technique = technique_mapping.get(technique_obj['name'], 3)
    else:
        pass_technique = 3  # null/unknown

    # Boolean flags (only present when true)
    is_inswinging = 1 if pass_data.get('inswinging', False) else 0
    is_outswinging = 1 if pass_data.get('outswinging', False) else 0

    return {
        'pass_angle': float(pass_angle),
        'pass_body_part': pass_body_part,
        'pass_technique': pass_technique,
        'is_inswinging': is_inswinging,
        'is_outswinging': is_outswinging
    }


def extract_pass_outcome_features(event):
    """
    Extract pass outcome and context features (4 features)

    Args:
        event: Corner event dictionary

    Returns:
        dict: 4 features (outcome, is_switch, has_recipient, is_shot_assist)
    """
    pass_data = event['pass']

    # Outcome encoding: Incomplete=0, Out=1, Pass Offside=2, Unknown=3, null=4
    outcome_mapping = {
        'Incomplete': 0,
        'Out': 1,
        'Pass Offside': 2,
        'Unknown': 3
    }
    outcome_obj = pass_data.get('outcome', None)
    if outcome_obj and 'name' in outcome_obj:
        pass_outcome = outcome_mapping.get(outcome_obj['name'], 4)
    else:
        pass_outcome = 4  # null

    # Boolean flags (only present when true)
    is_cross_field_switch = 1 if pass_data.get('switch', False) else 0
    has_recipient = 1 if pass_data.get('recipient', None) is not None else 0
    is_shot_assist = 1 if pass_data.get('shot_assist', False) else 0

    return {
        'pass_outcome': pass_outcome,
        'is_cross_field_switch': is_cross_field_switch,
        'has_recipient': has_recipient,
        'is_shot_assist': is_shot_assist
    }


def extract_goalkeeper_features(freeze_frame):
    """
    Extract goalkeeper and special player features (3 features)

    Args:
        freeze_frame: List of player positions

    Returns:
        dict: 3 features (num_attacking_keepers, num_defending_keepers, keeper_distance_to_goal)
    """
    attacking_keepers = [p for p in freeze_frame if p['teammate'] and p['keeper']]
    defending_keepers = [p for p in freeze_frame if not p['teammate'] and p['keeper']]

    num_attacking_keepers = len(attacking_keepers)
    num_defending_keepers = len(defending_keepers)

    # Calculate distance from defending keeper to goal center
    if defending_keepers:
        keeper_location = defending_keepers[0]['location']  # Usually only 1 keeper
        keeper_distance = np.sqrt(
            (keeper_location[0] - GOAL_CENTER[0])**2 +
            (keeper_location[1] - GOAL_CENTER[1])**2
        )
    else:
        keeper_distance = 0.0  # No keeper found (rare)

    return {
        'num_attacking_keepers': num_attacking_keepers,
        'num_defending_keepers': num_defending_keepers,
        'keeper_distance_to_goal': float(keeper_distance)
    }


def extract_match_context_score(corner, match_events=None):
    """
    Extract score state features (4 features)

    Args:
        corner: Dictionary with 'event', 'freeze_frame', and 'match_id' keys
        match_events: List of all events in the match (optional, loaded if not provided)

    Returns:
        dict: 4 features (attacking_team_goals, defending_team_goals, score_difference, match_situation)
    """
    if match_events is None:
        # Load match events if not provided
        match_id = corner['match_id']
        base_dir = Path(__file__).parent.parent
        events_file = base_dir / f'data/statsbomb/events/events/{match_id}.json'

        if not events_file.exists():
            # Return zeros if match file not found
            return {
                'attacking_team_goals': 0,
                'defending_team_goals': 0,
                'score_difference': 0,
                'match_situation': 0
            }

        with open(events_file, 'r') as f:
            match_events = json.load(f)

    corner_event = corner['event']
    corner_index = corner_event['index']
    corner_team_id = corner_event['team']['id']

    # Find all goals before this corner
    goals_before = [
        e for e in match_events
        if e['index'] < corner_index
        and e.get('type', {}).get('name') == 'Shot'
        and e.get('shot', {}).get('outcome', {}).get('name') == 'Goal'
    ]

    # Count goals by team
    attacking_team_goals = sum(1 for g in goals_before if g['team']['id'] == corner_team_id)
    defending_team_goals = len(goals_before) - attacking_team_goals

    score_difference = attacking_team_goals - defending_team_goals

    # Match situation: winning=1, drawing=0, losing=-1
    if score_difference > 0:
        match_situation = 1
    elif score_difference < 0:
        match_situation = -1
    else:
        match_situation = 0

    return {
        'attacking_team_goals': attacking_team_goals,
        'defending_team_goals': defending_team_goals,
        'score_difference': score_difference,
        'match_situation': match_situation
    }


def extract_match_context_substitutions(corner, match_events=None):
    """
    Extract substitution pattern features (3 features)

    Args:
        corner: Dictionary with 'event', 'freeze_frame', and 'match_id' keys
        match_events: List of all events in the match (optional, loaded if not provided)

    Returns:
        dict: 3 features (total_subs_before, recent_subs_5min, minutes_since_last_sub)
    """
    if match_events is None:
        # Load match events if not provided
        match_id = corner['match_id']
        base_dir = Path(__file__).parent.parent
        events_file = base_dir / f'data/statsbomb/events/events/{match_id}.json'

        if not events_file.exists():
            # Return defaults if match file not found
            return {
                'total_subs_before': 0,
                'recent_subs_5min': 0,
                'minutes_since_last_sub': 999.0
            }

        with open(events_file, 'r') as f:
            match_events = json.load(f)

    corner_event = corner['event']
    corner_index = corner_event['index']
    corner_minute = corner_event['minute']

    # Find all substitutions before this corner
    subs_before = [
        e for e in match_events
        if e['index'] < corner_index
        and e.get('type', {}).get('name') == 'Substitution'
    ]

    total_subs_before = len(subs_before)

    # Recent substitutions (last 5 minutes)
    recent_subs_5min = sum(1 for s in subs_before if corner_minute - s['minute'] <= 5)

    # Minutes since last substitution
    if subs_before:
        last_sub = max(subs_before, key=lambda x: x['index'])
        minutes_since_last_sub = float(corner_minute - last_sub['minute'])
    else:
        minutes_since_last_sub = 999.0  # No prior substitutions

    return {
        'total_subs_before': total_subs_before,
        'recent_subs_5min': recent_subs_5min,
        'minutes_since_last_sub': minutes_since_last_sub
    }


def extract_all_features(corner, match_events=None):
    """
    Extract all 49 features from a corner

    Args:
        corner: Dictionary with 'event', 'freeze_frame', and 'match_id' keys
        match_events: List of all events in the match (optional, for match context features)

    Returns:
        dict: All 49 features
    """
    features = {}

    # 5 basic metadata features
    features.update(extract_basic_metadata(corner['event']))

    # 3 temporal features
    features.update(extract_temporal_features(corner['event']))

    # 6 player count features
    features.update(extract_player_counts(corner['freeze_frame']))

    # 4 spatial density features
    features.update(extract_spatial_density(corner['freeze_frame']))

    # 8 positional features
    features.update(extract_positional_features(corner['freeze_frame']))

    # 4 pass trajectory features
    features.update(extract_pass_trajectory(corner['event']))

    # 5 pass technique & body part features
    features.update(extract_pass_technique_features(corner['event']))

    # 4 pass outcome & context features
    features.update(extract_pass_outcome_features(corner['event']))

    # 3 goalkeeper features
    features.update(extract_goalkeeper_features(corner['freeze_frame']))

    # 4 score state features
    features.update(extract_match_context_score(corner, match_events))

    # 3 substitution pattern features
    features.update(extract_match_context_substitutions(corner, match_events))

    return features
