"""
Progressive Feature Extraction for Ablation Study

This script implements the 9-step feature addition strategy from ABLATION_STUDY_PLAN.md:

Step 0 (Baseline): Raw features (27)
Step 1: + Player counts (6) = 33 features
Step 2: + Spatial density (4) = 37 features
Step 3: + Positional (8) = 45 features
Step 4: + Pass technique (2) = 47 features
Step 5: + Pass outcome context (4) = 51 features
Step 6: + Goalkeeper (3) = 54 features
Step 7: + Score state (4) = 58 features
Step 8: + Substitutions (3) = 61 features
Step 9: + Metadata (2) = 63 features (FULL)

Each function builds upon the previous step.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from extract_raw_features import extract_raw_features


# ============ STEP 1: Player Counts (6 features) ============

def extract_player_counts(freeze_frame):
    """
    Extract player count features from freeze frame.

    Returns 6 features:
    - total_attacking
    - total_defending
    - attacking_in_box
    - defending_in_box
    - attacking_near_goal
    - defending_near_goal
    """
    attacking_in_box = 0
    defending_in_box = 0
    attacking_near_goal = 0
    defending_near_goal = 0

    for player in freeze_frame:
        x, y = player['location']
        is_teammate = player.get('teammate', False)

        # In box (102 <= x <= 120, 18 <= y <= 62)
        in_box = (102 <= x <= 120) and (18 <= y <= 62)

        # Near goal (110 <= x <= 120, 30 <= y <= 50)
        near_goal = (110 <= x <= 120) and (30 <= y <= 50)

        if is_teammate:
            if in_box:
                attacking_in_box += 1
            if near_goal:
                attacking_near_goal += 1
        else:
            if in_box:
                defending_in_box += 1
            if near_goal:
                defending_near_goal += 1

    # Total counts already in raw features, so we extract detailed counts here
    return {
        'attacking_in_box': attacking_in_box,
        'defending_in_box': defending_in_box,
        'attacking_near_goal': attacking_near_goal,
        'defending_near_goal': defending_near_goal
    }


# ============ STEP 2: Spatial Density (4 features) ============

def extract_spatial_density(freeze_frame):
    """
    Extract spatial density metrics.

    Returns 4 features:
    - attacking_density
    - defending_density
    - numerical_advantage
    - attacker_defender_ratio
    """
    attacking_players = []
    defending_players = []

    for player in freeze_frame:
        x, y = player['location']
        is_teammate = player.get('teammate', False)

        # Only consider players in attacking third (x >= 80)
        if x >= 80:
            if is_teammate:
                attacking_players.append((x, y))
            else:
                defending_players.append((x, y))

    num_attacking = len(attacking_players)
    num_defending = len(defending_players)

    # Density: players per unit area (attacking third is 40x80 = 3200 sq units)
    attacking_density = num_attacking / 3200.0
    defending_density = num_defending / 3200.0

    # Numerical advantage
    numerical_advantage = num_attacking - num_defending

    # Ratio (avoid division by zero)
    attacker_defender_ratio = num_attacking / max(num_defending, 1)

    return {
        'attacking_density': attacking_density,
        'defending_density': defending_density,
        'numerical_advantage': numerical_advantage,
        'attacker_defender_ratio': attacker_defender_ratio
    }


# ============ STEP 3: Positional Features (8 features) ============

def extract_positional_features(freeze_frame):
    """
    Extract team positioning and shape features.

    Returns 8 features:
    - attacking_centroid_x
    - attacking_centroid_y
    - defending_centroid_x
    - defending_centroid_y
    - defending_compactness
    - defending_depth
    - attacking_to_goal_dist
    - defending_to_goal_dist
    """
    attacking_positions = []
    defending_positions = []

    for player in freeze_frame:
        x, y = player['location']
        is_teammate = player.get('teammate', False)

        if is_teammate:
            attacking_positions.append((x, y))
        else:
            defending_positions.append((x, y))

    # Centroids
    if attacking_positions:
        attacking_centroid_x = np.mean([p[0] for p in attacking_positions])
        attacking_centroid_y = np.mean([p[1] for p in attacking_positions])
        attacking_to_goal_dist = np.sqrt((120 - attacking_centroid_x)**2 + (40 - attacking_centroid_y)**2)
    else:
        attacking_centroid_x = 0.0
        attacking_centroid_y = 0.0
        attacking_to_goal_dist = 0.0

    if defending_positions:
        defending_centroid_x = np.mean([p[0] for p in defending_positions])
        defending_centroid_y = np.mean([p[1] for p in defending_positions])
        defending_to_goal_dist = np.sqrt((120 - defending_centroid_x)**2 + (40 - defending_centroid_y)**2)

        # Defending compactness: average distance from centroid
        distances = [np.sqrt((p[0] - defending_centroid_x)**2 + (p[1] - defending_centroid_y)**2)
                     for p in defending_positions]
        defending_compactness = np.mean(distances)

        # Defending depth: x-axis spread
        x_coords = [p[0] for p in defending_positions]
        defending_depth = max(x_coords) - min(x_coords)
    else:
        defending_centroid_x = 0.0
        defending_centroid_y = 0.0
        defending_to_goal_dist = 0.0
        defending_compactness = 0.0
        defending_depth = 0.0

    return {
        'attacking_centroid_x': attacking_centroid_x,
        'attacking_centroid_y': attacking_centroid_y,
        'defending_centroid_x': defending_centroid_x,
        'defending_centroid_y': defending_centroid_y,
        'defending_compactness': defending_compactness,
        'defending_depth': defending_depth,
        'attacking_to_goal_dist': attacking_to_goal_dist,
        'defending_to_goal_dist': defending_to_goal_dist
    }


# ============ STEP 4: Pass Technique (2 features) ============

def extract_pass_technique_encoding(event):
    """
    Encode pass technique as binary features.

    Returns 2 features:
    - is_inswinging
    - is_outswinging
    """
    pass_data = event['pass']
    technique = pass_data.get('technique', {}).get('name', '')

    return {
        'is_inswinging': int(technique == 'Inswinging'),
        'is_outswinging': int(technique == 'Outswinging')
    }


# ============ STEP 5: Pass Outcome Context (4 features) ============

def extract_pass_outcome_context(event):
    """
    Extract pass outcome and context.

    Returns 4 features:
    - pass_outcome_encoded
    - is_cross_field_switch
    - has_recipient
    - is_shot_assist
    """
    pass_data = event['pass']

    # Pass outcome encoding (0: Complete, 1: Incomplete, 2: Out, 3: Offside)
    outcome = pass_data.get('outcome', {}).get('name', 'Complete')
    outcome_mapping = {'Complete': 0, 'Incomplete': 1, 'Out': 2, 'Pass Offside': 3}
    pass_outcome_encoded = outcome_mapping.get(outcome, 0)

    # Cross-field switch (based on pass_switch flag)
    is_cross_field_switch = int(pass_data.get('switch', False))

    # Has recipient
    has_recipient = int(pass_data.get('recipient') is not None)

    # Shot assist
    is_shot_assist = int(pass_data.get('shot_assist', False))

    return {
        'pass_outcome_encoded': pass_outcome_encoded,
        'is_cross_field_switch': is_cross_field_switch,
        'has_recipient': has_recipient,
        'is_shot_assist': is_shot_assist
    }


# ============ STEP 6: Goalkeeper Features (3 features) ============

def extract_goalkeeper_features(freeze_frame):
    """
    Extract goalkeeper-specific features.

    Returns 3 features:
    - num_attacking_keepers
    - num_defending_keepers
    - keeper_distance_to_goal
    """
    num_attacking_keepers = 0
    num_defending_keepers = 0
    keeper_distance_to_goal = 0.0

    for player in freeze_frame:
        position_id = player.get('position', {}).get('id')
        is_teammate = player.get('teammate', False)

        if position_id == 1:  # Goalkeeper position ID
            x, y = player['location']
            dist = np.sqrt((120 - x)**2 + (40 - y)**2)

            if is_teammate:
                num_attacking_keepers += 1
            else:
                num_defending_keepers += 1
                keeper_distance_to_goal = dist  # Track defending keeper

    return {
        'num_attacking_keepers': num_attacking_keepers,
        'num_defending_keepers': num_defending_keepers,
        'keeper_distance_to_goal': keeper_distance_to_goal
    }


# ============ STEP 7: Score State (4 features) ============

def extract_score_state(corner, match_events=None):
    """
    Extract score state at time of corner.

    Returns 4 features:
    - attacking_team_goals
    - defending_team_goals
    - score_difference
    - match_situation (0: losing, 1: drawing, 2: winning)
    """
    if match_events is None:
        # Load match events if not provided
        match_id = corner['match_id']
        events_path = Path(__file__).parent.parent / 'data' / 'statsbomb' / 'events' / 'events' / f'{match_id}.json'
        with open(events_path, 'r') as f:
            match_events = json.load(f)

    event = corner['event']
    corner_timestamp = event['timestamp']
    corner_period = event['period']
    attacking_team_id = event['team']['id']

    # Count goals before this corner
    attacking_goals = 0
    defending_goals = 0

    for e in match_events:
        # Only count goals before this corner (same period and earlier time, or earlier period)
        if e['period'] < corner_period or (e['period'] == corner_period and e['timestamp'] < corner_timestamp):
            if e.get('type', {}).get('name') == 'Shot':
                shot_outcome = e.get('shot', {}).get('outcome', {}).get('name')
                if shot_outcome == 'Goal':
                    if e['team']['id'] == attacking_team_id:
                        attacking_goals += 1
                    else:
                        defending_goals += 1

    score_difference = attacking_goals - defending_goals

    # Match situation
    if score_difference < 0:
        match_situation = 0  # Losing
    elif score_difference == 0:
        match_situation = 1  # Drawing
    else:
        match_situation = 2  # Winning

    return {
        'attacking_team_goals': attacking_goals,
        'defending_team_goals': defending_goals,
        'score_difference': score_difference,
        'match_situation': match_situation
    }


# ============ STEP 8: Substitution Patterns (3 features) ============

def extract_substitution_patterns(corner, match_events=None):
    """
    Extract substitution patterns before corner.

    Returns 3 features:
    - total_subs_before
    - recent_subs_5min
    - minutes_since_last_sub
    """
    if match_events is None:
        # Load match events if not provided
        match_id = corner['match_id']
        events_path = Path(__file__).parent.parent / 'data' / 'statsbomb' / 'events' / 'events' / f'{match_id}.json'
        with open(events_path, 'r') as f:
            match_events = json.load(f)

    event = corner['event']
    corner_timestamp = event['timestamp']
    corner_period = event['period']
    corner_minute = event['minute']

    total_subs_before = 0
    recent_subs_5min = 0
    minutes_since_last_sub = 999  # Large default

    last_sub_minute = -999

    for e in match_events:
        # Only consider events before this corner
        if e['period'] < corner_period or (e['period'] == corner_period and e['timestamp'] < corner_timestamp):
            if e.get('type', {}).get('name') == 'Substitution':
                total_subs_before += 1
                sub_minute = e['minute']
                last_sub_minute = max(last_sub_minute, sub_minute)

                # Check if within 5 minutes
                if corner_minute - sub_minute <= 5:
                    recent_subs_5min += 1

    if last_sub_minute > -999:
        minutes_since_last_sub = corner_minute - last_sub_minute

    return {
        'total_subs_before': total_subs_before,
        'recent_subs_5min': recent_subs_5min,
        'minutes_since_last_sub': minutes_since_last_sub
    }


# ============ STEP 9: Metadata (2 features) ============

def extract_metadata_features(event):
    """
    Extract remaining metadata features.

    Returns 2 features:
    - corner_side (0: left, 1: right)
    - timestamp_seconds
    """
    location = event.get('location', [120, 40])
    corner_y = location[1]

    # Corner side (left: y < 40, right: y >= 40)
    corner_side = int(corner_y >= 40)

    # Timestamp in seconds
    timestamp_str = event['timestamp']
    time_parts = timestamp_str.split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])
    timestamp_seconds = hours * 3600 + minutes * 60 + seconds

    return {
        'corner_side': corner_side,
        'timestamp_seconds': timestamp_seconds
    }


# ============ PROGRESSIVE EXTRACTORS ============

def extract_features_step0(corner, match_events=None):
    """Baseline: Raw features only (27)"""
    return extract_raw_features(corner)


def extract_features_step1(corner, match_events=None):
    """Step 1: Baseline + Player Counts (33)"""
    features = extract_features_step0(corner, match_events)
    freeze_frame = corner.get('freeze_frame', [])
    features.update(extract_player_counts(freeze_frame))
    return features


def extract_features_step2(corner, match_events=None):
    """Step 2: Step 1 + Spatial Density (37)"""
    features = extract_features_step1(corner, match_events)
    freeze_frame = corner.get('freeze_frame', [])
    features.update(extract_spatial_density(freeze_frame))
    return features


def extract_features_step3(corner, match_events=None):
    """Step 3: Step 2 + Positional (45)"""
    features = extract_features_step2(corner, match_events)
    freeze_frame = corner.get('freeze_frame', [])
    features.update(extract_positional_features(freeze_frame))
    return features


def extract_features_step4(corner, match_events=None):
    """Step 4: Step 3 + Pass Technique (47)"""
    features = extract_features_step3(corner, match_events)
    event = corner['event']
    features.update(extract_pass_technique_encoding(event))
    return features


def extract_features_step5(corner, match_events=None):
    """Step 5: Step 4 + Pass Outcome Context (51)"""
    features = extract_features_step4(corner, match_events)
    event = corner['event']
    features.update(extract_pass_outcome_context(event))
    return features


def extract_features_step6(corner, match_events=None):
    """Step 6: Step 5 + Goalkeeper (54)"""
    features = extract_features_step5(corner, match_events)
    freeze_frame = corner.get('freeze_frame', [])
    features.update(extract_goalkeeper_features(freeze_frame))
    return features


def extract_features_step7(corner, match_events=None):
    """Step 7: Step 6 + Score State (58)"""
    features = extract_features_step6(corner, match_events)
    features.update(extract_score_state(corner, match_events))
    return features


def extract_features_step8(corner, match_events=None):
    """Step 8: Step 7 + Substitutions (61)"""
    features = extract_features_step7(corner, match_events)
    features.update(extract_substitution_patterns(corner, match_events))
    return features


def extract_features_step9(corner, match_events=None):
    """Step 9: Step 8 + Metadata (63 - FULL)"""
    features = extract_features_step8(corner, match_events)
    event = corner['event']
    features.update(extract_metadata_features(event))
    return features


# ============ MAIN EXTRACTION ============

def load_corners_with_freeze_frames(input_path):
    """Load corners with freeze frame data."""
    print(f"Loading corners from: {input_path}")
    with open(input_path, 'r') as f:
        corners = json.load(f)
    print(f"Loaded {len(corners)} corners")
    return corners


def extract_all_features_by_step(corners, step, match_events_cache=None):
    """
    Extract features for all corners at a specific step.

    Args:
        corners: List of corner dictionaries
        step: Feature step (0-9)
        match_events_cache: Optional dict mapping match_id -> events

    Returns:
        DataFrame with features and metadata
    """
    extractor_map = {
        0: extract_features_step0,
        1: extract_features_step1,
        2: extract_features_step2,
        3: extract_features_step3,
        4: extract_features_step4,
        5: extract_features_step5,
        6: extract_features_step6,
        7: extract_features_step7,
        8: extract_features_step8,
        9: extract_features_step9
    }

    extractor = extractor_map[step]
    feature_list = []

    for corner in tqdm(corners, desc=f"Extracting features (Step {step})"):
        try:
            match_id = corner['match_id']

            # Load match events if needed (for steps 7-8)
            match_events = None
            if step >= 7 and match_events_cache is not None:
                match_events = match_events_cache.get(match_id)

            features = extractor(corner, match_events)

            # Add metadata
            features['match_id'] = match_id
            features['event_id'] = corner['event']['id']
            features['event_timestamp'] = corner['event']['timestamp']

            feature_list.append(features)

        except Exception as e:
            print(f"Error processing corner {corner.get('event', {}).get('id', 'unknown')}: {e}")
            continue

    df = pd.DataFrame(feature_list)

    # Reorder columns: metadata first, then features
    metadata_cols = ['match_id', 'event_id', 'event_timestamp']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + feature_cols]

    return df


def main():
    """Extract features for all 10 steps (0-9)."""
    # Paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / 'data' / 'processed' / 'corners_with_freeze_frames.json'
    output_dir = project_root / 'data' / 'processed' / 'ablation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    corners = load_corners_with_freeze_frames(input_file)

    # Load match events cache (for steps 7-8)
    print("\n=== Loading Match Events Cache ===")
    match_events_cache = {}
    events_dir = project_root / 'data' / 'statsbomb' / 'events' / 'events'

    unique_matches = set(corner['match_id'] for corner in corners)
    for match_id in tqdm(unique_matches, desc="Loading match events"):
        events_path = events_dir / f'{match_id}.json'
        if events_path.exists():
            with open(events_path, 'r') as f:
                match_events_cache[match_id] = json.load(f)

    # Extract features for each step
    for step in range(10):
        print(f"\n=== Step {step} ===")
        df = extract_all_features_by_step(corners, step, match_events_cache)

        # Save to CSV
        output_file = output_dir / f'corners_features_step{step}.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ Saved to: {output_file}")
        print(f"  Shape: {df.shape}")
        print(f"  Features: {df.shape[1] - 3} (excluding metadata)")


if __name__ == '__main__':
    main()
