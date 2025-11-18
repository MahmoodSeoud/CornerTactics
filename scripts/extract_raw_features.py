"""
Extract RAW StatsBomb features (no engineering) for ablation study baseline.

This script extracts the 25 fixed raw StatsBomb fields plus simple freeze frame counts
(total_attacking, total_defending) to serve as the baseline for the ablation study.

Total Features: ~27 (25 fixed + 2 simple counts)

Raw Feature Categories:
1. Numeric/Continuous (12): period, minute, second, duration, index, possession,
   location_x, location_y, pass_length, pass_angle, pass_end_x, pass_end_y
2. Categorical IDs (10): team_id, player_id, position_id, play_pattern_id,
   possession_team_id, pass_height_id, pass_body_part_id, pass_type_id,
   pass_technique_id, pass_recipient_id
3. Boolean (3): under_pressure, is_aerial_won, is_out
4. Freeze Frame Counts (2): total_attacking, total_defending

Note: This does NOT include:
- Engineered spatial features (density, centroids, etc.)
- Derived features (numerical_advantage, ratios, etc.)
- Pass outcome encoding
- Score state
- Substitution patterns
- Any aggregations or transformations beyond simple counts
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def extract_raw_features(corner):
    """
    Extract raw StatsBomb fields without any engineering.

    Args:
        corner: Dictionary with keys 'match_id', 'event', 'freeze_frame'

    Returns:
        Dictionary of raw features
    """
    event = corner['event']
    pass_data = event['pass']
    freeze_frame = corner.get('freeze_frame', [])

    features = {}

    # ============ NUMERIC/CONTINUOUS (12) ============
    features['period'] = event['period']
    features['minute'] = event['minute']
    features['second'] = event['second']
    features['duration'] = event['duration']
    features['index'] = event['index']
    features['possession'] = event['possession']

    # Location coordinates
    location = event.get('location', [120, 40])  # Default to right corner
    features['location_x'] = location[0]
    features['location_y'] = location[1]

    # Pass metrics
    features['pass_length'] = pass_data.get('length', 0.0)
    features['pass_angle'] = pass_data.get('angle', 0.0)

    # Pass end location
    end_location = pass_data.get('end_location', [120, 40])
    features['pass_end_x'] = end_location[0]
    features['pass_end_y'] = end_location[1]

    # ============ CATEGORICAL IDs (10) ============
    features['team_id'] = event.get('team', {}).get('id', -1)
    features['player_id'] = event.get('player', {}).get('id', -1)
    features['position_id'] = event.get('position', {}).get('id', -1)
    features['play_pattern_id'] = event.get('play_pattern', {}).get('id', -1)
    features['possession_team_id'] = event.get('possession_team', {}).get('id', -1)

    # Pass-specific IDs
    features['pass_height_id'] = pass_data.get('height', {}).get('id', -1)
    features['pass_body_part_id'] = pass_data.get('body_part', {}).get('id', -1)
    features['pass_type_id'] = pass_data.get('type', {}).get('id', -1)
    features['pass_technique_id'] = pass_data.get('technique', {}).get('id', -1)
    features['pass_recipient_id'] = pass_data.get('recipient', {}).get('id', -1)

    # ============ BOOLEAN (3) ============
    features['under_pressure'] = int(event.get('under_pressure', False))

    # Pass outcome (exists if pass was incomplete)
    pass_outcome = pass_data.get('outcome', {})
    features['has_pass_outcome'] = int(pass_outcome is not None and len(pass_outcome) > 0)

    # Aerial won (for headers)
    features['is_aerial_won'] = int(pass_data.get('aerial_won', False))

    # ============ FREEZE FRAME COUNTS (2) ============
    # Simple counts of teammates and opponents (no positions)
    total_attacking = 0
    total_defending = 0

    for player in freeze_frame:
        if player.get('teammate', False):
            total_attacking += 1
        else:
            total_defending += 1

    features['total_attacking'] = total_attacking
    features['total_defending'] = total_defending

    return features


def load_corners_with_freeze_frames(input_path):
    """Load corners with freeze frame data."""
    print(f"Loading corners from: {input_path}")
    with open(input_path, 'r') as f:
        corners = json.load(f)
    print(f"Loaded {len(corners)} corners")
    return corners


def extract_all_raw_features(corners):
    """
    Extract raw features for all corners.

    Args:
        corners: List of corner dictionaries

    Returns:
        DataFrame with raw features and metadata
    """
    feature_list = []

    for corner in tqdm(corners, desc="Extracting raw features"):
        try:
            features = extract_raw_features(corner)

            # Add metadata
            features['match_id'] = corner['match_id']
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
    """Extract raw features from corners with freeze frames."""
    # Paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / 'data' / 'processed' / 'corners_with_freeze_frames.json'
    output_file = project_root / 'data' / 'processed' / 'corners_raw_features.csv'

    # Load data
    corners = load_corners_with_freeze_frames(input_file)

    # Extract features
    print("\n=== Extracting Raw Features (Baseline) ===")
    df = extract_all_raw_features(corners)

    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved raw features to: {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {df.shape[1] - 3} (excluding metadata)")

    # Print summary statistics
    print("\n=== Raw Feature Summary ===")
    print(f"Total corners: {len(df)}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Metadata columns: 3 (match_id, event_id, event_timestamp)")
    print(f"Feature columns: {df.shape[1] - 3}")

    print("\n=== Sample Features ===")
    print(df.head())

    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("No missing values!")


if __name__ == '__main__':
    main()
