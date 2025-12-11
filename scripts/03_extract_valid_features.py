#!/usr/bin/env python3
"""
Extract only temporally valid features for corner kick prediction.

This script removes all features that contain temporal data leakage
(features only known AFTER the corner kick is taken).

Author: CornerTactics Team
Date: 2025-11-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Define feature categories based on temporal availability
VALID_RAW_FEATURES = [
    # Temporal context (known at corner time)
    'second',           # When in match
    'index',            # Event sequence (if available)
    'possession',       # Possession number (if available)

    # Spatial context (corner starting position)
    'corner_x',         # Corner position X (should be ~120 or ~0)
    'corner_y',         # Corner position Y (should be ~0 or ~80)

    # Team/Player context
    'player_id',        # Who's taking corner (if available)
    'position_id',      # Player's position (if available)
    'play_pattern_id',  # How corner was earned (if available)
    'possession_team_id', # Team taking corner (if available)
    'team_id',          # Team ID (if different from possession_team_id)

    # Match state
    'period',           # First or second half
    'minute',           # Match minute

    # 360 freeze frame features (player positions at kick time)
    'total_attacking',
    'total_defending',
    'attacking_in_box',
    'defending_in_box',
    'attacking_near_goal',
    'defending_near_goal',
]

VALID_ENGINEERED_FEATURES = [
    # Spatial distributions from freeze frames
    'attacking_density',
    'defending_density',
    'numerical_advantage',
    'attacker_defender_ratio',

    # Valid engineered features
    'corner_side',              # Left/right corner
    'defending_depth',          # Defensive line position from freeze frame
    'attacking_to_goal_dist',   # Avg attacker distance to goal from freeze frame
    'defending_to_goal_dist',   # Avg defender distance to goal from freeze frame
    'keeper_distance_to_goal',  # GK distance to goal from freeze frame

    # Match state (known before corner)
    'attacking_team_goals',     # Goals scored by attacking team
    'defending_team_goals',     # Goals conceded (defending team's score)
]

# Features that contain temporal leakage (MUST EXCLUDE)
LEAKED_FEATURES = [
    # Event outcome features
    'duration',             # Only known after event completes
    'pass_end_x',          # Where ball actually landed
    'pass_end_y',          # Where ball actually landed
    'pass_length',         # Actual pass distance
    'pass_angle',          # Actual pass angle
    'pass_recipient_id',   # Who actually received it
    'has_pass_outcome',    # Pass outcome flag
    'pass_outcome',        # Pass outcome type
    'pass_outcome_encoded', # Encoded pass outcome
    'is_aerial_won',       # Aerial duel result

    # Shot-related outcomes
    'is_shot_assist',      # Whether corner led to shot
    'has_recipient',       # Whether there was successful recipient
    'is_cross_field_switch', # Whether ball switched sides

    # Any other outcome-related features
    'shot',                # Binary shot outcome (if present)
    'goal',                # Binary goal outcome (if present)
]

# Features that need investigation (might be intent or actual)
AMBIGUOUS_FEATURES = [
    'pass_body_part',      # Could be planned or actual
    'pass_body_part_id',   # Could be planned or actual
    'pass_technique',       # Inswing/outswing - planned or actual?
    'pass_technique_id',    # Inswing/outswing - planned or actual?
    'pass_height',          # Could be planned or actual
    'pass_height_id',       # Could be planned or actual
    'under_pressure',       # When is this assessed?
]


def extract_valid_features(input_path, output_path, include_ambiguous=False):
    """
    Extract only temporally valid features from the dataset.

    Args:
        input_path: Path to input CSV with all features
        output_path: Path to save cleaned CSV
        include_ambiguous: Whether to include ambiguous features (default: False)
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} corners with {len(df.columns)} features")

    # Get all column names
    all_columns = df.columns.tolist()

    # Identify metadata columns to keep
    metadata_columns = ['match_id', 'event_id', 'outcome']

    # Build list of columns to keep
    columns_to_keep = metadata_columns.copy()

    # Add valid raw features that exist
    for feature in VALID_RAW_FEATURES:
        if feature in all_columns:
            columns_to_keep.append(feature)
        # Also check for variations (e.g., location_x vs corner_x)
        elif feature.replace('corner_', 'location_') in all_columns:
            alt_name = feature.replace('corner_', 'location_')
            columns_to_keep.append(alt_name)

    # Add valid engineered features that exist
    for feature in VALID_ENGINEERED_FEATURES:
        if feature in all_columns:
            columns_to_keep.append(feature)

    # Optionally add ambiguous features
    if include_ambiguous:
        print("Including ambiguous features (use with caution)...")
        for feature in AMBIGUOUS_FEATURES:
            if feature in all_columns:
                columns_to_keep.append(feature)

    # Remove duplicates while preserving order
    columns_to_keep = list(dict.fromkeys(columns_to_keep))

    # Check for leaked features that might be included
    leaked_in_dataset = [col for col in all_columns if col in LEAKED_FEATURES]
    valid_in_dataset = [col for col in columns_to_keep if col not in metadata_columns]

    print(f"\n📊 Feature Summary:")
    print(f"  - Total features in dataset: {len(all_columns)}")
    print(f"  - Valid features found: {len(valid_in_dataset)}")
    print(f"  - Leaked features found: {len(leaked_in_dataset)}")
    print(f"  - Features to keep: {len(columns_to_keep)}")

    if leaked_in_dataset:
        print(f"\n⚠️ Removing leaked features:")
        for feat in leaked_in_dataset[:10]:  # Show first 10
            print(f"    - {feat}")
        if len(leaked_in_dataset) > 10:
            print(f"    ... and {len(leaked_in_dataset) - 10} more")

    # Create cleaned dataframe
    df_clean = df[columns_to_keep].copy()

    # Add binary shot label from original data if available
    # Note: The label itself is from the outcome (which we know from historical data)
    # but we only use pre-corner features for prediction
    if 'leads_to_shot' in df.columns:
        df_clean['shot'] = df['leads_to_shot'].astype(int)
        print(f"\n📊 Shot Label Distribution (for training):")
        print(f"  - Corners leading to shot: {df_clean['shot'].sum()} ({df_clean['shot'].mean()*100:.1f}%)")
        print(f"  - Corners not leading to shot: {len(df_clean) - df_clean['shot'].sum()} ({(1-df_clean['shot'].mean())*100:.1f}%)")
    elif 'is_shot_assist' in df.columns:
        # Fallback to is_shot_assist if leads_to_shot not available
        # Note: This is the outcome label for training, not a feature!
        df_clean['shot'] = df['is_shot_assist'].astype(int)
        print(f"\n📊 Shot Assist Label Distribution (for training):")
        print(f"  - Direct shot assists: {df_clean['shot'].sum()} ({df_clean['shot'].mean()*100:.1f}%)")
        print(f"  - Not direct assists: {len(df_clean) - df_clean['shot'].sum()} ({(1-df_clean['shot'].mean())*100:.1f}%)")

    # Save cleaned dataset
    print(f"\n💾 Saving cleaned dataset to {output_path}...")
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df_clean)} corners with {len(df_clean.columns)} features")

    # Save feature list for reference
    feature_info = {
        'valid_features': valid_in_dataset,
        'leaked_features_removed': leaked_in_dataset,
        'ambiguous_included': AMBIGUOUS_FEATURES if include_ambiguous else [],
        'total_features_kept': len(columns_to_keep),
        'total_features_removed': len(all_columns) - len(columns_to_keep)
    }

    feature_info_path = output_path.parent / 'temporal_valid_features.json'
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"📝 Feature info saved to {feature_info_path}")

    return df_clean


def main():
    """Main execution function."""
    # Set paths
    data_dir = Path('/home/mseo/CornerTactics/data/processed')

    # Process main dataset
    input_file = data_dir / 'corners_features_with_shot.csv'
    output_file = data_dir / 'corners_features_temporal_valid.csv'

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Please ensure you have run the feature extraction pipeline first.")
        return

    # Extract valid features (excluding ambiguous by default)
    df_clean = extract_valid_features(
        input_file,
        output_file,
        include_ambiguous=False  # Set to True to include ambiguous features
    )

    print("\n" + "="*60)
    print("✅ TEMPORAL FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print("\n⚠️ IMPORTANT NOTES:")
    print("1. This dataset removes all temporal leakage")
    print("2. Expected accuracy will be lower (~75-80%) but VALID")
    print("3. Use this dataset for all future model training")
    print("4. Do NOT use features like pass_end_x/y or is_shot_assist")

    # Create a sample feature vector for verification
    if len(df_clean) > 0:
        print("\n📋 Sample feature vector (first corner):")
        sample = df_clean.iloc[0]
        for col in df_clean.columns[:20]:  # Show first 20 features
            if col not in ['match_id', 'event_id', 'outcome']:
                print(f"    {col}: {sample[col]}")


if __name__ == "__main__":
    main()