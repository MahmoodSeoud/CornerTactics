#!/usr/bin/env python3
"""
Extract Node Features from Corner Kick Datasets

Implements Phase 2.1 of the Corner GNN project plan.

This script:
1. Loads labeled corner datasets (StatsBomb and SkillCorner)
2. Extracts 14-dimensional feature vectors per player
3. Saves enriched datasets with node features

Usage:
    python scripts/extract_corner_features.py

Output:
    - data/datasets/statsbomb/corners_with_node_features.csv
    - data/datasets/skillcorner/corners_with_node_features.csv
    - data/node_features/statsbomb_player_features.parquet
    - data/node_features/skillcorner_player_features.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering import FeatureEngineer, PlayerFeatures


def extract_statsbomb_features(
    corners_path: str = "data/results/statsbomb/corners_360_with_outcomes.csv",
    output_dir: str = "data/node_features"
) -> pd.DataFrame:
    """
    Extract features from StatsBomb corners dataset.

    Args:
        corners_path: Path to labeled StatsBomb corners
        output_dir: Output directory for features

    Returns:
        DataFrame with player-level features
    """
    print("\n" + "=" * 60)
    print("Extracting StatsBomb Corner Features")
    print("=" * 60)

    corners_df = pd.read_csv(corners_path)
    print(f"Loaded {len(corners_df)} corners from {corners_path}")

    engineer = FeatureEngineer()
    all_player_features = []

    # Process each corner
    for idx, corner_row in tqdm(corners_df.iterrows(), total=len(corners_df), desc="Processing corners"):
        try:
            # Extract features for all players
            features_list = engineer.extract_features_from_statsbomb_corner(corner_row)

            # Convert to DataFrame
            if features_list:
                features_df = engineer.features_to_dataframe(features_list)

                # Add corner metadata
                features_df['corner_id'] = corner_row['corner_id']
                features_df['match_id'] = corner_row['match_id']
                features_df['competition'] = corner_row['competition']
                features_df['season'] = corner_row['season']
                features_df['outcome_category'] = corner_row['outcome_category']
                features_df['goal_scored'] = corner_row['goal_scored']

                all_player_features.append(features_df)

        except Exception as e:
            print(f"\nError processing corner {corner_row.get('corner_id', idx)}: {e}")
            continue

    # Concatenate all results
    result_df = pd.concat(all_player_features, ignore_index=True)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as parquet (efficient for numerical data)
    output_file = output_path / "statsbomb_player_features.parquet"
    result_df.to_parquet(output_file, index=False)
    print(f"\n✅ Saved {len(result_df)} player features to {output_file}")

    # Also save as CSV for inspection
    csv_file = output_path / "statsbomb_player_features.csv"
    result_df.to_csv(csv_file, index=False)
    print(f"✅ Saved CSV to {csv_file}")

    # Print statistics
    print_feature_statistics(result_df, "StatsBomb")

    return result_df


def extract_skillcorner_features(
    corners_path: str = "data/results/skillcorner/skillcorner_corners_with_outcomes.csv",
    tracking_base_dir: str = "data/datasets/skillcorner/data/matches",
    output_dir: str = "data/node_features"
) -> pd.DataFrame:
    """
    Extract features from SkillCorner corners dataset with tracking data.

    Args:
        corners_path: Path to labeled SkillCorner corners
        tracking_base_dir: Base directory for tracking JSONL files
        output_dir: Output directory for features

    Returns:
        DataFrame with player-level features
    """
    print("\n" + "=" * 60)
    print("Extracting SkillCorner Corner Features")
    print("=" * 60)

    corners_df = pd.read_csv(corners_path)
    print(f"Loaded {len(corners_df)} corners from {corners_path}")

    # Filter for corners with tracking data
    corners_with_tracking = corners_df[corners_df['has_tracking'] == True]
    print(f"Corners with tracking data: {len(corners_with_tracking)}")

    engineer = FeatureEngineer()
    all_player_features = []
    tracking_cache = {}  # Cache loaded tracking files

    # Process each corner
    for idx, corner_row in tqdm(corners_with_tracking.iterrows(), total=len(corners_with_tracking), desc="Processing corners"):
        try:
            # Load tracking data (with caching)
            match_id = corner_row['match_id']
            if match_id not in tracking_cache:
                tracking_file = Path(tracking_base_dir) / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl"

                if tracking_file.exists():
                    # Load tracking data
                    tracking_data = pd.read_json(tracking_file, lines=True)
                    tracking_cache[match_id] = tracking_data
                else:
                    print(f"\n⚠️  Tracking file not found: {tracking_file}")
                    tracking_cache[match_id] = None

            tracking_data = tracking_cache[match_id]

            if tracking_data is None:
                continue

            # Extract features for all players
            features_list = engineer.extract_features_from_skillcorner_corner(
                corner_row, tracking_data
            )

            # Convert to DataFrame
            if features_list:
                features_df = engineer.features_to_dataframe(features_list)

                # Add corner metadata
                features_df['corner_id'] = f"skillcorner_{idx}"
                features_df['match_id'] = corner_row['match_id']
                features_df['competition'] = corner_row.get('competition', 'A-League')
                features_df['outcome_category'] = corner_row['outcome_category']

                all_player_features.append(features_df)

        except Exception as e:
            print(f"\nError processing corner idx={idx}: {e}")
            continue

    if not all_player_features:
        print("⚠️  No features extracted from SkillCorner data")
        return pd.DataFrame()

    # Concatenate all results
    result_df = pd.concat(all_player_features, ignore_index=True)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    output_file = output_path / "skillcorner_player_features.parquet"
    result_df.to_parquet(output_file, index=False)
    print(f"\n✅ Saved {len(result_df)} player features to {output_file}")

    # Also save as CSV
    csv_file = output_path / "skillcorner_player_features.csv"
    result_df.to_csv(csv_file, index=False)
    print(f"✅ Saved CSV to {csv_file}")

    # Print statistics
    print_feature_statistics(result_df, "SkillCorner")

    return result_df


def print_feature_statistics(df: pd.DataFrame, dataset_name: str):
    """Print summary statistics for extracted features."""
    print(f"\n{dataset_name} Feature Statistics")
    print("-" * 60)
    print(f"Total players: {len(df)}")
    print(f"Unique corners: {df['corner_id'].nunique()}")
    print(f"Attacking players: {(df['team'] == 'attacking').sum()}")
    print(f"Defending players: {(df['team'] == 'defending').sum()}")

    print("\nSpatial Features:")
    print(f"  Distance to goal: {df['distance_to_goal'].mean():.2f} ± {df['distance_to_goal'].std():.2f}")
    print(f"  Distance to ball: {df['distance_to_ball_target'].mean():.2f} ± {df['distance_to_ball_target'].std():.2f}")
    print(f"  In penalty box: {df['in_penalty_box'].sum()} ({100*df['in_penalty_box'].mean():.1f}%)")

    if df['velocity_magnitude'].sum() > 0:
        print("\nKinematic Features:")
        print(f"  Velocity magnitude: {df['velocity_magnitude'].mean():.2f} ± {df['velocity_magnitude'].std():.2f}")
        print(f"  Players moving (v > 0.1): {(df['velocity_magnitude'] > 0.1).sum()}")

    print("\nDensity Features:")
    print(f"  Players within 5m: {df['num_players_within_5m'].mean():.2f} ± {df['num_players_within_5m'].std():.2f}")
    print(f"  Local density score: {df['local_density_score'].mean():.2f} ± {df['local_density_score'].std():.2f}")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("Phase 2.1: Node Feature Engineering")
    print("=" * 60)
    print("Extracting 14-dimensional feature vectors per player:")
    print("  - Spatial: x, y, distance_to_goal, distance_to_ball_target (4)")
    print("  - Kinematic: vx, vy, velocity_magnitude, velocity_angle (4)")
    print("  - Contextual: angle_to_goal, angle_to_ball, team_flag, in_penalty_box (4)")
    print("  - Density: num_players_within_5m, local_density_score (2)")

    # Extract StatsBomb features
    statsbomb_features = extract_statsbomb_features()

    # Extract SkillCorner features
    skillcorner_features = extract_skillcorner_features()

    # Summary
    print("\n" + "=" * 60)
    print("Feature Extraction Complete")
    print("=" * 60)
    print(f"StatsBomb: {len(statsbomb_features)} player features extracted")
    print(f"SkillCorner: {len(skillcorner_features)} player features extracted")
    print(f"Total: {len(statsbomb_features) + len(skillcorner_features)} player features")
    print("\nOutput files:")
    print("  - data/node_features/statsbomb_player_features.parquet")
    print("  - data/node_features/statsbomb_player_features.csv")
    print("  - data/node_features/skillcorner_player_features.parquet")
    print("  - data/node_features/skillcorner_player_features.csv")


if __name__ == "__main__":
    main()
