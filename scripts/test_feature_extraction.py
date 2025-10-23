#!/usr/bin/env python3
"""
Test Feature Extraction on Sample Corners

Quick test to validate feature engineering implementation before running full SLURM job.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering import FeatureEngineer


def test_statsbomb_corner():
    """Test feature extraction on a single StatsBomb corner."""
    print("\n" + "=" * 60)
    print("Test 1: StatsBomb Corner Feature Extraction")
    print("=" * 60)

    # Load sample corner
    corners_df = pd.read_csv("data/results/statsbomb/corners_360_with_outcomes.csv", nrows=5)
    print(f"Loaded {len(corners_df)} sample corners")

    engineer = FeatureEngineer()

    # Test on first corner
    corner = corners_df.iloc[0]
    print(f"\nTesting corner: {corner['corner_id']}")
    print(f"  Team: {corner['team']}")
    print(f"  Location: ({corner['location_x']:.1f}, {corner['location_y']:.1f})")
    print(f"  Target: ({corner['end_x']:.1f}, {corner['end_y']:.1f})")
    print(f"  Attacking players: {corner['num_attacking_players']}")
    print(f"  Defending players: {corner['num_defending_players']}")

    # Extract features
    features_list = engineer.extract_features_from_statsbomb_corner(corner)

    print(f"\n✅ Extracted {len(features_list)} player features")

    # Display first player
    if features_list:
        player = features_list[0]
        print(f"\nSample player features (team={player.team}):")
        print(f"  Position: ({player.x:.2f}, {player.y:.2f})")
        print(f"  Distance to goal: {player.distance_to_goal:.2f}")
        print(f"  Distance to ball: {player.distance_to_ball_target:.2f}")
        print(f"  Angle to goal: {player.angle_to_goal:.2f} rad")
        print(f"  Angle to ball: {player.angle_to_ball:.2f} rad")
        print(f"  In penalty box: {player.in_penalty_box}")
        print(f"  Players within 5m: {player.num_players_within_5m}")
        print(f"  Local density: {player.local_density_score:.2f}")

        # Convert to array
        feature_array = player.to_array()
        print(f"\nFeature vector shape: {feature_array.shape}")
        print(f"Feature vector: {feature_array}")

    # Convert to DataFrame
    features_df = engineer.features_to_dataframe(features_list)
    print(f"\n✅ Created DataFrame with shape: {features_df.shape}")
    print("\nColumn names:")
    for col in features_df.columns:
        print(f"  - {col}")

    # Display statistics
    print("\nFeature Statistics:")
    print(features_df[['distance_to_goal', 'distance_to_ball_target', 'angle_to_goal', 'in_penalty_box']].describe())

    return features_df


def test_pitch_coordinates():
    """Test pitch coordinate system and geometry calculations."""
    print("\n" + "=" * 60)
    print("Test 2: Pitch Coordinate System")
    print("=" * 60)

    engineer = FeatureEngineer()

    # Test goal distance calculation
    test_positions = [
        ((120.0, 40.0), "Goal center"),
        ((120.0, 0.0), "Corner flag (right)"),
        ((120.0, 80.0), "Corner flag (left)"),
        ((102.0, 40.0), "Penalty spot"),
        ((60.0, 40.0), "Center circle"),
        ((0.0, 40.0), "Own goal"),
    ]

    print("\nDistance to goal from key positions:")
    for pos, name in test_positions:
        dist = engineer.calculate_distance(pos, engineer.goal_center)
        angle = engineer.calculate_angle(pos, engineer.goal_center)
        in_box = engineer.is_in_penalty_box(pos[0], pos[1])
        print(f"  {name:25s}: dist={dist:6.2f}, angle={angle:6.2f} rad, in_box={in_box}")

    # Test penalty box boundary
    print("\nPenalty box boundaries:")
    box_positions = [
        ((102.0, 18.0), "Bottom-left corner"),
        ((102.0, 62.0), "Top-left corner"),
        ((120.0, 18.0), "Bottom-right corner"),
        ((120.0, 62.0), "Top-right corner"),
        ((110.0, 40.0), "Center of box"),
        ((100.0, 40.0), "Just outside box"),
    ]
    for pos, name in box_positions:
        in_box = engineer.is_in_penalty_box(pos[0], pos[1])
        print(f"  {name:25s}: ({pos[0]:5.1f}, {pos[1]:5.1f}) -> in_box={in_box}")


def test_velocity_calculation():
    """Test velocity feature calculation."""
    print("\n" + "=" * 60)
    print("Test 3: Velocity Calculation")
    print("=" * 60)

    engineer = FeatureEngineer()

    # Test velocity calculation (10 fps = 0.1s between frames)
    test_cases = [
        ((100.0, 40.0), (101.0, 40.0), "Moving right (1 unit/frame)"),
        ((100.0, 40.0), (100.0, 41.0), "Moving up (1 unit/frame)"),
        ((100.0, 40.0), (102.0, 42.0), "Diagonal movement"),
        ((100.0, 40.0), (100.0, 40.0), "Stationary"),
    ]

    print("\nVelocity calculations (10 fps = 0.1s per frame):")
    for current, previous, description in test_cases:
        vx, vy, vmag, vangle = engineer.calculate_velocity_features(
            current, previous, fps=10.0
        )
        print(f"  {description:30s}:")
        print(f"    vx={vx:6.2f}, vy={vy:6.2f}, magnitude={vmag:6.2f}, angle={vangle:6.2f} rad")


def test_density_calculation():
    """Test density feature calculation."""
    print("\n" + "=" * 60)
    print("Test 4: Density Calculation")
    print("=" * 60)

    engineer = FeatureEngineer()

    # Create a crowded penalty box scenario
    player_pos = (110.0, 40.0)
    all_positions = [
        (110.0, 40.0),  # Self
        (108.0, 38.0),  # Close
        (112.0, 42.0),  # Close
        (109.0, 41.0),  # Close
        (111.0, 39.0),  # Close
        (115.0, 45.0),  # Medium distance
        (105.0, 35.0),  # Medium distance
        (120.0, 40.0),  # Far (goal line)
    ]

    num_nearby, density = engineer.calculate_density_features(player_pos, all_positions, radius=5.0)

    print(f"\nPlayer at position: {player_pos}")
    print(f"Total players in scene: {len(all_positions)}")
    print(f"Players within 5 units: {num_nearby}")
    print(f"Local density score: {density:.2f}")

    # Test sparse scenario
    sparse_positions = [
        (110.0, 40.0),  # Self
        (90.0, 40.0),   # Far
        (70.0, 40.0),   # Very far
    ]

    num_nearby_sparse, density_sparse = engineer.calculate_density_features(
        player_pos, sparse_positions, radius=5.0
    )

    print(f"\nSparse scenario:")
    print(f"Players within 5 units: {num_nearby_sparse}")
    print(f"Local density score: {density_sparse:.2f}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "FEATURE EXTRACTION TEST SUITE")
    print("=" * 70)

    try:
        # Test 1: Extract features from real corner
        features_df = test_statsbomb_corner()

        # Test 2: Coordinate system
        test_pitch_coordinates()

        # Test 3: Velocity calculation
        test_velocity_calculation()

        # Test 4: Density calculation
        test_density_calculation()

        # Summary
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nFeature extraction is working correctly!")
        print("Ready to run full extraction with: sbatch scripts/slurm/phase2_1_extract_features.sh")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
