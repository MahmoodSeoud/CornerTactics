#!/usr/bin/env python3
"""
Extract Temporal Features from SkillCorner Tracking Data

Creates multiple temporal frames per corner kick using 10fps tracking data.
Accesses data directly via GitHub media URLs (bypassing Git LFS).

Usage:
    python scripts/extract_skillcorner_temporal.py

Author: mseo
Date: October 2024
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering import FeatureEngineer, PlayerFeatures

warnings.filterwarnings('ignore')


def load_tracking_from_github(match_id: int) -> pd.DataFrame:
    """
    Load tracking data directly from GitHub media URL.

    Args:
        match_id: SkillCorner match ID

    Returns:
        DataFrame with tracking data
    """
    tracking_url = f'https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl'
    try:
        return pd.read_json(tracking_url, lines=True)
    except Exception as e:
        print(f"Error loading tracking for match {match_id}: {e}")
        return None


def load_match_metadata(match_id: int) -> Dict:
    """
    Load match metadata from GitHub.

    Args:
        match_id: SkillCorner match ID

    Returns:
        Match metadata dictionary
    """
    match_url = f'https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json'
    try:
        with urllib.request.urlopen(match_url) as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"Error loading metadata for match {match_id}: {e}")
        return None


def extract_frame_features(frame_data: pd.Series, match_meta: Dict,
                          corner_team: str, temporal_offset: float) -> List[PlayerFeatures]:
    """
    Extract player features from a tracking frame.

    Args:
        frame_data: Single frame from tracking data
        match_meta: Match metadata
        corner_team: Team taking the corner
        temporal_offset: Time offset from corner moment (seconds)

    Returns:
        List of PlayerFeatures for all players
    """
    features = []

    # Get team IDs
    home_id = match_meta['home_team']['id']
    away_id = match_meta['away_team']['id']
    home_name = match_meta['home_team']['name']
    away_name = match_meta['away_team']['name']

    # Determine which team is attacking based on corner team
    if corner_team in home_name or corner_team == home_name:
        attacking_team_id = home_id
    else:
        attacking_team_id = away_id

    # Ball position
    ball_x = frame_data['ball_data']['x']
    ball_y = frame_data['ball_data']['y']
    ball_z = frame_data['ball_data'].get('z', 0)

    # Extract features for each player
    for player in frame_data['player_data']:
        player_id = player['player_id']
        x = player['x']
        y = player['y']

        # Estimate team from player ID ranges (heuristic)
        # In SkillCorner data, player IDs often group by team
        is_attacking = (player_id < 50000 and attacking_team_id == home_id) or \
                      (player_id >= 50000 and attacking_team_id == away_id)

        # Calculate velocities (will be 0 for single frame, but structure for future)
        vx = 0.0  # Would need frame-to-frame calculation
        vy = 0.0
        velocity_magnitude = 0.0
        velocity_angle = 0.0

        # Spatial features
        distance_to_goal = np.sqrt((52.5 - x)**2 + y**2)  # Assuming 105x68m pitch
        distance_to_ball = np.sqrt((x - ball_x)**2 + (y - ball_y)**2)

        # Angular features
        angle_to_goal = np.arctan2(0 - y, 52.5 - x)
        angle_to_ball = np.arctan2(ball_y - y, ball_x - x)

        # Context features
        in_penalty_box = (x > 52.5 - 16.5) and abs(y) < 20.16

        # Create feature vector
        feature = PlayerFeatures(
            player_id=str(player_id),
            team='attacking' if is_attacking else 'defending',
            x=x,
            y=y,
            distance_to_goal=distance_to_goal,
            distance_to_ball_target=distance_to_ball,
            vx=vx,
            vy=vy,
            velocity_magnitude=velocity_magnitude,
            velocity_angle=velocity_angle,
            angle_to_goal=angle_to_goal,
            angle_to_ball=angle_to_ball,
            team_flag=1.0 if is_attacking else 0.0,
            in_penalty_box=1.0 if in_penalty_box else 0.0,
            num_players_within_5m=0,  # Will calculate after
            local_density_score=0.0
        )
        # Store temporal offset separately
        feature.temporal_offset = temporal_offset
        features.append(feature)

    # Calculate density features
    for i, f1 in enumerate(features):
        count = 0
        for j, f2 in enumerate(features):
            if i != j:
                dist = np.sqrt((f1.x - f2.x)**2 + (f1.y - f2.y)**2)
                if dist < 5.0:
                    count += 1
        features[i].num_players_within_5m = count
        features[i].local_density_score = count / len(features)

    return features


def process_skillcorner_corners():
    """
    Main function to process all SkillCorner corners with temporal expansion.
    """
    print("="*70)
    print("SkillCorner Temporal Feature Extraction")
    print("="*70)

    # Load corner events
    corners_file = Path("data/raw/skillcorner/skillcorner_corners_with_outcomes.csv")
    if not corners_file.exists():
        print(f"Error: {corners_file} not found!")
        return

    corners_df = pd.read_csv(corners_file)
    print(f"Found {len(corners_df)} corners from SkillCorner")

    # Group by match for efficient processing
    matches = corners_df['match_id'].unique()
    print(f"Processing {len(matches)} matches...")

    # Temporal offsets (in seconds)
    temporal_offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]

    all_features = []

    for match_id in tqdm(matches, desc="Processing matches"):
        # Load tracking data once per match
        tracking_data = load_tracking_from_github(match_id)
        if tracking_data is None:
            continue

        match_meta = load_match_metadata(match_id)
        if match_meta is None:
            continue

        # Process each corner in this match
        match_corners = corners_df[corners_df['match_id'] == match_id]

        for _, corner in match_corners.iterrows():
            corner_frame = corner['frame_start']
            corner_team = corner['team_shortname']
            corner_id = f"sc_{match_id}_{corner['event_id']}"

            # Extract features for temporal frames
            for offset_sec in temporal_offsets:
                offset_frames = int(offset_sec * 10)  # 10 fps
                target_frame = corner_frame + offset_frames

                # Check bounds
                if target_frame < 0 or target_frame >= len(tracking_data):
                    continue

                frame_data = tracking_data.iloc[target_frame]

                # Extract features
                features = extract_frame_features(
                    frame_data, match_meta, corner_team, offset_sec
                )

                # Store with metadata
                for feat in features:
                    feat_dict = feat.__dict__.copy()
                    feat_dict.update({
                        'corner_id': f"{corner_id}_t{offset_sec:+.1f}",
                        'match_id': match_id,
                        'frame': target_frame,
                        'temporal_offset': offset_sec,
                        'outcome': corner['outcome_category'],
                        'goal_scored': corner['goal_scored']
                    })
                    all_features.append(feat_dict)

    # Create DataFrame
    features_df = pd.DataFrame(all_features)

    print(f"\n{'='*70}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total temporal frames: {len(features_df['corner_id'].unique())}")
    print(f"Total player features: {len(features_df)}")
    print(f"Corners with goals: {features_df[features_df['goal_scored']]['corner_id'].nunique()}")

    # Save features
    output_dir = Path("data/features/temporal")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "skillcorner_temporal_features.parquet"
    features_df.to_parquet(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Also save as CSV for inspection
    csv_file = output_dir / "skillcorner_temporal_features.csv"
    features_df.to_csv(csv_file, index=False)
    print(f"Also saved as CSV: {csv_file}")

    # Summary statistics
    print(f"\nSummary:")
    print(f"  Unique corners: {len(corners_df)}")
    print(f"  Temporal frames per corner: {len(temporal_offsets)}")
    print(f"  Total graph samples: {len(features_df['corner_id'].unique())}")
    print(f"  Average players per frame: {features_df.groupby('corner_id').size().mean():.1f}")


if __name__ == "__main__":
    process_skillcorner_corners()