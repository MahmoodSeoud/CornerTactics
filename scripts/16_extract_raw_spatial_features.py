#!/usr/bin/env python3
"""
Extract raw spatial features from freeze frame data.

This script creates multiple feature sets for baseline comparison:
1. Raw coordinates (padded, sorted by distance to goal)
2. Pairwise distance features (marking structure)
3. Combined features

Author: CornerTactics Team
Date: 2025-11-27
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# Constants
GOAL_POSITION = (120.0, 40.0)  # StatsBomb coordinate system
MAX_ATTACKERS = 11
MAX_DEFENDERS = 11
PENALTY_BOX = {'x_min': 102, 'x_max': 120, 'y_min': 18, 'y_max': 62}


def distance_to_goal(x: float, y: float) -> float:
    """Calculate Euclidean distance from (x, y) to goal center."""
    return np.sqrt((x - GOAL_POSITION[0])**2 + (y - GOAL_POSITION[1])**2)


def extract_players(freeze_frame: List[Dict]) -> Tuple[List, List, Dict]:
    """
    Extract and separate attackers, defenders, and goalkeeper from freeze frame.

    Returns:
        attackers: List of (x, y, dist_to_goal) tuples
        defenders: List of (x, y, dist_to_goal) tuples
        goalkeeper: Dict with keeper info or None
    """
    attackers = []
    defenders = []
    goalkeeper = None

    for player in freeze_frame:
        loc = player.get('location', [0, 0])
        x, y = loc[0], loc[1]
        dist = distance_to_goal(x, y)

        is_teammate = player.get('teammate', False)
        is_keeper = player.get('keeper', False)
        is_actor = player.get('actor', False)  # Corner taker

        if is_keeper and not is_teammate:
            goalkeeper = {'x': x, 'y': y, 'dist': dist}
        elif is_teammate and not is_actor:  # Exclude corner taker
            attackers.append((x, y, dist))
        elif not is_teammate and not is_keeper:
            defenders.append((x, y, dist))

    # Sort by distance to goal (ascending - closest first)
    attackers.sort(key=lambda p: p[2])
    defenders.sort(key=lambda p: p[2])

    return attackers, defenders, goalkeeper


def extract_raw_coordinate_features(attackers: List, defenders: List,
                                     goalkeeper: Dict) -> Dict:
    """
    Extract raw coordinate features, padded to fixed size.

    Features:
    - att_0_x, att_0_y, att_1_x, att_1_y, ... (sorted by dist to goal)
    - def_0_x, def_0_y, def_1_x, def_1_y, ...
    - gk_x, gk_y
    """
    features = {}

    # Attacker positions (padded with -1)
    for i in range(MAX_ATTACKERS):
        if i < len(attackers):
            features[f'att_{i}_x'] = attackers[i][0]
            features[f'att_{i}_y'] = attackers[i][1]
        else:
            features[f'att_{i}_x'] = -1.0
            features[f'att_{i}_y'] = -1.0

    # Defender positions (padded with -1)
    for i in range(MAX_DEFENDERS):
        if i < len(defenders):
            features[f'def_{i}_x'] = defenders[i][0]
            features[f'def_{i}_y'] = defenders[i][1]
        else:
            features[f'def_{i}_x'] = -1.0
            features[f'def_{i}_y'] = -1.0

    # Goalkeeper position
    if goalkeeper:
        features['gk_x'] = goalkeeper['x']
        features['gk_y'] = goalkeeper['y']
    else:
        features['gk_x'] = -1.0
        features['gk_y'] = -1.0

    return features


def extract_pairwise_distance_features(attackers: List, defenders: List,
                                        goalkeeper: Dict) -> Dict:
    """
    Extract pairwise distance features capturing marking structure.

    Features:
    - Distance from each attacker to nearest defender
    - Distance from each attacker to goalkeeper
    - Aggregate distance statistics
    """
    features = {}

    att_positions = [(a[0], a[1]) for a in attackers]
    def_positions = [(d[0], d[1]) for d in defenders]

    # Distance from each attacker to nearest defender
    nearest_defender_dists = []
    for i, (ax, ay) in enumerate(att_positions):
        if def_positions:
            dists = [np.sqrt((ax - dx)**2 + (ay - dy)**2)
                    for dx, dy in def_positions]
            min_dist = min(dists)
            nearest_defender_dists.append(min_dist)
            if i < 5:  # Top 5 attackers (closest to goal)
                features[f'att_{i}_nearest_def_dist'] = min_dist
        else:
            if i < 5:
                features[f'att_{i}_nearest_def_dist'] = -1.0

    # Aggregate nearest defender distances
    if nearest_defender_dists:
        features['min_att_def_dist'] = min(nearest_defender_dists)
        features['max_att_def_dist'] = max(nearest_defender_dists)
        features['mean_att_def_dist'] = np.mean(nearest_defender_dists)
        features['std_att_def_dist'] = np.std(nearest_defender_dists) if len(nearest_defender_dists) > 1 else 0
    else:
        features['min_att_def_dist'] = -1.0
        features['max_att_def_dist'] = -1.0
        features['mean_att_def_dist'] = -1.0
        features['std_att_def_dist'] = -1.0

    # Distance from each attacker to goalkeeper
    gk_dists = []
    if goalkeeper:
        gk_x, gk_y = goalkeeper['x'], goalkeeper['y']
        for i, (ax, ay) in enumerate(att_positions):
            dist = np.sqrt((ax - gk_x)**2 + (ay - gk_y)**2)
            gk_dists.append(dist)
            if i < 5:
                features[f'att_{i}_gk_dist'] = dist

    if not goalkeeper:
        for i in range(5):
            features[f'att_{i}_gk_dist'] = -1.0

    # Aggregate goalkeeper distances
    if gk_dists:
        features['min_att_gk_dist'] = min(gk_dists)
        features['mean_att_gk_dist'] = np.mean(gk_dists)
    else:
        features['min_att_gk_dist'] = -1.0
        features['mean_att_gk_dist'] = -1.0

    # Count "unmarked" attackers (no defender within 2 meters)
    unmarked_count = sum(1 for d in nearest_defender_dists if d > 2.0)
    features['unmarked_attackers'] = unmarked_count

    # Count attackers in box with no close defender
    in_box_unmarked = 0
    for (ax, ay), dist in zip(att_positions, nearest_defender_dists):
        if (PENALTY_BOX['x_min'] <= ax <= PENALTY_BOX['x_max'] and
            PENALTY_BOX['y_min'] <= ay <= PENALTY_BOX['y_max'] and
            dist > 2.0):
            in_box_unmarked += 1
    features['in_box_unmarked'] = in_box_unmarked

    return features


def extract_spatial_structure_features(attackers: List, defenders: List) -> Dict:
    """
    Extract spatial structure features (team shape).

    Features:
    - Spread (std of positions)
    - Range (max - min)
    - Convex hull area (simplified)
    """
    features = {}

    att_positions = [(a[0], a[1]) for a in attackers]
    def_positions = [(d[0], d[1]) for d in defenders]

    # Attacker spread
    if len(att_positions) >= 2:
        att_x = [p[0] for p in att_positions]
        att_y = [p[1] for p in att_positions]
        features['att_x_spread'] = np.std(att_x)
        features['att_y_spread'] = np.std(att_y)
        features['att_x_range'] = max(att_x) - min(att_x)
        features['att_y_range'] = max(att_y) - min(att_y)
    else:
        features['att_x_spread'] = 0.0
        features['att_y_spread'] = 0.0
        features['att_x_range'] = 0.0
        features['att_y_range'] = 0.0

    # Defender spread
    if len(def_positions) >= 2:
        def_x = [p[0] for p in def_positions]
        def_y = [p[1] for p in def_positions]
        features['def_x_spread'] = np.std(def_x)
        features['def_y_spread'] = np.std(def_y)
        features['def_x_range'] = max(def_x) - min(def_x)
        features['def_y_range'] = max(def_y) - min(def_y)
    else:
        features['def_x_spread'] = 0.0
        features['def_y_spread'] = 0.0
        features['def_x_range'] = 0.0
        features['def_y_range'] = 0.0

    # Relative positioning
    if att_positions and def_positions:
        att_centroid = (np.mean([p[0] for p in att_positions]),
                       np.mean([p[1] for p in att_positions]))
        def_centroid = (np.mean([p[0] for p in def_positions]),
                       np.mean([p[1] for p in def_positions]))
        features['centroid_x_diff'] = att_centroid[0] - def_centroid[0]
        features['centroid_y_diff'] = att_centroid[1] - def_centroid[1]
    else:
        features['centroid_x_diff'] = 0.0
        features['centroid_y_diff'] = 0.0

    return features


def process_corner(corner: Dict) -> Dict:
    """Process a single corner and extract all feature sets."""
    freeze_frame = corner.get('freeze_frame', [])
    event = corner.get('event', {})

    # Extract players
    attackers, defenders, goalkeeper = extract_players(freeze_frame)

    # Extract all feature types
    features = {}

    # Metadata
    features['match_id'] = corner.get('match_id')
    features['event_id'] = event.get('id')

    # Raw coordinates
    raw_features = extract_raw_coordinate_features(attackers, defenders, goalkeeper)
    features.update(raw_features)

    # Pairwise distances
    pairwise_features = extract_pairwise_distance_features(attackers, defenders, goalkeeper)
    features.update(pairwise_features)

    # Spatial structure
    structure_features = extract_spatial_structure_features(attackers, defenders)
    features.update(structure_features)

    # Player counts
    features['n_attackers'] = len(attackers)
    features['n_defenders'] = len(defenders)
    features['has_goalkeeper'] = 1 if goalkeeper else 0

    return features


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'

    # Load freeze frame data
    print("Loading freeze frame data...")
    with open(data_dir / 'corners_with_freeze_frames.json', 'r') as f:
        corners = json.load(f)
    print(f"Loaded {len(corners)} corners")

    # Load labels
    print("Loading labels...")
    labels_df = pd.read_csv(data_dir / 'corners_features_temporal_valid.csv')
    labels_df = labels_df[['match_id', 'event_id', 'outcome', 'shot']]
    print(f"Loaded {len(labels_df)} labels")

    # Extract features
    print("Extracting raw spatial features...")
    feature_rows = []
    for i, corner in enumerate(corners):
        features = process_corner(corner)
        feature_rows.append(features)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(corners)} corners")

    # Create DataFrame
    df = pd.DataFrame(feature_rows)
    print(f"\nExtracted {len(df.columns)} features")

    # Ensure consistent types for merge
    df['match_id'] = df['match_id'].astype(str)
    df['event_id'] = df['event_id'].astype(str)
    labels_df['match_id'] = labels_df['match_id'].astype(str)
    labels_df['event_id'] = labels_df['event_id'].astype(str)

    # Merge with labels
    df = df.merge(labels_df, on=['match_id', 'event_id'], how='inner')
    print(f"After merge with labels: {len(df)} corners")

    # Feature summary
    raw_coord_cols = [c for c in df.columns if c.startswith('att_') and c.endswith('_x') or
                      c.startswith('att_') and c.endswith('_y') or
                      c.startswith('def_') and c.endswith('_x') or
                      c.startswith('def_') and c.endswith('_y') or
                      c.startswith('gk_')]
    pairwise_cols = [c for c in df.columns if 'dist' in c or 'unmarked' in c]
    structure_cols = [c for c in df.columns if 'spread' in c or 'range' in c or 'centroid' in c]

    print(f"\nFeature breakdown:")
    print(f"  Raw coordinates: {len(raw_coord_cols)} features")
    print(f"  Pairwise distances: {len(pairwise_cols)} features")
    print(f"  Spatial structure: {len(structure_cols)} features")

    # Save
    output_path = data_dir / 'corners_raw_spatial_features.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Save feature list
    feature_info = {
        'raw_coordinate_features': raw_coord_cols,
        'pairwise_distance_features': pairwise_cols,
        'spatial_structure_features': structure_cols,
        'total_features': len(df.columns) - 4  # Exclude metadata + target
    }
    with open(data_dir / 'raw_spatial_features.json', 'w') as f:
        json.dump(feature_info, f, indent=2)

    print("\nDone!")


if __name__ == '__main__':
    main()
