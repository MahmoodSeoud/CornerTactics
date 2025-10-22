#!/usr/bin/env python3
"""
Extract corner kick events from SkillCorner Open Data
Parses all matches for corner_for/corner_against events and links to tracking data
"""

import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm

def extract_corners_from_match(match_dir):
    """Extract corner events from a single match directory"""
    match_id = os.path.basename(match_dir)

    # Load dynamic events
    events_file = os.path.join(match_dir, f"{match_id}_dynamic_events.csv")
    if not os.path.exists(events_file):
        print(f"Warning: {events_file} not found")
        return pd.DataFrame()

    df = pd.read_csv(events_file)

    # Filter for corner events (game_interruption_before or game_interruption_after contains "corner")
    corner_mask = (
        df['game_interruption_before'].str.contains('corner', case=False, na=False) |
        df['game_interruption_after'].str.contains('corner', case=False, na=False)
    )

    corners = df[corner_mask].copy()

    if len(corners) == 0:
        return pd.DataFrame()

    # Load match metadata
    match_file = os.path.join(match_dir, f"{match_id}_match.json")
    if os.path.exists(match_file):
        with open(match_file, 'r') as f:
            match_data = json.load(f)
            corners['home_team'] = match_data.get('home_team', {}).get('name', 'Unknown')
            corners['away_team'] = match_data.get('away_team', {}).get('name', 'Unknown')
            corners['competition'] = match_data.get('competition', 'Unknown')
            corners['season'] = match_data.get('season', 'Unknown')
            corners['match_date'] = match_data.get('date_time', 'Unknown')

    # Check if tracking data exists
    tracking_file = os.path.join(match_dir, f"{match_id}_tracking_extrapolated.jsonl")
    corners['has_tracking'] = os.path.exists(tracking_file)
    corners['tracking_file'] = tracking_file if corners['has_tracking'].any() else None

    return corners

def main():
    """Extract corners from all SkillCorner matches"""

    # Setup paths
    base_dir = Path("/home/mseo/CornerTactics")
    skillcorner_dir = base_dir / "data" / "datasets" / "skillcorner" / "data" / "matches"
    output_dir = base_dir / "data" / "datasets" / "skillcorner"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all match directories
    match_dirs = sorted([d for d in skillcorner_dir.iterdir() if d.is_dir()])

    print(f"Found {len(match_dirs)} matches in {skillcorner_dir}")
    print("")

    # Extract corners from each match
    all_corners = []
    corner_counts = {}

    for match_dir in tqdm(match_dirs, desc="Processing matches"):
        corners = extract_corners_from_match(match_dir)
        if len(corners) > 0:
            all_corners.append(corners)
            match_id = os.path.basename(match_dir)
            corner_counts[match_id] = len(corners)

    # Combine all corners
    if len(all_corners) == 0:
        print("No corners found!")
        return

    corners_df = pd.concat(all_corners, ignore_index=True)

    # Select key columns for output
    output_columns = [
        'match_id', 'event_id', 'index',
        'home_team', 'away_team', 'competition', 'season', 'match_date',
        'period', 'minute_start', 'second_start', 'time_start',
        'frame_start', 'frame_end', 'duration',
        'attacking_side', 'team_shortname', 'player_name', 'player_position',
        'x_start', 'y_start', 'x_end', 'y_end',
        'game_interruption_before', 'game_interruption_after',
        'event_type', 'event_subtype',
        'has_tracking', 'tracking_file'
    ]

    # Keep only columns that exist
    output_columns = [col for col in output_columns if col in corners_df.columns]
    corners_output = corners_df[output_columns]

    # Save to CSV
    output_file = output_dir / "skillcorner_corners.csv"
    corners_output.to_csv(output_file, index=False)

    # Print summary
    print("\n" + "="*60)
    print("SkillCorner Corner Extraction Complete")
    print("="*60)
    print(f"Total matches processed: {len(match_dirs)}")
    print(f"Matches with corners: {len(corner_counts)}")
    print(f"Total corner events: {len(corners_df)}")
    print(f"\nOutput saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Print per-match breakdown
    print("\nCorners per match:")
    for match_id, count in sorted(corner_counts.items()):
        print(f"  Match {match_id}: {count} corners")

    # Sample of corner types
    print("\nCorner event types:")
    if 'game_interruption_before' in corners_df.columns:
        print("Before interruptions:")
        print(corners_df['game_interruption_before'].value_counts().head(5))
    if 'game_interruption_after' in corners_df.columns:
        print("\nAfter interruptions:")
        print(corners_df['game_interruption_after'].value_counts().head(5))

if __name__ == "__main__":
    main()
