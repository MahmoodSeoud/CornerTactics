#!/usr/bin/env python3
"""
Add outcome labels to SoccerNet corners dataset

Analyzes Labels-v2.json and Labels-v3.json files to determine corner outcomes.
"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.outcome_labeler import SoccerNetOutcomeLabeler, calculate_success_metrics


def parse_soccernet_time(game_time_str):
    """
    Parse SoccerNet time format to seconds

    Format: "1 - 12:34" (half - minute:second)

    Returns:
        Seconds from start of match
    """
    try:
        parts = game_time_str.split(' - ')
        if len(parts) == 2:
            half = int(parts[0])
            time_parts = parts[1].split(':')
            minutes = int(time_parts[0])
            seconds = int(time_parts[1])
            # Assume 45 min halves
            return (half - 1) * 2700 + minutes * 60 + seconds
    except:
        return 0.0
    return 0.0


def main():
    """Add outcome labels to existing SoccerNet corners"""

    base_dir = Path("/home/mseo/CornerTactics")
    input_file = base_dir / "data" / "datasets" / "soccernet" / "soccernet_corners.csv"
    output_file = base_dir / "data" / "datasets" / "soccernet" / "soccernet_corners_with_outcomes.csv"
    soccernet_data_dir = base_dir / "data" / "datasets" / "soccernet"

    print("="*80)
    print("SoccerNet Corner Outcome Labeling")
    print("="*80)

    # Load existing corners
    print(f"\nLoading corners from: {input_file}")
    if not input_file.exists():
        print(f"ERROR: {input_file} not found!")
        print("Run extract_soccernet_corners.py first")
        return

    corners_df = pd.read_csv(input_file)
    print(f"Loaded {len(corners_df)} corners")

    # Initialize labeler
    labeler = SoccerNetOutcomeLabeler(max_time_window=20.0)

    # Process each corner
    labeled_corners = []
    failed_corners = 0

    for idx, corner in tqdm(corners_df.iterrows(), total=len(corners_df), desc="Processing corners"):
        # Parse corner details
        league = corner['league']
        season = corner['season']
        game = corner['game']
        half = corner['half']
        corner_time_str = corner['time']
        corner_team = corner.get('team', 'Unknown')

        # Construct path to labels file
        labels_file = soccernet_data_dir / league / season / game / f"Labels-v2.json"

        # Fallback to v3 if v2 doesn't exist
        if not labels_file.exists():
            labels_file = soccernet_data_dir / league / season / game / f"Labels-v3.json"

        if not labels_file.exists():
            failed_corners += 1
            # Still add corner but without outcome
            corner_dict = corner.to_dict()
            corner_dict.update({
                'outcome_category': 'Unknown',
                'outcome_type': 'No Labels File',
                'outcome_team': None,
                'outcome_player': None,
                'same_team': None,
                'time_to_outcome': None,
                'events_to_outcome': None,
                'goal_scored': False,
                'shot_outcome': None,
                'outcome_location': None,
                'xthreat_delta': 0.0
            })
            labeled_corners.append(corner_dict)
            continue

        # Load labels file
        try:
            with open(labels_file, 'r') as f:
                labels_data = json.load(f)
        except Exception as e:
            print(f"\nWarning: Could not load labels file for {game}: {e}")
            failed_corners += 1
            continue

        # Parse corner timestamp
        corner_timestamp = parse_soccernet_time(corner_time_str)

        # Get outcome using unified labeler
        outcome = labeler.label_corner_outcome(
            labels_data,
            corner_timestamp,
            corner_team
        )

        # Merge with corner data
        corner_dict = corner.to_dict()
        corner_dict.update(outcome.to_dict())
        labeled_corners.append(corner_dict)

    # Create output dataframe
    output_df = pd.DataFrame(labeled_corners)

    # Save results
    output_df.to_csv(output_file, index=False)

    # Print summary
    print("\n" + "="*80)
    print("Outcome Labeling Complete")
    print("="*80)
    print(f"Labeled corners: {len(output_df)} / {len(corners_df)}")
    print(f"Failed corners: {failed_corners}")
    print(f"Output saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Outcome distribution
    print("\n" + "-"*80)
    print("Outcome Category Distribution:")
    print("-"*80)
    category_counts = output_df['outcome_category'].value_counts()
    for category, count in category_counts.items():
        pct = 100 * count / len(output_df)
        print(f"  {category:15s}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "-"*80)
    print("Detailed Outcome Types (Top 10):")
    print("-"*80)
    type_counts = output_df['outcome_type'].value_counts().head(10)
    for outcome_type, count in type_counts.items():
        pct = 100 * count / len(output_df)
        print(f"  {outcome_type:30s}: {count:4d} ({pct:5.1f}%)")

    # Success metrics
    print("\n" + "-"*80)
    print("Success Metrics:")
    print("-"*80)
    metrics = calculate_success_metrics(output_df[output_df['outcome_category'] != 'Unknown'])

    if metrics:
        print(f"  Total corners (with outcomes): {metrics['total_corners']}")
        print(f"  Goals: {metrics.get('goals', 0)} ({100*metrics.get('goal_rate', 0):.2f}%)")
        print(f"  Shots (inc. goals): {metrics.get('shots', 0)} ({100*metrics.get('shot_rate', 0):.1f}%)")
        print(f"  Clearances: {metrics.get('clearances', 0)} ({100*metrics.get('clearance_rate', 0):.1f}%)")
        print(f"  Possession maintained: {metrics.get('possession_maintained', 0)} ({100*metrics.get('possession_rate', 0):.1f}%)")

        # Time to outcome stats
        if 'avg_time_to_outcome' in metrics:
            print("\n" + "-"*80)
            print("Time to Outcome (seconds):")
            print("-"*80)
            print(f"  Mean: {metrics['avg_time_to_outcome']:.1f}s")
            print(f"  Median: {metrics['median_time_to_outcome']:.1f}s")

    print("\n" + "="*80)
    print("✓ SoccerNet outcome labeling complete!")
    print("="*80)
    print("\nNOTE: SoccerNet labels may not capture all event types.")
    print("      Some outcomes may be classified as 'Possession' by default.")
    print("      Consider cross-validation with video clips for research use.")

if __name__ == "__main__":
    main()
