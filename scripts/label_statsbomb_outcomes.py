#!/usr/bin/env python3
"""
Add outcome labels to StatsBomb corners dataset (Version 2 - Fixed)

Uses the unified OutcomeLabeler module for consistent classification across all data sources.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from statsbombpy import sb
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.outcome_labeler import StatsBombOutcomeLabeler, calculate_success_metrics


def main():
    """Add outcome labels to existing StatsBomb corners"""

    base_dir = Path("/home/mseo/CornerTactics")
    input_file = base_dir / "data" / "datasets" / "statsbomb" / "corners_360.csv"
    output_file = base_dir / "data" / "datasets" / "statsbomb" / "corners_360_with_outcomes.csv"

    print("="*80)
    print("StatsBomb Corner Outcome Labeling (Version 2 - Fixed)")
    print("="*80)

    # Load existing corners
    print(f"\nLoading corners from: {input_file}")
    corners_df = pd.read_csv(input_file)
    print(f"Loaded {len(corners_df)} corners")

    # Get unique matches
    unique_matches = corners_df['match_id'].unique()
    print(f"Found {len(unique_matches)} unique matches")

    # Initialize labeler
    labeler = StatsBombOutcomeLabeler(max_time_window=20.0)

    # Process each match
    labeled_corners = []
    failed_matches = []

    for match_id in tqdm(unique_matches, desc="Processing matches"):
        # Get corners for this match
        match_corners = corners_df[corners_df['match_id'] == match_id].copy()

        # Fetch full events for match
        try:
            events_df = sb.events(match_id=int(match_id))
        except Exception as e:
            print(f"\nWarning: Could not fetch events for match {match_id}: {e}")
            failed_matches.append(match_id)
            continue

        # Process each corner in this match
        for idx, corner in match_corners.iterrows():
            # Find corner in events dataframe by matching on pass_type='Corner'
            corner_mask = (
                (events_df['type'] == 'Pass') &
                (events_df['pass_type'].notna()) &
                (events_df['pass_type'].str.contains('Corner', na=False, case=False)) &
                (events_df['team'] == corner['team']) &
                (events_df['minute'] == corner['minute']) &
                (events_df['second'] == corner['second'])
            )

            matching_events = events_df[corner_mask]

            if len(matching_events) == 0:
                # Try without pass_type filter (fallback)
                corner_mask = (
                    (events_df['type'] == 'Pass') &
                    (events_df['team'] == corner['team']) &
                    (events_df['minute'] == corner['minute']) &
                    (events_df['second'] == corner['second'])
                )
                matching_events = events_df[corner_mask]

                if len(matching_events) == 0:
                    continue

            # Get first matching event index
            corner_idx = matching_events.index[0]

            # Get outcome using unified labeler
            outcome = labeler.label_corner_outcome(events_df, corner_idx)

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
    print(f"Failed matches: {len(failed_matches)}")
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
    metrics = calculate_success_metrics(output_df)

    print(f"  Total corners: {metrics['total_corners']}")
    print(f"  Goals: {metrics['goals']} ({100*metrics['goal_rate']:.2f}%)")
    print(f"  Shots (inc. goals): {metrics['shots']} ({100*metrics['shot_rate']:.1f}%)")
    if metrics['shots'] > 0:
        goal_conversion = metrics['goals'] / metrics['shots']
        print(f"  Goal conversion from shots: {100*goal_conversion:.1f}%")
    print(f"  Clearances: {metrics['clearances']} ({100*metrics['clearance_rate']:.1f}%)")
    print(f"  Possession maintained: {metrics['possession_maintained']} ({100*metrics['possession_rate']:.1f}%)")

    # Time to outcome stats
    if 'avg_time_to_outcome' in metrics:
        print("\n" + "-"*80)
        print("Time to Outcome (seconds):")
        print("-"*80)
        print(f"  Mean: {metrics['avg_time_to_outcome']:.1f}s")
        print(f"  Median: {metrics['median_time_to_outcome']:.1f}s")

    # xThreat analysis
    xthreat_data = output_df[output_df['xthreat_delta'].notna()]['xthreat_delta']
    if len(xthreat_data) > 0:
        print("\n" + "-"*80)
        print("xThreat Delta Analysis:")
        print("-"*80)
        print(f"  Mean: {xthreat_data.mean():.2f}")
        print(f"  Median: {xthreat_data.median():.2f}")
        print(f"  Positive (more dangerous): {len(xthreat_data[xthreat_data > 0])}")
        print(f"  Negative (less dangerous): {len(xthreat_data[xthreat_data < 0])}")

    print("\n" + "="*80)
    print("✓ StatsBomb outcome labeling complete!")
    print("="*80)

if __name__ == "__main__":
    main()
