#!/usr/bin/env python3
"""
Add outcome labels to SkillCorner corners dataset

Analyzes dynamic events following corner kicks to determine outcomes.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.outcome_labeler import SkillCornerOutcomeLabeler, calculate_success_metrics


def main():
    """Add outcome labels to existing SkillCorner corners"""

    base_dir = Path("/home/mseo/CornerTactics")
    input_file = base_dir / "data" / "datasets" / "skillcorner" / "skillcorner_corners.csv"
    output_file = base_dir / "data" / "datasets" / "skillcorner" / "skillcorner_corners_with_outcomes.csv"
    skillcorner_data_dir = base_dir / "data" / "datasets" / "skillcorner" / "data" / "matches"

    print("="*80)
    print("SkillCorner Corner Outcome Labeling")
    print("="*80)

    # Load existing corners
    print(f"\nLoading corners from: {input_file}")
    if not input_file.exists():
        print(f"ERROR: {input_file} not found!")
        print("Run extract_skillcorner_corners.py first")
        return

    corners_df = pd.read_csv(input_file)
    print(f"Loaded {len(corners_df)} corners")

    # Get unique matches
    unique_matches = corners_df['match_id'].unique()
    print(f"Found {len(unique_matches)} unique matches")

    # Initialize labeler
    labeler = SkillCornerOutcomeLabeler(max_time_window=20.0)

    # Process each match
    labeled_corners = []
    failed_matches = []

    for match_id in tqdm(unique_matches, desc="Processing matches"):
        # Get corners for this match
        match_corners = corners_df[corners_df['match_id'] == match_id].copy()

        # Load dynamic events for this match
        dynamic_events_file = skillcorner_data_dir / str(match_id) / f"{match_id}_dynamic_events.csv"
        phases_file = skillcorner_data_dir / str(match_id) / f"{match_id}_phases_of_play.csv"

        if not dynamic_events_file.exists():
            print(f"\nWarning: Dynamic events file not found for match {match_id}")
            failed_matches.append(match_id)
            continue

        # Load dynamic events
        try:
            dynamic_events_df = pd.read_csv(dynamic_events_file)
        except Exception as e:
            print(f"\nWarning: Could not load dynamic events for match {match_id}: {e}")
            failed_matches.append(match_id)
            continue

        # Load phases (optional)
        phases_df = None
        if phases_file.exists():
            try:
                phases_df = pd.read_csv(phases_file)
            except:
                pass

        # Process each corner in this match
        for idx, corner in match_corners.iterrows():
            # Get outcome using unified labeler
            outcome = labeler.label_corner_outcome(
                dynamic_events_df,
                corner,
                phases_df
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
    print("✓ SkillCorner outcome labeling complete!")
    print("="*80)
    print("\nNOTE: SkillCorner dynamic_events.csv has limited event types.")
    print("      Some outcomes may be classified as 'Possession' by default.")
    print("      Consider manual validation for research use.")

if __name__ == "__main__":
    main()
