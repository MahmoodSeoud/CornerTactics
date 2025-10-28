#!/usr/bin/env python3
"""
Test which StatsBomb competitions have 360 data available.

Quick script to identify which competitions actually have freeze frame data.
"""

from statsbombpy import sb
import pandas as pd
from tqdm import tqdm


def test_competition_for_360(competition_id, season_id, comp_name):
    """Test if a competition has 360 data."""
    try:
        # Get one match
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        if matches.empty:
            return False, 0

        # Try first match
        match_id = matches.iloc[0]['match_id']

        # Get events with 360
        events = sb.events(match_id=match_id, include_360_metrics=True)

        # Check for 360 column
        if '360_freeze_frame' not in events.columns:
            return False, 0

        # Count corners with 360
        corners = events[
            (events['type_name'] == 'Pass') &
            (events['pass_type_name'] == 'Corner') &
            (events['360_freeze_frame'].notna())
        ]

        return len(corners) > 0, len(corners)

    except Exception as e:
        return False, 0


def main():
    print("Testing StatsBomb competitions for 360 data availability...")
    print("=" * 70)

    # Get all competitions
    competitions = sb.competitions()
    competitions = competitions.drop_duplicates(subset=['competition_id', 'season_id'])

    results = []

    # Test each competition
    for _, comp in tqdm(competitions.iterrows(), total=len(competitions), desc="Testing competitions"):
        has_360, corner_count = test_competition_for_360(
            comp['competition_id'],
            comp['season_id'],
            comp['competition_name']
        )

        if has_360:
            results.append({
                'competition': comp['competition_name'],
                'season': comp['season_name'],
                'has_360': has_360,
                'sample_corners': corner_count
            })
            print(f"✓ {comp['competition_name']} {comp['season_name']}: {corner_count} corners with 360")

    # Summary
    print("\n" + "=" * 70)
    print("COMPETITIONS WITH 360 DATA")
    print("=" * 70)

    if results:
        df = pd.DataFrame(results)
        print(f"\nFound {len(df)} competition-seasons with 360 data")
        print("\nTop competitions:")
        for _, row in df.head(10).iterrows():
            print(f"  {row['competition']} {row['season']}: {row['sample_corners']} corners in first match")
    else:
        print("No competitions found with 360 data!")

    return df


if __name__ == "__main__":
    df = main()
    if not df.empty:
        df.to_csv('data/raw/statsbomb/competitions_with_360.csv', index=False)
        print(f"\nSaved to: data/raw/statsbomb/competitions_with_360.csv")