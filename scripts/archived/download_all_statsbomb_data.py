#!/usr/bin/env python3
"""
Download ALL Corner Kicks from ALL StatsBomb Open Data Competitions

This expanded version downloads corners from all 75 available competitions,
not just a subset. Expected to yield 5,000-10,000 corners with 360 data.

Key improvements:
- Downloads from ALL competitions (not just top 5)
- Includes international tournaments (World Cup, Euros, Copa America)
- Includes all available leagues (MLS, NWSL, Indian Super League, etc.)
- Adds progress tracking and resume capability

Usage:
    python scripts/download_all_statsbomb_data.py [--limit N]

Author: mseo
Date: October 2024
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import warnings
from statsbombpy import sb
import argparse

warnings.filterwarnings('ignore')


def get_all_available_competitions() -> pd.DataFrame:
    """
    Get ALL available competitions from StatsBomb.

    Returns:
        DataFrame with all competitions
    """
    print("Fetching all available competitions...")
    competitions = sb.competitions()

    # Get unique combinations (competition_id, season_id)
    competitions = competitions.drop_duplicates(
        subset=['competition_id', 'season_id']
    ).copy()

    print(f"Found {len(competitions)} unique competition-season pairs")
    print(f"Competitions span: {competitions['competition_name'].nunique()} different competitions")

    # Show distribution
    print("\nCompetitions by type:")
    comp_counts = competitions['competition_name'].value_counts()
    for comp, count in comp_counts.head(10).items():
        print(f"  {comp}: {count} seasons")

    return competitions


def download_corners_from_competition(
    competition_id: int,
    season_id: int,
    competition_name: str,
    season_name: str
) -> List[Dict]:
    """
    Download all corners with 360 data from a single competition-season.

    Args:
        competition_id: StatsBomb competition ID
        season_id: StatsBomb season ID
        competition_name: Competition name for logging
        season_name: Season name for logging

    Returns:
        List of corner kick dictionaries
    """
    corners_list = []

    try:
        # Get all matches for this competition-season
        matches = sb.matches(competition_id=competition_id, season_id=season_id)

        if matches.empty:
            return corners_list

        matches_with_360 = 0

        # Process each match
        for _, match in matches.iterrows():
            match_id = match['match_id']

            # Get events with 360 data
            try:
                events = sb.events(match_id=match_id, include_360_metrics=True)
            except:
                continue

            if events.empty:
                continue

            # Check if match has 360 data
            if '360_freeze_frame' not in events.columns:
                continue

            matches_with_360 += 1

            # Filter corner kicks
            corners = events[
                (events['type_name'] == 'Pass') &
                (events['pass_type_name'] == 'Corner') &
                (events['360_freeze_frame'].notna())
            ].copy()

            # Process each corner
            for _, corner in corners.iterrows():
                freeze_frame = corner['360_freeze_frame']
                if not freeze_frame:
                    continue

                # Count players by team
                attacking_players = []
                defending_players = []

                for player in freeze_frame:
                    pos = [player['location'][0], player['location'][1]]
                    if player['teammate']:
                        attacking_players.append(pos)
                    else:
                        defending_players.append(pos)

                # Create corner record
                corner_data = {
                    'match_id': match_id,
                    'competition': competition_name,
                    'season': season_name,
                    'match_date': match.get('match_date', ''),
                    'home_team': match.get('home_team_name', ''),
                    'away_team': match.get('away_team_name', ''),
                    'corner_id': corner.get('id', ''),
                    'minute': corner.get('minute', 0),
                    'second': corner.get('second', 0),
                    'team': corner.get('team_name', ''),
                    'player': corner.get('player_name', ''),
                    'location_x': corner.get('location', [0, 0])[0] if corner.get('location') else 0,
                    'location_y': corner.get('location', [0, 0])[1] if corner.get('location') else 0,
                    'end_x': corner.get('pass_end_location', [0, 0])[0] if corner.get('pass_end_location') else 0,
                    'end_y': corner.get('pass_end_location', [0, 0])[1] if corner.get('pass_end_location') else 0,
                    'num_attacking_players': len(attacking_players),
                    'num_defending_players': len(defending_players),
                    'total_visible_players': len(attacking_players) + len(defending_players),
                    'attacking_positions': json.dumps(attacking_players),
                    'defending_positions': json.dumps(defending_players)
                }

                corners_list.append(corner_data)

        if corners_list:
            print(f"  ✓ {competition_name} {season_name}: {len(corners_list)} corners from {matches_with_360} matches")

    except Exception as e:
        print(f"  ✗ Error processing {competition_name} {season_name}: {str(e)}")

    return corners_list


def main():
    """
    Main function to download all corners from all competitions.
    """
    parser = argparse.ArgumentParser(description='Download ALL StatsBomb corner kicks')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of competitions to process (for testing)')
    parser.add_argument('--output', type=str, default='data/raw/statsbomb/all_corners_360.csv',
                       help='Output CSV file path')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing file')
    args = parser.parse_args()

    # Setup output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STATSBOMB COMPREHENSIVE CORNER KICK DOWNLOAD")
    print("=" * 70)

    # Get all competitions
    competitions = get_all_available_competitions()

    # Optional limit for testing
    if args.limit:
        competitions = competitions.head(args.limit)
        print(f"\n⚠ Limited to first {args.limit} competitions for testing")

    # Check for existing data (resume capability)
    existing_corners = []
    if args.resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_corners = existing_df.to_dict('records')
        print(f"\n✓ Resuming from {len(existing_corners)} existing corners")

    # Process each competition
    all_corners = existing_corners.copy()
    processed_comps = set()

    if existing_corners:
        # Get already processed competitions
        existing_df = pd.DataFrame(existing_corners)
        for _, row in existing_df[['competition', 'season']].drop_duplicates().iterrows():
            processed_comps.add((row['competition'], row['season']))

    print(f"\n{'='*70}")
    print("DOWNLOADING CORNERS FROM ALL COMPETITIONS")
    print(f"{'='*70}")

    # Progress bar for competitions
    pbar = tqdm(total=len(competitions), desc="Processing competitions")

    for _, comp in competitions.iterrows():
        comp_name = comp['competition_name']
        season_name = comp['season_name']

        # Skip if already processed
        if (comp_name, season_name) in processed_comps:
            pbar.update(1)
            continue

        # Download corners from this competition
        corners = download_corners_from_competition(
            competition_id=comp['competition_id'],
            season_id=comp['season_id'],
            competition_name=comp_name,
            season_name=season_name
        )

        all_corners.extend(corners)

        # Save incrementally (in case of interruption)
        if corners and len(all_corners) % 100 == 0:
            temp_df = pd.DataFrame(all_corners)
            temp_df.to_csv(output_path, index=False)

        pbar.update(1)

        # Small delay to be nice to the API
        time.sleep(0.1)

    pbar.close()

    # Save final dataset
    final_df = pd.DataFrame(all_corners)
    final_df.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"Total corners downloaded: {len(final_df)}")
    print(f"Unique competitions: {final_df['competition'].nunique()}")
    print(f"Unique seasons: {len(final_df[['competition', 'season']].drop_duplicates())}")
    print(f"Average players per corner: {final_df['total_visible_players'].mean():.1f}")

    print(f"\nCorners by competition:")
    comp_summary = final_df['competition'].value_counts().head(15)
    for comp, count in comp_summary.items():
        print(f"  {comp}: {count} corners")

    print(f"\n✓ Data saved to: {output_path}")
    print(f"✓ File size: {len(final_df)} rows × {len(final_df.columns)} columns")

    # Show expected improvement
    print(f"\n{'='*70}")
    print("EXPECTED DATA IMPROVEMENT")
    print(f"{'='*70}")
    print(f"Previous dataset: 1,118 corners from ~5 competitions")
    print(f"New dataset: {len(final_df)} corners from {final_df['competition'].nunique()} competitions")
    print(f"Improvement: {len(final_df) / 1118:.1f}× more data!")

    if len(final_df) > 5000:
        print("\n🎉 SUCCESS! We now have enough data to match US Soccer Fed's scale!")
        print(f"With augmentation: {len(final_df)} corners × 5 temporal × 2 mirror = {len(final_df) * 10:,} samples")


if __name__ == "__main__":
    main()