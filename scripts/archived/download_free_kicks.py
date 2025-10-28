#!/usr/bin/env python3
"""
Download Wide Free Kicks from StatsBomb Open Data

Free kicks from wide positions are tactically similar to corners:
- Set piece situation
- Players positioned in the box
- Aerial duel potential
- Similar defensive setup

This script extracts free kicks that are:
- In the attacking third (x > 80)
- From wide angles (> 30 degrees from goal center)
- Have 360 freeze frame data

Usage:
    python scripts/download_free_kicks.py

Author: mseo
Date: October 2024
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import warnings
from statsbombpy import sb

warnings.filterwarnings('ignore')


def calculate_angle_to_goal(x: float, y: float) -> float:
    """
    Calculate angle from position to goal center.

    Args:
        x: X coordinate (0-120)
        y: Y coordinate (0-80)

    Returns:
        Angle in degrees from goal center
    """
    # Goal is at (120, 40)
    goal_x, goal_y = 120, 40

    # Calculate angle
    dx = goal_x - x
    dy = goal_y - y

    angle = math.degrees(math.atan2(abs(dy), dx))
    return angle


def is_wide_free_kick(event: pd.Series) -> bool:
    """
    Check if a free kick is from a wide position.

    Args:
        event: Event row from StatsBomb

    Returns:
        True if free kick is from wide position
    """
    if event.get('type_name') != 'Pass':
        return False

    if event.get('pass_type_name') != 'Free Kick':
        return False

    # Must have location
    location = event.get('location')
    if not location or len(location) != 2:
        return False

    x, y = location

    # Must be in attacking third
    if x < 80:
        return False

    # Calculate angle to goal
    angle = calculate_angle_to_goal(x, y)

    # Wide position (> 30 degrees from center)
    if angle < 30:
        return False

    # Must have 360 data
    if pd.isna(event.get('360_freeze_frame')):
        return False

    return True


def download_free_kicks_from_competition(
    competition_id: int,
    season_id: int,
    competition_name: str,
    season_name: str
) -> List[Dict]:
    """
    Download all wide free kicks with 360 data from a competition.

    Args:
        competition_id: StatsBomb competition ID
        season_id: StatsBomb season ID
        competition_name: Competition name
        season_name: Season name

    Returns:
        List of free kick dictionaries
    """
    free_kicks_list = []

    try:
        # Get all matches
        matches = sb.matches(competition_id=competition_id, season_id=season_id)

        if matches.empty:
            return free_kicks_list

        # Process each match
        for _, match in matches.iterrows():
            match_id = match['match_id']

            # Get events with 360 data
            try:
                events = sb.events(match_id=match_id, include_360_metrics=True)
            except:
                continue

            if events.empty or '360_freeze_frame' not in events.columns:
                continue

            # Process each event
            for _, event in events.iterrows():
                if not is_wide_free_kick(event):
                    continue

                freeze_frame = event['360_freeze_frame']
                if not freeze_frame:
                    continue

                # Count players
                attacking_players = []
                defending_players = []

                for player in freeze_frame:
                    pos = [player['location'][0], player['location'][1]]
                    if player['teammate']:
                        attacking_players.append(pos)
                    else:
                        defending_players.append(pos)

                # Calculate angle for metadata
                location = event.get('location', [0, 0])
                angle = calculate_angle_to_goal(location[0], location[1])

                # Create free kick record
                free_kick_data = {
                    'match_id': match_id,
                    'competition': competition_name,
                    'season': season_name,
                    'match_date': match.get('match_date', ''),
                    'home_team': match.get('home_team_name', ''),
                    'away_team': match.get('away_team_name', ''),
                    'free_kick_id': event.get('id', ''),
                    'minute': event.get('minute', 0),
                    'second': event.get('second', 0),
                    'team': event.get('team_name', ''),
                    'player': event.get('player_name', ''),
                    'location_x': location[0] if location else 0,
                    'location_y': location[1] if location else 0,
                    'end_x': event.get('pass_end_location', [0, 0])[0] if event.get('pass_end_location') else 0,
                    'end_y': event.get('pass_end_location', [0, 0])[1] if event.get('pass_end_location') else 0,
                    'angle_to_goal': angle,
                    'num_attacking_players': len(attacking_players),
                    'num_defending_players': len(defending_players),
                    'total_visible_players': len(attacking_players) + len(defending_players),
                    'attacking_positions': json.dumps(attacking_players),
                    'defending_positions': json.dumps(defending_players)
                }

                free_kicks_list.append(free_kick_data)

    except Exception as e:
        print(f"  Error processing {competition_name} {season_name}: {str(e)}")

    return free_kicks_list


def main():
    """
    Main function to download wide free kicks.
    """
    # Setup output
    output_dir = Path("data/raw/statsbomb")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WIDE FREE KICKS DOWNLOAD")
    print("=" * 70)
    print("\nDownloading tactically similar free kicks to augment corner dataset")
    print("Criteria: Attacking third, wide angle (>30°), with 360 data")

    # Get all competitions
    competitions = sb.competitions()

    # Remove duplicates
    competitions = competitions.drop_duplicates(
        subset=['competition_id', 'season_id']
    ).copy()

    print(f"\nProcessing {len(competitions)} competition-seasons...")

    all_free_kicks = []

    # Process each competition
    pbar = tqdm(total=len(competitions), desc="Processing competitions")

    for _, comp in competitions.iterrows():
        free_kicks = download_free_kicks_from_competition(
            competition_id=comp['competition_id'],
            season_id=comp['season_id'],
            competition_name=comp['competition_name'],
            season_name=comp['season_name']
        )

        if free_kicks:
            all_free_kicks.extend(free_kicks)
            print(f"  ✓ {comp['competition_name']} {comp['season_name']}: {len(free_kicks)} free kicks")

        pbar.update(1)

    pbar.close()

    # Save dataset
    df = pd.DataFrame(all_free_kicks)
    output_csv = output_dir / "wide_free_kicks_360.csv"
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"Total wide free kicks: {len(df)}")
    print(f"Competitions: {df['competition'].nunique()}")
    print(f"Average angle to goal: {df['angle_to_goal'].mean():.1f}°")
    print(f"Average players: {df['total_visible_players'].mean():.1f}")

    print(f"\nFree kicks by competition:")
    for comp, count in df['competition'].value_counts().head(10).items():
        print(f"  {comp}: {count}")

    print(f"\n✓ Data saved to: {output_csv}")
    print(f"\n💡 These {len(df)} free kicks can augment the corner dataset")
    print(f"   Combined with corners: More diverse set piece scenarios")


if __name__ == "__main__":
    main()