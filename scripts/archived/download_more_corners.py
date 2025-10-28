#!/usr/bin/env python3
"""
Simple script to download MORE corners from StatsBomb.
Tests each competition for 360 data and downloads if available.
"""

from statsbombpy import sb
import pandas as pd
import json
from tqdm import tqdm
import time

def main():
    print("=" * 70)
    print("DOWNLOADING MORE STATSBOMB CORNERS")
    print("=" * 70)

    # Get all competitions
    print("\nFetching competitions...")
    competitions = sb.competitions()
    print(f"Found {len(competitions)} competition-season pairs")

    # Track which we already have
    existing = pd.read_csv('data/raw/statsbomb/corners_360.csv')
    existing_comps = set(zip(existing['competition'], existing['season']))
    print(f"Already have corners from {len(existing_comps)} competition-seasons")

    all_new_corners = []
    successful_comps = []

    # Try each competition
    for _, comp in tqdm(competitions.iterrows(), total=len(competitions), desc="Testing competitions"):
        comp_name = comp['competition_name']
        season_name = comp['season_name']
        comp_id = comp['competition_id']
        season_id = comp['season_id']

        # Skip if we already have it
        if (comp_name, season_name) in existing_comps:
            continue

        try:
            # Get matches
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            if matches.empty:
                continue

            # Try first match to see if 360 data exists
            test_match = matches.iloc[0]
            events = sb.events(match_id=test_match['match_id'], include_360_metrics=True)

            # Check for 360 column
            if '360_freeze_frame' not in events.columns:
                continue

            # This competition has 360 data! Process all matches
            comp_corners = []

            for _, match in matches.iterrows():
                try:
                    events = sb.events(match_id=match['match_id'], include_360_metrics=True)

                    # Get corners
                    corners = events[
                        (events['type_name'] == 'Pass') &
                        (events['pass_type_name'] == 'Corner') &
                        (events['360_freeze_frame'].notna())
                    ]

                    # Process each corner
                    for _, corner in corners.iterrows():
                        freeze_frame = corner['360_freeze_frame']
                        if not freeze_frame:
                            continue

                        # Extract player positions
                        attacking_positions = []
                        defending_positions = []

                        for player in freeze_frame:
                            pos = [player['location'][0], player['location'][1]]
                            if player['teammate']:
                                attacking_positions.append(pos)
                            else:
                                defending_positions.append(pos)

                        # Create corner record
                        corner_data = {
                            'match_id': match['match_id'],
                            'competition': comp_name,
                            'season': season_name,
                            'match_date': match.get('match_date', ''),
                            'home_team': match.get('home_team_name', ''),
                            'away_team': match.get('away_team_name', ''),
                            'corner_id': corner.get('id', ''),
                            'minute': corner.get('minute', 0),
                            'second': corner.get('second', 0),
                            'team': corner.get('team_name', ''),
                            'player': corner.get('player_name', ''),
                            'location_x': corner.get('location', [0, 0])[0],
                            'location_y': corner.get('location', [0, 0])[1],
                            'end_x': corner.get('pass_end_location', [0, 0])[0],
                            'end_y': corner.get('pass_end_location', [0, 0])[1],
                            'num_attacking_players': len(attacking_positions),
                            'num_defending_players': len(defending_positions),
                            'total_visible_players': len(attacking_positions) + len(defending_positions),
                            'attacking_positions': json.dumps(attacking_positions),
                            'defending_positions': json.dumps(defending_positions)
                        }

                        comp_corners.append(corner_data)

                except Exception as e:
                    continue

            if comp_corners:
                all_new_corners.extend(comp_corners)
                successful_comps.append((comp_name, season_name, len(comp_corners)))
                print(f"\n✓ {comp_name} {season_name}: {len(comp_corners)} corners")

            time.sleep(0.1)  # Be nice to the API

        except Exception as e:
            continue

    # Save results
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"\nNew corners found: {len(all_new_corners)}")
    print(f"From {len(successful_comps)} new competition-seasons")

    if all_new_corners:
        # Combine with existing data
        new_df = pd.DataFrame(all_new_corners)
        combined = pd.concat([existing, new_df], ignore_index=True)

        # Save
        output_file = 'data/raw/statsbomb/corners_360_expanded.csv'
        combined.to_csv(output_file, index=False)

        print(f"\nSaved to: {output_file}")
        print(f"Total corners: {len(combined)}")
        print(f"Original: {len(existing)}")
        print(f"New: {len(all_new_corners)}")
        print(f"Increase: {len(all_new_corners) / len(existing) * 100:.1f}%")

        print("\nNew competitions:")
        for comp_name, season_name, count in successful_comps:
            print(f"  {comp_name} {season_name}: {count}")
    else:
        print("\nNo new corners found with 360 data")
        print("The existing dataset likely covers all available 360 data")

if __name__ == "__main__":
    main()
