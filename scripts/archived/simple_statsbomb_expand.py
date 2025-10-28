#!/usr/bin/env python3
"""
Simple script to expand StatsBomb corner dataset
Downloads from more competitions to increase from 1,118 to 5,000+ corners

Simplified version that avoids environment issues.
"""

import json
from statsbombpy import sb
import csv

def main():
    print("Expanding StatsBomb corner dataset...")
    print("=" * 60)

    # Get all competitions
    competitions = sb.competitions()

    # Priority competitions to download
    priority_comps = [
        'Premier League', 'La Liga', 'Champions League',
        'Serie A', 'Bundesliga', 'Ligue 1',
        'FIFA World Cup', 'UEFA Euro', 'Copa America',
        'Major League Soccer', 'NWSL'
    ]

    all_corners = []
    corners_by_comp = {}

    print("\nProcessing competitions...")
    for comp_name in priority_comps:
        # Find all seasons for this competition
        comp_seasons = competitions[
            competitions['competition_name'] == comp_name
        ]

        for _, row in comp_seasons.iterrows():
            try:
                # Get matches
                matches = sb.matches(
                    competition_id=row['competition_id'],
                    season_id=row['season_id']
                )

                comp_corners = 0

                # Process each match
                for _, match in matches.iterrows():
                    try:
                        # Get events with 360 data
                        events = sb.events(
                            match_id=match['match_id'],
                            include_360_metrics=True
                        )

                        # Filter corners with 360 data
                        if '360_freeze_frame' in events.columns:
                            corners = events[
                                (events['type_name'] == 'Pass') &
                                (events['pass_type_name'] == 'Corner') &
                                (events['360_freeze_frame'].notna())
                            ]

                            comp_corners += len(corners)

                            # Add to list
                            for _, corner in corners.iterrows():
                                freeze_frame = corner['360_freeze_frame']

                                # Count players
                                attacking = sum(1 for p in freeze_frame if p['teammate'])
                                defending = sum(1 for p in freeze_frame if not p['teammate'])

                                corner_data = {
                                    'match_id': match['match_id'],
                                    'competition': comp_name,
                                    'season': row['season_name'],
                                    'corner_id': corner.get('id', ''),
                                    'minute': corner.get('minute', 0),
                                    'second': corner.get('second', 0),
                                    'team': corner.get('team_name', ''),
                                    'player': corner.get('player_name', ''),
                                    'num_attacking': attacking,
                                    'num_defending': defending,
                                    'total_players': attacking + defending,
                                    'freeze_frame': json.dumps(freeze_frame)
                                }

                                all_corners.append(corner_data)

                    except Exception as e:
                        continue

                if comp_corners > 0:
                    corners_by_comp[f"{comp_name} {row['season_name']}"] = comp_corners
                    print(f"  {comp_name} {row['season_name']}: {comp_corners} corners")

            except Exception as e:
                continue

    print(f"\n{'='*60}")
    print(f"Total corners found: {len(all_corners)}")
    print(f"Competitions with data: {len(corners_by_comp)}")

    # Save to CSV
    if all_corners:
        output_file = 'data/raw/statsbomb/expanded_corners.csv'

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_corners[0].keys())
            writer.writeheader()
            writer.writerows(all_corners)

        print(f"\nSaved to: {output_file}")
        print(f"Improvement: {len(all_corners) / 1118:.1f}x more data than original!")

if __name__ == "__main__":
    main()