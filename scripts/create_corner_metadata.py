#!/usr/bin/env python3
"""Create corner metadata JSON with match info.

Enriches the corners CSV with competition, season, team names, etc.
"""

import json
import csv
import re
from pathlib import Path
from tqdm import tqdm


def parse_game_path(game_path: str) -> dict:
    """Parse game_path to extract competition, season, teams.

    Example: france_ligue-1/2016-2017/2016-09-23 - 21-45 Toulouse 2 - 0 Paris SG/
    """
    parts = game_path.rstrip('/').split('/')

    if len(parts) >= 3:
        competition = parts[0].replace('_', ' ').replace('-', ' ').title()
        season = parts[1]

        # Parse match info from last part: "2016-09-23 - 21-45 Toulouse 2 - 0 Paris SG"
        match_part = parts[-1]

        # Extract date, time, and score/teams
        date_match = re.match(r'(\d{4}-\d{2}-\d{2})\s*-\s*(\d{2}-\d{2})\s+(.+)', match_part)
        if date_match:
            date = date_match.group(1)
            time = date_match.group(2).replace('-', ':')
            teams_score = date_match.group(3)

            # Try to parse "Home X - Y Away"
            score_match = re.match(r'(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+)', teams_score)
            if score_match:
                home_team = score_match.group(1).strip()
                home_score = int(score_match.group(2))
                away_score = int(score_match.group(3))
                away_team = score_match.group(4).strip()
            else:
                home_team = teams_score
                away_team = "Unknown"
                home_score = away_score = 0
        else:
            date = time = "Unknown"
            home_team = away_team = "Unknown"
            home_score = away_score = 0
    else:
        competition = season = "Unknown"
        date = time = "Unknown"
        home_team = away_team = "Unknown"
        home_score = away_score = 0

    return {
        'competition': competition,
        'season': season,
        'date': date,
        'time': time,
        'home_team': home_team,
        'away_team': away_team,
        'home_score': home_score,
        'away_score': away_score,
    }


def main():
    # Load corners CSV
    corners_path = Path('/home/mseo/CornerTactics/data/processed/soccernet_corners.csv')

    corners = []
    with open(corners_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            corners.append(row)

    print(f"Loaded {len(corners)} corners")

    # Enrich with metadata
    metadata = []
    for corner in tqdm(corners, desc="Enriching corners"):
        game_info = parse_game_path(corner['game_path'])

        # Determine corner-taking team name
        if corner['team'] == 'home':
            corner_team = game_info['home_team']
        elif corner['team'] == 'away':
            corner_team = game_info['away_team']
        else:
            corner_team = 'Unknown'

        metadata.append({
            'corner_id': int(corner['corner_id']),
            'game_path': corner['game_path'],
            'half': int(corner['half']),
            'timestamp_seconds': int(corner['timestamp_seconds']),
            'position_ms': int(corner['position_ms']),
            'team': corner['team'],
            'corner_team': corner_team,
            'visibility': corner['visibility'],
            **game_info
        })

    # Save metadata JSON
    output_path = Path('/home/mseo/CornerTactics/data/processed/corner_metadata.json')
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to: {output_path}")

    # Print statistics
    competitions = set(m['competition'] for m in metadata)
    seasons = set(m['season'] for m in metadata)
    visible = sum(1 for m in metadata if m['visibility'] == 'visible')

    print(f"\nStatistics:")
    print(f"  Total corners: {len(metadata)}")
    print(f"  Visible: {visible}")
    print(f"  Competitions: {len(competitions)}")
    for comp in sorted(competitions):
        count = sum(1 for m in metadata if m['competition'] == comp)
        print(f"    - {comp}: {count}")
    print(f"  Seasons: {', '.join(sorted(seasons))}")


if __name__ == '__main__':
    main()
