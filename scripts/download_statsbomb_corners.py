#!/usr/bin/env python3
"""
Fast StatsBomb 360 downloader using pandas operations.

Instead of iterating through all competitions/matches manually,
this script uses pandas filtering to efficiently identify and download
only the data we need.
"""

import sys
from pathlib import Path
import json
import warnings

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

print("="*70)
print("StatsBomb 360 Fast Downloader")
print("="*70)

# Import statsbombpy
try:
    from statsbombpy import sb
    print("✓ statsbombpy imported successfully")
except ImportError:
    print("✗ Installing statsbombpy...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsbombpy", "--quiet"])
    from statsbombpy import sb
    print("✓ statsbombpy installed")

# Create output directory
output_dir = Path("data/raw/statsbomb")
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Get all competitions and matches
print("\n[1/3] Fetching competitions and matches...")
competitions = sb.competitions()
print(f"✓ Found {len(competitions)} total competitions")

# Filter for men's competitions only
competitions = competitions[competitions['competition_gender'] == 'male'].copy()
print(f"✓ Filtered to {len(competitions)} men's competitions")

# Exclude youth competitions (not professional)
youth_keywords = ['U20', 'U-20', 'U19', 'U-19', 'U17', 'U-17', 'Youth', 'Academy']
before_youth_filter = len(competitions)
for keyword in youth_keywords:
    competitions = competitions[~competitions['competition_name'].str.contains(keyword, case=False, na=False)]
excluded_youth = before_youth_filter - len(competitions)
if excluded_youth > 0:
    print(f"✓ Excluded {excluded_youth} youth competitions")
print(f"✓ Final: {len(competitions)} professional men's competitions")

# Priority leagues (focus on these first - ordered by importance)
priority_leagues = [
    # Top European Leagues
    'Champions League',
    'La Liga',
    'Premier League',
    'Bundesliga',
    'Serie A',
    'Ligue 1',
    # International Tournaments
    'FIFA World Cup',
    'UEFA Euro',
    'Copa America',
    'African Cup of Nations',
    # Other European Competitions
    'UEFA Europa League',
    'Copa del Rey',
    # Other Professional Leagues
    'Major League Soccer',
    'Indian Super league',
    'Liga Profesional',
    'North American League',
]

# Sort competitions - priority leagues first
def get_priority(comp_name):
    for i, league in enumerate(priority_leagues):
        if league.lower() in comp_name.lower():
            return i
    return 999

competitions['priority'] = competitions['competition_name'].apply(get_priority)
competitions = competitions.sort_values('priority').drop('priority', axis=1)

print(f"\n{'='*70}")
print("COMPETITIONS TO PROCESS")
print(f"{'='*70}")
# Show unique competition names
unique_comps = competitions['competition_name'].unique()
print(f"Total unique competitions: {len(unique_comps)}\n")
for comp_name in sorted(unique_comps):
    comp_seasons = competitions[competitions['competition_name'] == comp_name]
    num_seasons = len(comp_seasons)
    country = comp_seasons['country_name'].iloc[0]
    print(f"  • {comp_name} ({country}) - {num_seasons} season(s)")
print(f"\n{'='*70}\n")

# Get all matches for filtered competitions
all_matches = []
for _, comp in tqdm(competitions.iterrows(), total=len(competitions), desc="Fetching matches"):
    try:
        matches = sb.matches(
            competition_id=comp['competition_id'],
            season_id=comp['season_id']
        )
        # Add competition metadata
        matches['competition_name'] = comp['competition_name']
        matches['season_name'] = comp['season_name']
        all_matches.append(matches)
    except Exception as e:
        continue

matches_df = pd.concat(all_matches, ignore_index=True)
print(f"✓ Found {len(matches_df)} total matches across all competitions")

# Step 2: Try to get 360 data and events for each match
print("\n[2/3] Downloading 360 data and extracting corners...")
corners_list = []
matches_with_360 = 0
matches_without_360 = 0

for idx, match in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Processing matches"):
    match_id = match['match_id']

    try:
        # Try to get 360 frames - will fail if not available
        frames_df = sb.frames(match_id=match_id, fmt='dataframe')

        # Get events to find corners
        events_df = sb.events(match_id=match_id, fmt='dataframe')

        # Filter for corner kicks using pandas
        corner_events = events_df[
            (events_df['type'] == 'Pass') &
            (events_df['pass_type'].fillna('').str.contains('Corner', case=False, na=False))
        ].copy()

        if len(corner_events) == 0:
            matches_with_360 += 1
            continue

        # For each corner, match with 360 frames
        for _, corner in corner_events.iterrows():
            corner_frames = frames_df[frames_df['id'] == corner['id']]

            if len(corner_frames) > 0:
                # Use pandas to aggregate attacking/defending players
                attacking = corner_frames[corner_frames['teammate'] == True]
                defending = corner_frames[corner_frames['teammate'] == False]

                corners_list.append({
                    'match_id': match_id,
                    'competition': match['competition_name'],
                    'season': match['season_name'],
                    'match_date': match.get('match_date', 'Unknown'),
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'corner_id': corner['id'],
                    'minute': corner.get('minute'),
                    'second': corner.get('second'),
                    'team': corner.get('team'),
                    'player': corner.get('player'),
                    'location_x': corner.get('location', [None, None])[0] if isinstance(corner.get('location'), list) and len(corner.get('location', [])) >= 2 else None,
                    'location_y': corner.get('location', [None, None])[1] if isinstance(corner.get('location'), list) and len(corner.get('location', [])) >= 2 else None,
                    'end_x': corner.get('pass_end_location', [None, None])[0] if isinstance(corner.get('pass_end_location'), list) and len(corner.get('pass_end_location', [])) >= 2 else None,
                    'end_y': corner.get('pass_end_location', [None, None])[1] if isinstance(corner.get('pass_end_location'), list) and len(corner.get('pass_end_location', [])) >= 2 else None,
                    'num_attacking_players': len(attacking),
                    'num_defending_players': len(defending),
                    'total_visible_players': len(corner_frames),
                    'attacking_positions': json.dumps(attacking['location'].tolist()),
                    'defending_positions': json.dumps(defending['location'].tolist()),
                })

        matches_with_360 += 1

    except Exception as e:
        # Match doesn't have 360 data
        matches_without_360 += 1
        continue

print(f"✓ Matches with 360 data: {matches_with_360}")
print(f"  Matches without 360 data: {matches_without_360}")

# Step 3: Create DataFrame and save
print("\n[3/3] Saving results...")

if len(corners_list) == 0:
    print("✗ No corners with 360 data found!")
    sys.exit(1)

# Use pandas to create DataFrame from list
corners_df = pd.DataFrame(corners_list)

# Save to CSV
output_csv = output_dir / "corners_360.csv"
corners_df.to_csv(output_csv, index=False)

print(f"✓ Extracted {len(corners_df)} corners with 360 data")
print(f"✓ Saved to: {output_csv}")

# Summary statistics using pandas
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"Total matches processed: {len(matches_df)}")
print(f"Matches with 360 data: {matches_with_360}")
print(f"Total corners with 360: {len(corners_df)}")
print(f"Average players per corner: {corners_df['total_visible_players'].mean():.1f}")

print(f"\nPlayer count distribution:")
print(corners_df['total_visible_players'].value_counts().sort_index().head(10))

print(f"\nCorners by competition:")
print(corners_df['competition'].value_counts())

print(f"\nTop teams by corner count:")
print(corners_df['team'].value_counts().head(10))

print("\n" + "="*70)
print("DOWNLOAD COMPLETE")
print("="*70)
print(f"\nOutput: {output_csv}")
print(f"Size: {len(corners_df)} rows × {len(corners_df.columns)} columns")
