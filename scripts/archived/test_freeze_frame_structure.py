#!/usr/bin/env python3
"""
Quick test to check StatsBomb freeze frame structure without pandas import issues.
"""

import sys
sys.path.insert(0, '/home/mseo/.local/lib/python3.11/site-packages')

from statsbombpy import sb

# Get a sample match
print("Fetching competitions...")
competitions = sb.competitions()
la_liga = competitions[competitions['competition_name'] == 'La Liga'].iloc[0]

print("Fetching matches...")
matches = sb.matches(competition_id=la_liga['competition_id'], season_id=la_liga['season_id'])
match_id = matches.iloc[0]['match_id']

print(f"Loading frames for match {match_id}...")
frames_df = sb.frames(match_id=match_id, fmt='dataframe')

print(f"\nFrame columns: {list(frames_df.columns)}")
print(f"\nSample frame row:")
print(frames_df.iloc[0])

# Check if player info exists
if 'player' in frames_df.columns:
    print(f"\nPlayer column type: {type(frames_df['player'].iloc[0])}")
    print(f"Player value: {frames_df['player'].iloc[0]}")
