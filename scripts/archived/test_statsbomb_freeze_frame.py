#!/usr/bin/env python3
"""
Test script to check if StatsBomb freeze frames include player identity.
"""

from statsbombpy import sb
import pandas as pd

# Get a sample match
competitions = sb.competitions()
comp = competitions[competitions['competition_name'] == 'La Liga'].iloc[0]

matches = sb.matches(competition_id=comp['competition_id'], season_id=comp['season_id'])
match_id = matches.iloc[0]['match_id']

print(f"Loading events for match {match_id}")
events = sb.events(match_id=match_id)

# Find a corner kick with freeze frame
corners = events[events['type'] == 'Pass']
corners = corners[corners['pass_type'] == 'Corner']

if len(corners) > 0:
    corner = corners.iloc[0]
    print(f"\nFound corner kick:")
    print(f"Player: {corner['player']}")
    print(f"Team: {corner['team']}")

    if 'freeze_frame' in corner and corner['freeze_frame'] is not None:
        ff = corner['freeze_frame']
        print(f"\nFreeze frame type: {type(ff)}")
        print(f"Freeze frame length: {len(ff)}")

        if len(ff) > 0:
            print(f"\nFirst freeze frame player:")
            print(ff[0])

            # Check what fields are available
            if isinstance(ff[0], dict):
                print(f"\nAvailable fields: {list(ff[0].keys())}")
    else:
        print("\nNo freeze frame data")
else:
    print("No corners found")
