#!/usr/bin/env python3
"""
Debug a single corner in detail to see what events occur after it.
"""

import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.receiver_labeler import ReceiverLabeler

try:
    from statsbombpy import sb
except ImportError:
    print("ERROR: statsbombpy not installed")
    sys.exit(1)

# Load corners CSV
corners_df = pd.read_csv("data/raw/statsbomb/corners_360.csv")

# Pick first corner (which failed to find receiver)
corner_row = corners_df.iloc[0]
corner_id = corner_row['corner_id']
match_id = corner_row['match_id']

print(f"=== Analyzing Corner {corner_id} ===")
print(f"Match ID: {match_id}")
print(f"Team: {corner_row['team']}")
print(f"Player: {corner_row['player']}")
print(f"CSV has receiver: {pd.notna(corner_row['receiver_name'])}")
if pd.notna(corner_row['receiver_name']):
    print(f"CSV receiver: {corner_row['receiver_name']}")

# Load events
print("\nLoading events...")
events_df = sb.events(match_id=match_id, fmt='dataframe')

# Find corner event
corner_event_idx = events_df[events_df['id'] == corner_id].index
if len(corner_event_idx) == 0:
    print(f"ERROR: Corner event {corner_id} not found in events!")
    sys.exit(1)

corner_idx = corner_event_idx[0]
corner_event = events_df.loc[corner_idx]

print(f"\n=== Corner Event ===")
print(f"Type: {corner_event['type']}")
print(f"Timestamp: {corner_event['timestamp']}")
print(f"Player: {corner_event['player']}")
print(f"Player ID: {corner_event['player_id']}")

# Show next 10 events
print(f"\n=== Next 10 Events After Corner ===")
next_events = events_df.loc[corner_idx + 1 : corner_idx + 10]

for idx, event in next_events.iterrows():
    print(f"\nEvent {idx - corner_idx}:")
    print(f"  Type: {event['type']}")
    print(f"  Timestamp: {event['timestamp']}")
    print(f"  Player: {event.get('player', 'N/A')}")
    print(f"  Player ID: {event.get('player_id', 'N/A')}")
    print(f"  Team: {event.get('team', 'N/A')}")

    location = event.get('location')
    if location is not None:
        print(f"  Location: {location}")

# Try ReceiverLabeler
print(f"\n=== Testing ReceiverLabeler ===")
labeler = ReceiverLabeler()
player_id, player_name, location = labeler.find_receiver(events_df, corner_id)

if player_id is not None:
    print(f"✓ Found receiver: {player_name} (ID: {player_id})")
    if location:
        print(f"  Location: {location}")
else:
    print(f"✗ No receiver found")

# Check what the valid events are
print(f"\n=== ReceiverLabeler Valid Events ===")
print(labeler.VALID_RECEIVER_EVENTS)
