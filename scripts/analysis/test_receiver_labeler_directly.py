#!/usr/bin/env python3
"""
Test ReceiverLabeler directly on a sample of corners to see return rate.
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.receiver_labeler import ReceiverLabeler

try:
    from statsbombpy import sb
except ImportError:
    print("ERROR: statsbombpy not installed")
    sys.exit(1)

# Load corners CSV
print("Loading corners CSV...")
corners_df = pd.read_csv("data/raw/statsbomb/corners_360.csv")
print(f"Total corners: {len(corners_df)}")

# Test ReceiverLabeler on first 50 corners
labeler = ReceiverLabeler()
found_count = 0
not_found_count = 0

print("\n=== Testing ReceiverLabeler on first 50 corners ===\n")

for idx in range(min(50, len(corners_df))):
    row = corners_df.iloc[idx]
    corner_id = row['corner_id']
    match_id = row['match_id']

    # Load events for this match
    try:
        events_df = sb.events(match_id=match_id, fmt='dataframe')
    except Exception as e:
        print(f"❌ Corner {idx+1}: Failed to load events for match {match_id}")
        not_found_count += 1
        continue

    # Find receiver
    player_id, player_name, location = labeler.find_receiver(
        events_df,
        corner_event_id=corner_id
    )

    if player_id is not None:
        found_count += 1
        event_type = "Unknown"

        # Find the receiver event type
        receiver_events = events_df[
            (events_df['player_id'] == player_id) &
            (events_df['player'] == player_name)
        ]

        if len(receiver_events) > 0:
            # Get first event after corner
            corner_idx = events_df[events_df['id'] == corner_id].index
            if len(corner_idx) > 0:
                receiver_after_corner = receiver_events[
                    receiver_events.index > corner_idx[0]
                ]
                if len(receiver_after_corner) > 0:
                    event_type = receiver_after_corner.iloc[0]['type']

        print(f"✓ Corner {idx+1}: Found receiver (player_id={player_id}, event={event_type})")
    else:
        not_found_count += 1
        print(f"✗ Corner {idx+1}: No receiver found")

print(f"\n=== RESULTS ===")
print(f"Found receiver: {found_count}/50 ({found_count/50*100:.1f}%)")
print(f"No receiver: {not_found_count}/50 ({not_found_count/50*100:.1f}%)")

print(f"\n=== Comparing to old CSV approach ===")
# Count how many of these 50 have receiver in CSV
csv_receiver_count = corners_df.iloc[:50]['receiver_name'].notna().sum()
print(f"CSV had receiver labels: {csv_receiver_count}/50 ({csv_receiver_count/50*100:.1f}%)")
