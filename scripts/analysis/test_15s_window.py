#!/usr/bin/env python3
"""
Test ReceiverLabeler with 15-second window.
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
corners_df = pd.read_csv("data/raw/statsbomb/corners_360.csv")
print(f"Total corners: {len(corners_df)}")

# Test ReceiverLabeler on first 50 corners with 15s window
labeler = ReceiverLabeler()
found_with_5s = 0
found_with_15s = 0

print("\n=== Testing ReceiverLabeler with 5s vs 15s window ===\n")

for idx in range(min(50, len(corners_df))):
    row = corners_df.iloc[idx]
    corner_id = row['corner_id']
    match_id = row['match_id']

    # Load events for this match
    try:
        events_df = sb.events(match_id=match_id, fmt='dataframe')
    except Exception as e:
        continue

    # Test with 5s window (default)
    player_id_5s, _, _ = labeler.find_receiver(events_df, corner_id, max_time_diff=5.0)
    if player_id_5s is not None:
        found_with_5s += 1

    # Test with 15s window
    player_id_15s, _, _ = labeler.find_receiver(events_df, corner_id, max_time_diff=15.0)
    if player_id_15s is not None:
        found_with_15s += 1

print(f"Found with 5s window: {found_with_5s}/50 ({found_with_5s/50*100:.1f}%)")
print(f"Found with 15s window: {found_with_15s}/50 ({found_with_15s/50*100:.1f}%)")
print(f"Improvement: +{found_with_15s - found_with_5s} corners ({(found_with_15s - found_with_5s)/50*100:.1f}%)")
