#!/usr/bin/env python3
"""
Test different time windows to see coverage improvement.
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

print("="*80)
print("TESTING DIFFERENT TIME WINDOWS")
print("="*80)

# Load corners CSV
corners_df = pd.read_csv("data/raw/statsbomb/corners_360.csv")
print(f"\nTotal corners in dataset: {len(corners_df)}")

# Test different time windows on a sample
sample_size = 100
sample_corners = corners_df.iloc[:sample_size]

time_windows = [5, 10, 15, 20, 30, 60, 120]
results = {}

labeler = ReceiverLabeler()

print(f"\nTesting on first {sample_size} corners...")
print("(This may take a few minutes)\n")

for max_time in time_windows:
    found_count = 0
    events_cache = {}

    for idx, row in sample_corners.iterrows():
        corner_id = row['corner_id']
        match_id = row['match_id']

        # Load events
        if match_id not in events_cache:
            try:
                events_df = sb.events(match_id=match_id, fmt='dataframe')
                events_cache[match_id] = events_df
            except:
                continue
        else:
            events_df = events_cache[match_id]

        # Find receiver with this time window
        player_id, _, _ = labeler.find_receiver(
            events_df,
            corner_id,
            max_time_diff=max_time
        )

        if player_id is not None:
            found_count += 1

    coverage = found_count / sample_size * 100
    results[max_time] = (found_count, coverage)
    print(f"Window {max_time:>3}s: {found_count:>3}/{sample_size} ({coverage:>5.1f}%)")

print("\n" + "="*80)
print("MARGINAL GAIN ANALYSIS")
print("="*80)

prev_count = 0
for max_time in time_windows:
    count, coverage = results[max_time]
    gain = count - prev_count
    gain_pct = (gain / sample_size) * 100

    if prev_count > 0:
        print(f"{max_time:>3}s vs previous: +{gain:>2} corners (+{gain_pct:>4.1f}%)")
    prev_count = count

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Find diminishing returns point
for i in range(1, len(time_windows)):
    current_window = time_windows[i]
    prev_window = time_windows[i-1]

    current_count = results[current_window][0]
    prev_count = results[prev_window][0]

    gain = current_count - prev_count

    if gain <= 1:  # Less than 1% improvement
        print(f"\nDiminishing returns after {prev_window}s window")
        print(f"Recommended: Use {prev_window}s for optimal coverage/quality tradeoff")
        break
