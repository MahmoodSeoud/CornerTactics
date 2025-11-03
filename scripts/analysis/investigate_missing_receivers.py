#!/usr/bin/env python3
"""
Comprehensive investigation of why 32.3% of corners don't have receiver labels.
"""

import pickle
import pandas as pd
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.receiver_labeler import ReceiverLabeler

try:
    from statsbombpy import sb
except ImportError:
    print("ERROR: statsbombpy not installed")
    sys.exit(1)

print("="*80)
print("INVESTIGATING MISSING RECEIVERS")
print("="*80)

# Load graphs with v2 labels
print("\nLoading graphs...")
with open("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver_v2.pkl", 'rb') as f:
    graphs = pickle.load(f)

print(f"Total graphs: {len(graphs)}")

# Load corners CSV
print("Loading corners CSV...")
corners_df = pd.read_csv("data/raw/statsbomb/corners_360.csv")
print(f"Total corners in CSV: {len(corners_df)}")

# Split graphs by whether they have receivers
graphs_with_receiver = [g for g in graphs if g.receiver_player_id is not None]
graphs_without_receiver = [g for g in graphs if g.receiver_player_id is None]

print(f"\nGraphs with receiver: {len(graphs_with_receiver)} ({len(graphs_with_receiver)/len(graphs)*100:.1f}%)")
print(f"Graphs without receiver: {len(graphs_without_receiver)} ({len(graphs_without_receiver)/len(graphs)*100:.1f}%)")

# Get unique base corner IDs for corners without receivers
print("\n" + "="*80)
print("ANALYZING CORNERS WITHOUT RECEIVERS")
print("="*80)

missing_base_corners = set()
for graph in graphs_without_receiver:
    base_corner_id = graph.corner_id.split('_t')[0].split('_mirror')[0]
    missing_base_corners.add(base_corner_id)

print(f"\nUnique base corners without receivers: {len(missing_base_corners)}")

# Sample 20 corners without receivers to analyze
sample_missing = list(missing_base_corners)[:20]

print(f"\nAnalyzing first 20 corners without receivers...")
print(f"(Looking at events that occur after these corners)\n")

event_type_counter = Counter()
labeler = ReceiverLabeler()
events_cache = {}

for idx, corner_id in enumerate(sample_missing, 1):
    # Get match_id
    match_row = corners_df[corners_df['corner_id'] == corner_id]
    if len(match_row) == 0:
        print(f"{idx}. Corner {corner_id[:8]}... - NOT IN CSV")
        continue

    match_id = match_row.iloc[0]['match_id']

    # Load events
    if match_id not in events_cache:
        try:
            events_df = sb.events(match_id=match_id, fmt='dataframe')
            events_cache[match_id] = events_df
        except Exception as e:
            print(f"{idx}. Corner {corner_id[:8]}... - FAILED TO LOAD EVENTS")
            continue
    else:
        events_df = events_cache[match_id]

    # Find corner event
    corner_event_idx = events_df[events_df['id'] == corner_id].index
    if len(corner_event_idx) == 0:
        print(f"{idx}. Corner {corner_id[:8]}... - CORNER EVENT NOT FOUND")
        continue

    corner_idx = corner_event_idx[0]
    corner_event = events_df.loc[corner_idx]
    corner_taker_id = corner_event.get('player_id')
    corner_timestamp = corner_event.get('timestamp')
    corner_time = labeler._parse_timestamp(corner_timestamp)

    # Look at next 10 events
    next_events = events_df.loc[corner_idx + 1 : corner_idx + 10]

    print(f"{idx}. Corner {corner_id[:8]}... (match {match_id})")
    print(f"   Corner timestamp: {corner_timestamp}")

    for event_idx, event in next_events.iterrows():
        event_type = event.get('type')
        event_timestamp = event.get('timestamp')
        event_time = labeler._parse_timestamp(event_timestamp)

        # Calculate time diff
        if event_time and corner_time:
            time_diff = event_time - corner_time
        else:
            time_diff = None

        player_id = event.get('player_id')
        player_name = event.get('player', 'N/A')

        # Check if this is the corner taker
        is_taker = " (CORNER TAKER)" if player_id == corner_taker_id else ""

        # Track event type
        if time_diff and time_diff <= 15.0:
            event_type_counter[event_type] += 1

        time_str = f"{time_diff:.1f}s" if time_diff else "???"
        print(f"   +{time_str:>6} | {event_type:20} | {player_name[:25]:25}{is_taker}")

        # Stop after 15 seconds
        if time_diff and time_diff > 15.0:
            break

    print()

print("\n" + "="*80)
print("EVENT TYPE DISTRIBUTION (first 15s after corners without receivers)")
print("="*80)
for event_type, count in event_type_counter.most_common():
    print(f"{event_type:30} {count:>4}")

print("\n" + "="*80)
print("COMPARING TO VALID RECEIVER EVENTS")
print("="*80)
print(f"Valid receiver events: {labeler.VALID_RECEIVER_EVENTS}")

# Check which event types are NOT in valid list
all_event_types = set(event_type_counter.keys())
valid_events = set(labeler.VALID_RECEIVER_EVENTS)
missing_events = all_event_types - valid_events

print(f"\nEvent types appearing after corners but NOT in valid list:")
for event_type in sorted(missing_events):
    count = event_type_counter[event_type]
    print(f"  {event_type:30} {count:>4} occurrences")

print("\n" + "="*80)
print("CHECKING FOR CORNER TAKER ISSUE")
print("="*80)

# Check if many corners have corner taker as first touch
corner_taker_blocks = 0
sample_with_taker_block = []

for corner_id in list(missing_base_corners)[:50]:
    match_row = corners_df[corners_df['corner_id'] == corner_id]
    if len(match_row) == 0:
        continue

    match_id = match_row.iloc[0]['match_id']

    if match_id not in events_cache:
        try:
            events_df = sb.events(match_id=match_id, fmt='dataframe')
            events_cache[match_id] = events_df
        except:
            continue
    else:
        events_df = events_cache[match_id]

    corner_event_idx = events_df[events_df['id'] == corner_id].index
    if len(corner_event_idx) == 0:
        continue

    corner_idx = corner_event_idx[0]
    corner_event = events_df.loc[corner_idx]
    corner_taker_id = corner_event.get('player_id')

    # Check if first valid event is by corner taker
    next_events = events_df.loc[corner_idx + 1 : corner_idx + 20]
    for _, event in next_events.iterrows():
        event_type = event.get('type')
        if labeler._is_valid_receiver_event(event_type):
            if event.get('player_id') == corner_taker_id:
                corner_taker_blocks += 1
                sample_with_taker_block.append(corner_id[:8])
            break

print(f"Corners where corner taker has first valid touch: {corner_taker_blocks}/50")
if sample_with_taker_block:
    print(f"Examples: {', '.join(sample_with_taker_block[:5])}...")
