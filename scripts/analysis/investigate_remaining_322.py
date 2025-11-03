#!/usr/bin/env python3
"""
Deep investigation of the remaining 322 corners without receiver labels.
Can we reach 100% coverage or is there a fundamental limitation?
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
print("INVESTIGATING REMAINING 322 CORNERS WITHOUT RECEIVERS")
print("="*80)

# Load graphs with v2 labels (60s window)
print("\nLoading graphs...")
with open("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver_v2.pkl", 'rb') as f:
    graphs = pickle.load(f)

# Load corners CSV
corners_df = pd.read_csv("data/raw/statsbomb/corners_360.csv")

# Get graphs without receivers
graphs_without = [g for g in graphs if g.receiver_player_id is None]
print(f"Graphs without receiver: {len(graphs_without)}")

# Get unique base corner IDs
missing_base_corners = set()
for graph in graphs_without:
    base_corner_id = graph.corner_id.split('_t')[0].split('_mirror')[0]
    missing_base_corners.add(base_corner_id)

print(f"Unique base corners without receivers: {len(missing_base_corners)}")

# Analyze all remaining corners in detail
print("\n" + "="*80)
print("DETAILED ANALYSIS OF REMAINING CORNERS")
print("="*80)

labeler = ReceiverLabeler()
events_cache = {}

analysis = {
    'no_subsequent_events': [],
    'only_non_receiver_events': [],
    'corner_taker_only': [],
    'very_long_delay': [],
    'corner_not_in_events': [],
    'match_load_failed': []
}

event_type_after_corner = Counter()
time_to_first_event = []

print("\nAnalyzing all corners without receivers...\n")

for idx, corner_id in enumerate(sorted(missing_base_corners), 1):
    # Get match_id
    match_row = corners_df[corners_df['corner_id'] == corner_id]
    if len(match_row) == 0:
        analysis['corner_not_in_events'].append(corner_id)
        continue

    match_id = match_row.iloc[0]['match_id']

    # Load events
    if match_id not in events_cache:
        try:
            events_df = sb.events(match_id=match_id, fmt='dataframe')
            events_cache[match_id] = events_df
        except Exception as e:
            analysis['match_load_failed'].append((corner_id, str(e)))
            continue
    else:
        events_df = events_cache[match_id]

    # Find corner event
    corner_event_idx = events_df[events_df['id'] == corner_id].index
    if len(corner_event_idx) == 0:
        analysis['corner_not_in_events'].append(corner_id)
        continue

    corner_idx = corner_event_idx[0]
    corner_event = events_df.loc[corner_idx]
    corner_taker_id = corner_event.get('player_id')
    corner_timestamp = corner_event.get('timestamp')
    corner_time = labeler._parse_timestamp(corner_timestamp)

    # Look at subsequent events (up to 200 to find ANY event)
    next_events = events_df.loc[corner_idx + 1 : corner_idx + 200]

    if len(next_events) == 0:
        analysis['no_subsequent_events'].append(corner_id)
        if idx <= 20:
            print(f"{idx}. {corner_id[:8]}... - NO SUBSEQUENT EVENTS (end of match/half?)")
        continue

    # Find first event and its time
    first_event = next_events.iloc[0]
    first_event_time = labeler._parse_timestamp(first_event.get('timestamp'))
    if first_event_time and corner_time:
        time_diff = first_event_time - corner_time
        time_to_first_event.append(time_diff)
    else:
        time_diff = None

    # Find first valid receiver event
    found_valid_event = False
    first_valid_event_type = None
    first_valid_event_time_diff = None
    is_corner_taker = False

    for _, event in next_events.iterrows():
        event_type = event.get('type')
        event_timestamp = event.get('timestamp')
        event_time = labeler._parse_timestamp(event_timestamp)

        if labeler._is_valid_receiver_event(event_type):
            found_valid_event = True
            first_valid_event_type = event_type
            player_id = event.get('player_id')

            if event_time and corner_time:
                first_valid_event_time_diff = event_time - corner_time

            # Check if it's corner taker
            if player_id == corner_taker_id:
                is_corner_taker = True

            break

    # Categorize
    if not found_valid_event:
        analysis['only_non_receiver_events'].append(corner_id)
        event_type_after_corner[first_event.get('type')] += 1
        if idx <= 20:
            time_str = f"{time_diff:.1f}s" if time_diff else "???"
            print(f"{idx}. {corner_id[:8]}... - No valid receiver events (first: {first_event.get('type')} at +{time_str})")
    elif is_corner_taker:
        analysis['corner_taker_only'].append(corner_id)
        if idx <= 20:
            time_str = f"{first_valid_event_time_diff:.1f}s" if first_valid_event_time_diff else "???"
            print(f"{idx}. {corner_id[:8]}... - Corner taker first touch ({first_valid_event_type} at +{time_str})")
    else:
        # Should have been found by 60s window
        if first_valid_event_time_diff and first_valid_event_time_diff > 60:
            analysis['very_long_delay'].append((corner_id, first_valid_event_time_diff, first_valid_event_type))
            if idx <= 20:
                print(f"{idx}. {corner_id[:8]}... - Very long delay ({first_valid_event_type} at +{first_valid_event_time_diff:.1f}s)")
        else:
            # Mystery case - should have been caught
            if idx <= 20:
                time_str = f"{first_valid_event_time_diff:.1f}s" if first_valid_event_time_diff else "???"
                print(f"{idx}. {corner_id[:8]}... - MYSTERY: valid event at +{time_str} but not caught by labeler")

print("\n" + "="*80)
print("CATEGORIZATION SUMMARY")
print("="*80)
print(f"Total corners analyzed: {len(missing_base_corners)}")
print(f"\n1. No subsequent events (end of match/half): {len(analysis['no_subsequent_events'])}")
print(f"2. Only non-receiver events: {len(analysis['only_non_receiver_events'])}")
print(f"3. Corner taker is first touch: {len(analysis['corner_taker_only'])}")
print(f"4. Very long delay (>60s): {len(analysis['very_long_delay'])}")
print(f"5. Corner not in events: {len(analysis['corner_not_in_events'])}")
print(f"6. Match load failed: {len(analysis['match_load_failed'])}")

print("\n" + "="*80)
print("EVENT TYPES AFTER CORNERS (non-receiver events)")
print("="*80)
for event_type, count in event_type_after_corner.most_common():
    print(f"  {event_type:30} {count:>4}")

if time_to_first_event:
    import numpy as np
    print("\n" + "="*80)
    print("TIME TO FIRST EVENT (any type)")
    print("="*80)
    times = np.array(time_to_first_event)
    print(f"Mean: {np.mean(times):.1f}s")
    print(f"Median: {np.median(times):.1f}s")
    print(f"Max: {np.max(times):.1f}s")
    print(f"Min: {np.min(times):.1f}s")

print("\n" + "="*80)
print("RECOMMENDATIONS FOR 100% COVERAGE")
print("="*80)

total_fixable = 0

# Corner taker blocking
if analysis['corner_taker_only']:
    print(f"\n1. REMOVE CORNER TAKER EXCLUSION: {len(analysis['corner_taker_only'])} corners")
    print("   - Currently we skip if corner taker has first touch (short corners)")
    print("   - TacticAI paper doesn't mention excluding corner taker")
    print("   - FIX: Remove the corner_taker_id check in ReceiverLabeler")
    total_fixable += len(analysis['corner_taker_only'])

# Very long delays
if analysis['very_long_delay']:
    print(f"\n2. INCREASE TIME WINDOW FURTHER: {len(analysis['very_long_delay'])} corners")
    print("   - Events occurring >60s after corner")
    delays = [t for _, t, _ in analysis['very_long_delay']]
    if delays:
        import numpy as np
        max_delay = np.max(delays)
        print(f"   - Max delay: {max_delay:.1f}s")
        print(f"   - FIX: Increase to {max_delay + 10:.0f}s window")
    total_fixable += len(analysis['very_long_delay'])

# Non-receiver events
if analysis['only_non_receiver_events']:
    print(f"\n3. EXPAND VALID EVENT TYPES: {len(analysis['only_non_receiver_events'])} corners")
    print("   - Currently valid: Pass, Shot, Duel, Interception, Clearance, Miscontrol, Ball Receipt")
    print("   - Need to see what events appear after these corners")
    # Show examples
    if event_type_after_corner:
        print("   - Events appearing but not counted:")
        for event_type, count in event_type_after_corner.most_common(5):
            print(f"     • {event_type}: {count}")

# Unfixable
unfixable = len(analysis['no_subsequent_events'])
print(f"\n4. UNFIXABLE: {unfixable} corners")
print("   - No events after corner (end of match/half)")
print("   - Cannot assign receiver if no subsequent play")

print(f"\n" + "="*80)
print(f"POTENTIAL COVERAGE IF ALL FIXES APPLIED:")
print(f"Current: {len(graphs) - len(graphs_without)}/{len(graphs)} (94.5%)")
print(f"Fixable: +{total_fixable} corners")
print(f"Maximum possible: {len(graphs) - len(graphs_without) + total_fixable}/{len(graphs)} ({(len(graphs) - len(graphs_without) + total_fixable)/len(graphs)*100:.1f}%)")
print(f"Theoretical limit: {len(graphs) - unfixable}/{len(graphs)} ({(len(graphs) - unfixable)/len(graphs)*100:.1f}%)")
print("="*80)
