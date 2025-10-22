#!/usr/bin/env python3
"""
Add outcome labels to StatsBomb corners dataset
Fetches subsequent events (15-20 seconds window) and classifies outcomes:
- Goal: Goal scored within 20 seconds
- Shot: Shot attempt (saved/blocked/off target)
- Clearance: Defensive clearance
- Possession: Maintained possession without shot
- Loss: Lost possession (interception/duel lost)
"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from statsbombpy import sb

def parse_timestamp(timestamp_str):
    """Convert timestamp string to seconds"""
    if pd.isna(timestamp_str) or timestamp_str == '':
        return 0.0

    # Format: "HH:MM:SS.mmm"
    parts = str(timestamp_str).split(':')
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    return 0.0

def calculate_xthreat_delta(corner_location, outcome_location):
    """
    Calculate simplified xThreat change
    Higher value = more dangerous position

    StatsBomb pitch: 120 x 80
    Goal at (120, 40)
    """
    if not corner_location or not outcome_location:
        return 0.0

    try:
        corner_x, corner_y = corner_location
        outcome_x, outcome_y = outcome_location

        # Distance to goal (120, 40)
        corner_dist = ((120 - corner_x)**2 + (40 - corner_y)**2)**0.5
        outcome_dist = ((120 - outcome_x)**2 + (40 - outcome_y)**2)**0.5

        # Positive value means moved closer to goal (more dangerous)
        return corner_dist - outcome_dist
    except:
        return 0.0

def get_corner_outcome(events_df, corner_idx, max_time_window=20.0):
    """
    Analyze corner kick outcome with detailed classification

    Returns:
        dict with outcome classification and metrics
    """
    corner_event = events_df.iloc[corner_idx]
    corner_team = corner_event.get('team')
    corner_timestamp = corner_event.get('timestamp')
    corner_time_sec = parse_timestamp(corner_timestamp)
    corner_location = corner_event.get('pass_end_location')

    outcome = {
        'outcome_category': 'Unknown',
        'outcome_type': None,
        'outcome_team': None,
        'outcome_player': None,
        'same_team': None,
        'time_to_outcome': None,
        'events_to_outcome': None,
        'shot_outcome': None,
        'goal_scored': False,
        'xthreat_delta': 0.0,
        'outcome_location': None
    }

    # Scan forward through events
    for i in range(corner_idx + 1, min(corner_idx + 30, len(events_df))):
        next_event = events_df.iloc[i]
        next_timestamp = next_event.get('timestamp')
        next_time_sec = parse_timestamp(next_timestamp)
        time_diff = next_time_sec - corner_time_sec

        # Check time window
        if time_diff > max_time_window:
            break

        event_type = next_event.get('type')
        next_team = next_event.get('team')
        same_team = (next_team == corner_team)

        # Priority 1: Goal
        if event_type == 'Shot':
            shot_outcome_val = next_event.get('shot_outcome')
            location = next_event.get('location')

            if shot_outcome_val == 'Goal':
                outcome.update({
                    'outcome_category': 'Goal',
                    'outcome_type': 'Shot - Goal',
                    'outcome_team': next_team,
                    'outcome_player': next_event.get('player'),
                    'same_team': same_team,
                    'time_to_outcome': time_diff,
                    'events_to_outcome': i - corner_idx,
                    'shot_outcome': shot_outcome_val,
                    'goal_scored': True,
                    'outcome_location': location,
                    'xthreat_delta': calculate_xthreat_delta(corner_location, location)
                })
                return outcome
            else:
                # Shot but no goal
                outcome.update({
                    'outcome_category': 'Shot',
                    'outcome_type': f'Shot - {shot_outcome_val}',
                    'outcome_team': next_team,
                    'outcome_player': next_event.get('player'),
                    'same_team': same_team,
                    'time_to_outcome': time_diff,
                    'events_to_outcome': i - corner_idx,
                    'shot_outcome': shot_outcome_val,
                    'goal_scored': False,
                    'outcome_location': location,
                    'xthreat_delta': calculate_xthreat_delta(corner_location, location)
                })
                return outcome

        # Priority 2: Defensive actions (clearance/interception by defending team)
        if not same_team:
            if event_type == 'Clearance':
                location = next_event.get('location')
                outcome.update({
                    'outcome_category': 'Clearance',
                    'outcome_type': 'Clearance',
                    'outcome_team': next_team,
                    'outcome_player': next_event.get('player'),
                    'same_team': False,
                    'time_to_outcome': time_diff,
                    'events_to_outcome': i - corner_idx,
                    'outcome_location': location,
                    'xthreat_delta': calculate_xthreat_delta(corner_location, location)
                })
                return outcome

            elif event_type == 'Interception':
                location = next_event.get('location')
                outcome.update({
                    'outcome_category': 'Loss',
                    'outcome_type': 'Interception',
                    'outcome_team': next_team,
                    'outcome_player': next_event.get('player'),
                    'same_team': False,
                    'time_to_outcome': time_diff,
                    'events_to_outcome': i - corner_idx,
                    'outcome_location': location,
                    'xthreat_delta': calculate_xthreat_delta(corner_location, location)
                })
                return outcome

        # Priority 3: Second corner (retained possession, another chance)
        if same_team and event_type == 'Pass':
            pass_type = next_event.get('pass_type', '')
            if isinstance(pass_type, str) and 'corner' in pass_type.lower():
                outcome.update({
                    'outcome_category': 'Possession',
                    'outcome_type': 'Second Corner',
                    'outcome_team': next_team,
                    'outcome_player': next_event.get('player'),
                    'same_team': True,
                    'time_to_outcome': time_diff,
                    'events_to_outcome': i - corner_idx,
                })
                return outcome

        # Priority 4: Other possession loss
        if event_type in ['Foul Won', 'Duel'] and not same_team:
            location = next_event.get('location')
            outcome.update({
                'outcome_category': 'Loss',
                'outcome_type': event_type,
                'outcome_team': next_team,
                'outcome_player': next_event.get('player'),
                'same_team': False,
                'time_to_outcome': time_diff,
                'events_to_outcome': i - corner_idx,
                'outcome_location': location
            })
            return outcome

    # No clear outcome - possession retained but no shot/corner
    outcome.update({
        'outcome_category': 'Possession',
        'outcome_type': 'Maintained Possession',
        'same_team': True
    })

    return outcome

def main():
    """Add outcome labels to existing StatsBomb corners"""

    base_dir = Path("/home/mseo/CornerTactics")
    input_file = base_dir / "data" / "datasets" / "statsbomb" / "corners_360.csv"
    output_file = base_dir / "data" / "datasets" / "statsbomb" / "corners_360_with_outcomes.csv"

    print("="*60)
    print("StatsBomb Corner Outcome Labeling")
    print("="*60)

    # Load existing corners
    print(f"\nLoading corners from: {input_file}")
    corners_df = pd.read_csv(input_file)
    print(f"Loaded {len(corners_df)} corners")

    # Get unique matches
    unique_matches = corners_df['match_id'].unique()
    print(f"\nFound {len(unique_matches)} unique matches")

    # Process each match
    labeled_corners = []

    for match_id in tqdm(unique_matches, desc="Processing matches"):
        # Get corners for this match
        match_corners = corners_df[corners_df['match_id'] == match_id].copy()

        # Fetch full events for match
        try:
            events_df = sb.events(match_id=int(match_id))
        except Exception as e:
            print(f"\nWarning: Could not fetch events for match {match_id}: {e}")
            continue

        # Process each corner in this match
        for idx, corner in match_corners.iterrows():
            # Find corner in events dataframe
            # Match on team, player, minute, second
            corner_mask = (
                (events_df['type'] == 'Pass') &
                (events_df['team'] == corner['team']) &
                (events_df['minute'] == corner['minute']) &
                (events_df['second'] == corner['second'])
            )

            matching_events = events_df[corner_mask]

            if len(matching_events) == 0:
                continue

            # Get first matching event index
            corner_idx = matching_events.index[0]

            # Get outcome
            outcome = get_corner_outcome(events_df, corner_idx)

            # Merge with corner data
            corner_dict = corner.to_dict()
            corner_dict.update(outcome)
            labeled_corners.append(corner_dict)

    # Create output dataframe
    output_df = pd.DataFrame(labeled_corners)

    # Save results
    output_df.to_csv(output_file, index=False)

    # Print summary
    print("\n" + "="*60)
    print("Outcome Labeling Complete")
    print("="*60)
    print(f"Labeled corners: {len(output_df)}")
    print(f"Output saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Outcome distribution
    print("\nOutcome Category Distribution:")
    print(output_df['outcome_category'].value_counts())

    print("\nDetailed Outcome Types:")
    print(output_df['outcome_type'].value_counts().head(10))

    # Success metrics
    goals = len(output_df[output_df['goal_scored'] == True])
    shots = len(output_df[output_df['outcome_category'] == 'Shot'])
    shot_or_goal = len(output_df[output_df['outcome_category'].isin(['Goal', 'Shot'])])

    print(f"\nSuccess Metrics:")
    print(f"  Goals: {goals} ({100*goals/len(output_df):.1f}%)")
    print(f"  Shots (inc. goals): {shot_or_goal} ({100*shot_or_goal/len(output_df):.1f}%)")
    print(f"  Goal conversion from corners: {100*goals/len(output_df):.2f}%")
    if shot_or_goal > 0:
        print(f"  Goal conversion from shots: {100*goals/shot_or_goal:.1f}%")

    # Time to outcome stats
    outcome_times = output_df[output_df['time_to_outcome'].notna()]['time_to_outcome']
    if len(outcome_times) > 0:
        print(f"\nTime to Outcome (seconds):")
        print(f"  Mean: {outcome_times.mean():.1f}s")
        print(f"  Median: {outcome_times.median():.1f}s")
        print(f"  Min: {outcome_times.min():.1f}s")
        print(f"  Max: {outcome_times.max():.1f}s")

if __name__ == "__main__":
    main()
