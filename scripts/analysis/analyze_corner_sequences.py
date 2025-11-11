#!/usr/bin/env python3
"""
Analyze corner kick sequences from StatsBomb data to understand action patterns.
This script fetches raw event data to examine what happens after corner kicks.
"""

import sys
sys.path.append('/Users/mahmood/projects/CornerTactics')

from statsbombpy import sb
import pandas as pd
import json
from typing import Dict, List

def fetch_sample_match_events():
    """Fetch events from a sample match with corners."""
    # Get available competitions
    comps = sb.competitions()

    # Filter for men's competitions with likely corner data
    mens_comps = comps[comps['competition_gender'] == 'male']

    # Try Champions League first (usually good data quality)
    ucl = mens_comps[mens_comps['competition_name'] == 'Champions League']

    if not ucl.empty:
        comp_id = ucl.iloc[0]['competition_id']
        season_id = ucl.iloc[0]['season_id']

        # Get matches
        matches = sb.matches(competition_id=comp_id, season_id=season_id)

        # Get events from first match
        if not matches.empty:
            match_id = matches.iloc[0]['match_id']
            events = sb.events(match_id=match_id)
            return events, matches.iloc[0]

    return None, None

def analyze_corner_sequences(events_df: pd.DataFrame):
    """Analyze what happens after corner kicks."""

    # Find corner kicks
    corners = events_df[
        (events_df['type'] == 'Pass') &
        (events_df['pass_type'].notna()) &
        (events_df['pass_type'].str.contains('Corner', case=False, na=False))
    ]

    print(f"Found {len(corners)} corner kicks in this match")

    sequences = []

    for idx, corner in corners.iterrows():
        # Get corner index in full dataframe
        corner_idx = events_df.index.get_loc(idx)
        corner_team = corner['team']

        print(f"\n{'='*60}")
        print(f"CORNER at {corner['minute']}:{corner['second']} by {corner['team']}")
        print(f"Player: {corner['player']}")
        print(f"Location: {corner.get('location', 'N/A')}")
        print(f"End Location: {corner.get('pass_end_location', 'N/A')}")

        # Get next 10 events
        next_events = events_df.iloc[corner_idx+1:corner_idx+11]

        print(f"\nNext 10 events after corner:")
        print(f"{'#':<3} {'Type':<20} {'Team':<25} {'Same Team':<10} {'Details':<30}")
        print("-"*90)

        event_sequence = []

        for i, (_, event) in enumerate(next_events.iterrows(), 1):
            event_type = event['type']
            event_team = event.get('team', 'N/A')
            same_team = "Yes" if event_team == corner_team else "No"

            # Get additional details based on event type
            details = ""
            if event_type == 'Shot':
                shot_outcome = event.get('shot_outcome', 'N/A')
                details = f"Outcome: {shot_outcome}"
            elif event_type == 'Pass':
                pass_outcome = event.get('pass_outcome', 'Complete')
                details = f"Pass: {pass_outcome}"
            elif event_type == 'Clearance':
                details = "Ball cleared"
            elif event_type == 'Duel':
                duel_type = event.get('duel_type', 'N/A')
                details = f"Duel: {duel_type}"
            elif event_type == 'Ball Recovery':
                details = "Ball recovered"
            elif event_type == 'Foul Committed' or event_type == 'Foul Won':
                details = f"Foul"

            print(f"{i:<3} {event_type:<20} {event_team:<25} {same_team:<10} {details:<30}")

            event_sequence.append({
                'position': i,
                'type': event_type,
                'team': event_team,
                'same_team': same_team == "Yes",
                'details': details
            })

            # Stop if we find a clear outcome
            if event_type in ['Shot', 'Goal', 'Clearance'] and i <= 5:
                print(f"\n>>> Primary outcome: {event_type} by {'attacking' if same_team == 'Yes' else 'defending'} team")
                break

        sequences.append({
            'corner': corner.to_dict(),
            'sequence': event_sequence
        })

    return sequences

def summarize_patterns(sequences: List[Dict]):
    """Summarize patterns from corner sequences."""

    print(f"\n{'='*60}")
    print("PATTERN SUMMARY")
    print(f"{'='*60}")

    # Count first events
    first_events = {}
    second_events = {}
    outcomes = {}

    for seq in sequences:
        events = seq['sequence']

        if len(events) > 0:
            first = events[0]['type']
            first_events[first] = first_events.get(first, 0) + 1

        if len(events) > 1:
            second = events[1]['type']
            second_events[second] = second_events.get(second, 0) + 1

        # Find outcome (Shot, Clearance, or position 5+)
        for i, event in enumerate(events[:5]):
            if event['type'] in ['Shot', 'Clearance']:
                outcome = f"{event['type']} ({'same' if event['same_team'] else 'opp'})"
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                break
        else:
            outcomes['Possession'] = outcomes.get('Possession', 0) + 1

    print("\nFirst Event After Corner:")
    for event, count in sorted(first_events.items(), key=lambda x: x[1], reverse=True):
        print(f"  {event}: {count}")

    print("\nSecond Event After Corner:")
    for event, count in sorted(second_events.items(), key=lambda x: x[1], reverse=True):
        print(f"  {event}: {count}")

    print("\nOutcomes (within 5 events):")
    for outcome, count in sorted(outcomes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {outcome}: {count}")

if __name__ == "__main__":
    print("Fetching sample match data from StatsBomb...")

    events, match_info = fetch_sample_match_events()

    if events is not None:
        print(f"\nMatch: {match_info.get('home_team', 'N/A')} vs {match_info.get('away_team', 'N/A')}")
        print(f"Competition: {match_info.get('competition', 'N/A')}")

        sequences = analyze_corner_sequences(events)

        if sequences:
            summarize_patterns(sequences)
        else:
            print("No corner kicks found in this match")
    else:
        print("Could not fetch match data")