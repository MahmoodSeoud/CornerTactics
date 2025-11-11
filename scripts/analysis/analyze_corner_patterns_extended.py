#!/usr/bin/env python3
"""
Extended analysis of corner kick patterns from StatsBomb data.
Searches for matches with diverse corner outcomes including shots and clearances.
"""

import sys
sys.path.append('/Users/mahmood/projects/CornerTactics')

from statsbombpy import sb
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

def find_match_with_corner_outcomes():
    """Find a match with diverse corner outcomes (shots, clearances, etc.)."""
    comps = sb.competitions()
    mens_comps = comps[comps['competition_gender'] == 'male']

    # Try multiple competitions
    target_comps = ['Champions League', 'La Liga', 'Premier League']

    for comp_name in target_comps:
        comp_data = mens_comps[mens_comps['competition_name'] == comp_name]

        if not comp_data.empty:
            comp_id = comp_data.iloc[0]['competition_id']
            season_id = comp_data.iloc[0]['season_id']

            matches = sb.matches(competition_id=comp_id, season_id=season_id)

            # Try first few matches
            for _, match in matches.head(5).iterrows():
                match_id = match['match_id']
                events = sb.events(match_id=match_id)

                # Check if match has corners with varied outcomes
                corners = events[
                    (events['type'] == 'Pass') &
                    (events['pass_type'].notna()) &
                    (events['pass_type'].str.contains('Corner', case=False, na=False))
                ]

                if len(corners) >= 5:
                    # Check for shots and clearances
                    has_shots = (events['type'] == 'Shot').any()
                    has_clearances = (events['type'] == 'Clearance').any()

                    if has_shots and has_clearances:
                        return events, match, corners

    return None, None, None

def analyze_detailed_sequences(events_df: pd.DataFrame, corners_df: pd.DataFrame):
    """Analyze corner sequences with detailed event tracking."""

    all_event_types = set()
    all_outcomes = []
    detailed_sequences = []

    print(f"\nAnalyzing {len(corners_df)} corners...\n")

    for idx, corner in corners_df.iterrows():
        corner_idx = events_df.index.get_loc(idx)
        corner_team = corner['team']
        corner_minute = corner['minute']
        corner_second = corner['second']

        # Get next 20 events to find outcomes
        next_events = events_df.iloc[corner_idx+1:corner_idx+21]

        sequence_events = []
        outcome_found = False

        for i, (_, event) in enumerate(next_events.iterrows(), 1):
            event_type = event['type']
            event_team = event.get('team', 'N/A')
            same_team = event_team == corner_team

            all_event_types.add(event_type)

            # Track all event types
            event_detail = {
                'position': i,
                'type': event_type,
                'team': event_team,
                'same_team': same_team,
                'minute': event.get('minute'),
                'second': event.get('second')
            }

            # Add specific details
            if event_type == 'Shot':
                event_detail['shot_outcome'] = event.get('shot_outcome', 'Unknown')
                outcome_found = True
            elif event_type == 'Clearance':
                outcome_found = True
            elif event_type == 'Duel':
                event_detail['duel_type'] = event.get('duel_type', 'Unknown')
            elif event_type == 'Interception':
                outcome_found = True
            elif event_type == 'Foul Committed' or event_type == 'Foul Won':
                outcome_found = True

            sequence_events.append(event_detail)

            # Stop after finding clear outcome within 5 events
            if outcome_found and i <= 5:
                outcome_type = event_type
                outcome_team_type = 'attacking' if same_team else 'defending'

                all_outcomes.append({
                    'outcome': outcome_type,
                    'team_type': outcome_team_type,
                    'events_to_outcome': i
                })
                break

        if not outcome_found:
            all_outcomes.append({
                'outcome': 'Possession',
                'team_type': 'attacking',
                'events_to_outcome': None
            })

        detailed_sequences.append({
            'corner_time': f"{corner_minute}:{corner_second:02d}",
            'corner_team': corner_team,
            'sequence': sequence_events[:10]  # Keep first 10 events
        })

    return all_event_types, all_outcomes, detailed_sequences

def print_detailed_analysis(event_types, outcomes, sequences):
    """Print comprehensive analysis of corner patterns."""

    print("="*70)
    print("COMPREHENSIVE CORNER KICK ANALYSIS")
    print("="*70)

    # Event types found
    print("\n1. ALL EVENT TYPES FOLLOWING CORNERS:")
    print("-"*40)
    for event_type in sorted(event_types):
        print(f"  - {event_type}")

    # Outcome distribution
    print("\n2. OUTCOME DISTRIBUTION (within 5 events):")
    print("-"*40)
    outcome_counts = {}
    for outcome in outcomes:
        key = f"{outcome['outcome']} ({outcome['team_type']})"
        outcome_counts[key] = outcome_counts.get(key, 0) + 1

    for outcome, count in sorted(outcome_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {outcome}: {count}")

    # Show detailed examples
    print("\n3. DETAILED CORNER SEQUENCES:")
    print("-"*40)

    # Find examples of each major outcome
    shot_example = None
    clearance_example = None
    possession_example = None

    for i, (outcome, seq) in enumerate(zip(outcomes, sequences)):
        if outcome['outcome'] == 'Shot' and not shot_example:
            shot_example = (outcome, seq)
        elif outcome['outcome'] == 'Clearance' and not clearance_example:
            clearance_example = (outcome, seq)
        elif outcome['outcome'] == 'Possession' and not possession_example:
            possession_example = (outcome, seq)

    # Print examples
    for example_name, example_data in [
        ("SHOT EXAMPLE", shot_example),
        ("CLEARANCE EXAMPLE", clearance_example),
        ("POSSESSION EXAMPLE", possession_example)
    ]:
        if example_data:
            outcome, seq = example_data
            print(f"\n{example_name}:")
            print(f"Corner at {seq['corner_time']} by {seq['corner_team']}")
            print(f"Outcome: {outcome['outcome']} by {outcome['team_type']} team")

            print("\nEvent sequence:")
            for event in seq['sequence'][:5]:
                team_indicator = "↑" if event['same_team'] else "↓"
                print(f"  {event['position']:2}. {team_indicator} {event['type']:<15} ({event['team'][:20]})")

                # Add details
                if 'shot_outcome' in event:
                    print(f"      → Shot result: {event['shot_outcome']}")
                if 'duel_type' in event:
                    print(f"      → Duel type: {event['duel_type']}")

    # Event transition analysis
    print("\n4. COMMON EVENT TRANSITIONS:")
    print("-"*40)
    transitions = {}

    for seq in sequences:
        events = seq['sequence']
        for i in range(len(events) - 1):
            transition = f"{events[i]['type']} → {events[i+1]['type']}"
            transitions[transition] = transitions.get(transition, 0) + 1

    # Show top transitions
    top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
    for transition, count in top_transitions:
        print(f"  {transition}: {count}")

if __name__ == "__main__":
    print("Searching for match with diverse corner outcomes...")

    events, match_info, corners = find_match_with_corner_outcomes()

    if events is not None:
        print(f"\nMatch: {match_info.get('home_team', 'N/A')} vs {match_info.get('away_team', 'N/A')}")
        print(f"Competition: {match_info.get('competition', 'N/A')}")
        print(f"Season: {match_info.get('season', 'N/A')}")

        event_types, outcomes, sequences = analyze_detailed_sequences(events, corners)
        print_detailed_analysis(event_types, outcomes, sequences)
    else:
        print("Could not find a suitable match with varied outcomes")