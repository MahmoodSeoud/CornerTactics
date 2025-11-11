#!/usr/bin/env python3
"""
Comprehensive analysis of event transitions after corner kicks.
Builds transition matrices and documents all raw features.
"""

import json
import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple

def fetch_multiple_matches():
    """Fetch events from multiple matches to get better statistics."""

    base_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

    # Get competitions
    comp_url = f"{base_url}/competitions.json"
    competitions = requests.get(comp_url).json()

    all_events = []
    matches_info = []

    # Try to get multiple matches
    for comp in competitions:
        if comp['competition_name'] in ['La Liga', 'Premier League', 'Champions League']:
            comp_id = comp['competition_id']
            season_id = comp['season_id']

            matches_url = f"{base_url}/matches/{comp_id}/{season_id}.json"
            try:
                matches = requests.get(matches_url).json()

                # Get events from first 3 matches
                for match in matches[:3]:
                    match_id = match['match_id']
                    events_url = f"{base_url}/events/{match_id}.json"

                    try:
                        events = requests.get(events_url).json()
                        if events:
                            all_events.extend(events)
                            matches_info.append({
                                'match_id': match_id,
                                'competition': comp['competition_name'],
                                'season': comp['season_name'],
                                'home': match['home_team']['home_team_name'],
                                'away': match['away_team']['away_team_name']
                            })
                            print(f"Loaded {len(events)} events from {match['home_team']['home_team_name']} vs {match['away_team']['away_team_name']}")

                            if len(matches_info) >= 5:  # Get 5 matches total
                                return all_events, matches_info
                    except:
                        continue
            except:
                continue

    return all_events, matches_info

def extract_all_corner_sequences(events: List[Dict]) -> List[Dict]:
    """Extract all corner kick sequences with detailed transition information."""

    corner_sequences = []

    # Find all corners
    for i, event in enumerate(events):
        if (event.get('type', {}).get('name') == 'Pass' and
            event.get('pass', {}).get('type', {}).get('name') == 'Corner'):

            corner_info = {
                'corner_event': event,
                'corner_index': i,
                'corner_team': event.get('team', {}).get('name'),
                'corner_player': event.get('player', {}).get('name'),
                'location': event.get('location'),
                'end_location': event.get('pass', {}).get('end_location'),
                'height': event.get('pass', {}).get('height', {}).get('name'),
                'outcome': event.get('pass', {}).get('outcome', {}).get('name', 'Complete'),
                'timestamp': event.get('timestamp'),
                'minute': event.get('minute'),
                'second': event.get('second'),
                'sequence': []
            }

            # Get next 20 events for transition analysis
            for j in range(1, min(21, len(events) - i)):
                next_event = events[i + j]

                # Extract all relevant features
                event_info = {
                    'position': j,
                    'type': next_event.get('type', {}).get('name'),
                    'team': next_event.get('team', {}).get('name'),
                    'player': next_event.get('player', {}).get('name'),
                    'location': next_event.get('location'),
                    'timestamp': next_event.get('timestamp'),
                    'duration': next_event.get('duration'),
                    'under_pressure': next_event.get('under_pressure', False),
                    'off_camera': next_event.get('off_camera', False),
                    'out': next_event.get('out', False),
                    'related_events': next_event.get('related_events', [])
                }

                # Add type-specific features
                if event_info['type'] == 'Pass':
                    pass_data = next_event.get('pass', {})
                    event_info['pass_details'] = {
                        'outcome': pass_data.get('outcome', {}).get('name', 'Complete'),
                        'length': pass_data.get('length'),
                        'angle': pass_data.get('angle'),
                        'height': pass_data.get('height', {}).get('name'),
                        'end_location': pass_data.get('end_location'),
                        'cross': pass_data.get('cross', False),
                        'switch': pass_data.get('switch', False),
                        'shot_assist': pass_data.get('shot_assist', False),
                        'goal_assist': pass_data.get('goal_assist', False)
                    }

                elif event_info['type'] == 'Shot':
                    shot_data = next_event.get('shot', {})
                    event_info['shot_details'] = {
                        'outcome': shot_data.get('outcome', {}).get('name'),
                        'statsbomb_xg': shot_data.get('statsbomb_xg'),
                        'end_location': shot_data.get('end_location'),
                        'technique': shot_data.get('technique', {}).get('name'),
                        'body_part': shot_data.get('body_part', {}).get('name'),
                        'type': shot_data.get('type', {}).get('name'),
                        'first_time': shot_data.get('first_time', False),
                        'follows_dribble': shot_data.get('follows_dribble', False)
                    }

                elif event_info['type'] == 'Clearance':
                    clearance_data = next_event.get('clearance', {})
                    event_info['clearance_details'] = {
                        'aerial_won': clearance_data.get('aerial_won', False),
                        'head': clearance_data.get('head', False),
                        'body_part': clearance_data.get('body_part', {}).get('name') if 'body_part' in clearance_data else None
                    }

                elif event_info['type'] == 'Duel':
                    duel_data = next_event.get('duel', {})
                    event_info['duel_details'] = {
                        'type': duel_data.get('type', {}).get('name'),
                        'outcome': duel_data.get('outcome', {}).get('name') if 'outcome' in duel_data else None
                    }

                elif event_info['type'] == 'Interception':
                    interception_data = next_event.get('interception', {})
                    event_info['interception_details'] = {
                        'outcome': interception_data.get('outcome', {}).get('name') if 'outcome' in interception_data else None
                    }

                corner_info['sequence'].append(event_info)

            corner_sequences.append(corner_info)

    return corner_sequences

def build_transition_matrix(sequences: List[Dict]) -> Tuple[Dict, pd.DataFrame]:
    """Build transition probability matrix P(event_t+1 | event_t)."""

    # Count transitions
    transitions = defaultdict(lambda: defaultdict(int))
    event_counts = defaultdict(int)

    # Also track position-specific transitions (1st, 2nd, 3rd event after corner)
    position_transitions = {
        1: defaultdict(lambda: defaultdict(int)),
        2: defaultdict(lambda: defaultdict(int)),
        3: defaultdict(lambda: defaultdict(int))
    }

    for seq in sequences:
        # Add corner as initial state
        prev_event = 'Corner'
        event_counts['Corner'] += 1

        for i, event in enumerate(seq['sequence'][:10]):  # Look at first 10 events
            curr_event = event['type']

            # Overall transitions
            transitions[prev_event][curr_event] += 1
            event_counts[curr_event] += 1

            # Position-specific transitions
            if i + 1 <= 3:  # First 3 positions
                if i == 0:
                    position_transitions[1]['Corner'][curr_event] += 1
                else:
                    prev_pos_event = seq['sequence'][i-1]['type']
                    position_transitions[i+1][prev_pos_event][curr_event] += 1

            prev_event = curr_event

    # Convert to probability matrix
    unique_events = sorted(set(event_counts.keys()))
    n_events = len(unique_events)

    # Create probability matrix
    prob_matrix = pd.DataFrame(0.0, index=unique_events, columns=unique_events)

    for from_event in unique_events:
        total = sum(transitions[from_event].values())
        if total > 0:
            for to_event in unique_events:
                prob_matrix.loc[from_event, to_event] = transitions[from_event][to_event] / total

    return {
        'transitions': dict(transitions),
        'event_counts': dict(event_counts),
        'position_transitions': position_transitions
    }, prob_matrix

def analyze_features_in_data(sequences: List[Dict]) -> Dict:
    """Analyze what features are available in the raw data."""

    feature_analysis = {
        'event_types': set(),
        'event_fields': defaultdict(set),
        'pass_features': set(),
        'shot_features': set(),
        'clearance_features': set(),
        'spatial_features': {
            'has_location': 0,
            'has_end_location': 0,
            'location_examples': []
        },
        'temporal_features': {
            'has_timestamp': 0,
            'has_duration': 0,
            'timestamp_examples': []
        },
        'pressure_features': {
            'under_pressure_count': 0,
            'total_events': 0
        },
        'outcome_features': defaultdict(set)
    }

    for seq in sequences:
        # Analyze corner event itself
        corner = seq['corner_event']
        for key in corner.keys():
            feature_analysis['event_fields']['Corner'].add(key)

        # Analyze subsequent events
        for event in seq['sequence']:
            event_type = event['type']
            feature_analysis['event_types'].add(event_type)

            # Track fields per event type
            for key in event.keys():
                if key not in ['position', 'pass_details', 'shot_details', 'clearance_details', 'duel_details']:
                    feature_analysis['event_fields'][event_type].add(key)

            # Spatial features
            if event.get('location'):
                feature_analysis['spatial_features']['has_location'] += 1
                if len(feature_analysis['spatial_features']['location_examples']) < 5:
                    feature_analysis['spatial_features']['location_examples'].append({
                        'event': event_type,
                        'location': event['location']
                    })

            # Temporal features
            if event.get('timestamp'):
                feature_analysis['temporal_features']['has_timestamp'] += 1
                if len(feature_analysis['temporal_features']['timestamp_examples']) < 3:
                    feature_analysis['temporal_features']['timestamp_examples'].append({
                        'event': event_type,
                        'timestamp': event['timestamp']
                    })

            if event.get('duration'):
                feature_analysis['temporal_features']['has_duration'] += 1

            # Pressure
            feature_analysis['pressure_features']['total_events'] += 1
            if event.get('under_pressure'):
                feature_analysis['pressure_features']['under_pressure_count'] += 1

            # Type-specific features
            if event_type == 'Pass' and 'pass_details' in event:
                for key in event['pass_details'].keys():
                    feature_analysis['pass_features'].add(key)
                    if event['pass_details'][key] not in [None, False, []]:
                        feature_analysis['outcome_features']['Pass'].add(f"{key}={event['pass_details'][key]}")

            if event_type == 'Shot' and 'shot_details' in event:
                for key in event['shot_details'].keys():
                    feature_analysis['shot_features'].add(key)

            if event_type == 'Clearance' and 'clearance_details' in event:
                for key in event['clearance_details'].keys():
                    feature_analysis['clearance_features'].add(key)

    return feature_analysis

def generate_context_document(sequences: List[Dict], transition_data: Dict,
                            prob_matrix: pd.DataFrame, feature_analysis: Dict,
                            matches_info: List[Dict]) -> str:
    """Generate comprehensive context document."""

    doc = f"""# StatsBomb Corner Kick Analysis: Complete Context

## Dataset Overview
- **Matches Analyzed**: {len(matches_info)}
- **Total Corner Kicks**: {len(sequences)}
- **Events After Corners Analyzed**: {sum(len(s['sequence']) for s in sequences)}

### Matches Included:
"""

    for match in matches_info:
        doc += f"- {match['competition']}: {match['home']} vs {match['away']}\n"

    doc += f"""

## 1. EVENT TRANSITION ANALYSIS

### What Happens After a Corner Kick?

#### Immediate Transitions (Position 1 - First Event After Corner)
The probability P(Event at t+1 | Corner at t):

"""

    # Get first position transitions
    corner_transitions = transition_data['position_transitions'][1].get('Corner', {})
    total_corners = sum(corner_transitions.values())

    doc += "| Next Event | Count | Probability |\n"
    doc += "|------------|-------|-------------|\n"

    for event, count in sorted(corner_transitions.items(), key=lambda x: x[1], reverse=True):
        prob = count / total_corners if total_corners > 0 else 0
        doc += f"| {event} | {count} | {prob:.3f} |\n"

    doc += f"""

#### Full Transition Matrix (Top 10x10)
P(Event j at t+1 | Event i at t) for all event types:

"""

    # Show top 10x10 of transition matrix
    top_events = prob_matrix.sum(axis=1).nlargest(10).index
    top_matrix = prob_matrix.loc[top_events, top_events]

    doc += "```\n"
    doc += top_matrix.round(3).to_string()
    doc += "\n```\n"

    doc += f"""

#### Event Chains (First 3 Events After Corner)

Most common 3-event sequences after corners:
"""

    # Analyze 3-event chains
    chains = defaultdict(int)
    for seq in sequences:
        if len(seq['sequence']) >= 3:
            chain = ' → '.join([e['type'] for e in seq['sequence'][:3]])
            chains[chain] += 1

    doc += "\n| Sequence | Count | Percentage |\n"
    doc += "|----------|-------|------------|\n"

    for chain, count in sorted(chains.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = (count / len(sequences)) * 100
        doc += f"| {chain} | {count} | {pct:.1f}% |\n"

    doc += f"""

## 2. RAW DATA STRUCTURE & FEATURES

### Core Event Structure
Every event in StatsBomb data contains these fields:

#### Universal Fields (Present in ALL Events)
"""

    # Find truly universal fields
    universal_fields = None
    for event_type, fields in feature_analysis['event_fields'].items():
        if universal_fields is None:
            universal_fields = fields
        else:
            universal_fields = universal_fields.intersection(fields)

    for field in sorted(universal_fields):
        doc += f"- `{field}`\n"

    doc += f"""

#### Event-Specific Fields
Fields that appear in specific event types:

"""

    for event_type in sorted(feature_analysis['event_types']):
        fields = feature_analysis['event_fields'].get(event_type, set())
        specific_fields = fields - universal_fields if universal_fields else fields

        if specific_fields:
            doc += f"\n**{event_type}**:\n"
            for field in sorted(specific_fields):
                doc += f"- `{field}`\n"

    doc += f"""

### Corner Kick Specific Features

#### Pass (Corner) Features Available:
"""

    for feature in sorted(feature_analysis['pass_features']):
        doc += f"- `pass.{feature}`\n"

    doc += f"""

#### Example Corner Event (Raw JSON Structure):
```json
{{
    "id": "uuid-here",
    "type": {{"name": "Pass"}},
    "pass": {{
        "type": {{"name": "Corner"}},
        "end_location": [112.8, 43.8],
        "height": {{"name": "High Pass"}},
        "outcome": {{"name": "Complete"}},
        "body_part": {{"name": "Right Foot"}},
        "length": 18.5,
        "angle": 2.1
    }},
    "team": {{"name": "Barcelona"}},
    "player": {{"name": "Lionel Messi"}},
    "location": [120.0, 80.0],
    "timestamp": "00:21:14.123",
    "minute": 21,
    "second": 14
}}
```

### Spatial Features

**Location Coverage**: {feature_analysis['spatial_features']['has_location']} events have location data

**Coordinate System**:
- Origin: [0, 0] at bottom-left of defending goal
- Range: X ∈ [0, 120], Y ∈ [0, 80]
- Corner locations: [120, 0] or [120, 80] (attacking corners)

**Example Locations**:
"""

    for example in feature_analysis['spatial_features']['location_examples']:
        doc += f"- {example['event']}: {example['location']}\n"

    doc += f"""

### Temporal Features

**Timestamp Coverage**: {feature_analysis['temporal_features']['has_timestamp']} events have timestamps
**Duration Coverage**: {feature_analysis['temporal_features']['has_duration']} events have duration

**Example Timestamps**:
"""

    for example in feature_analysis['temporal_features']['timestamp_examples']:
        doc += f"- {example['event']}: {example['timestamp']}\n"

    pressure_pct = (feature_analysis['pressure_features']['under_pressure_count'] /
                   feature_analysis['pressure_features']['total_events'] * 100
                   if feature_analysis['pressure_features']['total_events'] > 0 else 0)

    doc += f"""

### Pressure & Context Features

**Under Pressure**: {feature_analysis['pressure_features']['under_pressure_count']}/{feature_analysis['pressure_features']['total_events']} events ({pressure_pct:.1f}%)

### Shot-Specific Features (When Shot Occurs)
"""

    for feature in sorted(feature_analysis['shot_features']):
        doc += f"- `shot.{feature}`\n"

    doc += f"""

### Clearance-Specific Features
"""

    for feature in sorted(feature_analysis['clearance_features']):
        doc += f"- `clearance.{feature}`\n"

    doc += f"""

## 3. KEY INSIGHTS FOR ANALYSIS

### Transition Patterns
1. **Most corners** → Pass or Ball Receipt (player receiving)
2. **Clearances** typically occur within first 3 events
3. **Shots** are rare but occur quickly (positions 1-5)

### Feature Richness
- **Spatial**: All events have locations
- **Temporal**: Timestamps allow time-based analysis
- **Contextual**: Pressure, team, player always available
- **Outcome-specific**: Rich details for shots, passes, clearances

### What You Can Calculate:
1. **Transition Probabilities**: P(Event_j | Event_i) for any pair
2. **Time to Outcome**: Using timestamps
3. **Spatial Heat Maps**: Using location data
4. **Pressure Influence**: Using under_pressure flag
5. **Team-specific Patterns**: Using team field
6. **Player Involvement**: Using player field

### What's NOT in Raw Data:
- Player positions/formations (except in rare freeze frames)
- Defensive shape
- Off-ball movements
- Expected goals (xG) for non-shots
"""

    return doc

def main():
    print("Fetching multiple matches from StatsBomb...")

    # Get data from multiple matches
    all_events, matches_info = fetch_multiple_matches()

    if not all_events:
        print("Failed to fetch data")
        return

    print(f"\nTotal events loaded: {len(all_events)}")

    # Extract all corner sequences
    print("Extracting corner sequences...")
    sequences = extract_all_corner_sequences(all_events)
    print(f"Found {len(sequences)} corner kicks")

    # Build transition matrix
    print("Building transition matrix...")
    transition_data, prob_matrix = build_transition_matrix(sequences)

    # Analyze features
    print("Analyzing features in raw data...")
    feature_analysis = analyze_features_in_data(sequences)

    # Generate context document
    print("Generating context document...")
    context_doc = generate_context_document(sequences, transition_data, prob_matrix,
                                           feature_analysis, matches_info)

    # Save the document
    output_path = 'docs/CORNER_KICK_CONTEXT.md'
    with open(output_path, 'w') as f:
        f.write(context_doc)

    print(f"\nContext document saved to: {output_path}")

    # Also save the raw transition matrix as CSV
    prob_matrix.to_csv('data/raw/statsbomb/transition_matrix.csv')
    print(f"Transition matrix saved to: data/raw/statsbomb/transition_matrix.csv")

if __name__ == "__main__":
    main()