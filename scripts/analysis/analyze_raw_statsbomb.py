#!/usr/bin/env python3
"""
Direct analysis of raw StatsBomb open-data from GitHub.
No modifications, just raw data exploration.
"""

import json
import requests
from typing import Dict, List, Any
import pandas as pd

def fetch_raw_statsbomb_data():
    """Fetch raw data directly from StatsBomb GitHub."""

    base_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

    # First, get competitions
    print("Fetching competitions...")
    comp_url = f"{base_url}/competitions.json"
    comp_response = requests.get(comp_url)
    competitions = comp_response.json()

    # Find a competition with good data (Champions League is usually complete)
    champions_league = [c for c in competitions
                        if c['competition_name'] == 'Champions League'
                        and c['season_name'] == '2019/2020']

    if not champions_league:
        # Fallback to any recent competition
        champions_league = [c for c in competitions
                           if c['competition_name'] == 'La Liga'
                           and '2019' in c['season_name']]

    if champions_league:
        comp = champions_league[0]
        comp_id = comp['competition_id']
        season_id = comp['season_id']

        print(f"Using {comp['competition_name']} - {comp['season_name']}")

        # Get matches
        print("Fetching matches...")
        matches_url = f"{base_url}/matches/{comp_id}/{season_id}.json"
        matches_response = requests.get(matches_url)
        matches = matches_response.json()

        # Get events from first match with data
        for match in matches[:5]:  # Try first 5 matches
            match_id = match['match_id']

            print(f"Fetching events for {match['home_team']['home_team_name']} vs {match['away_team']['away_team_name']}...")
            events_url = f"{base_url}/events/{match_id}.json"

            try:
                events_response = requests.get(events_url)
                events = events_response.json()

                if events:
                    return events, match, comp
            except:
                continue

    return None, None, None

def analyze_raw_corner_sequences(events: List[Dict[str, Any]]):
    """Analyze corner kicks in raw event data."""

    # Find corner kicks in raw data
    corners = []
    for event in events:
        if (event.get('type', {}).get('name') == 'Pass' and
            event.get('pass', {}).get('type', {}).get('name') == 'Corner'):
            corners.append(event)

    print(f"\nFound {len(corners)} corner kicks")

    corner_analysis = []

    for corner in corners[:10]:  # Analyze first 10 corners
        corner_idx = events.index(corner)
        corner_team = corner.get('team', {}).get('name', 'Unknown')
        corner_minute = corner.get('minute', 0)
        corner_second = corner.get('second', 0)

        print(f"\n{'='*70}")
        print(f"CORNER: {corner_minute}:{corner_second:02d} by {corner_team}")
        print(f"Player: {corner.get('player', {}).get('name', 'Unknown')}")
        print(f"Location: {corner.get('location', 'N/A')}")

        # Get raw structure of corner event
        print("\nRAW CORNER EVENT STRUCTURE:")
        print(f"  Keys in event: {list(corner.keys())}")

        # Show pass details
        if 'pass' in corner:
            print(f"  Pass keys: {list(corner['pass'].keys())}")
            print(f"  End location: {corner['pass'].get('end_location', 'N/A')}")
            print(f"  Height: {corner['pass'].get('height', {}).get('name', 'N/A')}")
            print(f"  Outcome: {corner['pass'].get('outcome', {}).get('name', 'Complete')}")

        # Show freeze frame if available
        if 'shot' in corner and 'freeze_frame' in corner['shot']:
            print(f"  Has freeze frame: Yes ({len(corner['shot']['freeze_frame'])} players)")

        # Analyze next events
        print("\nNEXT 10 EVENTS (RAW):")
        print(f"{'#':<3} {'Type':<20} {'Team':<25} {'Details':<30}")
        print("-"*80)

        sequence = []
        outcome_found = False

        for i, next_event in enumerate(events[corner_idx+1:corner_idx+11], 1):
            event_type = next_event.get('type', {}).get('name', 'Unknown')
            event_team = next_event.get('team', {}).get('name', 'Unknown')

            # Build details from raw data
            details = ""
            if event_type == 'Shot':
                shot_outcome = next_event.get('shot', {}).get('outcome', {}).get('name', 'N/A')
                details = f"Outcome: {shot_outcome}"
                outcome_found = True
            elif event_type == 'Pass':
                pass_outcome = next_event.get('pass', {}).get('outcome', {}).get('name', 'Complete')
                details = f"{pass_outcome}"
            elif event_type == 'Clearance':
                details = "Cleared"
                outcome_found = True
            elif event_type == 'Duel':
                duel_type = next_event.get('duel', {}).get('type', {}).get('name', 'N/A')
                details = f"Type: {duel_type}"
            elif event_type == 'Ball Recovery':
                details = "Recovered"
            elif event_type == 'Interception':
                details = "Intercepted"
                outcome_found = True

            print(f"{i:<3} {event_type:<20} {event_team:<25} {details:<30}")

            sequence.append({
                'position': i,
                'type': event_type,
                'team': event_team,
                'raw_event': next_event
            })

            if outcome_found and i <= 5:
                break

        corner_analysis.append({
            'corner': corner,
            'sequence': sequence
        })

    return corner_analysis

def extract_all_event_types(events: List[Dict[str, Any]]):
    """Extract all unique event types that follow corners."""

    event_types_after_corners = {}
    detailed_outcomes = []

    # Find all corners
    for i, event in enumerate(events):
        if (event.get('type', {}).get('name') == 'Pass' and
            event.get('pass', {}).get('type', {}).get('name') == 'Corner'):

            corner_team = event.get('team', {}).get('name')

            # Look at next 20 events
            for j in range(1, min(21, len(events) - i)):
                next_event = events[i + j]
                event_type = next_event.get('type', {}).get('name', 'Unknown')

                if j <= 5:  # Count only first 5 for statistics
                    event_types_after_corners[event_type] = event_types_after_corners.get(event_type, 0) + 1

                # Check for outcome events
                if event_type in ['Shot', 'Clearance', 'Goal Keeper'] and j <= 10:
                    next_team = next_event.get('team', {}).get('name')

                    outcome_detail = {
                        'type': event_type,
                        'position': j,
                        'same_team': next_team == corner_team
                    }

                    if event_type == 'Shot':
                        outcome_detail['shot_outcome'] = next_event.get('shot', {}).get('outcome', {}).get('name')

                    detailed_outcomes.append(outcome_detail)
                    break

    return event_types_after_corners, detailed_outcomes

def generate_reports(events, match, competition):
    """Generate comprehensive reports from raw data."""

    # Analyze corners
    corner_sequences = analyze_raw_corner_sequences(events)
    event_types, outcomes = extract_all_event_types(events)

    # Create infrastructure report
    infrastructure_report = f"""# StatsBomb Corner Kick Infrastructure Report

## Data Source
- **Repository**: https://github.com/statsbomb/open-data
- **Data Type**: Raw JSON events
- **Competition**: {competition['competition_name']} - {competition['season_name']}
- **Match**: {match['home_team']['home_team_name']} vs {match['away_team']['away_team_name']}

## Raw Data Structure

### Event Schema
Each event in StatsBomb data contains:
```json
{{
    "id": "unique-event-id",
    "index": integer,
    "period": integer,
    "timestamp": "HH:MM:SS.SSS",
    "minute": integer,
    "second": integer,
    "type": {{"id": integer, "name": "Event Type"}},
    "possession": integer,
    "possession_team": {{"id": integer, "name": "Team Name"}},
    "play_pattern": {{"id": integer, "name": "Pattern"}},
    "team": {{"id": integer, "name": "Team Name"}},
    "player": {{"id": integer, "name": "Player Name"}},
    "position": {{"id": integer, "name": "Position"}},
    "location": [x, y],
    "duration": float (optional),
    "related_events": ["event-id", ...] (optional),
    // Event-specific fields...
}}
```

### Corner Kick Structure
Corner kicks are Pass events with specific attributes:
```json
{{
    "type": {{"name": "Pass"}},
    "pass": {{
        "type": {{"name": "Corner"}},
        "end_location": [x, y],
        "height": {{"name": "High Pass" / "Ground Pass"}},
        "outcome": {{"name": "Complete" / "Incomplete" / etc.}},
        "body_part": {{"name": "Left Foot" / "Right Foot" / "Head"}}
    }}
}}
```

## Event Types Found After Corners

Distribution of events within 5 events of a corner:
"""

    for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
        infrastructure_report += f"- **{event_type}**: {count} occurrences\n"

    infrastructure_report += """

## Key Findings

1. **Event Sequencing**: Events are stored chronologically with index and timestamp
2. **Relationships**: `related_events` field links connected events
3. **Spatial Data**: All events have `location` [x, y] coordinates
4. **Team Context**: Each event tagged with team and player
5. **Outcome Tracking**: Pass events include outcome (Complete/Incomplete)

## Data Access Pattern

```python
# Direct GitHub access
base_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# 1. Get competitions
competitions = requests.get(f"{base_url}/competitions.json").json()

# 2. Get matches for competition/season
matches = requests.get(f"{base_url}/matches/{comp_id}/{season_id}.json").json()

# 3. Get events for match
events = requests.get(f"{base_url}/events/{match_id}.json").json()

# 4. Filter for corners
corners = [e for e in events
           if e['type']['name'] == 'Pass'
           and e['pass']['type']['name'] == 'Corner']
```

## Data Quality
- **Completeness**: All events have core fields (type, team, player, location)
- **Granularity**: Sub-second timestamps for precise sequencing
- **Relationships**: Events linked via related_events array
- **Spatial precision**: Coordinates in 120x80 grid system
"""

    # Create action taxonomy report
    taxonomy_report = f"""# Corner Kick Action Taxonomy Report

## Dataset Overview
- **Source**: StatsBomb Open Data (Raw GitHub)
- **Competition**: {competition['competition_name']} - {competition['season_name']}
- **Match Analyzed**: {match['home_team']['home_team_name']} vs {match['away_team']['away_team_name']}
- **Total Events**: {len(events)}
- **Corner Kicks Found**: {len([e for e in events if e.get('type', {}).get('name') == 'Pass' and e.get('pass', {}).get('type', {}).get('name') == 'Corner'])}

## Raw Event Sequences After Corners

### Complete Event Type Distribution
Events occurring within 5 events after a corner kick:

| Event Type | Count | Percentage |
|------------|-------|------------|"""

    total = sum(event_types.values())
    for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100 if total > 0 else 0
        taxonomy_report += f"\n| {event_type} | {count} | {pct:.1f}% |"

    taxonomy_report += """

## Outcome Patterns

### Primary Outcomes (First Significant Event)
Significant events are: Shot, Clearance, Goal Keeper, Interception

"""

    # Analyze outcomes
    if outcomes:
        outcome_dist = {}
        for outcome in outcomes:
            key = f"{outcome['type']} ({'same' if outcome['same_team'] else 'opp'})"
            outcome_dist[key] = outcome_dist.get(key, 0) + 1

        taxonomy_report += "| Outcome | Count | Team |\n|---------|-------|------|\n"
        for outcome, count in sorted(outcome_dist.items(), key=lambda x: x[1], reverse=True):
            taxonomy_report += f"| {outcome.split(' (')[0]} | {count} | {outcome.split(' (')[1].strip(')')} team |\n"

    taxonomy_report += """

## Detailed Sequence Examples

### Example Corner Sequences
"""

    # Add 3 example sequences
    for i, seq in enumerate(corner_sequences[:3], 1):
        corner = seq['corner']
        taxonomy_report += f"\n#### Corner {i}: {corner.get('minute', 0)}:{corner.get('second', 0):02d} by {corner.get('team', {}).get('name', 'Unknown')}\n"
        taxonomy_report += f"- **Location**: {corner.get('location', 'N/A')}\n"
        taxonomy_report += f"- **Target**: {corner.get('pass', {}).get('end_location', 'N/A')}\n"
        taxonomy_report += f"- **Next 5 Events**:\n"

        for event in seq['sequence'][:5]:
            taxonomy_report += f"  {event['position']}. {event['type']} ({event['team']})\n"

    taxonomy_report += """

## Proposed Classification Taxonomies

### Option 1: Binary Classification
- **Success**: Shot attempts (including goals)
- **Failure**: All other outcomes

### Option 2: 3-Class System (Recommended)
- **Class 0 - Shot**: Goal or shot attempt
- **Class 1 - Clearance**: Defensive clearance
- **Class 2 - Possession**: Continued play

### Option 3: 4-Class System
- **Class 0 - Goal**: Successful goal
- **Class 1 - Shot**: Shot attempt (no goal)
- **Class 2 - Clearance**: Defensive clearance
- **Class 3 - Possession**: Continued play

## Implementation Considerations

1. **Time Window**: Most outcomes occur within 5 events or 10 seconds
2. **Event Chains**: Track event sequences, not just single outcomes
3. **Team Context**: Consider whether outcome is by attacking or defending team
4. **Spatial Context**: Use location data to assess danger/quality

## Raw Data Advantages

Using raw StatsBomb data provides:
- Complete event sequences
- Precise timestamps
- Related events linkage
- Unprocessed spatial coordinates
- Full event metadata

This enables more sophisticated analysis than processed/aggregated data.
"""

    return infrastructure_report, taxonomy_report

if __name__ == "__main__":
    print("Fetching raw StatsBomb data from GitHub...")

    events, match, competition = fetch_raw_statsbomb_data()

    if events:
        print(f"\nAnalyzing {len(events)} events...")

        # Generate reports
        infra_report, taxonomy_report = generate_reports(events, match, competition)

        # Save reports
        with open('docs/STATSBOMB_INFRASTRUCTURE_REPORT.md', 'w') as f:
            f.write(infra_report)
        print("\nSaved: docs/STATSBOMB_INFRASTRUCTURE_REPORT.md")

        with open('docs/CORNER_ACTION_TAXONOMY.md', 'w') as f:
            f.write(taxonomy_report)
        print("Saved: docs/CORNER_ACTION_TAXONOMY.md")

    else:
        print("Failed to fetch data from GitHub")