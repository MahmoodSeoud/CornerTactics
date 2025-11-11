# StatsBomb Corner Kick Infrastructure Report

## Data Source
- **Repository**: https://github.com/statsbomb/open-data
- **Data Type**: Raw JSON events
- **Competition**: La Liga - 2019/2020
- **Match**: Barcelona vs Eibar

## Raw Data Structure

### Event Schema
Each event in StatsBomb data contains:
```json
{
    "id": "unique-event-id",
    "index": integer,
    "period": integer,
    "timestamp": "HH:MM:SS.SSS",
    "minute": integer,
    "second": integer,
    "type": {"id": integer, "name": "Event Type"},
    "possession": integer,
    "possession_team": {"id": integer, "name": "Team Name"},
    "play_pattern": {"id": integer, "name": "Pattern"},
    "team": {"id": integer, "name": "Team Name"},
    "player": {"id": integer, "name": "Player Name"},
    "position": {"id": integer, "name": "Position"},
    "location": [x, y],
    "duration": float (optional),
    "related_events": ["event-id", ...] (optional),
    // Event-specific fields...
}
```

### Corner Kick Structure
Corner kicks are Pass events with specific attributes:
```json
{
    "type": {"name": "Pass"},
    "pass": {
        "type": {"name": "Corner"},
        "end_location": [x, y],
        "height": {"name": "High Pass" / "Ground Pass"},
        "outcome": {"name": "Complete" / "Incomplete" / etc.},
        "body_part": {"name": "Left Foot" / "Right Foot" / "Head"}
    }
}
```

## Event Types Found After Corners

Distribution of events within 5 events of a corner:
- **Ball Receipt***: 13 occurrences
- **Carry**: 10 occurrences
- **Pass**: 9 occurrences
- **Clearance**: 3 occurrences
- **Goal Keeper**: 1 occurrences


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
