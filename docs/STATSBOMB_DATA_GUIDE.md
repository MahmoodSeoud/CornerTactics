# StatsBomb Data Guide

**Last Updated**: November 15, 2025
**Data Version**: StatsBomb Open Data (JSON format)

## Overview

This guide documents the StatsBomb corner kick data downloaded and stored in `data/raw/statsbomb/json_events/`. The data includes detailed event information, 360-degree freeze frames showing player positions at corner kick moments, and contextual events surrounding each corner.

## Dataset Statistics

```
Total Competitions:     75
Total Matches:          3,464
Total Events:           12,188,949
Total Corners:          34,049
Context Events:         366,281
Matches with Corners:   3,464 (100%)
```

## Directory Structure

```
data/raw/statsbomb/json_events/
├── competitions.json           # Competition metadata (75 competitions)
├── event_sequences.json        # Event context around corners (366,281 events)
├── <match_id>.json            # Full match events (3,464 files)
├── <match_id>.json
└── ...
```

## File Formats

### 1. competitions.json

Contains metadata for all 75 StatsBomb open data competitions.

**Structure**:
```json
[
  {
    "competition_id": 11,
    "competition_name": "La Liga",
    "country_name": "Spain",
    "season_id": 42,
    "season_name": "2020/2021",
    "match_updated": "2021-08-23T12:18:52.805",
    "match_available": "2021-08-23T12:18:52.805"
  }
]
```

**Fields**:
- `competition_id`: Unique competition identifier
- `competition_name`: Name of the competition (e.g., "Premier League", "La Liga")
- `country_name`: Country of the competition
- `season_id`: Unique season identifier
- `season_name`: Season year range (e.g., "2020/2021")
- `match_updated`: Last update timestamp
- `match_available`: Data availability timestamp

### 2. Match Event Files (`<match_id>.json`)

Each file contains all events for a single match, including corner kicks, passes, shots, etc.

**Structure**:
```json
[
  {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "index": 1,
    "period": 1,
    "timestamp": "00:00:00.000",
    "minute": 0,
    "second": 0,
    "type": {
      "id": 35,
      "name": "Starting XI"
    },
    "possession": 1,
    "possession_team": {
      "id": 217,
      "name": "Barcelona"
    },
    "play_pattern": {
      "id": 1,
      "name": "Regular Play"
    },
    "team": {
      "id": 217,
      "name": "Barcelona"
    },
    "duration": 0.0,
    "tactics": {
      "formation": 433,
      "lineup": [
        {
          "player": {
            "id": 5503,
            "name": "Lionel Andrés Messi Cuccittini"
          },
          "position": {
            "id": 21,
            "name": "Right Wing"
          },
          "jersey_number": 10
        }
      ]
    }
  }
]
```

**Common Event Fields**:
- `id`: Unique event UUID
- `index`: Event sequence number in match
- `period`: Match period (1 = first half, 2 = second half)
- `timestamp`: Time since period start (HH:MM:SS.mmm)
- `minute`: Minute of the match
- `second`: Second within the minute
- `type`: Event type object (id + name)
- `possession`: Possession number
- `possession_team`: Team with possession
- `play_pattern`: Pattern of play (Regular Play, From Corner, From Throw In, etc.)
- `team`: Team performing the action
- `player`: Player performing the action (if applicable)
- `position`: Player position
- `location`: [x, y] coordinates on pitch (if applicable)
- `duration`: Event duration in seconds

### 3. Corner Kick Events

Corner kicks have `type.id = 30` and `type.name = "Pass"` with additional corner-specific metadata.

**Corner Event Example**:
```json
{
  "id": "12345678-1234-5678-1234-567812345678",
  "index": 245,
  "period": 1,
  "timestamp": "00:12:34.567",
  "minute": 12,
  "second": 34,
  "type": {
    "id": 30,
    "name": "Pass"
  },
  "possession": 23,
  "possession_team": {
    "id": 217,
    "name": "Barcelona"
  },
  "play_pattern": {
    "id": 3,
    "name": "From Corner"
  },
  "team": {
    "id": 217,
    "name": "Barcelona"
  },
  "player": {
    "id": 5503,
    "name": "Lionel Andrés Messi Cuccittini"
  },
  "position": {
    "id": 21,
    "name": "Right Wing"
  },
  "location": [120.0, 0.5],
  "duration": 1.2,
  "pass": {
    "recipient": {
      "id": 6374,
      "name": "Gerard Piqué Bernabéu"
    },
    "length": 18.5,
    "angle": 1.57,
    "height": {
      "id": 3,
      "name": "High Pass"
    },
    "end_location": [102.0, 40.0],
    "type": {
      "id": 61,
      "name": "Corner"
    },
    "body_part": {
      "id": 40,
      "name": "Right Foot"
    },
    "outcome": {
      "id": 9,
      "name": "Incomplete"
    }
  },
  "freeze_frame": [
    {
      "location": [102.5, 38.2],
      "player": {
        "id": 6374,
        "name": "Gerard Piqué Bernabéu"
      },
      "position": {
        "id": 3,
        "name": "Center Back"
      },
      "teammate": true
    },
    {
      "location": [98.3, 42.1],
      "player": {
        "id": 9876,
        "name": "Sergio Ramos García"
      },
      "position": {
        "id": 3,
        "name": "Center Back"
      },
      "teammate": false
    }
  ]
}
```

**Corner-Specific Fields**:
- `pass.type.name`: Always "Corner" for corner kicks
- `pass.recipient`: Player receiving the corner
- `pass.end_location`: [x, y] where ball lands
- `pass.height`: Pass trajectory (High Pass, Ground Pass)
- `pass.body_part`: Kicking foot
- `pass.outcome`: Result (Complete, Incomplete, Out, etc.)
- `freeze_frame`: Array of player positions at corner moment (360 data)

### 4. 360 Freeze Frame Data

The `freeze_frame` array contains positions of all visible players at the moment of the corner kick.

**Freeze Frame Object**:
```json
{
  "location": [102.5, 38.2],
  "player": {
    "id": 6374,
    "name": "Gerard Piqué Bernabéu"
  },
  "position": {
    "id": 3,
    "name": "Center Back"
  },
  "teammate": true
}
```

**Fields**:
- `location`: [x, y] position on pitch (120x80 coordinate system)
- `player`: Player object with id and name
- `position`: Player's position/role
- `teammate`: Boolean - true if same team as corner taker, false if opponent

**Typical Freeze Frame Size**: 15-22 players (outfield players from both teams)

### 5. Event Sequences (`event_sequences.json`)

Contains 366,281 events that occur before and after corner kicks, providing context for analysis.

**Structure**: Same as match event files, but filtered to only include events in temporal proximity to corner kicks (typically ±20 seconds).

## Coordinate System

StatsBomb uses a standardized pitch coordinate system:

```
(0, 80) ─────────────────────────── (120, 80)
   │                                     │
   │         Attacking Direction →       │
   │                                     │
(0, 40) ─────────── Center ──────────(120, 40)
   │                                     │
   │                                     │
   │                                     │
(0, 0)  ─────────────────────────── (120, 0)

Defensive Goal                    Attacking Goal
```

**Dimensions**:
- **X-axis**: 0 (defensive goal) to 120 (attacking goal)
- **Y-axis**: 0 (bottom sideline) to 80 (top sideline)
- **Units**: Abstract units (not meters)

**Corner Kick Locations**:
- Right corner: (120, 0) or near (120, 0.5)
- Left corner: (120, 80) or near (120, 79.5)

**Penalty Box**:
- X: 102 to 120
- Y: 18 to 62

**Goal**:
- X: 120
- Y: 36 to 44 (width = 8 units)

## Event Types

Common event types you'll find in the data:

| Type ID | Event Name | Description |
|---------|------------|-------------|
| 30 | Pass | All passes including corners |
| 16 | Shot | Shot attempts |
| 6 | Block | Defensive blocks |
| 9 | Clearance | Defensive clearances |
| 10 | Interception | Ball interceptions |
| 2 | Ball Receipt | Receiving the ball |
| 3 | Dispossessed | Losing possession |
| 4 | Duel | 50/50 challenges |
| 17 | Pressure | Defensive pressure |
| 42 | Ball Recovery | Regaining possession |

## Competition Coverage

The dataset includes professional men's competitions only:

**Major Leagues**:
- La Liga (Spain)
- Premier League (England)
- Serie A (Italy)
- Bundesliga (Germany)
- Ligue 1 (France)

**International Tournaments**:
- UEFA Champions League
- UEFA Europa League
- FIFA World Cup
- UEFA European Championship
- Copa América

**Seasons**: Varies by competition, typically 2015/2016 onwards

## Data Quality Notes

1. **360 Coverage**: Not all events have freeze frame data. Only events with visible player tracking have the `freeze_frame` field.

2. **Corner Identification**:
   - `type.name = "Pass"` AND `pass.type.name = "Corner"`
   - OR `play_pattern.name = "From Corner"` (for subsequent events)

3. **Missing Data**: Some fields may be null or missing:
   - `pass.recipient` may be null if no clear recipient
   - `pass.outcome` may be missing for some passes
   - Not all players may be in freeze frame (camera angle limitations)

4. **Coordinate Precision**: Locations are typically precise to 0.1 units

## Using the Data

### Loading Match Events (Python)

```python
import json

# Load single match
with open('data/raw/statsbomb/json_events/123456.json', 'r') as f:
    events = json.load(f)

# Filter for corner kicks
corners = [
    e for e in events
    if e.get('type', {}).get('name') == 'Pass'
    and e.get('pass', {}).get('type', {}).get('name') == 'Corner'
]

print(f"Found {len(corners)} corners in this match")
```

### Extracting Freeze Frame Positions

```python
# Get freeze frame from corner event
corner = corners[0]
freeze_frame = corner.get('freeze_frame', [])

# Separate attacking and defending players
attacking_players = [p for p in freeze_frame if p['teammate']]
defending_players = [p for p in freeze_frame if not p['teammate']]

print(f"Attacking: {len(attacking_players)}, Defending: {len(defending_players)}")

# Extract positions
attacking_positions = [p['location'] for p in attacking_players]
defending_positions = [p['location'] for p in defending_players]
```

### Using StatsBombLoader

```python
from src.statsbomb_loader import StatsBombCornerLoader

# Initialize loader
loader = StatsBombCornerLoader(output_dir="data/raw/statsbomb")

# Get available competitions
competitions = loader.get_available_competitions()

# Build dataset for specific competition
df = loader.build_corner_dataset(
    country="England",
    division="Premier League",
    season="2019/2020",
    gender="male"
)

print(f"Loaded {len(df)} corners")
print(df.columns)
```

## Analysis Examples

### 1. Corner Outcome Analysis

Analyze what happens after corner kicks:

```python
# Look at next 3 events after corner
next_events = events[corner['index']:corner['index']+4]

# Common outcomes:
# - Shot (type.id = 16)
# - Clearance (type.id = 9)
# - Ball Receipt (type.id = 2)
# - Interception (type.id = 10)
```

### 2. Player Positioning Analysis

Analyze defensive vs attacking setups:

```python
import numpy as np

# Calculate average positions
attacking_avg = np.mean(attacking_positions, axis=0)
defending_avg = np.mean(defending_positions, axis=0)

# Count players in penalty box (x > 102, 18 < y < 62)
in_box = sum(
    1 for p in freeze_frame
    if p['location'][0] > 102
    and 18 < p['location'][1] < 62
)
```

### 3. Corner Success Rate

```python
# Check if corner led to shot within 10 events
def led_to_shot(corner_idx, events, window=10):
    for i in range(corner_idx + 1, min(corner_idx + window, len(events))):
        if events[i].get('type', {}).get('id') == 16:  # Shot
            return True
    return False

success_rate = sum(led_to_shot(c['index'], events) for c in corners) / len(corners)
print(f"Shot conversion rate: {success_rate:.1%}")
```

## Data License

StatsBomb open data is freely available for research and non-commercial use. Please review [StatsBomb's license terms](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf) before use.

**Attribution**: When using this data, cite StatsBomb as the source:
```
StatsBomb (2025). StatsBomb Open Data.
https://github.com/statsbomb/open-data
```

## References

- **StatsBomb Open Data**: https://github.com/statsbomb/open-data
- **StatsBomb API Documentation**: https://statsbomb.com/what-we-do/hub/free-data/
- **StatsBombPy Library**: https://github.com/statsbomb/statsbombpy

## Additional Resources

For more information on the download process and project structure, see:
- `README.md` - Project overview and installation
- `CLAUDE.md` - Development guide
- `docs/REFACTORING_SUMMARY.md` - Project refactoring history
