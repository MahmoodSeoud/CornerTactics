# StatsBomb Data Guide

**Last Updated**: November 15, 2025
**Data Version**: StatsBomb Open Data (JSON format)

## Overview

This guide documents the StatsBomb corner kick data downloaded and stored in `data/statsbomb/`. The data includes detailed event information and 360-degree freeze frames showing player positions at corner kick moments.

## Dataset Statistics

```
Total Competitions:     75
Total Matches:          3,464
Total Events:           12,188,949
Total Corners:          34,049
Corners with 360 Data:  1,933 (5.7%)
Context Events:         366,281
Matches with Corners:   3,464 (100%)
```

## Corner Outcome Distribution

Analysis of 1,933 corners with 360 freeze frame data shows the following distribution of immediate next events:

```
Event Type                Count    Percentage
------------------------------------------------------------
Ball Receipt*             1,050     54.3%
Clearance                 453       23.4%
Goal Keeper               196       10.1%
Duel                      73         3.8%
Pressure                  57         2.9%
Pass                      41         2.1%
Foul Committed            27         1.4%
Ball Recovery             18         0.9%
Block                     9          0.5%
Other (rare events)       9          0.5%
```

**Key Findings**:
- Ball Receipt is the most common outcome (54.3%)
- Defensive actions (Clearance + Goal Keeper) account for 33.5%
- Top 3 event types cover 87.8% of all corner outcomes
- Top 6 event types cover 96.6% of all corner outcomes

**Recommended Classification Approaches**:

1. **4-Class System** (Recommended for balanced classification):
   - Ball Receipt (54.3%)
   - Clearance (23.4%)
   - Goalkeeper (10.1%)
   - Other (12.2%)

2. **3-Class Tactical System** (For tactical analysis):
   - Offensive Retention: Ball Receipt + Pass (56.4%)
   - Defensive Action: Clearance + Goal Keeper + Block (33.9%)
   - Contest: Duel + Pressure + Foul (9.7%)

3. **6-Class Fine-Grained** (For detailed analysis, accepts class imbalance):
   - Ball Receipt (54.3%), Clearance (23.4%), Goal Keeper (10.1%), Duel (3.8%), Pressure (2.9%), Pass (2.1%)
   - Other: All remaining events (3.4%)

## Directory Structure

```
data/
├── misc/
│   ├── soccernet/
│   ├── soccersynth/
│   └── ussf_data_sample.pkl
└── statsbomb/
    ├── competitions.json        # Competition metadata
    ├── match_index.csv          # Match metadata index
    ├── events/                  # Match event files
    │   └── <match_id>.json     # Full match events (one per match)
    └── freeze-frames/           # 360 freeze frame data
        └── <match_id>.json     # Player positions at set piece moments
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

**Corner-Specific Fields** (Complete List):

**Event-Level Fields**:
- `id`: Unique event UUID (for matching with freeze frames)
- `index`: Event sequence number in match (for temporal ordering)
- `period`: Match period (1 = first half, 2 = second half)
- `timestamp`: Time since period start (HH:MM:SS.mmm format)
- `minute`: Minute of the match
- `second`: Second within the minute
- `duration`: Event duration in seconds
- `location`: [x, y] corner kick position (usually [120, 0] or [120, 80])
- `player`: Corner taker (id, name)
- `position`: Corner taker's position
- `team`: Team taking the corner (id, name)
- `possession`: Possession sequence number
- `possession_team`: Team with possession
- `play_pattern`: Pattern of play (usually "From Corner")

**Pass Object Fields**:
- `pass.type.name`: Always "Corner" for corner kicks
- `pass.end_location`: [x, y] where ball lands
- `pass.length`: Euclidean distance from corner to landing (in pitch units)
- `pass.angle`: Pass angle in radians (-π to π, 0 = horizontal right)
- `pass.height`: Pass trajectory object with `id` and `name`
  - **Observed values**: "High Pass" (most common), "Low Pass", "Ground Pass"
- `pass.body_part`: Kicking body part object with `id` and `name`
  - **Observed values in corners**: "Right Foot", "Left Foot" only (no Head/Other in corner kicks)
- `pass.technique`: Corner technique object with `id` and `name` (may be null)
  - **Observed values**: "Inswinging", "Outswinging", "Straight", null
  - **Coverage**: Present in ~95% of corners, null in ~5%
- `pass.inswinging`: Boolean (true if inswinging) - only present when true, absent otherwise
  - **Coverage**: Present in ~45% of corners
- `pass.outswinging`: Boolean (true if outswinging) - only present when true, absent otherwise
  - **Coverage**: Present in ~37% of corners
- `pass.outcome`: Pass result object with `id` and `name` (may be null)
  - **Observed values**: "Incomplete" (most common), "Out", "Pass Offside", "Unknown", null
  - **Coverage**: Present in ~85% of corners, null in ~15%
  - **Note**: "Complete" outcome is rare/absent in corners (corners rarely directly complete)
- `pass.recipient`: Intended recipient player object (id, name) - may be null
  - **Coverage**: Present in 1,111 / 1,933 corners (57.5%)
- `pass.switch`: Boolean (true if cross-field switch pass) - only present when true
  - **Coverage**: Present in 710 / 1,933 corners (36.7%)
- `pass.shot_assist`: Boolean (true if led to shot) - only present when true
  - **Coverage**: Present in 371 / 1,933 corners (19.2%)
- `pass.assisted_shot_id`: UUID of assisted shot - only present when shot_assist is true
  - **Coverage**: Present in 371 / 1,933 corners (19.2%)

**Freeze Frame Array** (360° player positioning):
- `freeze_frame`: Array of player positions at corner moment (see section 4 below)

---

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
  "teammate": true,
  "actor": false,
  "keeper": false
}
```

**Fields** (Complete List):
- `location`: [x, y] position on pitch (120x80 coordinate system)
- `player`: Player object with id and name (may be null for some players)
- `position`: Player's position/role (id and name)
- `teammate`: Boolean - true if same team as corner taker, false if opponent
- `actor`: Boolean - true if this player is the corner taker (always excluded from freeze frame)
- `keeper`: Boolean - true if this player is a goalkeeper

**Typical Freeze Frame Size**: 15-22 players (outfield players from both teams)

**Important Notes**:
- The corner taker (`actor=true`) is NOT included in freeze frames (always at corner flag)
- Goalkeeper identification via `keeper=true` is critical for defensive analysis
- Some players may have `player=null` if StatsBomb couldn't identify them

### 5. 360 Freeze Frame Files (`freeze-frames/<match_id>.json`)

StatsBomb 360 freeze frame data provides complete player positioning at the exact moment of set pieces (corners, free kicks, etc.). This data is stored separately from event data.

**Structure**:
```json
[
  {
    "event_uuid": "12345678-1234-5678-1234-567812345678",
    "visible_area": [[0, 0], [120, 80]],
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
      }
    ]
  }
]
```

**Fields**:
- `event_uuid`: UUID linking to the corner event in the events file
- `visible_area`: Visible pitch area captured by cameras
- `freeze_frame`: Array of all visible player positions at the moment of the corner

**Coverage**:
- ~62.6% of corners in matches with 360 data have freeze frames
- ~57.5% of corners with freeze frames also have recipients
- Typical freeze frame contains 15-22 players

**Matching with Events**:
To link 360 data with corner events, match the `event_uuid` from the freeze frame file with the `id` field from corner events in the events file.

---

## Data Characteristics & Field Coverage

This section documents the actual coverage and value distributions found in the **1,933 corners with 360° freeze frames** from StatsBomb Open Data.

### Pass Field Coverage Analysis

| Field | Coverage | Observed Values | Notes |
|-------|----------|-----------------|-------|
| `pass.type.name` | 100% | "Corner" | Always present, identifies corner kicks |
| `pass.end_location` | 100% | [x, y] coordinates | Always present |
| `pass.length` | 100% | Numeric (pitch units) | Always present |
| `pass.angle` | 100% | Numeric (radians) | Always present, range [-π, π] |
| `pass.height.name` | 100% | "High Pass", "Low Pass", "Ground Pass" | Always present, "High Pass" most common |
| `pass.body_part.name` | 100% | "Right Foot", "Left Foot" | Always present, no Head/Other in corners |
| `pass.technique.name` | ~95% | "Inswinging", "Outswinging", "Straight", null | Null in ~5% of corners |
| `pass.inswinging` | ~45% | true | Only present when true, absent otherwise |
| `pass.outswinging` | ~37% | true | Only present when true, absent otherwise |
| `pass.outcome.name` | ~85% | "Incomplete", "Out", "Pass Offside", "Unknown", null | Null in ~15% of corners |
| `pass.recipient` | 57.5% | Player object (id, name) | 1,111 / 1,933 corners |
| `pass.switch` | 36.7% | true | 710 / 1,933 corners, only when true |
| `pass.shot_assist` | 19.2% | true | 371 / 1,933 corners, only when true |
| `pass.assisted_shot_id` | 19.2% | UUID string | 371 / 1,933 corners, when shot_assist=true |

### Freeze Frame Field Coverage

| Field | Coverage | Observed Values | Notes |
|-------|----------|-----------------|-------|
| `location` | 100% | [x, y] coordinates | Always present for all players |
| `teammate` | 100% | true, false | Always present |
| `actor` | 100% | false | Always false (corner taker excluded from freeze frames) |
| `keeper` | 100% | true, false | Always present, typically 1-2 keepers per frame |
| `player` | ~95% | Player object or null | Some players unidentified by StatsBomb |
| `position` | 100% | Position object | Always present |

### Value Distribution Insights

**Pass Technique Distribution** (n=1,933):
- Inswinging: ~45% (870 corners)
- Outswinging: ~37% (710 corners)
- Straight: ~13% (250 corners)
- Null/Unspecified: ~5% (103 corners)

**Pass Outcome Distribution** (where not null, n=~1,643):
- Incomplete: ~75% (most common - corner is cleared/intercepted)
- Out: ~15% (ball goes out of play)
- Pass Offside: ~7% (recipient in offside position)
- Unknown: ~3% (uncertain outcome)

**Body Part Distribution** (n=1,933):
- Right Foot: ~55% (1,063 corners)
- Left Foot: ~45% (870 corners)
- Head/Other: 0% (never observed in corner kicks)

**Pass Height Distribution** (n=1,933):
- High Pass: ~85% (most corners are lofted into box)
- Low Pass: ~12% (driven corners)
- Ground Pass: ~3% (short corners or ground deliveries)

### Optional Field Handling

**Boolean Fields** (only present when true):
- `pass.inswinging`: Check with `.get('inswinging', False)`
- `pass.outswinging`: Check with `.get('outswinging', False)`
- `pass.switch`: Check with `.get('switch', False)`
- `pass.shot_assist`: Check with `.get('shot_assist', False)`

**Nullable Object Fields**:
- `pass.technique`: Check with `.get('technique', {}).get('name', None)`
- `pass.outcome`: Check with `.get('outcome', {}).get('name', None)`
- `pass.recipient`: Check with `.get('recipient', None)`

**Example Safe Access**:
```python
# Safe access to optional fields
technique = corner['pass'].get('technique', {}).get('name', 'Unknown')
is_inswinging = corner['pass'].get('inswinging', False)
has_recipient = corner['pass'].get('recipient') is not None
outcome = corner['pass'].get('outcome', {}).get('name', 'Unknown')
```

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
with open('data/statsbomb/events/123456.json', 'r') as f:
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
loader = StatsBombCornerLoader(output_dir="data/statsbomb")

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

### Loading 360 Freeze Frame Data

```python
import json
from pathlib import Path

# Load match events
match_id = "123456"
with open(f'data/statsbomb/events/{match_id}.json', 'r') as f:
    events = json.load(f)

# Load 360 freeze frames for same match
freeze_path = Path(f'data/statsbomb/freeze-frames/{match_id}.json')
if freeze_path.exists():
    with open(freeze_path, 'r') as f:
        freeze_data = json.load(f)

    # Create lookup dict: event_uuid -> freeze_frame
    freeze_lookup = {item['event_uuid']: item['freeze_frame'] for item in freeze_data}

    # Find corners and match with freeze frames
    for event in events:
        if (event.get('type', {}).get('name') == 'Pass' and
            event.get('pass', {}).get('type', {}).get('name') == 'Corner'):

            event_id = event['id']
            if event_id in freeze_lookup:
                freeze_frame = freeze_lookup[event_id]
                print(f"Corner by {event['player']['name']} has {len(freeze_frame)} players")
```

## Feature Extraction Reference

### Complete Feature Set (49 Features)

This section documents all features that can be extracted from StatsBomb corner kick data. See `PLAN.md` for implementation details.

#### **Category 1: Basic Corner Metadata (5 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `corner_side` | int | `event.location[1]` | Left corner (0) or Right corner (1), threshold at y=40 |
| `period` | int | `event.period` | Match period (1 or 2) |
| `minute` | int | `event.minute` | Minute of match |
| `corner_x` | float | `event.location[0]` | Corner kick x-coordinate (usually 120) |
| `corner_y` | float | `event.location[1]` | Corner kick y-coordinate (0-80) |

#### **Category 2: Temporal Features (3 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `second` | int | `event.second` | Second within the minute (0-59) |
| `timestamp_seconds` | float | `event.timestamp` | Total seconds from period start (convert HH:MM:SS.mmm) |
| `duration` | float | `event.duration` | Event duration in seconds |

#### **Category 3: Player Count Features (6 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `total_attacking` | int | `freeze_frame` | Total attacking players (teammate=true, excluding actor) |
| `total_defending` | int | `freeze_frame` | Total defending players (teammate=false) |
| `attacking_in_box` | int | `freeze_frame` | Attacking players in penalty box (x>102, 18<y<62) |
| `defending_in_box` | int | `freeze_frame` | Defending players in penalty box |
| `attacking_near_goal` | int | `freeze_frame` | Attacking players near goal (x>108, 30<y<50) |
| `defending_near_goal` | int | `freeze_frame` | Defending players near goal |

#### **Category 4: Spatial Density Features (4 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `attacking_density` | float | `freeze_frame` | Attacking player density in box (count / box_area) |
| `defending_density` | float | `freeze_frame` | Defending player density in box |
| `numerical_advantage` | int | `freeze_frame` | Attacking - defending players in box |
| `attacker_defender_ratio` | float | `freeze_frame` | Ratio of attackers to defenders in box |

#### **Category 5: Positional Features (8 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `attacking_centroid_x` | float | `freeze_frame` | Mean x-coordinate of attacking players |
| `attacking_centroid_y` | float | `freeze_frame` | Mean y-coordinate of attacking players |
| `defending_centroid_x` | float | `freeze_frame` | Mean x-coordinate of defending players |
| `defending_centroid_y` | float | `freeze_frame` | Mean y-coordinate of defending players |
| `defending_compactness` | float | `freeze_frame` | Std dev of defending y-positions (horizontal spread) |
| `defending_depth` | float | `freeze_frame` | Max x - min x of defending players (vertical depth) |
| `attacking_to_goal_dist` | float | `freeze_frame` | Distance from attacking centroid to goal center (120, 40) |
| `defending_to_goal_dist` | float | `freeze_frame` | Distance from defending centroid to goal center |

#### **Category 6: Pass Trajectory Features (4 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `pass_end_x` | float | `event.pass.end_location[0]` | Pass landing x-coordinate |
| `pass_end_y` | float | `event.pass.end_location[1]` | Pass landing y-coordinate |
| `pass_length` | float | `event.pass.length` | Euclidean distance from corner to landing |
| `pass_height` | int | `event.pass.height.name` | Ground Pass (0), Low Pass (1), High Pass (2) |

#### **Category 7: Pass Technique & Body Part (5 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `pass_angle` | float | `event.pass.angle` | Pass angle in radians (-π to π, 0=horizontal right). Always present. |
| `pass_body_part` | int | `event.pass.body_part.name` | Right Foot (0), Left Foot (1). **Note**: Only these 2 values observed in corners, no Head/Other. Always present. |
| `pass_technique` | int | `event.pass.technique.name` | Inswinging (0), Outswinging (1), Straight (2), Unknown/null (3). Present in ~95% of corners. |
| `is_inswinging` | bool | `event.pass.inswinging` | 1 if inswinging corner, 0 otherwise. Present in ~45% of corners when true. |
| `is_outswinging` | bool | `event.pass.outswinging` | 1 if outswinging corner, 0 otherwise. Present in ~37% of corners when true. |

#### **Category 8: Pass Outcome & Context (4 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `pass_outcome` | int | `event.pass.outcome.name` | Incomplete (0), Out (1), Pass Offside (2), Unknown (3), null (4). Present in ~85% of corners. **Note**: "Complete" outcome not observed in corners. |
| `is_cross_field_switch` | bool | `event.pass.switch` | 1 if switch pass across field, 0 otherwise. Present in 36.7% of corners (710/1,933). |
| `has_recipient` | bool | `event.pass.recipient` | 1 if recipient identified, 0 if null. Present in 57.5% of corners (1,111/1,933). |
| `is_shot_assist` | bool | `event.pass.shot_assist` | 1 if corner led to shot, 0 otherwise. Present in 19.2% of corners (371/1,933). |

#### **Category 9: Goalkeeper & Special Player Features (3 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `num_attacking_keepers` | int | `freeze_frame` | Count of attacking goalkeepers (keeper=true, teammate=true) |
| `num_defending_keepers` | int | `freeze_frame` | Count of defending goalkeepers (keeper=true, teammate=false) |
| `keeper_distance_to_goal` | float | `freeze_frame` | Distance from defending keeper to goal center (120, 40) |

#### **Category 10: Match Context - Score State (4 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `attacking_team_goals` | int | Match events | Goals by corner-taking team before this corner |
| `defending_team_goals` | int | Match events | Goals by defending team before this corner |
| `score_difference` | int | Match events | attacking_team_goals - defending_team_goals |
| `match_situation` | int | Match events | Winning (1), Drawing (0), Losing (-1) |

**Extraction**: Track all Shot events where `shot.outcome.name == "Goal"` with `index < corner.index`, count by team.

#### **Category 11: Match Context - Substitution Patterns (3 features)**
| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `total_subs_before` | int | Match events | Total substitutions before corner (both teams) |
| `recent_subs_5min` | int | Match events | Substitutions in last 5 minutes (both teams) |
| `minutes_since_last_sub` | float | Match events | Minutes since last substitution (999 if none) |

**Extraction**: Track all Substitution events where `type.name == "Substitution"` with `index < corner.index`.

---

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
