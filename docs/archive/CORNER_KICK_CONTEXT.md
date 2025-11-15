# StatsBomb Corner Kick Analysis: Complete Context

## Dataset Overview
- **Matches Analyzed**: 5
- **Total Corner Kicks**: 60
- **Events After Corners Analyzed**: 1200

### Matches Included:
- Champions League: Tottenham Hotspur vs Liverpool
- Champions League: Real Madrid vs Liverpool
- Champions League: Juventus vs Real Madrid
- Champions League: Real Madrid vs Atlético Madrid
- Champions League: Juventus vs Barcelona


## 1. EVENT TRANSITION ANALYSIS

### What Happens After a Corner Kick?

#### Immediate Transitions (Position 1 - First Event After Corner)
The probability P(Event at t+1 | Corner at t):

| Next Event | Count | Probability |
|------------|-------|-------------|
| Ball Receipt* | 33 | 0.550 |
| Clearance | 18 | 0.300 |
| Goal Keeper | 4 | 0.067 |
| Foul Committed | 3 | 0.050 |
| Pressure | 1 | 0.017 |
| Duel | 1 | 0.017 |


#### Full Transition Matrix (Top 10x10)
P(Event j at t+1 | Event i at t) for all event types:

```
                Ball Recovery  Block  Carry  Corner  Dispossessed  Dribble  Dribbled Past   Duel  Foul Committed  Foul Won
Ball Recovery           0.000    0.0  0.762     0.0           0.0    0.000          0.000  0.000            0.00       0.0
Block                   0.000    0.0  0.000     0.0           0.0    0.000          0.000  0.000            0.00       0.0
Carry                   0.000    0.0  0.000     0.0           0.0    0.013          0.013  0.000            0.00       0.0
Corner                  0.000    0.0  0.000     0.0           0.0    0.000          0.000  0.017            0.05       0.0
Dispossessed            0.000    0.0  0.000     0.0           0.0    0.000          0.000  1.000            0.00       0.0
Dribble                 0.000    0.0  0.333     0.0           0.0    0.000          0.000  0.667            0.00       0.0
Dribbled Past           0.000    0.0  0.000     0.0           0.0    1.000          0.000  0.000            0.00       0.0
Duel                    0.038    0.0  0.077     0.0           0.0    0.000          0.000  0.000            0.00       0.0
Foul Committed          0.000    0.0  0.000     0.0           0.0    0.000          0.000  0.000            0.00       1.0
Foul Won                0.000    0.0  0.000     0.0           0.0    0.000          0.000  0.000            0.00       0.0
```


#### Event Chains (First 3 Events After Corner)

Most common 3-event sequences after corners:

| Sequence | Count | Percentage |
|----------|-------|------------|
| Ball Receipt* → Carry → Pass | 7 | 11.7% |
| Clearance → Pass → Ball Receipt* | 6 | 10.0% |
| Ball Receipt* → Shot → Goal Keeper | 6 | 10.0% |
| Ball Receipt* → Duel → Clearance | 4 | 6.7% |
| Ball Receipt* → Carry → Pressure | 4 | 6.7% |
| Clearance → Ball Recovery → Carry | 4 | 6.7% |
| Ball Receipt* → Duel → Shot | 3 | 5.0% |
| Clearance → Ball Recovery → Shot | 3 | 5.0% |
| Foul Committed → Foul Won → Pass | 2 | 3.3% |
| Clearance → Pass → Clearance | 2 | 3.3% |


## 2. RAW DATA STRUCTURE & FEATURES

### Core Event Structure
Every event in StatsBomb data contains these fields:

#### Universal Fields (Present in ALL Events)
- `duration`
- `location`
- `off_camera`
- `player`
- `related_events`
- `team`
- `timestamp`
- `type`


#### Event-Specific Fields
Fields that appear in specific event types:


**50/50**:
- `out`
- `under_pressure`

**Ball Receipt***:
- `out`
- `under_pressure`

**Ball Recovery**:
- `out`
- `under_pressure`

**Block**:
- `out`
- `under_pressure`

**Carry**:
- `out`
- `under_pressure`

**Clearance**:
- `out`
- `under_pressure`

**Dispossessed**:
- `out`
- `under_pressure`

**Dribble**:
- `out`
- `under_pressure`

**Dribbled Past**:
- `out`
- `under_pressure`

**Duel**:
- `out`
- `under_pressure`

**Error**:
- `out`
- `under_pressure`

**Foul Committed**:
- `out`
- `under_pressure`

**Foul Won**:
- `out`
- `under_pressure`

**Goal Keeper**:
- `out`
- `under_pressure`

**Half End**:
- `out`
- `under_pressure`

**Half Start**:
- `out`
- `under_pressure`

**Injury Stoppage**:
- `out`
- `under_pressure`

**Interception**:
- `interception_details`
- `out`
- `under_pressure`

**Miscontrol**:
- `out`
- `under_pressure`

**Offside**:
- `out`
- `under_pressure`

**Pass**:
- `out`
- `under_pressure`

**Pressure**:
- `out`
- `under_pressure`

**Shield**:
- `out`
- `under_pressure`

**Shot**:
- `out`
- `under_pressure`

**Substitution**:
- `out`
- `under_pressure`

**Tactical Shift**:
- `out`
- `under_pressure`


### Corner Kick Specific Features

#### Pass (Corner) Features Available:
- `pass.angle`
- `pass.cross`
- `pass.end_location`
- `pass.goal_assist`
- `pass.height`
- `pass.length`
- `pass.outcome`
- `pass.shot_assist`
- `pass.switch`


#### Example Corner Event (Raw JSON Structure):
```json
{
    "id": "uuid-here",
    "type": {"name": "Pass"},
    "pass": {
        "type": {"name": "Corner"},
        "end_location": [112.8, 43.8],
        "height": {"name": "High Pass"},
        "outcome": {"name": "Complete"},
        "body_part": {"name": "Right Foot"},
        "length": 18.5,
        "angle": 2.1
    },
    "team": {"name": "Barcelona"},
    "player": {"name": "Lionel Messi"},
    "location": [120.0, 80.0],
    "timestamp": "00:21:14.123",
    "minute": 21,
    "second": 14
}
```

### Spatial Features

**Location Coverage**: 1181 events have location data

**Coordinate System**:
- Origin: [0, 0] at bottom-left of defending goal
- Range: X ∈ [0, 120], Y ∈ [0, 80]
- Corner locations: [120, 0] or [120, 80] (attacking corners)

**Example Locations**:
- Ball Receipt*: [109.6, 44.7]
- Duel: [110.6, 44.3]
- Clearance: [9.5, 35.8]
- Pressure: [34.7, 28.7]
- Ball Recovery: [82.5, 55.6]


### Temporal Features

**Timestamp Coverage**: 1200 events have timestamps
**Duration Coverage**: 637 events have duration

**Example Timestamps**:
- Ball Receipt*: 00:05:18.792
- Duel: 00:05:18.792
- Clearance: 00:05:18.792


### Pressure & Context Features

**Under Pressure**: 280/1200 events (23.3%)

### Shot-Specific Features (When Shot Occurs)
- `shot.body_part`
- `shot.end_location`
- `shot.first_time`
- `shot.follows_dribble`
- `shot.outcome`
- `shot.statsbomb_xg`
- `shot.technique`
- `shot.type`


### Clearance-Specific Features
- `clearance.aerial_won`
- `clearance.body_part`
- `clearance.head`


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
