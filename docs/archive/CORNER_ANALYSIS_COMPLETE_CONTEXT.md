# Corner Kick Analysis: Complete Context for ML

## Overview
This document provides all context needed to understand:
1. **P(Event at t+1 | Corner at t)** - What happens after corners
2. **Data structure & features** - What raw data exists

Dataset: 60 corner kicks from 5 Champions League matches (18,203 total events)

---

## 1. TRANSITION PROBABILITIES: What Happens After Corners?

### Direct Transitions: P(Event at t+1 | Corner at t)

When a corner kick occurs, the **immediate next event** probabilities are:

| Next Event | Probability | Interpretation |
|------------|-------------|----------------|
| **Ball Receipt*** | 0.550 | Player receives the ball (55% of corners) |
| **Clearance** | 0.300 | Defensive clearance (30% of corners) |
| **Goal Keeper** | 0.067 | Goalkeeper claims/punches (6.7%) |
| **Foul Committed** | 0.050 | Foul occurs (5%) |
| **Duel** | 0.017 | Aerial/ground duel (1.7%) |
| **Pressure** | 0.017 | Player under pressure (1.7%) |

### Multi-Step Sequences: Common 3-Event Chains

The most frequent 3-event sequences after corners:

| Sequence | Frequency | Description |
|----------|-----------|-------------|
| Ball Receipt* → Carry → Pass | 11.7% | Player receives, carries, passes |
| Clearance → Pass → Ball Receipt* | 10.0% | Clear, pass out, teammate receives |
| Ball Receipt* → Shot → Goal Keeper | 10.0% | Receive and shoot, keeper saves |
| Ball Receipt* → Duel → Clearance | 6.7% | Receive, aerial duel, cleared |
| Ball Receipt* → Duel → Shot | 5.0% | Receive, win duel, shoot |

### Full Transition Matrix

For any event type i, the probability of next event j is:

**Key transitions:**
- Pass → Ball Receipt*: 0.777 (passes usually completed)
- Ball Receipt* → Carry: 0.560 (receivers often dribble)
- Carry → Pass: 0.553 (carries lead to passes)
- Shot → Goal Keeper: 0.783 (most shots saved/blocked)
- Clearance → Ball Recovery: ~0.4 (clearances often recovered)

---

## 2. RAW DATA STRUCTURE & FEATURES

### Event Schema (JSON)

Every event in StatsBomb raw data contains:

```json
{
    // UNIVERSAL FIELDS (always present)
    "id": "unique-event-id",
    "type": {"id": 123, "name": "Pass"},  // Event type
    "team": {"id": 456, "name": "Barcelona"},
    "player": {"id": 789, "name": "Messi"},
    "location": [x, y],  // Spatial coordinates
    "timestamp": "HH:MM:SS.mmm",  // Precise timing
    "minute": 21,
    "second": 14,
    "duration": 1.2,  // Duration in seconds
    "related_events": ["event-id-1", "event-id-2"],  // Linked events

    // CONTEXTUAL FIELDS (common)
    "under_pressure": true/false,  // 23.3% of events
    "off_camera": false,
    "out": false,  // Ball out of play

    // TYPE-SPECIFIC FIELDS (varies by event type)
    "pass": {...},  // If Pass event
    "shot": {...},  // If Shot event
    "clearance": {...}  // If Clearance event
}
```

### Corner Kick Specific Structure

Corners are Pass events with additional fields:

```json
{
    "type": {"name": "Pass"},
    "pass": {
        "type": {"name": "Corner"},
        "end_location": [112.8, 43.8],  // Target location
        "length": 18.5,  // Distance in yards
        "angle": 2.1,  // Angle in radians
        "height": {"name": "High Pass"},  // High/Low/Ground
        "outcome": {"name": "Complete"},  // Complete/Incomplete
        "body_part": {"name": "Right Foot"},
        "cross": false,
        "switch": false
    },
    "location": [120.0, 80.0]  // Always from corner flag
}
```

### Available Features by Category

#### Spatial Features
- **location**: [x, y] for every event
- **Coordinate system**: 120×80 yards
  - Corners at: [120, 0] or [120, 80]
  - Goal at: [120, 40]
- **Coverage**: 98.4% of events have locations

#### Temporal Features
- **timestamp**: "HH:MM:SS.mmm" format
- **minute/second**: Integer time markers
- **duration**: Event duration (53% coverage)
- **Coverage**: 100% have timestamps

#### Pass-Specific Features (including corners)
- `pass.length`: Distance of pass
- `pass.angle`: Direction angle
- `pass.height`: High/Low/Ground
- `pass.outcome`: Complete/Incomplete
- `pass.end_location`: Target [x, y]
- `pass.cross`: Boolean
- `pass.switch`: Boolean
- `pass.shot_assist`: Boolean
- `pass.goal_assist`: Boolean

#### Shot-Specific Features
- `shot.statsbomb_xg`: Expected goals value
- `shot.outcome`: Goal/Saved/Blocked/Off Target
- `shot.end_location`: [x, y, z] including height
- `shot.technique`: Volley/Half Volley/Normal
- `shot.body_part`: Foot/Head
- `shot.first_time`: Boolean
- `shot.follows_dribble`: Boolean

#### Clearance-Specific Features
- `clearance.aerial_won`: Boolean
- `clearance.head`: Boolean
- `clearance.body_part`: Foot/Head/Other

#### Contextual Features
- **team**: Always present
- **player**: Always present
- **under_pressure**: 23.3% of events
- **related_events**: Links connected events

---

## 3. WHAT YOU HAVE vs WHAT YOU MIGHT ADD

### What You Have (Raw Data)

1. **Complete Event Sequences**: Every event with type, team, player
2. **Spatial Data**: Locations for 98%+ of events
3. **Temporal Data**: Precise timestamps for all events
4. **Outcome Labels**: Pass outcomes, shot outcomes, etc.
5. **Contextual Info**: Pressure, team possession, related events
6. **Rich Event Details**: Height, angle, technique for relevant events

### What You DON'T Have (Potential Additions)

1. **Player Positions/Formations**: Only available in rare freeze frames
2. **Defensive Shape**: No formation or defensive line data
3. **Off-Ball Movement**: Only on-ball events tracked
4. **Velocity/Acceleration**: No movement speeds
5. **Player Attributes**: No height, speed, skill ratings
6. **Team Tactical State**: No formation or strategy indicators
7. **Match Context**: No score, time remaining (must calculate)

### Potential Features to Calculate/Add

From existing data, you could derive:

1. **Time-based features**:
   - Time since corner kick
   - Time between events
   - Match phase (early/middle/late)

2. **Spatial features**:
   - Distance from goal
   - Distance from corner flag
   - Angle to goal
   - Zone/region of pitch

3. **Sequence features**:
   - Number of passes since corner
   - Possession changes
   - Event type counts

4. **Team-specific features**:
   - Team possession percentage
   - Attacking vs defending team
   - Home vs away

5. **Pressure metrics**:
   - Cumulative pressure in sequence
   - Pressure zones

---

## 4. KEY INSIGHTS

### For Predicting P(Event at t+1 | Corner at t)

**Most Important Features:**
1. **Event Type** (Corner) - Determines transition probabilities
2. **Location** - Corner flag position ([120,0] vs [120,80])
3. **End Location** - Where ball is aimed
4. **Height** - High/Low/Ground pass
5. **Team** - Which team taking corner

**Transition Patterns:**
- 55% → Ball Receipt (someone receives)
- 30% → Clearance (defended immediately)
- 85% of outcomes decided within 3 events
- Shots occur quickly (events 1-5) or not at all

### For Classification Tasks

**Natural Classes Based on Data:**
1. **Immediate Clearance** (30%)
2. **Successful Receipt** (55%)
3. **Set Piece Breakdown** (15%) - Fouls, goalkeeper claims

Or for outcome prediction:
1. **Shot** (~15-20% within 5 events)
2. **Clearance** (~30-35%)
3. **Possession** (~45-50%)

---

## Summary

You have access to:
- **Complete event sequences** with types and transitions
- **Rich spatial data** (locations for nearly all events)
- **Temporal precision** (millisecond timestamps)
- **Contextual information** (teams, players, pressure)
- **Detailed event features** (pass angles, shot xG, etc.)

The transition matrix shows clear patterns:
- Corner → Ball Receipt (55%) or Clearance (30%)
- Subsequent events follow predictable patterns
- Outcomes typically resolved within 3-5 events

This raw data provides everything needed for:
- Transition probability modeling
- Outcome classification
- Spatial-temporal analysis
- Team/player-specific patterns