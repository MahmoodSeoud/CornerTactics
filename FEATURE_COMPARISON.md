# Feature Comparison: USSF Counterattacks vs CornerTactics

Comparison between USSF SSAC 23 GNN repository (counterattacks) and our CornerTactics project (corner kicks).

---

## Data Scope

| Aspect | USSF SSAC 23 | CornerTactics (Our Project) |
|--------|--------------|----------------------------|
| **Event Type** | Counterattacks | Corner Kicks |
| **Data Sources** | StatsPerform + SkillCorner | StatsBomb + SkillCorner |
| **Dataset Size** | 20,863 (balanced) / 208,009 (imbalanced) | 1,435 corners |
| **Format** | Pickle files (graphs) | CSV + Parquet |

---

## Feature Comparison

### Node Features (Player/Ball Attributes)

| Feature | USSF | CornerTactics | Status |
|---------|------|---------------|--------|
| **x, y coordinates** | ✅ | ✅ (StatsBomb 360 + SkillCorner tracking) | ✅ Have |
| **vx, vy velocity** | ✅ | ❌ | ⚠️ Need to calculate |
| **Velocity magnitude** | ✅ | ❌ | ⚠️ Need to calculate |
| **Velocity angle** | ✅ | ❌ | ⚠️ Need to calculate |
| **Distance to goal** | ✅ | ❌ | ⚠️ Need to calculate |
| **Angle with goal** | ✅ | ❌ | ⚠️ Need to calculate |
| **Distance to ball** | ✅ | ❌ | ⚠️ Need to calculate |
| **Angle with ball** | ✅ | ❌ | ⚠️ Need to calculate |
| **Attacking team flag** | ✅ | ✅ (have team info) | ✅ Have |
| **Receiver indicator** | ✅ | ❌ | ⚠️ Could derive |

**Summary**: We have raw position data but need to calculate derived features (velocities, distances, angles).

### Edge Features (Player-to-Player)

| Feature | USSF | CornerTactics | Status |
|---------|------|---------------|--------|
| **Player distance** | ✅ | ❌ | ⚠️ Need to calculate |
| **Speed difference** | ✅ | ❌ | ⚠️ Need to calculate |
| **Positional sin/cos angles** | ✅ | ❌ | ⚠️ Need to calculate |
| **Velocity sin/cos angles** | ✅ | ❌ | ⚠️ Need to calculate |

**Summary**: All edge features need to be computed from our position data.

---

## Labels/Targets

### USSF Labels (Counterattacks)

**Balanced Dataset**:
- Binary: Success (1) = reaches penalty area, Failure (0) = doesn't reach
- 50/50 split

**Imbalanced Dataset**:
- Binary: Success (1) = goal scored, Failure (0) = no goal
- ~5% success rate

### CornerTactics Labels (Our Project)

**Multi-class Outcomes**:
- Goal (1.3%)
- Shot (16.9%)
- Clearance (51.8%)
- Loss (19.5%)
- Possession (10.6%)

**Additional Metadata**:
- `outcome_category` - categorical label
- `outcome_type` - detailed classification
- `same_team` - boolean (same team as corner taker)
- `time_to_outcome` - seconds after corner
- `outcome_location` - (x, y) coordinates
- `xthreat_delta` - threat change value

**Comparison**:
- ✅ We have richer outcome classification (5 categories vs binary)
- ✅ We have temporal information (time_to_outcome)
- ✅ We have spatial outcome data (outcome_location)
- ⚠️ Could create binary labels for specific tasks (e.g., "Shot or Goal" vs "Other")

---

## What We Have

### StatsBomb Data (1,118 corners)
```
Available Features:
- Player positions (x, y) at corner moment (360° freeze frame)
- Team affiliation (attacking/defending)
- Player roles/positions
- Match context (teams, competition, date)
- Outcome labels (Goal/Shot/Clearance/Loss/Possession)
- Time to outcome
- Outcome location
- xThreat delta

Missing:
- Velocities (static freeze frame)
- Continuous tracking (only single frame)
```

### SkillCorner Data (317 corners)
```
Available Features:
- Continuous tracking data (10 fps)
- Player positions over time
- Dynamic events with end_type
- Team affiliation
- Outcome labels (Shot/Clearance/Loss/Possession)

Missing:
- Pre-computed velocities (can calculate from tracking)
- Pre-computed distances/angles (can calculate)
```

---

## Gap Analysis

### What We Need to Add for Phase 2 (Graph Construction)

#### High Priority - Core Features
1. **Distance to goal** - Calculate from player (x,y) to goal position
2. **Angle to goal** - Calculate vector angle player → goal
3. **Distance to ball** - Calculate from player (x,y) to ball position
4. **Player-to-player distances** - Edge feature for all connections

#### Medium Priority - Motion Features
5. **Velocities (vx, vy)** - Calculate from SkillCorner tracking (frame differences)
6. **Velocity magnitude** - sqrt(vx² + vy²)
7. **Velocity angle** - atan2(vy, vx)

#### Lower Priority - Advanced
8. **Speed differences** - Edge feature |v_player1 - v_player2|
9. **Positional angles** - sin/cos angles between players
10. **Velocity angles** - sin/cos angles between velocity vectors
11. **Receiver indicators** - Who might receive the corner kick

---

## Implementation Plan for Phase 2

### Step 1: Feature Engineering Module
Create `src/feature_engineering.py`:
- `calculate_distances_to_goal(positions)`
- `calculate_angles_to_goal(positions)`
- `calculate_distances_to_ball(positions, ball_pos)`
- `calculate_player_distances(positions)` (edge features)
- `calculate_velocities(tracking_data)` (for SkillCorner only)

### Step 2: Graph Construction Module
Create `src/graph_constructor.py`:
- `build_corner_graph(corner_data, features)` → NetworkX graph
- Support multiple adjacency types (normal, delaunay, dense)
- Node features: positions + derived features
- Edge features: distances + angles

### Step 3: Dataset Export
Convert to GNN-ready format:
- Option A: PyTorch Geometric format
- Option B: Pickle format (like USSF)
- Option C: DGL format

---

## Validation Strategy

Since USSF doesn't have corner kick data, we validate by:

1. **Feature Calculations**: Test on sample data, verify distances/angles make sense
2. **Graph Structure**: Visualize graphs for sample corners, ensure connectivity is reasonable
3. **Distribution Checks**: Compare feature distributions to expected ranges
4. **Ablation Study**: Test GNN with different feature subsets to identify most important

---

## Key Differences: Counterattacks vs Corners

| Aspect | Counterattacks (USSF) | Corner Kicks (Us) |
|--------|----------------------|-------------------|
| **Spatial Dynamics** | Dynamic, fast-moving | Static → dynamic transition |
| **Player Positions** | Spread across pitch | Concentrated near goal box |
| **Key Features** | Speed, direction | Positioning, spacing |
| **Prediction Task** | Will it succeed? | What outcome? (multi-class) |
| **Time Horizon** | ~10-20 seconds | ~2-5 seconds |
| **Success Rate** | ~5% goals | ~1.3% goals, ~18% shots |

---

## Conclusion

✅ **What we have**:
- Raw position data (StatsBomb 360 + SkillCorner tracking)
- Rich outcome labels (5 categories)
- Temporal and spatial outcome metadata

⚠️ **What we need to build for Phase 2**:
- Derived features (distances, angles, velocities)
- Graph construction pipeline
- Edge feature calculations

💡 **Advantage over USSF**:
- Multi-class labels (not just binary success/failure)
- Richer outcome context (time, location, xThreat)
- Combination of static + tracking data sources

The USSF repository provides a great **methodology** for graph construction, but we need to implement our own feature engineering for corner kicks since they don't have corner kick data.
