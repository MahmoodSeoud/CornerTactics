# Additional Features for Multi-Class Corner Outcome Prediction

## Context
USSF GNN focused on binary prediction (counterattack success: yes/no).
Our task is multi-class: **Goal / Shot / Clearance / Loss / Possession**

Each outcome type has different spatial signatures that need specific features.

---

## 1. Receiver Indicator (from USSF)

**Definition**: Player most likely to make first contact with the ball

**Calculation**:
```python
# For each player, calculate probability of being receiver:
- Distance to ball landing zone (end_x, end_y)
- Velocity toward ball
- Positioning advantage over markers
# Flag the top 1-2 players as potential receivers (binary)
```

**Why it matters**:
- First contact often determines outcome
- Receiver's positioning predicts shot opportunity vs clearance

---

## 2. Corner-Specific Features

### 2.1 Ball Trajectory Features

| Feature | Description | Why Important |
|---------|-------------|---------------|
| `ball_landing_zone_x` | Where corner is aimed (x coord) | In-swinger vs out-swinger |
| `ball_landing_zone_y` | Where corner is aimed (y coord) | Near post vs far post |
| `distance_to_goal_line` | How deep the corner is | Deep corners = more dangerous |
| `distance_to_6yard_box` | Distance to 6-yard box center | Prime scoring zone |
| `inswinger_flag` | Curving toward goal (binary) | Higher goal probability |

**Calculation**:
```python
landing_zone = (end_x, end_y)  # from StatsBomb/SkillCorner
goal_center = (120, 40)  # StatsBomb coordinates
distance_to_goal_line = 120 - landing_zone[0]
inswinger = determine_curve_direction(location_x, location_y, end_x, end_y)
```

### 2.2 Spatial Zone Occupancy

Divide penalty area into tactical zones:

```
Penalty Area Zones (StatsBomb 120x80):
┌─────────────────────────────┐
│   Near Post    │  Far Post   │  (y > 40: far post)
│   (102-108)    │  (102-108)  │  (y < 40: near post)
├─────────────────────────────┤
│  Penalty Spot  │   6-Yard    │
│   (108-114)    │  (114-120)  │
└─────────────────────────────┘
```

| Feature | Description | Outcome Signal |
|---------|-------------|----------------|
| `attackers_near_post` | # attackers in near post zone | Shot opportunity |
| `defenders_near_post` | # defenders in near post zone | Clearance likely |
| `attackers_6yard` | # attackers in 6-yard box | High danger |
| `defenders_6yard` | # defenders in 6-yard box | Defensive strength |
| `zone_advantage` | Attackers - Defenders in target zone | Net advantage |

### 2.3 Marking Relationships

| Feature | Description | Why Important |
|---------|-------------|---------------|
| `marked_flag` | Player is closely marked (binary) | Reduces effectiveness |
| `marker_distance` | Distance to nearest opponent | Tighter marking = harder |
| `unmarked_attackers` | # of free attackers | Dangerous situation |
| `defensive_coverage_ratio` | Defenders/Attackers in box | Overall balance |

**Calculation**:
```python
for attacker in attacking_players:
    nearest_defender = min(distance(attacker, d) for d in defending_players)
    attacker.marked = (nearest_defender < 2.0)  # 2 meters threshold
    attacker.marker_distance = nearest_defender
```

---

## 3. Outcome-Specific Indicators

### 3.1 Shot Opportunity Features

These predict **Shot** or **Goal** outcomes:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `open_shooting_lanes` | # of clear paths to goal | More lanes = more shots |
| `shooting_angle` | Angle to goal from landing zone | Wider angle = better |
| `space_around_receiver` | Space within 3m of receiver | Room to shoot |
| `attackers_in_scoring_zone` | Attackers within 10m of goal | Multiple threats |

### 3.2 Clearance Prediction Features

These predict **Clearance** outcomes:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `defenders_on_ball_path` | Defenders between ball and goal | Interception likely |
| `first_contact_defender` | Defender closest to landing zone | Early clearance |
| `defensive_line_depth` | Avg y-position of defenders | Deep line = harder to clear |
| `goalkeeper_positioning` | GK distance from goal line | GK punch/catch |

### 3.3 Possession/Loss Features

These predict **Possession** or **Loss** (second ball):

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `players_at_edge_of_box` | Players at 18-yard line | Second ball winners |
| `space_behind_defense` | Open space for counter | Loss → counter danger |
| `midfield_presence` | Players outside box | Possession retention |

---

## 4. Defensive Organization Features

| Feature | Description | Why Important |
|---------|-------------|---------------|
| `defensive_compactness` | Std dev of defender positions | Tight = harder to score |
| `defensive_line_x` | Avg x-position of defenders | Defensive depth |
| `goalkeeper_x` | GK position on x-axis | GK involvement |
| `zonal_marking_indicator` | Defenders spread evenly (vs clustered) | Marking style |

**Calculation**:
```python
import numpy as np
defender_positions = [(x, y) for x, y in defending_players]
compactness = np.std(defender_positions)  # Lower = more compact
defensive_line_x = np.mean([x for x, y in defender_positions])
```

---

## 5. Attacking Pattern Features

| Feature | Description | Why Important |
|---------|-------------|---------------|
| `num_runners` | Attackers moving toward goal | More runs = confusion |
| `decoy_runners` | Attackers running away from ball | Create space |
| `stacked_attackers` | Attackers in same zone | Overload strategy |
| `short_corner_setup` | Player near corner flag | Short corner likely |

---

## 6. Temporal/Contextual Features

| Feature | Description | Why Important |
|---------|-------------|---------------|
| `match_minute` | When corner occurs | Late game = desperation |
| `score_difference` | Current goal difference | Affects urgency |
| `team_corner_count` | # corners so far | Corner routine variation |
| `recent_outcome` | Previous corner outcome | Momentum/adjustment |

---

## Feature Prioritization

### Tier 1: Must Have (Most Predictive)
1. ✅ Receiver indicator
2. ✅ Ball landing zone (x, y)
3. ✅ Zone advantage (attackers - defenders)
4. ✅ Marking relationships (marked_flag, marker_distance)
5. ✅ Defenders on ball path
6. ✅ Shooting angle from landing zone

### Tier 2: High Value
7. ✅ Defensive compactness
8. ✅ Open shooting lanes
9. ✅ Players at edge of box
10. ✅ Goalkeeper positioning
11. ✅ First contact (attacker vs defender)
12. ✅ Inswinger flag

### Tier 3: Nice to Have
13. ⚠️ Num runners (if tracking available)
14. ⚠️ Match context (minute, score)
15. ⚠️ Zonal marking indicator
16. ⚠️ Recent outcome history

---

## Updated Feature Count

**Original Plan**: 12 node features + 6 edge features = 18 total

**Enhanced Multi-Class Plan**:

### Node Features (22 dimensions)
**Spatial (6)**:
1. x, y coordinates (normalized)
2. distance_to_goal
3. distance_to_ball_target
4. distance_to_6yard_box
5. zone_id (near post / far post / edge)

**Kinematic (4)** (SkillCorner only):
6. velocity_x, velocity_y
7. velocity_magnitude, velocity_angle

**Contextual (8)**:
10. angle_to_goal
11. angle_to_ball
12. team_flag
13. in_penalty_box
14. **receiver_indicator** ← NEW
15. **marked_flag** ← NEW
16. **marker_distance** ← NEW
17. **shooting_angle** ← NEW

**Density (4)**:
18. num_players_within_5m
19. local_density_score
20. **zone_advantage** ← NEW
21. **defenders_on_path** ← NEW

### Edge Features (6 dimensions - unchanged)
1. Distance between players
2. Relative velocity
3. Angle differences (sine/cosine pairs)

### Graph-Level Features (8 dimensions)
Global features for the entire graph:
1. **ball_landing_x, ball_landing_y** ← NEW
2. **inswinger_flag** ← NEW
3. **defensive_compactness** ← NEW
4. **total_attackers_in_box** ← NEW
5. **total_defenders_in_box** ← NEW
6. **goalkeeper_distance** ← NEW
7. **unmarked_attackers_count** ← NEW

---

## Implementation Strategy

### Phase 2.1 Enhancement

Update `src/feature_engineering.py` to calculate:

```python
class CornerFeatureEngineer:
    def __init__(self):
        self.goal_position = (120, 40)  # StatsBomb coords
        self.six_yard_box = (114, 30, 50)  # (x_min, y_min, y_max)

    def calculate_node_features(self, player_positions, ball_target, team_flags):
        """Calculate 22-dimensional node features"""
        features = []
        for player in player_positions:
            spatial = self._spatial_features(player, ball_target)
            kinematic = self._kinematic_features(player)  # if tracking
            contextual = self._contextual_features(player, ball_target, team_flags)
            density = self._density_features(player, player_positions)
            features.append(spatial + kinematic + contextual + density)
        return np.array(features)

    def calculate_receiver_indicator(self, player_positions, ball_target):
        """Flag top 1-2 players most likely to receive"""
        distances = [euclidean(p, ball_target) for p in player_positions]
        receiver_scores = self._compute_receiver_probability(distances, velocities)
        return (receiver_scores > threshold).astype(int)

    def calculate_marking_relationships(self, attacking_players, defending_players):
        """Compute marking relationships"""
        marking = []
        for attacker in attacking_players:
            nearest_defender_dist = min([euclidean(attacker, d) for d in defending_players])
            marking.append({
                'marked': nearest_defender_dist < 2.0,
                'marker_distance': nearest_defender_dist
            })
        return marking
```

---

## Expected Impact on Multi-Class Prediction

| Outcome | Key Discriminative Features |
|---------|----------------------------|
| **Goal** | Receiver indicator + shooting angle + unmarked attackers + inswinger |
| **Shot** | Shooting angle + open lanes + zone advantage + space around receiver |
| **Clearance** | Defenders on path + first contact defender + defensive compactness |
| **Loss** | Players at edge + space behind defense + defensive line depth |
| **Possession** | Midfield presence + zone advantage negative + no clear threat |

---

## Next Steps

1. ✅ Add these features to Phase 2.1 in CORNER_GNN_PLAN.md
2. ✅ Implement `feature_engineering.py` with enhanced calculations
3. ✅ Test feature distributions on sample corners
4. ✅ Validate feature importance with ablation studies
5. ✅ Compare performance: 12 features vs 22 features vs 22+graph-level

---

*This enhancement builds on USSF's proven GNN approach while adding corner-specific intelligence for multi-class outcome prediction.*
