# USSF Counterattack Dataset: Feature Distribution Report

**Phase 0 Deliverable - Transfer Learning Experiment**

## Executive Summary

Key findings for DFL corner kick feature engineering:

| Aspect | USSF Convention | DFL Action Required |
|--------|-----------------|---------------------|
| Coordinates | Pitch-relative [0, 1] | Normalize DFL meters to [0, 1] |
| Velocities | Normalized to [-1, 1] | Divide by max velocity (~12 m/s?) |
| Velocity magnitude | Independently normalized [0, 1] | Separate normalization from vx, vy |
| Angles | Transformed to [0, 1] | Use (angle + π) / (2π) |
| Distances | Normalized by pitch diagonal | Use same max_dist |
| `potential_receiver` | **NOT purely binary** (has 0.1 values) | Consider dropping feature |

---

## 1. Data Structure Overview

**Data type:** `dict`
**Top-level keys:** `['delaunay', 'normal', 'dense', 'dense_ap', 'dense_dp', 'binary']`

### Adjacency Types Available

| Type | Description | Avg Density |
|------|-------------|-------------|
| normal | Team-based connectivity through ball | 0.543 |
| delaunay | Delaunay triangulation | 0.300 |
| dense | Fully connected (all-to-all) | 1.000 |
| dense_ap | Attackers fully connected + defenders | 0.771 |
| dense_dp | Defenders fully connected + attackers | 0.771 |

Each adjacency type contains:
- `x`: Node features - List of 22,479 graphs, shape (N_players, 12)
- `a`: Adjacency matrices - List of 22,479 graphs, shape (N_players, N_players)
- `e`: Edge features - List of 22,479 graphs, shape (N_edges, 6)

---

## 2. Node Feature Distributions (12 features)

| Feature | Mean | Std | Min | Max | Notes |
|---------|------|-----|-----|-----|-------|
| x | 0.4743 | 0.2200 | 0.0000 | 1.0000 | Pitch-relative X |
| y | 0.4988 | 0.2123 | -0.1147 | 1.0000 | Pitch-relative Y (has negatives!) |
| vx | 0.3981 | 0.7107 | -1.0000 | 1.0000 | Normalized velocity X |
| vy | -0.0028 | 0.5799 | -1.0000 | 1.0000 | Normalized velocity Y |
| velocity_mag | 0.2190 | 0.1357 | 0.0000 | 0.9950 | **NOT sqrt(vx²+vy²)** |
| velocity_angle | 0.4993 | 0.2240 | 0.0000 | 1.0000 | Normalized angle |
| dist_goal | 0.5192 | 0.2048 | 0.0014 | 0.9960 | Normalized distance |
| angle_goal | 0.4992 | 0.4634 | 0.0000 | 1.0000 | Normalized angle |
| dist_ball | 0.1791 | 0.1249 | 0.0000 | 0.7559 | Max < 1.0 (counterattack context) |
| angle_ball | 0.4808 | 0.3038 | 0.0000 | 1.0000 | Normalized angle |
| attacking_team_flag | 0.4781 | 0.4995 | 0.0000 | 1.0000 | Binary 0/1 |
| potential_receiver | 0.4870 | 0.4504 | 0.0000 | 1.0000 | **Has intermediate values (0.1)!** |

### Critical Observations

1. **Coordinate System:** Pitch-relative [0, 1] coordinates
   - y has slight negative values (-0.1147) suggesting extrapolation beyond pitch bounds

2. **Velocity Normalization:**
   - vx, vy are independently normalized to [-1, 1]
   - velocity_mag is **NOT** computed from normalized vx, vy (max diff = 0.999)
   - velocity_mag is independently normalized to [0, 1]

3. **Angle Normalization:**
   - All angles transformed to [0, 1] range
   - Likely formula: `(atan2(y, x) + π) / (2π)`

4. **Distance Normalization:**
   - dist_goal nearly spans [0, 1] - normalized by pitch diagonal
   - dist_ball max = 0.7559 - counterattacks are spatially constrained

5. **potential_receiver is NOT purely binary:**
   - Contains values: 0.0, 0.1, 1.0
   - ~48.7% of players marked as potential receivers
   - **Recommendation:** Drop this feature (retrain backbone with 11 features)

### Velocity Magnitude Percentiles

| Percentile | Value |
|------------|-------|
| 50th | 0.198 |
| 75th | 0.300 |
| 90th | 0.408 |
| 95th | 0.479 |
| 99th | 0.604 |

---

## 3. Edge Feature Distributions (6 features)

**Total edges across all graphs:** 6,427,947

| Feature | Mean | Std | Min | Max | Notes |
|---------|------|-----|-----|-----|-------|
| player_distance | 0.1897 | 0.1210 | 0.0000 | 0.7559 | Normalized |
| speed_difference | 0.0000 | 0.2006 | -0.9933 | 0.9933 | Symmetric around 0 |
| positional_sine_angle | 0.5401 | 0.3843 | 0.0000 | 1.0000 | (sin+1)/2 |
| positional_cosine_angle | 0.5000 | 0.3174 | 0.0000 | 1.0000 | (cos+1)/2 |
| velocity_sine_angle | 0.4969 | 0.3494 | 0.0000 | 1.0000 | (sin+1)/2 |
| velocity_cosine_angle | 0.7803 | 0.1711 | 0.5000 | 1.0000 | **Always ≥ 0.5** |

### Edge Feature Observations

- **velocity_cosine_angle min = 0.5**: Implies only acute angles between velocity vectors, or (cos+1)/2 with cos ≥ 0
- **Sine/cosine transformation:** `(val + 1) / 2` maps [-1, 1] → [0, 1]

---

## 4. Graph Statistics

### Graph Count
- **Total graphs:** 22,479
- **Expected (USSF paper):** 20,863
- **Difference:** +1,616 (possibly multiple frames per counterattack sequence)

### Players per Graph

| Metric | Value |
|--------|-------|
| Min | 21 |
| Max | 23 |
| Mean | 23.0 |
| Std | 0.2 |

| Players | Count | % |
|---------|-------|---|
| 21 | 57 | 0.3% |
| 22 | 910 | 4.0% |
| 23 | 21,512 | 95.7% |

---

## 5. Class Balance

| Label | Count | Percentage |
|-------|-------|------------|
| Positive (successful) | 11,571 | 51.5% |
| Negative (unsuccessful) | 10,908 | 48.5% |
| **Total** | **22,479** | |

Well-balanced dataset (unlike DFL corners with high class imbalance).

---

## 6. Inferred Normalization Formulas

Based on the data distributions, USSF likely uses:

### Node Features
```python
# Coordinates (pitch-relative)
x_norm = x / pitch_length  # → [0, 1]
y_norm = y / pitch_width   # → [0, 1] (with slight overflow)

# Velocities (normalized by max sprint speed)
vx_norm = clip(vx / max_velocity, -1, 1)
vy_norm = clip(vy / max_velocity, -1, 1)
velocity_mag_norm = |v| / max_velocity  # independent normalization

# Angles (normalized to [0, 1])
velocity_angle_norm = (atan2(vy, vx) + π) / (2π)
angle_goal_norm = (atan2(goal_y - y, goal_x - x) + π) / (2π)
angle_ball_norm = (atan2(ball_y - y, ball_x - x) + π) / (2π)

# Distances (normalized by pitch diagonal or max_dist)
dist_goal_norm = euclidean(pos, goal) / max_dist
dist_ball_norm = euclidean(pos, ball) / max_dist

# Binary flags
attacking_team_flag = 1 if attacking else 0
potential_receiver = ??? (model-derived, not simple binary)
```

### Edge Features
```python
# Between players i and j
player_distance = euclidean(pos_i, pos_j) / max_dist
speed_difference = (|v_i| - |v_j|) / max_velocity

# Angle between position vectors
pos_angle = atan2(y_j - y_i, x_j - x_i)
positional_sine_angle = (sin(pos_angle) + 1) / 2
positional_cosine_angle = (cos(pos_angle) + 1) / 2

# Angle between velocity vectors
vel_angle = angle_between(v_i, v_j)
velocity_sine_angle = (sin(vel_angle) + 1) / 2
velocity_cosine_angle = (cos(vel_angle) + 1) / 2  # ≥ 0.5 suggests acute angles only
```

---

## 7. DFL Feature Engineering Checklist

Based on this analysis, DFL corner kick features must:

- [x] **Analyzed:** USSF coordinate system is pitch-relative [0, 1]
- [ ] Determine DFL pitch dimensions (likely 105m × 68m)
- [ ] Normalize DFL coordinates to [0, 1] range
- [ ] Estimate USSF max_velocity (likely 10-12 m/s based on vel_mag distribution)
- [ ] Normalize DFL velocities to [-1, 1] using same max_velocity
- [ ] Compute velocity_mag independently (NOT from normalized vx, vy)
- [ ] Apply angle normalization: (angle + π) / (2π)
- [ ] Apply edge angle transformation: (sin/cos + 1) / 2
- [ ] Decide `potential_receiver` handling:
  - **Recommended:** Drop from both datasets, retrain backbone with 11 features
  - Alternative: Set to 0 for all corner kick players
- [ ] Run KS tests to compare USSF vs DFL distributions after engineering

---

## 8. Risk Assessment for Transfer

| Feature | Transfer Risk | Reasoning |
|---------|---------------|-----------|
| x, y | Low | Same coordinate system achievable |
| vx, vy | Medium | Need to match max_velocity constant |
| velocity_mag | Medium | Independent normalization required |
| angles | Low | Clear transformation formula |
| distances | Low | Clear normalization method |
| attacking_team_flag | Low | Direct mapping |
| potential_receiver | **High** | Not purely binary, unclear definition |

**Recommendation:** Drop `potential_receiver` feature and retrain USSF backbone with 11 node features. This removes uncertainty and provides a cleaner transfer experiment.

---

*Generated: 2026-02-11*
*Data source: [USSF SSAC 23 Soccer GNN](https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn)*
