# USSF Counterattack Dataset: Feature Distribution Report
## 1. Data Structure Overview
**Data type:** `dict`
**Top-level keys:** `['delaunay', 'normal', 'dense', 'dense_ap', 'dense_dp', 'binary']`

### Adjacency Type: `delaunay`
**Sub-keys:** `['a', 'x', 'e']`
- `a`: List of 22479 items
  - First item shape: `(23, 23)`
- `x`: List of 22479 items
  - First item shape: `(23, 12)`
- `e`: List of 22479 items
  - First item shape: `(159, 6)`

### Adjacency Type: `normal`
**Sub-keys:** `['a', 'x', 'e']`
- `a`: List of 22479 items
  - First item shape: `(23, 23)`
- `x`: List of 22479 items
  - First item shape: `(23, 12)`
- `e`: List of 22479 items
  - First item shape: `(287, 6)`

### Adjacency Type: `dense`
**Sub-keys:** `['a', 'x', 'e']`
- `a`: List of 22479 items
  - First item shape: `(23, 23)`
- `x`: List of 22479 items
  - First item shape: `(23, 12)`
- `e`: List of 22479 items
  - First item shape: `(529, 6)`

### Adjacency Type: `dense_ap`
**Sub-keys:** `['a', 'x', 'e']`
- `a`: List of 22479 items
  - First item shape: `(23, 23)`
- `x`: List of 22479 items
  - First item shape: `(23, 12)`
- `e`: List of 22479 items
  - First item shape: `(408, 6)`

### Adjacency Type: `dense_dp`
**Sub-keys:** `['a', 'x', 'e']`
- `a`: List of 22479 items
  - First item shape: `(23, 23)`
- `x`: List of 22479 items
  - First item shape: `(23, 12)`
- `e`: List of 22479 items
  - First item shape: `(408, 6)`

## 2. Node Feature Distributions
**Total graphs:** 22479
**Total nodes across all graphs:** 515993
**Features per node:** 12

| Feature | Mean | Std | Min | Max | Description |
|---------|------|-----|-----|-----|-------------|
| x | 0.4743 | 0.2200 | 0.0000 | 1.0000 | X position on pitch |
| y | 0.4988 | 0.2123 | -0.1147 | 1.0000 | Y position on pitch |
| vx | 0.3981 | 0.7107 | -1.0000 | 1.0000 | X velocity component |
| vy | -0.0028 | 0.5799 | -1.0000 | 1.0000 | Y velocity component |
| velocity_mag | 0.2190 | 0.1357 | 0.0000 | 0.9950 | Speed (magnitude of velocity) |
| velocity_angle | 0.4993 | 0.2240 | 0.0000 | 1.0000 | Direction of movement (radians) |
| dist_goal | 0.5192 | 0.2048 | 0.0014 | 0.9960 | Euclidean distance to goal |
| angle_goal | 0.4992 | 0.4634 | 0.0000 | 1.0000 | Angle toward goal |
| dist_ball | 0.1791 | 0.1249 | 0.0000 | 0.7559 | Euclidean distance to ball |
| angle_ball | 0.4808 | 0.3038 | 0.0000 | 1.0000 | Angle toward ball |
| attacking_team_flag | 0.4781 | 0.4995 | 0.0000 | 1.0000 | Binary: 1=attacking, 0=defending |
| potential_receiver | 0.4870 | 0.4504 | 0.0000 | 1.0000 | Binary: potential pass receiver |

### Coordinate System Analysis
**Coordinate system:** Normalized [0, 1] pitch-relative coordinates

## 3. Edge Feature Distributions
**Total edges across all graphs:** 6427947
**Features per edge:** 6

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| player_distance | 0.1897 | 0.1210 | 0.0000 | 0.7559 |
| speed_difference | 0.0000 | 0.2006 | -0.9933 | 0.9933 |
| positional_sine_angle | 0.5401 | 0.3843 | 0.0000 | 1.0000 |
| positional_cosine_angle | 0.5000 | 0.3174 | 0.0000 | 1.0000 |
| velocity_sine_angle | 0.4969 | 0.3494 | 0.0000 | 1.0000 |
| velocity_cosine_angle | 0.7803 | 0.1711 | 0.5000 | 1.0000 |

## 4. Adjacency Matrix Analysis
**Adjacency types available:** `['delaunay', 'normal', 'dense', 'dense_ap', 'dense_dp']`

| Adjacency Type | Avg Nodes | Avg Edges | Avg Density | Description |
|----------------|-----------|-----------|-------------|-------------|
| delaunay | 23.0 | 157.8 | 0.300 | Delaunay triangulation |
| normal | 23.0 | 286.0 | 0.543 | Team-based connectivity through ball |
| dense | 23.0 | 527.0 | 1.000 | Fully connected (all-to-all) |
| dense_ap | 23.0 | 406.5 | 0.771 | Attackers fully connected + defenders |
| dense_dp | 23.0 | 406.5 | 0.771 | Defenders fully connected + attackers |

## 5. Graph Size Distribution
**Min players per graph:** 21
**Max players per graph:** 23
**Mean players per graph:** 23.0
**Std players per graph:** 0.2

| Players | Count | Percentage |
|---------|-------|------------|
| 21 | 57 | 0.3% |
| 22 | 910 | 4.0% |
| 23 | 21512 | 95.7% |

## 6. Class Balance (Labels)
**Total samples:** 22479
**Positive (successful counterattacks):** 11571 (51.5%)
**Negative (unsuccessful):** 10908 (48.5%)

## 7. DFL Feature Engineering Checklist

Based on this analysis, DFL corner kick features must match:

- [ ] Coordinate system: Check if pitch-relative [0,1] or meters
- [ ] Velocity units: Match range and scale
- [ ] Goal position convention: Same reference point
- [ ] Angle conventions: atan2(y, x) vs atan2(x, y)
- [ ] Binary flags: Same encoding (0/1)
- [ ] Handle `potential_receiver`: Drop or set to 0

---

*Generated: 2026-02-11 13:11:51*
