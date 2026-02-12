# DFL to USSF Feature Distribution Comparison Report

**Adjacency Type:** dense
**Generated:** 2026-02-11T14:25:37.778264

## 1. DFL Transformed Feature Statistics

**Total nodes:** 714
**Total edges:** 15048

### Node Features

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| x | 0.4519 | 0.4035 | 0.0000 | 0.9975 |
| y | 0.4896 | 0.1330 | 0.0007 | 0.9875 |
| vx | 0.0961 | 0.6923 | -1.0000 | 1.0000 |
| vy | -0.0597 | 0.7127 | -1.0000 | 1.0000 |
| velocity_mag | 0.0529 | 0.0559 | 0.0000 | 1.0000 |
| velocity_angle | 0.4930 | 0.2699 | 0.0074 | 1.0000 |
| dist_goal | 0.4063 | 0.2746 | 0.0044 | 0.7801 |
| angle_goal | 0.4992 | 0.0837 | 0.2520 | 0.7389 |
| dist_ball | 0.1112 | 0.1031 | 0.0000 | 0.5078 |
| angle_ball | 0.4902 | 0.2665 | 0.0009 | 1.0000 |
| attacking_team_flag | 0.5140 | 0.4998 | 0.0000 | 1.0000 |
| potential_receiver | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Edge Features

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| player_distance | 0.1426 | 0.1139 | 0.0012 | 0.6284 |
| speed_difference | 0.0000 | 0.0788 | -1.0000 | 1.0000 |
| positional_sine_angle | 0.5000 | 0.3793 | 0.0000 | 1.0000 |
| positional_cosine_angle | 0.5000 | 0.3258 | 0.0000 | 1.0000 |
| velocity_sine_angle | 0.5000 | 0.3471 | 0.0000 | 1.0000 |
| velocity_cosine_angle | 0.5869 | 0.3492 | 0.0000 | 1.0000 |

## 2. Kolmogorov-Smirnov Test Results

Tests comparing DFL transformed features vs USSF original features.
**Significance threshold:** p < 0.01

### Node Features

| Feature | KS Stat | p-value | Significant | DFL Mean | USSF Mean |
|---------|---------|---------|-------------|----------|-----------|
| x | 0.3638 | 4.7542e-85 | **YES** | 0.4519 | 0.4743 |
| y | 0.1990 | 3.2458e-25 | **YES** | 0.4896 | 0.4988 |
| vx | 0.2785 | 2.1439e-49 | **YES** | 0.0961 | 0.3981 |
| vy | 0.1611 | 1.2298e-16 | **YES** | -0.0597 | -0.0028 |
| velocity_mag | 0.6757 | 1.0869e-322 | **YES** | 0.0529 | 0.2190 |
| velocity_angle | 0.1627 | 5.8826e-17 | **YES** | 0.4930 | 0.4993 |
| dist_goal | 0.2898 | 1.5580e-53 | **YES** | 0.4063 | 0.5192 |
| angle_goal | 0.5013 | 2.9691e-166 | **YES** | 0.4992 | 0.4992 |
| dist_ball | 0.2875 | 1.1268e-52 | **YES** | 0.1112 | 0.1791 |
| angle_ball | 0.1022 | 6.0980e-07 | **YES** | 0.4902 | 0.4808 |
| attacking_team_flag | 0.0359 | 3.0817e-01 | No | 0.5140 | 0.4781 |
| potential_receiver | 0.9564 | 0.0000e+00 | **YES** | 0.0000 | 0.4870 |

### Edge Features

| Feature | KS Stat | p-value | Significant | DFL Mean | USSF Mean |
|---------|---------|---------|-------------|----------|-----------|
| player_distance | 0.2724 | 0.0000e+00 | **YES** | 0.1426 | 0.2008 |
| speed_difference | 0.2061 | 0.0000e+00 | **YES** | 0.0000 | -0.0000 |
| positional_sine_angle | 0.0436 | 2.7345e-25 | **YES** | 0.5000 | 0.5218 |
| positional_cosine_angle | 0.0223 | 6.3271e-07 | **YES** | 0.5000 | 0.5000 |
| velocity_sine_angle | 0.0260 | 2.9619e-09 | **YES** | 0.5000 | 0.4940 |
| velocity_cosine_angle | 0.3923 | 0.0000e+00 | **YES** | 0.5869 | 0.7927 |

## 3. Summary

**Node features with significant distribution mismatch:** 11/12
- x, y, vx, vy, velocity_mag, velocity_angle, dist_goal, angle_goal, dist_ball, angle_ball, potential_receiver

**Edge features with significant distribution mismatch:** 6/6
- player_distance, speed_difference, positional_sine_angle, positional_cosine_angle, velocity_sine_angle, velocity_cosine_angle

## 4. Interpretation

Features with significant distribution mismatch (p < 0.01) may transfer poorly.
This is expected due to:

1. **Different game situations:** Counterattacks (USSF) vs corner kicks (DFL)
   - Counterattacks: players spread across half-pitch, high velocities
   - Corner kicks: dense clustering in penalty area, stationary setup

2. **Velocity differences:** Corner kicks start from static positions,
   counterattacks involve fast transitions

3. **Spatial distribution:** Corner kick positions concentrated near goal,
   counterattack positions more distributed

**Recommendation:** Proceed with transfer learning experiments despite distribution
differences. The CrystalConv layers may still capture useful spatial-relational
patterns. Document performance degradation attributable to distribution shift.
