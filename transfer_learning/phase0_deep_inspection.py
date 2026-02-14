#!/usr/bin/env python3
"""
Phase 0 Deep Inspection: Understand USSF normalization schemes
"""
import pickle
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "combined.pkl"

with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

# Get node features from normal adjacency type
nodes = np.vstack(data['normal']['x'])

print("="*60)
print("USSF NORMALIZATION ANALYSIS")
print("="*60)

feature_names = [
    'x', 'y', 'vx', 'vy', 'velocity_mag', 'velocity_angle',
    'dist_goal', 'angle_goal', 'dist_ball', 'angle_ball',
    'attacking_team_flag', 'potential_receiver'
]

print("\n### Feature Range Analysis ###\n")
for i, name in enumerate(feature_names):
    vals = nodes[:, i]
    print(f"{name:20s}: min={vals.min():8.4f}, max={vals.max():8.4f}, "
          f"mean={vals.mean():8.4f}, std={vals.std():8.4f}")

print("\n### Key Observations ###\n")

# Position analysis
x = nodes[:, 0]
y = nodes[:, 1]
print(f"Position (x,y): Both in [0,1] range - pitch-relative coordinates")
print(f"  x range: [{x.min():.4f}, {x.max():.4f}]")
print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")

# y has a slightly negative minimum - might be extrapolation
if y.min() < 0:
    print(f"  NOTE: y has negative values ({y.min():.4f}) - slight extrapolation beyond pitch?")

# Velocity analysis
vx = nodes[:, 2]
vy = nodes[:, 3]
vel_mag = nodes[:, 4]
vel_angle = nodes[:, 5]

print(f"\nVelocity (vx, vy): Range [-1, 1] - normalized velocities")
print(f"  vx range: [{vx.min():.4f}, {vx.max():.4f}]")
print(f"  vy range: [{vy.min():.4f}, {vy.max():.4f}]")
print(f"  vel_mag range: [{vel_mag.min():.4f}, {vel_mag.max():.4f}]")
print(f"  vel_angle range: [{vel_angle.min():.4f}, {vel_angle.max():.4f}]")

# Check if vel_mag = sqrt(vx^2 + vy^2)
computed_mag = np.sqrt(vx**2 + vy**2)
print(f"\n  Checking: vel_mag == sqrt(vx^2 + vy^2)?")
print(f"  Max difference: {np.max(np.abs(vel_mag - computed_mag)):.6f}")
if np.max(np.abs(vel_mag - computed_mag)) < 0.01:
    print(f"  MATCH: vel_mag is computed from normalized vx, vy")
else:
    print(f"  NO MATCH: vel_mag uses different normalization")
    # Check if vel_mag was normalized separately
    print(f"  Likely: vel_mag normalized to [0,1] independently")

# Angle analysis - check if normalized to [0,1] from [-pi, pi] or [0, 2pi]
print(f"\nAngle normalization:")
print(f"  vel_angle in [0,1] suggests: (angle + pi) / (2*pi) or angle / (2*pi)")
print(f"  angle_goal range: [{nodes[:, 7].min():.4f}, {nodes[:, 7].max():.4f}]")
print(f"  angle_ball range: [{nodes[:, 9].min():.4f}, {nodes[:, 9].max():.4f}]")

# Distance analysis
dist_goal = nodes[:, 6]
dist_ball = nodes[:, 8]
print(f"\nDistance normalization:")
print(f"  dist_goal range: [{dist_goal.min():.4f}, {dist_goal.max():.4f}]")
print(f"  dist_ball range: [{dist_ball.min():.4f}, {dist_ball.max():.4f}]")
print(f"  dist_goal max ~1.0 suggests normalized by pitch diagonal (~120m if 105x68)")

# Binary flags
att_flag = nodes[:, 10]
pot_recv = nodes[:, 11]
print(f"\nBinary flags:")
print(f"  attacking_team_flag unique values: {np.unique(att_flag)}")
print(f"  potential_receiver unique values: {np.unique(pot_recv)[:5]}...")  # First few unique values
print(f"  potential_receiver mean: {pot_recv.mean():.4f}")
print(f"  NOTE: potential_receiver has ~48.7% ones - actually used, not placeholder")

# Edge features
print("\n" + "="*60)
print("EDGE FEATURE ANALYSIS")
print("="*60)

edges = np.vstack(data['normal']['e'])
edge_names = [
    'player_distance', 'speed_difference',
    'positional_sine_angle', 'positional_cosine_angle',
    'velocity_sine_angle', 'velocity_cosine_angle'
]

for i, name in enumerate(edge_names):
    vals = edges[:, i]
    print(f"{name:25s}: min={vals.min():8.4f}, max={vals.max():8.4f}, "
          f"mean={vals.mean():8.4f}, std={vals.std():8.4f}")

print("\n### Edge Feature Observations ###\n")
print("player_distance: [0, 0.76] - normalized by pitch size")
print("speed_difference: [-1, 1] - difference in normalized velocities")
print("Sine/cosine angles: [0, 1] - normalized from [-1, 1]?")
print("  - Standard sine/cosine in [-1, 1], but these are [0, 1]")
print("  - Likely: (sin + 1) / 2 or similar transformation")

# Check velocity_cosine distribution
vel_cos = edges[:, 5]
print(f"\nvelocity_cosine_angle min: {vel_cos.min():.4f} (always >= 0.5)")
print(f"  This suggests: only acute angles considered, or (cos+1)/2 transform")

print("\n" + "="*60)
print("CRITICAL NORMALIZATION FORMULAS FOR DFL")
print("="*60)

print("""
Based on analysis, USSF uses these transformations:

NODE FEATURES:
  x, y         := pos / pitch_dims  (pitch-relative [0,1])
  vx, vy       := velocity / max_velocity  (capped to [-1,1])
  velocity_mag := |v| / max_velocity  (or computed from normalized vx,vy?)
  velocity_angle := (atan2(vy, vx) + pi) / (2*pi)  [0,1]
  dist_goal    := euclidean(pos, goal) / max_dist  [0,1]
  angle_goal   := (atan2(goal_y-y, goal_x-x) + pi) / (2*pi)  [0,1]
  dist_ball    := euclidean(pos, ball) / max_dist  [0,1]
  angle_ball   := (atan2(ball_y-y, ball_x-x) + pi) / (2*pi)  [0,1]
  attacking_team_flag := 0 or 1
  potential_receiver  := 0 or 1 (actually computed, ~48.7% are receivers)

EDGE FEATURES:
  player_distance := euclidean(p1, p2) / max_dist
  speed_difference := (|v1| - |v2|) / max_velocity  (or normalized)
  positional_sine_angle := (sin(angle) + 1) / 2
  positional_cosine_angle := (cos(angle) + 1) / 2
  velocity_sine_angle := (sin(angle) + 1) / 2
  velocity_cosine_angle := (cos(angle) + 1) / 2

FOR DFL CORNERS:
  - Need to determine max_velocity used in USSF (likely 10-12 m/s sprint speed)
  - Need to use same pitch dimension normalization
  - angles all shifted to [0,1] via (val + pi) / (2*pi) or (val + 1) / 2
""")

# Estimate max velocity used
# If vx in [-1,1] represents real velocities, and typical sprint is 10m/s
# Let's check velocity magnitude distribution
print("\n### Velocity Magnitude Distribution ###")
percentiles = [50, 75, 90, 95, 99]
for p in percentiles:
    print(f"  {p}th percentile: {np.percentile(vel_mag, p):.4f}")

print("\n### Graph Count Discrepancy ###")
print(f"Total graphs in data: {len(data['normal']['x'])}")
print(f"Expected from USSF paper: 20,863")
print(f"Difference: {len(data['normal']['x']) - 20863}")
print("Possible explanation: Multiple frames per counterattack sequence")
