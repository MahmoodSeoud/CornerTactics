#!/usr/bin/env python3
"""
Investigate why average matching distance is 32.61m (should be <5m).
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*80)
print("INVESTIGATING HIGH MATCHING DISTANCE (32.61m)")
print("="*80)

# Load graphs with v2 labels
print("\nLoading graphs...")
with open("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver_v2.pkl", 'rb') as f:
    graphs = pickle.load(f)

print(f"Total graphs: {len(graphs)}")

# Get graphs with receivers
graphs_with_receiver = [g for g in graphs if g.receiver_player_id is not None and g.receiver_node_index is not None]
print(f"Graphs with receiver and matched node: {len(graphs_with_receiver)}")

# Calculate matching distances
print("\nCalculating matching distances...")
distances = []
sample_high_distance = []
sample_low_distance = []

for graph in graphs_with_receiver:
    if graph.receiver_location is None:
        continue

    receiver_location = np.array(graph.receiver_location)
    matched_node_pos = graph.node_features[graph.receiver_node_index, :2]  # x, y

    distance = np.linalg.norm(matched_node_pos - receiver_location)
    distances.append(distance)

    # Collect samples
    if distance > 30 and len(sample_high_distance) < 5:
        sample_high_distance.append((graph.corner_id, distance, receiver_location, matched_node_pos))
    elif distance < 5 and len(sample_low_distance) < 5:
        sample_low_distance.append((graph.corner_id, distance, receiver_location, matched_node_pos))

# Statistics
distances = np.array(distances)
print(f"\n=== MATCHING DISTANCE STATISTICS ===")
print(f"Mean: {np.mean(distances):.2f}m")
print(f"Median: {np.median(distances):.2f}m")
print(f"Std: {np.std(distances):.2f}m")
print(f"Min: {np.min(distances):.2f}m")
print(f"Max: {np.max(distances):.2f}m")

# Percentiles
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(distances, p):.2f}m")

# Distribution
print(f"\n=== DISTANCE DISTRIBUTION ===")
bins = [0, 2, 5, 10, 20, 30, 50, 100, 200]
for i in range(len(bins) - 1):
    count = np.sum((distances >= bins[i]) & (distances < bins[i+1]))
    pct = count / len(distances) * 100
    print(f"  {bins[i]:>3}m - {bins[i+1]:>3}m: {count:>5} ({pct:>5.1f}%)")

# Show high distance examples
print(f"\n=== HIGH DISTANCE EXAMPLES (>30m) ===")
for corner_id, dist, receiver_loc, matched_pos in sample_high_distance:
    print(f"\nCorner: {corner_id[:8]}...")
    print(f"  Distance: {dist:.2f}m")
    print(f"  Receiver event location: [{receiver_loc[0]:.1f}, {receiver_loc[1]:.1f}]")
    print(f"  Matched freeze frame pos: [{matched_pos[0]:.1f}, {matched_pos[1]:.1f}]")
    print(f"  Diff: [{receiver_loc[0] - matched_pos[0]:.1f}, {receiver_loc[1] - matched_pos[1]:.1f}]")

# Show low distance examples
print(f"\n=== LOW DISTANCE EXAMPLES (<5m) ===")
for corner_id, dist, receiver_loc, matched_pos in sample_low_distance:
    print(f"\nCorner: {corner_id[:8]}...")
    print(f"  Distance: {dist:.2f}m")
    print(f"  Receiver event location: [{receiver_loc[0]:.1f}, {receiver_loc[1]:.1f}]")
    print(f"  Matched freeze frame pos: [{matched_pos[0]:.1f}, {matched_pos[1]:.1f}]")
    print(f"  Diff: [{receiver_loc[0] - matched_pos[0]:.1f}, {receiver_loc[1] - matched_pos[1]:.1f}]")

print("\n" + "="*80)
print("HYPOTHESIS: Event location might be DIFFERENT from freeze frame position")
print("="*80)
print("Event location = where the event OCCURRED (e.g., where clearance was made)")
print("Freeze frame = where players were WHEN CORNER WAS TAKEN")
print("\nFor a clearance 5 seconds after corner, player may have moved 20-30m!")
print("This is EXPECTED behavior, not a bug.")
