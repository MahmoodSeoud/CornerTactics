#!/usr/bin/env python3
"""
Generate comprehensive dataset documentation with statistics and distributions.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from collections import Counter
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*80)
print("GENERATING DATASET DOCUMENTATION")
print("="*80)

# Load graphs (100% coverage)
print("\nLoading graphs...")
with open("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl", 'rb') as f:
    graphs = pickle.load(f)

print(f"Total graphs: {len(graphs)}")

# Initialize statistics dictionary
stats = {
    'overview': {},
    'receiver_distribution': {},
    'outcome_distribution': {},
    'team_distribution': {},
    'temporal_augmentation': {},
    'graph_structure': {},
    'spatial_distribution': {},
}

print("\n" + "="*80)
print("1. DATASET OVERVIEW")
print("="*80)

stats['overview']['total_graphs'] = len(graphs)
stats['overview']['graphs_with_receivers'] = sum(1 for g in graphs if g.receiver_player_id is not None)
stats['overview']['receiver_coverage_pct'] = (stats['overview']['graphs_with_receivers'] / len(graphs)) * 100

# Extract unique base corners
base_corners = set()
for g in graphs:
    base_id = g.corner_id.split('_t')[0].split('_mirror')[0]
    base_corners.add(base_id)

stats['overview']['unique_base_corners'] = len(base_corners)
stats['overview']['augmentation_factor'] = len(graphs) / len(base_corners)

print(f"Total graphs: {stats['overview']['total_graphs']:,}")
print(f"Unique base corners: {stats['overview']['unique_base_corners']:,}")
print(f"Augmentation factor: {stats['overview']['augmentation_factor']:.1f}x")
print(f"Receiver coverage: {stats['overview']['receiver_coverage_pct']:.1f}%")

print("\n" + "="*80)
print("2. RECEIVER DISTRIBUTION")
print("="*80)

# Receiver team distribution
receiver_teams = Counter()
for g in graphs:
    if hasattr(g, 'teams') and g.receiver_node_index is not None:
        team = g.teams[g.receiver_node_index] if g.receiver_node_index < len(g.teams) else None
        if team:
            receiver_teams[team] += 1

stats['receiver_distribution']['attacking'] = receiver_teams.get('attacking', 0)
stats['receiver_distribution']['defending'] = receiver_teams.get('defending', 0)
stats['receiver_distribution']['attacking_pct'] = (stats['receiver_distribution']['attacking'] / len(graphs)) * 100
stats['receiver_distribution']['defending_pct'] = (stats['receiver_distribution']['defending'] / len(graphs)) * 100

print(f"Attacking receivers: {stats['receiver_distribution']['attacking']:,} ({stats['receiver_distribution']['attacking_pct']:.1f}%)")
print(f"Defending receivers: {stats['receiver_distribution']['defending']:,} ({stats['receiver_distribution']['defending_pct']:.1f}%)")

print("\n" + "="*80)
print("3. OUTCOME DISTRIBUTION")
print("="*80)

# Get outcomes
outcomes = Counter()
shot_situations = 0

for g in graphs:
    if hasattr(g, 'outcome'):
        outcomes[g.outcome] += 1
        # Shot situation = shot OR goal
        if g.outcome in ['shot', 'goal']:
            shot_situations += 1

stats['outcome_distribution']['outcomes'] = dict(outcomes)
stats['outcome_distribution']['shot_situations'] = shot_situations
stats['outcome_distribution']['shot_pct'] = (shot_situations / len(graphs)) * 100

print("Outcome breakdown:")
for outcome, count in outcomes.most_common():
    pct = (count / len(graphs)) * 100
    print(f"  {outcome:20} {count:>5} ({pct:>5.1f}%)")

print(f"\nShot situations (shot OR goal): {shot_situations:,} ({stats['outcome_distribution']['shot_pct']:.1f}%)")

print("\n" + "="*80)
print("4. TEMPORAL AUGMENTATION")
print("="*80)

# Count temporal frames and mirrors
temporal_counts = Counter()
mirror_count = 0

for g in graphs:
    if '_t' in g.corner_id:
        # Extract temporal frame
        parts = g.corner_id.split('_t')
        if len(parts) > 1:
            frame = parts[1].split('_')[0]
            temporal_counts[frame] += 1

    if '_mirror' in g.corner_id:
        mirror_count += 1

stats['temporal_augmentation']['frames'] = dict(temporal_counts)
stats['temporal_augmentation']['mirrored'] = mirror_count
stats['temporal_augmentation']['mirrored_pct'] = (mirror_count / len(graphs)) * 100

print("Temporal frame distribution:")
for frame in sorted(temporal_counts.keys()):
    count = temporal_counts[frame]
    pct = (count / len(graphs)) * 100
    print(f"  t={frame:>4}s: {count:>5} ({pct:>5.1f}%)")

print(f"\nMirrored graphs: {mirror_count:,} ({stats['temporal_augmentation']['mirrored_pct']:.1f}%)")

print("\n" + "="*80)
print("5. GRAPH STRUCTURE")
print("="*80)

# Node and edge statistics
num_nodes = []
num_edges = []
num_attacking = []
num_defending = []

for g in graphs:
    num_nodes.append(g.num_nodes)
    num_edges.append(g.num_edges)

    if hasattr(g, 'teams'):
        num_attacking.append(sum(1 for t in g.teams if t == 'attacking'))
        num_defending.append(sum(1 for t in g.teams if t == 'defending'))

num_nodes = np.array(num_nodes)
num_edges = np.array(num_edges)
num_attacking = np.array(num_attacking)
num_defending = np.array(num_defending)

stats['graph_structure']['nodes'] = {
    'mean': float(np.mean(num_nodes)),
    'median': float(np.median(num_nodes)),
    'min': int(np.min(num_nodes)),
    'max': int(np.max(num_nodes)),
    'std': float(np.std(num_nodes))
}

stats['graph_structure']['edges'] = {
    'mean': float(np.mean(num_edges)),
    'median': float(np.median(num_edges)),
    'min': int(np.min(num_edges)),
    'max': int(np.max(num_edges)),
    'std': float(np.std(num_edges))
}

stats['graph_structure']['attacking_players'] = {
    'mean': float(np.mean(num_attacking)),
    'median': float(np.median(num_attacking)),
    'min': int(np.min(num_attacking)),
    'max': int(np.max(num_attacking))
}

stats['graph_structure']['defending_players'] = {
    'mean': float(np.mean(num_defending)),
    'median': float(np.median(num_defending)),
    'min': int(np.min(num_defending)),
    'max': int(np.max(num_defending))
}

print(f"Nodes per graph:")
print(f"  Mean: {stats['graph_structure']['nodes']['mean']:.1f}")
print(f"  Median: {stats['graph_structure']['nodes']['median']:.0f}")
print(f"  Range: {stats['graph_structure']['nodes']['min']}-{stats['graph_structure']['nodes']['max']}")

print(f"\nEdges per graph:")
print(f"  Mean: {stats['graph_structure']['edges']['mean']:.1f}")
print(f"  Median: {stats['graph_structure']['edges']['median']:.0f}")
print(f"  Range: {stats['graph_structure']['edges']['min']}-{stats['graph_structure']['edges']['max']}")

print(f"\nAttacking players per graph:")
print(f"  Mean: {stats['graph_structure']['attacking_players']['mean']:.1f}")
print(f"  Range: {stats['graph_structure']['attacking_players']['min']}-{stats['graph_structure']['attacking_players']['max']}")

print(f"\nDefending players per graph:")
print(f"  Mean: {stats['graph_structure']['defending_players']['mean']:.1f}")
print(f"  Range: {stats['graph_structure']['defending_players']['min']}-{stats['graph_structure']['defending_players']['max']}")

print("\n" + "="*80)
print("6. SPATIAL DISTRIBUTION")
print("="*80)

# Analyze receiver positions
receiver_positions = []
for g in graphs:
    if g.receiver_location is not None:
        receiver_positions.append(g.receiver_location)

if receiver_positions:
    receiver_positions = np.array(receiver_positions)

    stats['spatial_distribution']['receiver_x'] = {
        'mean': float(np.mean(receiver_positions[:, 0])),
        'median': float(np.median(receiver_positions[:, 0])),
        'std': float(np.std(receiver_positions[:, 0]))
    }

    stats['spatial_distribution']['receiver_y'] = {
        'mean': float(np.mean(receiver_positions[:, 1])),
        'median': float(np.median(receiver_positions[:, 1])),
        'std': float(np.std(receiver_positions[:, 1]))
    }

    print(f"Receiver event locations (StatsBomb 120x80 pitch):")
    print(f"  X (length): mean={stats['spatial_distribution']['receiver_x']['mean']:.1f}, "
          f"median={stats['spatial_distribution']['receiver_x']['median']:.1f}, "
          f"std={stats['spatial_distribution']['receiver_x']['std']:.1f}")
    print(f"  Y (width):  mean={stats['spatial_distribution']['receiver_y']['mean']:.1f}, "
          f"median={stats['spatial_distribution']['receiver_y']['median']:.1f}, "
          f"std={stats['spatial_distribution']['receiver_y']['std']:.1f}")

# Save statistics to JSON
output_file = Path("docs/DATASET_STATISTICS.json")
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "="*80)
print(f"Statistics saved to: {output_file}")
print("="*80)
