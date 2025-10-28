#!/usr/bin/env python3
"""
Analyze class distribution in corner kick dataset.

Shows detailed breakdown of outcomes across the entire dataset
and within train/val/test splits.

Author: mseo
Date: October 2024
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import CornerDataset


def get_base_corner_id(corner_id):
    """Extract base corner ID by removing temporal suffix (_t...)"""
    if '_t' in corner_id:
        return corner_id.split('_t')[0]
    return corner_id


def analyze_class_distribution():
    """Analyze outcome distribution in the dataset."""
    print("=" * 80)
    print("CORNER KICK OUTCOME CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Load raw graphs to access outcome labels
    graph_path = 'data/graphs/adjacency_team/combined_temporal_graphs.pkl'
    print(f"\nLoading graphs from: {graph_path}")

    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)

    print(f"Total graphs: {len(graphs)}")

    # Get unique corners
    base_corners = {}
    for g in graphs:
        base_id = get_base_corner_id(g.corner_id)
        if base_id not in base_corners:
            base_corners[base_id] = {
                'goal_scored': g.goal_scored,
                'outcome_label': g.outcome_label if hasattr(g, 'outcome_label') else None
            }

    print(f"Unique corners: {len(base_corners)}")

    # Count outcomes at corner level (unique corners)
    print("\n" + "=" * 80)
    print("CORNER-LEVEL OUTCOMES (1,435 unique corners)")
    print("=" * 80)

    corner_goals = sum(1 for c in base_corners.values() if c['goal_scored'])
    corner_no_goals = len(base_corners) - corner_goals

    print(f"\nBinary Classification (goal_scored):")
    print(f"  Goals:     {corner_goals:4d} corners ({corner_goals/len(base_corners)*100:5.1f}%)")
    print(f"  No Goals:  {corner_no_goals:4d} corners ({corner_no_goals/len(base_corners)*100:5.1f}%)")

    # Count outcome labels if available
    outcome_counts = Counter(c['outcome_label'] for c in base_corners.values() if c['outcome_label'])

    if outcome_counts:
        print(f"\nMulti-class Outcomes (outcome_label):")
        total_with_labels = sum(outcome_counts.values())

        # Sort by count descending
        for outcome, count in sorted(outcome_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {outcome:20s}: {count:4d} corners ({count/total_with_labels*100:5.1f}%)")

        # Calculate "dangerous situations" (Shot OR Goal)
        dangerous = sum(count for outcome, count in outcome_counts.items()
                       if outcome in ['Shot', 'Goal'])
        print(f"\n  Dangerous (Shot+Goal): {dangerous:4d} corners ({dangerous/total_with_labels*100:5.1f}%)")

    # Count outcomes at graph level (all temporal frames)
    print("\n" + "=" * 80)
    print("GRAPH-LEVEL OUTCOMES (7,369 temporal frames)")
    print("=" * 80)

    graph_goals = sum(1 for g in graphs if g.goal_scored)
    graph_no_goals = len(graphs) - graph_goals

    print(f"\nBinary Classification (goal_scored):")
    print(f"  Goals:     {graph_goals:4d} graphs ({graph_goals/len(graphs)*100:5.1f}%)")
    print(f"  No Goals:  {graph_no_goals:4d} graphs ({graph_no_goals/len(graphs)*100:5.1f}%)")

    graph_outcome_counts = Counter(g.outcome_label for g in graphs if hasattr(g, 'outcome_label') and g.outcome_label)

    if graph_outcome_counts:
        print(f"\nMulti-class Outcomes (outcome_label):")
        total_graphs_with_labels = sum(graph_outcome_counts.values())

        for outcome, count in sorted(graph_outcome_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {outcome:20s}: {count:4d} graphs ({count/total_graphs_with_labels*100:5.1f}%)")

        dangerous_graphs = sum(count for outcome, count in graph_outcome_counts.items()
                              if outcome in ['Shot', 'Goal'])
        print(f"\n  Dangerous (Shot+Goal): {dangerous_graphs:4d} graphs ({dangerous_graphs/total_graphs_with_labels*100:5.1f}%)")

    # Analyze split distribution
    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION IN TRAIN/VAL/TEST SPLITS")
    print("=" * 80)

    # Load dataset and get splits
    dataset = CornerDataset(graph_path, outcome_type='shot')
    splits = dataset.get_split_indices(test_size=0.15, val_size=0.15, random_state=42)

    for split_name in ['train', 'val', 'test']:
        split_indices = splits[split_name]

        # Get corner-level stats
        split_corners = set(get_base_corner_id(dataset.data_list[i].corner_id)
                           for i in split_indices)

        # Get outcome stats
        split_goals = sum(1 for i in split_indices if dataset.data_list[i].y.item() == 1.0)
        split_total = len(split_indices)

        # Get unique corner outcomes
        corner_goal_count = sum(1 for base_id in split_corners
                               if base_corners[base_id]['goal_scored'])

        print(f"\n{split_name.upper()} SET:")
        print(f"  Corners:       {len(split_corners):4d}")
        print(f"  Graphs:        {split_total:4d}")
        print(f"  Positive rate: {split_goals/split_total*100:5.1f}% ({split_goals} / {split_total})")
        print(f"  Corner-level goals: {corner_goal_count:4d} / {len(split_corners)} ({corner_goal_count/len(split_corners)*100:5.1f}%)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"  - Dataset has {len(base_corners)} unique corners with {len(graphs)/len(base_corners):.1f} temporal frames each")
    print(f"  - Binary goal rate: {corner_goals/len(base_corners)*100:.1f}% at corner level")
    print(f"  - 'Dangerous situation' rate: {dangerous/total_with_labels*100:.1f}% (Shot OR Goal)")
    print(f"  - Class imbalance factor: {corner_no_goals/corner_goals:.1f}:1 (safe:dangerous)")
    print("\nRecommendation:")
    if corner_goals / len(base_corners) < 0.05:
        print("  ⚠️  Extreme class imbalance! Use 'shot' outcome (Shot OR Goal) for better balance")
    elif corner_goals / len(base_corners) < 0.20:
        print("  ⚠️  Moderate class imbalance. Consider using 'shot' outcome or class weights")
    else:
        print("  ✅ Reasonable class balance for binary classification")


if __name__ == "__main__":
    analyze_class_distribution()
