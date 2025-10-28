#!/usr/bin/env python3
"""
Merge 'Goal' outcome_label into 'Shot' category (standalone version).

This creates a cleaner 4-class distribution without importing graph_builder.

Author: mseo
Date: October 2024
"""

import pickle
import shutil
from pathlib import Path
from collections import Counter


def merge_goal_into_shot(input_path, output_path=None, backup=True):
    """Merge Goal labels into Shot category."""
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Create backup if requested
    if backup and output_path == input_path:
        backup_path = input_path.with_suffix('.pkl.backup_before_merge')
        print(f"Creating backup: {backup_path}")
        shutil.copy2(input_path, backup_path)

    # Load graphs with minimal dependencies
    print(f"\nLoading graphs from: {input_path}")
    with open(input_path, 'rb') as f:
        graphs = pickle.load(f)

    print(f"Total graphs: {len(graphs)}")

    # Count current distribution
    print("\n" + "=" * 70)
    print("CURRENT DISTRIBUTION")
    print("=" * 70)

    labeled_graphs = [g for g in graphs if hasattr(g, 'outcome_label') and g.outcome_label]
    current_dist = Counter(g.outcome_label for g in labeled_graphs)

    for outcome, count in sorted(current_dist.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(labeled_graphs) * 100
        print(f"  {outcome:20s}: {count:5d} graphs ({pct:5.1f}%)")

    # Merge Goal → Shot
    print("\n" + "=" * 70)
    print("MERGING Goal → Shot")
    print("=" * 70)

    merge_count = 0
    for graph in graphs:
        if hasattr(graph, 'outcome_label') and graph.outcome_label == 'Goal':
            graph.outcome_label = 'Shot'
            merge_count += 1

    print(f"Merged {merge_count} graphs from 'Goal' to 'Shot'")

    # Count new distribution
    print("\n" + "=" * 70)
    print("NEW DISTRIBUTION (4 classes)")
    print("=" * 70)

    labeled_graphs = [g for g in graphs if hasattr(g, 'outcome_label') and g.outcome_label]
    new_dist = Counter(g.outcome_label for g in labeled_graphs)

    for outcome, count in sorted(new_dist.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(labeled_graphs) * 100
        print(f"  {outcome:20s}: {count:5d} graphs ({pct:5.1f}%)")

    # Verify no Goal labels remain
    remaining_goals = sum(1 for g in graphs
                         if hasattr(g, 'outcome_label') and g.outcome_label == 'Goal')

    if remaining_goals > 0:
        print(f"\n⚠️  WARNING: {remaining_goals} 'Goal' labels still remain!")
        return False
    else:
        print(f"\n✅ SUCCESS: No 'Goal' labels remain (all merged into 'Shot')")

    # Save modified graphs
    print(f"\nSaving modified graphs to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Saved {len(graphs)} graphs with merged labels")

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("MERGE GOAL → SHOT LABELS")
    print("=" * 70)

    success = merge_goal_into_shot(
        input_path='data/graphs/adjacency_team/combined_temporal_graphs.pkl',
        output_path=None,  # Overwrite
        backup=True
    )

    if success:
        print("\n" + "=" * 70)
        print("✅ MERGE COMPLETE")
        print("=" * 70)
        print("\nYou now have 4 outcome classes:")
        print("  1. Shot (18.2%) - includes all goals")
        print("  2. Clearance (52.0%)")
        print("  3. Loss (19.4%)")
        print("  4. Possession (10.5%)")
        exit(0)
    else:
        print("\n❌ MERGE FAILED")
        exit(1)
