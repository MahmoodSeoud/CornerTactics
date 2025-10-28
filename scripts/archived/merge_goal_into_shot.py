#!/usr/bin/env python3
"""
Merge 'Goal' outcome_label into 'Shot' category.

This creates a cleaner 4-class distribution:
- Shot (includes goals): 18.2%
- Clearance: 52.0%
- Loss: 19.4%
- Possession: 10.5%

Author: mseo
Date: October 2024
"""

import pickle
import shutil
from pathlib import Path
from collections import Counter


def merge_goal_into_shot(input_path, output_path=None, backup=True):
    """
    Merge Goal labels into Shot category.

    Args:
        input_path: Path to input graph pickle file
        output_path: Path to save modified graphs (default: overwrites input)
        backup: Whether to create backup of original file
    """
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

    # Load graphs
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
    import argparse

    parser = argparse.ArgumentParser(description="Merge Goal labels into Shot category")
    parser.add_argument(
        '--input',
        type=str,
        default='data/graphs/adjacency_team/combined_temporal_graphs.pkl',
        help='Input graph pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path (default: overwrite input file)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup file'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MERGE GOAL → SHOT LABELS")
    print("=" * 70)

    success = merge_goal_into_shot(
        input_path=args.input,
        output_path=args.output,
        backup=not args.no_backup
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
        print("\nNext steps:")
        print("  1. Update data_loader.py to use 'multi' outcome type with 4 classes")
        print("  2. Re-train model with multi-class classification")
        exit(0)
    else:
        print("\n❌ MERGE FAILED")
        exit(1)
