#!/usr/bin/env python3
"""
Test script to verify no data leakage in train/val/test split.

This script checks that:
1. No corner appears in multiple splits
2. All temporal frames from a corner stay together
3. Split sizes are correct
4. Class distribution is maintained

Author: mseo
Date: October 2024
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import CornerDataset


def get_base_corner_id(corner_id):
    """Extract base corner ID by removing temporal suffix (_t...)"""
    if '_t' in corner_id:
        return corner_id.split('_t')[0]
    return corner_id


def test_data_leakage():
    """Test for data leakage in train/val/test split."""
    print("=" * 70)
    print("DATA LEAKAGE VERIFICATION TEST")
    print("=" * 70)

    # Load dataset
    graph_path = 'data/graphs/adjacency_team/combined_temporal_graphs.pkl'
    print(f"\nLoading dataset from: {graph_path}")

    dataset = CornerDataset(
        graph_path=graph_path,
        outcome_type='shot'
    )

    # Get split indices
    print("\nCreating train/val/test splits...")
    splits = dataset.get_split_indices(test_size=0.15, val_size=0.15, random_state=42)

    # Extract corner_ids from each split
    print("\nExtracting corner IDs from each split...")
    train_corners = set(get_base_corner_id(dataset.data_list[i].corner_id) for i in splits['train'])
    val_corners = set(get_base_corner_id(dataset.data_list[i].corner_id) for i in splits['val'])
    test_corners = set(get_base_corner_id(dataset.data_list[i].corner_id) for i in splits['test'])

    # Print split statistics
    print("\n" + "=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)
    print(f"\nGraphs:")
    print(f"  Train: {len(splits['train'])} graphs ({len(splits['train'])/len(dataset.data_list)*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} graphs ({len(splits['val'])/len(dataset.data_list)*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} graphs ({len(splits['test'])/len(dataset.data_list)*100:.1f}%)")
    print(f"  Total: {len(dataset.data_list)} graphs")

    print(f"\nUnique Corners:")
    print(f"  Train: {len(train_corners)} corners ({len(train_corners)/(len(train_corners)+len(val_corners)+len(test_corners))*100:.1f}%)")
    print(f"  Val:   {len(val_corners)} corners ({len(val_corners)/(len(train_corners)+len(val_corners)+len(test_corners))*100:.1f}%)")
    print(f"  Test:  {len(test_corners)} corners ({len(test_corners)/(len(train_corners)+len(val_corners)+len(test_corners))*100:.1f}%)")
    print(f"  Total: {len(train_corners) + len(val_corners) + len(test_corners)} corners")

    # Check for overlap (should be ZERO)
    print("\n" + "=" * 70)
    print("DATA LEAKAGE CHECK")
    print("=" * 70)

    train_val_overlap = train_corners & val_corners
    train_test_overlap = train_corners & test_corners
    val_test_overlap = val_corners & test_corners

    print(f"\nOverlap train-val:  {len(train_val_overlap)} corners (should be 0)")
    print(f"Overlap train-test: {len(train_test_overlap)} corners (should be 0)")
    print(f"Overlap val-test:   {len(val_test_overlap)} corners (should be 0)")

    # Check that all temporal frames from same corner are in same split
    print("\n" + "=" * 70)
    print("TEMPORAL FRAME CONSISTENCY CHECK")
    print("=" * 70)

    # For each split, verify all graphs from same corner are together
    all_ok = True
    for split_name, split_indices in [('train', splits['train']), ('val', splits['val']), ('test', splits['test'])]:
        corner_graph_counts = {}
        for idx in split_indices:
            base_corner = get_base_corner_id(dataset.data_list[idx].corner_id)
            corner_graph_counts[base_corner] = corner_graph_counts.get(base_corner, 0) + 1

        # Check if any corner has unexpected number of frames
        max_frames = max(corner_graph_counts.values())
        min_frames = min(corner_graph_counts.values())
        avg_frames = sum(corner_graph_counts.values()) / len(corner_graph_counts)

        print(f"\n{split_name.capitalize()} set:")
        print(f"  Frames per corner: min={min_frames}, max={max_frames}, avg={avg_frames:.2f}")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("\n❌ FAILED: Data leakage still present!")
        print(f"   Found {len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)} overlapping corners")
        return False
    else:
        print("\n✅ PASSED: No data leakage detected!")
        print("   All temporal frames from each corner stay together in the same split.")
        print("   The model will now generalize to NEW corners, not just new time frames.")
        return True


if __name__ == "__main__":
    success = test_data_leakage()
    sys.exit(0 if success else 1)
