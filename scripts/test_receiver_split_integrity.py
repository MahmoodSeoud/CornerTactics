#!/usr/bin/env python3
"""
Test Script: Verify Train/Val/Test Split Integrity

Verifies that no corner ID leakage occurs across train/val/test splits
for the ReceiverCornerDataset.

According to Day 3-4 Success Criteria:
- ✅ Batch shapes correct
- ✅ No data leakage (same corner_id stays in same split)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.receiver_data_loader import ReceiverCornerDataset

def test_split_integrity():
    """Test that no corner_id appears in multiple splits."""
    
    print("="*70)
    print("TESTING RECEIVER DATASET SPLIT INTEGRITY")
    print("="*70)
    
    # Load dataset
    graph_path = "data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl"
    dataset = ReceiverCornerDataset(graph_path, mask_velocities=True)
    
    # Get split indices
    splits = dataset.get_split_indices(test_size=0.15, val_size=0.15, random_state=42)
    
    # Extract base corner IDs for each split
    def get_base_corner_id(corner_id):
        """Extract base corner ID by removing temporal suffix."""
        if '_t' in corner_id:
            return corner_id.split('_t')[0]
        return corner_id
    
    train_corners = set()
    val_corners = set()
    test_corners = set()
    
    for idx in splits['train']:
        base_id = get_base_corner_id(dataset.data_list[idx].corner_id)
        train_corners.add(base_id)
    
    for idx in splits['val']:
        base_id = get_base_corner_id(dataset.data_list[idx].corner_id)
        val_corners.add(base_id)
    
    for idx in splits['test']:
        base_id = get_base_corner_id(dataset.data_list[idx].corner_id)
        test_corners.add(base_id)
    
    print(f"\nUnique corners per split:")
    print(f"  Train: {len(train_corners)} unique corners")
    print(f"  Val:   {len(val_corners)} unique corners")
    print(f"  Test:  {len(test_corners)} unique corners")
    print(f"  Total: {len(train_corners) + len(val_corners) + len(test_corners)} unique corners")
    
    # Check for overlap
    train_val_overlap = train_corners & val_corners
    train_test_overlap = train_corners & test_corners
    val_test_overlap = val_corners & test_corners
    
    print(f"\nOverlap check:")
    print(f"  Train ∩ Val:  {len(train_val_overlap)} corners")
    print(f"  Train ∩ Test: {len(train_test_overlap)} corners")
    print(f"  Val ∩ Test:   {len(val_test_overlap)} corners")
    
    # Verify no leakage
    total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    
    if total_overlap == 0:
        print(f"\n{'='*70}")
        print("✅ SPLIT INTEGRITY VERIFIED: Zero corner ID leakage!")
        print(f"{'='*70}")
        return True
    else:
        print(f"\n{'='*70}")
        print(f"❌ SPLIT INTEGRITY FAILED: {total_overlap} corner IDs leaked across splits!")
        print(f"{'='*70}")
        
        # Show examples of leaked corners
        if train_val_overlap:
            print(f"\nTrain-Val overlap examples: {list(train_val_overlap)[:5]}")
        if train_test_overlap:
            print(f"Train-Test overlap examples: {list(train_test_overlap)[:5]}")
        if val_test_overlap:
            print(f"Val-Test overlap examples: {list(val_test_overlap)[:5]}")
        
        return False

if __name__ == "__main__":
    success = test_split_integrity()
    sys.exit(0 if success else 1)
