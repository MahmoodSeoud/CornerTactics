#!/usr/bin/env python3
"""
Unit tests for ReceiverCornerDataset (Day 3-4 implementation)

Tests:
1. ReceiverCornerDataset initialization
2. Velocity masking (features 4-5 set to 0)
3. Receiver label field (torch.LongTensor, range 0-21)
4. Shot label field (torch.FloatTensor, binary 0.0/1.0)
5. Batch shapes verification
6. Split integrity (no corner ID leakage)
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.receiver_data_loader import ReceiverCornerDataset, load_receiver_dataset


class TestReceiverCornerDataset:
    """Test suite for ReceiverCornerDataset"""

    @pytest.fixture
    def graph_path(self):
        """Path to test graph pickle file"""
        # Use StatsBomb temporal augmented graphs with receiver labels
        path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl")
        if not path.exists():
            pytest.skip(f"Graph file not found: {path}")
        return str(path)

    def test_initialization(self, graph_path):
        """Test ReceiverCornerDataset initializes successfully"""
        dataset = ReceiverCornerDataset(graph_path)

        # Should load graphs and create data_list
        assert dataset is not None
        assert len(dataset.data_list) > 0
        assert dataset.num_graphs > 0

        print(f"✓ Loaded {dataset.num_graphs} corners with receiver labels")

    def test_velocity_masking(self, graph_path):
        """Test that velocity features (indices 4-5) are masked to 0.0"""
        dataset = ReceiverCornerDataset(graph_path)

        # Check first batch
        data = dataset.data_list[0]

        # Velocity features are at indices 4 and 5 (vx, vy)
        velocity_x = data.x[:, 4]
        velocity_y = data.x[:, 5]

        # All velocity values should be exactly 0.0
        assert torch.all(velocity_x == 0.0), "Velocity X not masked"
        assert torch.all(velocity_y == 0.0), "Velocity Y not masked"

        print(f"✓ Velocity features masked correctly")

    def test_receiver_label_field(self, graph_path):
        """Test that receiver_label field exists and has correct type/range"""
        dataset = ReceiverCornerDataset(graph_path)

        for data in dataset.data_list[:10]:
            # Check field exists
            assert hasattr(data, 'receiver_label'), "Missing receiver_label field"

            # Check type is LongTensor
            assert isinstance(data.receiver_label, torch.Tensor)
            assert data.receiver_label.dtype == torch.long

            # Check shape (should be scalar or [1])
            assert data.receiver_label.numel() == 1

            # Check range (0-21 for 22 players)
            label_value = data.receiver_label.item()
            assert 0 <= label_value <= 21, f"Label {label_value} out of range [0, 21]"

        print(f"✓ Receiver labels have correct type and range")

    def test_shot_label_field(self, graph_path):
        """Test that shot_label field exists and is binary float"""
        dataset = ReceiverCornerDataset(graph_path)

        for data in dataset.data_list[:10]:
            # Check field exists
            assert hasattr(data, 'shot_label'), "Missing shot_label field"

            # Check type is FloatTensor
            assert isinstance(data.shot_label, torch.Tensor)
            assert data.shot_label.dtype == torch.float32

            # Check shape
            assert data.shot_label.numel() == 1

            # Check binary (0.0 or 1.0)
            label_value = data.shot_label.item()
            assert label_value in [0.0, 1.0], f"Shot label {label_value} not binary"

        print(f"✓ Shot labels are binary floats")

    def test_batch_shapes(self, graph_path):
        """Test that batches have correct shapes"""
        from torch_geometric.loader import DataLoader

        dataset = ReceiverCornerDataset(graph_path)
        loader = DataLoader(dataset.data_list[:32], batch_size=8, shuffle=False)

        batch = next(iter(loader))

        # Check batch.x shape [num_nodes, 14]
        assert batch.x.ndim == 2
        assert batch.x.size(1) == 14, f"Expected 14 features, got {batch.x.size(1)}"

        # Check velocity columns are masked
        assert torch.all(batch.x[:, 4] == 0.0)
        assert torch.all(batch.x[:, 5] == 0.0)

        # Check receiver_label shape [batch_size]
        assert batch.receiver_label.ndim == 1
        assert batch.receiver_label.size(0) == 8, f"Expected batch_size=8, got {batch.receiver_label.size(0)}"

        # Check shot_label shape [batch_size]
        assert batch.shot_label.ndim == 1
        assert batch.shot_label.size(0) == 8

        print(f"✓ Batch shapes correct: x={batch.x.shape}, "
              f"receiver_label={batch.receiver_label.shape}, shot_label={batch.shot_label.shape}")

    def test_split_integrity(self, graph_path):
        """Test that train/val/test splits have no corner ID leakage"""
        dataset = ReceiverCornerDataset(graph_path)

        # Get split indices
        splits = dataset.get_split_indices(test_size=0.15, val_size=0.15, random_state=42)

        # Extract corner IDs for each split
        train_corners = set()
        val_corners = set()
        test_corners = set()

        def get_base_corner_id(corner_id):
            """Extract base corner ID (remove temporal suffix)"""
            if '_t' in corner_id:
                return corner_id.split('_t')[0]
            return corner_id

        for idx in splits['train']:
            corner_id = dataset.data_list[idx].corner_id
            train_corners.add(get_base_corner_id(corner_id))

        for idx in splits['val']:
            corner_id = dataset.data_list[idx].corner_id
            val_corners.add(get_base_corner_id(corner_id))

        for idx in splits['test']:
            corner_id = dataset.data_list[idx].corner_id
            test_corners.add(get_base_corner_id(corner_id))

        # Check no overlap
        assert len(train_corners & val_corners) == 0, "Train/Val corner overlap!"
        assert len(train_corners & test_corners) == 0, "Train/Test corner overlap!"
        assert len(val_corners & test_corners) == 0, "Val/Test corner overlap!"

        print(f"✓ No corner ID leakage: {len(train_corners)} train, "
              f"{len(val_corners)} val, {len(test_corners)} test unique corners")

    def test_skip_invalid_receivers(self, graph_path):
        """Test that corners without valid receiver labels are skipped"""
        dataset = ReceiverCornerDataset(graph_path)

        # All loaded corners should have valid receiver labels
        for data in dataset.data_list:
            assert hasattr(data, 'receiver_label')
            assert data.receiver_label is not None
            assert 0 <= data.receiver_label.item() <= 21

        print(f"✓ All {len(dataset.data_list)} corners have valid receiver labels")


class TestLoadReceiverDataset:
    """Test the convenience function load_receiver_dataset()"""

    def test_load_receiver_dataset(self):
        """Test load_receiver_dataset helper function"""
        graph_path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl")
        if not graph_path.exists():
            pytest.skip(f"Graph file not found: {graph_path}")

        # Load dataset and dataloaders
        dataset, train_loader, val_loader, test_loader = load_receiver_dataset(
            graph_path=str(graph_path),
            batch_size=16,
            test_size=0.15,
            val_size=0.15,
            random_state=42
        )

        # Verify dataset
        assert dataset is not None
        assert dataset.num_graphs > 0

        # Verify loaders
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Test iteration
        train_batch = next(iter(train_loader))
        assert train_batch.x.size(1) == 14
        assert hasattr(train_batch, 'receiver_label')
        assert hasattr(train_batch, 'shot_label')

        print(f"✓ load_receiver_dataset() works correctly")
        print(f"  Dataset: {dataset.num_graphs} corners")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
