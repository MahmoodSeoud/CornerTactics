#!/usr/bin/env python3
"""
Unit tests for Outcome Class Label in ReceiverCornerDataset

Tests the extension of receiver_data_loader.py to include outcome_class_label field
for multi-class outcome prediction (Goal/Shot/Clearance/Possession).

Following TDD approach: Write failing tests first.
"""

import sys
import torch
import pytest
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.receiver_data_loader import ReceiverCornerDataset


class TestOutcomeClassLabel:
    """Test suite for outcome_class_label field in dataset."""

    @pytest.fixture
    def dataset(self):
        """Load the dataset with receiver labels."""
        graph_path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl")
        if not graph_path.exists():
            pytest.skip(f"Graph file not found: {graph_path}")

        return ReceiverCornerDataset(str(graph_path), mask_velocities=True)

    def test_outcome_class_label_exists(self, dataset):
        """
        RED TEST: Verify outcome_class_label field exists in data objects.

        Expected to FAIL initially - outcome_class_label not yet implemented.
        """
        # Get first data object
        data = dataset.data_list[0]

        # Assert outcome_class_label field exists
        assert hasattr(data, 'outcome_class_label'), \
            "Data object should have outcome_class_label field"

        # Assert it's a LongTensor
        assert isinstance(data.outcome_class_label, torch.Tensor), \
            "outcome_class_label should be a torch.Tensor"
        assert data.outcome_class_label.dtype == torch.long, \
            "outcome_class_label should be LongTensor (dtype=torch.long)"

    def test_outcome_class_label_shape(self, dataset):
        """
        RED TEST: Verify outcome_class_label has correct shape.

        Expected shape: [1] (single scalar per corner)
        """
        data = dataset.data_list[0]

        assert data.outcome_class_label.shape == torch.Size([1]), \
            f"outcome_class_label should have shape [1], got {data.outcome_class_label.shape}"

    def test_outcome_class_label_range(self, dataset):
        """
        RED TEST: Verify outcome_class_label values are in valid range [0, 3].

        Class mapping:
        - 0: Goal
        - 1: Shot
        - 2: Clearance
        - 3: Possession (includes Loss)
        """
        for data in dataset.data_list:
            label = data.outcome_class_label.item()
            assert 0 <= label <= 3, \
                f"outcome_class_label should be in [0, 3], got {label}"

    def test_outcome_class_distribution(self, dataset):
        """
        RED TEST: Verify outcome class distribution matches expected ranges.

        Expected distribution:
        - Goal (0): ~1.3% (rare)
        - Shot (1): ~16.9% (minority)
        - Clearance (2): ~52.0% (common)
        - Possession (3): ~29.9% (Loss + Possession merged)
        """
        from collections import Counter

        labels = [data.outcome_class_label.item() for data in dataset.data_list]
        counter = Counter(labels)
        total = len(labels)

        # Goal: ~1-3%
        goal_pct = counter[0] / total * 100
        assert 0.5 <= goal_pct <= 5.0, \
            f"Goal class should be ~1.3%, got {goal_pct:.1f}%"

        # Shot: ~10-25%
        shot_pct = counter[1] / total * 100
        assert 10.0 <= shot_pct <= 25.0, \
            f"Shot class should be ~16.9%, got {shot_pct:.1f}%"

        # Clearance: ~45-60%
        clearance_pct = counter[2] / total * 100
        assert 45.0 <= clearance_pct <= 60.0, \
            f"Clearance class should be ~52.0%, got {clearance_pct:.1f}%"

        # Possession: ~20-40%
        possession_pct = counter[3] / total * 100
        assert 20.0 <= possession_pct <= 40.0, \
            f"Possession class should be ~29.9%, got {possession_pct:.1f}%"

    def test_outcome_mapping_correctness(self, dataset):
        """
        RED TEST: Verify outcome_class_label correctly maps from outcome_label string.

        Mapping:
        - "Goal" → 0
        - "Shot" → 1
        - "Clearance" → 2
        - "Possession" → 3
        - "Loss" → 3
        """
        # Sample a few data points and check mapping
        for i in range(min(50, len(dataset.data_list))):
            data = dataset.data_list[i]
            graph = dataset.graphs[i]

            outcome_str = graph.outcome_label
            outcome_class = data.outcome_class_label.item()

            # Verify mapping
            if outcome_str == "Goal":
                assert outcome_class == 0, \
                    f"Goal should map to 0, got {outcome_class}"
            elif outcome_str == "Shot":
                assert outcome_class == 1, \
                    f"Shot should map to 1, got {outcome_class}"
            elif outcome_str == "Clearance":
                assert outcome_class == 2, \
                    f"Clearance should map to 2, got {outcome_class}"
            elif outcome_str in ["Possession", "Loss"]:
                assert outcome_class == 3, \
                    f"Possession/Loss should map to 3, got {outcome_class}"

    def test_data_loader_batch_with_outcome_labels(self, dataset):
        """
        RED TEST: Verify data loader batches include outcome_class_label.
        """
        from torch_geometric.data import DataLoader

        # Create mini data loader
        loader = DataLoader(dataset.data_list[:32], batch_size=8, shuffle=False)

        # Get first batch
        batch = next(iter(loader))

        # Assert outcome_class_label exists in batch
        assert hasattr(batch, 'outcome_class_label'), \
            "Batch should have outcome_class_label field"

        # Assert correct shape: [batch_size]
        assert batch.outcome_class_label.shape == torch.Size([8]), \
            f"Batch outcome_class_label should have shape [8], got {batch.outcome_class_label.shape}"

        # Assert all values in valid range
        assert (batch.outcome_class_label >= 0).all(), \
            "All outcome_class_label values should be >= 0"
        assert (batch.outcome_class_label <= 3).all(), \
            "All outcome_class_label values should be <= 3"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
