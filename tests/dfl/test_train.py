"""Tests for Phase 3: Training utilities.

Following TDD, these tests are written first and should fail until
the implementation is complete.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


# Test data path
METRICA_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "metrica" / "data"


class TestOpenPlayExtraction:
    """Tests for extracting open-play sequences for pretraining."""

    def test_extract_open_play_sequences_exists(self):
        """extract_open_play_sequences function should exist."""
        from src.dfl.train import extract_open_play_sequences

        assert extract_open_play_sequences is not None

    def test_extract_open_play_sequences_returns_list(self):
        """extract_open_play_sequences should return a list."""
        from src.dfl.train import extract_open_play_sequences
        from src.dfl.data_loading import load_tracking_data, load_event_data

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        sequences = extract_open_play_sequences(tracking, events)

        assert isinstance(sequences, list)

    def test_extract_open_play_sequences_has_required_keys(self):
        """Each sequence should have frames and shot_label."""
        from src.dfl.train import extract_open_play_sequences
        from src.dfl.data_loading import load_tracking_data, load_event_data

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        sequences = extract_open_play_sequences(
            tracking, events, window_seconds=2, stride_seconds=4
        )

        if not sequences:
            pytest.skip("No sequences extracted")

        seq = sequences[0]
        assert "frames" in seq
        assert "shot_label" in seq
        assert seq["shot_label"] in [0, 1]

    def test_extract_open_play_sequences_window_size(self):
        """Sequences should have approximately correct number of frames."""
        from src.dfl.train import extract_open_play_sequences
        from src.dfl.data_loading import load_tracking_data, load_event_data

        tracking = load_tracking_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )
        events = load_event_data(
            provider="metrica",
            data_dir=METRICA_DATA_DIR / "Sample_Game_3",
        )

        window_seconds = 4
        fps = 25
        sequences = extract_open_play_sequences(
            tracking, events, window_seconds=window_seconds
        )

        if not sequences:
            pytest.skip("No sequences extracted")

        expected_frames = window_seconds * fps
        actual_frames = len(sequences[0]["frames"])

        # Allow some tolerance
        assert abs(actual_frames - expected_frames) < expected_frames * 0.2


class TestPretrainSpatialGNN:
    """Tests for spatial GNN pretraining on open-play data."""

    def test_pretrain_spatial_gnn_exists(self):
        """pretrain_spatial_gnn function should exist."""
        from src.dfl.train import pretrain_spatial_gnn

        assert pretrain_spatial_gnn is not None

    def test_pretrain_spatial_gnn_returns_model(self):
        """pretrain_spatial_gnn should return a model."""
        from src.dfl.train import pretrain_spatial_gnn
        from src.dfl.model import CornerKickPredictor
        import torch

        model = CornerKickPredictor()

        # Create minimal dummy data
        dummy_sequences = []
        for i in range(5):
            dummy_sequences.append(
                {
                    "graph": torch.randn(23, 8),  # Simple tensor for testing
                    "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                    "shot_label": i % 2,
                }
            )

        trained_model = pretrain_spatial_gnn(
            model, dummy_sequences, epochs=1, lr=1e-3
        )

        assert isinstance(trained_model, CornerKickPredictor)

    def test_pretrain_spatial_gnn_modifies_weights(self):
        """pretrain_spatial_gnn should modify model weights."""
        from src.dfl.train import pretrain_spatial_gnn
        from src.dfl.model import CornerKickPredictor
        import torch

        model = CornerKickPredictor()

        # Copy initial weights
        initial_weights = model.spatial_gnn.conv1.lin_l.weight.data.clone()

        dummy_sequences = []
        for i in range(10):
            dummy_sequences.append(
                {
                    "graph": torch.randn(23, 8),
                    "edge_index": torch.tensor(
                        [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
                    ),
                    "shot_label": i % 2,
                }
            )

        trained_model = pretrain_spatial_gnn(
            model, dummy_sequences, epochs=2, lr=1e-2
        )

        final_weights = trained_model.spatial_gnn.conv1.lin_l.weight.data

        # Weights should have changed
        assert not torch.allclose(initial_weights, final_weights)


class TestFinetuneOnCorners:
    """Tests for fine-tuning on corner kick dataset."""

    def test_finetune_on_corners_exists(self):
        """finetune_on_corners function should exist."""
        from src.dfl.train import finetune_on_corners

        assert finetune_on_corners is not None

    def test_finetune_on_corners_returns_results(self):
        """finetune_on_corners should return results dict."""
        from src.dfl.train import finetune_on_corners
        from src.dfl.model import CornerKickPredictor
        from torch_geometric.data import Data
        import torch

        model = CornerKickPredictor()

        # Create minimal dummy corner dataset
        dummy_dataset = []
        for i in range(6):  # 6 samples, 2 per match
            graphs = []
            for _ in range(3):  # 3 frames per corner
                graphs.append(
                    Data(
                        x=torch.randn(23, 8),
                        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                    )
                )
            dummy_dataset.append(
                {
                    "graphs": graphs,
                    "labels": {
                        "shot_binary": i % 2,
                        "goal_binary": 0,
                        "first_contact_team": "attacking",
                        "outcome_class": "other",
                    },
                    "match_id": f"match_{i // 2}",
                    "corner_time": float(i),
                }
            )

        results = finetune_on_corners(model, dummy_dataset, epochs=1, lr=1e-3)

        assert isinstance(results, list)

    def test_finetune_on_corners_returns_fold_metrics(self):
        """Each fold result should have test_match and metrics."""
        from src.dfl.train import finetune_on_corners
        from src.dfl.model import CornerKickPredictor
        from torch_geometric.data import Data
        import torch

        model = CornerKickPredictor()

        dummy_dataset = []
        for i in range(6):
            graphs = [
                Data(
                    x=torch.randn(23, 8),
                    edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                )
                for _ in range(3)
            ]
            dummy_dataset.append(
                {
                    "graphs": graphs,
                    "labels": {
                        "shot_binary": i % 2,
                        "goal_binary": 0,
                        "first_contact_team": "attacking",
                        "outcome_class": "other",
                    },
                    "match_id": f"match_{i // 2}",
                    "corner_time": float(i),
                }
            )

        results = finetune_on_corners(model, dummy_dataset, epochs=1)

        if results:
            result = results[0]
            assert "test_match" in result


class TestCrossValidation:
    """Tests for leave-one-match-out cross-validation."""

    def test_leave_one_match_out_split_exists(self):
        """leave_one_match_out_split function should exist."""
        from src.dfl.train import leave_one_match_out_split

        assert leave_one_match_out_split is not None

    def test_leave_one_match_out_split_returns_folds(self):
        """leave_one_match_out_split should return list of (train, test) tuples."""
        from src.dfl.train import leave_one_match_out_split

        dummy_dataset = [
            {"match_id": "match_A", "data": 1},
            {"match_id": "match_A", "data": 2},
            {"match_id": "match_B", "data": 3},
            {"match_id": "match_B", "data": 4},
            {"match_id": "match_C", "data": 5},
        ]

        folds = leave_one_match_out_split(dummy_dataset)

        assert isinstance(folds, list)
        assert len(folds) == 3  # 3 unique matches

    def test_leave_one_match_out_split_no_leakage(self):
        """Train and test sets should have no overlap in match_id."""
        from src.dfl.train import leave_one_match_out_split

        dummy_dataset = [
            {"match_id": "match_A", "data": 1},
            {"match_id": "match_A", "data": 2},
            {"match_id": "match_B", "data": 3},
            {"match_id": "match_B", "data": 4},
            {"match_id": "match_C", "data": 5},
        ]

        folds = leave_one_match_out_split(dummy_dataset)

        for train_data, test_data in folds:
            train_matches = set(d["match_id"] for d in train_data)
            test_matches = set(d["match_id"] for d in test_data)

            # No overlap
            assert len(train_matches & test_matches) == 0

            # Test has exactly one match
            assert len(test_matches) == 1


class TestMultiTaskLoss:
    """Tests for multi-task loss computation."""

    def test_compute_multi_task_loss_exists(self):
        """compute_multi_task_loss function should exist."""
        from src.dfl.train import compute_multi_task_loss

        assert compute_multi_task_loss is not None

    def test_compute_multi_task_loss_returns_tensor(self):
        """compute_multi_task_loss should return a scalar tensor."""
        from src.dfl.train import compute_multi_task_loss
        import torch

        predictions = {
            "shot": torch.tensor([[0.5], [0.7]]),
            "goal": torch.tensor([[0.3], [0.1]]),
            "contact": torch.tensor([[0.6, 0.4], [0.3, 0.7]]),
            "outcome": torch.randn(2, 6),
        }

        labels = {
            "shot_binary": torch.tensor([1, 0]),
            "goal_binary": torch.tensor([0, 0]),
            "first_contact_team": torch.tensor([0, 1]),  # 0=attacking, 1=defending
            "outcome_class": torch.tensor([0, 3]),
        }

        loss = compute_multi_task_loss(predictions, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_compute_multi_task_loss_positive(self):
        """Loss should be positive."""
        from src.dfl.train import compute_multi_task_loss
        import torch

        predictions = {
            "shot": torch.tensor([[0.5]]),
            "goal": torch.tensor([[0.5]]),
            "contact": torch.tensor([[0.5, 0.5]]),
            "outcome": torch.randn(1, 6),
        }

        labels = {
            "shot_binary": torch.tensor([1]),
            "goal_binary": torch.tensor([0]),
            "first_contact_team": torch.tensor([0]),
            "outcome_class": torch.tensor([0]),
        }

        loss = compute_multi_task_loss(predictions, labels)

        assert loss > 0


class TestAblationExperiment:
    """Tests for velocity ablation experiment."""

    def test_zero_out_velocity_features_exists(self):
        """zero_out_velocity_features function should exist."""
        from src.dfl.train import zero_out_velocity_features

        assert zero_out_velocity_features is not None

    def test_zero_out_velocity_features_modifies_dataset(self):
        """zero_out_velocity_features should set vx, vy to zero."""
        from src.dfl.train import zero_out_velocity_features
        from torch_geometric.data import Data
        import torch

        # Create dataset with non-zero velocities
        dataset = [
            {
                "graphs": [
                    Data(
                        x=torch.tensor(
                            [
                                [1.0, 2.0, 3.0, 4.0, 0.5, 0.0, 10.0, 5.0],  # vx=3, vy=4
                                [5.0, 6.0, 7.0, 8.0, 1.0, 0.0, 8.0, 3.0],
                            ]
                        ),
                        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                    )
                ],
                "labels": {"shot_binary": 0},
                "match_id": "test",
                "corner_time": 0.0,
            }
        ]

        modified = zero_out_velocity_features(dataset)

        # Check that vx (index 2) and vy (index 3) are now zero
        node_features = modified[0]["graphs"][0].x
        assert torch.all(node_features[:, 2] == 0.0)  # vx
        assert torch.all(node_features[:, 3] == 0.0)  # vy

        # Other features should be unchanged
        assert torch.all(node_features[:, 0] == dataset[0]["graphs"][0].x[:, 0])  # x
        assert torch.all(node_features[:, 1] == dataset[0]["graphs"][0].x[:, 1])  # y

    def test_zero_out_velocity_does_not_modify_original(self):
        """zero_out_velocity_features should not modify original dataset."""
        from src.dfl.train import zero_out_velocity_features
        from torch_geometric.data import Data
        import torch

        original_x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 0.5, 0.0, 10.0, 5.0]]
        )
        dataset = [
            {
                "graphs": [
                    Data(
                        x=original_x.clone(),
                        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                    )
                ],
                "labels": {},
                "match_id": "test",
                "corner_time": 0.0,
            }
        ]

        original_vx = dataset[0]["graphs"][0].x[0, 2].item()

        zero_out_velocity_features(dataset)

        # Original should be unchanged (deep copy inside function)
        assert dataset[0]["graphs"][0].x[0, 2].item() == original_vx

    def test_run_ablation_exists(self):
        """run_ablation function should exist."""
        from src.dfl.train import run_ablation

        assert run_ablation is not None
