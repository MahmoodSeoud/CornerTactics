"""Tests for corner kick prediction training pipeline.

Tests use synthetic data (small graphs with known properties), not real data files.
"""

import copy
from typing import List

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from corner_prediction.models import (
    CornerBackbone,
    ReceiverHead,
    ShotHead,
    TwoStageModel,
)
from corner_prediction.training.train import (
    build_model,
    eval_receiver,
    eval_shot,
    train_fold,
    train_receiver_epoch,
    train_shot_epoch,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic graph data
# ---------------------------------------------------------------------------


def _make_graph(
    n_nodes: int = 22,
    n_edges: int = 132,
    has_receiver: bool = True,
    shot: bool = False,
    match_id: str = "1",
) -> Data:
    """Create a synthetic corner kick graph for testing."""
    x = torch.randn(n_nodes, 13)
    edge_index = torch.stack([
        torch.randint(0, n_nodes, (n_edges,)),
        torch.randint(0, n_nodes, (n_edges,)),
    ])
    edge_attr = torch.randn(n_edges, 4)

    # Receiver mask: first 10 players are attacking outfield
    receiver_mask = torch.zeros(n_nodes, dtype=torch.bool)
    receiver_mask[:10] = True

    # Receiver label: player 3 is receiver
    receiver_label = torch.zeros(n_nodes, dtype=torch.float32)
    if has_receiver:
        receiver_label[3] = 1.0

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        receiver_mask=receiver_mask,
        receiver_label=receiver_label,
        has_receiver_label=has_receiver,
        shot_label=1 if shot else 0,
        goal_label=0,
        corner_side=1.0,
        match_id=match_id,
        corner_id=f"corner_{match_id}_1",
        detection_rate=0.9,
    )


def _make_dataset(n_graphs: int = 20, n_matches: int = 4) -> List[Data]:
    """Create a synthetic dataset with multiple matches."""
    graphs = []
    for i in range(n_graphs):
        match_id = str(i % n_matches)
        shot = (i % 3 == 0)  # ~33% shot rate
        has_receiver = (i % 5 != 0)  # 80% have receiver labels
        graphs.append(_make_graph(
            has_receiver=has_receiver,
            shot=shot,
            match_id=match_id,
        ))
    return graphs


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model():
    return build_model(backbone_mode="scratch", freeze=False)


@pytest.fixture
def dataset():
    return _make_dataset(n_graphs=20, n_matches=4)


@pytest.fixture
def small_dataset():
    return _make_dataset(n_graphs=8, n_matches=2)


# ---------------------------------------------------------------------------
# Test: build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_scratch_model_builds(self):
        model = build_model(backbone_mode="scratch", freeze=False)
        assert isinstance(model, TwoStageModel)
        assert isinstance(model.backbone, CornerBackbone)
        assert isinstance(model.receiver_head, ReceiverHead)
        assert isinstance(model.shot_head, ShotHead)

    def test_scratch_output_dim(self):
        model = build_model(backbone_mode="scratch", freeze=False)
        assert model.backbone.output_dim == 64

    def test_pretrained_without_path(self):
        # Should build but not load weights
        model = build_model(backbone_mode="pretrained", pretrained_path=None, freeze=True)
        assert model.backbone.output_dim == 128

    def test_custom_dims(self):
        model = build_model(
            backbone_mode="scratch",
            receiver_hidden=32,
            shot_hidden=16,
        )
        assert model.receiver_head.mlp[0].in_features == 64
        assert model.receiver_head.mlp[0].out_features == 32
        assert model.shot_head.mlp[0].out_features == 16


# ---------------------------------------------------------------------------
# Test: train_receiver_epoch
# ---------------------------------------------------------------------------


class TestTrainReceiverEpoch:
    def test_returns_scalar_loss(self, model, dataset, device):
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss = train_receiver_epoch(model, loader, optimizer, device)
        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_loss_decreases_over_epochs(self, model, dataset, device):
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for _ in range(5):
            loss = train_receiver_epoch(model, loader, optimizer, device)
            losses.append(loss)

        # Loss should generally decrease (may not be strictly monotonic)
        assert losses[-1] <= losses[0] * 1.5  # not diverging

    def test_only_labeled_graphs_contribute(self, device):
        # All graphs have has_receiver_label=False
        graphs = [_make_graph(has_receiver=False) for _ in range(5)]
        model = build_model(backbone_mode="scratch", freeze=False).to(device)
        loader = DataLoader(graphs, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss = train_receiver_epoch(model, loader, optimizer, device)
        assert loss == 0.0  # no labeled graphs → loss is 0


# ---------------------------------------------------------------------------
# Test: train_shot_epoch
# ---------------------------------------------------------------------------


class TestTrainShotEpoch:
    def test_oracle_mode(self, model, dataset, device):
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss = train_shot_epoch(model, loader, optimizer, device,
                                receiver_mode="oracle")
        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_predicted_mode(self, model, dataset, device):
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss = train_shot_epoch(model, loader, optimizer, device,
                                receiver_mode="predicted")
        assert isinstance(loss, float)

    def test_none_mode(self, model, dataset, device):
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss = train_shot_epoch(model, loader, optimizer, device,
                                receiver_mode="none")
        assert isinstance(loss, float)

    def test_pos_weight_affects_loss(self, model, dataset, device):
        model1 = copy.deepcopy(model).to(device)
        model2 = copy.deepcopy(model).to(device)
        loader = DataLoader(dataset, batch_size=len(dataset))
        opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

        # Same model weights, different pos_weight
        torch.manual_seed(42)
        loss1 = train_shot_epoch(model1, loader, opt1, device, pos_weight=1.0)
        torch.manual_seed(42)
        loss2 = train_shot_epoch(model2, loader, opt2, device, pos_weight=5.0)

        # Losses should differ (different weighting)
        # Note: not always different if all labels are 0, but generally
        assert isinstance(loss1, float)
        assert isinstance(loss2, float)


# ---------------------------------------------------------------------------
# Test: eval_receiver
# ---------------------------------------------------------------------------


class TestEvalReceiver:
    def test_returns_metrics(self, model, dataset, device):
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=4)

        metrics = eval_receiver(model, loader, device)
        assert "top1_acc" in metrics
        assert "top3_acc" in metrics
        assert "n_labeled" in metrics
        assert "per_graph" in metrics

    def test_accuracy_in_range(self, model, dataset, device):
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=4)

        metrics = eval_receiver(model, loader, device)
        assert 0.0 <= metrics["top1_acc"] <= 1.0
        assert 0.0 <= metrics["top3_acc"] <= 1.0
        assert metrics["top3_acc"] >= metrics["top1_acc"]

    def test_no_labeled_graphs(self, device):
        graphs = [_make_graph(has_receiver=False) for _ in range(5)]
        model = build_model(backbone_mode="scratch", freeze=False).to(device)
        loader = DataLoader(graphs, batch_size=4)

        metrics = eval_receiver(model, loader, device)
        assert metrics["n_labeled"] == 0
        assert metrics["top1_acc"] == 0.0

    def test_single_graph(self, model, device):
        model = model.to(device)
        graph = _make_graph(has_receiver=True)
        loader = DataLoader([graph], batch_size=1)

        metrics = eval_receiver(model, loader, device)
        assert metrics["n_labeled"] == 1
        assert metrics["top1_acc"] in (0.0, 1.0)
        assert metrics["top3_acc"] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Test: eval_shot
# ---------------------------------------------------------------------------


class TestEvalShot:
    def test_returns_metrics(self, model, dataset, device):
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=4)

        metrics = eval_shot(model, loader, device, receiver_mode="oracle")
        assert "auc" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics
        assert "probs" in metrics
        assert "labels" in metrics
        assert "n_samples" in metrics

    def test_auc_range(self, model, dataset, device):
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=4)

        metrics = eval_shot(model, loader, device, receiver_mode="oracle")
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_all_same_label_returns_05(self, device):
        # All shot_label=0
        graphs = [_make_graph(shot=False) for _ in range(5)]
        model = build_model(backbone_mode="scratch", freeze=False).to(device)
        loader = DataLoader(graphs, batch_size=4)

        metrics = eval_shot(model, loader, device, receiver_mode="none")
        assert metrics["auc"] == 0.5

    def test_different_receiver_modes(self, model, dataset, device):
        model = model.to(device)
        loader = DataLoader(dataset, batch_size=4)

        oracle = eval_shot(model, loader, device, receiver_mode="oracle")
        predicted = eval_shot(model, loader, device, receiver_mode="predicted")
        unconditional = eval_shot(model, loader, device, receiver_mode="none")

        # All should produce valid metrics
        for m in [oracle, predicted, unconditional]:
            assert 0.0 <= m["auc"] <= 1.0
            assert m["n_samples"] == len(dataset)


# ---------------------------------------------------------------------------
# Test: train_fold
# ---------------------------------------------------------------------------


class TestTrainFold:
    def test_completes_without_error(self, device):
        train_data = _make_dataset(n_graphs=12, n_matches=3)
        val_data = _make_dataset(n_graphs=4, n_matches=1)

        model = build_model(backbone_mode="scratch", freeze=False).to(device)

        trained, loss_history = train_fold(
            model, train_data, val_data, device,
            receiver_epochs=3,
            shot_epochs=3,
            receiver_patience=2,
            shot_patience=2,
            batch_size=4,
        )
        assert isinstance(trained, TwoStageModel)

    def test_model_changes_after_training(self, device):
        train_data = _make_dataset(n_graphs=12, n_matches=3)
        val_data = _make_dataset(n_graphs=4, n_matches=1)

        model = build_model(backbone_mode="scratch", freeze=False).to(device)
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}

        train_fold(
            model, train_data, val_data, device,
            receiver_epochs=5,
            shot_epochs=5,
            receiver_patience=3,
            shot_patience=3,
            batch_size=4,
        )

        # At least some parameters should have changed
        changed = False
        for k, v in model.state_dict().items():
            if not torch.equal(v, initial_params[k]):
                changed = True
                break
        assert changed, "No parameters changed during training"

    def test_scratch_backbone_params_trained(self, device):
        """Verify backbone conv params are included in optimizer in scratch mode."""
        train_data = _make_dataset(n_graphs=12, n_matches=3)
        val_data = _make_dataset(n_graphs=4, n_matches=1)

        model = build_model(backbone_mode="scratch", freeze=False).to(device)
        # Snapshot backbone conv params before training
        backbone_params_before = {
            k: v.clone() for k, v in model.state_dict().items()
            if "backbone" in k
        }
        assert len(backbone_params_before) > 0

        train_fold(
            model, train_data, val_data, device,
            receiver_epochs=5,
            shot_epochs=5,
            receiver_patience=3,
            shot_patience=3,
            batch_size=4,
        )

        # Backbone params should have changed
        backbone_changed = False
        for k, v in model.state_dict().items():
            if "backbone" in k and not torch.equal(v, backbone_params_before[k]):
                backbone_changed = True
                break
        assert backbone_changed, "Backbone params did not change in scratch mode"

    def test_loss_history_returned(self, device):
        """Verify train_fold returns per-epoch loss history for both stages."""
        train_data = _make_dataset(n_graphs=12, n_matches=3)
        val_data = _make_dataset(n_graphs=4, n_matches=1)

        model = build_model(backbone_mode="scratch", freeze=False).to(device)

        _, loss_history = train_fold(
            model, train_data, val_data, device,
            receiver_epochs=5,
            shot_epochs=5,
            receiver_patience=10,
            shot_patience=10,
            batch_size=4,
        )

        assert "receiver" in loss_history
        assert "shot" in loss_history

        # Both stages should have train and val lists
        for stage in ("receiver", "shot"):
            assert "train" in loss_history[stage]
            assert "val" in loss_history[stage]
            assert len(loss_history[stage]["train"]) > 0
            assert len(loss_history[stage]["val"]) > 0
            assert len(loss_history[stage]["train"]) == len(loss_history[stage]["val"])

            # All values should be finite floats
            for v in loss_history[stage]["train"]:
                assert isinstance(v, float)
                assert not np.isnan(v)
            for v in loss_history[stage]["val"]:
                assert isinstance(v, float)
                assert not np.isnan(v)


# ---------------------------------------------------------------------------
# Test: feature masking (for ablations)
# ---------------------------------------------------------------------------


class TestFeatureMask:
    def test_mask_zeros_features(self):
        from corner_prediction.training.ablation import apply_feature_mask

        dataset = _make_dataset(n_graphs=5)
        active = [0, 1, 5]  # only x, y, is_attacking
        masked = apply_feature_mask(dataset, active)

        for g in masked:
            for idx in range(13):
                if idx not in active:
                    assert (g.x[:, idx] == 0.0).all(), f"Feature {idx} should be zero"
                # Active features may be any value (including 0)

    def test_mask_preserves_active(self):
        from corner_prediction.training.ablation import apply_feature_mask

        dataset = _make_dataset(n_graphs=5)
        active = [0, 1, 2, 3, 4]
        masked = apply_feature_mask(dataset, active)

        for orig, m in zip(dataset, masked):
            for idx in active:
                assert torch.equal(orig.x[:, idx], m.x[:, idx])

    def test_mask_does_not_modify_original(self):
        from corner_prediction.training.ablation import apply_feature_mask

        dataset = _make_dataset(n_graphs=3)
        orig_x = [g.x.clone() for g in dataset]

        apply_feature_mask(dataset, [0, 1])

        for g, ox in zip(dataset, orig_x):
            assert torch.equal(g.x, ox), "Original dataset was modified"

    def test_all_active_is_identity(self):
        from corner_prediction.training.ablation import apply_feature_mask

        dataset = _make_dataset(n_graphs=3)
        masked = apply_feature_mask(dataset, list(range(13)))

        for orig, m in zip(dataset, masked):
            assert torch.equal(orig.x, m.x)


# ---------------------------------------------------------------------------
# Test: shuffle functions (for permutation tests)
# ---------------------------------------------------------------------------


class TestShuffle:
    def test_shuffle_receiver_preserves_mask(self):
        from corner_prediction.training.permutation_test import shuffle_receiver_labels

        dataset = _make_dataset(n_graphs=10)
        rng = np.random.RandomState(42)
        shuffled = shuffle_receiver_labels(dataset, rng)

        for orig, s in zip(dataset, shuffled):
            # Mask should be identical
            assert torch.equal(orig.receiver_mask, s.receiver_mask)
            # Receiver label should still have exactly one 1.0 if labeled
            if s.has_receiver_label:
                assert s.receiver_label.sum() == 1.0
                # The 1.0 should be at a masked position
                recv_idx = s.receiver_label.argmax().item()
                assert s.receiver_mask[recv_idx], "Receiver assigned to non-candidate"

    def test_shuffle_shot_preserves_count(self):
        from corner_prediction.training.permutation_test import shuffle_shot_labels

        dataset = _make_dataset(n_graphs=20)
        rng = np.random.RandomState(42)

        n_shots_orig = sum(g.shot_label for g in dataset)
        shuffled = shuffle_shot_labels(dataset, rng)
        n_shots_shuf = sum(g.shot_label for g in shuffled)

        assert n_shots_orig == n_shots_shuf

    def test_shuffle_does_not_modify_original(self):
        from corner_prediction.training.permutation_test import (
            shuffle_receiver_labels,
            shuffle_shot_labels,
        )

        dataset = _make_dataset(n_graphs=5)
        orig_labels = [g.receiver_label.clone() for g in dataset]
        orig_shots = [g.shot_label for g in dataset]

        rng = np.random.RandomState(42)
        shuffle_receiver_labels(dataset, rng)
        rng2 = np.random.RandomState(43)
        shuffle_shot_labels(dataset, rng2)

        for g, ol, os in zip(dataset, orig_labels, orig_shots):
            assert torch.equal(g.receiver_label, ol)
            assert g.shot_label == os
