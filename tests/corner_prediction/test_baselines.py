"""Tests for corner_prediction.baselines — random, heuristic, XGBoost, MLP."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from corner_prediction.baselines.heuristic_receiver import (
    GOAL_X,
    GOAL_Y,
    _heuristic_receiver_fold,
)
from corner_prediction.baselines.mlp_baseline import (
    FLAT_DIM,
    ShotMLP,
    _build_tensors,
    _flatten_graph,
)
from corner_prediction.baselines.random_baseline import (
    _random_receiver_fold,
    _random_shot_fold,
)
from corner_prediction.baselines.xgboost_baseline import (
    FEATURE_NAMES,
    extract_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(
    n_nodes: int = 22,
    n_attackers: int = 11,
    receiver_idx: int = 3,
    shot_label: int = 0,
    has_receiver: bool = True,
    match_id: str = "100",
) -> Data:
    """Create a synthetic corner kick graph."""
    x = torch.randn(n_nodes, 13)

    # Set team flags: first n_attackers are attacking
    x[:, 5] = 0.0
    x[:n_attackers, 5] = 1.0

    # Set goalkeeper flags: node 0 (atk GK), node n_attackers (def GK)
    x[:, 7] = 0.0
    x[0, 7] = 1.0
    if n_attackers < n_nodes:
        x[n_attackers, 7] = 1.0

    # Simple edges
    src = list(range(n_nodes))
    dst = [(i + 1) % n_nodes for i in range(n_nodes)]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.randn(n_nodes, 4)

    # Receiver mask: attacking outfield only (exclude node 0 = atk GK)
    receiver_mask = torch.zeros(n_nodes, dtype=torch.bool)
    for i in range(1, n_attackers):
        receiver_mask[i] = True

    receiver_label = torch.zeros(n_nodes, dtype=torch.float32)
    if has_receiver and receiver_mask[receiver_idx]:
        receiver_label[receiver_idx] = 1.0

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        receiver_mask=receiver_mask,
        receiver_label=receiver_label,
        has_receiver_label=has_receiver,
        shot_label=shot_label,
        goal_label=0,
        corner_side=0.0,
        match_id=match_id,
        corner_id=1,
        detection_rate=0.8,
    )


def _make_dataset(n_graphs: int = 20, n_matches: int = 4) -> list:
    """Create a synthetic dataset with multiple matches."""
    match_ids = [str(1000 + i) for i in range(n_matches)]
    graphs = []
    for i in range(n_graphs):
        mid = match_ids[i % n_matches]
        shot = 1 if i % 3 == 0 else 0  # ~33% shot rate
        g = _make_graph(
            match_id=mid,
            shot_label=shot,
            receiver_idx=1 + (i % 10),
        )
        graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
# Random baseline tests
# ---------------------------------------------------------------------------

class TestRandomBaseline:
    def test_random_receiver_returns_correct_format(self):
        graphs = [_make_graph() for _ in range(5)]
        rng = np.random.RandomState(42)
        result = _random_receiver_fold(graphs, rng)

        assert "top1_acc" in result
        assert "top3_acc" in result
        assert "n_labeled" in result
        assert "per_graph" in result
        assert result["n_labeled"] == 5

    def test_random_receiver_top1_near_chance(self):
        """With 10 candidates, random top-1 should be ~0.10."""
        graphs = [_make_graph() for _ in range(20)]
        rng = np.random.RandomState(42)
        result = _random_receiver_fold(graphs, rng)

        # 10 candidates → expected 0.10, allow tolerance
        assert 0.0 <= result["top1_acc"] <= 0.5
        assert result["top3_acc"] > result["top1_acc"]

    def test_random_receiver_no_labeled(self):
        graphs = [_make_graph(has_receiver=False) for _ in range(3)]
        rng = np.random.RandomState(42)
        result = _random_receiver_fold(graphs, rng)
        assert result["n_labeled"] == 0

    def test_random_shot_constant_predictor(self):
        train = [_make_graph(shot_label=(1 if i < 3 else 0)) for i in range(10)]
        test = [_make_graph(shot_label=(1 if i < 2 else 0)) for i in range(5)]
        rng = np.random.RandomState(42)
        result = _random_shot_fold(train, test, rng)

        assert result["auc"] == 0.5  # Constant predictor
        assert result["n_samples"] == 5
        assert 0.0 <= result["f1"] <= 1.0


# ---------------------------------------------------------------------------
# Heuristic receiver tests
# ---------------------------------------------------------------------------

class TestHeuristicReceiver:
    def test_picks_nearest_to_goal(self):
        """Player closest to goal center should be predicted receiver."""
        g = _make_graph(receiver_idx=3)

        # Place node 3 very close to goal (1.0, 0.0)
        g.x[3, 0] = 0.95  # x close to goal
        g.x[3, 1] = 0.0   # y at center

        # Place all other attackers far from goal
        for i in range(1, 11):
            if i != 3:
                g.x[i, 0] = -0.5
                g.x[i, 1] = 0.5

        result = _heuristic_receiver_fold([g])
        assert result["n_labeled"] == 1
        assert result["per_graph"][0]["top1"] is True

    def test_top3_includes_3_nearest(self):
        """Top-3 should include the 3 nearest to goal."""
        g = _make_graph(receiver_idx=2)

        # Place nodes 1, 2, 3 close to goal
        for i, dist in [(1, 0.1), (2, 0.15), (3, 0.2)]:
            g.x[i, 0] = GOAL_X - dist
            g.x[i, 1] = 0.0

        # Others far away
        for i in range(4, 11):
            g.x[i, 0] = -0.5
            g.x[i, 1] = 0.5

        result = _heuristic_receiver_fold([g])
        assert result["per_graph"][0]["top3"] is True

    def test_returns_correct_format(self):
        graphs = [_make_graph() for _ in range(3)]
        result = _heuristic_receiver_fold(graphs)
        assert "top1_acc" in result
        assert "top3_acc" in result
        assert result["n_labeled"] == 3


# ---------------------------------------------------------------------------
# XGBoost tests
# ---------------------------------------------------------------------------

class TestXGBoostBaseline:
    def test_feature_extraction_shape(self):
        g = _make_graph()
        features = extract_features(g)
        assert features.shape == (len(FEATURE_NAMES),)
        assert not np.any(np.isnan(features))

    def test_feature_extraction_consistency(self):
        """Same graph should produce same features."""
        g = _make_graph()
        f1 = extract_features(g)
        f2 = extract_features(g)
        np.testing.assert_array_equal(f1, f2)

    def test_corner_side_in_features(self):
        g = _make_graph()
        g.corner_side = 1.0
        features = extract_features(g)
        # corner_side is feature index 25
        idx = FEATURE_NAMES.index("corner_side")
        assert features[idx] == 1.0

    def test_detection_rate_in_features(self):
        g = _make_graph()
        g.detection_rate = 0.75
        features = extract_features(g)
        idx = FEATURE_NAMES.index("detection_rate")
        assert features[idx] == 0.75


# ---------------------------------------------------------------------------
# MLP tests
# ---------------------------------------------------------------------------

class TestMLPBaseline:
    def test_mlp_forward_pass(self):
        model = ShotMLP(input_dim=FLAT_DIM, hidden_dim=64, dropout=0.3)
        x = torch.randn(4, FLAT_DIM)
        out = model(x)
        assert out.shape == (4, 1)

    def test_flatten_graph(self):
        g = _make_graph(n_nodes=22)
        flat = _flatten_graph(g)
        assert flat.shape == (FLAT_DIM,)  # 22 * 13 = 286

    def test_flatten_graph_padding(self):
        """Graphs with fewer nodes should be zero-padded."""
        g = _make_graph(n_nodes=20, n_attackers=10, receiver_idx=3)
        flat = _flatten_graph(g)
        assert flat.shape == (FLAT_DIM,)
        # Last 2*13=26 entries should be zeros (padding)
        assert np.all(flat[-26:] == 0.0)

    def test_build_tensors(self):
        graphs = [_make_graph(shot_label=(1 if i < 3 else 0)) for i in range(5)]
        X, y = _build_tensors(graphs)
        assert X.shape == (5, FLAT_DIM)
        assert y.shape == (5,)
        assert y.sum() == 3  # 3 shots

    def test_mlp_gradients_flow(self):
        model = ShotMLP(input_dim=FLAT_DIM)
        x = torch.randn(2, FLAT_DIM)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# Integration: result format compatibility
# ---------------------------------------------------------------------------

class TestResultFormat:
    """Verify all baselines return compatible result dicts."""

    def _check_result_structure(self, results: dict):
        assert "config" in results
        assert "per_fold" in results
        assert "aggregated" in results

        agg = results["aggregated"]
        assert "receiver" in agg
        assert "shot_oracle" in agg

        r = agg["receiver"]
        assert "top1_mean" in r
        assert "top3_mean" in r

        s = agg["shot_oracle"]
        assert "auc_mean" in s
        assert "f1_mean" in s

    def test_random_baseline_format(self):
        dataset = _make_dataset(n_graphs=8, n_matches=2)
        from corner_prediction.baselines.random_baseline import random_baseline_lomo
        results = random_baseline_lomo(dataset, seed=42, verbose=False)
        self._check_result_structure(results)
        assert results["config"]["baseline"] == "random"

    def test_heuristic_baseline_format(self):
        dataset = _make_dataset(n_graphs=8, n_matches=2)
        from corner_prediction.baselines.heuristic_receiver import heuristic_receiver_lomo
        results = heuristic_receiver_lomo(dataset, seed=42, verbose=False)
        self._check_result_structure(results)
        assert results["config"]["baseline"] == "heuristic_receiver"

    def test_xgboost_baseline_format(self):
        dataset = _make_dataset(n_graphs=12, n_matches=3)
        from corner_prediction.baselines.xgboost_baseline import xgboost_baseline_lomo
        results = xgboost_baseline_lomo(dataset, seed=42, verbose=False)
        self._check_result_structure(results)
        assert results["config"]["baseline"] == "xgboost"

    def test_mlp_baseline_format(self):
        dataset = _make_dataset(n_graphs=12, n_matches=3)
        from corner_prediction.baselines.mlp_baseline import mlp_baseline_lomo
        results = mlp_baseline_lomo(
            dataset, seed=42, device=torch.device("cpu"), verbose=False,
        )
        self._check_result_structure(results)
        assert results["config"]["baseline"] == "mlp"
