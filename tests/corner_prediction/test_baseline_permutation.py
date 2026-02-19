"""Tests for corner_prediction.baselines.permutation_test_baselines."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from corner_prediction.baselines.permutation_test_baselines import (
    permutation_test_baseline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(shot_label: int = 0, match_id: str = "100") -> Data:
    """Minimal graph with the attributes baselines read."""
    n = 22
    x = torch.randn(n, 13)
    x[:, 5] = 0.0
    x[:11, 5] = 1.0  # attacking team
    x[:, 7] = 0.0
    x[0, 7] = 1.0    # atk GK
    x[11, 7] = 1.0   # def GK

    edge_index = torch.tensor(
        [list(range(n)), [(i + 1) % n for i in range(n)]], dtype=torch.long,
    )
    edge_attr = torch.randn(n, 4)

    receiver_mask = torch.zeros(n, dtype=torch.bool)
    receiver_mask[1:11] = True
    receiver_label = torch.zeros(n, dtype=torch.float32)
    receiver_label[3] = 1.0

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        receiver_mask=receiver_mask,
        receiver_label=receiver_label,
        has_receiver_label=True,
        shot_label=shot_label,
        goal_label=0,
        corner_side=0.0,
        match_id=match_id,
        corner_id=1,
        detection_rate=0.8,
    )


def _make_dataset(n_graphs: int = 20, n_matches: int = 4) -> list:
    match_ids = [str(1000 + i) for i in range(n_matches)]
    graphs = []
    for i in range(n_graphs):
        mid = match_ids[i % n_matches]
        shot = 1 if i % 3 == 0 else 0
        graphs.append(_make_graph(shot_label=shot, match_id=mid))
    return graphs


def _dummy_baseline_lomo(dataset, seed=42, verbose=False, **kwargs):
    """Fast dummy LOMO that returns a plausible result dict.

    Computes a trivial AUC proxy from the dataset labels so that
    shuffled labels produce a different metric value.
    """
    labels = [g.shot_label for g in dataset]
    n_pos = sum(labels)
    n_total = len(labels)
    # Deterministic but label-dependent "AUC"
    rng = np.random.RandomState(seed)
    auc = float(np.clip(n_pos / max(n_total, 1) + rng.uniform(-0.1, 0.1), 0.0, 1.0))

    captured_kwargs.update(kwargs)

    return {
        "config": {"baseline": "dummy", "seed": seed, "n_folds": 2},
        "per_fold": [],
        "aggregated": {
            "receiver": {
                "top1_mean": 0.0, "top1_std": 0.0,
                "top3_mean": 0.0, "top3_std": 0.0,
                "n_folds": 0, "per_fold_top1": [], "per_fold_top3": [],
            },
            "shot_oracle": {
                "auc_mean": auc, "auc_std": 0.1,
                "f1_mean": 0.3, "f1_std": 0.1,
                "n_folds": 2, "per_fold_auc": [auc, auc],
                "per_fold_f1": [0.3, 0.3],
            },
            "shot_predicted": {
                "auc_mean": auc, "auc_std": 0.1,
                "f1_mean": 0.3, "f1_std": 0.1,
                "n_folds": 2, "per_fold_auc": [auc, auc],
                "per_fold_f1": [0.3, 0.3],
            },
            "shot_unconditional": {
                "auc_mean": auc, "auc_std": 0.1,
                "f1_mean": 0.3, "f1_std": 0.1,
                "n_folds": 2, "per_fold_auc": [auc, auc],
                "per_fold_f1": [0.3, 0.3],
            },
        },
    }


# Global dict to capture kwargs forwarded to dummy
captured_kwargs: dict = {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPermutationTestBaseline:
    def setup_method(self):
        captured_kwargs.clear()

    def test_output_format_matches_gnn(self):
        """Output dict must have the same keys as GNN permutation tests."""
        dataset = _make_dataset(n_graphs=12, n_matches=3)
        result = permutation_test_baseline(
            dataset,
            baseline_fn=_dummy_baseline_lomo,
            baseline_name="dummy",
            n_permutations=5,
            seed=42,
            verbose=False,
        )

        required_keys = {
            "metric", "real_metric", "null_distribution",
            "null_mean", "null_std", "p_value",
            "n_permutations", "significant",
        }
        assert required_keys.issubset(result.keys())
        assert result["baseline"] == "dummy"

    def test_p_value_bounds(self):
        """p-value must be in (0, 1]."""
        dataset = _make_dataset(n_graphs=12, n_matches=3)
        result = permutation_test_baseline(
            dataset,
            baseline_fn=_dummy_baseline_lomo,
            baseline_name="dummy",
            n_permutations=5,
            seed=42,
            verbose=False,
        )

        assert 0.0 < result["p_value"] <= 1.0

    def test_null_distribution_length(self):
        """Null distribution should have n_permutations entries."""
        n_perms = 7
        dataset = _make_dataset(n_graphs=12, n_matches=3)
        result = permutation_test_baseline(
            dataset,
            baseline_fn=_dummy_baseline_lomo,
            baseline_name="dummy",
            n_permutations=n_perms,
            seed=42,
            verbose=False,
        )

        assert len(result["null_distribution"]) == n_perms
        assert result["n_permutations"] == n_perms

    def test_significant_flag_consistent(self):
        """significant flag must match p_value < 0.05."""
        dataset = _make_dataset(n_graphs=12, n_matches=3)
        result = permutation_test_baseline(
            dataset,
            baseline_fn=_dummy_baseline_lomo,
            baseline_name="dummy",
            n_permutations=5,
            seed=42,
            verbose=False,
        )

        assert result["significant"] == (result["p_value"] < 0.05)

    def test_kwargs_forwarded(self):
        """Extra kwargs should reach the baseline function."""
        dataset = _make_dataset(n_graphs=12, n_matches=3)
        permutation_test_baseline(
            dataset,
            baseline_fn=_dummy_baseline_lomo,
            baseline_name="dummy",
            n_permutations=2,
            seed=42,
            verbose=False,
            device=torch.device("cpu"),
            hidden_dim=32,
        )

        assert captured_kwargs.get("device") == torch.device("cpu")
        assert captured_kwargs.get("hidden_dim") == 32

    def test_real_metric_is_float(self):
        dataset = _make_dataset(n_graphs=12, n_matches=3)
        result = permutation_test_baseline(
            dataset,
            baseline_fn=_dummy_baseline_lomo,
            baseline_name="dummy",
            n_permutations=3,
            seed=42,
            verbose=False,
        )

        assert isinstance(result["real_metric"], float)
        assert isinstance(result["null_mean"], float)
        assert isinstance(result["null_std"], float)

    def test_metric_name_includes_baseline(self):
        dataset = _make_dataset(n_graphs=12, n_matches=3)
        result = permutation_test_baseline(
            dataset,
            baseline_fn=_dummy_baseline_lomo,
            baseline_name="my_model",
            n_permutations=2,
            seed=42,
            verbose=False,
        )

        assert "my_model" in result["metric"]
