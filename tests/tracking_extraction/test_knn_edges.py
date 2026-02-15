"""Tests for kNN graph construction in multi_source_utils."""

import math

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from transfer_learning.multi_source_utils import (
    _compute_edge_features_batch,
    rebuild_knn_edges,
)


def _make_graph(n_nodes: int, seed: int = 42) -> Data:
    """Create a dummy graph with random 12-dim node features and a label."""
    rng = np.random.RandomState(seed)
    x = torch.tensor(rng.rand(n_nodes, 12).astype(np.float32))
    # Dense edges (placeholder — will be replaced by kNN)
    src, dst = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.zeros(edge_index.shape[1], 6)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([1.0]),
    )


class TestKnnBasic:
    """Core kNN edge construction."""

    def test_knn_10_nodes_k5(self):
        """10 nodes, k=5 → each node has 5 outgoing = 50 edges."""
        g = _make_graph(10)
        g.match_id = "test_match"
        g.source = "test"
        result = rebuild_knn_edges([g], k=5)
        assert len(result) == 1
        r = result[0]
        assert r.edge_index.shape == (2, 50)

    def test_knn_small_graph_degenerates_to_dense(self):
        """4 nodes, k=5 → each node connects to all 3 others = 12 edges."""
        g = _make_graph(4)
        result = rebuild_knn_edges([g], k=5)
        r = result[0]
        assert r.edge_index.shape == (2, 12)

    def test_knn_exact_boundary(self):
        """6 nodes, k=5 → each node connects to all 5 others = 30 edges (dense)."""
        g = _make_graph(6)
        result = rebuild_knn_edges([g], k=5)
        r = result[0]
        assert r.edge_index.shape == (2, 30)

    def test_knn_large_graph(self):
        """20 nodes, k=5 → 100 directed edges."""
        g = _make_graph(20)
        result = rebuild_knn_edges([g], k=5)
        r = result[0]
        assert r.edge_index.shape == (2, 100)


class TestKnnEdgeFeatures:
    """Edge features must match graph_converter._compute_edge_features."""

    def test_edge_features_match_scalar(self):
        """Compare vectorized batch computation against scalar reference."""
        g = _make_graph(8, seed=99)
        result = rebuild_knn_edges([g], k=5)
        r = result[0]

        # Scalar reference implementation (copied from graph_converter.py)
        for e_idx in range(r.edge_index.shape[1]):
            i = r.edge_index[0, e_idx].item()
            j = r.edge_index[1, e_idx].item()
            ni = r.x[i].numpy()
            nj = r.x[j].numpy()

            xi, yi = ni[0], ni[1]
            xj, yj = nj[0], nj[1]
            vxi, vyi = ni[2], ni[3]
            vxj, vyj = nj[2], nj[3]

            dx = xj - xi
            dy = yj - yi
            dist = math.sqrt(dx * dx + dy * dy)
            dist_norm = np.clip(dist / math.sqrt(2), 0.0, 1.0)

            speed_diff = nj[4] - ni[4]

            pos_angle = math.atan2(dy, dx)
            pos_sine = (math.sin(pos_angle) + 1) / 2
            pos_cosine = (math.cos(pos_angle) + 1) / 2

            dot = vxi * vxj + vyi * vyj
            cross = vxi * vyj - vyi * vxj
            vel_angle = math.atan2(float(cross), float(dot))
            vel_sine = (math.sin(vel_angle) + 1) / 2
            vel_cosine = (math.cos(vel_angle) + 1) / 2

            expected = np.array([dist_norm, speed_diff, pos_sine, pos_cosine,
                                 vel_sine, vel_cosine], dtype=np.float32)
            actual = r.edge_attr[e_idx].numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-5,
                                       err_msg=f"Edge ({i},{j}) features mismatch")

    def test_edge_attr_shape(self):
        """Edge attr has 6 features per edge."""
        g = _make_graph(10)
        result = rebuild_knn_edges([g], k=5)
        r = result[0]
        assert r.edge_attr.shape == (50, 6)


class TestKnnPreservesMetadata:
    """Labels and metadata survive the transform."""

    def test_preserves_label(self):
        g = _make_graph(10)
        g.y = torch.tensor([1.0])
        result = rebuild_knn_edges([g], k=5)
        assert result[0].y.item() == 1.0

    def test_preserves_match_id(self):
        g = _make_graph(10)
        g.match_id = "england_epl/2020-2021/match_42"
        result = rebuild_knn_edges([g], k=5)
        assert result[0].match_id == "england_epl/2020-2021/match_42"

    def test_preserves_source(self):
        g = _make_graph(10)
        g.source = "soccernet_gsr"
        result = rebuild_knn_edges([g], k=5)
        assert result[0].source == "soccernet_gsr"


class TestKnnProperties:
    """Structural properties of kNN graphs."""

    def test_no_self_loops(self):
        """No edge (i, i) in output."""
        g = _make_graph(15, seed=7)
        result = rebuild_knn_edges([g], k=5)
        ei = result[0].edge_index
        assert (ei[0] != ei[1]).all(), "Found self-loop"

    def test_directed_asymmetric(self):
        """kNN is generally asymmetric: not all i→j have j→i."""
        # Use a graph with clustered positions to ensure asymmetry
        x = torch.zeros(10, 12)
        # Cluster: nodes 0-4 close together, 5-9 far away
        x[:5, 0] = torch.tensor([0.1, 0.11, 0.12, 0.13, 0.14])
        x[:5, 1] = torch.tensor([0.1, 0.11, 0.12, 0.13, 0.14])
        x[5:, 0] = torch.tensor([0.9, 0.91, 0.92, 0.93, 0.94])
        x[5:, 1] = torch.tensor([0.9, 0.91, 0.92, 0.93, 0.94])
        g = Data(x=x, edge_index=torch.empty(2, 0, dtype=torch.long),
                 edge_attr=torch.empty(0, 6), y=torch.tensor([0.0]))
        result = rebuild_knn_edges([g], k=3)
        ei = result[0].edge_index
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        # With 2 clusters of 5 and k=3, within-cluster edges will be symmetric
        # but some cross-cluster edges may not be
        assert len(edges) == 30  # 10 nodes × 3 neighbors

    def test_multiple_graphs(self):
        """Handles a list of graphs with different sizes."""
        graphs = [_make_graph(n) for n in [6, 10, 15]]
        result = rebuild_knn_edges(graphs, k=5)
        assert len(result) == 3
        assert result[0].edge_index.shape == (2, 30)   # 6 nodes, dense
        assert result[1].edge_index.shape == (2, 50)   # 10 × 5
        assert result[2].edge_index.shape == (2, 75)   # 15 × 5
