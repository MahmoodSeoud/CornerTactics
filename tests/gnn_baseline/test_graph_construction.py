"""Tests for graph construction from StatsBomb freeze-frame data.

TDD: Write tests first, then implement to make them pass.
"""

import pytest
import torch
import numpy as np


class TestCornerToGraph:
    """Test conversion of corner freeze-frame data to PyTorch Geometric graph."""

    @pytest.fixture
    def sample_corner(self):
        """Sample corner data in StatsBomb format."""
        return {
            "match_id": "12345",
            "event": {
                "id": "test-corner-1",
                "location": [120.0, 80.0],  # Corner from right side
            },
            "freeze_frame": [
                # Attacking team (teammate=True)
                {"location": [108.0, 35.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [110.0, 40.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [112.0, 38.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [105.0, 42.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [120.0, 80.0], "teammate": True, "keeper": False, "actor": True},  # Corner taker
                # Defending team (teammate=False)
                {"location": [115.0, 36.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [116.0, 42.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [113.0, 40.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [118.0, 38.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [120.0, 40.0], "teammate": False, "keeper": True, "actor": False},  # Goalkeeper
            ],
            "shot_outcome": 1,
        }

    def test_corner_to_graph_returns_data_object(self, sample_corner):
        """Graph construction should return a PyTorch Geometric Data object."""
        from experiments.gnn_baseline.graph_construction import corner_to_graph

        graph = corner_to_graph(sample_corner)

        # Should return a PyG Data object
        from torch_geometric.data import Data
        assert isinstance(graph, Data)

    def test_node_features_shape(self, sample_corner):
        """Node features should have correct shape: [num_players, num_features]."""
        from experiments.gnn_baseline.graph_construction import corner_to_graph

        graph = corner_to_graph(sample_corner)

        # Should have 10 nodes (players) with 5 features each
        # Features: x, y, team_indicator, dist_to_goal, dist_to_ball
        assert graph.x.shape[0] == 10  # 10 players
        assert graph.x.shape[1] == 5   # 5 features

    def test_node_features_contain_positions(self, sample_corner):
        """Node features should contain normalized x, y positions."""
        from experiments.gnn_baseline.graph_construction import corner_to_graph

        graph = corner_to_graph(sample_corner)

        # First two features should be positions (normalized 0-1)
        positions = graph.x[:, :2]
        assert positions.min() >= 0.0
        assert positions.max() <= 1.0

    def test_node_features_team_indicator(self, sample_corner):
        """Node features should include team indicator (0=defender, 1=attacker)."""
        from experiments.gnn_baseline.graph_construction import corner_to_graph

        graph = corner_to_graph(sample_corner)

        # Team indicator is the 3rd feature (index 2)
        team_indicators = graph.x[:, 2]
        # Should have both 0s and 1s
        assert 0.0 in team_indicators
        assert 1.0 in team_indicators
        # First 5 players are attackers (teammate=True)
        assert team_indicators[:5].sum() == 5.0
        # Last 5 are defenders
        assert team_indicators[5:].sum() == 0.0

    def test_edge_index_shape(self, sample_corner):
        """Edge index should have shape [2, num_edges]."""
        from experiments.gnn_baseline.graph_construction import corner_to_graph

        graph = corner_to_graph(sample_corner)

        # Should have shape [2, num_edges]
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0  # Should have some edges

    def test_edge_attr_matches_edges(self, sample_corner):
        """Edge attributes should match number of edges."""
        from experiments.gnn_baseline.graph_construction import corner_to_graph

        graph = corner_to_graph(sample_corner)

        num_edges = graph.edge_index.shape[1]
        assert graph.edge_attr.shape[0] == num_edges

    def test_label_is_tensor(self, sample_corner):
        """Graph should have label as tensor."""
        from experiments.gnn_baseline.graph_construction import corner_to_graph

        graph = corner_to_graph(sample_corner)

        assert hasattr(graph, 'y')
        assert isinstance(graph.y, torch.Tensor)
        assert graph.y.item() == 1  # shot_outcome = 1


class TestKNNEdges:
    """Test k-nearest neighbor edge construction."""

    @pytest.fixture
    def positions(self):
        """Simple positions for testing."""
        return torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ])

    def test_knn_edges_correct_k(self, positions):
        """k-NN should connect each node to exactly k neighbors."""
        from experiments.gnn_baseline.graph_construction import create_knn_edges

        k = 2
        edge_index = create_knn_edges(positions, k=k)

        # Each node should have k outgoing edges
        # Total edges = n * k (since we use directed k-NN)
        assert edge_index.shape[1] == len(positions) * k

    def test_knn_edges_no_self_loops(self, positions):
        """k-NN edges should not include self-loops."""
        from experiments.gnn_baseline.graph_construction import create_knn_edges

        edge_index = create_knn_edges(positions, k=2)

        # Check no self-loops: source != target
        sources, targets = edge_index[0], edge_index[1]
        assert not torch.any(sources == targets)

    def test_knn_edges_nearest_neighbors(self, positions):
        """k-NN should connect to actual nearest neighbors."""
        from experiments.gnn_baseline.graph_construction import create_knn_edges

        edge_index = create_knn_edges(positions, k=1)

        # Node 0 at (0,0) should connect to node 1 at (1,0) or node 2 at (0,1)
        # Both are distance 1.0, node 3 at (2,0) is distance 2.0
        node_0_neighbors = edge_index[1, edge_index[0] == 0]
        assert node_0_neighbors[0].item() in [1, 2]


class TestEdgeFeatures:
    """Test edge feature computation."""

    @pytest.fixture
    def sample_corner(self):
        """Sample corner for edge feature testing."""
        return {
            "match_id": "12345",
            "event": {"id": "test", "location": [120.0, 0.0]},
            "freeze_frame": [
                {"location": [100.0, 40.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [110.0, 40.0], "teammate": False, "keeper": False, "actor": False},
            ],
            "shot_outcome": 0,
        }

    def test_edge_features_distance(self, sample_corner):
        """Edge features should include Euclidean distance."""
        from experiments.gnn_baseline.graph_construction import corner_to_graph

        graph = corner_to_graph(sample_corner)

        # Edge features: [distance, angle_to_goal]
        assert graph.edge_attr.shape[1] == 2

        # Distance should be positive
        distances = graph.edge_attr[:, 0]
        assert (distances > 0).all()


class TestGraphStructureVariations:
    """Test different graph structure options."""

    @pytest.fixture
    def sample_corner(self):
        """Sample corner data."""
        return {
            "match_id": "12345",
            "event": {"id": "test", "location": [120.0, 0.0]},
            "freeze_frame": [
                {"location": [100.0, 30.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [105.0, 35.0], "teammate": True, "keeper": False, "actor": False},
                {"location": [108.0, 40.0], "teammate": False, "keeper": False, "actor": False},
                {"location": [112.0, 38.0], "teammate": False, "keeper": True, "actor": False},
            ],
            "shot_outcome": 1,
        }

    def test_full_connectivity(self, sample_corner):
        """Full connectivity should create edges between all node pairs."""
        from experiments.gnn_baseline.graph_construction import corner_to_graph

        graph = corner_to_graph(sample_corner, edge_type='full')

        n = 4  # 4 players
        # Full connectivity: n * (n-1) edges (no self-loops)
        expected_edges = n * (n - 1)
        assert graph.edge_index.shape[1] == expected_edges

    def test_knn_connectivity(self, sample_corner):
        """k-NN should limit edges to k neighbors per node."""
        from experiments.gnn_baseline.graph_construction import corner_to_graph

        k = 2
        graph = corner_to_graph(sample_corner, edge_type='knn', k=k)

        n = 4
        # k-NN: each node has k outgoing edges
        expected_edges = n * k
        assert graph.edge_index.shape[1] == expected_edges


class TestDatasetConstruction:
    """Test dataset construction from multiple corners."""

    def test_build_dataset_returns_list(self):
        """Dataset builder should return list of graphs."""
        from experiments.gnn_baseline.graph_construction import build_graph_dataset

        corners = [
            {
                "match_id": "1",
                "event": {"id": "c1", "location": [120.0, 0.0]},
                "freeze_frame": [
                    {"location": [100.0, 40.0], "teammate": True, "keeper": False, "actor": False},
                    {"location": [110.0, 40.0], "teammate": False, "keeper": True, "actor": False},
                ],
                "shot_outcome": 0,
            },
            {
                "match_id": "2",
                "event": {"id": "c2", "location": [120.0, 80.0]},
                "freeze_frame": [
                    {"location": [105.0, 35.0], "teammate": True, "keeper": False, "actor": True},
                    {"location": [115.0, 42.0], "teammate": False, "keeper": True, "actor": False},
                ],
                "shot_outcome": 1,
            },
        ]

        dataset = build_graph_dataset(corners)

        assert len(dataset) == 2
        assert dataset[0].y.item() == 0
        assert dataset[1].y.item() == 1
