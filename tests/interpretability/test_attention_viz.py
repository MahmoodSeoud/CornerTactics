"""Tests for Attention Visualization.

TDD: Write tests first, then implement to make them pass.

Visualize attention weights from GAT models to understand
which player relationships the model focuses on.
"""

import pytest
import torch
import numpy as np


class TestAttentionExtractor:
    """Test extraction of attention weights from GAT."""

    def test_instantiation(self):
        """AttentionExtractor should instantiate with a GAT model."""
        from experiments.interpretability.attention_viz import AttentionExtractor
        from experiments.gnn_baseline.models import GATModel

        model = GATModel(in_channels=5, hidden_channels=64, num_layers=2, heads=4)
        extractor = AttentionExtractor(model)

        assert extractor is not None

    def test_extract_returns_attention_weights(self):
        """Should extract attention weights for a batch using explicit method."""
        from experiments.interpretability.attention_viz import AttentionExtractor
        from experiments.gnn_baseline.models import GATModel
        from torch_geometric.data import Data, Batch

        model = GATModel(in_channels=5, hidden_channels=64, num_layers=2, heads=4)
        extractor = AttentionExtractor(model)

        # Create sample graph
        x = torch.randn(10, 5)
        edge_index = torch.randint(0, 10, (2, 30))
        graph = Data(x=x, edge_index=edge_index)
        batch = Batch.from_data_list([graph])

        # Use extract_with_attention for reliable extraction
        _, attention_weights = extractor.extract_with_attention(batch)

        assert attention_weights is not None
        # Should be a dict with layer keys
        assert isinstance(attention_weights, dict)
        assert len(attention_weights) > 0

    def test_attention_weights_shape(self):
        """Attention weights should have correct shape."""
        from experiments.interpretability.attention_viz import AttentionExtractor
        from experiments.gnn_baseline.models import GATModel
        from torch_geometric.data import Data, Batch

        model = GATModel(in_channels=5, hidden_channels=64, num_layers=2, heads=4)
        extractor = AttentionExtractor(model)

        x = torch.randn(10, 5)
        edge_index = torch.randint(0, 10, (2, 30))
        graph = Data(x=x, edge_index=edge_index)
        batch = Batch.from_data_list([graph])

        attention_weights = extractor.extract(batch)

        # Each layer should have weights of shape [num_edges, num_heads]
        for layer_name, weights in attention_weights.items():
            assert weights.ndim == 2  # [num_edges, num_heads]


class TestAttentionAggregation:
    """Test aggregation of attention weights."""

    def test_aggregate_per_node(self):
        """Should aggregate attention to per-node importance."""
        from experiments.interpretability.attention_viz import aggregate_attention_per_node

        # Mock attention weights: 20 edges, 4 heads
        edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 2, 0, 2, 0]])
        attention = torch.randn(5, 4)

        node_importance = aggregate_attention_per_node(
            edge_index, attention, num_nodes=3
        )

        assert node_importance.shape == (3,)  # One score per node

    def test_aggregate_per_edge(self):
        """Should aggregate attention across heads per edge."""
        from experiments.interpretability.attention_viz import aggregate_attention_per_edge

        attention = torch.randn(20, 4)  # 20 edges, 4 heads

        edge_importance = aggregate_attention_per_edge(attention)

        assert edge_importance.shape == (20,)  # One score per edge


class TestAttentionVisualization:
    """Test attention visualization utilities."""

    @pytest.fixture
    def sample_corner_graph(self):
        """Create a sample corner graph with positions."""
        # 10 players with positions
        positions = torch.rand(10, 2) * torch.tensor([120.0, 80.0])
        team_indicators = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32)

        x = torch.randn(10, 5)  # Node features
        edge_index = torch.randint(0, 10, (2, 30))

        return {
            'x': x,
            'edge_index': edge_index,
            'positions': positions,
            'team_indicators': team_indicators,
        }

    def test_create_attention_plot_data(self, sample_corner_graph):
        """Should create data suitable for visualization."""
        from experiments.interpretability.attention_viz import create_attention_plot_data

        attention = torch.randn(30, 4)  # attention for 30 edges

        plot_data = create_attention_plot_data(
            positions=sample_corner_graph['positions'],
            edge_index=sample_corner_graph['edge_index'],
            attention_weights=attention,
            team_indicators=sample_corner_graph['team_indicators'],
        )

        assert 'node_positions' in plot_data
        assert 'edge_weights' in plot_data
        assert 'team_colors' in plot_data

    def test_plot_attention_on_pitch(self, sample_corner_graph):
        """Should create matplotlib figure of attention on pitch."""
        from experiments.interpretability.attention_viz import plot_attention_on_pitch

        attention = torch.randn(30, 4)

        fig = plot_attention_on_pitch(
            positions=sample_corner_graph['positions'],
            edge_index=sample_corner_graph['edge_index'],
            attention_weights=attention,
            team_indicators=sample_corner_graph['team_indicators'],
        )

        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestAttentionComparison:
    """Test comparison of attention patterns between classes."""

    @pytest.fixture
    def sample_batches(self):
        """Create sample batches for shot and no-shot corners."""
        from torch_geometric.data import Data, Batch

        def make_batch(n_graphs, label):
            graphs = []
            for _ in range(n_graphs):
                x = torch.randn(10, 5)
                edge_index = torch.randint(0, 10, (2, 30))
                y = torch.tensor([label], dtype=torch.float32)
                pos = torch.rand(10, 2) * torch.tensor([120.0, 80.0])
                graphs.append(Data(x=x, edge_index=edge_index, y=y, pos=pos))
            return Batch.from_data_list(graphs)

        return {
            'shot': make_batch(5, 1),
            'no_shot': make_batch(5, 0),
        }

    def test_compare_attention_patterns(self, sample_batches):
        """Should compare attention between shot and no-shot corners."""
        from experiments.interpretability.attention_viz import compare_attention_patterns
        from experiments.gnn_baseline.models import GATModel

        model = GATModel(in_channels=5, hidden_channels=64, num_layers=2, heads=4)

        comparison = compare_attention_patterns(
            model=model,
            shot_batch=sample_batches['shot'],
            no_shot_batch=sample_batches['no_shot'],
        )

        assert 'shot_mean_attention' in comparison
        assert 'no_shot_mean_attention' in comparison
        assert 'attention_difference' in comparison
