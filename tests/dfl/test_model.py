"""Tests for Phase 3: Model Architecture & Training.

Following TDD, these tests are written first and should fail until
the implementation is complete.
"""

import pytest
import torch
import numpy as np


class TestSpatialGNN:
    """Tests for the SpatialGNN module that processes single frame graphs."""

    def test_spatial_gnn_exists(self):
        """SpatialGNN class should exist in model module."""
        from src.dfl.model import SpatialGNN

        assert SpatialGNN is not None

    def test_spatial_gnn_is_nn_module(self):
        """SpatialGNN should be a torch.nn.Module."""
        from src.dfl.model import SpatialGNN

        model = SpatialGNN()
        assert isinstance(model, torch.nn.Module)

    def test_spatial_gnn_default_dimensions(self):
        """SpatialGNN should have correct default dimensions."""
        from src.dfl.model import SpatialGNN

        model = SpatialGNN()

        # Default: in_channels=8, hidden_channels=64, out_channels=32
        assert model.in_channels == 8
        assert model.hidden_channels == 64
        assert model.out_channels == 32

    def test_spatial_gnn_custom_dimensions(self):
        """SpatialGNN should accept custom dimensions."""
        from src.dfl.model import SpatialGNN

        model = SpatialGNN(in_channels=6, hidden_channels=32, out_channels=16)

        assert model.in_channels == 6
        assert model.hidden_channels == 32
        assert model.out_channels == 16

    def test_spatial_gnn_forward_returns_tensor(self):
        """SpatialGNN forward should return a tensor."""
        from src.dfl.model import SpatialGNN

        model = SpatialGNN(in_channels=8)

        # Create dummy graph: 23 nodes (22 players + 1 ball), 8 features
        x = torch.randn(23, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

        output = model(x, edge_index)

        assert isinstance(output, torch.Tensor)

    def test_spatial_gnn_forward_output_shape(self):
        """SpatialGNN forward should output (1, out_channels) for single graph."""
        from src.dfl.model import SpatialGNN

        model = SpatialGNN(in_channels=8, out_channels=32)

        x = torch.randn(23, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

        output = model(x, edge_index)

        # Should pool to graph-level: (1, 32)
        assert output.shape == (1, 32)

    def test_spatial_gnn_forward_with_batch(self):
        """SpatialGNN forward should handle batched graphs."""
        from src.dfl.model import SpatialGNN

        model = SpatialGNN(in_channels=8, out_channels=32)

        # Two graphs in a batch (each with 10 nodes)
        x = torch.randn(20, 8)  # 20 nodes total
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 10, 11, 12, 13], [1, 0, 3, 2, 11, 10, 13, 12]], dtype=torch.long
        )
        batch = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)

        output = model(x, edge_index, batch=batch)

        # Should output (2, 32) for 2 graphs
        assert output.shape == (2, 32)

    def test_spatial_gnn_uses_gatv2conv(self):
        """SpatialGNN should use GATv2Conv layers."""
        from src.dfl.model import SpatialGNN
        from torch_geometric.nn import GATv2Conv

        model = SpatialGNN()

        # Check that conv layers are GATv2Conv
        assert isinstance(model.conv1, GATv2Conv)
        assert isinstance(model.conv2, GATv2Conv)


class TestTemporalAggregator:
    """Tests for the TemporalAggregator module that processes sequences."""

    def test_temporal_aggregator_exists(self):
        """TemporalAggregator class should exist in model module."""
        from src.dfl.model import TemporalAggregator

        assert TemporalAggregator is not None

    def test_temporal_aggregator_is_nn_module(self):
        """TemporalAggregator should be a torch.nn.Module."""
        from src.dfl.model import TemporalAggregator

        model = TemporalAggregator()
        assert isinstance(model, torch.nn.Module)

    def test_temporal_aggregator_default_dimensions(self):
        """TemporalAggregator should have correct default dimensions."""
        from src.dfl.model import TemporalAggregator

        model = TemporalAggregator()

        assert model.input_dim == 32
        assert model.hidden_dim == 64

    def test_temporal_aggregator_forward_returns_tensor(self):
        """TemporalAggregator forward should return a tensor."""
        from src.dfl.model import TemporalAggregator

        model = TemporalAggregator(input_dim=32, hidden_dim=64)

        # Batch of 4 sequences, each 200 frames, 32 features
        x = torch.randn(4, 200, 32)

        output = model(x)

        assert isinstance(output, torch.Tensor)

    def test_temporal_aggregator_forward_output_shape(self):
        """TemporalAggregator forward should output (batch, hidden_dim)."""
        from src.dfl.model import TemporalAggregator

        model = TemporalAggregator(input_dim=32, hidden_dim=64)

        x = torch.randn(4, 200, 32)  # 4 sequences
        output = model(x)

        assert output.shape == (4, 64)

    def test_temporal_aggregator_variable_length(self):
        """TemporalAggregator should handle variable length sequences."""
        from src.dfl.model import TemporalAggregator

        model = TemporalAggregator(input_dim=32, hidden_dim=64)

        # Different sequence lengths
        x1 = torch.randn(2, 100, 32)
        x2 = torch.randn(2, 250, 32)

        out1 = model(x1)
        out2 = model(x2)

        assert out1.shape == (2, 64)
        assert out2.shape == (2, 64)


class TestCornerKickPredictor:
    """Tests for the full ST-GNN model with multi-head outputs."""

    def test_corner_kick_predictor_exists(self):
        """CornerKickPredictor class should exist in model module."""
        from src.dfl.model import CornerKickPredictor

        assert CornerKickPredictor is not None

    def test_corner_kick_predictor_is_nn_module(self):
        """CornerKickPredictor should be a torch.nn.Module."""
        from src.dfl.model import CornerKickPredictor

        model = CornerKickPredictor()
        assert isinstance(model, torch.nn.Module)

    def test_corner_kick_predictor_has_spatial_gnn(self):
        """CornerKickPredictor should have a SpatialGNN component."""
        from src.dfl.model import CornerKickPredictor, SpatialGNN

        model = CornerKickPredictor()

        assert hasattr(model, "spatial_gnn")
        assert isinstance(model.spatial_gnn, SpatialGNN)

    def test_corner_kick_predictor_has_temporal_aggregator(self):
        """CornerKickPredictor should have a TemporalAggregator component."""
        from src.dfl.model import CornerKickPredictor, TemporalAggregator

        model = CornerKickPredictor()

        assert hasattr(model, "temporal")
        assert isinstance(model.temporal, TemporalAggregator)

    def test_corner_kick_predictor_has_prediction_heads(self):
        """CornerKickPredictor should have multi-head prediction layers."""
        from src.dfl.model import CornerKickPredictor

        model = CornerKickPredictor()

        assert hasattr(model, "head_shot")
        assert hasattr(model, "head_goal")
        assert hasattr(model, "head_contact")
        assert hasattr(model, "head_outcome")

    def test_corner_kick_predictor_forward_with_graph_list(self):
        """CornerKickPredictor forward should accept list of graph sequences."""
        from src.dfl.model import CornerKickPredictor
        from torch_geometric.data import Data

        model = CornerKickPredictor()

        # Create 2 corner sequences, each with 5 frames
        sequences = []
        for _ in range(2):
            seq = []
            for _ in range(5):
                x = torch.randn(23, 8)
                edge_index = torch.tensor(
                    [[0, 1, 2], [1, 0, 2]], dtype=torch.long
                )
                seq.append(Data(x=x, edge_index=edge_index))
            sequences.append(seq)

        output = model(sequences)

        assert isinstance(output, dict)

    def test_corner_kick_predictor_forward_returns_all_heads(self):
        """CornerKickPredictor forward should return dict with all prediction heads."""
        from src.dfl.model import CornerKickPredictor
        from torch_geometric.data import Data

        model = CornerKickPredictor()

        # Create 2 corner sequences
        sequences = []
        for _ in range(2):
            seq = []
            for _ in range(5):
                x = torch.randn(23, 8)
                edge_index = torch.tensor(
                    [[0, 1, 2], [1, 0, 2]], dtype=torch.long
                )
                seq.append(Data(x=x, edge_index=edge_index))
            sequences.append(seq)

        output = model(sequences)

        assert "shot" in output
        assert "goal" in output
        assert "contact" in output
        assert "outcome" in output

    def test_corner_kick_predictor_shot_head_shape(self):
        """Shot head should output (batch_size, 1) with sigmoid activation."""
        from src.dfl.model import CornerKickPredictor
        from torch_geometric.data import Data

        model = CornerKickPredictor()

        sequences = []
        for _ in range(3):  # 3 samples
            seq = [
                Data(x=torch.randn(23, 8), edge_index=torch.tensor([[0], [1]]))
                for _ in range(5)
            ]
            sequences.append(seq)

        output = model(sequences)

        assert output["shot"].shape == (3, 1)
        # Should be between 0 and 1 (sigmoid)
        assert torch.all(output["shot"] >= 0)
        assert torch.all(output["shot"] <= 1)

    def test_corner_kick_predictor_goal_head_shape(self):
        """Goal head should output (batch_size, 1) with sigmoid activation."""
        from src.dfl.model import CornerKickPredictor
        from torch_geometric.data import Data

        model = CornerKickPredictor()

        sequences = []
        for _ in range(3):
            seq = [
                Data(x=torch.randn(23, 8), edge_index=torch.tensor([[0], [1]]))
                for _ in range(5)
            ]
            sequences.append(seq)

        output = model(sequences)

        assert output["goal"].shape == (3, 1)
        assert torch.all(output["goal"] >= 0)
        assert torch.all(output["goal"] <= 1)

    def test_corner_kick_predictor_contact_head_shape(self):
        """Contact head should output (batch_size, 2) for attacking/defending."""
        from src.dfl.model import CornerKickPredictor
        from torch_geometric.data import Data

        model = CornerKickPredictor()

        sequences = []
        for _ in range(3):
            seq = [
                Data(x=torch.randn(23, 8), edge_index=torch.tensor([[0], [1]]))
                for _ in range(5)
            ]
            sequences.append(seq)

        output = model(sequences)

        assert output["contact"].shape == (3, 2)

    def test_corner_kick_predictor_outcome_head_shape(self):
        """Outcome head should output (batch_size, num_classes)."""
        from src.dfl.model import CornerKickPredictor
        from torch_geometric.data import Data

        model = CornerKickPredictor(num_classes_outcome=6)

        sequences = []
        for _ in range(3):
            seq = [
                Data(x=torch.randn(23, 8), edge_index=torch.tensor([[0], [1]]))
                for _ in range(5)
            ]
            sequences.append(seq)

        output = model(sequences)

        assert output["outcome"].shape == (3, 6)
