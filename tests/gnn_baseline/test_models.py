"""Tests for GNN models.

TDD: Write tests first, then implement to make them pass.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch


def create_sample_graph(num_nodes=10, num_features=5, num_edge_features=2, label=1):
    """Create a sample graph for testing."""
    x = torch.randn(num_nodes, num_features)
    # Create random edges (k-NN style, ~3 edges per node)
    sources = torch.arange(num_nodes).repeat_interleave(3)
    targets = torch.randint(0, num_nodes, (num_nodes * 3,))
    # Remove self-loops
    mask = sources != targets
    edge_index = torch.stack([sources[mask], targets[mask]])
    edge_attr = torch.randn(edge_index.shape[1], num_edge_features)
    y = torch.tensor([label], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def create_sample_batch(batch_size=4):
    """Create a batch of sample graphs."""
    graphs = [create_sample_graph(num_nodes=10 + i, label=i % 2) for i in range(batch_size)]
    return Batch.from_data_list(graphs)


class TestGATModel:
    """Test Graph Attention Network model."""

    def test_gat_instantiation(self):
        """GAT model should instantiate with default parameters."""
        from experiments.gnn_baseline.models import GATModel

        model = GATModel(
            in_channels=5,
            hidden_channels=64,
            num_layers=2,
            heads=4,
            dropout=0.1,
        )

        assert model is not None

    def test_gat_forward_single_graph(self):
        """GAT should process a single graph and return prediction."""
        from experiments.gnn_baseline.models import GATModel

        model = GATModel(in_channels=5, hidden_channels=64, num_layers=2)
        graph = create_sample_graph()

        # Need to batch even a single graph for pooling
        batch = Batch.from_data_list([graph])
        output = model(batch.x, batch.edge_index, batch.batch)

        assert output.shape == (1, 1)  # [batch_size, 1]

    def test_gat_forward_batch(self):
        """GAT should process a batch of graphs."""
        from experiments.gnn_baseline.models import GATModel

        model = GATModel(in_channels=5, hidden_channels=64, num_layers=2)
        batch = create_sample_batch(batch_size=4)

        output = model(batch.x, batch.edge_index, batch.batch)

        assert output.shape == (4, 1)  # [batch_size, 1]

    def test_gat_output_range(self):
        """GAT output should be in [0, 1] (sigmoid activated)."""
        from experiments.gnn_baseline.models import GATModel

        model = GATModel(in_channels=5, hidden_channels=64, num_layers=2)
        batch = create_sample_batch(batch_size=8)

        output = model(batch.x, batch.edge_index, batch.batch)

        assert (output >= 0).all()
        assert (output <= 1).all()


class TestGraphSAGEModel:
    """Test GraphSAGE model."""

    def test_graphsage_instantiation(self):
        """GraphSAGE model should instantiate with default parameters."""
        from experiments.gnn_baseline.models import GraphSAGEModel

        model = GraphSAGEModel(
            in_channels=5,
            hidden_channels=64,
            num_layers=2,
            dropout=0.1,
        )

        assert model is not None

    def test_graphsage_forward_batch(self):
        """GraphSAGE should process a batch of graphs."""
        from experiments.gnn_baseline.models import GraphSAGEModel

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        batch = create_sample_batch(batch_size=4)

        output = model(batch.x, batch.edge_index, batch.batch)

        assert output.shape == (4, 1)

    def test_graphsage_output_range(self):
        """GraphSAGE output should be in [0, 1]."""
        from experiments.gnn_baseline.models import GraphSAGEModel

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        batch = create_sample_batch(batch_size=8)

        output = model(batch.x, batch.edge_index, batch.batch)

        assert (output >= 0).all()
        assert (output <= 1).all()


class TestMPNNModel:
    """Test Message Passing Neural Network model."""

    def test_mpnn_instantiation(self):
        """MPNN model should instantiate with default parameters."""
        from experiments.gnn_baseline.models import MPNNModel

        model = MPNNModel(
            in_channels=5,
            edge_channels=2,
            hidden_channels=64,
            num_layers=2,
            dropout=0.1,
        )

        assert model is not None

    def test_mpnn_forward_batch(self):
        """MPNN should process a batch of graphs."""
        from experiments.gnn_baseline.models import MPNNModel

        model = MPNNModel(
            in_channels=5,
            edge_channels=2,
            hidden_channels=64,
            num_layers=2,
        )
        batch = create_sample_batch(batch_size=4)

        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        assert output.shape == (4, 1)

    def test_mpnn_output_range(self):
        """MPNN output should be in [0, 1]."""
        from experiments.gnn_baseline.models import MPNNModel

        model = MPNNModel(
            in_channels=5,
            edge_channels=2,
            hidden_channels=64,
            num_layers=2,
        )
        batch = create_sample_batch(batch_size=8)

        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        assert (output >= 0).all()
        assert (output <= 1).all()


class TestModelFactory:
    """Test model factory for creating models by name."""

    def test_create_gat(self):
        """Factory should create GAT model."""
        from experiments.gnn_baseline.models import create_model

        model = create_model('gat', in_channels=5)

        from experiments.gnn_baseline.models import GATModel
        assert isinstance(model, GATModel)

    def test_create_graphsage(self):
        """Factory should create GraphSAGE model."""
        from experiments.gnn_baseline.models import create_model

        model = create_model('graphsage', in_channels=5)

        from experiments.gnn_baseline.models import GraphSAGEModel
        assert isinstance(model, GraphSAGEModel)

    def test_create_mpnn(self):
        """Factory should create MPNN model."""
        from experiments.gnn_baseline.models import create_model

        model = create_model('mpnn', in_channels=5, edge_channels=2)

        from experiments.gnn_baseline.models import MPNNModel
        assert isinstance(model, MPNNModel)

    def test_invalid_model_name(self):
        """Factory should raise error for invalid model name."""
        from experiments.gnn_baseline.models import create_model

        with pytest.raises(ValueError):
            create_model('invalid_model', in_channels=5)


class TestGradientFlow:
    """Test that gradients flow correctly through models."""

    def test_gat_gradients(self):
        """GAT should allow gradient backpropagation."""
        from experiments.gnn_baseline.models import GATModel

        model = GATModel(in_channels=5, hidden_channels=64, num_layers=2)
        batch = create_sample_batch(batch_size=4)

        output = model(batch.x, batch.edge_index, batch.batch)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_graphsage_gradients(self):
        """GraphSAGE should allow gradient backpropagation."""
        from experiments.gnn_baseline.models import GraphSAGEModel

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        batch = create_sample_batch(batch_size=4)

        output = model(batch.x, batch.edge_index, batch.batch)
        loss = output.sum()
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None

    def test_mpnn_gradients(self):
        """MPNN should allow gradient backpropagation."""
        from experiments.gnn_baseline.models import MPNNModel

        model = MPNNModel(in_channels=5, edge_channels=2, hidden_channels=64, num_layers=2)
        batch = create_sample_batch(batch_size=4)

        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = output.sum()
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
