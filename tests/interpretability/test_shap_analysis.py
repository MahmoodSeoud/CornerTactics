"""Tests for SHAP/Integrated Gradients Analysis.

TDD: Write tests first, then implement to make them pass.

Feature importance analysis for GNN models using:
- Integrated Gradients for node feature attribution
- SHAP for embedding-level feature importance
"""

import pytest
import torch
import numpy as np


class TestIntegratedGradients:
    """Test Integrated Gradients for node feature attribution."""

    def test_instantiation(self):
        """IntegratedGradientsExplainer should instantiate."""
        from experiments.interpretability.shap_analysis import IntegratedGradientsExplainer
        from experiments.gnn_baseline.models import GraphSAGEModel

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        explainer = IntegratedGradientsExplainer(model)

        assert explainer is not None

    def test_attribution_shape(self):
        """Attribution should have same shape as input features."""
        from experiments.interpretability.shap_analysis import IntegratedGradientsExplainer
        from experiments.gnn_baseline.models import GraphSAGEModel
        from torch_geometric.data import Data, Batch

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        explainer = IntegratedGradientsExplainer(model)

        x = torch.randn(10, 5)
        edge_index = torch.randint(0, 10, (2, 30))
        graph = Data(x=x, edge_index=edge_index, y=torch.tensor([1.0]))
        batch = Batch.from_data_list([graph])

        attribution = explainer.explain(batch)

        assert attribution.shape == x.shape  # [num_nodes, num_features]

    def test_attribution_values(self):
        """Attributions should be finite and real."""
        from experiments.interpretability.shap_analysis import IntegratedGradientsExplainer
        from experiments.gnn_baseline.models import GraphSAGEModel
        from torch_geometric.data import Data, Batch

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        explainer = IntegratedGradientsExplainer(model)

        x = torch.randn(10, 5)
        edge_index = torch.randint(0, 10, (2, 30))
        graph = Data(x=x, edge_index=edge_index, y=torch.tensor([1.0]))
        batch = Batch.from_data_list([graph])

        attribution = explainer.explain(batch)

        assert torch.isfinite(attribution).all()


class TestFeatureImportance:
    """Test feature importance aggregation."""

    def test_aggregate_by_feature(self):
        """Should aggregate attributions by feature dimension."""
        from experiments.interpretability.shap_analysis import aggregate_feature_importance

        # Attribution for 10 nodes, 5 features
        attribution = torch.randn(10, 5)

        importance = aggregate_feature_importance(attribution)

        assert importance.shape == (5,)  # One score per feature

    def test_aggregate_multiple_graphs(self):
        """Should aggregate across multiple graphs."""
        from experiments.interpretability.shap_analysis import aggregate_feature_importance

        # List of attributions from different graphs
        attributions = [torch.randn(10, 5), torch.randn(12, 5), torch.randn(8, 5)]

        importance = aggregate_feature_importance(attributions)

        assert importance.shape == (5,)


class TestSHAPAnalysis:
    """Test SHAP-based explanation."""

    def test_shap_explainer_instantiation(self):
        """SHAPExplainer should instantiate."""
        from experiments.interpretability.shap_analysis import SHAPExplainer
        from experiments.gnn_baseline.models import GraphSAGEModel

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        explainer = SHAPExplainer(model)

        assert explainer is not None

    def test_shap_on_embeddings(self):
        """SHAP should explain embedding classifier."""
        from experiments.interpretability.shap_analysis import explain_embedding_classifier

        # Sample embeddings and predictions
        embeddings = torch.randn(50, 64)
        labels = torch.randint(0, 2, (50,))

        # Simple classifier
        classifier = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

        shap_values = explain_embedding_classifier(
            classifier=classifier,
            embeddings=embeddings,
            background_size=20,
        )

        # SHAP values should have shape [n_samples, embedding_dim]
        assert shap_values.shape == (50, 64)


class TestFeatureNameMapping:
    """Test mapping feature indices to interpretable names."""

    def test_feature_names(self):
        """Should provide meaningful feature names."""
        from experiments.interpretability.shap_analysis import get_feature_names

        names = get_feature_names()

        assert len(names) == 5  # 5 node features
        assert 'x_position' in names
        assert 'y_position' in names
        assert 'team' in names


class TestImportancePlotting:
    """Test importance visualization."""

    def test_plot_feature_importance(self):
        """Should create matplotlib figure of feature importance."""
        from experiments.interpretability.shap_analysis import plot_feature_importance

        importance = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1])
        feature_names = ['x_pos', 'y_pos', 'team', 'dist_goal', 'dist_ball']

        fig = plot_feature_importance(importance, feature_names)

        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestComprehensiveAnalysis:
    """Test full analysis pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample graphs for analysis."""
        from torch_geometric.data import Data, Batch

        graphs = []
        for i in range(20):
            x = torch.randn(10, 5)
            edge_index = torch.randint(0, 10, (2, 30))
            y = torch.tensor([i % 2], dtype=torch.float32)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))

        return graphs

    def test_analyze_feature_importance(self, sample_data):
        """Should run complete feature importance analysis."""
        from experiments.interpretability.shap_analysis import analyze_feature_importance
        from experiments.gnn_baseline.models import GraphSAGEModel
        from torch_geometric.loader import DataLoader

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)

        results = analyze_feature_importance(
            model=model,
            graphs=sample_data,
            num_samples=10,
        )

        assert 'feature_importance' in results
        assert 'feature_names' in results
        assert len(results['feature_importance']) == 5
