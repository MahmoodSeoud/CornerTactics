"""Tests for oversampling on graph embeddings.

TDD: Write tests first, then implement to make them pass.

SMOTE/ADASYN: Oversampling minority classes in embedding space.
"""

import pytest
import torch
import numpy as np


class TestEmbeddingExtractor:
    """Test extraction of graph embeddings from GNN."""

    def test_instantiation(self):
        """EmbeddingExtractor should instantiate with a model."""
        from experiments.class_imbalance.oversampling import EmbeddingExtractor
        from experiments.gnn_baseline.models import GraphSAGEModel

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        extractor = EmbeddingExtractor(model)

        assert extractor is not None

    def test_extract_single_graph(self):
        """Should extract embedding from a single graph."""
        from experiments.class_imbalance.oversampling import EmbeddingExtractor
        from experiments.gnn_baseline.models import GraphSAGEModel
        from torch_geometric.data import Data, Batch

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        extractor = EmbeddingExtractor(model)

        # Create sample graph
        x = torch.randn(10, 5)
        edge_index = torch.randint(0, 10, (2, 30))
        graph = Data(x=x, edge_index=edge_index)
        batch = Batch.from_data_list([graph])

        embedding = extractor.extract(batch)

        # Should return embedding before final classifier
        assert embedding.shape[0] == 1  # batch size
        assert embedding.shape[1] > 0  # embedding dimension

    def test_extract_batch(self):
        """Should extract embeddings from a batch of graphs."""
        from experiments.class_imbalance.oversampling import EmbeddingExtractor
        from experiments.gnn_baseline.models import GraphSAGEModel
        from torch_geometric.data import Data, Batch

        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        extractor = EmbeddingExtractor(model)

        # Create batch of graphs
        graphs = []
        for i in range(4):
            x = torch.randn(10, 5)
            edge_index = torch.randint(0, 10, (2, 30))
            graphs.append(Data(x=x, edge_index=edge_index))
        batch = Batch.from_data_list(graphs)

        embeddings = extractor.extract(batch)

        assert embeddings.shape[0] == 4  # batch size


class TestSMOTEOversampling:
    """Test SMOTE oversampling on embeddings."""

    def test_smote_instantiation(self):
        """SMOTEOversampler should instantiate with default parameters."""
        from experiments.class_imbalance.oversampling import SMOTEOversampler

        oversampler = SMOTEOversampler()
        assert oversampler is not None

    def test_smote_custom_k_neighbors(self):
        """SMOTEOversampler should accept custom k_neighbors."""
        from experiments.class_imbalance.oversampling import SMOTEOversampler

        oversampler = SMOTEOversampler(k_neighbors=3)
        assert oversampler.k_neighbors == 3

    def test_smote_fit_resample(self):
        """SMOTE should balance class distribution."""
        from experiments.class_imbalance.oversampling import SMOTEOversampler

        oversampler = SMOTEOversampler(k_neighbors=3, random_state=42)

        # Imbalanced data: 80 class 0, 20 class 1
        embeddings = np.random.randn(100, 64)
        labels = np.array([0] * 80 + [1] * 20)

        X_resampled, y_resampled = oversampler.fit_resample(embeddings, labels)

        # Should have more samples after resampling
        assert len(X_resampled) > len(embeddings)
        # Classes should be balanced
        assert np.sum(y_resampled == 0) == np.sum(y_resampled == 1)

    def test_smote_with_tensor_input(self):
        """SMOTE should work with PyTorch tensor input."""
        from experiments.class_imbalance.oversampling import SMOTEOversampler

        oversampler = SMOTEOversampler(k_neighbors=3, random_state=42)

        embeddings = torch.randn(100, 64)
        labels = torch.tensor([0] * 80 + [1] * 20)

        X_resampled, y_resampled = oversampler.fit_resample(embeddings, labels)

        assert len(X_resampled) > 100


class TestADASYNOversampling:
    """Test ADASYN oversampling on embeddings."""

    def test_adasyn_instantiation(self):
        """ADASYNOversampler should instantiate with default parameters."""
        from experiments.class_imbalance.oversampling import ADASYNOversampler

        oversampler = ADASYNOversampler()
        assert oversampler is not None

    def test_adasyn_fit_resample(self):
        """ADASYN should oversample minority class adaptively."""
        from experiments.class_imbalance.oversampling import ADASYNOversampler

        oversampler = ADASYNOversampler(n_neighbors=3, random_state=42)

        # Imbalanced data
        embeddings = np.random.randn(100, 64)
        labels = np.array([0] * 80 + [1] * 20)

        X_resampled, y_resampled = oversampler.fit_resample(embeddings, labels)

        # Should have more minority samples
        assert np.sum(y_resampled == 1) > 20


class TestOversamplingPipeline:
    """Test full oversampling pipeline with GNN embeddings."""

    @pytest.fixture
    def sample_corners(self):
        """Create sample corner data."""
        corners = []
        for i in range(50):
            corner = {
                "match_id": str(i // 5),
                "event": {"id": f"corner-{i}", "location": [120.0, 40.0]},
                "freeze_frame": [
                    {"location": [100.0 + j, 30.0 + j * 2], "teammate": j < 5,
                     "keeper": j == 9, "actor": j == 0}
                    for j in range(10)
                ],
                "shot_outcome": 0 if i < 40 else 1,  # Imbalanced: 80% class 0
            }
            corners.append(corner)
        return corners

    def test_pipeline_extract_and_oversample(self, sample_corners):
        """Full pipeline should extract embeddings and oversample."""
        from experiments.class_imbalance.oversampling import (
            EmbeddingExtractor, SMOTEOversampler, create_oversampled_classifier_data
        )
        from experiments.gnn_baseline.models import GraphSAGEModel
        from experiments.gnn_baseline.graph_construction import build_graph_dataset
        from torch_geometric.loader import DataLoader

        # Build graphs
        graphs = build_graph_dataset(sample_corners)
        labels = [c['shot_outcome'] for c in sample_corners]

        # Extract embeddings
        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)
        extractor = EmbeddingExtractor(model)
        loader = DataLoader(graphs, batch_size=10, shuffle=False)

        embeddings, ys = [], []
        for batch in loader:
            emb = extractor.extract(batch)
            embeddings.append(emb)
            ys.append(batch.y)

        embeddings = torch.cat(embeddings, dim=0)
        ys = torch.cat(ys, dim=0)

        # Oversample
        oversampler = SMOTEOversampler(k_neighbors=3, random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(
            embeddings.numpy(), ys.numpy().astype(int)
        )

        # Should be balanced
        assert X_resampled.shape[0] > embeddings.shape[0]

    def test_create_oversampled_classifier_data(self, sample_corners):
        """Helper function should create balanced classifier dataset."""
        from experiments.class_imbalance.oversampling import create_oversampled_classifier_data
        from experiments.gnn_baseline.models import GraphSAGEModel
        from experiments.gnn_baseline.graph_construction import build_graph_dataset
        from torch_geometric.loader import DataLoader

        graphs = build_graph_dataset(sample_corners)
        model = GraphSAGEModel(in_channels=5, hidden_channels=64, num_layers=2)

        X_train, y_train = create_oversampled_classifier_data(
            model=model,
            graphs=graphs,
            method='smote',
            k_neighbors=3,
            random_state=42,
        )

        # Output should be numpy arrays
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        # Should be balanced
        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) == 2:
            assert counts[0] == counts[1]
