"""Oversampling on graph embeddings for class imbalance.

Provides SMOTE and ADASYN oversampling on GNN embeddings to handle
severe class imbalance in corner kick outcome prediction.

Usage:
    1. Train a GNN model or use pretrained
    2. Extract graph-level embeddings using EmbeddingExtractor
    3. Apply SMOTE/ADASYN to oversample minority class embeddings
    4. Train a classifier on the augmented embedding space
"""

from typing import Literal, Tuple, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool

from imblearn.over_sampling import SMOTE, ADASYN


class EmbeddingExtractor:
    """Extract graph-level embeddings from a trained GNN.

    Hooks into the model before the final classification layer to
    extract the graph-level representation.

    Args:
        model: A trained GNN model (GAT, GraphSAGE, or MPNN)
        device: Device to use for extraction
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

        # Detect model type by class name
        self._is_mpnn = 'MPNN' in model.__class__.__name__

        # Store embeddings via hook
        self._embedding = None
        self._hook_handle = None
        self._setup_hook()

    def _setup_hook(self):
        """Register forward hook on the layer before classifier."""
        # The classifier is typically the last module
        # We want to capture input to the classifier
        classifier = self.model.classifier

        def hook(module, input, output):
            self._embedding = input[0].detach()

        self._hook_handle = classifier.register_forward_hook(hook)

    def extract(self, batch: Batch) -> torch.Tensor:
        """Extract embeddings from a batch of graphs.

        Args:
            batch: PyTorch Geometric batch of graphs

        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        batch = batch.to(self.device)

        with torch.no_grad():
            # Forward pass to trigger hook
            if self._is_mpnn:
                # MPNN model requires edge_attr
                _ = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                # GAT or GraphSAGE
                _ = self.model(batch.x, batch.edge_index, batch.batch)

        return self._embedding.cpu()

    def __del__(self):
        """Clean up hook on deletion."""
        if self._hook_handle is not None:
            self._hook_handle.remove()


class SMOTEOversampler:
    """SMOTE oversampling for graph embeddings.

    Synthetic Minority Over-sampling Technique creates synthetic samples
    by interpolating between minority class samples and their k-nearest
    neighbors.

    Args:
        k_neighbors: Number of nearest neighbors for SMOTE
        sampling_strategy: Target ratio of minority to majority class
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: Union[str, float] = 'auto',
        random_state: Optional[int] = None,
    ):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        self._smote = SMOTE(
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

    def fit_resample(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Oversample minority class using SMOTE.

        Args:
            embeddings: Shape [n_samples, embedding_dim]
            labels: Shape [n_samples]

        Returns:
            Tuple of (resampled_embeddings, resampled_labels)
        """
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        # Flatten labels if needed
        labels = labels.flatten().astype(int)

        X_resampled, y_resampled = self._smote.fit_resample(embeddings, labels)

        return X_resampled, y_resampled


class ADASYNOversampler:
    """ADASYN oversampling for graph embeddings.

    Adaptive Synthetic Sampling focuses on generating synthetic samples
    for minority class samples that are harder to learn.

    Args:
        n_neighbors: Number of nearest neighbors
        sampling_strategy: Target ratio of minority to majority class
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        sampling_strategy: Union[str, float] = 'auto',
        random_state: Optional[int] = None,
    ):
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        self._adasyn = ADASYN(
            n_neighbors=n_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

    def fit_resample(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Oversample minority class using ADASYN.

        Args:
            embeddings: Shape [n_samples, embedding_dim]
            labels: Shape [n_samples]

        Returns:
            Tuple of (resampled_embeddings, resampled_labels)
        """
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        # Flatten labels if needed
        labels = labels.flatten().astype(int)

        X_resampled, y_resampled = self._adasyn.fit_resample(embeddings, labels)

        return X_resampled, y_resampled


def create_oversampled_classifier_data(
    model: nn.Module,
    graphs: List[Data],
    method: Literal['smote', 'adasyn'] = 'smote',
    k_neighbors: int = 5,
    batch_size: int = 32,
    random_state: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create oversampled training data from graph embeddings.

    Complete pipeline for:
    1. Extracting graph-level embeddings from a GNN
    2. Oversampling minority class using SMOTE or ADASYN

    Args:
        model: Trained GNN model
        graphs: List of PyTorch Geometric Data objects
        method: Oversampling method ('smote' or 'adasyn')
        k_neighbors: Number of neighbors for oversampling
        batch_size: Batch size for embedding extraction
        random_state: Random seed for reproducibility
        device: Device for embedding extraction

    Returns:
        Tuple of (oversampled_embeddings, oversampled_labels)
    """
    # Extract embeddings
    extractor = EmbeddingExtractor(model, device=device)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    embeddings_list = []
    labels_list = []

    for batch in loader:
        emb = extractor.extract(batch)
        embeddings_list.append(emb)
        labels_list.append(batch.y)

    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy().flatten().astype(int)

    # Apply oversampling
    if method == 'smote':
        oversampler = SMOTEOversampler(
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
    elif method == 'adasyn':
        oversampler = ADASYNOversampler(
            n_neighbors=k_neighbors,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'smote' or 'adasyn'")

    X_resampled, y_resampled = oversampler.fit_resample(embeddings, labels)

    return X_resampled, y_resampled
