"""SHAP and Integrated Gradients analysis for GNN interpretability.

Provides feature importance analysis for understanding which input
features contribute most to corner kick outcome predictions.

Methods:
- Integrated Gradients: Attribution for node features
- SHAP: KernelSHAP for embedding-level importance
"""

from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# Feature names for corner kick node features
FEATURE_NAMES = [
    'x_position',
    'y_position',
    'team',
    'dist_goal',
    'dist_ball',
]


def get_feature_names() -> List[str]:
    """Get human-readable feature names.

    Returns:
        List of feature names
    """
    return FEATURE_NAMES.copy()


class IntegratedGradientsExplainer:
    """Integrated Gradients for node feature attribution.

    Computes attribution scores showing how much each input feature
    contributes to the model's prediction.

    Args:
        model: GNN model to explain
        baseline: Baseline input (default: zeros)
        n_steps: Number of interpolation steps
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.baseline = baseline
        self.n_steps = n_steps
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

    def explain(
        self,
        batch: Batch,
        target_class: int = 1,
    ) -> torch.Tensor:
        """Compute Integrated Gradients attribution.

        Args:
            batch: PyTorch Geometric batch containing graph(s)
            target_class: Target class for attribution (0 or 1)

        Returns:
            Attribution tensor [num_nodes, num_features]
        """
        batch = batch.to(self.device)
        x = batch.x.clone().requires_grad_(True)

        # Baseline (zeros by default)
        if self.baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = self.baseline.to(self.device)

        # Compute integrated gradients
        attributions = torch.zeros_like(x)

        for step in range(self.n_steps):
            # Interpolate between baseline and input
            alpha = step / self.n_steps
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.detach().requires_grad_(True)
            interpolated.retain_grad()

            # Forward pass
            self.model.eval()
            output = self.model(interpolated, batch.edge_index, batch.batch)

            # Backward pass
            output.sum().backward()

            # Accumulate gradients
            if interpolated.grad is not None:
                attributions += interpolated.grad.detach()

            self.model.zero_grad()

        # Scale by (input - baseline)
        attributions = attributions * (x - baseline) / self.n_steps

        return attributions.detach().cpu()


def aggregate_feature_importance(
    attributions: Union[torch.Tensor, List[torch.Tensor]],
    aggregation: str = 'mean_abs',
) -> torch.Tensor:
    """Aggregate attributions to feature-level importance.

    Args:
        attributions: Single attribution tensor or list of tensors
        aggregation: Aggregation method ('mean_abs', 'mean', 'sum')

    Returns:
        Feature importance tensor [num_features]
    """
    if isinstance(attributions, list):
        # Stack all attributions
        all_attr = torch.cat(attributions, dim=0)
    else:
        all_attr = attributions

    if aggregation == 'mean_abs':
        importance = all_attr.abs().mean(dim=0)
    elif aggregation == 'mean':
        importance = all_attr.mean(dim=0)
    elif aggregation == 'sum':
        importance = all_attr.sum(dim=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return importance


class SHAPExplainer:
    """SHAP explainer for GNN models.

    Uses gradient-based approximation for efficiency.

    Args:
        model: GNN model to explain
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

    def explain(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        """Compute SHAP-style attributions using gradients.

        Args:
            batch: PyTorch Geometric batch

        Returns:
            Attribution tensor [num_nodes, num_features]
        """
        batch = batch.to(self.device)
        x = batch.x.clone().requires_grad_(True)

        self.model.eval()
        output = self.model(x, batch.edge_index, batch.batch)
        output.sum().backward()

        attributions = x.grad.detach() if x.grad is not None else torch.zeros_like(x)

        return attributions.cpu()


def explain_embedding_classifier(
    classifier: nn.Module,
    embeddings: torch.Tensor,
    background_size: int = 100,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Explain an embedding classifier using gradient-based SHAP approximation.

    Args:
        classifier: MLP classifier on embeddings
        embeddings: Input embeddings [n_samples, embedding_dim]
        background_size: Number of background samples
        device: Device for computation

    Returns:
        SHAP values [n_samples, embedding_dim]
    """
    device = device or torch.device('cpu')
    classifier = classifier.to(device)
    classifier.eval()

    embeddings = embeddings.to(device)
    n_samples = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]

    shap_values = np.zeros((n_samples, embedding_dim))

    # Use gradient-based approximation
    for i in range(n_samples):
        x = embeddings[i:i+1].clone().requires_grad_(True)
        output = classifier(x)
        output.backward()

        if x.grad is not None:
            shap_values[i] = (x.grad * x).detach().cpu().numpy()

    return shap_values


def plot_feature_importance(
    importance: torch.Tensor,
    feature_names: List[str],
    ax: Optional[plt.Axes] = None,
    title: str = 'Feature Importance',
) -> plt.Figure:
    """Plot feature importance as a bar chart.

    Args:
        importance: Feature importance scores [num_features]
        feature_names: Names for each feature
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    importance = importance.numpy() if isinstance(importance, torch.Tensor) else importance

    # Sort by importance
    indices = np.argsort(importance)[::-1]
    sorted_importance = importance[indices]
    sorted_names = [feature_names[i] for i in indices]

    # Plot
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(importance)))
    ax.barh(range(len(importance)), sorted_importance, color=colors[::-1])
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title)

    return fig


def analyze_feature_importance(
    model: nn.Module,
    graphs: List[Data],
    num_samples: int = 50,
    n_steps: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run complete feature importance analysis.

    Args:
        model: GNN model to analyze
        graphs: List of graph Data objects
        num_samples: Number of samples to analyze
        n_steps: Steps for integrated gradients
        device: Device for computation

    Returns:
        Dictionary with feature_importance and feature_names
    """
    device = device or torch.device('cpu')
    explainer = IntegratedGradientsExplainer(model, n_steps=n_steps, device=device)

    # Sample graphs
    sample_indices = np.random.choice(
        len(graphs), min(num_samples, len(graphs)), replace=False
    )

    all_attributions = []
    for idx in sample_indices:
        graph = graphs[idx]
        batch = Batch.from_data_list([graph])
        attribution = explainer.explain(batch)
        all_attributions.append(attribution)

    # Aggregate
    feature_importance = aggregate_feature_importance(all_attributions)

    return {
        'feature_importance': feature_importance,
        'feature_names': get_feature_names(),
        'attributions': all_attributions,
    }
