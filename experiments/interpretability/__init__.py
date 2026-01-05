"""Interpretability module for GNN models.

Provides:
- AttentionExtractor: Extract attention weights from GAT models
- Attention visualization utilities
- SHAP analysis for feature importance
"""

from experiments.interpretability.attention_viz import (
    AttentionExtractor,
    aggregate_attention_per_node,
    aggregate_attention_per_edge,
    create_attention_plot_data,
    plot_attention_on_pitch,
    compare_attention_patterns,
)

__all__ = [
    'AttentionExtractor',
    'aggregate_attention_per_node',
    'aggregate_attention_per_edge',
    'create_attention_plot_data',
    'plot_attention_on_pitch',
    'compare_attention_patterns',
]
