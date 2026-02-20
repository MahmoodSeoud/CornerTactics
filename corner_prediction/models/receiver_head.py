"""Stage 1: Receiver prediction head.

Predicts which attacking outfield player receives the corner kick delivery.
Uses per-node classification with masked softmax over valid candidates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class ReceiverHead(nn.Module):
    """Per-node classification head for receiver prediction.

    Takes per-node embeddings from the backbone and produces a logit per node.
    Use :func:`masked_softmax` to convert logits to probabilities over
    valid receiver candidates.

    Args:
        input_dim: Backbone output dimension (e.g. 128 for pretrained).
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.3,
                 linear_only: bool = False):
        super().__init__()
        if linear_only:
            self.mlp = nn.Linear(input_dim, 1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute per-node logits.

        Args:
            node_embeddings: [N, input_dim] from backbone.

        Returns:
            Logits [N] (one per node).
        """
        return self.mlp(node_embeddings).squeeze(-1)


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Compute softmax over masked nodes, independently per graph.

    Uses only differentiable operations (no in-place assignment) so
    gradients flow through properly.

    Args:
        logits: Per-node logits [N].
        mask: Boolean mask [N] — True for valid receiver candidates.
        batch: Graph membership [N] — maps each node to its graph index.

    Returns:
        Probabilities [N]. Masked-out nodes get 0.0.
        Within each graph, valid nodes sum to 1.0.
    """
    # Use torch.where for differentiable masking (no in-place ops)
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    masked_logits = torch.where(mask, logits, neg_inf)

    # Per-graph max for numerical stability
    max_per_graph = scatter(masked_logits, batch, dim=0, reduce="max")
    shifted = masked_logits - max_per_graph[batch]
    # Re-mask after shift (inf - inf = nan)
    shifted = torch.where(mask, shifted, neg_inf)

    exp_vals = torch.exp(shifted)
    # Zero out masked positions (exp(-inf) may give tiny values)
    exp_vals = exp_vals * mask.float()

    # Normalize per graph
    sum_per_graph = scatter(exp_vals, batch, dim=0, reduce="sum")
    probs = exp_vals / sum_per_graph[batch].clamp(min=1e-12)
    probs = probs * mask.float()

    return probs


def receiver_loss(
    logits: torch.Tensor,
    receiver_label: torch.Tensor,
    receiver_mask: torch.Tensor,
    batch: torch.Tensor,
    has_receiver_label: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for receiver prediction over masked candidates.

    Only graphs with has_receiver_label=True contribute to the loss.

    Args:
        logits: Per-node logits [N].
        receiver_label: Per-node binary labels [N] (1.0 at true receiver).
        receiver_mask: Boolean mask [N] for valid candidates.
        batch: Graph membership [N].
        has_receiver_label: Per-graph boolean [B] — True if graph has a label.

    Returns:
        Scalar loss (mean over labeled graphs). Returns 0 if no labeled graphs.
    """
    if not has_receiver_label.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Compute log-softmax per graph over masked candidates
    probs = masked_softmax(logits, receiver_mask, batch)
    log_probs = torch.log(probs.clamp(min=1e-12))

    # For each graph with a label, pick the log-prob at the receiver node
    # receiver_label is 1.0 at the true receiver, 0.0 elsewhere
    per_node_loss = -log_probs * receiver_label

    # Sum per graph (each graph has exactly one receiver with label=1.0)
    per_graph_loss = scatter(per_node_loss, batch, dim=0, reduce="sum")

    # Only average over graphs that have labels
    labeled_loss = per_graph_loss[has_receiver_label]
    return labeled_loss.mean()
