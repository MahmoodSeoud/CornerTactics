"""Two-stage corner kick prediction model.

Combines backbone + receiver head + shot head for sequential inference:
    Stage 1: Predict receiver from corner graph
    Stage 2: Predict shot conditioned on receiver identity
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .backbone import CornerBackbone
from .receiver_head import ReceiverHead, masked_softmax
from .shot_head import ShotHead


class TwoStageModel(nn.Module):
    """Combined two-stage model for corner kick prediction.

    The model augments node features with a receiver indicator column
    (14th feature = is_predicted_receiver). For Stage 1 this is all zeros.
    For Stage 2 it is 1.0 at the predicted (or oracle) receiver node.

    Args:
        backbone: GNN backbone producing per-node embeddings.
        receiver_head: Stage 1 head for receiver prediction.
        shot_head: Stage 2 head for shot prediction.
    """

    # Index of the receiver indicator feature (appended as 14th column)
    RECEIVER_FEATURE_IDX = 13

    def __init__(
        self,
        backbone: CornerBackbone,
        receiver_head: ReceiverHead,
        shot_head: ShotHead,
    ):
        super().__init__()
        self.backbone = backbone
        self.receiver_head = receiver_head
        self.shot_head = shot_head

    @staticmethod
    def _augment_with_receiver(
        x: torch.Tensor,
        receiver_indicator: torch.Tensor = None,
    ) -> torch.Tensor:
        """Append receiver indicator as 14th node feature.

        Args:
            x: Node features [N, 13].
            receiver_indicator: [N] tensor with 1.0 at receiver node(s),
                0.0 elsewhere. If None, appends all zeros.

        Returns:
            Augmented features [N, 14].
        """
        n_nodes = x.shape[0]
        if receiver_indicator is None:
            indicator = torch.zeros(n_nodes, 1, device=x.device, dtype=x.dtype)
        else:
            indicator = receiver_indicator.unsqueeze(-1)
        return torch.cat([x, indicator], dim=-1)

    def predict_receiver(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        receiver_mask: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Stage 1: Predict receiver probabilities.

        Args:
            x: Node features [N, 13] (from build_graphs).
            edge_index: Edge indices [2, E].
            edge_attr: Edge features [E, 4].
            receiver_mask: Boolean mask [N] — True for valid candidates.
            batch: Graph membership [N].

        Returns:
            Receiver probabilities [N]. Valid candidates sum to 1.0 per graph.
        """
        # Augment with zeros (no receiver info for Stage 1)
        x_aug = self._augment_with_receiver(x)

        # Backbone → per-node embeddings
        node_emb = self.backbone(x_aug, edge_index, edge_attr)

        # Receiver head → logits → masked softmax
        logits = self.receiver_head(node_emb)
        probs = masked_softmax(logits, receiver_mask, batch)
        return probs

    def predict_shot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        graph_features: torch.Tensor = None,
        receiver_indicator: torch.Tensor = None,
    ) -> torch.Tensor:
        """Stage 2: Predict shot probability.

        Args:
            x: Node features [N, 13].
            edge_index: Edge indices [2, E].
            edge_attr: Edge features [E, 4].
            batch: Graph membership [N].
            graph_features: Optional graph-level features [B, graph_feature_dim].
            receiver_indicator: Optional [N] tensor with 1.0 at predicted receiver.
                If None, runs unconditional (no receiver info).

        Returns:
            Shot logit [B, 1].
        """
        # Augment with receiver indicator (or zeros for unconditional)
        x_aug = self._augment_with_receiver(x, receiver_indicator)

        # Backbone → per-node embeddings
        node_emb = self.backbone(x_aug, edge_index, edge_attr)

        # Global mean pool → graph embedding
        graph_emb = global_mean_pool(node_emb, batch)

        # Default graph_features to zeros if ShotHead expects them
        if graph_features is None and self.shot_head.graph_feature_dim > 0:
            n_graphs = graph_emb.shape[0]
            graph_features = torch.zeros(
                n_graphs, self.shot_head.graph_feature_dim,
                device=graph_emb.device, dtype=graph_emb.dtype,
            )

        # Shot head
        return self.shot_head(graph_emb, graph_features)

    def forward_two_stage(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        receiver_mask: torch.Tensor,
        batch: torch.Tensor,
        graph_features: torch.Tensor = None,
    ) -> dict:
        """Full two-stage inference: predict receiver then shot.

        Args:
            x: Node features [N, 13].
            edge_index: Edge indices [2, E].
            edge_attr: Edge features [E, 4].
            receiver_mask: Boolean mask [N].
            batch: Graph membership [N].
            graph_features: Optional graph-level features [B, graph_feature_dim].

        Returns:
            Dict with:
                "receiver_probs": [N] probabilities
                "predicted_receiver": [N] one-hot indicator
                "shot_logit": [B, 1] shot logits
        """
        # Stage 1: Predict receiver
        receiver_probs = self.predict_receiver(
            x, edge_index, edge_attr, receiver_mask, batch,
        )

        # Select predicted receiver (argmax per graph among masked nodes)
        receiver_indicator = torch.zeros_like(receiver_probs)
        n_graphs = batch.max().item() + 1
        for g in range(n_graphs):
            graph_mask = batch == g
            graph_probs = receiver_probs[graph_mask]
            if graph_probs.sum() > 0:
                local_idx = graph_probs.argmax()
                global_indices = graph_mask.nonzero(as_tuple=True)[0]
                receiver_indicator[global_indices[local_idx]] = 1.0

        # Stage 2: Predict shot conditioned on predicted receiver
        shot_logit = self.predict_shot(
            x, edge_index, edge_attr, batch, graph_features, receiver_indicator,
        )

        return {
            "receiver_probs": receiver_probs,
            "predicted_receiver": receiver_indicator,
            "shot_logit": shot_logit,
        }
