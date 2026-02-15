"""Tests for corner_prediction.models — backbone, receiver head, shot head, two-stage model."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from corner_prediction.models.backbone import CornerBackbone
from corner_prediction.models.receiver_head import (
    ReceiverHead,
    masked_softmax,
    receiver_loss,
)
from corner_prediction.models.shot_head import ShotHead
from corner_prediction.models.two_stage import TwoStageModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_graph(
    n_nodes: int = 22,
    n_attackers: int = 11,
    node_dim: int = 13,
    edge_k: int = 6,
    has_receiver: bool = True,
    receiver_idx: int = 3,
    shot_label: int = 0,
) -> Data:
    """Create a synthetic corner kick graph with correct shapes."""
    x = torch.randn(n_nodes, node_dim)

    # KNN-like edges: k neighbors per node
    src, dst = [], []
    for i in range(n_nodes):
        neighbors = [(i + j + 1) % n_nodes for j in range(edge_k)]
        for j in neighbors:
            src.append(i)
            dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 4)

    # Receiver mask: first n_attackers are attacking; exclude node 0 as GK
    receiver_mask = torch.zeros(n_nodes, dtype=torch.bool)
    for i in range(1, n_attackers):  # skip 0 (GK)
        receiver_mask[i] = True

    # Receiver label
    receiver_label = torch.zeros(n_nodes, dtype=torch.float32)
    if has_receiver and receiver_mask[receiver_idx]:
        receiver_label[receiver_idx] = 1.0

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        receiver_mask=receiver_mask,
        receiver_label=receiver_label,
        has_receiver_label=has_receiver,
        shot_label=shot_label,
        goal_label=0,
        corner_side=1.0,
        match_id="test_match_1",
        corner_id="test_corner_1",
        detection_rate=0.85,
    )


def _make_batch(n_graphs: int = 4, **kwargs) -> Batch:
    """Create a batch of dummy graphs."""
    graphs = []
    for i in range(n_graphs):
        g = _make_dummy_graph(
            shot_label=1 if i % 3 == 0 else 0,
            has_receiver=i % 2 == 0,
            **kwargs,
        )
        g.match_id = f"match_{i}"
        g.corner_id = f"corner_{i}"
        graphs.append(g)
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# TestCornerBackbone
# ---------------------------------------------------------------------------

class TestCornerBackbone:
    """Tests for CornerBackbone."""

    def test_pretrained_output_shape(self):
        backbone = CornerBackbone(mode="pretrained", freeze=False)
        batch = _make_batch()
        # Augment to 14 features (as TwoStageModel would)
        x_aug = torch.cat([batch.x, torch.zeros(batch.x.shape[0], 1)], dim=-1)
        out = backbone(x_aug, batch.edge_index, batch.edge_attr)
        assert out.shape == (batch.x.shape[0], 128)

    def test_scratch_output_shape(self):
        backbone = CornerBackbone(mode="scratch")
        batch = _make_batch()
        x_aug = torch.cat([batch.x, torch.zeros(batch.x.shape[0], 1)], dim=-1)
        out = backbone(x_aug, batch.edge_index, batch.edge_attr)
        assert out.shape == (batch.x.shape[0], 64)

    def test_pretrained_frozen_params(self):
        backbone = CornerBackbone(mode="pretrained", freeze=True)
        # Conv and lin_in params should be frozen
        for param in backbone.conv1.parameters():
            assert not param.requires_grad
        for param in backbone.lin_in.parameters():
            assert not param.requires_grad
        for conv in backbone.convs:
            for param in conv.parameters():
                assert not param.requires_grad

    def test_pretrained_projection_trainable(self):
        backbone = CornerBackbone(mode="pretrained", freeze=True)
        # Projection layers should be trainable
        for param in backbone.node_proj.parameters():
            assert param.requires_grad
        for param in backbone.edge_proj.parameters():
            assert param.requires_grad

    def test_scratch_all_trainable(self):
        backbone = CornerBackbone(mode="scratch")
        for param in backbone.parameters():
            assert param.requires_grad

    def test_output_dim_property(self):
        assert CornerBackbone(mode="pretrained", freeze=False).output_dim == 128
        assert CornerBackbone(mode="scratch").output_dim == 64

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            CornerBackbone(mode="invalid")

    def test_forward_no_nan(self):
        backbone = CornerBackbone(mode="scratch")
        batch = _make_batch()
        x_aug = torch.cat([batch.x, torch.zeros(batch.x.shape[0], 1)], dim=-1)
        out = backbone(x_aug, batch.edge_index, batch.edge_attr)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# TestReceiverHead
# ---------------------------------------------------------------------------

class TestReceiverHead:
    """Tests for ReceiverHead and related functions."""

    def test_output_shape(self):
        head = ReceiverHead(input_dim=64)
        emb = torch.randn(22, 64)
        logits = head(emb)
        assert logits.shape == (22,)

    def test_masked_softmax_sums_to_one(self):
        logits = torch.randn(22)
        mask = torch.zeros(22, dtype=torch.bool)
        mask[1:11] = True  # 10 candidates
        batch = torch.zeros(22, dtype=torch.long)

        probs = masked_softmax(logits, mask, batch)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_masked_softmax_zeros_masked(self):
        logits = torch.randn(22)
        mask = torch.zeros(22, dtype=torch.bool)
        mask[1:11] = True
        batch = torch.zeros(22, dtype=torch.long)

        probs = masked_softmax(logits, mask, batch)
        # Non-candidate nodes should have 0 probability
        assert (probs[~mask] == 0).all()

    def test_masked_softmax_multi_graph(self):
        """Each graph's candidates sum to 1.0 independently."""
        logits = torch.randn(44)
        mask = torch.zeros(44, dtype=torch.bool)
        mask[1:11] = True   # graph 0 candidates
        mask[23:33] = True  # graph 1 candidates
        batch = torch.cat([torch.zeros(22, dtype=torch.long),
                           torch.ones(22, dtype=torch.long)])

        probs = masked_softmax(logits, mask, batch)

        # Each graph sums to 1
        graph0_sum = probs[batch == 0].sum()
        graph1_sum = probs[batch == 1].sum()
        assert abs(graph0_sum.item() - 1.0) < 1e-5
        assert abs(graph1_sum.item() - 1.0) < 1e-5

    def test_receiver_loss_computes(self):
        head = ReceiverHead(input_dim=64)
        emb = torch.randn(22, 64)
        logits = head(emb)

        receiver_label = torch.zeros(22)
        receiver_label[3] = 1.0
        receiver_mask = torch.zeros(22, dtype=torch.bool)
        receiver_mask[1:11] = True
        batch = torch.zeros(22, dtype=torch.long)
        has_label = torch.tensor([True])

        loss = receiver_loss(logits, receiver_label, receiver_mask, batch, has_label)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_receiver_loss_gradient_flows(self):
        head = ReceiverHead(input_dim=64)
        emb = torch.randn(22, 64, requires_grad=True)
        logits = head(emb)

        receiver_label = torch.zeros(22)
        receiver_label[3] = 1.0
        receiver_mask = torch.zeros(22, dtype=torch.bool)
        receiver_mask[1:11] = True
        batch = torch.zeros(22, dtype=torch.long)
        has_label = torch.tensor([True])

        loss = receiver_loss(logits, receiver_label, receiver_mask, batch, has_label)
        loss.backward()
        assert emb.grad is not None
        assert not torch.isnan(emb.grad).any()

    def test_receiver_loss_no_labels_returns_zero(self):
        logits = torch.randn(22)
        receiver_label = torch.zeros(22)
        receiver_mask = torch.zeros(22, dtype=torch.bool)
        receiver_mask[1:11] = True
        batch = torch.zeros(22, dtype=torch.long)
        has_label = torch.tensor([False])

        loss = receiver_loss(logits, receiver_label, receiver_mask, batch, has_label)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# TestShotHead
# ---------------------------------------------------------------------------

class TestShotHead:
    """Tests for ShotHead."""

    def test_output_shape(self):
        head = ShotHead(input_dim=128, graph_feature_dim=1)
        graph_emb = torch.randn(4, 128)
        graph_feat = torch.randn(4, 1)
        out = head(graph_emb, graph_feat)
        assert out.shape == (4, 1)

    def test_with_graph_features(self):
        head = ShotHead(input_dim=64, graph_feature_dim=2)
        graph_emb = torch.randn(3, 64)
        graph_feat = torch.randn(3, 2)
        out = head(graph_emb, graph_feat)
        assert out.shape == (3, 1)

    def test_without_graph_features(self):
        head = ShotHead(input_dim=64, graph_feature_dim=0)
        graph_emb = torch.randn(3, 64)
        out = head(graph_emb)
        assert out.shape == (3, 1)

    def test_gradient_flows(self):
        head = ShotHead(input_dim=64, graph_feature_dim=1)
        graph_emb = torch.randn(2, 64, requires_grad=True)
        graph_feat = torch.randn(2, 1)
        out = head(graph_emb, graph_feat)
        loss = out.sum()
        loss.backward()
        assert graph_emb.grad is not None


# ---------------------------------------------------------------------------
# TestTwoStageModel
# ---------------------------------------------------------------------------

class TestTwoStageModel:
    """Tests for TwoStageModel end-to-end."""

    @pytest.fixture
    def model(self):
        backbone = CornerBackbone(mode="scratch")  # 64-dim output
        receiver_head = ReceiverHead(input_dim=64)
        shot_head = ShotHead(input_dim=64, graph_feature_dim=1)
        return TwoStageModel(backbone, receiver_head, shot_head)

    def test_predict_receiver_output(self, model):
        batch = _make_batch()
        probs = model.predict_receiver(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.receiver_mask, batch.batch,
        )
        assert probs.shape == (batch.x.shape[0],)
        # Probabilities should be non-negative
        assert (probs >= 0).all()
        # Masked nodes should have 0 probability
        assert (probs[~batch.receiver_mask] == 0).all()

    def test_predict_shot_unconditional(self, model):
        batch = _make_batch()
        n_graphs = batch.batch.max().item() + 1
        logit = model.predict_shot(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch,
        )
        assert logit.shape == (n_graphs, 1)

    def test_predict_shot_with_receiver(self, model):
        batch = _make_batch()
        n_graphs = batch.batch.max().item() + 1
        # Set node 3 as receiver in each graph
        receiver_indicator = torch.zeros(batch.x.shape[0])
        for g in range(n_graphs):
            graph_nodes = (batch.batch == g).nonzero(as_tuple=True)[0]
            receiver_indicator[graph_nodes[3]] = 1.0

        logit = model.predict_shot(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch,
            receiver_indicator=receiver_indicator,
        )
        assert logit.shape == (n_graphs, 1)

    def test_predict_shot_with_graph_features(self, model):
        batch = _make_batch()
        n_graphs = batch.batch.max().item() + 1
        graph_features = torch.randn(n_graphs, 1)
        logit = model.predict_shot(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch,
            graph_features=graph_features,
        )
        assert logit.shape == (n_graphs, 1)

    def test_end_to_end_pipeline(self, model):
        batch = _make_batch()
        result = model.forward_two_stage(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.receiver_mask, batch.batch,
        )

        n_nodes = batch.x.shape[0]
        n_graphs = batch.batch.max().item() + 1

        assert result["receiver_probs"].shape == (n_nodes,)
        assert result["predicted_receiver"].shape == (n_nodes,)
        assert result["shot_logit"].shape == (n_graphs, 1)

        # predicted_receiver should have exactly one 1.0 per graph
        for g in range(n_graphs):
            graph_mask = batch.batch == g
            n_selected = result["predicted_receiver"][graph_mask].sum()
            assert n_selected.item() == 1.0

    def test_augment_with_receiver_dimensions(self):
        x = torch.randn(22, 13)

        # No receiver → zeros appended
        x_aug = TwoStageModel._augment_with_receiver(x)
        assert x_aug.shape == (22, 14)
        assert (x_aug[:, 13] == 0).all()

        # With receiver at node 5
        indicator = torch.zeros(22)
        indicator[5] = 1.0
        x_aug = TwoStageModel._augment_with_receiver(x, indicator)
        assert x_aug.shape == (22, 14)
        assert x_aug[5, 13] == 1.0
        assert x_aug[0, 13] == 0.0

    def test_batch_inference(self, model):
        """Works with different batch sizes."""
        for n in [1, 2, 8]:
            batch = _make_batch(n_graphs=n)
            result = model.forward_two_stage(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.receiver_mask, batch.batch,
            )
            assert result["shot_logit"].shape[0] == n

    def test_receiver_conditioning_changes_output(self, model):
        """Adding receiver indicator must change shot prediction."""
        model.eval()
        batch = _make_batch(n_graphs=2)
        n_graphs = batch.batch.max().item() + 1

        with torch.no_grad():
            # Unconditional (no receiver info)
            logit_uncond = model.predict_shot(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
            )

            # With receiver indicator at node 3 of each graph
            indicator = torch.zeros(batch.x.shape[0])
            for g in range(n_graphs):
                nodes = (batch.batch == g).nonzero(as_tuple=True)[0]
                indicator[nodes[3]] = 1.0

            logit_cond = model.predict_shot(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                receiver_indicator=indicator,
            )

        assert not torch.allclose(logit_uncond, logit_cond), \
            "Receiver conditioning should change shot prediction"

    def test_receiver_loss_mixed_batch(self):
        """Loss works with a batch where some graphs have labels and some don't."""
        from corner_prediction.models.receiver_head import receiver_loss, ReceiverHead

        head = ReceiverHead(input_dim=64)
        # 2 graphs, 22 nodes each
        emb = torch.randn(44, 64)
        logits = head(emb)

        receiver_label = torch.zeros(44)
        receiver_label[3] = 1.0  # graph 0 has a receiver at node 3

        receiver_mask = torch.zeros(44, dtype=torch.bool)
        receiver_mask[1:11] = True   # graph 0 candidates
        receiver_mask[23:33] = True  # graph 1 candidates

        batch = torch.cat([torch.zeros(22, dtype=torch.long),
                           torch.ones(22, dtype=torch.long)])
        has_label = torch.tensor([True, False])  # only graph 0 is labeled

        loss = receiver_loss(logits, receiver_label, receiver_mask, batch, has_label)
        assert loss.shape == ()
        assert loss.item() > 0
