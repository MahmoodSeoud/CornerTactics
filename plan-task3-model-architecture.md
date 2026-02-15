# Task 3: Model Architecture — Implementation Plan

## Overview

Build the two-stage corner kick prediction model: receiver prediction (Stage 1) → conditional shot prediction (Stage 2), with pretrained USSF backbone integration.

## Files to Create

```
corner_prediction/
├── models/
│   ├── __init__.py              # Export main classes
│   ├── backbone.py              # GNN backbone (pretrained + from-scratch)
│   ├── receiver_head.py         # Stage 1: per-node receiver prediction
│   ├── shot_head.py             # Stage 2: graph-level shot prediction
│   └── two_stage.py             # Combined two-stage inference model
tests/
└── corner_prediction/
    └── test_models.py           # All model tests
```

## File 1: `corner_prediction/models/__init__.py`

Export: `CornerBackbone`, `ReceiverHead`, `ShotHead`, `TwoStageModel`

## File 2: `corner_prediction/models/backbone.py`

### Class: `CornerBackbone(nn.Module)`

Wraps CrystalConv GNN layers. Two modes: pretrained (with feature projection) and from-scratch.

**Constructor args:**
- `mode: str = "pretrained"` — "pretrained" or "scratch"
- `node_features: int = 13` — input node feature dim (from build_graphs)
- `edge_features: int = 4` — input edge feature dim (from build_graphs)
- `pretrained_path: str = None` — path to USSF backbone weights
- `hidden_channels: int = 128` — backbone hidden dim (128 for pretrained, 64 for scratch)
- `num_conv_layers: int = 3` — number of CGConv layers
- `freeze: bool = True` — freeze backbone conv layers

**Pretrained mode architecture:**
```
Input: x [B*22, 13], edge_attr [B*E, 4]
  → node_proj: Linear(13, 12)        # Trainable projection
  → edge_proj: Linear(4, 6)          # Trainable projection
  → conv1: CGConv(channels=12, dim=6)  # FROZEN, loaded from USSF
  → ReLU
  → lin_in: Linear(12, 128)            # FROZEN, loaded from USSF
  → conv2: CGConv(channels=128, dim=6)  # FROZEN
  → ReLU
  → conv3: CGConv(channels=128, dim=6)  # FROZEN
  → ReLU
Output: node_embeddings [B*22, 128]
```

Note: edge features are projected once and reused across all layers. CGConv layers use `dim=6` (matches pretrained).

**From-scratch mode architecture:**
```
Input: x [B*22, 13], edge_attr [B*E, 4]
  → conv1: CGConv(channels=13, dim=4)    # Trainable
  → ReLU
  → lin_in: Linear(13, 64)               # Trainable
  → conv2: CGConv(channels=64, dim=4)    # Trainable
  → ReLU
  → conv3: CGConv(channels=64, dim=4)    # Trainable
  → ReLU
Output: node_embeddings [B*22, 64]
```

No projection layers needed — CGConv accepts any feature dim directly.

**Methods:**
- `forward(x, edge_index, edge_attr) -> Tensor` — returns per-node embeddings [N, hidden_channels]
- `load_pretrained(path)` — loads USSF backbone state dict
- `output_dim` property — returns hidden_channels (128 for pretrained, 64 for scratch)

**Key detail**: edge_proj output is cached per forward call since all layers use the same projected edge features.

## File 3: `corner_prediction/models/receiver_head.py`

### Class: `ReceiverHead(nn.Module)`

Per-node classification with masked softmax over receiver candidates.

**Constructor args:**
- `input_dim: int = 128` — backbone output dim
- `hidden_dim: int = 64`
- `dropout: float = 0.3`

**Architecture:**
```
Input: node_embeddings [N, input_dim]
  → Linear(input_dim, hidden_dim)
  → ReLU
  → Dropout(dropout)
  → Linear(hidden_dim, 1)              # Per-node logit
  → Squeeze → [N]
Output: per-node logits [N]
```

**Methods:**
- `forward(node_embeddings: Tensor) -> Tensor` — returns raw logits [N]

**Separate function** (not method, module-level):
- `masked_softmax(logits, mask, batch=None) -> Tensor` — applies mask, computes per-graph softmax
  - Sets logits to -inf where mask=False
  - Applies softmax per graph (using scatter_softmax or manual batch indexing)
  - Returns probabilities [N]

**Separate function:**
- `receiver_loss(logits, receiver_label, receiver_mask, batch) -> Tensor`
  - Computes cross-entropy over masked candidates per graph
  - For each graph: log_softmax over masked nodes, pick the receiver node
  - Returns mean loss across graphs that have receiver labels

## File 4: `corner_prediction/models/shot_head.py`

### Class: `ShotHead(nn.Module)`

Graph-level binary classification with optional graph-level features.

**Constructor args:**
- `input_dim: int = 128` — backbone output dim (after pooling)
- `graph_feature_dim: int = 1` — graph-level features (corner_side)
- `hidden_dim: int = 32`
- `dropout: float = 0.3`

**Architecture:**
```
Input: graph_embedding [B, input_dim], graph_features [B, graph_feature_dim]
  → cat(graph_embedding, graph_features) → [B, input_dim + graph_feature_dim]
  → Linear(input_dim + graph_feature_dim, hidden_dim)
  → ReLU
  → Dropout(dropout)
  → Linear(hidden_dim, 1)
Output: shot logit [B, 1]
```

**Methods:**
- `forward(graph_embedding, graph_features=None) -> Tensor` — returns logit [B, 1]
  - If graph_features is None, uses input_dim only (no concat)

## File 5: `corner_prediction/models/two_stage.py`

### Class: `TwoStageModel(nn.Module)`

Combines backbone + receiver head + shot head for joint or sequential use.

**Constructor args:**
- `backbone: CornerBackbone`
- `receiver_head: ReceiverHead`
- `shot_head: ShotHead`

**Methods:**
- `predict_receiver(data) -> Tensor` — Stage 1 forward pass
  - Runs backbone on graph → node_embeddings
  - Runs receiver_head → logits
  - Applies masked_softmax with receiver_mask → probabilities
  - Returns probabilities [N]

- `predict_shot(data, receiver_node_idx=None) -> Tensor` — Stage 2 forward pass
  - If receiver_node_idx provided: adds is_predicted_receiver feature to node features
    - Creates augmented x: [N, 14] by concatenating a receiver indicator column
    - Backbone must handle 14 features → need a separate projection for Stage 2
  - Runs backbone on (augmented) graph → node_embeddings
  - Global mean pool → graph_embedding [B, hidden_dim]
  - Runs shot_head(graph_embedding, graph_features) → logit [B, 1]

- `forward(data, mode="receiver") -> Tensor` — dispatch to either stage

### Design choice: receiver conditioning

**Problem**: Adding a receiver feature changes input dim from 13 to 14, breaking the backbone projection.

**Solution**: The backbone's node_proj handles the mismatch. Two options:

**Option A** (simpler): Backbone has two projection layers — `node_proj_stage1: Linear(13, 12)` and `node_proj_stage2: Linear(14, 12)`. Pass a `stage` argument to backbone.forward().

**Option B** (cleaner): TwoStageModel adds receiver indicator *after* the backbone, concatenating it to the node embeddings before receiver_head/shot_head. For Stage 2 shot prediction with receiver conditioning, the receiver indicator goes into the graph embedding rather than through message passing.

**Decision**: Option A. The spec says "Add receiver features to graph" before running backbone, and message passing with receiver info is more expressive. Backbone takes a `node_dim` parameter in forward to select the right projection.

Actually, let me reconsider. The simpler approach is:

**Option C**: Always use 14-dim input. For Stage 1, the 14th feature (is_predicted_receiver) is always 0. For Stage 2, it's 1 at the predicted receiver node. This way there's one projection layer: `Linear(14, 12)` for pretrained, and from-scratch uses `CGConv(channels=14, dim=4)`.

**Decision**: Option C. One backbone, one projection, receiver conditioning is a 14th input feature that's 0 during Stage 1 and set during Stage 2.

### Updated backbone node_features = 14

This changes build_graphs output → NO. Keep build_graphs at 13 features. TwoStageModel is responsible for concatenating the 14th feature (zeros for Stage 1, receiver indicator for Stage 2) before passing to backbone.

### Updated File 2 (backbone):
- `node_features: int = 14` for pretrained mode (13 original + 1 receiver indicator)
- Projection: `Linear(14, 12)`
- From-scratch: `CGConv(channels=14, dim=4)`

## File 6: `tests/corner_prediction/test_models.py`

### Test helpers
- `_make_dummy_graph(n_nodes=22, n_attackers=11)` — creates a synthetic PyG Data object with correct shapes (node features [22, 13], edge features, labels, masks)
- `_make_batch(n_graphs=4)` — creates a batch of graphs via `Batch.from_data_list`

### TestCornerBackbone (~6 tests)
- `test_pretrained_output_shape` — output [N, 128]
- `test_scratch_output_shape` — output [N, 64]
- `test_pretrained_frozen_params` — backbone params have requires_grad=False
- `test_pretrained_projection_trainable` — node_proj and edge_proj have requires_grad=True
- `test_forward_with_receiver_feature` — works with 14-dim input (13 + receiver)
- `test_load_pretrained_weights` — loads USSF weights without error (skip if weights not available)

### TestReceiverHead (~5 tests)
- `test_output_shape` — logits [N]
- `test_masked_softmax_sums_to_one` — per-graph probabilities sum to 1.0
- `test_masked_softmax_zeros_masked` — masked positions have 0 probability
- `test_receiver_loss_computes` — returns scalar loss
- `test_receiver_loss_gradient_flows` — loss.backward() succeeds

### TestShotHead (~4 tests)
- `test_output_shape` — logit [B, 1]
- `test_with_graph_features` — accepts graph features
- `test_without_graph_features` — works without graph features (graph_feature_dim=0)
- `test_gradient_flows` — loss.backward() succeeds

### TestTwoStageModel (~5 tests)
- `test_predict_receiver_output` — returns probabilities [N]
- `test_predict_shot_unconditional` — works without receiver
- `test_predict_shot_with_receiver` — works with receiver node index
- `test_end_to_end_pipeline` — receiver → shot sequential inference
- `test_batch_inference` — works with batched graphs

## Step-by-step execution order

1. Create `corner_prediction/models/__init__.py`
2. Create `corner_prediction/models/backbone.py` — CornerBackbone
3. Create `corner_prediction/models/receiver_head.py` — ReceiverHead + masked_softmax + receiver_loss
4. Create `corner_prediction/models/shot_head.py` — ShotHead
5. Create `corner_prediction/models/two_stage.py` — TwoStageModel
6. Create `tests/corner_prediction/test_models.py` — all tests
7. Run tests — verify all pass
8. Smoke test: load real graphs + pretrained weights, run forward pass

## Acceptance criteria

- All tests pass
- Pretrained backbone loads USSF weights and produces [N, 128] embeddings
- From-scratch backbone produces [N, 64] embeddings
- Receiver head produces per-node probabilities summing to 1.0 per graph (over mask)
- Shot head produces [B, 1] logits
- Two-stage model runs end-to-end: data → receiver probs → shot logit
- Frozen params don't receive gradients; trainable params do
- Stage 2 correctly incorporates receiver identity as 14th feature
