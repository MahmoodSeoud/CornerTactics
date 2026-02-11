# Phase 3: Model Architecture & Training

## Status: In Progress

## Goal
Build ST-GNN, pretrain on open play, fine-tune on corners, run velocity ablation experiment.

## Key Components (from plan.md)

### 3.1 Spatio-Temporal GNN Architecture
- **SpatialGNN**: GATv2Conv layers to process single frame's graph
  - Input: 8 node features (x, y, vx, vy, team_flag, is_kicker, dist_to_goal, dist_to_ball)
  - Output: 32-dim graph-level embedding
- **TemporalAggregator**: GRU to aggregate frame-level representations over time
  - Input: sequence of 32-dim embeddings
  - Output: 64-dim temporal embedding
- **CornerKickPredictor**: Full ST-GNN with multi-head outputs
  - head_shot: Binary shot prediction
  - head_goal: Binary goal prediction
  - head_contact: First contact team (2-class)
  - head_outcome: Outcome class (6-class)

### 3.2 Pretrain on Open-Play Sequences
- Extract overlapping windows from all 7 DFL matches (not just corners)
- Label: will there be a shot within 6 seconds?
- Train spatial GNN only (no temporal) on ~1000-3000 sequences
- This enables transfer learning for the small corner dataset (~70 samples)

### 3.3 Fine-tune on Corner Kicks
- Leave-one-match-out cross-validation (7 folds)
- Multi-task loss: shot + goal + contact + outcome
- Train full model (spatial + temporal + all heads)

### 3.4 Velocity Ablation (KEY EXPERIMENT)
- Condition A: Position-only features (zero out vx, vy)
- Condition B: Position + Velocity features
- Statistical test: paired t-test on AUC across folds
- Expected: Position+Velocity AUC > 0.55, Position-only AUC ~ 0.50

## Implementation Order (TDD)
1. SpatialGNN module - process single frame graph
2. TemporalAggregator module - GRU for temporal
3. CornerKickPredictor - full model with multi-head outputs
4. Open-play sequence extraction
5. Pretraining loop
6. Fine-tuning with cross-validation
7. Ablation experiment runner

## Dependencies
- torch>=2.0
- torch-geometric>=2.4
- Phase 2 modules (graph_construction, data_loading)

## Questions/Clarifications
(None yet)
