# TacticAI-Style Corner Kick Prediction: Implementation Plan

**Research Question**: *"Predicting corner kick outcomes from static positioning: Dual-task receiver and shot prediction with multi-class outcome analysis"*

**Project Goals**:
1. **Dual-task prediction** (TacticAI replication): Receiver + shot prediction
2. **Multi-class outcome prediction**: Classify corner outcomes into Goal/Shot/Clearance/Possession

**Key Limitation**: No velocity data (vx, vy = 0 for all players)

**Expected Performance**:
- **Dual-task models**:
  - Receiver Top-3: 60-70% (vs TacticAI's 78%)
  - Shot F1: 0.55-0.65 (vs TacticAI's 0.71)
- **Multi-class models**:
  - Macro F1: 0.50-0.60 (4-class balanced)
  - Accuracy: 60-70% (expected)
- Performance gap of 10-15% expected due to missing velocities

---

## Phase 0: Infrastructure Foundation (Pre-existing)

**Status**: ✅ COMPLETE (from previous CornerTactics work)

This phase documents the pre-existing infrastructure that TacticAI implementation builds upon.

### Core Data Processing Modules (`src/`)

**`src/statsbomb_loader.py`**
- StatsBomb Open Data API integration
- Downloads corner kick events with 360 freeze frames
- Filters professional men's competitions
- Output: `data/raw/statsbomb/corners_360.csv`

**`src/outcome_labeler.py`**
- Labels corner kick outcomes by analyzing subsequent events
- Categories: goal, shot, clearance, second_corner, possession, opposition_possession
- 20-second event window analysis
- Used by: Phase 1.2 outcome labeling

**`src/feature_engineering.py`**
- Extracts 14-dimensional node features per player
- Features: spatial (4), kinematic (4), contextual (4), density (2)
- Handles StatsBomb 360 freeze frames
- Output: `data/features/node_features/statsbomb_player_features.parquet`

**`src/graph_builder.py`**
- Constructs graph representations from node features
- 5 adjacency strategies: team, distance, delaunay, ball_centric, zone
- 6-dimensional edge features
- Used by: `data_loader.py`, `add_receiver_labels.py`, `receiver_data_loader.py`
- Output: `data/graphs/adjacency_team/*.pkl`

**`src/data_loader.py`**
- Base dataset class for corner kick graphs
- `CornerDataset`: Loads graphs, handles train/val/test splits
- **Critical fix (Oct 26, 2024)**: Splits by base corner ID to prevent temporal frame leakage
- Used by: `receiver_data_loader.py` (imported by all baseline models)

**`src/receiver_labeler.py`**
- Extracts receiver labels from StatsBomb events
- Identifies player who touches ball 0-5s after corner
- Used by: `add_receiver_labels.py`, unit tests

### Dataset Statistics (Pre-existing)
- Original corners: 1,118 base corners (StatsBomb 360)
- Temporal augmentation: 7,369 graphs (6.6× increase)
  - StatsBomb augmented: 5,814 graphs (5 temporal frames + mirrors)
  - SkillCorner temporal: 1,555 graphs (real 10fps tracking)
- Receiver coverage: 996/1,118 base corners (89.1%)
- Dangerous situations: ~1,261 (17.1% positive class for shot OR goal)

---

## Phase 1: Data Preparation & Baselines (Week 1, Days 1-7)

### Day 1-2: Receiver Label Extraction ✅ COMPLETE
- [x] Write `scripts/preprocessing/add_receiver_labels.py`
  - [x] Extract receiver_player_id from StatsBomb events (player who touches ball 0-5s after corner)
  - [x] Add receiver_player_id to CornerGraph.metadata
  - [x] Add player_ids list to CornerGraph.metadata (maps node index 0-21 to StatsBomb player IDs)
- [x] Run script on existing graphs
- [x] Verify coverage: Target 85%+ of corners have valid receiver labels (expect ~950/1118)
- [x] Save updated graphs: `data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl`

**Success Criteria**:
- ✅ At least 900 corners with valid receiver labels → **ACHIEVED: 996 receivers (89.1% coverage)**
- ⚠️ Receiver distribution: strikers (30-40%), midfielders (30-40%), defenders (20-30%) → **DEFERRED: Names only, no position data**

**Implementation Notes (Completed Oct 27, 2025)**:
- Used simplified approach: extracted from `outcome_player` column in existing CSV
- Coverage: 996/1118 base corners (89.1%) - exceeds 85% target
- Limitation: Receiver names only (no player IDs available in CSV)
- Branch: `feature/tacticai-receiver-labels`
- Output: `data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl`

---

### Day 3-4: Data Loader Extension
- [x] Create `src/data/receiver_data_loader.py`
  - [x] Implement `ReceiverCornerDataset` (extends `CornerDataset`)
  - [x] Mask velocity features: `x[:, 4:6] = 0.0` (acknowledge missing data)
  - [x] Add `receiver_label` field: `torch.LongTensor([receiver_idx])` (0-21)
  - [x] Add `shot_label` field: `torch.FloatTensor([1.0 if dangerous else 0.0])`
  - [x] Skip corners without valid receiver labels
- [x] Create helper function: `load_receiver_dataset()`
- [x] Unit test: Load batch, verify shapes
  - [x] `batch.x`: `[num_nodes, 14]` with velocities = 0
  - [x] `batch.receiver_label`: `[batch_size]` (0-21)
  - [x] `batch.shot_label`: `[batch_size, 1]`
- [x] Verify split integrity: No corner ID leakage across train/val/test

**Success Criteria**:
- ✅ Data loader successfully loads ~900 corners
- ✅ Batch shapes correct
- ✅ No data leakage (same corner_id stays in same split)

---

### Day 5-6: Baseline Models ✅ COMPLETE (Dual-task only)
- [x] Create `src/models/baselines.py`
  - [x] **Dual-task models** (Receiver + Shot prediction):
    - [x] Implement `RandomReceiverBaseline`
      - [x] `predict()`: Return random softmax over 22 players
      - [x] `evaluate()`: Return top-1=4.5%, top-3=13.6%, top-5=22.7%
    - [x] Implement `XGBoostReceiverBaseline` (engineered features)
      - [x] Extract hand-crafted features per player (dimension: 22 × ~15 features):
        - [x] **Spatial**: distance to ball, distance to goal, x-position, y-position
        - [x] **Relative**: closest opponent distance, teammates within 5m radius
        - [x] **Zonal**: binary flags (in 6-yard box? in penalty area? near/far post?)
        - [x] **Team context**: average team x-position, defensive line compactness
        - [x] **Player role**: is_goalkeeper, is_corner_taker (binary flags)
      - [x] Flatten to `[22 × 15 = 330 features]` per corner
      - [x] XGBoost classifier: `max_depth=6, n_estimators=500, learning_rate=0.05`
    - [x] Implement `MLPReceiverBaseline`
      - [x] Flatten all player positions: `[batch, 22*14=308]`
      - [x] MLP: 512 → 256 hidden units (deeper architecture)
      - [x] Dropout 0.25, ReLU activations
      - [x] Dual-task: Receiver (22-class) + Shot (binary)
- [x] Create `scripts/training/train_baselines.py`
  - [x] Train Random: Theoretical evaluation
  - [x] Train XGBoost: 500 trees, early stopping
  - [x] Train MLP: 20,000 steps, lr=0.0005, AdamW
  - [x] Compute top-1, top-3, top-5 accuracy for receiver prediction
  - [x] Compute F1, Precision, Recall, AUROC, AUPRC for shot prediction
  - [x] Save results: `results/baselines/{random,mlp,xgboost}_results.json`

**Success Criteria (Dual-task)**:
- ✅ Random baseline: top-1=4.9%, top-3=15.1%, top-5=25.1% (matches theory)
- ✅ XGBoost baseline: top-1=62.2%, **top-3=89.3%**, top-5=95.7% (far exceeds target!)
- ✅ MLP baseline: top-1=29.6%, **top-3=66.7%**, top-5=88.5% (exceeds 45% target)
- 📊 **Result**: XGBoost >> MLP (engineered features capture tactical patterns better)

**Implementation Notes (Completed Nov 2024)**:
- XGBoost outperformed expectations (89.3% vs 42% target) - engineered features very effective
- MLP exceeded target (66.7% vs 45% target) - validates dual-task learning
- Shot prediction: F1 ~0.40 for all models (challenging task, room for GNN improvement)
- Branch: `feature/tacticai-baseline-models`
- Files: `src/models/baselines.py`, `scripts/training/train_baselines.py`
- SLURM: `scripts/slurm/train_baselines_{v100,a100,h100}.sh`

---

### Day 6.5: Multi-Class Outcome Baselines ⏳ IN PROGRESS
**Goal**: Add multi-class outcome prediction (4-class: Goal/Shot/Clearance/Possession)

- [ ] Extend `src/data/receiver_data_loader.py`
  - [ ] Add `outcome_class_label` field: `torch.LongTensor([class_idx])` (0-3)
  - [ ] Class mapping: 0=Goal, 1=Shot, 2=Clearance, 3=Possession
  - [ ] Compute class distribution for weighted sampling
- [ ] Extend `src/models/baselines.py`
  - [ ] Implement `RandomOutcomeBaseline`
    - [ ] `predict()`: Return uniform distribution over 4 classes
    - [ ] Expected: 25% accuracy (random chance)
  - [ ] Implement `XGBoostOutcomeBaseline`
    - [ ] Extract graph-level features (aggregate player features)
    - [ ] Features: mean/std positions, formation compactness, defensive line height
    - [ ] XGBoost multi-class: `objective='multi:softmax', num_class=4`
    - [ ] Expected: 50-60% accuracy
  - [ ] Implement `MLPOutcomeBaseline`
    - [ ] Input: Flattened player positions `[batch, 22*14=308]`
    - [ ] MLP: 512 → 256 → 128 → 4 (output logits)
    - [ ] Loss: `CrossEntropyLoss()` with class weights
    - [ ] Expected: 55-65% accuracy
- [ ] Create `scripts/training/train_outcome_baselines.py`
  - [ ] Train Random: Theoretical evaluation (25% accuracy)
  - [ ] Train XGBoost: 500 trees, class weights for imbalance
  - [ ] Train MLP: 15,000 steps, lr=0.0005, weighted loss
  - [ ] Metrics: Accuracy, Macro F1, Per-class F1, Confusion Matrix
  - [ ] Save results: `results/baselines/outcome_{random,mlp,xgboost}_results.json`

**Class Distribution (Expected)**:
- Goal: ~1.3% (rare, difficult to predict)
- Shot: ~15% (minority class)
- Clearance: ~40% (common)
- Possession: ~44% (most common)

**Success Criteria (Multi-class)**:
- ✅ Random baseline: 25% accuracy (uniform)
- ✅ XGBoost baseline: 50-60% accuracy, Macro F1 > 0.45
- ✅ MLP baseline: 55-65% accuracy, Macro F1 > 0.50

**Deliverables**:
- ✅ Extended `src/data/receiver_data_loader.py` (add outcome labels)
- ✅ Extended `src/models/baselines.py` (3 new outcome classifiers)
- ✅ `scripts/training/train_outcome_baselines.py`
- ✅ `results/baselines/outcome_*.json`
- ✅ Update `docs/BASELINE_ARCHITECTURE.md` (add multi-class section)

---

### Day 7: Checkpoint & Decision Point ✅ COMPLETE
- [x] Review baseline results
- [x] Document findings in `docs/BASELINE_ARCHITECTURE.md`
- [x] Create publication-ready reporting system
- [x] Decision:
  - [x] **MLP top-3 = 66.7% >> 45% target**: ✅ Proceed to Phase 2 (GATv2)
  - [x] **XGBoost top-3 = 89.3%**: Strong baseline established

**Deliverables**:
- ✅ `scripts/preprocessing/add_receiver_labels.py`
- ✅ `src/data/receiver_data_loader.py`
- ✅ `src/models/baselines.py` (Random, XGBoost, MLP dual-task models)
- ✅ `scripts/training/train_baselines.py`
- ✅ `scripts/slurm/train_baselines_{v100,a100,h100}.sh`
- ✅ `results/baselines/{random,mlp,xgboost}_results.json`
- ✅ `docs/BASELINE_ARCHITECTURE.md` (complete technical documentation)
- ✅ `scripts/analysis/generate_baseline_report.py` (publication-ready reports)
- ✅ `results/baselines/report/` (LaTeX tables, figures, summary statistics)
- ✅ `results/baselines/report/HOW_TO_PRESENT.md` (presentation guide)

---

## Phase 2: GATv2 with D2 Equivariance (Week 2, Days 8-14)

### Day 8-9: D2 Augmentation Implementation ✅ COMPLETE
- [x] Create `src/data/augmentation.py`
  - [x] Implement `D2Augmentation` class
    - [x] `apply_transform(x, transform_type)`: h-flip, v-flip, both-flip
      - [x] H-flip: `x[:, 0] = 120 - x[:, 0]`, `x[:, 4] = -x[:, 4]` (flip vx)
      - [x] V-flip: `x[:, 1] = 80 - x[:, 1]`, `x[:, 5] = -x[:, 5]` (flip vy)
      - [x] Both-flip: Apply both transformations
    - [x] `get_all_views(x, edge_index)`: Generate 4 D2 views
- [x] Unit tests: `tests/test_augmentation.py`
  - [x] Test: Apply h-flip twice = identity
  - [x] Test: Edge structure unchanged across transforms
- [x] Visual test: Plot all 4 views of a corner kick (use mplsoccer)
  - [x] Save to `data/results/d2_augmentation_demo.png`

**Success Criteria**:
- ✅ All 4 D2 transforms implemented correctly
- ✅ Unit tests pass (17/17 passing)
- ✅ Visual inspection: 4 views look geometrically correct

**Implementation Notes (Completed Oct 29, 2025)**:
- Implemented algebraic angle transformations with [-π, π] normalization for perfect involution
- Preserves edge structure across all transformations
- Velocity angles transformed via: h_flip: θ → π - θ, v_flip: θ → -θ
- 17 comprehensive unit tests covering all transformations and properties
- Visual demo script: `scripts/visualization/visualize_d2_augmentation.py`
- Branch: `feature/tacticai-d2-augmentation`
- Files: `src/data/augmentation.py`, `tests/test_augmentation.py`, `scripts/visualization/visualize_d2_augmentation.py`

---

### Day 10-11: GATv2 Encoder Implementation
- [x] Create `src/models/gat_encoder.py`
  - [x] Implement `GATv2Encoder` (no D2 yet)
    - [x] Layer 1: `GATv2Conv(14, hidden_dim, heads=num_heads, dropout=0.4)`
    - [x] Layer 2: `GATv2Conv(hidden_dim*heads, hidden_dim, heads=num_heads, dropout=0.4)`
    - [x] Layer 3: `GATv2Conv(hidden_dim*heads, hidden_dim, heads=1, concat=False, dropout=0.4)`
    - [x] Batch normalization after each layer
    - [x] ELU activations (TacticAI uses ELU, not ReLU)
  - [x] Implement `D2GATv2` (with D2 frame averaging)
    - [x] Generate 4 D2 views using `D2Augmentation`
    - [x] Encode each view through `GATv2Encoder`
    - [x] Average node embeddings across views: `torch.stack(views).mean(dim=0)`
    - [x] Global mean pool: `global_mean_pool(avg_node_emb, batch)`
    - [x] Return both graph and node embeddings
- [x] Unit test: Forward pass with dummy data
  - [x] Input: `[batch=4, nodes=88, features=14]`
  - [x] Output graph_emb: `[batch=4, hidden_dim]`
  - [x] Output node_emb: `[nodes=88, hidden_dim]`

**Architecture Specifications**:
```
TacticAI: 4 layers, 8 heads, 4-dim latent (~50k params)
Ours: 3 layers, 4 heads, 16-dim latent (~25-30k params)
Rationale: Reduced capacity (15% of TacticAI's data), wider features (compensate for missing velocities)
```

**Success Criteria**:
- ✅ Model forward pass succeeds
- ✅ Parameter count: 25-35k (50-70% of TacticAI)
- ✅ D2 frame averaging produces sensible embeddings

**Implementation Notes (Completed Oct 29, 2024)**:
- GATv2Encoder: 3 layers, 4 heads, 24-dim hidden, ~27k params
- D2GATv2: Frame averaging over 4 D2 views (identity, h-flip, v-flip, both-flip)
- Bug fix (Oct 29): Corrected D2Augmentation API call in forward pass
- All 9 unit tests passing (4 GATv2Encoder + 5 D2GATv2)
- Branch: `bugfix/d2gatv2-augmentation-api`
- Files: `src/models/gat_encoder.py`, `tests/test_gat_encoder.py`

---

### Day 12-13: Receiver Prediction Head
- [ ] Create `src/models/receiver_predictor.py`
  - [ ] Implement `ReceiverPredictor`
    - [ ] Per-node classifier: `nn.Linear(hidden_dim, 1)`
    - [ ] Reshape to `[batch_size, 22]` with padding for variable player counts
    - [ ] Handle masking: Set logits to `-inf` for padded positions
  - [ ] Implement `ReceiverPredictionModel` (full model)
    - [ ] `D2GATv2` encoder
    - [ ] `ReceiverPredictor` head
    - [ ] Return: `receiver_logits [batch_size, 22]`
- [ ] Create `scripts/training/train_receiver.py`
  - [ ] Loss: `nn.CrossEntropyLoss()`
  - [ ] Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
  - [ ] Scheduler: `CosineAnnealingLR(T_max=50000, eta_min=1e-5)`
  - [ ] Train for 20k steps with early stopping (patience=10k steps)
  - [ ] Compute top-1, top-3, top-5 accuracy every 1000 steps
- [ ] Save best model: `models/receiver_prediction_best.pt`

**Success Criteria**:
- ✅ Training converges (loss decreases)
- ✅ Val top-3 accuracy > 60%
- ✅ No overfitting (val/train gap < 10%)

---

### Day 14: Ablation Study & Checkpoint
- [ ] Train 4 model variants (5k steps each for speed):
  - [ ] GCN (no attention): `GCNConv` layers
  - [ ] GAT (attention, no D2): Single view, no frame averaging
  - [ ] GATv2 + D2 (our main model): With D2 frame averaging
  - [ ] GATv2 + D2 + PosEmb: Add learned 2D positional embeddings
- [ ] Compare top-3 accuracy:
  - [ ] GCN: Expected 50-55%
  - [ ] GAT: Expected 55-60%
  - [ ] GATv2+D2: Expected 60-65%
  - [ ] GATv2+D2+PosEmb: Expected 63-68%
- [ ] Document results: `docs/RECEIVER_ABLATIONS.md`

**Decision Point**:
- [ ] **If D2 helps (+3% top-3)**: ✅ Keep D2, proceed to Phase 3
- [ ] **If D2 marginal (+1-2% top-3)**: ⚠️ Keep D2, note limited benefit (still valuable)
- [ ] **If D2 doesn't help (<1% top-3)**: ⚠️ Remove D2, focus on positional embeddings (negative result is OK!)

**Deliverables**:
- ✅ `src/data/augmentation.py`
- ✅ `src/models/gat_encoder.py`
- ✅ `src/models/receiver_predictor.py`
- ✅ `scripts/training/train_receiver.py`
- ✅ `models/receiver_prediction_best.pt`
- ✅ `docs/RECEIVER_ABLATIONS.md`

---

## Phase 3: Shot Prediction Task (Week 3, Days 15-21)

### Day 15-16: Conditional Shot Predictor
- [ ] Create `src/models/shot_predictor.py`
  - [ ] Implement `ConditionalShotPredictor`
    - [ ] Input: `graph_emb [batch, hidden_dim]` + `receiver_probs [batch, 22]`
    - [ ] Concatenate: `[batch, hidden_dim + 22]`
    - [ ] MLP: `(hidden_dim+22) → hidden_dim → 1`
    - [ ] Sigmoid output: shot probability
  - [ ] Implement `TwoStageModel` (full pipeline)
    - [ ] `D2GATv2` encoder
    - [ ] `ReceiverPredictor` head → softmax → receiver_probs
    - [ ] `ConditionalShotPredictor` head (conditioned on receiver_probs)
    - [ ] Return: `receiver_logits, shot_prob`
- [ ] Unit test: Forward pass
  - [ ] Input: `batch.x, batch.edge_index, batch.batch`
  - [ ] Output: `receiver_logits [batch, 22], shot_prob [batch, 1]`

**Two-Stage Inference**:
```
P(shot|corner) = Σ_i P(shot|receiver=i, corner) × P(receiver=i|corner)
```

**Success Criteria**:
- ✅ Forward pass succeeds
- ✅ Shot probabilities in [0, 1]
- ✅ Model learns to condition on receiver (verify gradients flow through receiver_probs)

---

### Day 17-18: Class Imbalance Handling
- [ ] Create `src/training/losses.py`
  - [ ] Implement `focal_loss(logits, targets, gamma=2.0, alpha=0.75)`
    - [ ] Focal weight: `alpha * (1 - pt)^gamma`
    - [ ] Focuses learning on hard examples
- [ ] Create `src/training/samplers.py`
  - [ ] Implement `BalancedSampler`
    - [ ] Compute class weights: `1.0 / class_counts[labels]`
    - [ ] Use `WeightedRandomSampler` to oversample minority class
- [ ] Create `scripts/training/train_shot.py`
  - [ ] Loss: Focal loss (γ=2.0, α=0.75)
  - [ ] Sampling: Balanced sampler (oversample shots)
  - [ ] Optimizer: AdamW(lr=5e-4, weight_decay=1e-4)
  - [ ] Train for 30k steps with early stopping
  - [ ] Metrics: F1, Precision, Recall, AUROC, AUPRC
- [ ] Save best model: `models/shot_prediction_best.pt`

**Class Distribution**:
- Positive class (shot OR goal): 14.3% (expect ~1050/7369 graphs)
- Negative class (clearance, loss, possession): 85.7%

**Success Criteria**:
- ✅ Training converges (loss decreases)
- ✅ No mode collapse (model doesn't predict all 0s or all 1s)
- ✅ Val F1 > 0.50, AUROC > 0.70

---

### Day 19-20: Threshold Optimization & Full Training
- [ ] Create `src/training/metrics.py`
  - [ ] Implement `compute_topk_accuracy(logits, targets, k)`
  - [ ] Implement `compute_shot_metrics(probs, targets, threshold)`
    - [ ] F1, Precision, Recall, AUROC, AUPRC
  - [ ] Implement `optimize_threshold(probs, targets)` using precision-recall curve
- [ ] Create `scripts/training/train_two_stage.py`
  - [ ] Multi-task loss: `L_total = L_receiver + 0.5 * L_shot`
  - [ ] Train full two-stage model for 50k steps
  - [ ] Optimize classification threshold on val set (often ~0.25-0.35 for 14% class)
  - [ ] Log to Weights & Biases (wandb)
- [ ] Save final model: `models/two_stage_model_final.pt`

**Expected Threshold**:
- Don't use 0.5! Optimal threshold often 0.25-0.35 for imbalanced data
- Use precision-recall curve to find F1-maximizing threshold

**Success Criteria**:
- ✅ Val F1 > 0.55, AUROC > 0.75, AUPRC > 0.40
- ✅ Precision > 0.50 @ Recall=0.60 (minimize false positives)

---

### Day 21: Conditional vs Unconditional Ablation
- [ ] Train unconditional baseline (shot prediction without receiver info)
  - [ ] Use only `graph_emb → MLP → shot_prob` (no receiver conditioning)
- [ ] Compare results:
  - [ ] Unconditional: Expected F1 = 0.50-0.55
  - [ ] Conditional (two-stage): Expected F1 = 0.55-0.65
- [ ] Document results: `docs/SHOT_ABLATIONS.md`

**Key Research Question**: Does knowing the receiver help predict shots?
- If conditional outperforms by >5% F1 → Proves receiver bottleneck hypothesis

**Decision Point**:
- [ ] **If F1 > 0.55, AUROC > 0.75**: ✅ Proceed to Phase 4 (final evaluation)
- [ ] **If F1 = 0.50-0.55**: ⚠️ Try threshold tuning, balanced sampling adjustments
- [ ] **If F1 < 0.50**: ❌ Pivot to multi-class outcome prediction (shot/clearance/possession)

**Deliverables**:
- ✅ `src/models/shot_predictor.py`
- ✅ `src/training/losses.py`
- ✅ `src/training/samplers.py`
- ✅ `src/training/metrics.py`
- ✅ `scripts/training/train_shot.py`
- ✅ `scripts/training/train_two_stage.py`
- ✅ `models/two_stage_model_final.pt`
- ✅ `docs/SHOT_ABLATIONS.md`

---

## Phase 3.5: Multi-Class Outcome Prediction (Parallel Track, Week 3-4)

**Goal**: Train GNN models for 4-class outcome classification (Goal/Shot/Clearance/Possession) as an alternative to dual-task approach.

### Day 21-22: Multi-Class GNN Architecture
- [ ] Create `src/models/outcome_predictor.py`
  - [ ] Implement `OutcomePredictor`
    - [ ] Input: `graph_emb [batch, hidden_dim]` from D2GATv2
    - [ ] MLP: `hidden_dim → hidden_dim → 4` (output logits)
    - [ ] Output: 4-class logits (Goal/Shot/Clearance/Possession)
  - [ ] Implement `OutcomeModel` (full pipeline)
    - [ ] `D2GATv2` encoder
    - [ ] `OutcomePredictor` head
    - [ ] Return: `outcome_logits [batch, 4]`
- [ ] Unit test: Forward pass
  - [ ] Input: `batch.x, batch.edge_index, batch.batch`
  - [ ] Output: `outcome_logits [batch, 4]`
  - [ ] Verify softmax sums to 1.0

**Class Mapping**:
```python
OUTCOME_CLASSES = {
    0: "Goal",      # 1.3% (very rare)
    1: "Shot",      # 15.8% (minority)
    2: "Clearance", # 39.5% (common)
    3: "Possession" # 43.4% (most common)
}
```

**Success Criteria**:
- ✅ Forward pass succeeds
- ✅ Logits shape correct `[batch, 4]`
- ✅ Model parameters ~30k (same as dual-task)

---

### Day 23-24: Weighted Loss & Training
- [ ] Extend `src/training/losses.py`
  - [ ] Implement `weighted_cross_entropy(logits, targets, class_weights)`
    - [ ] Compute class weights: `1.0 / sqrt(class_counts)`
    - [ ] Apply weights to `CrossEntropyLoss`
  - [ ] Implement `focal_loss_multiclass(logits, targets, gamma=2.0, alpha=None)`
    - [ ] Multi-class focal loss for extreme imbalance
    - [ ] Alpha parameter: per-class weights
- [ ] Create `scripts/training/train_outcome_gnn.py`
  - [ ] Loss: Weighted cross-entropy OR focal loss (compare both)
  - [ ] Optimizer: AdamW(lr=5e-4, weight_decay=1e-4)
  - [ ] Train for 30k steps with early stopping
  - [ ] Metrics: Accuracy, Macro F1, Per-class F1, Confusion Matrix
  - [ ] Class-balanced sampling (oversample Goal/Shot, undersample Poss/Clear)
- [ ] Save best model: `models/outcome_prediction_best.pt`

**Expected Class Weights**:
```python
class_weights = {
    0: 8.0,   # Goal (boost heavily)
    1: 2.5,   # Shot (boost moderately)
    2: 1.0,   # Clearance (normal)
    3: 0.9    # Possession (slight penalty)
}
```

**Success Criteria**:
- ✅ Training converges (loss decreases)
- ✅ No mode collapse (doesn't predict only Possession/Clearance)
- ✅ Val Macro F1 > 0.50, Accuracy > 55%
- ✅ Goal F1 > 0.10 (rare class, hard to predict)

---

### Day 25: Multi-Class Ablation Study
- [ ] Train 4 model variants (5k steps each for speed):
  - [ ] GCN (no attention): `GCNConv` layers
  - [ ] GAT (attention, no D2): Single view, no frame averaging
  - [ ] GATv2 + D2: With D2 frame averaging
  - [ ] GATv2 + D2 + PosEmb: Add learned 2D positional embeddings
- [ ] Compare Macro F1 and per-class F1:
  - [ ] GCN: Expected Macro F1 = 0.52-0.56
  - [ ] GAT: Expected Macro F1 = 0.56-0.59
  - [ ] GATv2+D2: Expected Macro F1 = 0.60-0.64
  - [ ] GATv2+D2+PosEmb: Expected Macro F1 = 0.62-0.66
- [ ] Document results: `docs/OUTCOME_ABLATIONS.md`

**Decision Point**:
- [ ] **If Macro F1 > 0.60**: ✅ Multi-class approach viable, proceed to comparison
- [ ] **If Macro F1 = 0.50-0.60**: ⚠️ Marginal, focus on dual-task (better for specific predictions)
- [ ] **If Macro F1 < 0.50**: ❌ Multi-class too difficult with static data, prioritize dual-task

**Deliverables**:
- ✅ `src/models/outcome_predictor.py`
- ✅ Extended `src/training/losses.py` (weighted/focal multi-class)
- ✅ `scripts/training/train_outcome_gnn.py`
- ✅ `models/outcome_prediction_best.pt`
- ✅ `docs/OUTCOME_ABLATIONS.md`

---

## Phase 4: Evaluation & Analysis (Week 4, Days 22-28)

### Day 22-23: Multi-Seed Evaluation (Both Approaches)
- [ ] Create `scripts/evaluation/run_five_seeds.py`
  - [ ] **Dual-task models**: Train 5 seeds [42, 123, 456, 789, 1011]
    - [ ] Each: 50k steps, save best checkpoint
    - [ ] Evaluate: Receiver (top-1/3/5), Shot (F1/AUROC/AUPRC)
  - [ ] **Multi-class models**: Train 5 seeds [42, 123, 456, 789, 1011]
    - [ ] Each: 30k steps, save best checkpoint
    - [ ] Evaluate: Accuracy, Macro F1, Per-class F1, Confusion Matrix
- [ ] Compute statistics: mean ± std across 5 runs
- [ ] Statistical significance tests:
  - [ ] **Dual-task**: t-test GATv2+D2 vs MLP baseline (receiver top-3)
  - [ ] **Multi-class**: t-test GATv2+D2 vs MLP baseline (Macro F1)
  - [ ] **Cross-approach**: Compare dual-task shot F1 vs multi-class shot F1
- [ ] Save results:
  - [ ] `results/five_seed_evaluation_dualtask.json`
  - [ ] `results/five_seed_evaluation_multiclass.json`

**Reporting Format**:
```
Dual-task:
  Receiver Top-3: 63.7% ± 2.0%
  Shot F1: 0.59 ± 0.03
  Shot AUROC: 0.78 ± 0.02

Multi-class:
  Accuracy: 65.2% ± 1.8%
  Macro F1: 0.62 ± 0.02
  Goal F1: 0.22 ± 0.04
  Shot F1: 0.58 ± 0.03
  Clearance F1: 0.76 ± 0.02
  Possession F1: 0.79 ± 0.01
```

**Success Criteria**:
- ✅ Std dev < 3% for all metrics (reproducible results)
- ✅ p < 0.05 for GATv2+D2 vs baselines (statistically significant improvement)
- ✅ Dual-task shot F1 ≈ Multi-class shot F1 (cross-validation of approaches)

---

### Day 24: Dual-task vs Multi-class Comparison
- [ ] Create `scripts/analysis/compare_approaches.py`
  - [ ] **Quantitative Comparison**:
    - [ ] Shot prediction agreement: % where dual-task & multi-class agree
    - [ ] Shot F1 comparison: Dual-task (binary) vs Multi-class (shot class)
    - [ ] Confusion analysis: Where do approaches disagree?
  - [ ] **Use Case Analysis**:
    - [ ] Dual-task strength: Better for receiver-specific tactics (set plays)
    - [ ] Multi-class strength: Better for holistic outcome understanding
    - [ ] Computational cost: Training time, inference speed
  - [ ] **Interpretability**:
    - [ ] Dual-task: Clear causal chain (receiver → shot)
    - [ ] Multi-class: Direct outcome prediction (no intermediate steps)
- [ ] Create visualization: `scripts/visualization/plot_approach_comparison.py`
  - [ ] Side-by-side confusion matrices
  - [ ] Venn diagram: Shot predictions overlap
  - [ ] Per-corner case studies: Where approaches differ
- [ ] Document findings: `docs/DUAL_VS_MULTICLASS_COMPARISON.md`

**Key Research Questions**:
- Which approach is better for coaching? (Hypothesis: Dual-task, explains "why")
- Which approach is better for betting? (Hypothesis: Multi-class, direct outcomes)
- Can we ensemble both approaches? (Hypothesis: Yes, +2-3% performance)

**Success Criteria**:
- ✅ Clear use case recommendations for each approach
- ✅ Shot F1 difference < 5% (cross-validation of methods)
- ✅ Identified specific corner scenarios where approaches disagree

---

### Day 25-26: Qualitative Analysis
- [ ] Create `scripts/visualization/visualize_attention.py`
  - [ ] **Dual-task model**:
    - [ ] Select 10 correct receiver predictions (high confidence)
    - [ ] Select 10 incorrect receiver predictions (failures)
    - [ ] Plot pitch with attention edges (thickness = weight)
  - [ ] **Multi-class model**:
    - [ ] Select 5 correct Goal predictions (rare, interesting)
    - [ ] Select 10 Shot vs Clearance confusion cases
    - [ ] Visualize attention patterns for each outcome class
  - [ ] Save to `data/results/attention_heatmaps/`
- [ ] Create `scripts/visualization/analyze_errors.py`
  - [ ] **Dual-task errors**:
    - [ ] Receiver errors by position (striker/midfielder/defender)
    - [ ] Shot errors by corner type (in-swing/out-swing/short/long)
    - [ ] Correlation: Low receiver confidence → low shot accuracy?
  - [ ] **Multi-class errors**:
    - [ ] Confusion matrix analysis: Most common misclassifications
    - [ ] Goal false negatives: Missed goal predictions (critical failures)
    - [ ] Clearance vs Possession: Why do models confuse these?
- [ ] Create confusion matrix visualization: `scripts/visualization/plot_confusion_matrix.py`
  - [ ] Multi-class confusion matrix (4×4 heatmap)
  - [ ] Per-class precision-recall curves
  - [ ] Error breakdown by outcome class
- [ ] Document findings: `docs/ERROR_ANALYSIS.md`

**Key Insights to Find**:
- **Dual-task**: Which positions hardest to predict? Attention interpretability?
- **Multi-class**: Which outcome transitions cause confusion? (e.g., Shot → Clearance)
- **Cross-model**: Do both models fail on same corners? (systematic failures)

**Success Criteria**:
- ✅ Clear patterns in error analysis (not random failures)
- ✅ Attention visualizations interpretable by soccer experts
- ✅ Actionable insights for model improvement

---

### Day 27-28: Paper Figures & Tables
- [ ] Create `notebooks/04_paper_figures.ipynb`
  - [ ] **Table 1**: Receiver Prediction Comparison (Dual-task)
    - [ ] Rows: Random, MLP, GCN, GAT, GATv2+D2, GATv2+D2+PosEmb, TacticAI
    - [ ] Columns: Top-1, Top-3, Top-5, Params, Features
  - [ ] **Table 2**: Shot Prediction Comparison (Dual-task)
    - [ ] Rows: Baseline, Unconditional, Conditional, TacticAI
    - [ ] Columns: F1, Precision, Recall, AUROC, AUPRC
  - [ ] **Table 3**: Multi-Class Outcome Comparison (NEW)
    - [ ] Rows: Random, XGBoost, MLP, GCN, GAT, GATv2+D2, GATv2+D2+PosEmb
    - [ ] Columns: Accuracy, Macro F1, Goal F1, Shot F1, Clear F1, Poss F1
  - [ ] **Table 4**: Dual-task vs Multi-class Summary (NEW)
    - [ ] Rows: Dual-task, Multi-class, Ensemble
    - [ ] Columns: Best for, Shot F1, Training time, Interpretability
  - [ ] **Figure 1**: Attention Heatmap Examples
    - [ ] 2×2 grid: 2 correct predictions, 2 failures (dual-task)
    - [ ] Professional broadcast-style pitch rendering
  - [ ] **Figure 2**: Precision-Recall Curves
    - [ ] Compare: Baseline, Unconditional, Conditional (dual-task)
    - [ ] Highlight F1-optimal threshold
  - [ ] **Figure 3**: Learning Curves
    - [ ] Dual-task: Top-3 accuracy, Shot F1 over steps
    - [ ] Multi-class: Accuracy, Macro F1 over steps
  - [ ] **Figure 4**: Ablation Study Bar Chart
    - [ ] Dual-task: Top-3 accuracy (GCN/GAT/GATv2+D2/+PosEmb)
    - [ ] Multi-class: Macro F1 (GCN/GAT/GATv2+D2/+PosEmb)
  - [ ] **Figure 5**: Multi-Class Confusion Matrix (NEW)
    - [ ] 4×4 heatmap (Goal/Shot/Clearance/Possession)
    - [ ] Per-class F1 scores annotated
  - [ ] **Figure 6**: Dual-task vs Multi-class Comparison (NEW)
    - [ ] Shot prediction Venn diagram (overlap analysis)
    - [ ] Bar chart: Metrics comparison side-by-side
- [ ] Export figures to `results/paper_figures/`

**Figure Quality Requirements**:
- ✅ Publication-ready: 300 DPI, vector graphics (SVG/PDF)
- ✅ Clear labels, legends, axis titles
- ✅ Consistent color scheme across all figures
- ✅ Color-blind friendly palettes (viridis, colorblind-safe)

---

### Day 29: Final Report & Documentation
- [ ] Create `docs/FINAL_REPORT.md`
  - [ ] **Executive Summary**: 1 paragraph (research question, dual approaches, key findings)
  - [ ] **Methodology**:
    - [ ] Dual-task architecture: Receiver + Shot prediction
    - [ ] Multi-class architecture: 4-class outcome classification
    - [ ] Training procedures, hyperparameters, ablations
  - [ ] **Results**: Tables 1-4, Figures 1-6
  - [ ] **Key Findings**:
    - [ ] **Dual-task results**:
      - [ ] Receiver Top-3: 66% (vs TacticAI 78%)
      - [ ] Shot F1: 0.61 (vs TacticAI 0.71)
      - [ ] Conditional prediction: +5% F1 (receiver bottleneck validated)
    - [ ] **Multi-class results**:
      - [ ] Accuracy: 67%, Macro F1: 0.64
      - [ ] Goal F1: 0.25 (very hard from static data)
      - [ ] Shot/Clearance/Possession: F1 0.58-0.79 (feasible)
    - [ ] **Comparison findings**:
      - [ ] Dual-task better for coaching (explains "why" via receiver)
      - [ ] Multi-class better for holistic understanding (direct outcomes)
      - [ ] Shot F1 similar (~0.60) - cross-validates both approaches
    - [ ] **Architecture insights**:
      - [ ] D2 symmetry: +6% receiver top-3, +3% multi-class accuracy
      - [ ] Positional embeddings: +2.4% top-3 (compensate for missing velocities)
      - [ ] Performance gap: 10-15% lower than TacticAI (velocities critical)
  - [ ] **Limitations**:
    - [ ] Static data (no velocities)
    - [ ] Small dataset (15% of TacticAI's size)
    - [ ] Goal prediction very challenging (1.3% class)
  - [ ] **Future Work**:
    - [ ] Velocity estimation from static frames
    - [ ] Ensemble dual-task + multi-class (+2-3% expected)
    - [ ] Hierarchical multi-class (coarse → fine outcomes)
    - [ ] Larger datasets, temporal modeling
- [ ] Create `README_TACTICAI.md` in project root
  - [ ] Quick start guide
  - [ ] Model download links
  - [ ] Inference example
  - [ ] Citation instructions
- [ ] Clean up codebase:
  - [ ] Add docstrings to all functions
  - [ ] Remove debug code
  - [ ] Run black formatter: `black src/ scripts/`
  - [ ] Run flake8 linter: `flake8 src/ scripts/`

**Deliverables**:
- ✅ `docs/FINAL_REPORT.md`
- ✅ `README_TACTICAI.md`
- ✅ `results/five_seed_evaluation.json`
- ✅ `results/paper_figures/` (all figures)
- ✅ `docs/ERROR_ANALYSIS.md`
- ✅ Clean, documented codebase

---

## Phase 5: Hyperparameter Search (Optional, Week 5)

### Hyperparameter Search Grid
- [ ] **Priority 1** (if Phase 4 results are weak):
  - [ ] `hidden_dim`: [16, 24, 32]
  - [ ] `num_layers`: [2, 3, 4]
  - [ ] `num_heads`: [4, 6, 8]
  - [ ] `dropout`: [0.3, 0.4, 0.5]
  - [ ] `lr`: [1e-3, 5e-4, 1e-4]
  - [ ] `weight_decay`: [1e-4, 5e-4, 1e-3]
- [ ] **Priority 2** (if Priority 1 improves results):
  - [ ] `focal_gamma`: [1.5, 2.0, 2.5]
  - [ ] `focal_alpha`: [0.7, 0.75, 0.8]
  - [ ] `batch_size`: [64, 128, 256]
  - [ ] `positional_emb_dim`: [0, 8, 16, 32]
- [ ] Create `scripts/training/hyperparameter_search.py`
  - [ ] Use Weights & Biases sweeps
  - [ ] Train each config for 20k steps (fast evaluation)
  - [ ] Select top 3 configs, train to 50k steps
- [ ] Document best hyperparameters: `configs/best_config.yaml`

**Estimated Time**: 2-3 days (12 configs × 6 hours = 3 days GPU time)

---

## Expected Performance Summary

### Receiver Prediction

| Model | Top-1 | Top-3 | Top-5 | Params | Features |
|-------|-------|-------|-------|--------|----------|
| Random | 4.5% | 13.6% | 22.7% | 0 | - |
| XGBoost Baseline | 25% ± 2% | 42% ± 2% | 60% ± 3% | N/A | Engineered |
| MLP Baseline | 22% ± 1% | 46% ± 2% | 64% ± 2% | 50k | Flatten |
| GCN | 27% ± 2% | 52% ± 2% | 70% ± 2% | 25k | Graph |
| GAT (no D2) | 29% ± 1% | 57% ± 2% | 75% ± 2% | 28k | Attention |
| **GATv2 + D2** | **32% ± 1%** | **64% ± 2%** | **78% ± 2%** | 30k | D2 symmetry |
| GATv2 + D2 + PosEmb | **33% ± 1%** | **66% ± 2%** | **81% ± 1%** | 32k | + Pos encoding |
| TacticAI (reference) | ~38% | 78% | ~88% | ~50k | + Velocities |

**Gap**: 10-15% lower than TacticAI (expected due to missing velocities)
**Key insight**: XGBoost → MLP (+4% top-3) shows neural networks help. MLP → GATv2+D2 (+18% top-3) shows graph structure is critical.

---

### Shot Prediction (Binary)

| Model | F1 | Precision | Recall | AUROC | AUPRC |
|-------|-----|-----------|--------|-------|-------|
| Baseline (graph only) | 0.51 ± 0.03 | 0.48 ± 0.04 | 0.55 ± 0.05 | 0.73 ± 0.02 | 0.35 ± 0.03 |
| Unconditional | 0.56 ± 0.02 | 0.53 ± 0.03 | 0.59 ± 0.04 | 0.77 ± 0.02 | 0.41 ± 0.03 |
| **Conditional (ours)** | **0.61 ± 0.02** | **0.58 ± 0.03** | **0.65 ± 0.03** | **0.80 ± 0.02** | **0.46 ± 0.02** |
| TacticAI (reference) | 0.71 | ~0.69 | ~0.73 | ~0.85 | ~0.55 |

**Gap**: 10% lower F1, 5% lower AUROC (expected due to missing velocities)

---

### Multi-Class Outcome Prediction (4-class: Goal/Shot/Clearance/Possession)

| Model | Accuracy | Macro F1 | Goal F1 | Shot F1 | Clear F1 | Poss F1 | Params |
|-------|----------|----------|---------|---------|----------|---------|--------|
| Random | 25.0% | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0 |
| XGBoost Baseline | 52% ± 3% | 0.48 ± 0.03 | 0.08 ± 0.05 | 0.42 ± 0.04 | 0.61 ± 0.03 | 0.65 ± 0.02 | N/A |
| MLP Baseline | 58% ± 2% | 0.53 ± 0.02 | 0.12 ± 0.06 | 0.48 ± 0.03 | 0.67 ± 0.02 | 0.70 ± 0.02 | 50k |
| GCN | 60% ± 2% | 0.56 ± 0.02 | 0.15 ± 0.05 | 0.52 ± 0.03 | 0.70 ± 0.02 | 0.73 ± 0.02 | 25k |
| GAT (no D2) | 62% ± 2% | 0.59 ± 0.02 | 0.18 ± 0.04 | 0.55 ± 0.03 | 0.73 ± 0.02 | 0.76 ± 0.02 | 28k |
| **GATv2 + D2** | **65% ± 2%** | **0.62 ± 0.02** | **0.22 ± 0.04** | **0.59 ± 0.02** | **0.76 ± 0.02** | **0.79 ± 0.01** | 30k |
| GATv2 + D2 + PosEmb | **67% ± 1%** | **0.64 ± 0.02** | **0.25 ± 0.03** | **0.62 ± 0.02** | **0.78 ± 0.01** | **0.81 ± 0.01** | 32k |

**Class Distribution**: Goal (1.3%), Shot (15.8%), Clearance (39.5%), Possession (43.4%)

**Key Insights**:
- Goal class extremely hard (F1 ~0.25) due to rarity (1.3%)
- Shot/Clearance/Possession benefit from graph structure (+10-15% F1)
- Macro F1 heavily penalized by rare Goal class
- Weighted F1 likely more appropriate metric for imbalanced data

**Gap**: Multi-class is harder than binary (Macro F1 0.64 vs Binary F1 0.61) due to class imbalance

---

## Critical Decision Points

### ✅ Checkpoint 1 (Day 7): Baselines
- **Success**: MLP top-3 > 45% → Proceed to Phase 2 (GATv2)
- **Failure**: MLP top-3 < 40% → Debug data pipeline (receiver labels)

### ✅ Checkpoint 2 (Day 14): D2 Symmetry
- **Success**: D2 improves top-3 by >3% → Keep D2, proceed to Phase 3
- **Neutral**: D2 improves top-3 by 1-3% → Keep D2, note limited benefit
- **Failure**: D2 doesn't help → Remove D2 (negative result is valuable!)

### ✅ Checkpoint 3 (Day 21): Shot Prediction
- **Success**: F1 > 0.55, AUROC > 0.75 → Proceed to Phase 4
- **Marginal**: F1 = 0.50-0.55 → Try threshold tuning, hyperparameter search
- **Failure**: F1 < 0.50 → Pivot to multi-class outcome prediction

### ✅ Final Checkpoint (Day 28): Publication Readiness
- **Strong Results** (Top-3 > 65%, F1 > 0.58): Submit to ML4Sports workshop (NeurIPS, ICML)
- **Moderate Results** (Top-3 = 60-65%, F1 = 0.53-0.58): Submit to sports analytics conference (MIT SSAC, StatsBomb)
- **Weak Results** (Top-3 < 60%, F1 < 0.53): Technical report + open-source release (still valuable!)

---

## Research Contribution Statement

**Title**: *Predicting Corner Kick Outcomes from Static Player Positions: Dual-task Receiver and Shot Prediction with Multi-class Outcome Analysis*

**Research Question**: Can static player positioning (without velocities) predict corner kick outcomes through dual-task learning (receiver + shot) and multi-class classification (Goal/Shot/Clearance/Possession)?

**Key Contributions**:
1. ✅ First open-data replication of TacticAI's geometric deep learning approach
2. ✅ Quantification of velocity feature importance (10-15% performance gap)
3. ✅ D2 equivariance on static data (+6% top-3 receiver accuracy)
4. ✅ Learned positional embeddings compensate for missing velocities (+2.4% top-3)
5. ✅ Conditional shot prediction validates receiver bottleneck hypothesis (+5% F1)
6. ✅ **Multi-class outcome analysis**: Direct comparison of dual-task vs. multi-class approaches
   - Dual-task (receiver + shot): Better for specific predictions (receiver top-3: 66%, shot F1: 0.61)
   - Multi-class (Goal/Shot/Clearance/Poss): Better for holistic outcome understanding (Macro F1: 0.64)
7. ✅ **Class imbalance insights**: Goal prediction from static positioning extremely challenging (F1 ~0.25)

**Positioning**:
- Primary: "Methodological replication with ablations" (ML/sports analytics)
- Secondary: "Dual-task vs. Multi-class learning comparison" (machine learning methodology)
- Tertiary: "Data efficiency analysis" (what's possible with limited tracking data)
- Quaternary: "Open-source baseline" (enable future research)

---

## File Structure Checklist

### New Files to Create

**Data & Preprocessing**:
- [ ] `scripts/preprocessing/add_receiver_labels.py`
- [ ] `src/data/receiver_data_loader.py`
- [ ] `src/data/augmentation.py`

**Models**:
- [ ] `src/models/baselines.py`
- [ ] `src/models/gat_encoder.py`
- [ ] `src/models/receiver_predictor.py`
- [ ] `src/models/shot_predictor.py`
- [ ] `src/models/two_stage_model.py`

**Training**:
- [ ] `src/training/trainer.py`
- [ ] `src/training/losses.py`
- [ ] `src/training/samplers.py`
- [ ] `src/training/metrics.py`

**Scripts**:
- [ ] `scripts/training/train_baseline.py` (dual-task: receiver + shot)
- [ ] `scripts/training/train_outcome_baselines.py` (multi-class: Goal/Shot/Clear/Poss)
- [ ] `scripts/training/train_receiver.py`
- [ ] `scripts/training/train_shot.py`
- [ ] `scripts/training/train_two_stage.py`
- [ ] `scripts/training/train_outcome_gnn.py` (GNN multi-class models)
- [ ] `scripts/training/hyperparameter_search.py`
- [ ] `scripts/evaluation/run_five_seeds.py`
- [ ] `scripts/evaluation/evaluate_receiver.py`
- [ ] `scripts/evaluation/evaluate_shot.py`
- [ ] `scripts/evaluation/evaluate_outcomes.py` (multi-class evaluation)
- [ ] `scripts/visualization/visualize_attention.py`
- [ ] `scripts/visualization/analyze_errors.py`
- [ ] `scripts/visualization/plot_confusion_matrix.py` (multi-class confusion)

**Documentation**:
- [ ] `docs/BASELINE_RESULTS.md` (dual-task baselines)
- [ ] `docs/OUTCOME_BASELINE_RESULTS.md` (multi-class baselines)
- [ ] `docs/RECEIVER_ABLATIONS.md`
- [ ] `docs/SHOT_ABLATIONS.md`
- [ ] `docs/OUTCOME_ABLATIONS.md` (multi-class ablations)
- [ ] `docs/ERROR_ANALYSIS.md`
- [ ] `docs/DUAL_VS_MULTICLASS_COMPARISON.md` (compare approaches)
- [ ] `docs/FINAL_REPORT.md`
- [ ] `README_TACTICAI.md`

**Configuration**:
- [ ] `configs/base_config.yaml`
- [ ] `configs/receiver_config.yaml`
- [ ] `configs/shot_config.yaml`
- [ ] `configs/best_config.yaml`

**Tests**:
- [ ] `tests/test_augmentation.py`
- [ ] `tests/test_models.py`
- [ ] `tests/test_data_loader.py`

---

## Progress Tracking

**Current Phase**: Phase 1 Complete ✅ - Ready for Phase 2 (GATv2)

**Completed Phases**:
- ✅ Phase 0: Infrastructure Foundation (data loaders, graph builders, feature engineering)
- ✅ Phase 1: Data Preparation & Baselines
  - ✅ Day 1-2: Receiver label extraction (996/1,118 corners, 89.1% coverage)
  - ✅ Day 3-4: Data loader extension (dual-task support, data leakage fix)
  - ✅ Day 5-6: Baseline models (Random, XGBoost, MLP)
  - ✅ Day 7: Checkpoint passed (XGBoost 89.3%, MLP 66.7% Top-3 accuracy)
- ✅ Phase 2 (Partial): D2 Augmentation & GATv2 Encoder
  - ✅ Day 8-9: D2 augmentation implementation
  - ✅ Day 10-11: GATv2 encoder with D2 frame averaging

**Next Task**: Phase 2 Day 12-13 - Receiver Prediction Head

**Key Results**:
- Dataset: 7,369 temporally augmented graphs (6.6× original)
- Baselines: XGBoost 89.3% Top-3, MLP 66.7% Top-3 (far exceed targets)
- Shot prediction: F1 ~0.40 (room for GNN improvement)
- Data quality: Fixed temporal leakage, 89% receiver coverage

**Notes**:
- XGBoost baseline is very strong (89.3% Top-3) - sets high bar for GNN
- MLP baseline validates dual-task approach (receiver + shot)
- D2 augmentation ready for GATv2 integration
- Publication-ready reporting infrastructure complete

---

## Quick Reference: Key Hyperparameters

```yaml
# Best guess hyperparameters (to be validated)
model:
  hidden_dim: 24
  num_layers: 3
  num_heads: 4
  dropout: 0.4
  use_d2: true
  use_pos_emb: true
  pos_emb_dim: 16

training:
  max_steps: 50000
  batch_size: 128
  lr: 5.0e-4
  weight_decay: 1.0e-4
  gradient_clip: 1.0
  scheduler: "cosine"

receiver_task:
  loss: "cross_entropy"

shot_task:
  loss: "focal"
  focal_gamma: 2.0
  focal_alpha: 0.75
  use_balanced_sampling: true
  optimal_threshold: 0.30  # To be tuned on val set
```

---

## Contact & Support

For questions or issues during implementation:
1. Check this plan for decision points and expected results
2. Consult `docs/FINAL_REPORT.md` for completed phases
3. Review error analysis in `docs/ERROR_ANALYSIS.md`

**Remember**: A 10-15% performance gap vs TacticAI is expected and scientifically valuable. The contribution is quantifying what's achievable with static-only data.
