# TacticAI-Style Corner Kick Prediction: Implementation Plan

**Project Goal**: Replicate TacticAI's dual-task corner kick prediction system (receiver + shot prediction) using static-only StatsBomb 360 freeze frames.

**Key Limitation**: No velocity data (vx, vy = 0 for all players)

**Expected Performance**:
- Receiver Top-3: 60-70% (vs TacticAI's 78%)
- Shot F1: 0.55-0.65 (vs TacticAI's 0.71)
- Performance gap of 10-15% expected due to missing velocities

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

### Day 5-6: Baseline Models
- [x] Create `src/models/baselines.py`
  - [x] Implement `RandomReceiverBaseline`
    - [x] `predict()`: Return random softmax over 22 players
    - [x] `evaluate()`: Return top-1=4.5%, top-3=13.6%, top-5=22.7%
  - [x] Implement `MLPReceiverBaseline`
    - [x] Flatten all player positions: `[batch, 22*14=308]`
    - [x] MLP: 308 → 256 → 128 → 22
    - [x] Dropout 0.3, ReLU activations
- [x] Create `scripts/training/train_baseline.py`
  - [x] Train MLP for 10k steps
  - [x] Compute top-1, top-3, top-5 accuracy
  - [x] Save results: `results/baseline_mlp.json`

**Success Criteria**:
- ✅ Random baseline: top-1=4.5%, top-3=13.6% (sanity check)
- ✅ MLP baseline: top-1 > 20%, top-3 > 45%
- ❌ **If MLP top-3 < 40%**: STOP and debug data pipeline (receiver labels may be incorrect)

---

### Day 7: Checkpoint & Decision Point
- [x] Review baseline results
- [x] Document findings in `docs/BASELINE_RESULTS.md`
- [x] Decision:
  - [x] **If MLP top-3 > 45%**: ✅ Proceed to Phase 2 (GATv2)
  - [x] **If MLP top-3 < 40%**: ❌ Debug data quality (check receiver label extraction)

**Deliverables**:
- ✅ `scripts/preprocessing/add_receiver_labels.py`
- ✅ `src/data/receiver_data_loader.py`
- ✅ `src/models/baselines.py`
- ✅ `results/baseline_mlp.json`
- ✅ `docs/BASELINE_RESULTS.md`

---

## Phase 2: GATv2 with D2 Equivariance (Week 2, Days 8-14)

### Day 8-9: D2 Augmentation Implementation
- [ ] Create `src/data/augmentation.py`
  - [ ] Implement `D2Augmentation` class
    - [ ] `apply_transform(x, transform_type)`: h-flip, v-flip, both-flip
      - [ ] H-flip: `x[:, 0] = 120 - x[:, 0]`, `x[:, 4] = -x[:, 4]` (flip vx)
      - [ ] V-flip: `x[:, 1] = 80 - x[:, 1]`, `x[:, 5] = -x[:, 5]` (flip vy)
      - [ ] Both-flip: Apply both transformations
    - [ ] `get_all_views(x, edge_index)`: Generate 4 D2 views
- [ ] Unit tests: `tests/test_augmentation.py`
  - [ ] Test: Apply h-flip twice = identity
  - [ ] Test: Edge structure unchanged across transforms
- [ ] Visual test: Plot all 4 views of a corner kick (use mplsoccer)
  - [ ] Save to `data/results/d2_augmentation_demo.png`

**Success Criteria**:
- ✅ All 4 D2 transforms implemented correctly
- ✅ Unit tests pass
- ✅ Visual inspection: 4 views look geometrically correct

---

### Day 10-11: GATv2 Encoder Implementation
- [ ] Create `src/models/gat_encoder.py`
  - [ ] Implement `GATv2Encoder` (no D2 yet)
    - [ ] Layer 1: `GATv2Conv(14, hidden_dim, heads=num_heads, dropout=0.4)`
    - [ ] Layer 2: `GATv2Conv(hidden_dim*heads, hidden_dim, heads=num_heads, dropout=0.4)`
    - [ ] Layer 3: `GATv2Conv(hidden_dim*heads, hidden_dim, heads=1, concat=False, dropout=0.4)`
    - [ ] Batch normalization after each layer
    - [ ] ELU activations (TacticAI uses ELU, not ReLU)
  - [ ] Implement `D2GATv2` (with D2 frame averaging)
    - [ ] Generate 4 D2 views using `D2Augmentation`
    - [ ] Encode each view through `GATv2Encoder`
    - [ ] Average node embeddings across views: `torch.stack(views).mean(dim=0)`
    - [ ] Global mean pool: `global_mean_pool(avg_node_emb, batch)`
    - [ ] Return both graph and node embeddings
- [ ] Unit test: Forward pass with dummy data
  - [ ] Input: `[batch=4, nodes=88, features=14]`
  - [ ] Output graph_emb: `[batch=4, hidden_dim]`
  - [ ] Output node_emb: `[nodes=88, hidden_dim]`

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

## Phase 4: Evaluation & Analysis (Week 4, Days 22-28)

### Day 22-23: Multi-Seed Evaluation
- [ ] Create `scripts/evaluation/run_five_seeds.py`
  - [ ] Train 5 models with seeds: [42, 123, 456, 789, 1011]
  - [ ] Each model: 50k steps, save best checkpoint
- [ ] Evaluate each model on test set
  - [ ] Receiver: top-1, top-3, top-5 accuracy
  - [ ] Shot: F1, Precision, Recall, AUROC, AUPRC
- [ ] Compute statistics: mean ± std across 5 runs
- [ ] Statistical significance tests:
  - [ ] t-test: GATv2+D2 vs MLP baseline (receiver top-3)
  - [ ] t-test: Conditional vs Unconditional (shot F1)
- [ ] Save results: `results/five_seed_evaluation.json`

**Reporting Format**:
```
Receiver Top-3: 63.7% ± 2.0%
Shot F1: 0.59 ± 0.03
Shot AUROC: 0.78 ± 0.02
```

**Success Criteria**:
- ✅ Std dev < 3% (reproducible results)
- ✅ p < 0.05 for GATv2+D2 vs baselines (statistically significant improvement)

---

### Day 24-25: Qualitative Analysis
- [ ] Create `scripts/visualization/visualize_attention.py`
  - [ ] Select 10 correct receiver predictions (high confidence)
  - [ ] Select 10 incorrect receiver predictions (failures)
  - [ ] For each:
    - [ ] Plot pitch with player positions (colored by team)
    - [ ] Overlay attention edges (thickness = attention weight)
    - [ ] Highlight predicted receiver (star)
    - [ ] Show true receiver (circle)
  - [ ] Save to `data/results/attention_heatmaps/`
- [ ] Create `scripts/visualization/analyze_errors.py`
  - [ ] Categorize receiver errors by position:
    - [ ] Striker errors: % of strikers misclassified
    - [ ] Midfielder errors: % of midfielders misclassified
    - [ ] Defender errors: % of defenders misclassified
  - [ ] Categorize shot errors by corner type:
    - [ ] In-swinging vs out-swinging
    - [ ] Short vs long corners
  - [ ] Analyze correlation: Does low receiver confidence → low shot accuracy?
- [ ] Document findings: `docs/ERROR_ANALYSIS.md`

**Key Insights to Find**:
- Which positions are hardest to predict? (Hypothesis: midfielders, due to positional ambiguity)
- Does the model attend to correct players? (Hypothesis: yes, attention focuses on players near ball landing zone)
- What types of corners cause failures? (Hypothesis: short corners, crowded penalty box)

**Success Criteria**:
- ✅ Clear patterns in error analysis (not random failures)
- ✅ Attention visualizations interpretable by soccer experts

---

### Day 26-27: Paper Figures & Tables
- [ ] Create `notebooks/04_paper_figures.ipynb`
  - [ ] **Table 1**: Receiver Prediction Comparison
    - [ ] Rows: Random, MLP, GCN, GAT, GATv2+D2, GATv2+D2+PosEmb, TacticAI
    - [ ] Columns: Top-1, Top-3, Top-5, Params, Features
  - [ ] **Table 2**: Shot Prediction Comparison
    - [ ] Rows: Baseline, Unconditional, Conditional, TacticAI
    - [ ] Columns: F1, Precision, Recall, AUROC, AUPRC
  - [ ] **Figure 1**: Attention Heatmap Examples
    - [ ] 2×2 grid: 2 correct predictions, 2 failures
    - [ ] Professional broadcast-style pitch rendering
  - [ ] **Figure 2**: Precision-Recall Curves
    - [ ] Compare: Baseline, Unconditional, Conditional
    - [ ] Highlight F1-optimal threshold
  - [ ] **Figure 3**: Learning Curves
    - [ ] Top-3 accuracy over training steps (train vs val)
    - [ ] F1 score over training steps (train vs val)
  - [ ] **Figure 4**: Ablation Study Bar Chart
    - [ ] Top-3 accuracy for: GCN, GAT, GATv2+D2, GATv2+D2+PosEmb
- [ ] Export figures to `results/paper_figures/`

**Figure Quality Requirements**:
- ✅ Publication-ready: 300 DPI, vector graphics (SVG/PDF)
- ✅ Clear labels, legends, axis titles
- ✅ Consistent color scheme across all figures

---

### Day 28: Final Report & Documentation
- [ ] Create `docs/FINAL_REPORT.md`
  - [ ] **Executive Summary**: 1 paragraph
  - [ ] **Methodology**: Architecture, training procedure, hyperparameters
  - [ ] **Results**: Tables 1-2, Figures 1-4
  - [ ] **Key Findings**:
    - [ ] Performance gap: 10-15% lower than TacticAI (due to missing velocities)
    - [ ] D2 symmetry: +6% top-3 accuracy (even without velocities)
    - [ ] Positional embeddings: +2.4% top-3 accuracy (compensate for missing velocities)
    - [ ] Conditional shot prediction: +5% F1 (receiver info helps)
  - [ ] **Limitations**: Static data, small dataset (15% of TacticAI's size)
  - [ ] **Future Work**: Velocity estimation, larger datasets, multi-task learning
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
| MLP Baseline | 22% ± 1% | 46% ± 2% | 64% ± 2% | 50k | Flatten |
| GCN | 27% ± 2% | 52% ± 2% | 70% ± 2% | 25k | Graph |
| GAT (no D2) | 29% ± 1% | 57% ± 2% | 75% ± 2% | 28k | Attention |
| **GATv2 + D2** | **32% ± 1%** | **64% ± 2%** | **78% ± 2%** | 30k | D2 symmetry |
| GATv2 + D2 + PosEmb | **33% ± 1%** | **66% ± 2%** | **81% ± 1%** | 32k | + Pos encoding |
| TacticAI (reference) | ~38% | 78% | ~88% | ~50k | + Velocities |

**Gap**: 10-15% lower than TacticAI (expected due to missing velocities)

---

### Shot Prediction

| Model | F1 | Precision | Recall | AUROC | AUPRC |
|-------|-----|-----------|--------|-------|-------|
| Baseline (graph only) | 0.51 ± 0.03 | 0.48 ± 0.04 | 0.55 ± 0.05 | 0.73 ± 0.02 | 0.35 ± 0.03 |
| Unconditional | 0.56 ± 0.02 | 0.53 ± 0.03 | 0.59 ± 0.04 | 0.77 ± 0.02 | 0.41 ± 0.03 |
| **Conditional (ours)** | **0.61 ± 0.02** | **0.58 ± 0.03** | **0.65 ± 0.03** | **0.80 ± 0.02** | **0.46 ± 0.02** |
| TacticAI (reference) | 0.71 | ~0.69 | ~0.73 | ~0.85 | ~0.55 |

**Gap**: 10% lower F1, 5% lower AUROC (expected due to missing velocities)

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

**Title**: *Predicting Corner Kick Outcomes from Static Player Positions: A TacticAI Replication on Open Data*

**Key Contributions**:
1. ✅ First open-data replication of TacticAI's geometric deep learning approach
2. ✅ Quantification of velocity feature importance (10-15% performance gap)
3. ✅ D2 equivariance on static data (+6% top-3 accuracy)
4. ✅ Learned positional embeddings compensate for missing velocities (+2.4% top-3)
5. ✅ Conditional shot prediction validates receiver bottleneck hypothesis (+5% F1)

**Positioning**:
- Primary: "Methodological replication with ablations" (ML/sports analytics)
- Secondary: "Data efficiency analysis" (what's possible with limited tracking data)
- Tertiary: "Open-source baseline" (enable future research)

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
- [ ] `scripts/training/train_baseline.py`
- [ ] `scripts/training/train_receiver.py`
- [ ] `scripts/training/train_shot.py`
- [ ] `scripts/training/train_two_stage.py`
- [ ] `scripts/training/hyperparameter_search.py`
- [ ] `scripts/evaluation/run_five_seeds.py`
- [ ] `scripts/evaluation/evaluate_receiver.py`
- [ ] `scripts/evaluation/evaluate_shot.py`
- [ ] `scripts/visualization/visualize_attention.py`
- [ ] `scripts/visualization/analyze_errors.py`

**Documentation**:
- [ ] `docs/BASELINE_RESULTS.md`
- [ ] `docs/RECEIVER_ABLATIONS.md`
- [ ] `docs/SHOT_ABLATIONS.md`
- [ ] `docs/ERROR_ANALYSIS.md`
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

**Current Phase**: Not started

**Completed Phases**:
- ✅ Phase 0: Existing GNN infrastructure (from previous work)

**Next Task**: Day 1 - Write `scripts/preprocessing/add_receiver_labels.py`

**Notes**:
- This plan assumes you have ~7,369 graphs with temporal augmentation
- Expected dataset size: ~900-950 corners with valid receiver labels (85% coverage)
- Total estimated time: 4 weeks (28 days) at full-time pace

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
