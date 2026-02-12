# USSF → Corner Kick Transfer Learning: Task Checklist

**Goal:** Test whether GNN representations learned from counterattack prediction (USSF, 20,863 graphs) transfer to corner kick shot prediction (DFL, 65 graphs).

**Reference:** Full experimental plan in `new-plan.md`

---

## Phase 0: Data Inspection

**Objective:** Understand USSF data distributions before writing training code.

- [ ] Download USSF `combined.pkl` from S3 bucket
- [ ] Inspect data structure (type, keys, nested structure)
- [ ] Document node feature distributions for all 12 features:
  - [ ] x, y (positions)
  - [ ] vx, vy (velocities)
  - [ ] velocity_mag, velocity_angle
  - [ ] dist_goal, angle_goal
  - [ ] dist_ball, angle_ball
  - [ ] attacking_team_flag
  - [ ] potential_receiver
- [ ] Document edge feature distributions for all 6 features:
  - [ ] player_distance
  - [ ] speed_difference
  - [ ] positional_sine_angle, positional_cosine_angle
  - [ ] velocity_sine_angle, velocity_cosine_angle
- [ ] Check adjacency matrix structure (`normal` vs `dense`)
- [ ] Analyze class balance (success rate)
- [ ] Document variable graph sizes (min/max/mean players per graph)
- [ ] Create `ussf_feature_distribution_report.md` documenting all findings

**Deliverable:** Feature distribution report with means, stds, ranges for all features.

---

## Phase 1: Train USSF Backbone

**Objective:** Reproduce USSF results and save trained CrystalConv weights.

### Environment Setup
- [ ] Set up environment with pinned versions (`spektral==1.3.1`, `tensorflow==2.12`)
- [ ] Verify GPU availability for training

### Architecture Implementation
- [ ] Implement CrystalConv architecture:
  - [ ] 3 conv layers
  - [ ] 128 hidden channels
  - [ ] Dense head: 128 → dropout(0.5) → 128 → dropout(0.5) → 1 (sigmoid)
- [ ] Implement data loading with sequence-aware 70/30 splitting (seed=15)

### Training
- [ ] Train backbone with `dense` adjacency type
  - [ ] Log training curves
  - [ ] Record final AUC
- [ ] Train backbone with `normal` adjacency type
  - [ ] Log training curves
  - [ ] Record final AUC

### Validation
- [ ] Reproduce USSF reported AUC (must match before proceeding)
- [ ] Document any discrepancies

### Save Weights
- [ ] Save conv layer weights for `dense` backbone (`ussf_backbone_dense.pkl`)
- [ ] Save conv layer weights for `normal` backbone (`ussf_backbone_normal.pkl`)
- [ ] Save full models for reference (`ussf_full_model_dense.h5`, `ussf_full_model_normal.h5`)

**Deliverable:** Two trained backbones with verified AUCs, saved weights.

---

## Phase 2: Engineer DFL Corner Features

**Objective:** Transform 57 DFL corner kick graphs to match USSF feature schema exactly.

**Status:** ✅ COMPLETE

### Coordinate System Alignment
- [x] Analyze USSF coordinate system (0-1 pitch-relative)
- [x] Document DFL coordinate system (0-1 normalized, goal at x=0)
- [x] Implement normalization to match USSF conventions (flip x-axis)
- [x] Verify goal position conventions match (goal at x=1 after flip)

### Velocity Computation
- [x] Implement velocity computation from 25fps tracking:
  - [x] vx, vy already computed in DFL data (m/frame)
  - [x] Convert to m/s by multiplying by 25
- [x] Transform to unit vectors + normalized magnitude (matching USSF)

### Derived Features
- [x] Compute velocity_mag = sqrt(vx² + vy²), normalized to [0,1]
- [x] Compute velocity_angle = (atan2(vy, vx) + π) / (2π)
- [x] Compute dist_goal (Euclidean to goal center, normalized)
- [x] Compute angle_goal = (atan2(goal_y - y, goal_x - x) + π) / (2π)
- [x] Compute dist_ball (Euclidean to ball position, normalized)
- [x] Compute angle_ball = (atan2(ball_y - y, ball_x - x) + π) / (2π)
- [x] Set attacking_team_flag using position heuristic (team labels unavailable)

### Handle `potential_receiver`
- [x] Decide on approach: **Option B - Set to 0 for all corner players**
  - Rationale: Keeps 12-feature architecture compatible with pretrained weights
  - Alternative (Option A) would require retraining backbone
- [x] Document decision and implement

### Build Graph Structure
- [x] Compute edge features for all player pairs (6 features)
- [x] Build `dense` adjacency matrices (fully connected)
- [x] Build `normal` adjacency matrices (kNN + team-based + ball connectivity)

### Distribution Alignment Verification
- [x] Run KS (Kolmogorov-Smirnov) tests comparing USSF vs DFL distributions:
  - [x] All 12 node features
  - [x] All 6 edge features
- [x] Flag features with p < 0.01 (significant distribution mismatch)
- [x] Create `dfl_ussf_distribution_comparison.md` with KS test results

### Results Summary
- **11/12 node features** show significant distribution mismatch (expected)
- **attacking_team_flag is NOT significantly different** (p=0.31)
- Key mismatches: velocity_mag (0.053 vs 0.219), potential_receiver (0.0 vs 0.487)

**Deliverables:**
- `transfer_learning/data/dfl_corners_ussf_format_dense.pkl` (57 corners)
- `transfer_learning/data/dfl_corners_ussf_format_normal.pkl` (57 corners)
- `transfer_learning/reports/dfl_ussf_distribution_comparison_dense.md`
- `transfer_learning/reports/dfl_ussf_distribution_comparison_normal.md`
- `transfer_learning/reports/ks_test_results_dense.pkl`
- `transfer_learning/reports/ks_test_results_normal.pkl`

---

## Phase 3: Transfer Learning Experiments

**Objective:** Run 6 experimental conditions to test transfer learning.

**Status:** ✅ COMPLETE

### Architecture Implementation
- [x] Implement `TransferGNN` class:
  - [x] Load pretrained conv layers
  - [x] Frozen backbone option
  - [x] New head: 32 → dropout(0.3) → 1 (sigmoid)
  - [x] Support for unfreezing conv layers with low lr

### Data Split
- [x] Create match-based train/val/test split for 57 corners:
  - [x] Keep all corners from same match together
  - [x] 5 matches train, 1 val, 1 test (varies by seed)
  - [x] Document match distribution

### Training Config
- [x] Epochs: 50 with early stopping (patience=10)
- [x] Batch size: 8
- [x] Learning rate: 1e-4 (frozen), 1e-5 (unfrozen)
- [x] Loss: BinaryCrossentropy with class weights

### Run Experimental Conditions

- [x] **Condition A:** USSF pretrained + dense + frozen
  - [x] Train model across 5 seeds
  - [x] Test AUC: 0.57 ± 0.24

- [x] **Condition B:** USSF pretrained + normal + frozen
  - [x] Train model across 5 seeds
  - [x] Test AUC: 0.51 ± 0.21

- [x] **Condition C:** USSF pretrained + dense + unfrozen (lr=1e-5)
  - [x] Train model across 5 seeds
  - [x] Test AUC: 0.56 ± 0.23

- [x] **Condition D:** Random init + dense
  - [x] Train model across 5 seeds
  - [x] Test AUC: 0.39 ± 0.29

- [x] **Condition E:** Random init + normal
  - [x] Train model across 5 seeds
  - [x] Test AUC: 0.43 ± 0.27

- [x] **Condition F:** Majority class baseline
  - [x] Test AUC: 0.50 ± 0.00

### Results Summary

| Condition | Description | Test AUC (mean±std) | Test Acc (mean±std) |
|-----------|-------------|---------------------|---------------------|
| A | USSF pretrained + dense + frozen | **0.57 ± 0.24** | 0.58 ± 0.23 |
| B | USSF pretrained + normal + frozen | 0.51 ± 0.21 | 0.53 ± 0.20 |
| C | USSF pretrained + dense + unfrozen | 0.56 ± 0.23 | 0.58 ± 0.23 |
| D | Random init + dense | 0.39 ± 0.29 | 0.56 ± 0.24 |
| E | Random init + normal | 0.43 ± 0.27 | 0.54 ± 0.27 |
| F | Majority baseline | 0.50 ± 0.00 | 0.72 ± 0.01 |

### Key Findings

1. **Marginal transfer benefit:** Pretrained models (A, B, C) slightly outperform random AUC baseline (0.50), with A achieving 0.57
2. **High variance:** Standard deviations ~0.24 indicate results are not statistically significant with 57 samples
3. **Random init fails:** Conditions D, E perform worse than baseline - severe overfitting
4. **Dense > normal:** Dense adjacency (A: 0.57) outperforms normal (B: 0.51)
5. **Frozen ≈ unfrozen:** Similar performance (A: 0.57 vs C: 0.56) suggests pretrained features are stable

**Interpretation:** Transfer from counterattack prediction to corner kick prediction shows marginal (non-significant) benefit. The expected failure mode is confirmed - 57 samples insufficient for reliable conclusions, but pretrained models at least don't hurt.

**Deliverables:**
- `transfer_learning/phase3_transfer_learning.py` - Experiment script
- `transfer_learning/results/phase3_multiseed_*.pkl` - Full results
- `transfer_learning/results/phase3_multiseed_summary_*.json` - Summary

---

## Phase 4: Velocity Ablation

**Objective:** Determine whether velocity features are critical for predictions.

**Status:** ✅ COMPLETE

### Implement Permutation Importance
- [x] Implement shuffled dataset class (following USSF's `ShuffledCounterDataset` methodology)
- [x] Select best transfer condition (A and C tested)

### Run Ablation Experiments
- [x] Baseline: Evaluate on test set → AUC_baseline
- [x] Velocity ablation: Shuffle vx, vy, velocity_mag, velocity_angle → AUC_no_velocity
- [x] Position ablation: Shuffle x, y → AUC_no_position
- [x] Position derived ablation: Shuffle dist_goal, angle_goal, dist_ball, angle_ball

### Results Summary

| Condition | Baseline AUC | Velocity Drop | Position (x,y) Drop | Position Derived Drop |
|-----------|--------------|---------------|---------------------|----------------------|
| A (frozen) | 0.64 ± 0.20 | 0.13 ± 0.21 | 0.05 ± 0.05 | 0.23 ± 0.15 |
| C (unfrozen) | 0.61 ± 0.13 | 0.10 ± 0.10 | 0.03 ± 0.17 | 0.05 ± 0.14 |

### Analysis
- [x] Compare AUC drops:
  - [x] Velocity features (vx, vy, mag, angle) cause moderate AUC drop (~0.10-0.13)
  - [x] Raw position features (x, y) cause minimal AUC drop (~0.03-0.05)
  - [x] Position derived features (dist/angle to goal/ball) cause largest drop in frozen model (0.23)
- [x] Document conclusions connecting to 7.5 ECTS findings

### Key Findings

1. **Velocity > Raw Position:** Velocity ablation causes larger AUC drops than raw position ablation in both conditions, supporting the hypothesis that velocity is more important than static position.

2. **High Variance:** Standard deviations are large relative to means (e.g., velocity drop 0.13 ± 0.21), indicating results are not statistically significant with 57 samples across 5 seeds.

3. **Spatial Relationships Matter:** In the frozen model (A), derived position features (dist_goal, angle_goal, dist_ball, angle_ball) show the largest importance, suggesting the model relies on spatial relationships rather than raw coordinates.

4. **Connection to 7.5 ECTS:** Results support the finding that position-only achieved AUC=0.50. Velocity features provide signal above random, but the effect is modest and noisy due to small sample size.

**Deliverables:**
- `transfer_learning/phase4_velocity_ablation.py` - Ablation experiment script
- `transfer_learning/results/phase4_ablation_*.pkl` - Full results
- `transfer_learning/results/phase4_ablation_summary_*.json` - Summary

---

## Phase 5: DFL Open-Play Comparison

**Objective:** Compare USSF pretraining vs DFL open-play pretraining.

### Train DFL Open-Play Backbone
- [ ] Prepare DFL open-play frames (use available sequences from 7 matches)
- [ ] Train CrystalConv architecture on DFL open-play data
- [ ] Save trained weights

### Fine-tune on Corners
- [ ] Fine-tune DFL pretrained model on 65 corners (same procedure as Phase 3)
- [ ] Record AUC with confidence interval

### Comparison
- [ ] Create final comparison table:
  - [ ] USSF pretrained (best condition)
  - [ ] DFL open-play pretrained
  - [ ] Random init baseline
  - [ ] Majority class baseline
- [ ] Document which pretraining source works better and why

**Deliverable:** Final comparison table with conclusions about transfer learning effectiveness.

---

## Final Documentation

- [ ] Compile all results into thesis section draft
- [ ] Update `CLAUDE.md` with final results summary
- [ ] Archive all trained models and reports

---

## Progress Summary

| Phase | Status | Notes |
|-------|--------|-------|
| 0. Data Inspection | ✅ Complete | USSF feature distributions documented |
| 1. Train USSF Backbone | ✅ Complete | Dense AUC=0.693, Normal AUC=0.683 |
| 2. Engineer DFL Features | ✅ Complete | 57 corners transformed, 11/12 features differ significantly |
| 3. Transfer Experiments | ✅ Complete | Best: 0.57±0.24 AUC (pretrained+dense+frozen) |
| 4. Velocity Ablation | ✅ Complete | Velocity drop: 0.13±0.21, Position drop: 0.05±0.05 |
| 5. DFL Comparison | Not Started | |

**Last Updated:** 2026-02-11
