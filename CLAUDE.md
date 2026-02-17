# CLAUDE.md

Context for Claude Code when working with this repository.

## Project Overview

CornerTactics predicts corner kick shot outcomes using a two-stage GNN pipeline with USSF-pretrained CrystalConv backbone. Stage 1 predicts the receiver, Stage 2 predicts whether the corner leads to a shot. Earlier work used FAANTRA (video-based) and StatsBomb (event-based); the current approach uses tracking data from SkillCorner and DFL.

## Current Status

| Component | Status | Key Result |
|-----------|--------|------------|
| FAANTRA (video) | Complete | mAP@∞ = 50% binary (random) |
| StatsBomb (events) | Complete | AUC = 0.43 |
| Transfer Learning | Complete | 0.86 AUC open-play; corners ~0.55 (n=57) |
| **Two-Stage GNN** | **Complete** | **Shot AUC = 0.730 (combined), p=0.010** |

## Results Summary

All results use Leave-One-Match-Out (LOMO) cross-validation. **Dataset matters** — numbers differ between SkillCorner-only and combined.

### Two-Stage GNN (Main Results)

**Combined dataset (143 corners, 17 LOMO folds):**

| Metric | Value |
|--------|-------|
| Receiver Top-1 | 0.289 ± 0.226 (10 folds with labels) |
| Receiver Top-3 | 0.458 ± 0.341 (10 folds with labels) |
| Shot AUC (oracle) | 0.730 ± 0.202 (17 folds) |
| Shot AUC (predicted) | 0.715 ± 0.193 (17 folds) |
| Shot AUC (unconditional) | 0.730 ± 0.204 (17 folds) |
| Permutation p-value (shot) | **0.010** (100 perms) |
| Permutation p-value (receiver) | **0.050** (100 perms) |

**SkillCorner-only (86 corners, 10 LOMO folds):**

| Metric | Value |
|--------|-------|
| Receiver Top-3 | 0.308 ± 0.279 |
| Shot AUC (oracle) | 0.751 ± 0.213 |
| Permutation p-value (shot) | **0.020** (100 perms) |
| Permutation p-value (receiver) | 0.406 (not significant) |

### Ablations (SkillCorner-only, 10 folds)

| Config | Active Features | Receiver Top-3 | Shot AUC |
|--------|----------------|----------------|----------|
| position_only | x,y + team/role flags (9 feat) | 0.408 ± 0.290 | 0.583 ± 0.280 |
| plus_velocity | + vx, vy, speed (12 feat) | 0.398 ± 0.330 | 0.747 ± 0.238 |
| plus_detection | + is_detected (all 13) | 0.308 ± 0.279 | 0.751 ± 0.213 |
| full_features | all 13, KNN k=6 (default) | 0.308 ± 0.279 | 0.748 ± 0.218 |
| full_fc_edges | all 13, fully connected | 0.525 ± 0.364 | 0.713 ± 0.198 |

### Baselines (SkillCorner-only, 10 folds)

| Model | Input | Shot AUC |
|-------|-------|----------|
| MLP | 22×13 flattened (286 feat) | 0.802 ± 0.214 |
| XGBoost | 27 aggregate features | 0.743 ± 0.232 |
| GNN (pretrained) | graph (13 node, 4 edge feat) | 0.751 ± 0.213 |
| Random baseline | — | 0.500 |

**Note:** Baselines have no permutation tests. MLP and XGBoost only run on SkillCorner-only (86 corners). Combined LOMO (143 corners) only has GNN results.

### Earlier Approaches

| Task | Model | Result |
|------|-------|--------|
| 8-class outcome | FAANTRA (video) | mAP@∞ = 12.6% |
| Binary shot/no-shot | FAANTRA (video) | mAP@∞ = 50% (random) |
| Binary shot | StatsBomb (events) | AUC = 0.43 |
| Open-play transfer | USSF→DFL linear probe | AUC = 0.86 |

## Project Structure

```
CornerTactics/
├── corner_prediction/                 # Two-stage GNN pipeline (MAIN)
│   ├── config.py                      # All hyperparameters and paths
│   ├── run_all.py                     # Entry point (eval, ablation, baselines, permutation)
│   ├── data/
│   │   ├── extract_corners.py         # SkillCorner corner extraction
│   │   ├── extract_dfl_corners.py     # DFL Bundesliga corner extraction
│   │   ├── build_graphs.py            # Graph construction (nodes, edges)
│   │   ├── merge_datasets.py          # Combine SkillCorner + DFL
│   │   ├── dataset.py                 # LOMO splits, data loading
│   │   ├── extracted_corners.pkl      # 86 SkillCorner corners
│   │   ├── dfl_extracted_corners.pkl  # 57 DFL corners
│   │   └── combined_corners.pkl       # 143 combined corners
│   ├── models/
│   │   ├── backbone.py                # CrystalConv GNN (pretrained/scratch)
│   │   ├── two_stage.py               # TwoStageModel orchestration
│   │   ├── receiver_head.py           # Stage 1: per-node receiver prediction
│   │   └── shot_head.py               # Stage 2: graph-level shot prediction
│   ├── training/
│   │   ├── train.py                   # Training loops for both stages
│   │   ├── evaluate.py                # LOMO cross-validation
│   │   ├── permutation_test.py        # Statistical validation
│   │   └── ablation.py                # Feature ablation experiments
│   ├── baselines/
│   │   ├── mlp_baseline.py            # Flattened MLP (286 features)
│   │   └── xgboost_baseline.py        # Aggregate features (27 features)
│   └── visualization/                 # Thesis-ready plots
├── transfer_learning/                 # USSF transfer experiments
├── FAANTRA/                           # Video-based model (earlier work)
│   └── venv/                          # Python environment (shared)
├── results/corner_prediction/         # All result JSON/pickle files
├── scripts/                           # Data pipeline + SLURM jobs
├── tests/                             # pytest test suite
└── CLAUDE.md                          # This file
```

## Data Pipeline

All scripts run from CornerTactics root directory:

```bash
cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

# Step 1: Build corner dataset (already complete)
python scripts/01_build_corner_dataset.py

# Step 2: Extract video clips (already complete)
python scripts/02_extract_video_clips.py

# Step 2b: Verify and repair corrupt clips
python scripts/02_extract_video_clips.py --verify --repair

# Step 3a: Create train/val/test splits
python scripts/03_prepare_faantra_data.py --splits-only

# Step 3b: Extract frames (parallel via SLURM)
sbatch scripts/slurm/extract_frames.sbatch                        # Train
SPLIT=valid CHUNKS=5 sbatch scripts/slurm/extract_frames.sbatch   # Valid
SPLIT=test CHUNKS=5 sbatch scripts/slurm/extract_frames.sbatch    # Test
```

## Dataset

### Video Clips
- **Location**: `FAANTRA/data/corners/clips/corner_XXXX/720p.mp4`
- **Count**: 4,836 clips
- **Duration**: 30 seconds (25s observation + 5s anticipation)
- **Size**: 114GB total

### Outcome Distribution
| Outcome | Count | % |
|---------|-------|---|
| NOT_DANGEROUS | 1,939 | 40.1% |
| CLEARED | 1,138 | 23.5% |
| SHOT_OFF_TARGET | 713 | 14.7% |
| SHOT_ON_TARGET | 387 | 8.0% |
| FOUL | 384 | 7.9% |
| GOAL | 172 | 3.6% |
| OFFSIDE | 77 | 1.6% |
| CORNER_WON | 26 | 0.5% |

### Train/Val/Test Splits
- Train: 3,868 clips (80%)
- Valid: 483 clips (10%)
- Test: 485 clips (10%)

## Environment Setup

```bash
# Activate Python environment
source FAANTRA/venv/bin/activate

# Load FFmpeg (for video processing)
module load GCCcore/12.3.0 FFmpeg/6.0-GCCcore-12.3.0
```

## Common Commands

```bash
# Check video clip extraction status
ls FAANTRA/data/corners/clips/ | wc -l  # Should be 4836

# Check frame extraction progress
ls -d FAANTRA/data/corner_anticipation/train/clip_*/ 2>/dev/null | wc -l

# Verify corrupt clips
python scripts/02_extract_video_clips.py --verify

# Monitor SLURM jobs
squeue -u $USER
```

## Training (after frame extraction)

```bash
cd FAANTRA
source venv/bin/activate
python main.py config/corner_config.json corner_model
```

## Important Notes

1. **Run scripts from CornerTactics root** - All paths are relative to project root
2. **FFmpeg module required** - Load before video processing
3. **Large data** - Clips are 114GB, source videos are 1.1TB (gitignored)
4. **Class imbalance** - 40% NOT_DANGEROUS vs 0.5% CORNER_WON - use weighted loss
5. **SLURM for parallelism** - Use array jobs for frame extraction

## Two-Stage GNN Pipeline (Complete)

### Dataset

| Source | Corners | Matches | Receiver Labels | Shots | Shot Rate |
|--------|---------|---------|-----------------|-------|-----------|
| SkillCorner (A-League) | 86 | 10 | 66 (76.7%) | 29 | 33.7% |
| DFL (Bundesliga) | 57 | 7 | 0 (0%) | 16 | 28.1% |
| **Combined** | **143** | **17** | **66 (46.2%)** | **45** | **31.5%** |

**Receiver labels (SkillCorner only):** 3 methods in priority order — (1) `player_targeted_name`, (2) `passing_option` targeted+received, (3) `player_possession`. DFL XML lacks event-level data.

**Shot labels:** SkillCorner uses `lead_to_shot` column directly; DFL derives from `ShotAtGoal` XML events matched via `BuildUp=cornerKick`.

**Quality filters:** SkillCorner `min_detection_rate=0.0` (no filtering). DFL requires 20-24 players, `max_gap=25` frames (1s at 25Hz).

### Graph Construction

**13 node features per player** (indices for ablation configs):
- 0-1: x_norm (x/52.5), y_norm (y/34.0) — range [-1, 1]
- 2-4: vx, vy, speed — raw m/s, backward frame difference, not smoothed
- 5-7: is_attacking, is_corner_taker, is_goalkeeper — binary
- 8: is_detected — binary
- 9-12: position group one-hot (GK, DEF, MID, FWD)

**14th feature** appended at forward time: receiver indicator (0 for Stage 1, 1.0 at receiver for Stage 2).

**4 edge features:** dx, dy, distance, same_team.

**Edge construction:** KNN k=6 (default, 132 edges) or fully connected (462 edges).

**Coordinate normalization:** All corners flipped so attacking team attacks toward +x. Both SkillCorner and DFL apply identical direction normalization.

**Velocity:** Raw backward difference. SkillCorner: `(x_t - x_{t-1}) / 0.1` at 10Hz. DFL: `(x_nearest - x_prev) / dt` at 25Hz.

### Architecture

```
Input: [N=22, 13] node features + [E, 4] edge features
  → Augment with receiver indicator → [N, 14]
  → node_proj: Linear(14, 12)                 [TRAINABLE]
  → edge_proj: Linear(4, 6)                   [TRAINABLE]
  → conv1: CGConv(12, dim=6) + ReLU           [FROZEN — USSF pretrained]
  → lin_in: Linear(12, 128)                   [FROZEN]
  → convs[0]: CGConv(128, dim=6) + ReLU       [FROZEN]
  → convs[1]: CGConv(128, dim=6) + ReLU       [FROZEN]
  → Per-node embeddings: [N, 128]

Stage 1 (Receiver):
  → ReceiverHead: Linear(128,64) + ReLU + Dropout(0.3) + Linear(64,1)
  → masked_softmax over valid candidates → [N] probabilities

Stage 2 (Shot):
  → global_mean_pool → [B, 128]
  → cat(graph_emb, corner_side) → [B, 129]
  → ShotHead: Linear(129,32) + ReLU + Dropout(0.3) + Linear(32,1) → logit
```

**USSF backbone:** Trained on 20,863 counterattack graphs, AUC=0.693 on original task. Weights: `transfer_learning/weights/ussf_backbone_dense.pt`.

### Training

| Parameter | Stage 1 (Receiver) | Stage 2 (Shot) |
|-----------|-------------------|-----------------|
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 1e-3 |
| Weight decay | 1e-3 | 1e-3 |
| Max epochs | 100 | 100 |
| Early stopping patience | 20 (val CE loss) | 20 (val BCE loss) |
| Batch size | 8 | 8 |
| Class weight | — | pos_weight=2.0 |

Stage 2 trains with **oracle receiver** conditioning. Evaluated in all three modes (oracle, predicted, unconditional). Receiver head frozen during Stage 2 training.

**Seed:** 42 (single seed). `SEEDS = [42, 123, 456, 789, 1234]` defined but multi-seed averaging not implemented.

### Per-Fold Combined LOMO Results (Pretrained, Frozen)

**Shot AUC (oracle), 17 folds:**

| Fold | Match | Source | n_test | AUC |
|------|-------|--------|--------|-----|
| 1 | 1886347 | SK | 8 | 1.000 |
| 2 | 1899585 | SK | 7 | 0.600 |
| 3 | 1925299 | SK | 4 | 1.000 |
| 4 | 1953632 | SK | 8 | 0.875 |
| 5 | 1996435 | SK | 14 | 0.825 |
| 6 | 2006229 | SK | 4 | 1.000 |
| 7 | 2011166 | SK | 12 | 0.600 |
| 8 | 2013725 | SK | 13 | 0.533 |
| 9 | 2015213 | SK | 4 | 1.000 |
| 10 | 2017461 | SK | 12 | 0.700 |
| 11 | J03WMX | DFL | 10 | 0.571 |
| 12 | J03WN1 | DFL | 7 | 0.700 |
| 13 | J03WOH | DFL | 6 | 0.750 |
| 14 | J03WOY | DFL | 2 | 0.500 |
| 15 | J03WPY | DFL | 11 | 0.900 |
| 16 | J03WQQ | DFL | 15 | 0.364 |
| 17 | J03WR9 | DFL | 6 | 0.500 |

4 folds hit AUC=1.0 (all small SK matches, 4-8 corners). Fold 16 worst at 0.364.

### Permutation Tests

| Dataset | Metric | Real | Null mean±std | Null range | p-value |
|---------|--------|------|---------------|------------|---------|
| Combined (143, 17 folds) | Shot AUC (oracle) | 0.715 | 0.508±0.059 | [0.341, 0.633] | **0.010** |
| Combined (143, 17 folds) | Receiver Top-3 | 0.458 | 0.300±0.086 | [0.058, 0.535] | **0.050** |
| SkillCorner (86, 10 folds) | Shot AUC (oracle) | 0.751 | 0.502±0.087 | — | **0.020** |
| SkillCorner (86, 10 folds) | Receiver Top-3 | 0.308 | 0.284±0.087 | — | 0.406 |

100 permutations each. Both real and shuffled use identical LOMO CV (apples-to-apples). No multi-seed permutation stability testing.

### Known Gaps

1. **Baselines on combined dataset (PARTIAL):** MLP and XGBoost now evaluated on combined (143 corners, 17 folds). GNN has highest mean AUC (0.730 vs XGBoost 0.695 vs MLP 0.665), but no formal paired significance test between models. Random and heuristic baselines not yet run on combined.
2. **No permutation tests on baselines:** MLP/XGBoost significance not formally tested on either dataset.
3. **Single seed:** Multi-seed averaging not implemented despite SEEDS list in config.
4. **DFL has no receiver labels:** Stage 1 receiver evaluation limited to 10 SkillCorner folds.

### Running the Pipeline

```bash
cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

# Full evaluation (combined dataset, pretrained backbone)
python -m corner_prediction.run_all --eval --combined

# Ablations (SkillCorner-only)
python -m corner_prediction.run_all --ablation position_only plus_velocity plus_detection full_features full_fc_edges

# Baselines (SkillCorner-only)
python -m corner_prediction.run_all --baselines

# Permutation tests (combined dataset)
python -m corner_prediction.run_all --permutation-only --combined
```

## Transfer Learning Experiment (Complete)

**Goal:** Test whether GNN representations from USSF counterattack prediction (20,863 graphs) transfer to corner kick shot prediction (57 DFL graphs).

### Phase Summary

| Phase | Status | Key Result |
|-------|--------|------------|
| 0. Data Inspection | ✅ | USSF uses 12 node features, 6 edge features |
| 1. Train USSF Backbone | ✅ | Dense AUC=0.693, Normal AUC=0.683 |
| 2. Engineer DFL Features | ✅ | 57 corners transformed, 11/12 features differ significantly (KS test) |
| 3. Transfer Experiments | ✅ | Best: 0.57±0.24 AUC (pretrained+dense+frozen) |
| 4. Velocity Ablation | ✅ | Velocity features more important than raw position |
| 5. DFL Open-Play Comparison | ✅ | Corner results underpowered (n=57) |
| 5b. Transfer Validation | ✅ | **0.86 AUC** on open-play (n=1,796) - transfer works |

### Transfer Learning Results

| Condition | Description | Test AUC (mean±std) |
|-----------|-------------|---------------------|
| A | USSF pretrained + dense + frozen | **0.57 ± 0.24** |
| B | USSF pretrained + normal + frozen | 0.51 ± 0.21 |
| C | USSF pretrained + dense + unfrozen | 0.56 ± 0.23 |
| D | Random init + dense | 0.39 ± 0.29 |
| E | Random init + normal | 0.43 ± 0.27 |
| F | Majority baseline | 0.50 ± 0.00 |

### Velocity Ablation Results (Permutation Importance)

| Condition | Baseline AUC | Velocity Drop | Position (x,y) Drop | Position Derived Drop |
|-----------|--------------|---------------|---------------------|----------------------|
| A (frozen) | 0.64 ± 0.20 | 0.13 ± 0.21 | 0.05 ± 0.05 | 0.23 ± 0.15 |
| C (unfrozen) | 0.61 ± 0.13 | 0.10 ± 0.10 | 0.03 ± 0.17 | 0.05 ± 0.14 |

### Key Findings

1. **Marginal Transfer Benefit:** Pretrained models (A: 0.57) slightly outperform random baseline (0.50), but high variance means results are not statistically significant with 57 samples.

2. **Dense > Normal Adjacency:** Dense adjacency (fully connected) outperforms normal (team-based) for corner kicks, as expected since corners are densely packed situations.

3. **Random Init Fails:** Training from scratch (D, E) performs worse than baseline - severe overfitting with 57 samples.

4. **Velocity > Raw Position:** Velocity ablation causes 2-3x larger AUC drops than raw position ablation, supporting the hypothesis that dynamic features matter more than static coordinates.

5. **Spatial Relationships Important:** The frozen model relies heavily on derived position features (dist_goal, angle_goal, dist_ball, angle_ball), suggesting the USSF backbone learned spatial relationship patterns.

6. **Connection to 7.5 ECTS Finding:** Confirms that position-only achieves ~0.50 AUC (random). Velocity provides signal, but effect is modest and noisy due to small sample size.

### Phase 5: DFL Open-Play Comparison Results

**Goal:** Compare USSF counterattack pretraining vs DFL open-play pretraining.

**DFL Open-Play Pretraining:**
- Extracted 11,967 graphs from 7 DFL matches (1s sampling interval)
- Task: Predict shot within next 5 seconds (365 positive, 3.1%)
- Trained CrystalConv backbone (same architecture as USSF)

| Condition | Description | Test AUC (mean±std) |
|-----------|-------------|---------------------|
| G | DFL pretrained + frozen | 0.55 ± 0.22 |
| H | DFL pretrained + unfrozen | 0.55 ± 0.22 |
| A | USSF pretrained + frozen | 0.57 ± 0.24 |

**Corner Results:** All conditions achieve ~0.55-0.57 AUC with ±0.22-0.24 std. With 57 samples, these results are **not statistically significant** - the confidence intervals span from worse-than-random to good performance.

### Phase 5b: Transfer Validation (Statistically Powered)

**Critical validation:** Test USSF pretrained model on DFL open-play data (n=11,967).

| Metric | Value |
|--------|-------|
| Test samples | 1,796 |
| Test AUC | **0.8632** |
| Test Accuracy | 83.1% |

**Conclusion:** USSF representations **do transfer** to DFL data. A simple linear probe on frozen USSF features achieves 0.86 AUC for shot prediction. The method works - but corner-specific prediction remains at chance level even with proven representations.

### Files

```
transfer_learning/
├── phase0_inspect_ussf_data.py      # USSF data inspection
├── phase1_train_ussf_backbone.py    # Train CrystalConv backbone
├── phase2_engineer_dfl_features.py  # Transform DFL corners to USSF schema
├── phase3_transfer_learning.py      # Run 6 experimental conditions
├── phase4_velocity_ablation.py      # Permutation importance testing
├── phase5_dfl_openplay_comparison.py # DFL vs USSF pretraining comparison
├── phase5b_ussf_on_dfl_openplay.py   # Transfer validation (0.86 AUC)
├── data/
│   ├── dfl_corners_ussf_format_dense.pkl   # 57 corners (dense adj)
│   ├── dfl_corners_ussf_format_normal.pkl  # 57 corners (normal adj)
│   └── dfl_openplay_graphs.pkl             # 11,967 open-play graphs
├── weights/
│   ├── ussf_backbone_dense.pt       # Pretrained backbone (dense)
│   ├── ussf_backbone_normal.pt      # Pretrained backbone (normal)
│   └── dfl_backbone_dense.pt        # DFL open-play pretrained backbone
└── results/
    ├── phase3_multiseed_*.pkl       # Transfer experiment results
    ├── phase4_ablation_*.pkl        # Velocity ablation results
    └── phase5_comparison_*.pkl      # DFL vs USSF comparison results
```

### Running Transfer Learning

```bash
cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

# Phase 3: Transfer experiments (all conditions, 5 seeds)
python transfer_learning/phase3_transfer_learning.py --multi-seed

# Phase 4: Velocity ablation
python transfer_learning/phase4_velocity_ablation.py --conditions A C
```

### Conclusions

- **Transfer works for open-play:** USSF pretrained features achieve 0.86 AUC on DFL open-play shot prediction (n=1,796). This validates that the GNN representations are useful and transfer cross-dataset.
- **Corner prediction remains random:** Despite proven representations, corner-specific prediction achieves only ~0.55-0.57 AUC (not statistically different from 0.50 with n=57).
- **Velocity shows directional signal:** Ablation suggests velocity features matter more than position (0.13 vs 0.05 AUC drop), but the difference is not statistically significant with current sample size.
- **Interpretation:** The representations work. Shot prediction is solvable. But corner outcomes specifically are fundamentally harder to predict from pre-delivery state - the signal isn't in the setup, it's in the execution.
