# CLAUDE.md

Context for Claude Code when working with this repository.

## Project Overview

CornerTactics predicts corner kick outcomes from SoccerNet broadcast videos using FAANTRA (Football Action ANticipation TRAnsformer).

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Build Dataset | **Complete** | Corner metadata from SoccerNet labels |
| 2. Video Clip Extraction | **Complete** | 4,836 clips extracted (114GB) |
| 3. Frame Extraction | **Complete** | Frames extracted for all splits |
| 4. Model Training | **Complete** | FAANTRA trained on corner outcomes |
| 5. Evaluation | **Complete** | 12.6% mAP@∞ (8-class), 50% mAP@∞ (binary) |

## Results Summary

| Task | Model | Result |
|------|-------|--------|
| 8-class outcome | FAANTRA | mAP@∞ = 12.6% |
| Binary shot/no-shot | FAANTRA | mAP@∞ = 50% (random) |
| Binary shot (StatsBomb) | Classical ML | AUC = 0.43 |
| Ball actions (baseline) | FAANTRA | mAP@∞ = 18.48% |

**Key Finding**: Shot prediction from pre-corner observation achieves random performance (~50%), confirming outcome depends on post-corner events.

## Project Structure

```
CornerTactics/
├── scripts/                           # Data pipeline (run from project root)
│   ├── 01_build_corner_dataset.py     # Build metadata from SoccerNet labels
│   ├── 02_extract_video_clips.py      # Extract 30s video clips
│   ├── 03_prepare_faantra_data.py     # Create splits + extract frames
│   └── slurm/
│       └── extract_frames.sbatch      # Parallel frame extraction
├── FAANTRA/                           # FAANTRA model code
│   ├── data/
│   │   ├── corners/
│   │   │   ├── corner_dataset.json    # Metadata for 4,836 corners
│   │   │   └── clips/                 # Video clips (114GB)
│   │   └── corner_anticipation/       # FAANTRA-formatted frames
│   │       ├── train/, valid/, test/  # Split directories
│   │       └── *.json, class.txt      # Labels and config
│   ├── main.py, train.py, test.py     # Training/evaluation
│   └── venv/                          # Python environment
├── data/misc/soccernet/videos/        # Source SoccerNet videos (1.1TB)
├── plan.md                            # Detailed project plan
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
