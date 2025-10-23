# CornerTactics SLURM Pipeline Guide

Complete guide for reproducing the CornerTactics GNN dataset and model training from scratch.

## Overview

This pipeline processes corner kick data from StatsBomb and SkillCorner, applies temporal augmentation, and trains a Graph Neural Network to predict dangerous situations (shots and goals).

**Total Pipeline Runtime**: ~30-45 minutes
**Final Dataset**: 7,369 temporally augmented graphs
**Target Metric**: AUC-ROC > 0.60 for dangerous situation prediction

---

## Quick Start

### Option 1: Run Full Pipeline (Automated)
```bash
bash scripts/slurm/RUN_FULL_PIPELINE.sh
```
This script will automatically submit all phases in order and wait for each to complete.

### Option 2: Run Phases Individually (Manual)
```bash
# Phase 1: Data Integration & Labeling
sbatch scripts/slurm/phase1_1_complete.sh        # ~5 min
sbatch scripts/slurm/phase1_2_label_outcomes.sh  # ~5 min

# Phase 2: Feature Engineering & Graph Construction
sbatch scripts/slurm/phase2_1_extract_features.sh              # ~10 min
sbatch scripts/slurm/phase2_2_build_graphs.sh                  # ~5 min
sbatch scripts/slurm/phase2_3_skillcorner_temporal.sh          # ~10 min
sbatch scripts/slurm/phase2_4_statsbomb_augment.sh             # ~2 min
sbatch scripts/slurm/phase2_5_build_skillcorner_graphs.sh      # ~1 min

# Phase 3: Model Training
sbatch scripts/slurm/phase3_train_gnn.sh         # ~5 min
```

---

## Phase-by-Phase Details

### Phase 1.1: Data Integration
**Script**: `phase1_1_complete.sh`
**Runtime**: ~5 minutes
**Resources**: 8GB RAM, 4 CPUs

**Purpose**: Downloads and integrates corner kick data from StatsBomb and SkillCorner.

**Outputs**:
- `data/raw/statsbomb/corners_360.csv` - StatsBomb freeze-frame data (1,118 corners)
- `data/raw/skillcorner/` - SkillCorner tracking data (10 matches, 317 corners)
- `data/processed/unified_corners_dataset.parquet` - Combined dataset

**Check Success**:
```bash
tail logs/phase1_1_complete_*.out
wc -l data/raw/statsbomb/corners_360.csv  # Should be 1119 (including header)
```

---

### Phase 1.2: Outcome Labeling
**Script**: `phase1_2_label_outcomes.sh`
**Runtime**: ~5 minutes
**Resources**: 8GB RAM, 4 CPUs

**Purpose**: Labels each corner with its outcome (goal, shot, clearance, etc.) by analyzing subsequent events.

**Outputs**:
- Augmented `corners_360.csv` with outcome columns
- `goal_scored`, `outcome_label`, `time_to_outcome` columns

**Target Distribution**:
- Goals: ~14 (1.3%)
- Shots (dangerous): ~203 (18.2%)

**Check Success**:
```bash
tail logs/phase1_2_label_outcomes_*.out
# Look for "Labeled X corners" in output
```

---

### Phase 2.1: Node Feature Extraction
**Script**: `phase2_1_extract_features.sh`
**Runtime**: ~10 minutes
**Resources**: 8GB RAM, 4 CPUs

**Purpose**: Extracts 14-dimensional feature vectors for each player in every corner.

**Features Extracted** (14 dimensions):
1. x, y - Position coordinates
2. distance_to_goal - Distance from goal
3. distance_to_ball_target - Distance from corner kick target
4. vx, vy - Velocity components
5. velocity_magnitude, velocity_angle - Velocity metrics
6. angle_to_goal, angle_to_ball - Angular features
7. team_flag - Attacking (1) or Defending (0)
8. in_penalty_box - Boolean flag
9. num_players_within_5m - Local crowding
10. local_density_score - Spatial density

**Outputs**:
- `data/features/node_features/statsbomb_player_features.parquet` (21,231 player features)

**Check Success**:
```bash
tail logs/phase2_1_features_*.out
wc -l data/features/node_features/statsbomb_player_features.csv
```

---

### Phase 2.2: Graph Construction (StatsBomb Baseline)
**Script**: `phase2_2_build_graphs.sh`
**Runtime**: ~5 minutes
**Resources**: 8GB RAM, 4 CPUs

**Purpose**: Converts player features into graph representations with adjacency matrices.

**Adjacency Strategy**: Team-based (connect teammates only)

**Edge Features** (6 dimensions):
- Distance between players
- Relative velocity (x, y components)
- Relative velocity magnitude
- Angle between players (sine, cosine)

**Outputs**:
- `data/graphs/adjacency_team/statsbomb_graphs.pkl` (1,118 graphs)

**Check Success**:
```bash
tail logs/phase2_2_graphs_*.out
ls -lh data/graphs/adjacency_team/statsbomb_graphs.pkl  # Should be ~15MB
```

---

### Phase 2.3: SkillCorner Temporal Extraction
**Script**: `phase2_3_skillcorner_temporal.sh`
**Runtime**: ~10 minutes
**Resources**: 16GB RAM, 4 CPUs

**Purpose**: Extracts temporal features from SkillCorner 10fps tracking data.

**Temporal Frames**: 5 frames per corner
- t = -2.0s, -1.0s, 0.0s, +1.0s, +2.0s

**Data Access**: GitHub media URLs (bypasses Git LFS limits)

**Outputs**:
- `data/features/temporal/skillcorner_temporal_features.parquet` (1,555 temporal graphs from 317 corners)

**Check Success**:
```bash
tail logs/phase2_3_sc_temporal_*.out
wc -l data/features/temporal/skillcorner_temporal_features.csv  # Should be ~34,211
```

---

### Phase 2.4: StatsBomb Temporal Augmentation
**Script**: `phase2_4_statsbomb_augment.sh`
**Runtime**: ~2 minutes
**Resources**: 8GB RAM, 4 CPUs

**Purpose**: Applies US Soccer Federation temporal augmentation to StatsBomb freeze-frames.

**Augmentation Strategy**:
1. **Temporal offsets**: 5 frames (t = -2s, -1s, 0s, +1s, +2s)
2. **Position perturbations**: Gaussian noise proportional to temporal distance
3. **Mirror augmentation**: Left/right flip for t=0 frames

**Outputs**:
- `data/graphs/adjacency_team/statsbomb_temporal_augmented.pkl` (5,814 graphs)
  - 5,590 temporal variations (5× original)
  - 224 mirror augmentations

**Target Update**: Changed from "goal" (1.3%) to "dangerous situation" (shot OR goal, 18.2%)

**Check Success**:
```bash
tail logs/phase2_4_sb_augment_*.out
ls -lh data/graphs/adjacency_team/statsbomb_temporal_augmented.pkl  # Should be ~75MB
```

---

### Phase 2.5: Build SkillCorner Graphs and Merge
**Script**: `phase2_5_build_skillcorner_graphs.sh`
**Runtime**: ~1 minute
**Resources**: 8GB RAM, 4 CPUs

**Purpose**: Converts SkillCorner temporal features to graphs and merges with StatsBomb.

**Process**:
1. Build 1,555 SkillCorner graphs from temporal features
2. Merge with 5,814 StatsBomb augmented graphs
3. Create final combined dataset

**Outputs**:
- `data/graphs/adjacency_team/skillcorner_temporal_graphs.pkl` (1,555 graphs)
- `data/graphs/adjacency_team/combined_temporal_graphs.pkl` (7,369 graphs)

**Final Dataset**:
- Total graphs: 7,369 (6.6× increase from original)
- Dangerous situations: 1,258 (17.1% positive class)
- Average nodes per graph: 19.6
- Average edges per graph: 187.9

**Check Success**:
```bash
tail logs/phase2_5_sc_graphs_*.out
ls -lh data/graphs/adjacency_team/combined_temporal_graphs.pkl  # Should be ~95MB
```

---

### Phase 3: GNN Model Training
**Script**: `phase3_train_gnn.sh`
**Runtime**: ~5 minutes (100 epochs with early stopping)
**Resources**: 8GB RAM, 4 CPUs, 1 GPU

**Model Architecture**:
```
GraphConv1: 14 → 64 (ReLU, Dropout=0.3)
GraphConv2: 64 → 128 (ReLU, Dropout=0.3)
GraphConv3: 128 → 64 (ReLU)
Global Pooling: Mean + Max concatenation
Dense1: 128 → 64 (ReLU)
Dense2: 64 → 32 (ReLU)
Output: 32 → 1 (Sigmoid)
```

**Training Configuration**:
- Dataset: `combined_temporal_graphs.pkl` (7,369 graphs)
- Target: Dangerous situation (shot OR goal)
- Loss: Weighted binary cross-entropy
- Optimizer: Adam with cosine annealing
- Early stopping: Patience = 15 epochs

**Data Split**:
- Train: ~5,158 graphs (70%)
- Val: ~1,105 graphs (15%)
- Test: ~1,106 graphs (15%)

**Target Metrics**:
- Test AUC > 0.60 (baseline: 0.22, previous best: 0.58)
- Reduced val-test gap (less overfitting)
- Improved generalization with 6.6× more training data

**Outputs**:
- `models/corner_gnn_gcn_shot_<timestamp>/`
  - `best_model.pth` - Best validation checkpoint
  - `config.json` - Model configuration
  - `results.json` - Training metrics

**Check Success**:
```bash
tail -100 logs/phase3_train_gnn_*.out
# Look for "Test AUC:" in final output
```

**Monitor Training**:
```bash
# While training is running:
tail -f logs/phase3_train_gnn_<JOB_ID>.out

# Check if training is still running:
squeue -u $USER
```

---

## Troubleshooting

### Check Job Status
```bash
# See all your jobs
squeue -u $USER

# Check specific job
squeue -j <JOB_ID>

# View job output
tail -f logs/<phase_name>_<JOB_ID>.out
tail -f logs/<phase_name>_<JOB_ID>.err
```

### Common Issues

**Job fails immediately**:
- Check error log: `cat logs/<phase_name>_<JOB_ID>.err`
- Verify conda environment exists: `conda env list | grep robo`
- Check partition/account: `sinfo`

**Out of memory**:
- Increase `--mem` in SLURM script
- Reduce batch size for training

**Missing dependencies**:
- Activate environment: `conda activate robo`
- Install missing packages: `pip install <package>`

**Can't find data files**:
- Check if previous phase completed successfully
- Verify file exists: `ls -lh <data_file>`

---

## File Outputs Summary

After successful pipeline completion, you should have:

```
data/
├── raw/
│   ├── statsbomb/
│   │   └── corners_360.csv                    # 1,118 corners
│   └── skillcorner/                           # 10 matches
│
├── processed/
│   └── unified_corners_dataset.parquet         # Combined dataset
│
├── features/
│   ├── node_features/
│   │   └── statsbomb_player_features.parquet  # 21,231 players
│   └── temporal/
│       └── skillcorner_temporal_features.parquet  # 34,210 temporal features
│
└── graphs/
    └── adjacency_team/
        ├── statsbomb_graphs.pkl                    # 1,118 baseline graphs
        ├── statsbomb_temporal_augmented.pkl        # 5,814 augmented graphs
        ├── skillcorner_temporal_graphs.pkl         # 1,555 temporal graphs
        └── combined_temporal_graphs.pkl            # 7,369 final graphs ✓

models/
└── corner_gnn_gcn_shot_<timestamp>/
    ├── best_model.pth                          # Trained model weights
    ├── config.json                             # Model configuration
    └── results.json                            # Training metrics
```

---

## Next Steps After Pipeline

1. **Evaluate Model Performance**:
   ```bash
   python scripts/evaluate_model.py --model-dir models/corner_gnn_gcn_shot_<timestamp>
   ```

2. **Run Ablation Studies**:
   ```bash
   # Test different adjacency strategies
   python scripts/build_graph_dataset.py --strategy distance
   python scripts/build_graph_dataset.py --strategy delaunay
   ```

3. **Analyze Feature Importance**:
   ```bash
   python scripts/analyze_feature_importance.py
   ```

4. **Visualize Predictions**:
   ```bash
   python scripts/visualize_predictions.py --corner-id <ID>
   ```

---

## References

- **US Soccer Federation GNN**: https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn
- **Bekkers & Sahasrabudhe (2024)**: "A Graph Neural Network Deep-Dive into Successful Counterattacks"
- **StatsBomb Open Data**: https://github.com/statsbomb/open-data
- **SkillCorner Open Data**: https://github.com/SkillCorner/opendata

---

*Last Updated: October 23, 2024*
*Pipeline Version: 2.0 (Temporal Augmentation)*
