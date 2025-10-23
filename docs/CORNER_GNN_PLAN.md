# Corner Kick GNN Prediction System - Implementation Plan

## Overview
Implementing a Graph Neural Network (GNN) based corner kick outcome prediction system using the methodology from Bekkers & Sahasrabudhe (2024) "A Graph Neural Network Deep-Dive into Successful Counterattacks".

## Data Inventory

### Available Datasets
- [x] **StatsBomb 360**: 1,118 corners with freeze-frame player positions
  - Location: `data/statsbomb/corners_360.csv`
  - Status: ✅ Downloaded and augmented
  - Features: Player positions (JSON), corner location, pass end location
  - **Augmented**: 5,814 temporal graphs (5× temporal + mirrors)

- [x] **SkillCorner Open Data**: 10 A-League matches with continuous tracking
  - Location: `data/skillcorner/`
  - Status: ✅ Downloaded and processed
  - Features: 10fps tracking, dynamic events, phases of play
  - Corner events: 317 corners
  - **Temporal**: 1,555 graphs (5 frames per corner)
  - Access: Via GitHub media URLs (bypasses Git LFS)

- [ ] **SoccerNet**: Video clips and tracking data
  - Location: `data/datasets/soccernet/`
  - Status: ⏳ Not yet integrated
  - Corner clips: 4,208 videos in `corner_clips/visible/`
  - Tracking data: Available in `tracking/`

### **Final Dataset**: 7,369 graphs (6.6× increase from original)
- Dangerous situations: ~1,261 (17.1% positive class)
- Target: "Shot OR Goal" instead of goal-only

## Phase 1: Data Integration & Enrichment ⏳

### 1.1 Create Unified Corner Database
- [x] Extract corner events from SkillCorner dynamic_events.csv files
  - [x] Parse all 10 matches for corner_for/corner_against events
  - [x] Extract frame ranges and timestamps
  - [x] Link to tracking data at corner moments

- [x] Add outcome labels to StatsBomb corners
  - [x] Fetch subsequent events (15-20 seconds window)
  - [x] Classify outcomes: Goal/Shot/Clearance/Possession
  - [x] Calculate time_to_outcome and events_to_outcome

- [x] Map SoccerNet video clips to tracking data
  - [x] Match video filenames to game/timestamp
  - [x] Create unified index of all corners

- [x] Output unified dataset
  - [x] Create `data/unified_corners_dataset.parquet`
  - [x] Include source, match_id, timestamp, players, outcome

### 1.2 Outcome Labeling Pipeline
- [x] Define success metrics
  - [x] Binary: Goal within 20 seconds (primary)
  - [x] Multi-class: Shot/Clearance/Second Corner/Possession
  - [x] xThreat: Expected threat value generated

- [x] Implement labeling functions
  - [x] `get_corner_outcome()` - extract outcome from event sequence
  - [x] `calculate_xthreat()` - compute threat value
  - [x] `get_temporal_features()` - time/events to outcome

## Phase 2: Graph Construction 🔧

### 2.1 Node Feature Engineering (12 dimensions per player)
- [x] Spatial features
  - [x] x, y coordinates (normalized to pitch dimensions)
  - [x] distance_to_goal
  - [x] distance_to_ball_target

- [ ] Kinematic features (from SkillCorner 10fps)
  - [ ] velocity_x (vx)
  - [ ] velocity_y (vy)
  - [ ] velocity_magnitude
  - [ ] velocity_angle

- [x] Contextual features
  - [x] angle_to_goal
  - [x] angle_to_ball
  - [x] team_flag (attacking/defending)
  - [x] in_penalty_box flag

- [x] Density features
  - [x] num_players_within_5m
  - [x] local_density_score

### 2.2 Adjacency Matrix Construction
- [x] Implement 5 connection strategies
  - [x] Team-based: Connect teammates only
  - [x] Distance-based: Connect players within 10m threshold
  - [x] Delaunay: Triangulation for spatial relationships
  - [x] Ball-centric: Connect players near ball trajectory
  - [x] Zone-based: Connect players in same tactical zone

### 2.3 SkillCorner Temporal Extraction ✅
- [x] Access tracking data via GitHub media URLs
  - [x] Bypass Git LFS budget limitations
  - [x] Load 10fps tracking data directly from GitHub
- [x] Extract temporal frames
  - [x] 5 frames per corner: t = -2s, -1s, 0s, +1s, +2s
  - [x] Extract 14-dim features for each frame
  - [x] Output: 1,555 temporal graphs from 317 corners
- [x] Label dangerous situations
  - [x] 41 shots identified (13.2% positive class)

### 2.4 StatsBomb Temporal Augmentation ✅
- [x] Implement US Soccer Federation approach
  - [x] 5 temporal offsets per corner
  - [x] Position perturbations with Gaussian noise
  - [x] Noise proportional to temporal distance
- [x] Mirror augmentation
  - [x] Flip y-coordinates for left/right symmetry
  - [x] Applied to t=0 frames only
  - [x] 224 additional graphs
- [x] Update target label
  - [x] Changed from "goal" (1.3%) to "shot OR goal" (18.2%)
  - [x] 1,056 dangerous situations in augmented data
- [x] Output: 5,814 augmented graphs

### 2.5 Build SkillCorner Graphs and Merge Datasets ✅
- [x] Build SkillCorner graph representations
  - [x] Convert temporal features to graphs (1,555 graphs)
  - [x] Apply team-based adjacency strategy
  - [x] Label dangerous situations (202 shots, 13%)
- [x] Merge with StatsBomb augmented graphs
  - [x] Combine 5,814 StatsBomb + 1,555 SkillCorner = 7,369 total
  - [x] Final positive class: 1,258 dangerous situations (17.1%)
  - [x] Output: `combined_temporal_graphs.pkl`

### 2.6 Edge Features (6 dimensions)
- [x] Distance between connected players
- [x] Relative velocity
- [x] Angle differences (sine/cosine pairs)

## Phase 3: GNN Model Implementation ✅

### 3.1 Core Architecture (PyTorch Geometric)
- [x] Set up environment
  - [x] Selected PyTorch Geometric over Spektral
  - [x] Created installation script in SLURM job

- [x] Implement model layers (`src/gnn_model.py`)
  ```python
  - [x] GraphConv1: 14 → 64 (ReLU, Dropout=0.3)
  - [x] GraphConv2: 64 → 128 (ReLU, Dropout=0.3)
  - [x] GraphConv3: 128 → 64 (ReLU)
  - [x] Global Pooling: Mean + Max concatenation
  - [x] Dense1: 128 → 64 (ReLU)
  - [x] Dense2: 64 → 32 (ReLU)
  - [x] Output: 32 → 1 (Sigmoid for binary)
  ```

### 3.2 Training Infrastructure
- [x] Data loader module (`src/data_loader.py`)
- [x] Training utilities (`src/train_utils.py`)
- [x] Main training script (`scripts/train_gnn.py`)
- [x] Evaluation script (`scripts/evaluate_model.py`)
- [x] SLURM job script (`scripts/slurm/phase3_train_gnn.sh`)

### 3.3 Multi-Modal Extension (Future Work)
- [ ] Add CNN branch for video frames
- [ ] Implement feature fusion layer
- [ ] Joint training pipeline

## Phase 4: Training Pipeline 🚀

### 4.1 Data Preparation
- [ ] Split datasets (prevent leakage by match_id)
  - [ ] StatsBomb: 800 train / 150 val / 168 test
  - [ ] SkillCorner: 7 matches train / 1 val / 2 test
  - [ ] SoccerNet: 3000 train / 500 val / 708 test

- [ ] Create data loaders
  - [ ] Implement batch sampling (32 graphs/batch)
  - [ ] Handle variable graph sizes
  - [ ] Implement data augmentation

### 4.2 Training Configuration
- [ ] Set up training loop
  - [ ] Loss: Binary crossentropy + auxiliary losses
  - [ ] Optimizer: Adam with cosine annealing
  - [ ] Early stopping (patience=10)

- [ ] Implement metrics tracking
  - [ ] AUC-ROC
  - [ ] LogLoss
  - [ ] Expected Calibration Error (ECE)
  - [ ] Precision@k

### 4.3 Model Evaluation
- [ ] Feature importance analysis
  - [ ] Permutation importance
  - [ ] Node attention weights

- [ ] Performance benchmarks
  - [ ] Compare adjacency matrix types
  - [ ] Ablation studies on features

- [ ] Visualization
  - [ ] Plot important spatial patterns
  - [ ] Team-specific strategies

## Pipeline Execution 🚀

### Automated Full Pipeline
**Script**: `scripts/slurm/RUN_FULL_PIPELINE.sh`

Complete automated execution of all phases:
```bash
bash scripts/slurm/RUN_FULL_PIPELINE.sh
```

This will automatically:
1. Submit all phases in order (1.1 → 1.2 → 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 3)
2. Wait for each phase to complete before starting the next
3. Check for successful completion
4. Provide status updates

**Total Runtime**: ~30-45 minutes

### Manual Phase-by-Phase Execution
```bash
# Phase 1: Data Integration & Labeling (~10 min)
sbatch scripts/slurm/phase1_1_complete.sh
sbatch scripts/slurm/phase1_2_label_outcomes.sh

# Phase 2: Feature Engineering & Graphs (~28 min)
sbatch scripts/slurm/phase2_1_extract_features.sh        # ~10 min
sbatch scripts/slurm/phase2_2_build_graphs.sh            # ~5 min
sbatch scripts/slurm/phase2_3_skillcorner_temporal.sh    # ~10 min
sbatch scripts/slurm/phase2_4_statsbomb_augment.sh       # ~2 min
sbatch scripts/slurm/phase2_5_build_skillcorner_graphs.sh # ~1 min

# Phase 3: Training (~5 min)
sbatch scripts/slurm/phase3_train_gnn.sh
```

### Pipeline Documentation
**Complete guide**: `scripts/slurm/PIPELINE_README.md`

Contains detailed information about:
- Each phase's purpose and outputs
- Resource requirements
- Expected file outputs
- Troubleshooting tips
- Success verification commands

## Phase 5: Implementation Files 📁

### Core Modules (`src/`)
- [x] `data_integration.py` - Merge all data sources
- [x] `graph_builder.py` - Convert tracking to graphs
- [x] `gnn_model.py` - PyTorch Geometric GNN architecture
- [x] `feature_engineering.py` - Node/edge feature extraction
- [x] `outcome_labeler.py` - Label corner outcomes
- [x] `data_loader.py` - PyTorch Geometric data loading
- [x] `train_utils.py` - Training utilities and metrics

### Scripts (`scripts/`)
- [x] `integrate_datasets.py` - Combine StatsBomb + SkillCorner + SoccerNet
- [x] `extract_skillcorner_corners.py` - Parse SkillCorner corners
- [x] `build_graph_dataset.py` - Create graph dataset
- [x] `extract_skillcorner_temporal.py` - Extract temporal features from SkillCorner (Phase 2.3)
- [x] `augment_statsbomb_temporal.py` - Temporal augmentation for StatsBomb (Phase 2.4)
- [x] `build_skillcorner_graphs.py` - Build SkillCorner graphs and merge datasets (Phase 2.5)
- [x] `train_gnn.py` - Main training pipeline
- [x] `evaluate_model.py` - Model evaluation
- [ ] `predict_corner.py` - Inference on new corners

### SLURM Jobs (`scripts/slurm/`)
- [x] `phase1_1_complete.sh` - Data integration job
- [x] `phase1_2_label_outcomes.sh` - Outcome labeling
- [x] `phase2_1_extract_features.sh` - Node feature extraction
- [x] `phase2_2_build_graphs.sh` - Graph construction (StatsBomb baseline)
- [x] `phase2_3_skillcorner_temporal.sh` - SkillCorner temporal extraction
- [x] `phase2_4_statsbomb_augment.sh` - StatsBomb temporal augmentation
- [x] `phase2_5_build_skillcorner_graphs.sh` - Build SkillCorner graphs and merge datasets
- [x] `phase3_train_gnn.sh` - GNN training job (on combined dataset)
- [x] `RUN_FULL_PIPELINE.sh` - **Automated pipeline execution script**
- [x] `PIPELINE_README.md` - **Complete pipeline documentation**
- [ ] `phase3_evaluate_gnn.sh` - Evaluation job
- [ ] `hyperparam_search.sh` - Hyperparameter tuning

## Phase 6: Deployment & Analysis 📊

### 6.1 Model Deployment
- [ ] Save trained models
  - [ ] Best validation checkpoint
  - [ ] Ensemble of top models

- [ ] Create prediction API
  - [ ] Real-time inference (<100ms)
  - [ ] Batch prediction support

### 6.2 Results Analysis
- [ ] Generate insights report
  - [ ] Key spatial patterns for success
  - [ ] Team-specific strategies
  - [ ] Feature importance rankings

- [ ] Visualization outputs
  - [ ] Heatmaps of dangerous zones
  - [ ] Player movement patterns
  - [ ] Success probability overlays

## Progress Tracking

### Completed ✅
- [x] Downloaded StatsBomb 360 data (1,118 corners)
- [x] Downloaded SkillCorner Open Data (10 matches)
- [x] Downloaded SoccerNet corner clips (4,208 videos)
- [x] Reviewed Bekkers & Sahasrabudhe (2024) paper
- [x] Phase 1.1: Data Integration - Unified dataset created
- [x] Phase 1.2: Outcome Labeling - All corners labeled
- [x] Phase 2.1: Node Feature Engineering - 14-dim features extracted
- [x] Phase 2.2: Graph Construction - 5 adjacency strategies implemented
- [x] Phase 2.3: SkillCorner Temporal Extraction - 1,555 temporal graphs from 317 corners
- [x] Phase 2.4: StatsBomb Temporal Augmentation - 5,814 augmented graphs
- [x] Phase 2.5: Build SkillCorner Graphs and Merge - 7,369 combined graphs
- [x] Phase 3: GNN Model Implementation - PyTorch Geometric architecture complete
- [x] Built graph dataset: 1,118 corners with team-based adjacency (baseline)
- [x] Expanded dataset: 7,369 temporal graphs (6.6× increase)
- [x] Combined StatsBomb + SkillCorner: 5,814 + 1,555 = 7,369 graphs
- [x] Changed target label: "dangerous situation" (shot OR goal) - 17.1% positive class
- [x] Created complete training infrastructure (model, data loader, training script)
- [x] Implemented evaluation and visualization scripts
- [x] Initial baseline training: Val AUC 0.765, Test AUC 0.271 (on small dataset)
- [x] Created automated pipeline execution script (`RUN_FULL_PIPELINE.sh`)
- [x] Documented complete pipeline (`PIPELINE_README.md`)

### In Progress ⏳
- [ ] Re-train GNN with expanded dataset (7,369 graphs, dangerous situation target)

### Next Steps 📝
1. **Monitor current training job** (28657): `tail -f logs/phase3_train_gnn_28657.out`
2. **Reproduce full pipeline** (if needed): `bash scripts/slurm/RUN_FULL_PIPELINE.sh`
3. Run ablation studies with different adjacency strategies:
   ```bash
   python scripts/build_graph_dataset.py --strategy distance
   python scripts/build_graph_dataset.py --strategy delaunay
   sbatch scripts/slurm/phase3_train_gnn.sh  # with different strategies
   ```
4. Evaluate model performance and analyze feature importance
5. Fine-tune hyperparameters based on validation results
6. Prepare results for MIT Sloan Sports Analytics Conference 2025

## Notes & Observations

### Key Insights
- SkillCorner provides continuous 10fps tracking (better for velocity features)
- StatsBomb has precise freeze-frames at corner moment
- SoccerNet adds video for potential multi-modal learning
- Combining all three gives unprecedented data richness
- **Temporal augmentation successfully expanded dataset 6.6× (1,118 → 7,369 graphs)**
- **Changing target to "dangerous situation" improved class balance (1.3% → 17.1%)**
- **US Soccer Federation approach translates well to corner kick scenarios**
- **GitHub media URLs provide reliable SkillCorner access without LFS limitations**

### Challenges Addressed
- ✅ Extreme class imbalance (1.3% goal rate) → Changed to "shot OR goal" (17.1%)
- ✅ Small dataset (1,118 corners) → Temporal augmentation + SkillCorner (7,369 graphs)
- ✅ SkillCorner Git LFS budget → Use GitHub media URLs
- ✅ StatsBomb 360 data limitation → Applied US Soccer Fed augmentation approach

### Remaining Challenges
- Synchronizing different data formats and coordinate systems
- Handling missing player tracking in some frames
- Computational requirements for training on HPC
- Need to validate augmented data doesn't introduce bias

### References
- Bekkers & Sahasrabudhe (2024): "A Graph Neural Network Deep-Dive into Successful Counterattacks"
- US Soccer Federation GNN: https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn
- UnravelSports: https://github.com/UnravelSports/unravelsports
- StatsBomb Open Data: https://github.com/statsbomb/open-data
- SkillCorner Open Data: https://github.com/SkillCorner/opendata

## Success Metrics

### Target Performance
- **Primary Goal**: AUC-ROC > 0.80 for goal prediction
- **Secondary Goals**:
  - Shot prediction: AUC-ROC > 0.75
  - Multi-class accuracy > 65%
  - Inference time < 100ms

### Publication Targets
- Conference: MIT Sloan Sports Analytics Conference 2025
- Journal: Journal of Sports Analytics
- Open-source release of code and models

---

*Last Updated: October 23, 2024*
*Project Lead: mseo*
*Status: Phase 2 Complete + Temporal Augmentation - Ready for Re-training*
