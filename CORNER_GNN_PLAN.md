# Corner Kick GNN Prediction System - Implementation Plan

## Overview
Implementing a Graph Neural Network (GNN) based corner kick outcome prediction system using the methodology from Bekkers & Sahasrabudhe (2024) "A Graph Neural Network Deep-Dive into Successful Counterattacks".

## Data Inventory

### Available Datasets
- [ ] **StatsBomb 360**: 1,118 corners with freeze-frame player positions
  - Location: `data/statsbomb/corners_360.csv`
  - Status: ✅ Downloaded and ready
  - Features: Player positions (JSON), corner location, pass end location

- [ ] **SkillCorner Open Data**: 10 A-League matches with continuous tracking
  - Location: `data/skillcorner/`
  - Status: ✅ Downloaded
  - Features: 10fps tracking, dynamic events, phases of play
  - Corner events: ~30 per match (estimated ~300 total)

- [ ] **SoccerNet**: Video clips and tracking data
  - Location: `data/datasets/soccernet/`
  - Status: ✅ Downloaded
  - Corner clips: 4,208 videos in `corner_clips/visible/`
  - Tracking data: Available in `tracking/`

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
- [ ] Spatial features
  - [ ] x, y coordinates (normalized to pitch dimensions)
  - [ ] distance_to_goal
  - [ ] distance_to_ball_target

- [ ] Kinematic features (from SkillCorner 10fps)
  - [ ] velocity_x (vx)
  - [ ] velocity_y (vy)
  - [ ] velocity_magnitude
  - [ ] velocity_angle

- [ ] Contextual features
  - [ ] angle_to_goal
  - [ ] angle_to_ball
  - [ ] team_flag (attacking/defending)
  - [ ] in_penalty_box flag

- [ ] Density features
  - [ ] num_players_within_5m
  - [ ] local_density_score

### 2.2 Adjacency Matrix Construction
- [ ] Implement 5 connection strategies
  - [ ] Team-based: Connect teammates only
  - [ ] Distance-based: Connect players within 10m threshold
  - [ ] Delaunay: Triangulation for spatial relationships
  - [ ] Ball-centric: Connect players near ball trajectory
  - [ ] Zone-based: Connect players in same tactical zone

### 2.3 Edge Features (6 dimensions)
- [ ] Distance between connected players
- [ ] Relative velocity
- [ ] Angle differences (sine/cosine pairs)

## Phase 3: GNN Model Implementation 🤖

### 3.1 Core Architecture (Spektral-based)
- [ ] Set up environment
  - [ ] Install spektral==1.2.0, tensorflow==2.14.0
  - [ ] Install unravelsports package

- [ ] Implement model layers
  ```python
  - [ ] GraphConv1: 12 → 64 (ReLU, Dropout=0.3)
  - [ ] GraphConv2: 64 → 128 (ReLU, Dropout=0.3)
  - [ ] GraphConv3: 128 → 64 (ReLU)
  - [ ] Global Pooling: Mean + Max concatenation
  - [ ] Dense1: 128 → 64 (ReLU)
  - [ ] Dense2: 64 → 32 (ReLU)
  - [ ] Output: 32 → 1 (Sigmoid for binary)
  ```

### 3.2 Multi-Modal Extension (Optional)
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

## Phase 5: Implementation Files 📁

### Core Modules (`src/`)
- [ ] `data_integration.py` - Merge all data sources
- [ ] `graph_builder.py` - Convert tracking to graphs
- [ ] `gnn_model.py` - Spektral GNN architecture
- [ ] `feature_engineering.py` - Node/edge feature extraction
- [ ] `outcome_labeler.py` - Label corner outcomes
- [ ] `train_gnn.py` - Training pipeline

### Scripts (`scripts/`)
- [ ] `integrate_datasets.py` - Combine StatsBomb + SkillCorner + SoccerNet
- [ ] `extract_skillcorner_corners.py` - Parse SkillCorner corners
- [ ] `build_graph_dataset.py` - Create graph dataset
- [ ] `evaluate_model.py` - Model evaluation
- [ ] `predict_corner.py` - Inference on new corners

### SLURM Jobs (`scripts/slurm/`)
- [ ] `integrate_data.sh` - Data integration job
- [ ] `train_corner_gnn.sh` - GNN training job
- [ ] `evaluate_gnn.sh` - Evaluation job
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
- [x] Identified UnravelSports and US Soccer Federation codebases

### In Progress ⏳
- [ ] Data integration pipeline
- [ ] Feature engineering

### Next Steps 📝
1. Start with SkillCorner corner extraction script
2. Implement StatsBomb outcome labeling
3. Create first version of graph builder
4. Set up Spektral environment on HPC

## Notes & Observations

### Key Insights
- SkillCorner provides continuous 10fps tracking (better for velocity features)
- StatsBomb has precise freeze-frames at corner moment
- SoccerNet adds video for potential multi-modal learning
- Combining all three gives unprecedented data richness

### Challenges to Address
- Synchronizing different data formats and coordinate systems
- Handling missing player tracking in some frames
- Balancing dataset sizes from different sources
- Computational requirements for training on HPC

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

*Last Updated: October 22, 2024*
*Project Lead: mseo*
*Status: Planning Phase*
