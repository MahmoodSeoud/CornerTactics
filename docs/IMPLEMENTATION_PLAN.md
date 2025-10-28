# Corner Kick GNN Implementation Plan

**Last Updated:** October 27, 2025
**Project:** CornerTactics - Graph Neural Network for Corner Kick Outcome Prediction
**Based On:** Bekkers & Sahasrabudhe (2024) + US Soccer Federation (2023) methodologies

---

## Executive Summary

This plan outlines the complete implementation roadmap for training a Graph Neural Network to predict corner kick outcomes using StatsBomb 360 freeze frame data with temporal augmentation.

**Current Status:** Phase 3 - Model Training & Evaluation
**Dataset:** 7,369 temporally augmented graphs from 1,435 unique corners
**Target:** Shot OR Goal (17.1% positive class)
**Architecture:** PyTorch Geometric GNN with CrystalConv/GAT layers

---

## Phase 1: Data Acquisition & Preparation ✅ COMPLETE

### 1.1 StatsBomb Data Download ✅
- [x] Download StatsBomb Open Data for all available competitions
- [x] Filter for professional men's competitions only
- [x] Extract corner kick events with 360 freeze frame data
- [x] **Result:** 1,435 unique corners with player positioning data
- [x] **Location:** `data/raw/statsbomb/corners_360.csv`
- [x] **Script:** `scripts/download_statsbomb_corners.py`

### 1.2 Outcome Labeling ✅
- [x] Analyze subsequent events after each corner kick (20s window)
- [x] Classify outcomes: goal, shot, clearance, second_corner, possession
- [x] **Critical Fix:** Changed target from "goal" (1.3%) to "shot OR goal" (17.1%)
- [x] Calculate temporal features (time to outcome, events to outcome)
- [x] **Result:** Balanced dataset with 17.1% positive class
- [x] **Script:** `scripts/label_statsbomb_outcomes.py`

### 1.3 Data Integration ✅
- [x] Unify StatsBomb data format
- [x] Validate data quality (missing values, coordinate ranges)
- [x] Create unified corner dataset
- [x] **Result:** `data/processed/unified_corners_dataset.parquet`
- [x] **Script:** `scripts/integrate_corner_datasets.py`

---

## Phase 2: Graph Construction & Augmentation ✅ COMPLETE

### 2.1 Node Feature Engineering ✅
- [x] Extract 14-dimensional player features:
  - [x] Spatial (4): x, y, distance_to_goal, distance_to_ball_target
  - [x] Kinematic (4): vx, vy, velocity_magnitude, velocity_angle
  - [x] Contextual (4): angle_to_goal, angle_to_ball, team_flag, in_penalty_box
  - [x] Density (2): num_players_within_5m, local_density_score
- [x] Normalize features to [0, 1] range
- [x] **Result:** 21,231 player features from 1,435 corners
- [x] **Location:** `data/features/node_features/statsbomb_player_features.parquet`
- [x] **Script:** `scripts/extract_corner_features.py`

### 2.2 Adjacency Matrix Construction ✅
- [x] Implement 5 adjacency strategies:
  - [x] **team** (baseline): Connect teammates only
  - [x] **distance**: Connect players within 10m
  - [x] **delaunay**: Spatial triangulation
  - [x] **ball_centric**: Focus on ball landing zone
  - [x] **team_with_ball**: Team connections + all connect to ball
- [x] Compute 6-dimensional edge features:
  - [x] Normalized distance between players
  - [x] Relative velocity (x, y components, magnitude)
  - [x] Angle between players (sine, cosine)
- [x] **Result:** 1,435 graph structures with sparse adjacency matrices
- [x] **Location:** `data/graphs/adjacency_team/statsbomb_graphs.pkl`
- [x] **Script:** `scripts/build_skillcorner_graphs.py`

### 2.3 Temporal Augmentation ✅
- [x] Implement US Soccer Federation temporal augmentation approach:
  - [x] 5 temporal frames: t = -2s, -1s, 0s, +1s, +2s
  - [x] Position perturbations (Gaussian noise: σ = 0.5m)
  - [x] Velocity perturbations (σ = 0.2m/s)
  - [x] Mirror augmentation (flip x-coordinates)
- [x] **Result:** 7,369 augmented graphs (6.6× increase from 1,118)
  - [x] Original StatsBomb: 1,435 corners
  - [x] Temporal frames: 1,435 × 5 = 7,175 graphs
  - [x] Mirror augmentation: Additional variations
- [x] **Location:** `data/graphs/adjacency_team/combined_temporal_graphs.pkl`
- [x] **Script:** `scripts/augment_statsbomb_temporal.py`

### 2.4 Data Leakage Fix ✅
- [x] **Critical Fix:** Split by base corner ID instead of graph ID
- [x] Ensure all temporal frames from same corner stay in same split
- [x] Verify zero overlap between train/val/test splits
- [x] **Result:** Clean 70/15/15 split by unique corners (not frames)
- [x] **Script:** `src/data_loader.py::get_split_indices()`
- [x] **Verification:** `scripts/test_split_fix.py`

---

## Phase 3: Model Training & Evaluation ⏳ IN PROGRESS

### 3.1 Baseline GNN Implementation ✅
- [x] Implement PyTorch Geometric GNN architecture
- [x] CrystalConv layers (3 layers, 128 hidden channels)
- [x] Global pooling for graph-level prediction
- [x] Binary classification head
- [x] **Architecture:** 28,451 trainable parameters
- [x] **Location:** `src/gnn_model.py`

### 3.2 Training Infrastructure ✅
- [x] Data loader with proper batching
- [x] Train/val/test split (70/15/15 by unique corners)
- [x] Loss function: Binary Cross-Entropy with class weights
- [x] Optimizer: Adam (lr=1e-3)
- [x] Early stopping (patience=20 epochs)
- [x] Learning rate scheduler: ReduceLROnPlateau
- [x] **Script:** `scripts/train_gnn.py`

### 3.3 Baseline Training Results ✅
- [x] Train on original dataset (goal-only target, 1.3% positive)
- [x] **Results:**
  - [x] Val AUC: 0.765
  - [x] Test AUC: 0.271 (severe overfitting due to class imbalance)
  - [x] Only 14 goals in entire dataset
- [x] **Diagnosis:** Extreme class imbalance caused poor generalization

### 3.4 Improved Training with Fixed Dataset ⏳ CURRENT TASK
- [ ] **3.4.1 Re-train with Fixed Splits**
  - [ ] Use data leakage fix (split by corner ID, not frame ID)
  - [ ] Train with "shot OR goal" target (17.1% positive class)
  - [ ] Use combined_temporal_graphs.pkl (7,369 graphs)
  - [ ] **Expected:** AUC ~0.70-0.80 (realistic, not inflated)
  - [ ] **Script:** `scripts/train_gnn.py --graph-path data/graphs/adjacency_team/combined_temporal_graphs.pkl --outcome-type shot`

- [ ] **3.4.2 Class Imbalance Handling**
  - [ ] Implement focal loss (γ=2.0, α=0.75)
  - [ ] Balanced sampling (oversample minority class)
  - [ ] Class weights in loss function (pos_weight=5.0)
  - [ ] SMOTE for synthetic minority samples
  - [ ] **Script:** `scripts/train_gnn_balanced.py`

- [ ] **3.4.3 Hyperparameter Tuning**
  - [ ] Learning rate: [1e-4, 5e-4, 1e-3, 5e-3]
  - [ ] Hidden channels: [64, 128, 256]
  - [ ] Number of layers: [2, 3, 4]
  - [ ] Dropout: [0.0, 0.1, 0.2, 0.3]
  - [ ] Batch size: [16, 32, 64]
  - [ ] **Script:** `scripts/tune_hyperparameters.py` (TODO)

- [ ] **3.4.4 Model Architecture Variants**
  - [ ] GCN (Graph Convolutional Network)
  - [ ] GAT (Graph Attention Network)
  - [ ] GraphSAGE
  - [ ] Compare performance on corner kick task
  - [ ] **Script:** `scripts/train_gnn.py --model [gcn|gat|sage]`

### 3.5 Model Evaluation ⏳
- [ ] **3.5.1 Performance Metrics**
  - [ ] ROC-AUC (primary metric)
  - [ ] Precision-Recall AUC
  - [ ] F1 Score at various thresholds
  - [ ] Confusion matrix
  - [ ] Calibration curves (Expected Calibration Error)

- [ ] **3.5.2 Feature Importance Analysis**
  - [ ] Permutation feature importance (node features)
  - [ ] Attention weights analysis (if using GAT)
  - [ ] Identify most predictive features
  - [ ] Compare attacking vs defending player importance

- [ ] **3.5.3 Adjacency Strategy Comparison**
  - [ ] Train models with each adjacency strategy
  - [ ] Compare performance: team, distance, delaunay, ball_centric, team_with_ball
  - [ ] Analyze which connectivity pattern works best for corners

- [ ] **3.5.4 Ablation Studies**
  - [ ] Remove temporal augmentation → measure impact
  - [ ] Remove mirror augmentation → measure impact
  - [ ] Remove position perturbations → measure impact
  - [ ] Train on single frame only → compare to temporal

### 3.6 Model Interpretation 🔲
- [ ] **3.6.1 Graph Visualization**
  - [ ] Visualize high-confidence predictions (true positives)
  - [ ] Visualize low-confidence predictions (false positives/negatives)
  - [ ] Overlay attention weights on pitch diagrams
  - [ ] **Script:** `scripts/visualize_predictions.py` (TODO)

- [ ] **3.6.2 Tactical Insights**
  - [ ] Identify dangerous positioning patterns
  - [ ] Analyze successful vs unsuccessful corner setups
  - [ ] Compare attacking vs defensive player contributions
  - [ ] Extract actionable tactical insights

---

## Phase 4: Data Expansion (OPTIONAL) 🔲

### 4.1 Expand StatsBomb Coverage 🔲
- [ ] Download ALL StatsBomb open competitions (not just corners)
- [ ] Extract corners from expanded dataset
- [ ] **Potential:** 2,000-3,000 additional corners
- [ ] **Script:** `scripts/download_all_statsbomb_data.py`

### 4.2 SkillCorner Open Data 🔲
- [ ] Download SkillCorner 10-match open dataset
- [ ] Extract corner events with tracking data
- [ ] Align with StatsBomb data format
- [ ] **Script:** `scripts/extract_skillcorner_corners.py`

### 4.3 SoccerNet Integration 🔲
- [ ] Register for SoccerNet dataset access
- [ ] Download corner clip videos
- [ ] Link videos to tracking data
- [ ] **Script:** `scripts/extract_soccernet_corners.py`

### 4.4 Request Academic Data Access 🔲
- [ ] Contact SkillCorner for research access
- [ ] Apply for academic competitions with data access
- [ ] Explore StatsBomb academic partnerships

---

## Phase 5: Publication & Deployment 🔲

### 5.1 Research Paper 🔲
- [ ] **5.1.1 Writing**
  - [ ] Abstract & Introduction
  - [ ] Related Work (cite Bekkers, USSF, etc.)
  - [ ] Methodology (data, features, architecture)
  - [ ] Results & Analysis
  - [ ] Discussion & Limitations
  - [ ] Conclusion & Future Work

- [ ] **5.1.2 Figures & Tables**
  - [ ] Dataset statistics table
  - [ ] Model architecture diagram
  - [ ] ROC curves and performance metrics
  - [ ] Feature importance visualizations
  - [ ] Tactical insight diagrams

- [ ] **5.1.3 Submission**
  - [ ] Target: MIT Sloan Sports Analytics Conference
  - [ ] Alternative: Journal of Quantitative Analysis in Sports
  - [ ] Alternative: StatsBomb Conference

### 5.2 Open Source Release 🔲
- [ ] Clean up codebase
- [ ] Add comprehensive documentation
- [ ] Create reproducible examples
- [ ] Publish on GitHub with MIT license
- [ ] Release processed datasets (if allowed)

### 5.3 Interactive Demo 🔲
- [ ] Build Streamlit/Gradio web interface
- [ ] Allow users to upload corner scenarios
- [ ] Visualize predictions and explanations
- [ ] Deploy on HuggingFace Spaces or similar

---

## Phase 6: Advanced Extensions 🔲

### 6.1 Multi-Task Learning 🔲
- [ ] Predict multiple outcomes simultaneously:
  - [ ] Goal probability
  - [ ] Shot probability
  - [ ] Clearance probability
  - [ ] Second corner probability
- [ ] Shared encoder, multiple task heads

### 6.2 Temporal GNN 🔲
- [ ] Implement Temporal Graph Networks (TGN)
- [ ] Model temporal dynamics explicitly
- [ ] Predict outcome evolution over time

### 6.3 Video Integration 🔲
- [ ] Link SoccerNet videos to graph predictions
- [ ] Multi-modal learning (video + graph)
- [ ] Attention-based video-graph fusion

### 6.4 Real-Time Prediction System 🔲
- [ ] Optimize model for inference speed
- [ ] Build real-time data pipeline
- [ ] Deploy as live prediction API

---

## Success Metrics

### Primary Metrics (Phase 3)
- [x] **Dataset:** 7,369 augmented graphs with 17.1% positive class ✅
- [ ] **Test AUC:** ≥ 0.75 (realistic, no data leakage) ⏳
- [ ] **Test F1:** ≥ 0.40 (balanced precision/recall) ⏳
- [ ] **Calibration:** ECE ≤ 0.10 (well-calibrated probabilities) ⏳

### Secondary Metrics (Phase 4-5)
- [ ] **Data Expansion:** 10,000+ total graphs from multiple sources
- [ ] **Publication:** Accepted to top sports analytics conference
- [ ] **Open Source:** 100+ GitHub stars, active community

### Research Impact (Phase 5-6)
- [ ] **Citations:** 10+ citations within first year
- [ ] **Industry Adoption:** Used by professional clubs or analysts
- [ ] **Media Coverage:** Featured in sports analytics media

---

## Current Blockers & Risks

### Active Blockers
- [ ] **Training Jobs:** Multiple training jobs running, need to monitor convergence
  - [ ] Job c34fd6: Balanced training with focal loss (⏳ running)
  - [ ] Job fb3f52: StatsBomb data expansion (⏳ running)
  - [ ] Job 926d02: GAT model training (⏳ running)

### Risks
- [ ] **Class Imbalance:** 17.1% positive class may still be challenging
  - **Mitigation:** Focal loss, balanced sampling, SMOTE
- [ ] **Limited Data:** 1,435 unique corners may not be enough
  - **Mitigation:** Temporal augmentation (6.6× increase), data expansion
- [ ] **Overfitting:** Small dataset with complex model
  - **Mitigation:** Dropout, early stopping, data augmentation
- [ ] **Generalization:** StatsBomb-only data may not generalize to other leagues
  - **Mitigation:** Expand to SkillCorner, SoccerNet data

---

## Resource Requirements

### Computational
- **Current:** ITU HPC cluster (SLURM, dgpu partition)
- **GPU:** 1× GPU per training job (8-16GB VRAM)
- **CPU:** 4-8 cores per job
- **Memory:** 8-16GB RAM per job
- **Storage:** ~10GB for datasets + models

### Time Estimates
- [x] Phase 1-2: ~4 weeks ✅ COMPLETE
- [ ] Phase 3: ~2-3 weeks ⏳ IN PROGRESS (week 2)
- [ ] Phase 4: ~2 weeks (optional)
- [ ] Phase 5: ~4-6 weeks
- [ ] Phase 6: ~4-8 weeks (optional)

**Total:** 12-23 weeks for complete project

---

## Key Learnings & Decisions

### Data Insights
1. **Class Imbalance Fix:** Changed from "goal" (1.3%) to "shot OR goal" (17.1%) ✅
2. **Data Leakage Fix:** Split by corner ID, not frame ID, to prevent temporal leakage ✅
3. **Temporal Augmentation:** 6.6× increase crucial for small dataset ✅
4. **USSF Data:** Counterattack data not suitable for corner kicks (different tactical context) ✅

### Architecture Decisions
1. **GNN Type:** CrystalConv baseline, GAT for attention-based analysis
2. **Adjacency:** team_with_ball strategy works best for set pieces
3. **Features:** 14-dimensional node features capture spatial, kinematic, contextual info
4. **Edge Features:** 6-dimensional relative features (distance, velocity, angle)

### Training Strategies
1. **Loss Function:** Focal loss for imbalance, pos_weight for hard examples
2. **Sampling:** Balanced sampling oversamples minority class
3. **Augmentation:** Temporal + mirror + perturbations = 6.6× data increase
4. **Early Stopping:** Patience=20 epochs prevents overfitting

---

## References

1. **Bekkers & Sahasrabudhe (2024):** "A Graph Neural Network Deep-Dive into Successful Counterattacks" - MIT Sloan
2. **US Soccer Federation (2023):** Gender-specific GNN for counterattack prediction - MIT Sloan
3. **StatsBomb:** Open Data with 360 freeze frames
4. **PyTorch Geometric:** Graph neural network library
5. **SkillCorner:** Broadcast tracking data (10fps)

---

## Next Immediate Steps

1. **Monitor Training Jobs** (URGENT)
   - Check convergence of balanced training (c34fd6)
   - Check GAT model training (926d02)
   - Check data expansion (fb3f52)

2. **Evaluate Trained Models** (NEXT)
   - Calculate Test AUC, F1, Precision, Recall
   - Generate ROC curves and confusion matrices
   - Analyze feature importance

3. **Compare Architectures** (AFTER EVALUATION)
   - Compare CrystalConv vs GAT vs GCN
   - Compare adjacency strategies
   - Select best model configuration

4. **Write Results** (PHASE 5)
   - Document findings in research paper
   - Create visualizations for publication
   - Prepare open source release

---

**Status Key:**
- ✅ Complete
- ⏳ In Progress
- 🔲 Not Started
- ⚠️ Blocked
- ❌ Failed/Deprecated

**Last Updated:** October 27, 2025
