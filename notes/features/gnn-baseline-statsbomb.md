# GNN Baseline on StatsBomb Data

## Feature Overview

Implement a Graph Neural Network approach for corner kick outcome prediction using StatsBomb 360° freeze-frame data, following TacticAI-style methodology.

## Data Understanding

- **Dataset**: 1,933 corners with freeze frames from StatsBomb 360° data
- **Labels**: Binary shot/no-shot (`shot_outcome`: 1=shot, 0=no-shot)
- **Distribution**: 560 shots (29%), 1373 no-shots (71%)
- **Splits**: Train ~1155, Val ~371, Test ~407 (80/10/10)

### Freeze Frame Structure

Each corner has a freeze frame with ~18-22 players:
- `location`: [x, y] in 120×80 pitch coordinates
- `teammate`: Boolean (True = attacking team)
- `keeper`: Boolean (True = goalkeeper)
- `actor`: Boolean (True = corner taker)

## Implementation Plan

### 1. Graph Construction (`graph_construction.py`)

**Node Features** (per player):
- Position (x, y)
- Team indicator (0/1)
- Distance to goal (Euclidean to [120, 40])
- Distance to ball (corner location)
- Keeper indicator (0/1)

**Edge Construction Options**:
- k-NN within teams + cross-team marking edges
- Full connectivity (all-to-all)
- Delaunay triangulation

**Edge Features**:
- Euclidean distance between players
- Angle to goal

### 2. Models (`models.py`)

Implement three GNN architectures:
1. **GAT** (Graph Attention Network)
2. **GraphSAGE**
3. **MPNN** (Message Passing NN)

All with:
- Global mean/max pooling for graph-level prediction
- Binary classification head
- Dropout for regularization

### 3. Training (`train.py`)

- Loss: Binary cross-entropy (with class weights for imbalance)
- Optimizer: Adam
- Early stopping on validation AUC

### 4. Evaluation (`evaluate.py`)

- Primary metric: AUC-ROC
- Bootstrap 95% confidence intervals
- Permutation test vs random baseline

## Success Criteria

- AUC > 0.55 with statistical significance → Found signal (challenges "unpredictable" claim)
- AUC ≈ 0.50 → Strengthens claim with better methodology

## Progress Log

- [2024-01-05] Feature branch created, data structure understood
