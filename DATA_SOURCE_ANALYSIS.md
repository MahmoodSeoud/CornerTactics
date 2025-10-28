# Data Source Analysis: Do We Need SkillCorner/SoccerNet?

## Current Data Sources

### 1. StatsBomb Open Data
- **Type**: Event data + 360 freeze frames
- **Coverage**: ~1,118 corners with player positions
- **Temporal**: Single snapshot at corner kick moment
- **Augmentation**: 5 temporal offsets (t = -2s, -1s, 0s, +1s, +2s) + mirrors = 5,814 graphs
- **Key Features**: Player positions (x, y), teammate/opponent flags, keeper flags
- **Receiver Data**: ✅ Available via pass_recipient + Ball Receipt location

### 2. SkillCorner Tracking Data
- **Type**: Continuous tracking data (10 fps)
- **Coverage**: ~317 corners
- **Temporal**: Real continuous tracking
- **Result**: 1,555 temporal graphs
- **Key Features**: Player positions + velocities over time
- **Receiver Data**: ❓ Would need to be added

### 3. SoccerNet
- **Type**: Video footage + action spotting labels
- **Coverage**: Unknown corners
- **Temporal**: Video-based
- **Result**: Not currently processed
- **Receiver Data**: ❓ Would need to be extracted

---

## TacticAI Requirements

For **receiver prediction** task, we need:
1. ✅ Player positions at corner kick moment
2. ✅ Graph structure (adjacency matrix)
3. ✅ Receiver labels (player who receives the ball)
4. ✅ Temporal context (optional but helps)

### StatsBomb 360 Provides:
- ✅ Player positions at corner moment
- ✅ Can build graphs from positions
- ✅ Receiver labels (via Ball Receipt events)
- ✅ Temporal augmentation (simulated via offsets)

### SkillCorner Provides:
- ✅ Continuous tracking (real velocities)
- ✅ More temporal granularity
- ❌ Smaller dataset (317 vs 1,118 corners)
- ❓ Receiver labels need to be added

### SoccerNet Provides:
- ❌ Not currently integrated
- ❓ Would require video processing
- ❓ Receiver labels would need extraction

---

## Recommendation: **StatsBomb-Only Pipeline**

### Reasons:
1. **Sufficient for TacticAI**: StatsBomb 360 freeze frames provide all necessary data
2. **Larger dataset**: 1,118 corners → 5,814 augmented graphs (vs 1,555 from SkillCorner)
3. **Simpler pipeline**: Single data source, no multi-source integration
4. **Receiver labels ready**: Ball Receipt events provide ground truth
5. **Temporal augmentation**: Simulates continuous tracking via time offsets

### What We Lose:
- Real velocity data (SkillCorner provides actual velocities)
- Continuous tracking (10fps real data)
- ~1,555 additional temporal graphs

### What We Gain:
- Simplified codebase (remove 20+ files)
- Single-source pipeline (easier maintenance)
- Larger base dataset (1,118 vs 317 corners)
- Clear data provenance (all from StatsBomb)

---

## Files to Remove (StatsBomb-Only)

### SkillCorner Files (11 files)
```
scripts/extract_skillcorner_corners.py
scripts/extract_skillcorner_temporal.py
scripts/build_skillcorner_graphs.py
scripts/label_skillcorner_outcomes.py
scripts/slurm/phase2_3_skillcorner_temporal.sh
scripts/slurm/phase2_5_build_skillcorner_graphs.sh
src/feature_engineering.py - Remove SkillCorner methods
src/outcome_labeler.py - Remove SkillCornerOutcomeLabeler class
```

### SoccerNet Files (2 files)
```
scripts/extract_soccernet_corners.py
scripts/label_soccernet_outcomes.py
```

### Multi-Source Integration (1 file)
```
scripts/integrate_corner_datasets.py
scripts/slurm/phase1_1_complete.sh
```

---

## Updated Pipeline (StatsBomb-Only)

### Phase 1: Data Acquisition
1. `download_statsbomb_corners.py` → Download corners with receiver info
2. `label_statsbomb_outcomes.py` → Label outcomes (goal/shot/clearance)

### Phase 2: Graph Construction
1. `extract_corner_features.py` → Extract 14-dim node features
2. `augment_statsbomb_temporal.py` → Create temporal augmentation (5 frames)
3. Build graphs with adjacency matrices

### Phase 3: Receiver Prediction (TacticAI Day 3-4)
1. `preprocessing/add_receiver_labels.py` → Map receivers to nodes
2. `data/receiver_data_loader.py` → Load with receiver labels
3. Train GNN for receiver prediction

---

## Decision

**YES - Remove SkillCorner and SoccerNet entirely.**

Use **StatsBomb-only pipeline** with temporal augmentation for TacticAI receiver prediction task.
