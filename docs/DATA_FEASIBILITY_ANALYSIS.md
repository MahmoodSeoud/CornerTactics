# Data Feasibility Analysis for GNN Implementation

**Date**: October 23, 2025
**Question**: Can we proceed with Phase 2.2 (Graph Construction) and Phase 3 (GNN Training) with the current data?

---

## TL;DR: YES, but with StatsBomb only (for now)

**✅ We CAN build and train the GNN** using StatsBomb 360° data (1,118 corners, 21,231 player instances).

**⚠️ We CANNOT use SkillCorner tracking** data because we only have Git LFS pointer files, not the actual tracking data.

**🤔 We COULD potentially use SoccerNet** tracking data (52GB available), but format is different and requires investigation.

---

## Current Data Inventory

### ✅ StatsBomb 360° - READY TO USE

**Status**: ✅ Fully available and extracted
**Source**: StatsBomb Open Data
**Size**: 1,118 corners from major competitions

**What we have**:
- Player positions (x, y) at corner moment (freeze frame)
- Team affiliations (attacking/defending)
- Corner outcomes (Goal/Shot/Clearance/Loss/Possession)
- Ball trajectory (start location → end location)
- Player features extracted (21,231 player instances)

**What we're missing**:
- Velocity data (vx, vy) - freeze frame only, no motion
- Continuous tracking over time

**Node features extracted** (14 dimensions):
```
✅ x, y (position)
✅ distance_to_goal
✅ distance_to_ball_target
❌ vx, vy (zero - no motion data)
❌ velocity_magnitude (zero)
❌ velocity_angle (zero)
✅ angle_to_goal
✅ angle_to_ball
✅ team_flag
✅ in_penalty_box
✅ num_players_within_5m
✅ local_density_score
```

**Verdict**: **Sufficient for GNN training**. We can use 10/14 features (the 4 velocity features will be zero).

---

### ❌ SkillCorner Tracking - NOT AVAILABLE

**Status**: ❌ Git LFS pointer files only
**Issue**: Tracking JSONL files are stored with Git LFS but not downloaded

**What we see**:
```bash
$ ls -lh data/datasets/skillcorner/data/matches/*/\*tracking*.jsonl
-rw-rw-r-- 1 mseo mseo 133 Oct 22 14:32 1886347_tracking_extrapolated.jsonl
```

**What's in the file** (Git LFS pointer):
```
version https://git-lfs.github.com/spec/v1
oid sha256:3577e2803da95390f8b2f85d47829eb55fbbfee6211c94adc65db0dc78e46b41
size 89280839
```

The actual file should be 89MB, but we only have a 133-byte pointer.

**What we DO have from SkillCorner**:
- ✅ Dynamic events CSV (has data)
- ✅ Phases of play CSV (has data)
- ✅ Match metadata JSON (has data)
- ❌ Tracking JSONL (Git LFS pointer only)

**To get the tracking data**, you would need to:
```bash
cd data/datasets/skillcorner
git lfs pull
```

**Verdict**: **Not usable without Git LFS pull**. Dynamic events might be useful but don't contain player positions.

---

### 🤔 SoccerNet Tracking - AVAILABLE but Different Format

**Status**: ⚠️ Available (52GB) but in different format
**Source**: SoccerNet SNMOT (Soccer Network Multiple Object Tracking)

**What we have**:
- 52GB of tracking data
- MOT (Multiple Object Tracking) format
- Image sequences and detection/tracking annotations

**Format**:
```
data/datasets/soccernet/tracking/train/SNMOT-060/
├── det/           # Detections
├── gt/            # Ground truth tracking
├── img1/          # Video frames
├── gameinfo.ini   # Game metadata
└── seqinfo.ini    # Sequence info
```

**Challenges**:
1. **Different format**: Not same as SkillCorner/StatsBomb coordinate systems
2. **Video-focused**: Designed for multi-object tracking in videos
3. **Requires parsing**: Need to extract player positions from MOT format
4. **Alignment needed**: Need to synchronize with corner kick moments
5. **No direct corner labels**: Would need to identify corner events from video/tracking

**Potential**: Could be used, but requires significant additional work to:
- Parse MOT format
- Extract player positions
- Identify corner kick events
- Convert to same coordinate system as StatsBomb

**Verdict**: **Possible but significant work required**. Better to focus on StatsBomb first.

---

## Comparison with USSF GNN Repository

### What USSF Used

**Data sources** (from their README):
- StatsPerform on-ball event data
- SkillCorner spatiotemporal tracking data
- 632 matches (MLS 2022, NWSL 2022, International women's)
- 20,863 counterattack frames (balanced dataset)

**Their node features** (12 dimensions):
```
1. x, y (position)
2. vx, vy (velocity)
3. distance_to_goal
4. angle_to_goal
5. distance_to_ball
6. angle_to_ball
7. team_flag
8. receiver_indicator
```

**Their data format**:
- Pre-processed pickle files with graphs
- Adjacency matrices already constructed
- Node and edge features pre-computed

### What We Have

**Data sources**:
- StatsBomb 360° (freeze frames)
- SkillCorner (Git LFS pointers only)
- SoccerNet (different format)
- 1,118 corners from major competitions

**Our node features** (14 dimensions):
```
1-2. x, y ✅
3-6. vx, vy, velocity_magnitude, velocity_angle ❌ (zero for StatsBomb)
7. distance_to_goal ✅
8. angle_to_goal ✅
9. distance_to_ball_target ✅
10. angle_to_ball ✅
11. team_flag ✅
12. in_penalty_box ✅
13. num_players_within_5m ✅
14. local_density_score ✅
```

**Key differences**:
- ❌ No velocity features (StatsBomb is static)
- ✅ More spatial features (density, penalty box)
- ✅ More outcomes (5 classes vs binary)
- ✅ Corner-specific context (not counterattacks)

---

## Can We Build a GNN? YES!

### Approach 1: StatsBomb Only (RECOMMENDED)

**Use StatsBomb 360° data exclusively**

**Pros**:
- ✅ Data ready to use (21,231 player instances extracted)
- ✅ High-quality freeze frames at corner moment
- ✅ Rich outcome labels (Goal/Shot/Clearance/Loss/Possession)
- ✅ Sufficient spatial features for GNN
- ✅ Can proceed immediately to Phase 2.2

**Cons**:
- ❌ No velocity features (4/14 features will be zero)
- ❌ Single time point (no temporal dynamics)
- ❌ Smaller dataset (1,118 vs USSF's 20,863)

**GNN Architecture Adaptations**:
- Use 10 effective node features (drop velocity features)
- OR keep all 14 features (velocities = 0) and let model learn
- Focus on spatial relationships over temporal dynamics
- Adjacency matrix: distance-based, Delaunay, team-based (no velocity-based)

**Expected Performance**:
- Should still work well - spatial positioning is key for corners
- Velocity less important for corners than counterattacks
- Similar to analyzing chess positions (static) vs chess games (dynamic)

---

### Approach 2: Download SkillCorner Tracking (FUTURE)

**Pull Git LFS files to get tracking data**

**Steps**:
```bash
cd data/datasets/skillcorner
git lfs install
git lfs pull
```

**This would give us**:
- 10 matches with full tracking (10 fps)
- Player positions over time
- Velocity calculations possible
- ~317 corners with tracking data

**Benefits**:
- ✅ Full 14-feature vectors with real velocities
- ✅ Can validate velocity features matter
- ✅ Temporal dynamics for corner sequences
- ✅ Can compare static vs dynamic models

**Cost**:
- ⚠️ Large download (~890MB per match = ~9GB total)
- ⚠️ Requires Git LFS setup
- ⚠️ Additional preprocessing needed

---

### Approach 3: Integrate SoccerNet (RESEARCH PROJECT)

**Parse SoccerNet MOT tracking data**

**Would require**:
1. Parse MOT format (bounding boxes → player positions)
2. Identify corner kick events from video/annotations
3. Convert coordinates to standard pitch coordinates
4. Align with corner kick timing
5. Extract player positions at corner moments

**Benefits**:
- ✅ Huge dataset (52GB of tracking)
- ✅ Could complement StatsBomb

**Challenges**:
- ⏰ Significant engineering effort (weeks of work)
- 🔧 Format conversion complexity
- 📊 No corner labels (need to detect from video)

**Verdict**: **Research project on its own** - not feasible for immediate GNN training.

---

## Recommendation: Proceed with StatsBomb Only

### Phase 2.2: Graph Construction (StatsBomb)

**What to build**:
1. **Adjacency matrices** (5 types):
   - Team-based (teammates only)
   - Distance-based (within 10m threshold)
   - Delaunay triangulation
   - Ball-centric (near ball trajectory)
   - Zone-based (tactical zones)

2. **Edge features** (6 dimensions):
   - Player-to-player distance
   - Angle difference (sin/cos pairs)
   - ~~Speed difference~~ (skip - no velocities)
   - Positional relationship features

3. **Graph format**:
   - Convert to PyTorch Geometric `Data` objects
   - Node features: 10 effective dimensions (or 14 with zeros)
   - Edge features: 4-6 dimensions
   - Labels: 5-class outcome

### Phase 3: GNN Training (StatsBomb)

**Model architecture** (adapted from USSF):
```python
- GraphConv1: 10 → 64 (ReLU, Dropout=0.3)
- GraphConv2: 64 → 128 (ReLU, Dropout=0.3)
- GraphConv3: 128 → 64 (ReLU)
- Global Pooling: Mean + Max
- Dense1: 128 → 64 (ReLU)
- Dense2: 64 → 32 (ReLU)
- Output: 32 → 5 (Softmax for multi-class)
```

**Training setup**:
- Train/val/test split: 800/150/168 corners
- Loss: CrossEntropyLoss (multi-class)
- Optimizer: Adam with cosine annealing
- Metrics: Accuracy, AUC-ROC (per class), F1-score

**Expected outcomes**:
- Demonstrate GNN approach works for corners
- Baseline performance with static features
- Foundation for future velocity-enhanced models

---

## Future Enhancements

### Option 1: Add SkillCorner Tracking (Easy)
```bash
# Just pull Git LFS
git lfs pull
```
- Adds 317 corners with velocities
- Validates if velocity features matter
- ~1 hour of work to integrate

### Option 2: Hybrid Model (Medium)
- Use StatsBomb for majority (1,118 corners)
- Use SkillCorner subset (317 corners) for velocity validation
- Train separate models and compare
- ~1 week of work

### Option 3: SoccerNet Integration (Hard)
- Parse MOT format
- Detect corner events
- Massive dataset expansion
- ~1 month of work

---

## Bottom Line

### ✅ YES - We can proceed with the GNN implementation

**Current capabilities**:
- **Phase 2.1**: ✅ COMPLETE (node features extracted)
- **Phase 2.2**: ✅ READY (can build graphs from StatsBomb)
- **Phase 3**: ✅ READY (can train GNN on StatsBomb)

**What we'll build**:
- Spatial-focused GNN for corner kick outcome prediction
- 1,118 corners with 10-14 node features
- 5-class multi-class prediction
- Multiple adjacency matrix types

**What we won't have** (initially):
- Velocity-based features
- Temporal dynamics
- Large-scale dataset (thousands of corners)

**Is this enough for research?**
- ✅ YES for proof-of-concept
- ✅ YES for demonstrating GNN approach on corners
- ✅ YES for baseline model
- ⚠️ Maybe not for publication without velocity comparison

**Recommendation**: **Proceed with StatsBomb-only implementation**, get a working GNN, then decide if downloading SkillCorner tracking is worth it for the velocity features.

---

*Analysis Date: October 23, 2025*
*Conclusion: PROCEED with Phase 2.2 using StatsBomb 360° data*
