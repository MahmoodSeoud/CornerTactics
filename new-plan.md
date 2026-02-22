# TASK: USSF-Aligned Features + Joint Pretraining — CornerTactics

## Context

Master's thesis on corner kick shot prediction. Defense: **March 16, 2026**. Current pipeline uses a pretrained USSF backbone with learned adapter layers (node_proj, edge_proj) to bridge the feature gap between our corner data and the backbone's expected input. This wastes trainable parameters on a feature mapping that can be computed deterministically. We're also not making optimal use of the 20,863 USSF open-play graphs available for pretraining.

**Two experiments, in order of priority:**

1. **Experiment A (MUST DO):** Replace learned adapters with deterministic USSF-aligned feature engineering
2. **Experiment B (DO IF TIME):** Joint pretraining on 20,863 open-play graphs + 143 corner graphs

## Repository Layout

```
/home/mseo/CornerTactics/
├── corner_prediction/
│   ├── data/
│   │   ├── extract_corners.py      # Raw data → corner records
│   │   ├── build_graphs.py         # Corner records → PyG Data objects (MODIFY THIS)
│   │   ├── dataset.py              # CornerKickDataset class
│   │   ├── extracted_corners.pkl   # 86 SK corners
│   │   ├── dfl_extracted_corners.pkl  # 57 DFL corners
│   │   └── combined_corners.pkl    # 143 combined
│   ├── models/
│   │   ├── backbone.py             # CornerBackbone with node_proj/edge_proj (MODIFY THIS)
│   │   ├── receiver_head.py
│   │   ├── shot_head.py
│   │   └── two_stage.py
│   ├── training/
│   │   ├── train.py                # Training loops
│   │   ├── evaluate.py             # LOMO cross-validation
│   │   ├── permutation_test.py
│   │   └── ablation.py
│   ├── config.py                   # Hyperparameters
│   └── run_all.py
├── transfer_learning/
│   ├── phase1_train_ussf_backbone.py  # CRITICAL: How our PyTorch backbone was trained
│   ├── phase2_engineer_dfl_features.py  # CRITICAL: Contains USSF feature mapping logic
│   ├── data/
│   │   └── dfl_openplay_graphs.pkl     # 11,967 open-play graphs (PyG format)
│   └── weights/
│       ├── ussf_backbone_dense.pt      # Pretrained backbone (dense adjacency)
│       └── ussf_backbone_normal.pt     # Pretrained backbone (normal adjacency)
└── results/corner_prediction/          # All result JSONs and pickles
```

---

## VERIFIED USSF GROUND TRUTH (from GitHub repo + Context7 + notebook source)

The USSF project is at: https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn/

### Original USSF Architecture (TensorFlow/Spektral)
```python
class GNN(Model):
    def __init__(self, n_layers=3):
        self.conv1 = CrystalConv()  # No explicit channel dims — Spektral infers from data
        self.convs = [CrystalConv() for _ in range(1, n_layers)]  # 2 more = 3 total
        self.pool = GlobalAvgPool()
        self.dense1 = Dense(128, activation="relu")
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(128, activation="relu")
        self.dense3 = Dense(1, activation="sigmoid")
```
**NOTE:** The original USSF model does NOT have a `lin_in` layer between conv1 and the remaining convs. Our PyTorch reimplementation in `backbone.py` has `conv1 → lin_in → convs[0] → convs[1]` which adds an extra linear layer not in the original. This is a known architectural difference — the pretrained weights in `ussf_backbone_dense.pt` were trained with this PyTorch architecture, NOT the original TF one. So the feature alignment must match what `phase1_train_ussf_backbone.py` trained on, which may differ from the original USSF pickle format.

### USSF Training Config
- Learning rate: 1e-3
- Epochs: 150
- Batch size: 16
- Hidden channels: 128
- CrystalConv layers: 3
- Train/test split: 70/30 (seed=15)
- No validation set in original — just train and test
- Loss: BinaryCrossentropy
- Optimizer: Adam

### 12 Node Features (VERIFIED from notebook source code)
```python
node_feature_names = ['x', 'y', 'vx', 'vy', 'velocity', 'velocity_angle',
                      'dist_goal', 'angle_goal', 'dist_ball', 'angle_ball',
                      'attacking_team', 'potential_receiver']
```

| Idx | Name | Description (from USSF README) |
|-----|------|-------------------------------|
| 0 | x | x coordinate on the 2D pitch |
| 1 | y | y coordinate on the 2D pitch |
| 2 | vx | Velocity vector's x coordinate |
| 3 | vy | Velocity vector's y coordinate |
| 4 | velocity | Magnitude of the velocity |
| 5 | velocity_angle | Angle made by the velocity vector |
| 6 | dist_goal | Distance of the player from the goal post |
| 7 | angle_goal | Angle made by the player with the goal |
| 8 | dist_ball | Distance from the ball (always 0 for the ball) |
| 9 | angle_ball | Angle made with the ball (always 0 for the ball) |
| 10 | attacking_team | 1 if attacking, 0 if not, and for the ball |
| 11 | potential_receiver | 1 if potential receiver, 0 otherwise |

**⚠️ CRITICAL UNKNOWN: Feature normalization.**
The USSF README describes features as names only. It does NOT document:
- Whether `vx`/`vy` are raw m/s or unit direction vectors
- Whether `velocity` is raw speed or normalized to [0,1]
- Whether `velocity_angle` is raw radians or normalized
- Whether `x`/`y` are raw coordinates, normalized to pitch dimensions, or in some other range
- Whether `dist_goal`/`angle_goal`/`dist_ball`/`angle_ball` are raw or normalized

**The features are pre-computed in the pickle files.** The notebook loads pre-baked `.pkl` files and passes them directly to the GNN. There is no feature engineering step visible in the notebook.

**To determine the actual normalization scheme, you MUST:**
1. Download and inspect the USSF pickle data directly:
   ```python
   import pickle, requests
   url = "https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/combined.pkl"
   # Download or check if cached locally
   ```
   Then inspect `data['normal']['x'][0]` to see actual numerical ranges.
2. **Also** inspect `phase1_train_ussf_backbone.py` to see what features our PyTorch backbone was trained on. Our backbone was NOT trained on the original USSF TF model weights — it was retrained in PyTorch on data that `phase2_engineer_dfl_features.py` prepared.
3. **Also** inspect `phase2_engineer_dfl_features.py` which explicitly engineers features from raw DFL tracking data to match the USSF schema. This file IS the Rosetta Stone — it documents every transformation.

**DO NOT ASSUME NORMALIZATIONS. INSPECT THE DATA FIRST.**

### 6 Edge Features (VERIFIED from notebook source code)
```python
edge_feature_names = ['distance', 'speed_difference', 'pos_sin_angle', 
                      'pos_cos_angle', 'vel_sin_angle', 'vel_cos_angle']
```

| Idx | Name | Description (from USSF README) |
|-----|------|-------------------------------|
| 0 | Player Distance | Euclidean distance between two connected players |
| 1 | Speed Difference | Speed difference between two connected players |
| 2 | Positional Sine angle | sin(angle between two players' positions) |
| 3 | Positional Cosine angle | cos(angle between two players' positions) |
| 4 | Velocity Sine angle | sin(angle between velocity vectors of two players) |
| 5 | Velocity Cosine angle | cos(angle between velocity vectors of two players) |

**Same unknown:** Are these raw or normalized? Inspect the pickles.

### Graph Structure (VERIFIED)
- **Ball node is included** as the LAST node (index -1). The `ShuffledCounterDataset` code explicitly does `arr[0:-1]` to separate players from ball, and `ball = arr[-1].copy()`.
- Total nodes per graph: **23** (22 players + 1 ball)
- 5 adjacency types available: normal, delaunay, dense, dense_ap, dense_dp
- `dense` = fully connected (all 23 nodes connected to each other)
- `normal` = attackers connected to attackers, defenders to defenders, both linked through ball

### Data (VERIFIED)
- Combined: 20,863 graphs from 632 MLS + NWSL + int'l women's games
- Source: StatsPerform events + SkillCorner broadcast tracking
- Balanced: 50% success / 50% failure
- Labels: "success" = counterattack reaches opposing penalty area

---

## COMPARISON: USSF vs Our Current Pipeline

| Aspect | USSF (original) | Our PyTorch Backbone | Our Corner Pipeline |
|--------|-----------------|---------------------|-------------------|
| Framework | TF/Spektral | PyTorch/PyG | PyTorch/PyG |
| Node features | 12 (pre-baked in pkl) | 12 (from phase2) | 13 → node_proj → 12 |
| Edge features | 6 (pre-baked in pkl) | 6 (from phase2) | 4 → edge_proj → 6 |
| Nodes per graph | 23 (22 players + ball) | 23 | **22 (no ball node!)** |
| Adjacency | 5 types, default normal | dense (for our backbone) | KNN k=6 (132 edges) |
| Architecture | CrystalConv ×3 → Pool → Dense ×3 | CrystalConv → Linear → CrystalConv ×2 | Same backbone + heads |
| Extra linear layer | NO `lin_in` | YES `lin_in: Linear(12, 128)` | Inherited from backbone |

**Three mismatches to fix:**
1. **Node count:** We use 22 nodes, USSF uses 23 (has ball). Ball was the last node.
2. **Edge features:** We compute 4 (dx, dy, distance, same_team). USSF uses 6 (distance, speed_diff, pos_sin, pos_cos, vel_sin, vel_cos). We're missing 4 features and have 2 the USSF never used (dx, dy).
3. **Adjacency:** We use KNN k=6. USSF backbone was trained on dense. This is a structural mismatch.

---

## CRITICAL FIRST STEPS (before writing ANY code)

### Step 0a: Inspect the USSF pickle to determine feature normalizations
```python
import pickle, numpy as np

# Option 1: Check if already cached locally
import os
for root, dirs, files in os.walk('/home/mseo/CornerTactics'):
    for f in files:
        if 'combined.pkl' in f or 'men.pkl' in f or 'women.pkl' in f:
            print(os.path.join(root, f))

# Option 2: Download if not cached
import requests
url = "https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/combined.pkl"
# This is ~200MB, cache it
if not os.path.exists("ussf_combined.pkl"):
    r = requests.get(url)
    with open("ussf_combined.pkl", "wb") as f:
        f.write(r.content)

with open("ussf_combined.pkl", "rb") as f:
    ussf_data = pickle.load(f)

# Inspect structure
print(ussf_data.keys())
print(ussf_data['normal'].keys())

# Inspect node features of first graph
x = ussf_data['normal']['x'][0]  # or ussf_data['dense']['x'][0]
print(f"Node matrix shape: {x.shape}")  # Expected: (23, 12)
print(f"Node feature ranges:")
for i in range(x.shape[1]):
    col = x[:, i]
    print(f"  Feature {i}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}, std={col.std():.4f}")

# Inspect edge features
e = ussf_data['normal']['e'][0]
print(f"\nEdge matrix shape: {e.shape}")
print(f"Edge feature ranges:")
for i in range(e.shape[1]):
    col = e[:, i]
    print(f"  Feature {i}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}, std={col.std():.4f}")

# Inspect adjacency
a = ussf_data['normal']['a'][0]
print(f"\nAdjacency shape: {a.shape}")  # Expected: (23, 23)
print(f"Adjacency sum per node: {a.sum(axis=1)}")

# Check DENSE adjacency too (this is what our backbone was trained on)
x_dense = ussf_data['dense']['x'][0]
e_dense = ussf_data['dense']['e'][0]
a_dense = ussf_data['dense']['a'][0]
print(f"\nDense adjacency shape: {a_dense.shape}")
print(f"Dense edge shape: {e_dense.shape}")
print(f"Dense node shape: {x_dense.shape}")

# Check ball node (last node)
print(f"\nBall node features (last row): {x[22, :]}")
print(f"Ball dist_ball (should be 0): {x[22, 8]}")
print(f"Ball angle_ball (should be 0): {x[22, 9]}")
```

**Report the output of this step before proceeding.** The entire experiment depends on getting the normalizations right.

### Step 0b: Inspect how our PyTorch backbone was trained
```python
# Read phase1_train_ussf_backbone.py
# Identify: what features did it train on? Did it use the USSF pickle directly?
# Did it transform the features or use them as-is?
cat /home/mseo/CornerTactics/transfer_learning/phase1_train_ussf_backbone.py
```

### Step 0c: Inspect how phase2 engineers features
```python
# This is the Rosetta Stone — it maps raw DFL tracking to USSF schema
cat /home/mseo/CornerTactics/transfer_learning/phase2_engineer_dfl_features.py
```

### Step 0d: Inspect our current corner graph construction
```python
cat /home/mseo/CornerTactics/corner_prediction/data/build_graphs.py
```

**Only after completing Steps 0a-0d should you write any code.** Report your findings to me so I can verify before proceeding.

---

## EXPERIMENT A: USSF-Aligned Feature Engineering (Priority 1)

### Goal
Replace the learned `node_proj: Linear(14→12)` and `edge_proj: Linear(4→6)` with deterministic feature engineering that produces the exact 12 node features and 6 edge features the backbone expects. Also fix the node count (add ball node) and adjacency type (use dense).

### Step A1: Engineer USSF-aligned node features

Based on what you learn from Step 0, create a new function `build_ussf_aligned_graph()` in `build_graphs.py`. This function must:

1. Take the same corner record as input (has per-player x, y, vx, vy, speed, team info)
2. Compute all 12 USSF features using the EXACT same transformations/normalizations you found in the pickle data and/or `phase2_engineer_dfl_features.py`
3. For each player compute: x, y (with USSF normalization), vx, vy (with USSF normalization), velocity (magnitude, with USSF normalization), velocity_angle, dist_goal, angle_goal, dist_ball, angle_ball, attacking_team, potential_receiver
4. **Add a 23rd ball node** with: x=ball_x, y=ball_y, vx=0, vy=0, velocity=0, velocity_angle=0, dist_goal=(from ball pos), angle_goal=(from ball pos), dist_ball=0, angle_ball=0, attacking_team=0 (or check USSF ball value), potential_receiver=0

### Step A2: Engineer USSF-aligned edge features

For each pair of connected nodes, compute:
1. `distance`: Euclidean distance between them
2. `speed_difference`: |speed_i - speed_j| (or signed? check USSF data)
3. `pos_sin_angle`: sin(atan2(y_j - y_i, x_j - x_i))
4. `pos_cos_angle`: cos(atan2(y_j - y_i, x_j - x_i))
5. `vel_sin_angle`: sin(angle between velocity vectors of i and j)
6. `vel_cos_angle`: cos(angle between velocity vectors of i and j)

**⚠️ VERIFY from USSF data:** "Positional angle" could mean atan2(dy, dx) OR the angle of the position vector relative to some reference. "Velocity angle" could mean the angle BETWEEN the two velocity vectors OR the difference in their individual angles. Inspect the data to disambiguate.

### Step A3: Use dense adjacency

Switch from KNN k=6 to fully connected. Every node connected to every other node (including ball). For 23 nodes with dense adjacency, this is 23×22 = 506 directed edges (or 23×23 = 529 if self-loops). Check USSF adjacency matrix diagonal to determine self-loops.

### Step A4: Modify backbone.py

Add `mode="ussf_aligned"` that:
- Sets `node_proj = None`, `edge_proj = None`
- Expects 12 node features and 6 edge features directly
- Loads the same `ussf_backbone_dense.pt` weights
- Freezes backbone

**Receiver indicator handling:** The `potential_receiver` slot (feature index 11) serves double duty:
- In USSF: marks "potential receiver" in open-play (always available)
- In our corners: marks the predicted/oracle receiver for Stage 2

For Stage 1 (receiver prediction): set potential_receiver=0 for all players (we don't know the receiver yet). This matches USSF behavior where not all graphs have a clear receiver.
For Stage 2 (shot prediction): set potential_receiver=1 for the oracle/predicted receiver node. This is exactly how receiver conditioning should work — it's a built-in feature slot, not an extra dimension.

### Step A5: Run evaluation

```bash
# USSF-aligned, combined dataset, 5 seeds
for seed in 42 123 456 789 1234; do
    python -m corner_prediction.run_all --eval --combined --seed $seed --feature-mode ussf_aligned
done

# Velocity ablation on USSF-aligned (position-only = zero out features 2,3,4,5)
python -m corner_prediction.run_all --ablation position_only plus_velocity --feature-mode ussf_aligned

# Permutation test
python -m corner_prediction.run_all --permutation-only --combined --seed 42 --feature-mode ussf_aligned
```

### Step A6: Update MLP and XGBoost baselines

The baselines also need updating:
- **MLP**: Now takes [23 × 12] = 276 features (was [22 × 13] = 286). Retrain with same hyperparameters.
- **XGBoost**: Aggregate features need recalculation with USSF feature names. Velocity aggregates should use the same normalization.

Run baselines on combined with all 5 seeds for fair comparison.

### Expected results for Experiment A
- Multi-seed AUC variance should **decrease** (fewer stochastic parameters)
- Mean AUC might go up slightly (better feature alignment) or stay similar
- Velocity ablation delta should be similar to current +0.164 (feature representation doesn't change the underlying signal)
- If AUC drops significantly, something is wrong with the feature alignment — double check normalizations

---

## EXPERIMENT B: Joint Pretraining (Priority 2 — only if Experiment A complete)

### Goal
Fine-tune the backbone jointly on open-play + corner data instead of using a frozen off-the-shelf backbone.

### Step B1: Load USSF open-play data

Check what we have locally first:
```bash
find /home/mseo/CornerTactics -name "*openplay*" -o -name "*combined*" -path "*/transfer_learning/*"
```

If `dfl_openplay_graphs.pkl` exists (11,967 graphs), use that. If you can also get the full USSF 20,863-graph dataset (`combined.pkl` from S3), even better — more data = more regularization.

**IMPORTANT:** The open-play graphs must use the same feature schema as your USSF-aligned corner graphs from Experiment A. If the open-play data was saved in a different format (raw USSF pkl format vs PyG format), you'll need to convert.

### Step B2: Create joint training loop

Create `corner_prediction/training/joint_pretrain.py`:

```python
def joint_pretrain(
    corner_graphs,        # 143 USSF-aligned corner graphs (PyG Data)
    openplay_graphs,      # 11,967+ open-play graphs (PyG Data)
    backbone,             # CrystalConv backbone — ALL params UNFROZEN
    shot_head,            # Simple shot prediction head
    n_epochs=50,
    corner_weight=5.0,    # Upweight corner loss to balance gradient contribution
    backbone_lr=1e-4,     # Lower LR for backbone fine-tuning
    head_lr=1e-3,         # Normal LR for head
    weight_decay=1e-3,
    seed=42,
):
```

**Batch composition:** Use a weighted random sampler or alternating strategy. Each epoch should see ALL corners multiple times (143 is tiny) but only sample from open-play. Target: ~15% corners per batch.

**Loss:** Binary cross-entropy for both. `corner_weight` multiplies corner loss to compensate for underrepresentation.

**Only shot prediction during pretraining.** No receiver head — we're just teaching the backbone football + corner patterns.

### Step B3: Data leakage management

**Per-fold joint pretraining is correct but expensive.** For each of 17 LOMO folds, you'd exclude that match's corners from joint pretraining, pretrain, then evaluate. That's 17 pretraining runs.

**Acceptable shortcut:** Pretrain once on all corners + open-play. Note the leakage: corners are <0.7% of training data (143/20,863). The backbone cannot memorize individual corners from such diluted signal. Note this as a limitation.

### Step B4: Evaluation

After joint pretraining, FREEZE the backbone and evaluate with LOMO exactly as in Experiment A:

```bash
# Joint pretrain (once)
python -m corner_prediction.training.joint_pretrain \
    --openplay-data transfer_learning/data/dfl_openplay_graphs.pkl \
    --corner-data corner_prediction/data/combined_corners_ussf_aligned.pkl \
    --epochs 50 --seed 42

# Evaluate with joint-pretrained backbone
for seed in 42 123 456 789 1234; do
    python -m corner_prediction.run_all --eval --combined --seed $seed \
        --backbone-weights joint_pretrained_backbone.pt --feature-mode ussf_aligned
done
```

---

## Output Checklist

### Experiment A deliverables:
- [ ] Step 0a-0d inspection results reported (feature ranges, normalizations, actual code review)
- [ ] `build_graphs.py` has `build_ussf_aligned_graph()` with verified feature transforms
- [ ] Ball node (23rd) added to corner graphs
- [ ] Dense adjacency used (matching backbone pretraining)
- [ ] 6 USSF edge features computed (not our old 4)
- [ ] `backbone.py` has `mode="ussf_aligned"` (no projection layers)
- [ ] Receiver indicator uses feature slot 11 (potential_receiver) — no extra dimension
- [ ] 5-seed results on combined dataset
- [ ] Velocity ablation on USSF-aligned features
- [ ] Permutation test on USSF-aligned features
- [ ] Updated MLP and XGBoost baselines with new feature dimensions
- [ ] All results saved with `ussf_aligned` in filenames

### Experiment B deliverables (if time):
- [ ] `joint_pretrain.py` with mixed batching
- [ ] Joint-pretrained backbone weights saved
- [ ] 5-seed LOMO evaluation with joint backbone
- [ ] Data leakage documented as limitation
- [ ] Comparison table: original frozen vs USSF-aligned frozen vs joint-pretrained

### Naming convention:
```
# Experiment A:
combined_lomo_ussf_aligned_seed{42,123,456,789,1234}_YYYYMMDD_HHMMSS.json
ablation_ussf_aligned_{config}_YYYYMMDD_HHMMSS.json
combined_perm_ussf_aligned_shot_YYYYMMDD_HHMMSS.json

# Experiment B:
combined_lomo_joint_pretrained_seed{42,123,456,789,1234}_YYYYMMDD_HHMMSS.json
```

---

## What NOT to change

- Do NOT modify LOMO cross-validation logic
- Do NOT modify permutation test logic
- Do NOT change hyperparameters (LR, epochs, patience) for fair comparison
- Do NOT delete existing code — add alongside
- Keep `mode="pretrained"` working for backward compatibility
- Do NOT re-train the USSF backbone from scratch — we're aligning features TO the existing backbone

## Priority

**Steps 0a-0d FIRST (inspection). Then Experiment A. Then Experiment B only if A is done.**

If at ANY point you're uncertain about a normalization or feature definition, STOP and inspect the actual data. The whole point is deterministic alignment — guessing defeats the purpose.
