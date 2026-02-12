# USSF → Corner Kick Transfer Learning Plan

## Thesis Framing

**Experiment Title:** Cross-Task Transfer Learning for Corner Kick Prediction Using Pre-trained Counterattack GNN Representations

**Research Question:** Can graph neural network representations learned from counterattack prediction (20,863 graphs, USSF/MLS data) transfer to corner kick shot prediction (65 graphs, DFL data)?

**Why this matters:** If it works (even partially), you've shown that football GNNs learn generalizable spatial-tactical representations across game situations. If it fails, you've established that counterattack dynamics don't transfer to set-piece prediction — and the comparison between this approach vs. your DFL open-play pretraining tells you WHERE useful representations come from (same-distribution open play vs. cross-task counterattacks).

**Expected outcome:** Probably fails. But the failure mode is informative and publishable.

---

## The Three Problems (And How We Handle Them)

### Problem 1: Feature Schema Mismatch

**USSF node features (12):**
```
[x, y, vx, vy, velocity_mag, velocity_angle,
 dist_goal, angle_goal, dist_ball, angle_ball,
 attacking_team_flag, potential_receiver]
```

**USSF edge features (6):**
```
[player_distance, speed_difference,
 positional_sine_angle, positional_cosine_angle,
 velocity_sine_angle, velocity_cosine_angle]
```

**Your DFL data provides:** 25fps (x,y) tracking for all players → you can derive ALL of these features.

**Solution:** Engineer your DFL corner kick features to match USSF schema exactly.

| USSF Feature | DFL Derivation | Notes |
|---|---|---|
| x, y | Direct from tracking | MUST normalize to same coordinate system (0-1 pitch-relative, or meters — inspect USSF data to confirm) |
| vx, vy | Finite difference: `(pos[t] - pos[t-1]) * fps` | Use t=0 (delivery frame), compute from preceding frames. Smoothing window TBD after inspecting USSF velocity distributions |
| velocity_mag | `sqrt(vx² + vy²)` | Derived |
| velocity_angle | `atan2(vy, vx)` | Derived |
| dist_goal | Euclidean to goal center | Must use same goal position convention as USSF |
| angle_goal | `atan2(goal_y - y, goal_x - x)` | Derived |
| dist_ball | Euclidean to ball position | Ball position from DFL tracking |
| angle_ball | `atan2(ball_y - y, ball_x - x)` | Derived |
| attacking_team_flag | 1 for attacking, 0 for defending | Direct from DFL team labels |
| potential_receiver | ??? | USSF-specific. Set to 0 for all players in corner data, or drop this feature entirely |

**Critical first step:** Download `combined.pkl`, inspect actual feature distributions (means, stds, ranges). Your DFL features MUST match these distributions or the pretrained weights see garbage.

### Problem 2: Graph Topology Mismatch

**Counterattacks:** ~10-15 players spread across half-pitch, sparse connectivity
**Corner kicks:** 20+ players packed in penalty area, dense clustering

**Solution:** Use `dense` adjacency type for pretraining, NOT `normal`.

Rationale: The `normal` type (team-based connectivity through ball) encodes counterattack-specific assumptions about passing lanes. The `dense` type (all players fully connected) makes no task-specific assumptions — the CrystalConv layers learn which connections matter from the edge features alone. Dense connectivity transfers better because corner kicks ARE dense situations.

**Alternative worth testing:** `delaunay` — spatial triangulation adapts to whatever geometry the players form. Principled, topology-agnostic. Test both.

### Problem 3: No Pretrained Weights Released

**Solution:** Train the backbone yourself. This is Phase 1 of the plan.

---

## Execution Plan

### Phase 0: Data Inspection (Day 1)

**Goal:** Understand exactly what USSF data looks like before writing any training code.

```python
import pickle
import numpy as np

# Download
data = get_data('combined.pkl')  # 20,863 graphs

# Inspect structure
print(type(data))
print(data.keys())
print(data['normal'].keys())

# Node feature distributions
all_nodes = np.concatenate(data['normal']['x'])
print(f"Node shape per graph: {data['normal']['x'][0].shape}")
print(f"Total nodes across all graphs: {all_nodes.shape}")

for i, name in enumerate(['x','y','vx','vy','vel_mag','vel_angle',
                           'dist_goal','angle_goal','dist_ball',
                           'angle_ball','att_flag','pot_receiver']):
    vals = all_nodes[:, i]
    print(f"{name}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
          f"min={vals.min():.3f}, max={vals.max():.3f}")

# Edge feature distributions
all_edges = np.concatenate(data['normal']['e'])
print(f"\nEdge shape per graph: {data['normal']['e'][0].shape}")
for i, name in enumerate(['distance','speed_diff','pos_sin','pos_cos',
                           'vel_sin','vel_cos']):
    vals = all_edges[:, i]
    print(f"{name}: mean={vals.mean():.3f}, std={vals.std():.3f}")

# Adjacency structure
print(f"\nAdj shape: {data['normal']['a'][0].shape}")
print(f"Density: {data['normal']['a'][0].sum() / data['normal']['a'][0].size:.3f}")

# Also inspect 'dense' adjacency
print(f"Dense adj density: {data['dense']['a'][0].sum() / data['dense']['a'][0].size:.3f}")

# Class balance
labels = np.array(data['binary']).flatten()
print(f"\nLabels: {np.bincount(labels.astype(int))}")
print(f"Success rate: {labels.mean():.3f}")

# Variable graph sizes?
sizes = [x.shape[0] for x in data['normal']['x']]
print(f"\nPlayers per graph: min={min(sizes)}, max={max(sizes)}, "
      f"mean={np.mean(sizes):.1f}")
```

**Output of this phase:** A document listing exact feature distributions, normalization conventions, coordinate systems, graph sizes. This is your compatibility checklist.

### Phase 1: Train USSF Backbone (Days 2-3)

**Goal:** Reproduce USSF results, save trained CrystalConv weights.

**Architecture (from USSF, slightly modified):**
```
CrystalConv layers: 3 (USSF default)
Hidden channels: 128 (USSF default)
Dense head: 128 → dropout(0.5) → 128 → dropout(0.5) → 1 (sigmoid)
Optimizer: Adam, lr=1e-3
Loss: BinaryCrossentropy
Epochs: 150
Batch: 16
```

**Adjacency types to train:** Train TWO separate backbones:
1. `dense` adjacency (hypothesis: transfers better to corners)
2. `normal` adjacency (USSF default, control condition)

**Data split:** Use their sequence-aware splitting (70/30, seed=15 for reproducibility).

**Validation:** You MUST reproduce their reported AUC before proceeding. If you can't match their results on their data with their code, the backbone is broken and nothing downstream matters.

**Save weights:**
```python
# After training, save ONLY the conv layers
backbone_weights = {
    'conv1': model.conv1.get_weights(),
    'convs': [conv.get_weights() for conv in model.convs],
    'pool': model.pool.get_weights() if model.pool.trainable else None
}
pickle.dump(backbone_weights, open('ussf_backbone_dense.pkl', 'wb'))

# Also save full model for reference
model.save_weights('ussf_full_model_dense.h5')
```

**Deliverable:** Two trained backbones (`dense`, `normal`), verified AUCs logged.

### Phase 2: Engineer DFL Corner Features to Match USSF Schema (Days 4-6)

**Goal:** Transform your 65 DFL corner kick graphs into EXACT same feature format as USSF data.

**Steps:**

1. **Coordinate normalization:** Inspect Phase 0 output. If USSF uses pitch-relative [0,1] coordinates, normalize DFL's meter-based coordinates to match. If USSF uses meters, convert DFL to same pitch dimensions.

2. **Velocity computation from 25fps tracking:**
   ```python
   # At delivery frame t=0
   # Use 3-frame centered difference for smoother velocity estimate
   # (or match whatever USSF used — inspect their preprocessing)
   vx = (x[t+1] - x[t-1]) / (2 * dt)  # dt = 1/25 = 0.04s
   vy = (y[t+1] - y[t-1]) / (2 * dt)
   ```

3. **Derived features:** vel_mag, vel_angle, dist_goal, angle_goal, dist_ball, angle_ball — compute identically to USSF.

4. **`potential_receiver` handling:**
   - Option A: Drop from both USSF and DFL (retrain backbone without it — 11 node features)
   - Option B: Set to 0 for all corner players (weakens transfer but keeps architecture identical)
   - Option C: Heuristic — set to 1 for attacking players in 6-yard box (approximation)
   - **Recommendation: Option A** — retrain backbone without it. Cleaner experiment.

5. **Edge features:** Compute identically from player pairs.

6. **Adjacency matrix:** Build `dense` (fully connected) and `normal` (team-based) for each corner graph, matching USSF construction logic exactly.

7. **Distribution alignment check:**
   ```python
   # CRITICAL: Compare feature distributions
   ussf_nodes = np.concatenate(ussf_data['dense']['x'])
   dfl_nodes = np.concatenate(dfl_corner_graphs_x)

   for i, name in enumerate(feature_names):
       ks_stat, p_val = scipy.stats.ks_2samp(ussf_nodes[:, i], dfl_nodes[:, i])
       print(f"{name}: KS={ks_stat:.3f}, p={p_val:.4f}")
       # If p < 0.01, distributions are significantly different
       # → that feature will NOT transfer well
   ```

**Deliverable:** 65 corner kick graphs in USSF-compatible format, with distribution comparison report.

### Phase 3: Transfer Learning (Days 7-9)

**Goal:** Load pretrained conv weights, freeze them, train new head on 65 corners.

**Architecture for fine-tuning:**
```python
class TransferGNN(Model):
    def __init__(self, pretrained_backbone, n_out=1):
        super().__init__()
        # Load pretrained conv layers — FROZEN
        self.conv1 = pretrained_backbone.conv1
        self.convs = pretrained_backbone.convs
        self.pool = GlobalAvgPool()

        # Freeze conv layers
        self.conv1.trainable = False
        for conv in self.convs:
            conv.trainable = False

        # NEW head — small because 65 samples
        self.dense1 = Dense(32, activation="relu")     # Down from 128
        self.dropout = Dropout(0.3)                      # Down from 0.5
        self.dense2 = Dense(n_out, activation="sigmoid")
        # NO second dense layer — too many params for 65 samples

    def call(self, inputs):
        x, a, e, i = inputs
        x = self.conv1([x, a, e])
        for conv in self.convs:
            x = conv([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)
```

**Training config for 65 samples:**
```
Epochs: 50 (with early stopping, patience=10)
Batch size: 8 (gives ~6 steps per epoch for train set)
Learning rate: 1e-4 (10x lower than pretraining — don't blast the head)
Loss: BinaryCrossentropy with class weights
```

**Data split for 65 corners:**
- Match-based split (MUST keep all corners from same match together)
- ~45 train / ~10 val / ~10 test (rough, depends on match distribution)
- This is brutally small. Report confidence intervals.

**Experimental conditions (6 total):**

| Condition | Backbone | Adjacency | Conv Frozen? | Purpose |
|---|---|---|---|---|
| A | USSF pretrained | dense | Yes | Main hypothesis |
| B | USSF pretrained | normal | Yes | Adjacency type comparison |
| C | USSF pretrained | dense | No (lr=1e-5) | Partial fine-tuning |
| D | Random init | dense | No | Train from scratch baseline |
| E | Random init | normal | No | Train from scratch baseline |
| F | – | – | – | Majority class baseline |

**Why all 6:** Conditions A/B test transfer. C tests if unfreezing helps. D/E prove that 65 samples can't train a GNN from scratch (expected: worse than majority baseline). F is the floor.

### Phase 4: Ablation — Velocity Feature Importance (Days 10-11)

**Goal:** THE headline result. Does velocity matter in the transferred model?

**Use USSF's own ShuffledCounterDataset methodology:**
```python
# For the BEST transfer condition (probably A or C):
# 1. Evaluate on test set → baseline AUC
# 2. Shuffle vx, vy for all players → evaluate → AUC_no_velocity
# 3. Shuffle x, y for all players → evaluate → AUC_no_position

# If AUC_no_velocity << AUC_baseline: velocity is critical
# If AUC_no_position << AUC_baseline: position is critical
# If both drop equally: both matter
# If neither drops: model learned nothing (transfer failed)
```

**This connects directly to your 7.5 ECTS finding:** You proved position-only = AUC 0.50. If this transfer model shows velocity shuffling tanks performance but position shuffling doesn't, you've experimentally isolated velocity as the missing ingredient.

### Phase 5: Compare Against DFL Open-Play Pretraining (Days 12-14)

**Goal:** The real comparison. Is USSF counterattack pretraining better or worse than pretraining on open-play sequences from your own DFL data?

**Setup:**
- Train same CrystalConv architecture on DFL open-play frames (thousands available from 7 matches)
- Same fine-tuning procedure on 65 corners
- Compare AUCs

**Expected result:** DFL open-play pretraining wins because:
1. Same coordinate system (no distribution shift)
2. Same tracking precision
3. Includes dense situations (set pieces, congested play)

**But if USSF pretraining wins:** That's a genuinely surprising and publishable finding — it would mean cross-league, cross-task transfer works in football GNNs. That's a real contribution.

---

## Reporting Structure for Thesis

### Section: Cross-Task Transfer Learning Experiment

1. **Motivation:** Can representations learned from counterattack prediction generalize to corner kick prediction?

2. **Source domain:** USSF counterattack dataset (20,863 graphs, MLS/NWSL), CrystalConv architecture

3. **Target domain:** DFL corner kicks (65 graphs), binary shot prediction

4. **Feature alignment:** How we matched schemas, distribution comparison (KS tests), known gaps

5. **Results table:** All 6 conditions + DFL open-play pretraining comparison

6. **Ablation:** Velocity importance via permutation testing

7. **Discussion:**
   - If transfer works: "Football GNNs learn generalizable spatial-tactical representations"
   - If transfer fails: "Counterattack dynamics are domain-specific; set-piece prediction requires set-piece-like pretraining data. The topology mismatch between sparse counterattack graphs and dense corner kick graphs prevents meaningful transfer."
   - Either way: "Velocity features are [critical/not critical] for corner kick prediction, as shown by permutation importance analysis"

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Can't reproduce USSF results | Low (code is public) | Blocks everything | Use their exact code first, then adapt |
| Feature distribution mismatch too large | High | Conv weights useless | Report KS statistics honestly, proceed anyway |
| 65 samples insufficient for ANY signal | Very High | All conditions = baseline | Frame as "establishing that N>65 is needed even with transfer" |
| USSF S3 bucket goes offline | Low | No pretraining data | Download immediately, store locally |
| Spektral/TF version conflicts | Medium | Delays | Pin versions: `spektral==1.3.1`, `tensorflow==2.12` |

---

## Timeline

| Day | Task | Deliverable |
|---|---|---|
| 1 | Phase 0: Download USSF data, inspect distributions | Feature distribution report |
| 2-3 | Phase 1: Train two backbones (dense, normal) | Verified AUCs, saved weights |
| 4-6 | Phase 2: Engineer DFL features to match USSF | 65 compatible graphs, KS test report |
| 7-9 | Phase 3: Run all 6 transfer conditions | Results table |
| 10-11 | Phase 4: Velocity permutation importance | Ablation results |
| 12-14 | Phase 5: DFL open-play comparison | Final comparison table |

**Total: ~14 working days for a complete, well-controlled experiment.**

---

## What Makes This Defensible

1. **You're not just trying transfer learning** — you're testing it against a proper baseline (DFL open-play pretraining) AND a from-scratch baseline AND a majority-class baseline. That's 6+ conditions.

2. **You predict it will probably fail** — and you explain WHY (topology mismatch, distribution shift). An examiner can't attack you for "not knowing this would fail" when you document the expected failure modes upfront.

3. **The velocity ablation is the real payload** — even if transfer fails completely, the permutation importance analysis on whatever model works best (probably DFL pretrained) delivers your thesis's core claim.

4. **It connects your 7.5 ECTS and 15 ECTS work** — "Phase 1 showed no signal in static data. Phase 2 asks: can we find signal with velocity data, and where should we get our representations from?"
