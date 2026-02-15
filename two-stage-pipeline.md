# Two-Stage Corner Kick Prediction Pipeline: Receiver Prediction → Conditional Shot Prediction

## Project Context

I'm building a TacticAI-style two-stage corner kick prediction system for my master's thesis (defense: March 16, 2026). The approach:

- **Stage 1**: Predict which player receives the corner kick delivery (receiver prediction)
- **Stage 2**: Predict whether a shot results, conditioned on the receiver (conditional shot prediction)

TacticAI achieved F1=0.71 with this two-stage approach vs F1=0.52 unconditional. My previous 7.5 ECTS work showed AUC≈0.50 (random chance) using static positions only. I now have velocity-capable tracking data from SkillCorner.

I have a **pretrained GNN backbone** from a USSF soccer project that achieved 0.86 AUC on open-play shot prediction using CrystalConv layers. The plan is transfer learning — freeze backbone, train task-specific heads on corner data.

---

## Data Available

### Source
SkillCorner Open Data — 10 A-League 2024/2025 matches, broadcast tracking at 10Hz.

**86 total corners, 80 usable after quality filtering (excluding 0% detection frames).**

### Full data audit
A comprehensive audit of all corner kick data in the SkillCorner dataset has already been completed. The full report with per-corner breakdowns, velocity analysis, detection rates, event chain reconstruction, and TacticAI comparison matrix is at:

```
/home/mseo/CornerTactics/docs/skillcorner-audit-report.md
```

**Read this file first before writing any code.** It contains exact column names, field mappings, edge cases, quality issues, and validated extraction methods. Do not re-derive what's already documented there.

### File locations
```
data/skillcorner/data/matches/{match_id}/
├── {id}_match.json                    # Lineups, player metadata, pitch dims
├── {id}_tracking_extrapolated.jsonl   # Frame-by-frame tracking at 10Hz
├── {id}_dynamic_events.csv            # Events: possessions, passes, runs, pressing
└── {id}_phases_of_play.csv            # Phase labels per frame range
```

### Match IDs (10 matches)
```
2017461, 2015213, 2013725, 2011166, 2006229,
1996435, 1953632, 1925299, 1899585, 1886347
```

### Corner detection
- Column: `game_interruption_before` in `dynamic_events.csv`
- Values: `corner_for` (attacking team) or `corner_against` (defending team)
- Events within 20 frames (2s) of each other in same period → same corner
- Raw rows: 117 → 86 unique corners after dedup → 80 after quality filter

### Per-corner data audit results

| Metric | Value |
|--------|-------|
| Total usable corners | 80 |
| Shot rate | 33.7% (29 shots, 3 goals) |
| Mean detection rate at delivery | 69.8% (15.3/22 detected) |
| Ball detected at delivery | 92% |
| Corners with receiver labels | 74/86 (86%) |
| Corners with passing_option events | 71/86 (83%) |
| Mean passing_option candidates per corner | ~6.8 |
| Corner taker identified | 74% |
| lead_to_shot coverage | 100% |
| lead_to_goal coverage | 100% |

### Coordinate system
- Center-origin meters: x ∈ [-52.5, 52.5], y ∈ [-34, 34]
- x-axis = long side (touchline), y-axis = short side (goal-to-goal)
- Pitch dimensions in match.json (`pitch_length`, `pitch_width`)

### Velocity derivation (validated clean)
```python
vx = (x[t] - x[t-1]) / 0.1  # 10Hz → 0.1s between frames
vy = (y[t] - y[t-1]) / 0.1
```
- Zero teleportation artifacts across 2,926 samples
- All speeds < 12 m/s (physically plausible)
- No smoothing needed for raw frame-to-frame
- Extrapolated player velocities are systematically dampened (0.61x speed vs detected)

### Key field mappings
- **Tracking `player_id`** → match.json `players[].id` (NOT `trackable_object`)
- **Team assignment**: `player_id` → match.json `players[].team_id` → compare to `home_team.id`/`away_team.id`
- **Receiver**: `player_targeted_name` in delivery event OR passing_option with `targeted=True` + `received=True`
- **Shot outcome**: `lead_to_shot` / `lead_to_goal` columns
- **Player role**: match.json `player_role.name` (e.g., "Goalkeeper", "Right Center Back", "Left Winger")
- **Direction normalization needed**: Teams switch sides at halftime. Use `home_team.side` from match.json to determine direction per period, then normalize all corners to attack left-to-right.

---

## Task 1: Data Extraction Pipeline

Build a complete extraction pipeline that produces one structured record per corner kick.

### Step 1.1: Identify all corners

```python
# Pseudocode
for each match_id:
    load dynamic_events.csv
    filter rows where game_interruption_before in ['corner_for', 'corner_against']
    group events within 20 frames of each other in same period → one corner
    record: match_id, period, delivery_frame, corner_team, corner_taker
```

### Step 1.2: Extract tracking snapshot at delivery

For each corner, from the tracking JSONL:
- Pull frame at `delivery_frame` (t=0)
- Pull frames at `delivery_frame - 1` through `delivery_frame - 3` (for velocity)
- Extract for each of 22 players: `player_id`, `x`, `y`, `is_detected`
- Extract ball: `x`, `y`, `z`, `is_detected`
- Compute velocity: `vx`, `vy` from frame difference

### Step 1.3: Enrich with match metadata

From match.json, for each player_id:
- `team_id` → is_attacking (1 if same team as corner taker, 0 if defending)
- `player_role.name` → one-hot or categorical encoding
- `player_role.position_group` → coarser grouping
- `number` (shirt number)

### Step 1.4: Extract labels

**Receiver label (Stage 1):**
- Primary: `player_targeted_name` from the delivery event row where `game_interruption_before = corner_for`
- Secondary: passing_option events within 3 seconds with `targeted=True` and `received=True`
- Tertiary: next `player_possession` event after delivery
- Map receiver name → `player_id` via match.json
- If receiver is on attacking team → valid receiver label
- If no receiver identified → exclude from Stage 1 training (keep for Stage 2 if shot label exists)

**Shot label (Stage 2):**
- `lead_to_shot` column on any event in the corner's event chain
- Binary: 1 if any event has `lead_to_shot=True`, 0 otherwise
- Also extract `lead_to_goal` for analysis

### Step 1.5: Normalize coordinates

All corners must be normalized to a canonical orientation:
- Attacking team attacks left-to-right (toward x = +52.5)
- Corner taken from positive-x end
- Use `home_team.side` and `period` to determine flip
- If corner is from the left side of the goal (y < 0 after normalization), note as `corner_side=left`; if y > 0, `corner_side=right`

### Expected output

One record per corner:
```python
{
    "match_id": int,
    "corner_id": str,  # unique identifier
    "period": int,
    "delivery_frame": int,
    "corner_team_id": int,
    "corner_taker_id": int,  # player_id, or None
    "corner_side": str,  # "left" or "right"
    
    # Per-player data (22 players)
    "players": [
        {
            "player_id": int,
            "x": float,  # normalized
            "y": float,  # normalized
            "vx": float,
            "vy": float,
            "speed": float,  # sqrt(vx² + vy²)
            "is_attacking": bool,
            "is_corner_taker": bool,
            "is_goalkeeper": bool,
            "role": str,  # e.g., "CB", "LW", "CF"
            "is_detected": bool,
            "is_receiver": bool,  # label for Stage 1
        }
        # ... 22 entries
    ],
    
    # Ball data
    "ball_x": float,
    "ball_y": float,
    "ball_z": float,
    "ball_detected": bool,
    
    # Labels
    "receiver_id": int or None,  # player_id of receiver
    "has_receiver_label": bool,
    "lead_to_shot": bool,
    "lead_to_goal": bool,
    
    # Quality
    "detection_rate": float,  # fraction of 22 players detected
    "n_detected": int,
    "n_extrapolated": int,
    
    # Event context
    "n_passing_options": int,
    "passing_option_ids": list,  # player_ids flagged as passing options
    "n_off_ball_runs": int,
    "pass_outcome": str or None,  # "successful", "unsuccessful", None
}
```

**Deliver**: Complete Python script that processes all 10 matches and outputs a list of these records, saved as both JSON and a pickle file for direct loading. Include data validation: assert 22 players per corner, assert coordinates within pitch bounds, flag any anomalies.

---

## Task 2: Graph Construction

Build PyTorch Geometric `Data` objects from the extracted corner records.

### Node features (per player)

Each of the 22 players is a node. Feature vector per node:

```python
node_features = [
    x,              # normalized x position (float)
    y,              # normalized y position (float)
    vx,             # x velocity (float)
    vy,             # y velocity (float)
    speed,          # scalar speed (float)
    is_attacking,   # 1.0 or 0.0
    is_corner_taker,# 1.0 or 0.0
    is_goalkeeper,  # 1.0 or 0.0
    is_detected,    # 1.0 or 0.0 (quality flag)
    # Role one-hot (choose grouping — ~6 categories):
    # GK, DEF, MID, FWD, or finer: GK, CB, FB, DM, CM, AM, W, CF
]
```

Decide on role encoding granularity. Finer roles = more features but sparser per category. Recommend position_group level: GK, Defender, Midfielder, Forward (4 categories, one-hot → 4 features). Total: 9 + 4 = 13 features per node.

### Edge construction

**Option A: Fully connected** — edges between all 22 players (22×21 = 462 edges). Simple, lets GNN learn which edges matter. May be noisy with 86 samples.

**Option B: k-nearest neighbors** — connect each player to k=6 nearest by Euclidean distance. Reduces edges to 22×6 = 132. Encodes spatial locality. TacticAI used this approach.

**Option C: Team-based bipartite** — edges between attackers and nearest defenders (marking assignments). More domain-specific.

**Recommendation**: Start with Option B (k=6 KNN) to match TacticAI. Also implement Option A for ablation.

### Edge features

```python
edge_features = [
    dx,          # x difference between connected players
    dy,          # y difference between connected players  
    distance,    # Euclidean distance
    same_team,   # 1.0 if same team, 0.0 if opponents
]
```

### Graph-level features

```python
graph_features = [
    corner_side,        # 0 or 1 (left/right)
    # Optional:
    score_diff,         # attacking_team_goals - defending_team_goals
    period,             # 1 or 2
    minute,             # match minute (normalized)
]
```

### Label tensors

**Stage 1 (receiver prediction):**
```python
# Per-node binary label: 1 for receiver, 0 for all others
# Only defined for attacking players (defending players masked out)
receiver_mask = tensor of shape [22] — True for attacking outfield players
receiver_label = tensor of shape [22] — 1.0 at receiver node index, 0.0 elsewhere
```

**Stage 2 (shot prediction):**
```python
# Graph-level binary label
shot_label = 1 if lead_to_shot else 0
```

### PyTorch Geometric Data object

```python
from torch_geometric.data import Data

data = Data(
    x=node_features,           # [22, 13]
    edge_index=edge_index,     # [2, num_edges]
    edge_attr=edge_features,   # [num_edges, 4]
    
    # Stage 1 labels
    receiver_mask=receiver_mask,    # [22] bool
    receiver_label=receiver_label,  # [22] float
    has_receiver_label=bool,
    
    # Stage 2 labels
    shot_label=shot_label,          # scalar
    goal_label=goal_label,          # scalar
    
    # Metadata
    match_id=match_id,
    corner_id=corner_id,
    detection_rate=detection_rate,
    
    # Graph-level features
    graph_attr=graph_features,      # [num_graph_features]
)
```

**Deliver**: Complete graph construction code. Load the extracted corner records from Task 1, build Data objects, store as a list. Include a `CornerKickDataset` class inheriting from `torch_geometric.data.Dataset` or `InMemoryDataset`. Print summary statistics: number of graphs, feature dimensions, label distributions, edge count distributions.

---

## Task 3: Model Architecture

### Stage 1: Receiver Prediction

```
Input: Corner kick graph (22 nodes)
       ↓
[Pretrained GNN Backbone — FROZEN]
  - CrystalConv layers (from USSF project)
  - Produces per-node embeddings: [22, hidden_dim]
       ↓
[Receiver Head — TRAINABLE]
  - Linear(hidden_dim, 64)
  - ReLU
  - Linear(64, 1)  → per-node logit
       ↓
[Masked Softmax]
  - Apply receiver_mask (only attacking outfield players)
  - Softmax over masked nodes → receiver probability per candidate
       ↓
Loss: Cross-entropy over masked candidates
Metric: Top-1 accuracy, Top-3 accuracy
```

### Stage 2: Conditional Shot Prediction

```
Input: Corner kick graph (22 nodes) + receiver identity
       ↓
[Add receiver features to graph]
  - Set is_predicted_receiver=1.0 on predicted receiver node (or ground-truth for oracle)
  - Optionally: add receiver-specific features (distance_to_goal, defensive_pressure, speed_toward_goal)
       ↓
[Pretrained GNN Backbone — FROZEN]
  - Same backbone as Stage 1
  - Produces per-node embeddings: [22, hidden_dim]
       ↓
[Graph-level pooling]
  - Global mean pool or attention-weighted pool
  - Produces graph embedding: [hidden_dim]
       ↓
[Shot Head — TRAINABLE]
  - Linear(hidden_dim + graph_features_dim, 32)
  - ReLU  
  - Dropout(0.3)
  - Linear(32, 1) → shot logit
       ↓
Loss: Binary cross-entropy with class weights (33.7% shots)
Metric: AUC-ROC, F1
```

### Pretrained backbone integration

The USSF pretrained backbone uses CrystalConv layers. I need you to:

1. **Define an adapter** — the USSF model was trained on open-play graphs which may have different node feature dimensions than our 13-feature corner graphs. Build a projection layer: `Linear(13, ussf_input_dim)` that maps our features to the backbone's expected input.

2. **Freeze strategy** — Freeze all backbone parameters. Only train: input projection, receiver head, shot head. This prevents overfitting on 80 samples.

3. **If no pretrained backbone available** — Build a lightweight GNN from scratch:
   ```
   CrystalConv(13, 64) → ReLU → CrystalConv(64, 64) → ReLU → CrystalConv(64, 64)
   ```
   With only 80 samples, keep this small. Max 3 layers, hidden_dim ≤ 64. Heavy dropout (0.3-0.5). Weight decay 1e-3.

### Two-stage inference

```python
# At inference time:
# 1. Run Stage 1 on corner graph → predicted receiver (argmax of masked softmax)
# 2. Add receiver indicator to graph
# 3. Run Stage 2 on augmented graph → P(shot | receiver, corner)

# Compare against:
# - Unconditional baseline: P(shot | corner) without receiver info
# - Oracle receiver: Use ground-truth receiver in Stage 2
# - Random receiver: Use random attacking player as receiver
```

**Deliver**: Complete model code with both stages. Include the adapter for pretrained backbone AND the from-scratch lightweight version. Both should be runnable.

---

## Task 4: Training & Evaluation

### Cross-validation strategy

**Leave-One-Match-Out (LOMO)** — 10 folds, one match held out per fold.

```python
matches = [2017461, 2015213, 2013725, 2011166, 2006229,
           1996435, 1953632, 1925299, 1899585, 1886347]

for held_out_match in matches:
    train_corners = [c for c in all_corners if c.match_id != held_out_match]
    test_corners = [c for c in all_corners if c.match_id == held_out_match]
    # Train on ~72-76 corners, test on 4-14
```

This is the only valid split strategy — random splitting would leak match-level patterns.

### Training details

```python
# Stage 1: Receiver prediction
optimizer = Adam(receiver_head.parameters(), lr=1e-3, weight_decay=1e-3)
epochs = 100  # small dataset, will converge fast
early_stopping = patience=20 on validation loss (use 1 match from train as val)

# Stage 2: Shot prediction  
optimizer = Adam(shot_head.parameters(), lr=1e-3, weight_decay=1e-3)
epochs = 100
class_weights = [1.0, 2.0]  # upweight shots (33.7% minority)
early_stopping = patience=20
```

### Evaluation metrics

**Stage 1 — Receiver prediction:**
- Top-1 accuracy (did we get the exact receiver?)
- Top-3 accuracy (is true receiver in top 3 predictions?)
- Random baseline: 1/n_candidates per corner (~1/6.8 ≈ 14.7% for top-1, ~3/6.8 ≈ 44.1% for top-3)
- Report mean ± std across 10 LOMO folds

**Stage 2 — Shot prediction:**
- AUC-ROC (primary — threshold-independent)
- F1 at optimal threshold
- Baseline: majority class (always predict no-shot) = 66.3% accuracy
- Random baseline AUC = 0.50
- Report mean ± std across 10 LOMO folds

**Two-stage combined:**
- Run Stage 2 with: (a) oracle receiver, (b) predicted receiver from Stage 1, (c) no receiver info (unconditional)
- Compare AUC across all three conditions
- This directly tests whether receiver conditioning improves shot prediction

### Statistical validation

**Permutation test** (same as my 7.5 ECTS work):
```python
# For each stage:
# 1. Record real-label metric (accuracy or AUC)
# 2. Shuffle labels N=100 times, retrain, record shuffled metrics
# 3. Compute p-value: fraction of shuffled >= real
# 4. If p < 0.05, signal is real
```

### Ablation experiments

Run the full pipeline in these configurations to isolate what matters:

| Experiment | Node features | Edge type | Notes |
|-----------|--------------|-----------|-------|
| Position only | x, y, team, role | KNN k=6 | Replicates 7.5 ECTS finding on new data |
| + Velocity | x, y, vx, vy, speed, team, role | KNN k=6 | Tests velocity hypothesis |
| + Detection flag | Above + is_detected | KNN k=6 | Tests if quality awareness helps |
| Full features | All 13 features | KNN k=6 | Full model |
| Full + FC edges | All 13 features | Fully connected | Tests edge construction impact |
| No GNN (baseline) | Aggregate features | N/A (XGBoost) | Traditional ML comparison |

For each ablation, report both Stage 1 and Stage 2 metrics.

**Deliver**: Complete training loop, evaluation code, LOMO cross-validation, permutation test, and ablation runner. Should produce a results table that can go directly into the thesis.

---

## Task 5: Baseline Comparisons

To contextualize the GNN results, implement these baselines:

### Baseline 1: Random prediction
- Stage 1: Uniform random over attacking outfield players
- Stage 2: Predict shot with probability = dataset shot rate (33.7%)

### Baseline 2: Heuristic receiver
- Predict receiver = nearest attacking outfield player to goal center
- Or: nearest to typical delivery zone (6-yard box area)

### Baseline 3: XGBoost on aggregate features (from 7.5 ECTS)
- Same 22 aggregate features as previous work
- Now with added velocity-based features:
  - mean_attacker_speed, mean_defender_speed
  - max_attacker_speed, speed_toward_goal (per attacker, aggregated)
  - speed_differential (attackers vs defenders)
- For shot prediction only (receiver prediction doesn't make sense with aggregate features)

### Baseline 4: Unconditional GNN shot prediction
- Same GNN but Stage 2 only, no receiver conditioning
- Direct comparison: does knowing the receiver help?

### Baseline 5: MLP on per-player features (no graph structure)
- Flatten all 22 players' features into one vector: [22 × 13] = [286]
- MLP: Linear(286, 64) → ReLU → Linear(64, 1)
- Tests whether graph structure adds value beyond just having the right features

**Deliver**: All baseline implementations with same LOMO evaluation as the main model.

---

## Task 6: Results Visualization

Generate thesis-ready figures:

### Figure 1: Receiver prediction example
- Plot one corner kick: pitch, all 22 players colored by team
- Overlay receiver probabilities as node size or color intensity
- Show true receiver vs predicted receiver
- Arrow from corner taker to predicted receiver

### Figure 2: Ablation comparison
- Bar chart: Top-1 accuracy and Top-3 accuracy across ablation configs
- Error bars from LOMO folds
- Horizontal line at random baseline

### Figure 3: Shot prediction AUC comparison
- Bar chart: AUC for unconditional vs oracle-receiver vs predicted-receiver
- Include XGBoost baseline and position-only GNN
- Error bars from LOMO folds
- Horizontal line at AUC=0.50 (random)

### Figure 4: Two-stage benefit
- Scatter: x-axis = receiver prediction accuracy, y-axis = conditional shot AUC
- One point per LOMO fold
- Shows whether better receiver prediction → better shot prediction

### Figure 5: Detection rate sensitivity
- x-axis = detection rate threshold for inclusion
- y-axis = model performance (AUC or accuracy)
- Shows how data quality filtering affects results

**Deliver**: Matplotlib/seaborn code for all figures. Save as PDF for LaTeX inclusion.

---

## Task 7: DFL Integration (Bonus)

I also have 57 corner kicks from the DFL Bundesliga dataset (25fps optical tracking). These have:
- Clean (x, y) for all 22 players (no extrapolation)
- Derivable velocity vectors at higher temporal resolution
- Different coordinate system (needs normalization)
- No event-level annotations (corners identified from ball position)

**If time permits**, merge the 57 DFL corners with the 80 SkillCorner corners for a combined 137-corner dataset. This requires:
- Coordinate normalization to common reference frame
- Feature alignment (DFL has no `is_detected` flag — all detected)
- Domain adaptation consideration (A-League vs Bundesliga)
- Source domain indicator as graph-level feature

This is lower priority than the core pipeline. Only attempt if Tasks 1-6 are complete and working.

---

## Constraints & Priorities

1. **March 16 defense deadline** — Code must work, not be perfect. Prefer simple and correct over complex and buggy.
2. **80 corners** — Every design decision must account for extreme data scarcity. When in doubt, simpler model.
3. **Reproducibility** — All random seeds fixed. All results must be reproducible from a single script run.
4. **Thesis-ready output** — Results tables and figures should be directly insertable into LaTeX.
5. **Honest reporting** — If the model doesn't beat random, report that. Negative results are valid.

### Priority order
1. Task 1 (extraction) — nothing works without clean data
2. Task 2 (graph construction) — foundation for everything
3. Task 3 (model) — core contribution
4. Task 4 (training/eval) — produces thesis results
5. Task 5 (baselines) — contextualizes results
6. Task 6 (visualization) — makes it presentable
7. Task 7 (DFL merge) — only if time allows

---

## Code Structure

```
corner_prediction/
├── data/
│   ├── extract_corners.py          # Task 1: raw data → corner records
│   ├── build_graphs.py             # Task 2: corner records → PyG Data objects
│   └── dataset.py                  # CornerKickDataset class
├── models/
│   ├── backbone.py                 # GNN backbone (pretrained or from-scratch)
│   ├── receiver_head.py            # Stage 1: receiver prediction head
│   ├── shot_head.py                # Stage 2: conditional shot prediction head
│   └── two_stage.py                # Combined two-stage model
├── baselines/
│   ├── random_baseline.py
│   ├── heuristic_receiver.py
│   ├── xgboost_baseline.py
│   └── mlp_baseline.py
├── training/
│   ├── train_receiver.py           # Stage 1 training loop
│   ├── train_shot.py               # Stage 2 training loop
│   ├── evaluate.py                 # LOMO cross-validation
│   ├── permutation_test.py         # Statistical validation
│   └── ablation.py                 # Run all ablation configs
├── visualization/
│   ├── plot_corner.py              # Single corner visualization
│   ├── plot_results.py             # Results bar charts
│   └── plot_sensitivity.py         # Detection rate sensitivity
├── config.py                       # All hyperparameters, paths, seeds
├── run_all.py                      # Single entry point: extract → train → evaluate → plot
└── README.md
```

**Deliver**: The complete codebase following this structure. Every script should be runnable independently AND via `run_all.py`. Include a README with setup instructions and expected runtime.

---

## Reference: TacticAI Comparison Points

For the thesis, I need to compare my results against these benchmarks:

| Metric | TacticAI | My target | Random baseline |
|--------|----------|-----------|----------------|
| Receiver Top-1 | ~0.35 (estimated) | >0.20 | ~0.147 |
| Receiver Top-3 | 0.782 | >0.50 | ~0.441 |
| Shot F1 (unconditional) | 0.52 | >0.30 | ~0.25 |
| Shot F1 (two-stage) | 0.71 | >0.40 | ~0.25 |
| Shot AUC (unconditional) | N/R | >0.55 | 0.50 |
| Shot AUC (two-stage) | N/R | >0.60 | 0.50 |

These targets are deliberately modest. Exceeding random baseline with statistical significance on 80 corners would be a real contribution. Matching TacticAI is not the goal.

---

## What Success Looks Like

**Minimum viable result for thesis:**
- Receiver top-3 accuracy > 50% (above 44% random baseline)
- Conditional shot AUC > unconditional shot AUC (two-stage helps)
- Velocity features improve over position-only (connects to 7.5 ECTS finding)

**Bonus results:**
- Permutation test confirms signal (p < 0.05 for any metric)
- Clear ablation showing velocity is necessary
- DFL integration shows cross-source generalization

**Acceptable negative result:**
- Everything at random chance, but with proper statistical validation
- Documented as evidence that 80 corners is insufficient even with velocity + GNN
- Still contributes methodology + data quality analysis
