# CornerTactics Dataset - Quick Reference

A quick reference guide for working with the CornerTactics dataset.

---

## Dataset at a Glance

```
📦 Dataset: statsbomb_temporal_augmented_with_receiver_v2.pkl
📊 Size: 5,814 graphs (1,118 unique corners × 5.2 augmentation)
✅ Coverage: 100% receiver labels
🎯 Purpose: GNN-based corner kick outcome prediction
```

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total Graphs | 5,814 |
| Unique Corners | 1,118 |
| Receiver Coverage | 100% |
| Attacking Receivers | 3,645 (62.7%) |
| Defending Receivers | 2,169 (37.3%) |
| Avg Players/Graph | 19 |
| Avg Edges/Graph | 166 |
| Augmentation | 5.2× |

---

## Quick Load

```python
import pickle

# Load dataset
with open("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver_v2.pkl", 'rb') as f:
    graphs = pickle.load(f)

# Access graph
graph = graphs[0]
print(graph.corner_id)          # Unique ID
print(graph.num_nodes)          # Number of players
print(graph.node_features)      # [num_nodes, 14] features
print(graph.receiver_player_name)  # Receiver name
```

---

## Data Structure

### CornerGraph Object

```python
graph.corner_id              # str: Unique identifier
graph.match_id               # int: StatsBomb match ID
graph.num_nodes              # int: Number of players (8-22)
graph.num_edges              # int: Number of connections (26-220)
graph.node_features          # np.array [num_nodes, 14]
graph.edge_index             # np.array [2, num_edges]
graph.edge_features          # np.array [num_edges, 6]
graph.teams                  # List[str]: ['attacking', 'defending', ...]
graph.receiver_player_id     # int: Player ID
graph.receiver_player_name   # str: Player name
graph.receiver_location      # List[float]: [x, y]
graph.receiver_node_index    # int: Graph node index (0-based)
graph.outcome                # str: 'goal', 'shot', 'clearance', etc.
```

---

## Feature Dimensions

### Node Features (14D)

| Index | Feature | Description |
|-------|---------|-------------|
| 0-1 | `x, y` | Position on pitch (0-120, 0-80) |
| 2-3 | `dist_to_goal, dist_to_ball` | Distances |
| 4-5 | `vx, vy` | Velocity components |
| 6-7 | `vel_mag, vel_angle` | Speed & direction |
| 8-9 | `angle_to_goal, angle_to_ball` | Angles |
| 10-11 | `team_flag, in_penalty_box` | Binary flags |
| 12-13 | `num_nearby, density` | Crowding metrics |

### Edge Features (6D)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `distance` | Normalized distance |
| 1-2 | `rel_vx, rel_vy` | Relative velocity |
| 3 | `rel_vel_mag` | Speed difference |
| 4-5 | `angle_sin, angle_cos` | Angle encoding |

---

## Common Operations

### Filter by Receiver Team

```python
# Attacking receivers
attacking = [g for g in graphs if g.teams[g.receiver_node_index] == 'attacking']

# Defending receivers
defending = [g for g in graphs if g.teams[g.receiver_node_index] == 'defending']
```

### Filter by Temporal Frame

```python
# Original frames only (t=0s)
original = [g for g in graphs if '_t+0.0' in g.corner_id]

# Specific temporal frame
t_minus_2 = [g for g in graphs if '_t-2.0' in g.corner_id]
```

### Get Unique Base Corners

```python
# Extract base corner IDs (without temporal suffix)
base_corners = set()
for g in graphs:
    base_id = g.corner_id.split('_t')[0].split('_mirror')[0]
    base_corners.add(base_id)

print(f"Unique corners: {len(base_corners)}")  # 1,118
```

### Train/Val/Test Split

```python
from src.data.receiver_data_loader import get_split_indices

# Corner-based split (prevents temporal leakage)
train_idx, val_idx, test_idx = get_split_indices(
    graphs,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)
```

---

## Receiver Labeling Info

### Methodology (v2)

- **Definition**: First player to touch ball after corner
- **Time Window**: 270 seconds (captures VAR/injury delays)
- **Includes**: Attacking AND defending players
- **Includes**: Corner taker (short corners)
- **Valid Events**: Pass, Shot, Duel, Interception, Clearance, Miscontrol, Ball Receipt

### Evolution

| Version | Coverage | Key Change |
|---------|----------|------------|
| v1 (CSV) | 60.1% | Attacking only, 5s window |
| v2 (Events) | 100.0% | Both teams, 270s window ✅ |

---

## Temporal Augmentation

### Frame Types

```python
# Check temporal frame
if '_t-2.0' in graph.corner_id:     # 2s before corner
if '_t-1.0' in graph.corner_id:     # 1s before corner
if '_t+0.0' in graph.corner_id:     # Corner moment (original)
if '_t+1.0' in graph.corner_id:     # 1s after corner
if '_t+2.0' in graph.corner_id:     # 2s after corner

# Check if mirrored
if '_mirror' in graph.corner_id:    # Mirrored augmentation
```

### Distribution

- **Original (t=0s)**: 1,342 graphs (23.1%)
- **Augmented**: 4,472 graphs (76.9%)
- **Mirrored**: 224 graphs (3.9%)

---

## Validation Checks

```python
# Check receiver coverage
total = len(graphs)
with_receiver = sum(1 for g in graphs if g.receiver_player_id is not None)
print(f"Coverage: {with_receiver/total*100:.1f}%")  # Should be 100%

# Check node count distribution
import numpy as np
num_nodes = [g.num_nodes for g in graphs]
print(f"Avg nodes: {np.mean(num_nodes):.1f}")  # Should be ~19

# Check team balance
attacking = sum(1 for g in graphs if g.teams[g.receiver_node_index] == 'attacking')
defending = sum(1 for g in graphs if g.teams[g.receiver_node_index] == 'defending')
print(f"Attacking: {attacking/total*100:.1f}%")  # Should be ~63%
print(f"Defending: {defending/total*100:.1f}%")  # Should be ~37%
```

---

## Data Loader Example

```python
from src.data.receiver_data_loader import ReceiverDataLoader

# Initialize loader
loader = ReceiverDataLoader(
    graph_path="data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver_v2.pkl",
    batch_size=32,
    mask_velocities=True  # Mask velocities for receiver prediction
)

# Get data splits
train_loader, val_loader, test_loader = loader.get_data_loaders()

# Iterate
for batch in train_loader:
    node_features = batch['node_features']  # [batch_size, num_nodes, 14]
    edge_index = batch['edge_index']        # [batch_size, 2, num_edges]
    receiver_labels = batch['receiver']     # [batch_size] node indices
    # ... train model
```

---

## Common Pitfalls

### ❌ Don't: Random Split by Graphs

```python
# BAD: Temporal frames leak across splits
random.shuffle(graphs)
train = graphs[:4000]
test = graphs[4000:]
```

### ✅ Do: Split by Base Corners

```python
# GOOD: Group temporal frames together
from src.data.receiver_data_loader import get_split_indices
train_idx, val_idx, test_idx = get_split_indices(graphs)
```

### ❌ Don't: Use Raw Positions as Features

```python
# BAD: Positions contain temporal info
features = graph.node_features[:, :2]  # x, y only
```

### ✅ Do: Mask Velocities for Receiver Prediction

```python
# GOOD: Mask velocities to prevent leakage
loader = ReceiverDataLoader(mask_velocities=True)
```

---

## File Locations

```
📁 Dataset
  └─ data/graphs/adjacency_team/
     └─ statsbomb_temporal_augmented_with_receiver_v2.pkl

📁 Documentation
  ├─ docs/DATASET_DOCUMENTATION.md        (Full docs)
  ├─ docs/DATASET_STATISTICS.json         (JSON stats)
  └─ docs/QUICK_REFERENCE.md              (This file)

📁 Scripts
  ├─ scripts/preprocessing/add_receiver_labels_v2.py
  └─ scripts/analysis/generate_dataset_documentation.py

📁 Source
  ├─ src/receiver_labeler.py              (Receiver extraction)
  └─ src/data/receiver_data_loader.py     (PyTorch loader)
```

---

## Support

- **Full Documentation**: `docs/DATASET_DOCUMENTATION.md`
- **Statistics**: `docs/DATASET_STATISTICS.json`
- **Issues**: https://github.com/MahmoodSeoud/CornerTactics/issues

---

**Version**: v2
**Last Updated**: November 2025
**Status**: Production Ready ✅
