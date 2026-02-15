# Task 2: Graph Construction — Implementation Plan

## Overview

Build PyTorch Geometric `Data` objects from the 86 extracted corner records (`corner_prediction/data/extracted_corners.pkl`). Create two files:

1. `corner_prediction/data/build_graphs.py` — Graph construction logic
2. `corner_prediction/data/dataset.py` — `CornerKickDataset` class
3. `tests/corner_prediction/test_build_graphs.py` — Tests

## Files to Create

### File 1: `corner_prediction/data/build_graphs.py`

#### Constants

```python
# Position group categories for one-hot encoding (4 categories)
POSITION_GROUPS = ["GK", "DEF", "MID", "FWD"]

# Map from role_abbrev (stored in extracted records) to coarse group
ROLE_TO_GROUP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "DM": "MID", "LM": "MID", "RM": "MID", "AM": "MID",
    "LW": "FWD", "RW": "FWD", "LF": "FWD", "RF": "FWD", "CF": "FWD",
    "SUB": "MID",
}

HALF_LENGTH = 52.5  # from extract_corners.py
HALF_WIDTH = 34.0
```

#### Function: `build_node_features(player: dict) -> list[float]`

Build 13-dim feature vector per player as specified in pipeline doc:

| Index | Feature | Source | Range |
|-------|---------|--------|-------|
| 0 | x | `player["x"] / HALF_LENGTH` | [-1, 1] → normalized |
| 1 | y | `player["y"] / HALF_WIDTH` | [-1, 1] → normalized |
| 2 | vx | `player["vx"]` | raw m/s |
| 3 | vy | `player["vy"]` | raw m/s |
| 4 | speed | `player["speed"]` | raw m/s |
| 5 | is_attacking | `float(player["is_attacking"])` | 0.0 or 1.0 |
| 6 | is_corner_taker | `float(player["is_corner_taker"])` | 0.0 or 1.0 |
| 7 | is_goalkeeper | `float(player["is_goalkeeper"])` | 0.0 or 1.0 |
| 8 | is_detected | `float(player["is_detected"])` | 0.0 or 1.0 |
| 9 | group_GK | one-hot | 0.0 or 1.0 |
| 10 | group_DEF | one-hot | 0.0 or 1.0 |
| 11 | group_MID | one-hot | 0.0 or 1.0 |
| 12 | group_FWD | one-hot | 0.0 or 1.0 |

**Design note**: Normalize x,y by dividing by half-pitch dims. This gives [-1, 1] range which preserves sign (useful for corner side info). Velocities left raw — small enough range (max ~12 m/s) and meaningful units.

#### Function: `build_edge_features(node_i_xy, node_j_xy, team_i, team_j) -> list[float]`

4-dim edge features as specified:

| Index | Feature | Computation |
|-------|---------|-------------|
| 0 | dx | `x_j - x_i` (normalized coords) |
| 1 | dy | `y_j - y_i` (normalized coords) |
| 2 | distance | `sqrt(dx² + dy²)` |
| 3 | same_team | `1.0 if same team else 0.0` |

#### Function: `build_knn_edges(positions: Tensor, k: int = 6) -> Tensor`

- Input: `[22, 2]` tensor of (x, y) positions
- Compute pairwise Euclidean distances with `scipy.spatial.distance.cdist`
- For each node, select k nearest neighbors
- Return `edge_index` of shape `[2, 22*k]` (directed: i→j for each neighbor j of i)
- Handle degenerate case: if n_nodes ≤ k, fall back to fully connected

#### Function: `build_dense_edges(n_nodes: int) -> Tensor`

- Return `edge_index` of shape `[2, n*(n-1)]` — all pairs i≠j

#### Function: `corner_record_to_graph(record: dict, edge_type: str = "knn", k: int = 6) -> Data`

Main conversion function. Steps:

1. Build node features for all 22 players → `x` tensor `[22, 13]`
2. Build edge_index via KNN or dense
3. Build edge_attr from node positions and team flags → `[n_edges, 4]`
4. Build receiver labels:
   - `receiver_mask`: `[22]` bool — True for attacking outfield players (is_attacking=True AND is_goalkeeper=False)
   - `receiver_label`: `[22]` float — 1.0 at receiver node index, 0.0 elsewhere
   - `has_receiver_label`: bool scalar
5. Build shot labels:
   - `shot_label`: 0 or 1 (from `record["lead_to_shot"]`)
   - `goal_label`: 0 or 1 (from `record["lead_to_goal"]`)
6. Graph-level features:
   - `corner_side`: 0.0 for "left", 1.0 for "right"
7. Metadata:
   - `match_id`: str (for LOMO splits)
   - `corner_id`: str
   - `detection_rate`: float

Return `Data(x=..., edge_index=..., edge_attr=..., receiver_mask=..., receiver_label=..., has_receiver_label=..., shot_label=..., goal_label=..., corner_side=..., match_id=..., corner_id=..., detection_rate=...)`

#### Function: `build_graph_dataset(records: list[dict], edge_type="knn", k=6) -> list[Data]`

- Load records from pickle or accept list
- Convert each record → Data object
- Return list of Data objects

#### Function: `print_summary(graphs: list[Data])`

Print:
- Total graphs, shots/no-shots counts
- Matches represented
- Node feature dimensions and ranges (min/max/mean/std per feature)
- Edge count distribution (min/max/mean)
- Receiver label coverage
- Detection rate distribution

#### CLI (`if __name__ == "__main__"`)

```
python -m corner_prediction.data.build_graphs \
    --input corner_prediction/data/extracted_corners.pkl \
    --output corner_prediction/data/graphs.pkl \
    --edge-type knn --k 6
```

### File 2: `corner_prediction/data/dataset.py`

#### Class: `CornerKickDataset(InMemoryDataset)`

- `__init__(root, records=None, edge_type="knn", k=6, transform=None, pre_transform=None)`
- `raw_file_names` → `["extracted_corners.pkl"]`
- `processed_file_names` → `["graphs_knn6.pt"]` (varies by edge_type/k)
- `process()` → loads records, calls `build_graph_dataset()`, saves via `torch.save`
- Standard PyG `__len__`, `__getitem__`

#### Function: `lomo_split(dataset, held_out_match_id) -> (train_data, test_data)`

- Returns two lists of Data objects: train (all matches except held_out) and test (held_out match only)
- Used for leave-one-match-out cross-validation

#### Function: `get_match_ids(dataset) -> list[str]`

- Returns sorted unique match IDs from dataset

### File 3: `tests/corner_prediction/test_build_graphs.py`

All tests use synthetic corner records — no real data needed.

#### Synthetic helpers

- `_make_player(player_id, x, y, vx, vy, is_attacking, role, ...)` — builds a player dict
- `_make_corner_record(n_attacking=11, n_defending=11, has_receiver=True, lead_to_shot=False, ...)` — builds a full 22-player corner record with realistic positions

#### Test classes

**`TestNodeFeatures`** (~6 tests):
- `test_feature_dimension` — output is 13-dim
- `test_position_normalization` — x=52.5 maps to 1.0, x=-52.5 to -1.0
- `test_velocity_passthrough` — vx/vy/speed preserved as-is
- `test_binary_flags` — is_attacking/is_corner_taker/is_goalkeeper/is_detected are 0.0 or 1.0
- `test_position_group_onehot` — exactly one of 4 group features is 1.0
- `test_unknown_role_defaults_mid` — unknown role maps to MID

**`TestEdgeConstruction`** (~5 tests):
- `test_knn_edge_count` — 22 nodes, k=6 → 132 edges
- `test_knn_degenerates_to_dense` — 5 nodes, k=6 → 20 edges (fully connected)
- `test_dense_edge_count` — 22 nodes → 462 edges
- `test_no_self_loops` — no edge i→i
- `test_edge_feature_dimension` — each edge has 4 features

**`TestEdgeFeatures`** (~3 tests):
- `test_same_team_flag` — 1.0 when both attacking, 0.0 when different
- `test_distance_nonnegative` — distance ≥ 0
- `test_dx_dy_antisymmetric` — dx(i→j) = -dx(j→i)

**`TestGraphConstruction`** (~6 tests):
- `test_full_graph_shape` — x=[22,13], edge_index=[2,132], edge_attr=[132,4]
- `test_receiver_mask_only_attacking_outfield` — mask True for attacking non-GK players
- `test_receiver_label_one_hot` — exactly one 1.0 in receiver_label when has_receiver_label
- `test_no_receiver_label` — receiver_label all zeros when has_receiver_label=False
- `test_shot_label` — matches input record
- `test_metadata_preserved` — match_id, corner_id, detection_rate on Data object

**`TestDataset`** (~4 tests):
- `test_dataset_length` — matches number of records
- `test_lomo_split` — train + test = total, test only contains held_out match
- `test_get_match_ids` — returns correct unique match IDs
- `test_indexing` — dataset[0] returns Data object

## Step-by-step execution order

1. Create `corner_prediction/data/build_graphs.py` with all functions
2. Create `corner_prediction/data/dataset.py` with `CornerKickDataset` and LOMO split
3. Create `tests/corner_prediction/test_build_graphs.py` with all tests
4. Run tests — verify all pass
5. Run build_graphs.py on real data as smoke test, print summary

## Acceptance criteria

- All tests pass
- `build_graphs.py` processes all 86 corners without errors
- Summary shows: 86 graphs, 22 nodes each, 13 features, correct shot/receiver counts
- No NaN/Inf in any feature tensors
- KNN with k=6 produces 132 edges per graph
- Receiver mask correctly identifies attacking outfield players only
