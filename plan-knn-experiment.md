# Plan: kNN Graph Adjacency Experiment

## Goal
Replace dense adjacency with kNN (k=5) graph construction based on Euclidean distance.
Retrain CrystalConv on the full 3,078 multi-source dataset. Report AUC.

## Hypothesis
Dense graphs (90-342 edges) drown signal in noise for corners, where spatial proximity
defines tactical relationships (marking assignments). kNN (k=5) preserves local structure.

## Baseline (exp1 dense results)
- MS-A (pretrained+frozen): AUC = 0.493 ± 0.019
- MS-B (pretrained+fine-tuned): AUC = 0.489 ± 0.019
- MS-C (random init): AUC = 0.509 ± 0.016
- MS-D (majority): AUC = 0.500 ± 0.000

---

## Implementation Plan

### Step 1: Add `rebuild_knn_edges()` to `transfer_learning/multi_source_utils.py`

**Location**: After `zero_velocity_features()` (line ~199), before training utilities section.

**Function signature**:
```python
def rebuild_knn_edges(data_list: List[Data], k: int = 5) -> List[Data]:
```

**Logic**:
1. For each PyG Data object in data_list:
   a. Extract node features: `x` (shape [n_nodes, 12])
   b. Extract positions: use `x[:, 0:2]` (normalized x,y from node features — these are
      the actual coordinates used in distance calculations, matching how edge features
      reference positions)
   c. Compute pairwise Euclidean distance matrix: `scipy.spatial.distance.cdist(pos, pos)`
   d. Set diagonal to inf (no self-loops)
   e. For each node i, find k nearest neighbors (or n_nodes-1 if n_nodes <= k)
   f. Build directed edge_index: i → j for each j in kNN(i)
   g. Recompute edge_attr using the same 6-feature formula from graph_converter.py:
      - For each edge (i,j): extract node_i = x[i].numpy(), node_j = x[j].numpy()
      - Compute: dist_norm, speed_diff, pos_sine, pos_cosine, vel_sine, vel_cosine
      - This is identical to `_compute_edge_features()` in graph_converter.py
   h. Create new Data object with same x, y, match_id, etc. but new edge_index and edge_attr
2. Return new list

**Edge case**: When n_nodes <= k+1, kNN degenerates to dense (every node connects to
every other). This is correct — small graphs stay dense.

**Import needed**: `from scipy.spatial.distance import cdist` (already used in graph_converter.py)

### Step 2: Add `_compute_edge_features_batch()` helper in same file

**Location**: Just before `rebuild_knn_edges()`.

**Purpose**: Vectorized edge feature computation matching graph_converter.py exactly.

```python
def _compute_edge_features_batch(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
```

**Logic** (vectorized over all edges):
1. src_nodes = x[edge_index[0]]  # shape [n_edges, 12]
2. dst_nodes = x[edge_index[1]]  # shape [n_edges, 12]
3. Compute all 6 features in batch using torch ops:
   - dist_norm = clamp(sqrt((xj-xi)^2 + (yj-yi)^2) / sqrt(2), 0, 1)
   - speed_diff = vel_mag_j - vel_mag_i
   - pos_angle = atan2(yj-yi, xj-xi) → sin/cos normalized to [0,1]
   - vel_angle = atan2(cross, dot) → sin/cos normalized to [0,1]
4. Stack into [n_edges, 6] tensor

### Step 3: Create `transfer_learning/exp5_knn_adjacency.py`

**Structure**: Follows exp1_full_dataset.py pattern exactly.

**Conditions**:
- **kNN-A**: USSF pretrained + frozen backbone, kNN edges (k=5)
- **kNN-B**: USSF pretrained + fine-tuned backbone, kNN edges (k=5)
- **kNN-C**: Random init, kNN edges (k=5)
- **kNN-D**: Majority baseline (same as MS-D — adjacency-independent)

**Flow**:
1. Load splits via `load_splits()` (dense pickle data)
2. Convert to PyG via `prepare_pyg_data()`
3. **NEW**: Call `rebuild_knn_edges(data, k=5)` on train, val, test
4. Run same training loop as exp1 for each condition (5 seeds)
5. Bootstrap CI + permutation test (N=20)
6. Print comparison table: kNN results vs dense baseline (load exp1 results)
7. Save to `results/multi_source_experiments/exp5_knn_adjacency.pkl`

**Hyperparameters**: Same as exp1 (epochs=100, batch_size=32, patience=15, head_hidden=64).

**CLI args**: Same as exp1 + `--k` (default 5) for kNN neighborhood size.

### Step 4: Add exp5 to SLURM batch script

**File**: `scripts/slurm/run_multi_source_exp.sbatch`

**Change**: Add case 5 for the kNN experiment:
```
    5)
        echo ">>> Experiment 5: kNN Adjacency (k=5)"
        python3 -u transfer_learning/exp5_knn_adjacency.py \
            $COMMON_ARGS --epochs 100 --patience 15 --n-permutations 20 --k 5
        ;;
```

Update the header comment to mention task 5.

### Step 5: Add test for `rebuild_knn_edges()`

**File**: New test in `tests/tracking_extraction/test_knn_edges.py`
(or could add to existing test file, but this is a new utility)

**Tests**:
1. **test_knn_basic**: 10 nodes, k=5 → each node has exactly 5 outgoing edges (50 total)
2. **test_knn_small_graph**: 4 nodes, k=5 → degenerates to dense (each node has 3 edges = 12 total)
3. **test_knn_edge_features_match**: Compare `_compute_edge_features_batch()` output against
   `graph_converter._compute_edge_features()` for same node pairs — must be numerically identical
4. **test_knn_preserves_labels**: Ensure y, match_id, source survive the transform
5. **test_knn_symmetry**: If i→j exists, j→i may or may not exist (directed kNN is asymmetric)
6. **test_knn_no_self_loops**: No edge (i,i) in output

---

## Execution Order
1. Step 2 — `_compute_edge_features_batch()` helper
2. Step 1 — `rebuild_knn_edges()` function
3. Step 5 — Tests (run to verify correctness)
4. Step 3 — `exp5_knn_adjacency.py` experiment script
5. Step 4 — SLURM batch update

## Success Criteria
- All 6 tests pass
- Experiment runs to completion on 5 seeds
- AUC results reported for all 4 conditions
- Comparison with dense baseline printed

## Files Modified
- `transfer_learning/multi_source_utils.py` (add 2 functions, ~60 lines)
- `scripts/slurm/run_multi_source_exp.sbatch` (add case 5, ~5 lines)

## Files Created
- `transfer_learning/exp5_knn_adjacency.py` (~240 lines, follows exp1 pattern)
- `tests/tracking_extraction/test_knn_edges.py` (~120 lines)
