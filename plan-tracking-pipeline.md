# Implementation Plan: Complete Tracking Data Extraction Pipeline

## Overview

Transform 86 extracted SkillCorner corners + 57 existing DFL corners into USSF-schema PyTorch Geometric graphs for GNN training. Optionally extract more corners from SkillCorner opendata v2 if StatsBomb event overlap exists.

**Current state:** 86 SkillCorner corners in unified JSON, 57 DFL corners in PyG pickle only, no GNN converter.
**End state:** ~143+ corners as USSF-schema PyG graphs with train/val/test splits, ready for transfer learning.

---

## Step 1: Build the GNN Graph Converter

**File:** `tracking_extraction/graph_converter.py`

**Purpose:** Convert unified JSON (`CornerTrackingData`) → USSF-schema PyTorch Geometric `Data` objects.

**Implementation details:**

1.1. Function `corner_to_ussf_graph(corner: CornerTrackingData, adjacency: str = "dense") -> Data`:
  - Takes a single `CornerTrackingData` object (from `core.py`)
  - Uses the **delivery frame** (the single snapshot at corner delivery) to build one graph per corner
  - This matches the existing 57 DFL corners format: each corner = 1 graph (not a temporal sequence)

1.2. Node feature engineering (12 features, matching USSF schema exactly):
  - `x, y`: Normalize player positions to [0, 1] by dividing by pitch dimensions (105, 68)
  - `vx, vy`: Convert velocity (m/s) to unit vector. If speed=0, use (1, 0).
  - `velocity_mag`: `sqrt(vx² + vy²)` normalized to [0, 1] by dividing by `max_velocity=10.0`
  - `velocity_angle`: `atan2(vy, vx)` normalized to [0, 1] via `(angle + pi) / (2*pi)`
  - `dist_goal`: Euclidean distance to goal center (105, 34) in normalized coords, clipped to [0, 1]
  - `angle_goal`: `atan2(goal_y - y, goal_x - x)` normalized to [0, 1]
  - `dist_ball`: Euclidean distance to ball in normalized coords, clipped to [0, 1]
  - `angle_ball`: `atan2(ball_y - y, ball_x - x)` normalized to [0, 1]
  - `attacking_team_flag`: 1.0 for "attacking", 0.0 for "defending", 0.5 for "unknown"
  - `potential_receiver`: 0.0 for all players (not applicable to corner kicks)
  - Ball node: same features but with `attacking_team_flag=-1.0`, `dist_ball=0.0`

1.3. Edge feature engineering (6 features, dense adjacency):
  - For dense adjacency: all-to-all edges (n*(n-1) directed edges)
  - `player_distance`: Normalized Euclidean distance between nodes (in normalized coords)
  - `speed_difference`: Signed difference in velocity magnitude, normalized to [-1, 1]
  - `positional_sine_angle`: `(sin(angle_between_positions) + 1) / 2`
  - `positional_cosine_angle`: `(cos(angle_between_positions) + 1) / 2`
  - `velocity_sine_angle`: `(sin(angle_between_velocities) + 1) / 2`
  - `velocity_cosine_angle`: `(cos(angle_between_velocities) + 1) / 2`

1.4. Function `convert_dataset(corners: List[CornerTrackingData], adjacency: str = "dense") -> List[dict]`:
  - Converts a list of corners to the output format
  - Each entry: `{'graphs': [Data(...)], 'labels': {'shot_binary': int, 'goal_binary': int, ...}, 'match_id': str, 'corner_time': float}`
  - Maps `outcome="shot"` → `shot_binary=1`, `outcome="no_shot"` → `shot_binary=0`
  - `goal_binary=0` for all (we don't distinguish goals from shots in unified format)

1.5. Function `save_graph_dataset(dataset: List[dict], path: Path)`:
  - Pickle dump, same format as existing `dfl_corners_ussf_format_dense.pkl`

**Test criteria:**
  - Output graph `x` tensor shape = `[n_players+1, 12]` (includes ball node)
  - All node features in [0, 1] range (except attacking_team_flag which can be -1, 0, 0.5, 1)
  - Edge index shape = `[2, n*(n-1)]` for dense adjacency
  - Edge attr shape = `[n*(n-1), 6]`
  - All edge features in [0, 1] range
  - Output dict has same keys as existing `dfl_corners_ussf_format_dense.pkl` entries

---

## Step 2: Build the CLI Script

**File:** `tracking_extraction/scripts/build_graph_dataset.py`

**Purpose:** CLI to convert unified JSON dataset → USSF-schema pickle for training.

2.1. Arguments:
  - `--input-dir`: Path to unified JSON directory (default: `tracking_extraction/output/unified`)
  - `--output-path`: Output pickle path (default: `transfer_learning/data/multi_source_corners_dense.pkl`)
  - `--adjacency`: "dense" or "normal" (default: "dense")
  - `--verbose`

2.2. Flow:
  - Load corners from `input-dir` using `core.load_dataset()`
  - Convert using `graph_converter.convert_dataset()`
  - Save pickle using `graph_converter.save_graph_dataset()`
  - Print summary: total corners, shot/no_shot distribution, source breakdown, mean nodes/edges per graph

**Test criteria:**
  - Script runs end-to-end on existing `tracking_extraction/output/skillcorner/` data
  - Output pickle is loadable and graphs have correct shapes

---

## Step 3: Run DFL Adapter → Unified JSON

**File:** No new files. Use existing `tracking_extraction/dfl_adapter.py`.

3.1. Check raw DFL data availability at `/home/mseo/CornerTactics/data/dfl/`:
  - Need XML position and event files for 7 matches
  - If files exist: run `convert_dfl_from_paths()` for each match
  - Save output to `tracking_extraction/output/dfl/`

3.2. Write a small script `tracking_extraction/scripts/extract_dfl.py`:
  - Similar to `extract_skillcorner.py`
  - Iterates over DFL match directories, runs adapter, saves unified JSON
  - Cross-validates: count should be 57, matches should be the same 7 match IDs

3.3. If raw DFL data is NOT accessible (files may have been cleaned up):
  - Alternative: write a converter from existing `dfl_corners_ussf_format_dense.pkl` → unified JSON
  - This is lossy (can't recover raw m/s velocities from unit vectors) but gives us the unified format for consistency
  - The GNN converter will re-normalize anyway, so the lossy roundtrip doesn't matter for graph output

**Test criteria:**
  - 57 DFL corners in unified JSON format at `tracking_extraction/output/dfl/`
  - Shot labels match existing: 19 shots, 2 goals
  - 7 match IDs match existing dataset

---

## Step 4: Re-consolidate All Sources

**File:** No new files. Use existing `tracking_extraction/scripts/consolidate_dataset.py`.

4.1. Run consolidation with both sources:
  ```
  python -m tracking_extraction.scripts.consolidate_dataset \
    --skillcorner-dir tracking_extraction/output/skillcorner \
    --dfl-dir tracking_extraction/output/dfl \
    --output-dir tracking_extraction/output/unified
  ```

4.2. Expected output: ~143 corners (86 SkillCorner + 57 DFL)

**Test criteria:**
  - Unified dataset has corners from both sources
  - No duplicate corner IDs
  - Quality validation passes

---

## Step 5: Convert Unified Dataset → USSF-Schema Graphs

5.1. Run the graph converter:
  ```
  python -m tracking_extraction.scripts.build_graph_dataset \
    --input-dir tracking_extraction/output/unified \
    --output-path transfer_learning/data/multi_source_corners_dense.pkl
  ```

5.2. Validate output:
  - Compare SkillCorner-derived graphs with existing DFL graphs (similar feature distributions)
  - Check node feature ranges match USSF schema
  - Verify shot label distribution

**Test criteria:**
  - Output pickle has ~143 entries
  - Each graph: x=[N, 12], edge_index=[2, E], edge_attr=[E, 6], pos=[N, 2]
  - Feature distributions are reasonable (no NaN, no inf, ranges within expected bounds)

---

## Step 6: Create Train/Val/Test Splits

6.1. Add function `create_splits(dataset, test_size=0.15, val_size=0.15, seed=42)` to `graph_converter.py`:
  - Match-based stratification: all corners from the same match go to the same split
  - Stratify by outcome (shot/no_shot) as much as possible
  - With 17 matches (10 SkillCorner + 7 DFL), allocate ~2-3 matches to test, ~2-3 to val, rest to train

6.2. Save splits as separate pickle files:
  - `transfer_learning/data/multi_source_train.pkl`
  - `transfer_learning/data/multi_source_val.pkl`
  - `transfer_learning/data/multi_source_test.pkl`
  - Also save a combined file with split labels

**Test criteria:**
  - No match overlap between splits
  - Each split has both shot and no_shot examples
  - Split ratios approximately 70/15/15

---

## Step 7: Validation Report

7.1. Add function to `graph_converter.py` or a new script that prints:
  - Per-source statistics (n corners, shot rate, mean players/frame)
  - Per-split statistics
  - Feature distribution comparison: SkillCorner vs DFL (mean, std of each of the 12 node features)
  - Flag any systematic differences between sources

**Test criteria:**
  - Report is generated without errors
  - No NaN or inf values in any features
  - Feature distributions are within reasonable ranges

---

## Execution Order

1. Step 1 (graph_converter.py) — the critical new code
2. Step 2 (CLI script) — thin wrapper
3. Step 3 (DFL extraction) — depends on data availability, may need fallback
4. Step 4 (consolidation) — just running existing script
5. Step 5 (convert to graphs) — just running new script
6. Step 6 (splits) — small addition to converter
7. Step 7 (validation) — reporting

**Estimated new code:** ~300-400 lines across 2-3 files.

---

## Files to create/modify

| Action | File | Lines |
|--------|------|-------|
| CREATE | `tracking_extraction/graph_converter.py` | ~250 |
| CREATE | `tracking_extraction/scripts/build_graph_dataset.py` | ~80 |
| CREATE | `tracking_extraction/scripts/extract_dfl.py` | ~60 |
| MODIFY | `tracking_extraction/__init__.py` | +2 lines (export graph_converter) |

## Out of scope (deferred)

- SkillCorner opendata v2 (9 top-5 league matches) — skip unless StatsBomb overlap confirmed
- SoccerNet GSR pipeline — requires GPU cluster setup, separate effort
- Training experiments with new data — separate task after this pipeline is complete
