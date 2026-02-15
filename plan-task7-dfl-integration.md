# Task 7: DFL Integration — Implementation Plan

## Goal
Merge 57 DFL Bundesliga corners with 86 SkillCorner A-League corners for a combined 143-corner dataset, then run the full two-stage evaluation pipeline on the combined data. Submit as SLURM jobs.

## Approach
Extract DFL corners from raw XML/tracking data via kloppy + XML parsing, producing flat records in the exact same format as SkillCorner's `extract_corners.py` output. Then merge, build graphs, and evaluate.

### Key design decisions
- **Corner identification**: Parse DFL event XML directly for `<CornerKick>` elements (kloppy doesn't expose these). Get `StartFrame`, `Team`, `Side` from the XML.
- **Tracking data**: Load via kloppy's sportec loader to get player positions, team assignments, starting positions, jersey numbers. Only keep the delivery frame + prev frame (for velocity).
- **Team assignment**: From kloppy `player.team` attribute (properly set on Player objects in tracking data).
- **Player roles**: Map kloppy `player.starting_position` (e.g., "Goalkeeper", "Right Back", "Striker") to the same abbreviation scheme as SkillCorner (GK, CB, RB, CF, etc.).
- **Coordinates**: kloppy gives [0,1] normalized. Convert to center-origin meters: `x_m = x_norm * 105 - 52.5`, `y_m = y_norm * 68 - 34`. Then normalize direction (attacking toward +x).
- **Velocities**: Backward difference at 25fps: `vx = (x[t] - x[t-1]) * 25.0` (converted from frame-diff to m/s using pitch dimensions).
- **Shot/goal labels**: Parse subsequent events in XML within 10s window for `<Shot>`, `<GoalScored>`, `<BlockedShot>` elements.
- **Receiver labels**: `has_receiver_label=False` for all DFL corners (no receiver annotation available).
- **Detection**: `is_detected=True` for all players (optical tracking, no extrapolation).
- **Source indicator**: Add `source` field ("dfl" or "skillcorner") to each record/graph for domain-aware analysis.

## Files to create/modify

### New files

1. **`corner_prediction/data/extract_dfl_corners.py`** — DFL corner extraction pipeline
2. **`corner_prediction/data/merge_datasets.py`** — Merge SkillCorner + DFL records
3. **`scripts/slurm/run_dfl_integration.sbatch`** — SLURM job script

### Modified files

4. **`corner_prediction/config.py`** — Add DFL match IDs, DFL data path, combined dataset paths
5. **`corner_prediction/data/build_graphs.py`** — Add `source` attribute to Data objects
6. **`corner_prediction/data/dataset.py`** — Support loading combined dataset
7. **`corner_prediction/run_all.py`** — Add `--combined` flag for DFL integration

---

## Step-by-step implementation

### Step 1: `corner_prediction/data/extract_dfl_corners.py`

```
Functions:
  DFL_MATCH_IDS = ['DFL-MAT-J03WMX', ..., 'DFL-MAT-J03WR9']
  DFL_DATA_DIR = PROJECT_ROOT / 'data' / 'dfl'
  DFL_FPS = 25.0

  DFL_POSITION_MAP = {
    "Goalkeeper": "GK",
    "Left Center Back": "CB", "Right Center Back": "CB",
    "Left Back": "LB", "Right Back": "RB",
    "Left Defensive Midfield": "DM", "Right Defensive Midfield": "DM",
    "Center Attacking Midfield": "AM",
    "Left Midfield": "LM", "Right Midfield": "RM",
    "Left Forward": "LF", "Right Forward": "RF", "Left Winger": "LW", "Right Winger": "RW",
    "Striker": "CF",
    "Unknown": "SUB",
  }

  find_corners_from_xml(event_xml_path) -> List[Dict]:
    - Parse XML, find <CornerKick> elements inside <Event> parents
    - Extract: StartFrame, EndFrame, Team, Side, Placing
    - Return list of {frame, team_id, side, ...}

  find_shots_from_xml(event_xml_path, corner_frame, window_frames=250) -> Dict:
    - Parse subsequent <Event> elements after corner_frame
    - Look for <Shot>, <GoalScored>, <BlockedShot>, <ChanceWithoutShot> within window
    - Return {lead_to_shot: bool, lead_to_goal: bool}

  load_dfl_tracking_frame(tracking_dataset, target_frame_time, period) -> Dict:
    - Search tracking.records for frame closest to target_frame_time
    - Return player positions/data from that frame and prev frame

  build_dfl_player_record(player, pdata, prev_pdata, corner_team_id, fps) -> Dict:
    - Convert [0,1] coords to center-origin meters
    - Compute velocity from backward difference
    - Map starting_position to role abbreviation
    - Return {player_id, x, y, vx, vy, speed, is_attacking, is_corner_taker, is_goalkeeper, role, is_detected, is_receiver}

  normalize_dfl_direction(players, ball, corner_team_id, tracking_metadata) -> (players, ball, corner_side):
    - Determine which direction corner_team attacks
    - Flip all coords/velocities if needed so attacking -> +x
    - Determine corner_side from ball y position

  extract_dfl_match_corners(match_id, data_dir) -> List[Dict]:
    - Load tracking via kloppy sportec
    - Parse event XML for corners and shots
    - For each corner: extract tracking frame, build player records, normalize, build labels
    - Return list of flat records in SkillCorner format

  extract_all_dfl_corners(data_dir) -> List[Dict]:
    - Process all 7 matches
    - Validate records (22 players, bounds, etc.)
    - Save as dfl_extracted_corners.pkl + .json

  CLI: python -m corner_prediction.data.extract_dfl_corners
```

### Step 2: `corner_prediction/data/merge_datasets.py`

```
Functions:
  merge_records(skillcorner_records, dfl_records) -> List[Dict]:
    - Add source="skillcorner" to each SC record
    - Add source="dfl" to each DFL record
    - Concatenate
    - Validate: unique corner_ids, consistent features
    - Return combined list

  CLI: python -m corner_prediction.data.merge_datasets
    --skillcorner corner_prediction/data/extracted_corners.pkl
    --dfl corner_prediction/data/dfl_extracted_corners.pkl
    --output corner_prediction/data/combined_corners.pkl
```

### Step 3: Modify `corner_prediction/data/build_graphs.py`

```
Changes:
  - In corner_record_to_graph(): add source attribute to Data object
    data.source = record.get("source", "skillcorner")
  - In print_summary(): add source distribution printout
```

### Step 4: Modify `corner_prediction/config.py`

```
Add:
  DFL_DATA_DIR = PROJECT_ROOT / "data" / "dfl"
  DFL_MATCH_IDS = [
    "DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH",
    "DFL-MAT-J03WOY", "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9",
  ]
  COMBINED_DATA_DIR = DATA_DIR  # Same dir, different filename
```

### Step 5: Modify `corner_prediction/data/dataset.py`

```
Changes:
  - CornerKickDataset: accept optional records_file parameter
    Default: "extracted_corners.pkl" (SkillCorner only)
    Combined: "combined_corners.pkl" (SkillCorner + DFL)
  - Update processed_file_names to include source indicator
```

### Step 6: Modify `corner_prediction/run_all.py`

```
Add:
  --combined flag: loads combined_corners.pkl instead of extracted_corners.pkl
  --extract-dfl flag: runs DFL extraction before evaluation

  When --combined:
    - Load combined records
    - Build CornerKickDataset from combined records
    - LOMO uses 17 match IDs (10 SC + 7 DFL) instead of 10
    - Results saved with "combined_" prefix
```

### Step 7: `scripts/slurm/run_dfl_integration.sbatch`

```
SLURM array job with 3 tasks:

Task 0: DFL extraction (CPU, 32G RAM, 4h)
  - Load 7 DFL matches via kloppy
  - Extract corners, build flat records
  - Save dfl_extracted_corners.pkl
  - Merge with SkillCorner records -> combined_corners.pkl

Task 1: Combined LOMO evaluation - pretrained (GPU, 16G RAM, 6h)
  - Run LOMO CV on combined 143 corners, 17 folds
  - Save results

Task 2: Combined LOMO evaluation - scratch (GPU, 16G RAM, 6h)
  - Run LOMO CV on combined 143 corners, 17 folds, scratch backbone
  - Save results
```

---

## Data flow

```
data/dfl/*.xml  ──────────────────┐
                                  ▼
              extract_dfl_corners.py
                                  │
                                  ▼
              dfl_extracted_corners.pkl (57 corners)
                                  │
data/skillcorner/  ──────────┐    │
                             ▼    ▼
              extracted_corners.pkl    merge_datasets.py
              (86 corners)             │
                                       ▼
                          combined_corners.pkl (143 corners)
                                       │
                                       ▼
                            build_graphs.py (KNN k=6)
                                       │
                                       ▼
                        combined_graphs_knn6.pkl (143 graphs)
                                       │
                                       ▼
                          LOMO CV (17-fold, one match per fold)
                                       │
                                       ▼
                        results/corner_prediction/combined_*.pkl
```

## Coordinate normalization detail

DFL kloppy: [0, 1] normalized, origin at top-left(?), pitch 105x68m.
Target: center-origin meters, attacking toward +x.

```python
# Step 1: [0,1] -> meters (center-origin)
x_meters = x_norm * 105.0 - 52.5  # range [-52.5, 52.5]
y_meters = y_norm * 68.0 - 34.0   # range [-34, 34]

# Step 2: velocity [0,1]/frame -> m/s
# Position delta per frame in [0,1] space, convert to meters then to m/s
vx_meters = (x_norm[t] - x_norm[t-1]) * 105.0  # meters per frame
vy_meters = (y_norm[t] - y_norm[t-1]) * 68.0    # meters per frame
vx_mps = vx_meters * 25.0  # m/s (at 25fps)
vy_mps = vy_meters * 25.0  # m/s

# Step 3: Determine attacking direction
# Use corner team + kloppy coordinate system
# If corner team attacks toward -x, flip all coordinates
```

## DFL position -> role abbreviation mapping

```python
DFL_POSITION_MAP = {
    "Goalkeeper": "GK",
    "Left Center Back": "CB",
    "Right Center Back": "CB",
    "Left Back": "LB",
    "Right Back": "RB",
    "Left Defensive Midfield": "DM",
    "Right Defensive Midfield": "DM",
    "Center Attacking Midfield": "AM",
    "Left Midfield": "LM",
    "Right Midfield": "RM",
    "Left Forward": "LF",
    "Right Forward": "RF",
    "Striker": "CF",
    "Unknown": "SUB",
}
```

## Test criteria

1. DFL extraction produces exactly 57 corners across 7 matches (matching corner_dataset.pkl count)
2. Each record has exactly 22 players (no ball node)
3. All coordinates within pitch bounds after normalization
4. All velocities < 15 m/s (physically plausible)
5. Shot labels match existing corner_dataset.pkl labels: 19 shots, 2 goals
6. Combined dataset has 143 records (86 SC + 57 DFL)
7. LOMO CV runs without errors on combined dataset (17 folds)
8. Results pickle + JSON saved successfully
