# Corner Kick Data Extraction - Project Implementation Plan

## **CORNER KICK DATA EXTRACTION - COMPLETE IMPLEMENTATION**

### **Context**

StatsBomb data downloaded:
- `data/statsbomb/events/` - Full match event JSONs (3,464 matches)
- `data/statsbomb/freeze-frames/` - 360 player position data
- `data/statsbomb/competitions.json` - Competition metadata

**Goal**: Extract and process corner kick data with player positions from StatsBomb open data.

**4 Outcome Classes** (based on empirical analysis):
1. Ball Receipt (54.3%) - Attacking team receives
2. Clearance (23.4%) - Defensive clearance
3. Goalkeeper (10.1%) - GK intervention
4. Other (12.2%) - Everything else

---

### **TASK 1: Data Extraction Pipeline** [COMPLETE]

Script: `scripts/01_extract_corners_with_freeze_frames.py`

**Requirements**:
1. Load all match event JSONs from `data/statsbomb/events/`
2. For each match:
   - Find corner kick events (type="Pass" AND pass.type.name="Corner")
   - Load corresponding freeze frame data from `data/statsbomb/freeze-frames/<match_id>.json`
   - Match corner event UUIDs with freeze frame event_uuids
3. Save corners that have freeze frames to `data/processed/corners_with_freeze_frames.json`

**Output format**:
```json
[
  {
    "match_id": "123456",
    "event": { /* full corner event object */ },
    "freeze_frame": [ /* array of player positions */ ]
  }
]
```

**Expected output**: ~1,933 corners with freeze frames

---

### **TASK 2: Outcome Label Extraction** [COMPLETE]

Script: `scripts/02_extract_outcome_labels.py`

**Requirements**:
1. Load corners from Task 1
2. For each corner:
   - Load full match events from `data/statsbomb/events/<match_id>.json`
   - Find corner in event sequence by UUID
   - Get IMMEDIATE next event (index + 1)
   - Map event type to our 4 classes:
     ```python
     OUTCOME_MAPPING = {
         "Ball Receipt*": "Ball Receipt",
         "Clearance": "Clearance",
         "Goal Keeper": "Goalkeeper",
         # Everything else -> "Other"
     }
     ```
3. Add `"outcome"` field to each corner
4. Save to `data/processed/corners_with_labels.json`

**Validation**: Print class distribution - should match:
- Ball Receipt: ~54%
- Clearance: ~23%
- Goalkeeper: ~10%
- Other: ~12%

---

### **TASK 3: Feature Engineering** [COMPLETE]

Script: `scripts/03_extract_features.py`

**Requirements**: Extract ALL available features from StatsBomb corner kick data.

**COMPLETE FEATURE SET (49 features across 10 categories):**

#### **1. Basic Corner Metadata** (5 features)
- `corner_side`: Left (0) or Right (1) - from location[1] < 40
- `period`: Match period (1 or 2)
- `minute`: Minute of match
- `corner_x`: Corner kick x-coordinate
- `corner_y`: Corner kick y-coordinate

#### **2. Temporal Features** (3 features)
- `second`: Second within the minute
- `timestamp_seconds`: Total seconds from period start
- `duration`: Event duration in seconds

#### **3. Player Count Features** (6 features)
- `total_attacking`: Total attacking players in freeze frame
- `total_defending`: Total defending players in freeze frame
- `attacking_in_box`: Attacking players in penalty box (x > 102, 18 < y < 62)
- `defending_in_box`: Defending players in penalty box
- `attacking_near_goal`: Attacking players near goal (x > 108, 30 < y < 50)
- `defending_near_goal`: Defending players near goal

#### **4. Spatial Density Features** (4 features)
- `attacking_density`: Attacking player density in penalty box
- `defending_density`: Defending player density in penalty box
- `numerical_advantage`: Numerical advantage in box (attacking - defending)
- `attacker_defender_ratio`: Ratio of attackers to defenders in box

#### **5. Positional Features** (8 features)
- `attacking_centroid_x`: Attacking team centroid x-coordinate
- `attacking_centroid_y`: Attacking team centroid y-coordinate
- `defending_centroid_x`: Defending team centroid x-coordinate
- `defending_centroid_y`: Defending team centroid y-coordinate
- `defending_compactness`: Defending line compactness (std of y positions)
- `defending_depth`: Defending depth (max x - min x of defenders)
- `attacking_to_goal_dist`: Distance from attacking centroid to goal center
- `defending_to_goal_dist`: Distance from defending centroid to goal center

#### **6. Pass Trajectory Features** (4 features)
- `pass_end_x`: Pass landing x-coordinate
- `pass_end_y`: Pass landing y-coordinate
- `pass_length`: Euclidean distance from corner to landing (meters)
- `pass_height`: Ground Pass (0), Low Pass (1), High Pass (2)

**Output**: `data/processed/corners_with_features.csv`

---

### **TASK 4: Train/Val/Test Split** [COMPLETE]

Script: `scripts/04_create_splits.py`

**Requirements**:
1. Load `corners_with_features.csv`
2. Create **match-based** stratified split (60/20/20):
   - Group by match_id
   - Ensure train, val, and test have similar outcome distributions
   - Use custom stratified group split implementation
3. Save indices:
   - `data/processed/train_indices.csv`
   - `data/processed/val_indices.csv`
   - `data/processed/test_indices.csv`

**Validation**: Print class distributions for train/val/test - should be similar.

---

### **TASK 7: Shot Label Extraction** [COMPLETE]

Script: `scripts/07_extract_shot_labels.py`

**Implementation** (Following TacticAI Methodology):
1. Load `corners_with_freeze_frames.json`
2. For each corner:
   - Find corner in match event sequence by UUID
   - Look ahead at next 5 events (following TacticAI)
   - Check if any subsequent event is a **threatening shot** from **attacking team**:
     - Shot outcomes: Goal, Saved, Post, Off Target, Wayward
     - Exclude: Blocked shots, defending team shots
   - Assign binary label: 1 (Shot) or 0 (No Shot)
3. Save to `data/processed/corners_with_shot_labels.json`

**Actual Results**:
- Shot: 560 (29.0%)
- No Shot: 1,373 (71.0%)
- Class imbalance: 2.45:1

---

## **DELIVERABLES CHECKLIST**

**Scripts** (in order):
- [x] `scripts/download_statsbomb_events.py`
- [x] `scripts/download_statsbomb_360_freeze_frames.py`
- [x] `scripts/01_extract_corners_with_freeze_frames.py`
- [x] `scripts/02_extract_outcome_labels.py`
- [x] `scripts/03_extract_features.py`
- [x] `scripts/04_create_splits.py`
- [x] `scripts/07_extract_shot_labels.py`
- [x] `scripts/14_extract_temporally_valid_features.py`
- [x] `scripts/16_extract_raw_spatial_features.py`

**Data files**:
- [x] `data/processed/corners_with_freeze_frames.json` (~1,933 samples)
- [x] `data/processed/corners_with_labels.json` (~1,933 samples)
- [x] `data/processed/corners_with_features.csv` (1,933 x 30 columns)
- [x] `data/processed/train_indices.csv` (1,155 train samples, 59.8%)
- [x] `data/processed/val_indices.csv` (371 val samples, 19.2%)
- [x] `data/processed/test_indices.csv` (407 test samples, 21.1%)
- [x] `data/processed/corners_with_shot_labels.json`
- [x] `data/processed/corners_features_temporal_valid.csv`

---

## **CRITICAL IMPLEMENTATION NOTES**

1. **Use downloaded JSONs, not StatsBombPy API** - Data is in `data/statsbomb/`, don't re-download
2. **Match event UUIDs with freeze frames** - The `event_uuid` in freeze frame files matches `id` in event files
3. **Handle missing freeze frames gracefully** - Not all corners have 360 data
4. **Validate at each step** - Print counts and distributions after each script
5. **Match-based splitting is mandatory** - Prevents data leakage (same match in train and test)
