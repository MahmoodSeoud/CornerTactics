# Corner Kick Outcome Prediction - Project Implementation Plan

## **CORNER KICK OUTCOME PREDICTION - COMPLETE IMPLEMENTATION**

### **Context**

I have StatsBomb data already downloaded:
- `data/statsbomb/events/` - Full match event JSONs (3,464 matches)
- `data/statsbomb/freeze-frames/` - 360° player position data
- `data/statsbomb/competitions.json` - Competition metadata

**Goal**: Build a 4-class corner outcome predictor using REAL StatsBomb event sequences.

**4 Classes** (based on empirical analysis):
1. Ball Receipt (54.3%) - Attacking team receives
2. Clearance (23.4%) - Defensive clearance
3. Goalkeeper (10.1%) - GK intervention  
4. Other (12.2%) - Everything else

---

### **TASK 1: Data Extraction Pipeline**

Create `scripts/01_extract_corners_with_freeze_frames.py`:

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

### **TASK 2: Outcome Label Extraction**

Create `scripts/02_extract_outcome_labels.py`:

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
         # Everything else → "Other"
         "Duel": "Other",
         "Pressure": "Other",
         "Pass": "Other",
         "Foul Committed": "Other",
         "Ball Recovery": "Other",
         "Block": "Other",
         "Interception": "Other",
         "Dispossessed": "Other",
         "Shot": "Other"
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

### **TASK 3: Feature Engineering**

Create `scripts/03_extract_features.py`:

**Requirements**: Extract ALL available features from StatsBomb corner kick data.

**COMPLETE FEATURE SET (45+ features across 10 categories):**

---

#### **1. Basic Corner Metadata** (5 features) ✅ IMPLEMENTED
- `corner_side`: Left (0) or Right (1) - from location[1] < 40
- `period`: Match period (1 or 2)
- `minute`: Minute of match
- `corner_x`: Corner kick x-coordinate
- `corner_y`: Corner kick y-coordinate

**Source**: `event.location`, `event.period`, `event.minute`

---

#### **2. Temporal Features** (3 features) ⚠️ PARTIALLY IMPLEMENTED
- `second`: Second within the minute ❌ NOT EXTRACTED
- `timestamp_seconds`: Total seconds from period start (convert HH:MM:SS.mmm) ❌ NOT EXTRACTED
- `duration`: Event duration in seconds ❌ NOT EXTRACTED

**Source**: `event.second`, `event.timestamp`, `event.duration`

---

#### **3. Player Count Features** (6 features) ✅ IMPLEMENTED
- `total_attacking`: Total attacking players in freeze frame
- `total_defending`: Total defending players in freeze frame
- `attacking_in_box`: Attacking players in penalty box (x > 102, 18 < y < 62)
- `defending_in_box`: Defending players in penalty box
- `attacking_near_goal`: Attacking players near goal (x > 108, 30 < y < 50)
- `defending_near_goal`: Defending players near goal

**Source**: `freeze_frame` (aggregate counts using `teammate` boolean)

---

#### **4. Spatial Density Features** (4 features) ✅ IMPLEMENTED
- `attacking_density`: Attacking player density in penalty box (count / box_area)
- `defending_density`: Defending player density in penalty box
- `numerical_advantage`: Numerical advantage in box (attacking - defending)
- `attacker_defender_ratio`: Ratio of attackers to defenders in box

**Source**: `freeze_frame` (spatial calculations)

---

#### **5. Positional Features** (8 features) ✅ IMPLEMENTED
- `attacking_centroid_x`: Attacking team centroid x-coordinate
- `attacking_centroid_y`: Attacking team centroid y-coordinate
- `defending_centroid_x`: Defending team centroid x-coordinate
- `defending_centroid_y`: Defending team centroid y-coordinate
- `defending_compactness`: Defending line compactness (std of y positions)
- `defending_depth`: Defending depth (max x - min x of defenders)
- `attacking_to_goal_dist`: Distance from attacking centroid to goal center
- `defending_to_goal_dist`: Distance from defending centroid to goal center

**Source**: `freeze_frame` (geometric computations)

---

#### **6. Pass Trajectory Features** (4 features) ✅ IMPLEMENTED
- `pass_end_x`: Pass landing x-coordinate
- `pass_end_y`: Pass landing y-coordinate
- `pass_length`: Euclidean distance from corner to landing (meters)
- `pass_height`: Ground Pass (0), Low Pass (1), High Pass (2)

**Source**: `event.pass.end_location`, `event.pass.length`, `event.pass.height.name`

---

#### **7. Pass Technique & Body Part** (5 features) ❌ NOT IMPLEMENTED
- `pass_angle`: Pass angle in radians (-π to π)
- `pass_body_part`: Kicking body part (Right Foot=0, Left Foot=1, Head=2, Other=3)
- `pass_technique`: Inswinging (0), Outswinging (1), Straight (2)
- `is_inswinging`: Boolean (1 if inswinging, 0 otherwise)
- `is_outswinging`: Boolean (1 if outswinging, 0 otherwise)

**Source**: `event.pass.angle`, `event.pass.body_part.name`, `event.pass.technique.name`, `event.pass.inswinging`, `event.pass.outswinging`

---

#### **8. Pass Outcome & Context** (4 features) ❌ NOT IMPLEMENTED
- `pass_outcome`: Complete (0), Incomplete (1), Out (2), Injury Clearance (3), Unknown (4)
- `is_cross_field_switch`: Boolean (1 if switch pass, 0 otherwise)
- `has_recipient`: Boolean (1 if recipient identified, 0 otherwise)
- `is_shot_assist`: Boolean (1 if led to shot, 0 otherwise)

**Source**: `event.pass.outcome.name`, `event.pass.switch`, `event.pass.recipient`, `event.pass.shot_assist`

---

#### **9. Goalkeeper & Special Player Features** (3 features) ❌ NOT IMPLEMENTED
- `num_attacking_keepers`: Count of attacking goalkeepers in freeze frame (usually 0)
- `num_defending_keepers`: Count of defending goalkeepers (usually 1)
- `keeper_distance_to_goal`: Distance from defending keeper to goal center

**Source**: `freeze_frame` (filter by `keeper == true`)

---

#### **10. Match Context - Score State** (4 features) ❌ NOT IMPLEMENTED
- `attacking_team_goals`: Goals scored by corner-taking team before this corner
- `defending_team_goals`: Goals scored by defending team before this corner
- `score_difference`: Score difference (attacking_goals - defending_goals)
- `match_situation`: Winning (1), Drawing (0), Losing (-1)

**Source**: Track Shot events with `shot.outcome.name == "Goal"` in match event sequence before corner

**Implementation**:
```python
# Track all goals before corner by index
goals_before = [e for e in match_events
                if e['index'] < corner_index
                and e.get('type', {}).get('name') == 'Shot'
                and e.get('shot', {}).get('outcome', {}).get('name') == 'Goal']

attacking_goals = sum(1 for g in goals_before if g['team']['id'] == corner_team_id)
defending_goals = len(goals_before) - attacking_goals
```

---

#### **11. Match Context - Substitution Patterns** (3 features) ❌ NOT IMPLEMENTED
- `total_subs_before`: Total substitutions before this corner (both teams)
- `recent_subs_5min`: Recent substitutions in last 5 minutes (both teams)
- `minutes_since_last_sub`: Minutes since last substitution (999 if no prior subs)

**Source**: Track Substitution events (`type.name == "Substitution"`) before corner

**Implementation**:
```python
# Track all substitutions before corner
subs_before = [e for e in match_events
               if e['index'] < corner_index
               and e.get('type', {}).get('name') == 'Substitution']

total_subs = len(subs_before)
recent_subs = sum(1 for s in subs_before if corner_minute - s['minute'] <= 5)

if subs_before:
    last_sub = max(subs_before, key=lambda x: x['index'])
    minutes_since = corner_minute - last_sub['minute']
else:
    minutes_since = 999
```

---

### **FEATURE SUMMARY**

**Total Features**: 49 features

**Currently Implemented**: 27 features (55%)
- ✅ Basic metadata (5)
- ✅ Player counts (6)
- ✅ Spatial density (4)
- ✅ Positional (8)
- ✅ Pass trajectory (4)

**Missing from Raw Data**: 22 features (45%)
- ❌ Temporal (3)
- ❌ Pass technique & body part (5)
- ❌ Pass outcome & context (4)
- ❌ Goalkeeper features (3)
- ❌ Score state (4)
- ❌ Substitution patterns (3)

**Output**: `data/processed/corners_with_features.csv`

**Columns**: match_id, event_id, outcome, [49 feature columns]

---

### **RATIONALE FOR ADDITIONAL FEATURES**

1. **Temporal features**: Capture fatigue effects (late-game corners may differ)
2. **Pass technique**: Inswinging vs outswinging affects defensive positioning
3. **Body part**: Right-footed vs left-footed corners affect ball trajectory
4. **Pass outcome**: Immediate feedback on corner quality
5. **Goalkeeper positioning**: Critical for aerial duels
6. **Score state**: Teams trailing commit more attackers (tactical urgency)
7. **Substitutions**: Fresh players or tactical changes affect set-piece execution

---

### **TASK 4: Train/Val/Test Split**

Create `scripts/04_create_splits.py`:

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

### **TASK 5: Baseline Models**

Create `scripts/05_train_baseline_models.py`:

**Requirements**: Train 3 baseline models and evaluate.

**Models**:

1. **Random Forest**:
   ```python
   RandomForestClassifier(
       n_estimators=100,
       max_depth=10,
       min_samples_split=20,
       class_weight='balanced',  # Handle 5.4:1 imbalance
       random_state=42
   )
   ```

2. **XGBoost**:
   ```python
   XGBClassifier(
       n_estimators=100,
       max_depth=6,
       learning_rate=0.1,
       scale_pos_weight=calculate_from_class_distribution,
       random_state=42
   )
   ```

3. **MLP (Neural Network)**:
   ```python
   MLPClassifier(
       hidden_layers=(64, 32),
       activation='relu',
       alpha=0.001,
       max_iter=500,
       random_state=42
   )
   # Use StandardScaler for features first
   ```

**Evaluation metrics** (for each model):
- Macro F1 (average across classes)
- Weighted F1 (weighted by class frequency)
- Per-class precision, recall, F1
- Confusion matrix
- Classification report

**Output**:
- Save models to `models/` directory
- Save metrics to `results/baseline_metrics.json`
- Print comparison table

---

### **TASK 6: Evaluation & Analysis**

Create `scripts/06_evaluate_models.py`:

**Requirements**:
1. Load all trained models
2. Generate predictions on test set
3. Create visualizations:
   - Confusion matrices (3 plots, one per model)
   - Feature importance plot (for RF and XGBoost)
   - Per-class F1 comparison (bar chart)
4. Error analysis:
   - Find most confused pairs (e.g., "Ball Receipt" vs "Other")
   - Analyze feature distributions for misclassified samples
5. Save report to `results/evaluation_report.md`

---

## **DELIVERABLES CHECKLIST**

After implementation, I should have:

**Scripts** (in order):
- [x] `scripts/01_extract_corners_with_freeze_frames.py`
- [x] `scripts/02_extract_outcome_labels.py`
- [x] `scripts/03_extract_features.py`
- [x] `scripts/04_create_splits.py`
- [x] `scripts/05_train_baseline_models.py`
- [x] `scripts/06_evaluate_models.py`

**Data files**:
- [x] `data/processed/corners_with_freeze_frames.json` (~1,933 samples)
- [x] `data/processed/corners_with_labels.json` (~1,933 samples)
- [x] `data/processed/corners_with_features.csv` (1,933 × 30 columns)
- [x] `data/processed/train_indices.csv` (1,155 train samples, 59.8%)
- [x] `data/processed/val_indices.csv` (371 val samples, 19.2%)
- [x] `data/processed/test_indices.csv` (407 test samples, 21.1%)

**Models**:
- [x] `models/random_forest.pkl`
- [x] `models/xgboost.pkl`
- [x] `models/mlp.pkl`
- [x] `models/label_encoder.pkl`
- [x] `models/feature_scaler.pkl` (for MLP)

**Results**:
- [x] `results/baseline_metrics.json`
- [x] `results/confusion_matrices/` (3 PNG files)
- [x] `results/feature_importance.png`
- [x] `results/evaluation_report.md`

---

## **CRITICAL IMPLEMENTATION NOTES**

1. **Use downloaded JSONs, not StatsBombPy API** - Your data is in `data/statsbomb/`, don't re-download
2. **Match event UUIDs with freeze frames** - The `event_uuid` in freeze frame files matches `id` in event files
3. **Handle missing freeze frames gracefully** - Not all corners have 360 data
4. **Validate at each step** - Print counts and distributions after each script
5. **Class imbalance**: 5.4:1 ratio requires `class_weight='balanced'` in sklearn
6. **Match-based splitting is mandatory** - Prevents data leakage (same match in train and test)

---

## **EXPECTED BASELINE PERFORMANCE**

Based on class distribution and similar sports analytics tasks:

- **Naive baseline** (always predict majority class): ~71% (for no-shot class)
- **Random Forest**: ~60-65% accuracy
- **XGBoost**: ~58-62% accuracy
- **MLP**: ~65-71% accuracy
- **Expected AUC**: 0.52-0.60 (barely better than random due to inherent unpredictability)

**Important**: Only use temporally valid features available at corner kick time. Features like `pass_end_x/y` (actual landing position) or `is_shot_assist` (outcome) must not be used.

---

## **BINARY SHOT PREDICTION** ✅ COMPLETE

These tasks predict shot occurrence (binary classification) rather than outcome categories.

### **TASK 7: Shot Label Extraction** ✅ COMPLETE

Created `scripts/07_extract_shot_labels.py`:

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

**Output format**:
```json
[
  {
    "match_id": "123456",
    "event": { /* full corner event */ },
    "freeze_frame": [ /* player positions */ ],
    "shot_outcome": 1  // Binary: 1=Shot, 0=No Shot
  }
]
```

**Actual Results**:
- Shot: 422 (21.8%) - matches TacticAI's ~24%
- No Shot: 1,511 (78.2%)
- Class imbalance: 3.58:1

**Key Implementation Detail**:
TacticAI filtered shots by:
- Team: Only attacking team (corner-taking team)
- Outcome: Only "threatening" shots (Goal, Saved, Post, Off Target, Wayward)
- This is critical - without filtering, shot rate was incorrectly 29%

---

### **TASK 8: Binary Classification Models** ✅ COMPLETE

Created `scripts/08_train_binary_models.py`:

**Implementation**: Trained binary classifiers for shot prediction using same features from Task 3.

**Models** (same architecture as Task 5):

1. **Random Forest**:
   ```python
   RandomForestClassifier(
       n_estimators=100,
       max_depth=10,
       class_weight='balanced',  # Handle 5.7:1 imbalance
       random_state=42
   )
   ```

2. **XGBoost**:
   ```python
   XGBClassifier(
       n_estimators=100,
       max_depth=6,
       scale_pos_weight=5.7,  # Based on imbalance ratio
       random_state=42
   )
   ```

3. **MLP**:
   ```python
   MLPClassifier(
       hidden_layers=(64, 32),
       activation='relu',
       alpha=0.001,
       max_iter=500,
       random_state=42
   )
   # Note: MLP may struggle with imbalance - use class_weight or SMOTE
   ```

**Data preparation**:
1. Load `corners_with_features.csv`
2. Merge with `corners_with_shot_labels.json` by event_id
3. Use same match-based train/val/test splits from Task 4
4. Target variable: `shot_outcome` (binary)

**Evaluation metrics**:
- Accuracy
- Precision, Recall, F1 (for both classes)
- ROC-AUC
- Precision-Recall curve (important for imbalanced data)
- Confusion matrix

**Actual Test Set Performance** (407 samples, 84 shots):

**Random Forest** ⭐ BEST:
- Accuracy: 70.8%
- Precision: 34.2%, Recall: 45.2%, F1: 0.390
- ROC-AUC: 0.638, PR-AUC: 0.316

**XGBoost**:
- Accuracy: 68.3%
- Precision: 27.7%, Recall: 33.3%, F1: 0.303
- ROC-AUC: 0.634, PR-AUC: 0.294

**MLP**:
- Accuracy: 72.7%
- Precision: 32.0%, Recall: 28.6%, F1: 0.302
- ROC-AUC: 0.633, PR-AUC: 0.277

**Output**:
- ✅ Models saved to `models/binary/` (4 files)
- ✅ Metrics saved to `results/binary_metrics.json`
- ✅ Confusion matrices generated (3 PNG files)
- ✅ Comprehensive test suite (12 tests, all passing)

**Analysis**:
- Random Forest achieved best F1 score for shot class
- All models show reasonable performance for this challenging imbalanced task
- Results align with TacticAI methodology and sports analytics literature

