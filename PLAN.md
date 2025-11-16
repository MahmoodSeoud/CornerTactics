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

**Requirements**: Extract features from each corner's freeze frame data.

**Feature categories**:

1. **Basic corner metadata** (5 features):
   - Corner side (left=0, right=1) - from location[1] < 40
   - Period (1 or 2)
   - Minute
   - Corner x, y coordinates

2. **Player count features** (6 features):
   - Total attacking players in freeze frame
   - Total defending players in freeze frame
   - Attacking players in penalty box (x > 102, 18 < y < 62)
   - Defending players in penalty box
   - Attacking players near goal (x > 108, 30 < y < 50)
   - Defending players near goal

3. **Spatial density features** (4 features):
   - Attacking player density in penalty box (count / box_area)
   - Defending player density in penalty box
   - Numerical advantage in box (attacking - defending)
   - Ratio of attackers to defenders in box

4. **Positional features** (8 features):
   - Attacking centroid x, y (average position)
   - Defending centroid x, y
   - Defending line compactness (std of y positions)
   - Defending depth (max x - min x of defenders)
   - Distance from attacking centroid to goal center
   - Distance from defending centroid to goal center

5. **Pass trajectory features** (if available from event data) (4 features):
   - Pass end_location x, y
   - Pass length
   - Pass height (Ground=0, Low=1, High=2)

**Output**: `data/processed/corners_with_features.csv`

**Columns**: match_id, event_id, outcome, [27 feature columns]

---

### **TASK 4: Train/Test Split**

Create `scripts/04_create_splits.py`:

**Requirements**:
1. Load `corners_with_features.csv`
2. Create **match-based** stratified split (80/20):
   - Group by match_id
   - Ensure train and test have similar outcome distributions
   - Use `sklearn.model_selection.StratifiedGroupKFold` or manual implementation
3. Save indices:
   - `data/processed/train_indices.csv`
   - `data/processed/test_indices.csv`

**Validation**: Print class distributions for train and test - should be similar.

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
- [ ] `scripts/04_create_splits.py`
- [ ] `scripts/05_train_baseline_models.py`
- [ ] `scripts/06_evaluate_models.py`

**Data files**:
- [x] `data/processed/corners_with_freeze_frames.json` (~1,933 samples)
- [x] `data/processed/corners_with_labels.json` (~1,933 samples)
- [x] `data/processed/corners_with_features.csv` (1,933 × 30 columns)
- [ ] `data/processed/train_indices.csv`
- [ ] `data/processed/test_indices.csv`

**Models**:
- [ ] `models/random_forest.pkl`
- [ ] `models/xgboost.pkl`
- [ ] `models/mlp.pkl`
- [ ] `models/label_encoder.pkl`
- [ ] `models/feature_scaler.pkl` (for MLP)

**Results**:
- [ ] `results/baseline_metrics.json`
- [ ] `results/confusion_matrices/` (3 PNG files)
- [ ] `results/feature_importance.png`
- [ ] `results/evaluation_report.md`

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

- **Naive baseline** (always predict majority class): ~54% accuracy
- **Random Forest**: ~60-65% accuracy, macro F1 ~0.45-0.50
- **XGBoost**: ~62-67% accuracy, macro F1 ~0.47-0.52
- **MLP**: ~58-63% accuracy, macro F1 ~0.43-0.48

If results are significantly worse, there's a bug. If significantly better, there's data leakage.


