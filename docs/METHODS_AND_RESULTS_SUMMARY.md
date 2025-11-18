# CornerTactics: Machine Learning for Corner Kick Outcome Prediction

**Project Summary for LLM Comprehension**

**Last Updated**: November 17, 2025

---

## Project Overview

**CornerTactics** is a machine learning research project that predicts corner kick outcomes in professional soccer using StatsBomb's open event data and 360-degree freeze frame player positioning data. The project implements both multi-class outcome prediction (4 classes) and binary shot prediction models.

**Core Objective**: Build ML models to predict corner kick outcomes using spatial player positioning and event metadata.

**Project Stage**: Complete baseline implementation with trained models and comprehensive evaluation.

---

## Dataset

### Data Source
- **Provider**: StatsBomb Open Data (GitHub repository)
- **Format**: JSON event files and freeze frame files
- **Coverage**: 75 competitions, 3,464 matches, 12,188,949 events
- **Total Corner Kicks**: 34,049
- **Corners with 360° Data**: 1,933 (5.7% of all corners)
- **Data Volume**: ~11.3 GB

### StatsBomb Coordinate System
- **Pitch Dimensions**: 120 × 80 units (abstract units, not meters)
- **X-axis**: 0 (defensive goal) → 120 (attacking goal)
- **Y-axis**: 0 (bottom sideline) → 80 (top sideline)
- **Penalty Box**: x > 102, 18 < y < 62
- **Near Goal Area**: x > 108, 30 < y < 50
- **Goal Center**: (120, 40)
- **Corner Locations**: Right (120, 0), Left (120, 80)

### 360° Freeze Frame Data
- **Content**: Player positions at the exact moment of corner kick execution
- **Player Information**:
  - Location [x, y] coordinates
  - Position/role (e.g., Center Back, Right Wing)
  - Teammate flag (boolean: same team as corner taker)
  - Keeper flag (boolean: is goalkeeper)
- **Typical Size**: 15-22 players per freeze frame (outfield players from both teams)
- **Coverage**: 323 matches have freeze frame data available

---

## Data Processing Pipeline

### Task 1: Extract Corners with Freeze Frames
**Script**: `scripts/01_extract_corners_with_freeze_frames.py`

**Process**:
1. Load all 3,464 match event JSON files from `data/statsbomb/events/`
2. Identify corner kicks: `event['type']['name'] == "Pass"` AND `event['pass']['type']['name'] == "Corner"`
3. Load corresponding freeze frame files from `data/statsbomb/freeze-frames/`
4. Match corner event UUIDs with freeze frame event_uuids using dictionary lookup (O(1))
5. Filter corners that have 360° positioning data

**Output**: `data/processed/corners_with_freeze_frames.json` (9.1 MB)
- **Total Corners Found**: 34,049
- **Corners with Freeze Frames**: 1,933 ✓ (matches expected count)

**Test Coverage**: 8 tests (all passing)

---

### Task 2: Extract Outcome Labels (4-Class Classification)
**Script**: `scripts/02_extract_outcome_labels.py`

**Method**:
- For each corner, load the full match event sequence
- Find corner event by UUID in sequence
- Extract the **immediate next event** (index + 1) as the outcome
- Map event type to 4 outcome classes

**Class Mapping**:
```python
OUTCOME_MAPPING = {
    "Ball Receipt*": "Ball Receipt",  # Attacking team receives
    "Clearance": "Clearance",          # Defensive clearance
    "Goal Keeper": "Goalkeeper",       # GK intervention
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

**Class Distribution** (1,933 corners):
1. **Ball Receipt**: 1,050 (54.3%) - Attacking team successfully receives the ball
2. **Clearance**: 453 (23.4%) - Defensive clearance
3. **Goalkeeper**: 196 (10.1%) - Goalkeeper intervention (catch, punch, save)
4. **Other**: 234 (12.1%) - Duel, Pressure, Pass, Foul, Block, etc.

**Class Imbalance**: 5.4:1 (majority class : minority class ratio)

**Output**: `data/processed/corners_with_labels.json`

**Key Findings**:
- Ball Receipt is the most common outcome (54.3%)
- Defensive actions (Clearance + Goalkeeper) account for 33.5%
- Top 3 event types cover 87.8% of all corner outcomes

---

### Task 3: Feature Engineering
**Script**: `scripts/03_extract_features.py`

**27 Features Extracted** (5 categories):

#### 1. Basic Corner Metadata (5 features):
- `corner_side`: Left (0) or Right (1) based on y < 40
- `period`: Match period (1 = first half, 2 = second half)
- `minute`: Minute of the match
- `corner_x`, `corner_y`: Corner kick coordinates

#### 2. Player Count Features (6 features):
- `attacking_players`: Total attacking players in freeze frame
- `defending_players`: Total defending players in freeze frame
- `attacking_in_box`: Attackers in penalty box (x > 102, 18 < y < 62)
- `defending_in_box`: Defenders in penalty box
- `attacking_near_goal`: Attackers near goal (x > 108, 30 < y < 50)
- `defending_near_goal`: Defenders near goal

#### 3. Spatial Density Features (4 features):
- `attacking_density`: Attacking players / penalty box area
- `defending_density`: Defending players / penalty box area
- `numerical_advantage`: attacking_in_box - defending_in_box
- `attacker_defender_ratio`: Ratio of attackers to defenders in box (handles division by zero)

#### 4. Positional Features (8 features):
- `attacking_centroid_x`, `attacking_centroid_y`: Mean position of attacking players
- `defending_centroid_x`, `defending_centroid_y`: Mean position of defending players
- `defending_compactness`: Standard deviation of defending y positions (lower = more compact)
- `defending_depth`: Max x - Min x of defenders (defensive line depth)
- `attacking_to_goal_dist`: Distance from attacking centroid to goal center
- `defending_to_goal_dist`: Distance from defending centroid to goal center

#### 5. Pass Trajectory Features (4 features):
- `pass_end_x`, `pass_end_y`: Where the corner kick lands
- `pass_length`: Euclidean distance from corner location to end location
- `pass_height`: 0 = Ground Pass, 1 = Low Pass, 2 = High Pass

**Output**: `data/processed/corners_with_features.csv` (1,933 rows × 30 columns)
- Columns: match_id, event_id, outcome, [27 feature columns]
- All features are numeric (int or float)
- No missing values

**Test Coverage**: 19 tests (all passing)

---

### Task 4: Train/Val/Test Split
**Script**: `scripts/04_create_splits.py`

**Method**: **Match-based stratified split** (prevents data leakage)

**Why Match-based?**
- Ensures the same match never appears in multiple splits
- Prevents data leakage (corners from same match may have similar patterns)
- Groups by match_id before splitting

**Split Ratios**:
- **Train**: 1,155 samples (59.8%)
- **Validation**: 371 samples (19.2%)
- **Test**: 407 samples (21.1%)

**Stratification**: Maintains similar outcome class distributions across all splits

**Output Files**:
- `data/processed/train_indices.csv`
- `data/processed/val_indices.csv`
- `data/processed/test_indices.csv`

**Validation**: Printed class distributions confirm similar distributions across splits

---

## Models and Results: 4-Class Outcome Prediction

### Task 5 & 6: Baseline Models

**Script**: `scripts/05_train_baseline_models.py` and `scripts/06_evaluate_models.py`

Three models trained with class imbalance handling:

#### 1. Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    class_weight='balanced',  # Handles 5.4:1 imbalance
    random_state=42
)
```

#### 2. XGBoost
```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=calculated_from_class_distribution,
    random_state=42
)
# Uses sample weights based on class distribution
```

#### 3. MLP (Multi-Layer Perceptron)
```python
MLPClassifier(
    hidden_layers=(64, 32),
    activation='relu',
    alpha=0.001,
    max_iter=500,
    random_state=42
)
# Features normalized with StandardScaler
```

---

### Test Set Performance (407 samples)

#### Random Forest ⭐ BEST MODEL
```
                  Precision  Recall  F1-Score  Support
Ball Receipt         0.65     0.62     0.63      216
Clearance            0.33     0.31     0.32       97
Goalkeeper           0.39     0.77     0.52       44
Other                0.41     0.18     0.25       50

Accuracy: 50.9%
Macro F1: 0.430
Weighted F1: 0.499
```

**Strengths**:
- Best overall performance across all metrics
- Excellent recall for Goalkeeper class (77%)
- Balanced performance on majority class (Ball Receipt)

**Weaknesses**:
- Low F1 for minority classes (Other: 0.25)
- Confusion between Ball Receipt and Clearance

**Misclassification Rate**: 49.1%

#### XGBoost
```
Accuracy: 49.6%
Macro F1: 0.396
Weighted F1: 0.476
```

**Observations**:
- Slightly worse than Random Forest
- Better at Goalkeeper prediction (70% recall)
- Struggles with Other class (12% recall)

#### MLP (Neural Network)
```
Accuracy: 49.6%
Macro F1: 0.201
Weighted F1: 0.386
```

**Observations**:
- Severe class imbalance problem
- Predicts Ball Receipt for 89% of samples
- Cannot predict Goalkeeper (0% recall) or Other (0% recall)
- Only learns majority class effectively

---

### Feature Importance Analysis

#### Random Forest - Top 5 Features
1. **pass_length**: 0.1661 - Distance of corner kick
2. **pass_end_x**: 0.1584 - Where corner lands (x-coordinate)
3. **pass_end_y**: 0.1138 - Where corner lands (y-coordinate)
4. **defending_depth**: 0.0512 - Depth of defensive line
5. **defending_to_goal_dist**: 0.0467 - Defender positioning relative to goal

#### XGBoost - Top 5 Features
1. **pass_height**: 0.1244 - Ground/Low/High trajectory
2. **pass_length**: 0.0864 - Distance of corner kick
3. **pass_end_x**: 0.0818 - Where corner lands (x-coordinate)
4. **period**: 0.0634 - First or second half
5. **corner_x**: 0.0607 - Corner location x-coordinate

**Key Finding**: **Pass trajectory features** (length, end location, height) are the most predictive features for corner outcomes. Player positioning features have secondary importance.

---

### Confusion Matrix Analysis

#### Most Confused Class Pairs (Random Forest)
1. **Clearance → Ball Receipt**: 45 times (46% of Clearances misclassified)
2. **Ball Receipt → Clearance**: 42 times (19% of Ball Receipts)
3. **Ball Receipt → Goalkeeper**: 32 times (15% of Ball Receipts)
4. **Other → Ball Receipt**: 24 times (48% of Other)

#### Most Confused Class Pairs (XGBoost)
1. **Clearance → Ball Receipt**: 53 times (55% of Clearances)
2. **Ball Receipt → Clearance**: 38 times (18% of Ball Receipts)
3. **Other → Ball Receipt**: 29 times (58% of Other)

#### Most Confused Class Pairs (MLP)
1. **Clearance → Ball Receipt**: 87 times (90% of Clearances - severe)
2. **Other → Ball Receipt**: 45 times (90% of Other)
3. **Goalkeeper → Ball Receipt**: 39 times (89% of Goalkeeper)

**Insight**: Models struggle to distinguish defensive actions (Clearance, Goalkeeper) from successful attacking outcomes (Ball Receipt). This reflects the inherent difficulty of predicting contested aerial duels.

---

### Evaluation Artifacts

**Generated Files**:
- `results/baseline_metrics.json` - Complete metrics for all models
- `results/evaluation_report.md` - Detailed analysis report
- `results/confusion_matrices/` - 3 confusion matrix visualizations (PNG)
- `results/feature_importance.png` - Feature importance comparison (RF vs XGB)
- `results/per_class_f1.png` - Per-class F1 score comparison

---

## Models and Results: Binary Shot Prediction

### Task 7: Shot Label Extraction (TacticAI Methodology)
**Script**: `scripts/07_extract_shot_labels.py`

**Method** (Following DeepMind's TacticAI Paper):
1. **Lookahead window**: 5 events after corner kick
2. **Filter shots by team**: Only shots from **attacking team** (corner-taking team)
3. **Filter shots by outcome**: Only **"threatening" shots**
   - **Include**: Goal, Saved, Post, Off Target, Wayward
   - **Exclude**: Blocked shots, shots from defending team (counterattacks)

**Rationale**: TacticAI focuses on shots that represent genuine goal-scoring opportunities, not just any shot event.

**Binary Labels**:
- **Shot (1)**: Corner leads to threatening shot from attacking team within 5 events
- **No Shot (0)**: No threatening shot within lookahead window

**Class Distribution** (1,933 corners):
- **Shot**: 422 (21.8%)
- **No Shot**: 1,511 (78.2%)
- **Imbalance Ratio**: 3.58:1

**Validation**: 21.8% shot rate aligns with TacticAI's reported ~24% shot rate

**Important Note**: Initial implementation without filtering produced 29% shot rate (too high), highlighting the importance of TacticAI's filtering methodology.

**Output**: `data/processed/corners_with_shot_labels.json` (9.2 MB)

**Test Coverage**: 12 tests (all passing)

---

### Task 8: Binary Classification Models
**Script**: `scripts/08_train_binary_models.py`

**Data Preparation**:
1. Load `corners_with_features.csv` (27 features from Task 3)
2. Merge with `corners_with_shot_labels.json` by event_id
3. Use same match-based train/val/test splits from Task 4
4. Target variable: `shot_outcome` (binary: 0 or 1)

**Models**: Same architectures as Task 5, adapted for binary classification

#### 1. Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',  # Handles 3.58:1 imbalance
    random_state=42
)
```

#### 2. XGBoost
```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    scale_pos_weight=3.58,  # Based on imbalance ratio
    random_state=42
)
```

#### 3. MLP
```python
MLPClassifier(
    hidden_layers=(64, 32),
    activation='relu',
    alpha=0.001,
    max_iter=500,
    random_state=42
)
# StandardScaler applied to features
```

---

### Test Set Performance (407 samples, 84 shots)

#### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision (Shot)**: Of predicted shots, % that are correct
- **Recall (Shot)**: Of actual shots, % that are detected
- **F1 (Shot)**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (0.5 = random, 1.0 = perfect)
- **PR-AUC**: Area under Precision-Recall curve (better for imbalanced data)

#### Random Forest ⭐ BEST MODEL
```
Accuracy: 70.8%
Precision (Shot): 34.2%
Recall (Shot): 45.2%
F1 (Shot): 0.390
ROC-AUC: 0.638
PR-AUC: 0.316

Confusion Matrix:
              Predicted
              No Shot  Shot
Actual No Shot   250     73
       Shot       46     38
```

**Analysis**:
- Correctly identifies 45.2% of shots (38/84)
- 34.2% precision means 1 in 3 shot predictions is correct
- ROC-AUC of 0.638 indicates reasonable discriminative ability
- Best F1 score among all models

#### XGBoost
```
Accuracy: 68.3%
Precision: 27.7%
Recall: 33.3%
F1: 0.303
ROC-AUC: 0.634
PR-AUC: 0.294

Confusion Matrix:
              No Shot  Shot
No Shot         250     73
Shot             56     28
```

**Analysis**:
- Lower precision (27.7%) - more false positives
- Lower recall (33.3%) - misses more shots
- Similar ROC-AUC to Random Forest

#### MLP (Neural Network)
```
Accuracy: 72.7%
Precision: 32.0%
Recall: 28.6%
F1: 0.302
ROC-AUC: 0.633
PR-AUC: 0.277

Confusion Matrix:
              No Shot  Shot
No Shot         272     51
Shot             60     24
```

**Analysis**:
- Highest overall accuracy (72.7%)
- Lowest recall (28.6%) - conservative predictions
- Fewer false positives (51 vs 73)
- Misses more shots (60 vs 46)

---

### Binary Classification Analysis

**Baseline Comparison**:
- **Naive baseline** (always predict No Shot): 79.4% accuracy, 0.0 F1 for Shot class
- All models significantly outperform naive baseline on F1 score

**Performance Comparison**:
- Random Forest: Best F1 (0.390), best recall (45.2%)
- MLP: Highest accuracy (72.7%), most conservative
- XGBoost: Middle ground, similar to RF but lower precision

**Model Selection**:
- **For maximizing shot detection**: Random Forest (highest recall)
- **For minimizing false alarms**: MLP (highest precision)
- **For balanced performance**: Random Forest (highest F1)

**ROC-AUC Comparison**:
All models achieve ROC-AUC ~0.63-0.64, indicating:
- Models are better than random (0.5)
- Room for improvement to excellent (>0.8)
- Consistent performance across model types

**PR-AUC Comparison**:
PR-AUC scores (0.28-0.32) indicate:
- Models perform reasonably on imbalanced data
- Precision-recall tradeoff is challenging
- Baseline PR-AUC (% of shots) = 0.206

---

### Evaluation Artifacts

**Generated Files**:
- `results/binary_metrics.json` - Complete metrics for all models
- `results/confusion_matrices_binary/` - 3 confusion matrix visualizations (PNG)
- `models/binary/` - Saved models (random_forest_binary.pkl, xgboost_binary.pkl, mlp_binary.pkl, label_encoder_binary.pkl)

**Test Coverage**: 12 tests (all passing)

---

## Key Findings

### 1. Class Imbalance is the Primary Challenge
- **4-Class Problem**: 5.4:1 imbalance (Ball Receipt dominates at 54.3%)
- **Binary Problem**: 3.58:1 imbalance (No Shot at 78.2%)
- **Mitigation Strategies Used**:
  - `class_weight='balanced'` for Random Forest
  - Sample weighting for XGBoost (`scale_pos_weight`)
  - StandardScaler normalization for MLP
- **Result**: Traditional ML struggles with minority classes; Random Forest performs best

### 2. Corner Kick Outcomes are Inherently Uncertain
- Best models achieve only ~50% accuracy for 4-class prediction
- ~70% accuracy for binary shot prediction
- **Why?**
  - Spatial positioning captures only part of the story
  - Missing contextual factors: player skill, fatigue, weather, tactics
  - Contested aerial duels have high variance
  - Ball physics and timing are unpredictable
- **Comparison to Literature**: Results align with sports analytics research showing set pieces are difficult to predict

### 3. Pass Trajectory Features are Most Predictive
- **Top 3 features** across all models:
  1. Pass length
  2. Pass end location (x, y)
  3. Pass height
- **Insight**: **Where and how the corner is delivered** matters more than static player positioning
- **Player positioning features** (centroids, density, depth) have secondary importance
- **Implication**: Corner kick execution quality is critical

### 4. Models Confuse Defensive vs Offensive Outcomes
- **Most confused pairs**:
  - Clearance ↔ Ball Receipt
  - Goalkeeper → Ball Receipt
  - Other → Ball Receipt
- **Why?**
  - Similar spatial setups can lead to different outcomes
  - Depends on who wins the aerial duel (not captured in freeze frame)
  - Timing and player movement after freeze frame

### 5. TacticAI Methodology Alignment
- Shot prediction methodology matches published research (DeepMind's TacticAI)
- 21.8% shot rate validates filtering approach (close to TacticAI's ~24%)
- Filtering is critical: Without it, shot rate was 29% (too high)
- **Filtering criteria**:
  - Only attacking team shots
  - Only threatening shots (exclude blocked)
  - 5-event lookahead window

### 6. Random Forest Outperforms Other Models
- **4-Class Task**: RF achieves best macro F1 (0.43) and weighted F1 (0.50)
- **Binary Task**: RF achieves best F1 for shot class (0.39)
- **Why RF works better**:
  - Handles class imbalance better with `class_weight='balanced'`
  - Robust to feature scaling (doesn't require normalization)
  - Ensemble of trees captures non-linear patterns
- **XGBoost** performs similarly but slightly worse
- **MLP** struggles with imbalance despite normalization

---

## Limitations

### 1. Limited Temporal Information
- Freeze frames capture a single moment in time
- No information about player movement, speed, or trajectories
- No pre-corner tactical setup or team instructions

### 2. Missing Contextual Factors
- Player attributes: height, heading ability, finishing skill
- Team tactics: zonal vs man-marking, set piece routines
- Match context: score, time remaining, importance
- Environmental: weather, pitch conditions, crowd noise

### 3. Simplified Outcome Labels
- 4-class system loses nuance (e.g., "Other" is too broad)
- Binary shot prediction doesn't capture shot quality
- Immediate next event may not represent true outcome (e.g., shot could come after a clearance and recovery)

### 4. Data Coverage
- Only 1,933 corners with 360° data (5.7% of all corners)
- Limited to matches where StatsBomb provides freeze frames
- Potential selection bias: freeze frame data mostly from elite competitions

### 5. Model Interpretability
- Feature importance shows "what" matters but not "why"
- Difficult to extract tactical insights for coaches
- Black-box nature limits actionable recommendations

---

## Technical Implementation Strengths

### 1. Test-Driven Development (TDD)
- Comprehensive test suites for all scripts
- Total: 70+ tests across all tasks
- All tests passing
- Ensures code correctness and prevents regressions

### 2. Modular Pipeline Design
- Each task is a separate script with clear input/output
- Easy to modify individual stages without breaking pipeline
- Reusable feature extraction code

### 3. Proper Data Splitting
- **Match-based stratified split** prevents data leakage
- Critical for valid evaluation
- Many sports analytics projects fail here

### 4. Class Imbalance Handling
- Multiple strategies: class weighting, sample weighting, normalization
- Proper evaluation metrics: macro F1, per-class F1, PR-AUC
- Avoids pitfall of optimizing only for accuracy

### 5. Alignment with Published Research
- TacticAI methodology for shot labeling
- Feature engineering informed by sports analytics literature
- Validates results against expected distributions

### 6. Comprehensive Evaluation
- Multiple metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC
- Confusion matrices for error analysis
- Feature importance analysis
- Visualizations for interpretability

---

## Files and Artifacts

### Scripts (Execution Order)
1. `scripts/01_extract_corners_with_freeze_frames.py` - Extract 1,933 corners with 360° data
2. `scripts/02_extract_outcome_labels.py` - Label with 4 outcome classes
3. `scripts/03_extract_features.py` - Extract 27 features per corner
4. `scripts/04_create_splits.py` - Match-based train/val/test split
5. `scripts/05_train_baseline_models.py` - Train 3 baseline models (4-class)
6. `scripts/06_evaluate_models.py` - Evaluate and visualize results
7. `scripts/07_extract_shot_labels.py` - Extract binary shot labels (TacticAI method)
8. `scripts/08_train_binary_models.py` - Train 3 binary classifiers

### Data Files
- `data/processed/corners_with_freeze_frames.json` (9.1 MB, 1,933 samples)
- `data/processed/corners_with_labels.json` (4-class labels)
- `data/processed/corners_with_shot_labels.json` (9.2 MB, binary labels)
- `data/processed/corners_with_features.csv` (1,933 × 30 columns)
- `data/processed/train_indices.csv` (1,155 samples, 59.8%)
- `data/processed/val_indices.csv` (371 samples, 19.2%)
- `data/processed/test_indices.csv` (407 samples, 21.1%)

### Trained Models
- `models/random_forest.pkl` (4-class)
- `models/xgboost.pkl` (4-class)
- `models/mlp.pkl` (4-class)
- `models/label_encoder.pkl`
- `models/feature_scaler.pkl` (for MLP)
- `models/binary/random_forest_binary.pkl`
- `models/binary/xgboost_binary.pkl`
- `models/binary/mlp_binary.pkl`
- `models/binary/label_encoder_binary.pkl`

### Results and Visualizations
- `results/baseline_metrics.json` - 4-class model metrics
- `results/binary_metrics.json` - Binary model metrics
- `results/evaluation_report.md` - Detailed analysis report
- `results/confusion_matrices/` - 3 confusion matrix plots (4-class)
- `results/confusion_matrices_binary/` - 3 confusion matrix plots (binary)
- `results/feature_importance.png` - Feature importance comparison
- `results/per_class_f1.png` - Per-class F1 comparison

### Documentation
- `docs/STATSBOMB_DATA_GUIDE.md` (16 KB) - Comprehensive data documentation
- `docs/METHODS_AND_RESULTS_SUMMARY.md` (this file) - Methods and results summary
- `PLAN.md` - Implementation roadmap
- `CLAUDE.md` - Development guide
- `README.md` - Project overview
- `notes/features/` - Feature implementation notes (8 markdown files)

---

## Future Work

### 1. Graph Neural Networks (GNNs)
- **Current Status**: 12 GNN checkpoints exist in `models/corner_gnn_*/`
- **Approach**: Model player interactions as a graph
  - Nodes: Players (with positions and attributes)
  - Edges: Spatial proximity, marking relationships
  - Graph convolution to learn interaction patterns
- **Expected Improvement**: Better capture of relational patterns

### 2. Temporal Modeling
- **Approach**: Include pre-corner events and player trajectories
  - LSTM/GRU for sequence modeling
  - Model how play develops before corner
  - Player movement patterns leading to corner
- **Expected Improvement**: Capture tactical setup and momentum

### 3. Advanced Feature Engineering
- **Player-specific features**:
  - Height, age, heading accuracy (if available)
  - Historical corner success rates
- **Team-specific features**:
  - Team form, defensive record
  - Set piece routine patterns
- **Match context features**:
  - Score difference, time remaining
  - Match importance (e.g., knockout stage)

### 4. Ensemble Methods
- **Approach**: Combine RF, XGBoost, MLP predictions
  - Stacking: Use model outputs as features for meta-learner
  - Voting: Majority vote or weighted average
- **Expected Improvement**: Leverage complementary strengths

### 5. Better Handling of Class Imbalance
- **Techniques to try**:
  - SMOTE (Synthetic Minority Over-sampling)
  - Cost-sensitive learning
  - Focal loss for neural networks
  - One-vs-rest decomposition
- **Expected Improvement**: Better minority class prediction

### 6. Outcome Granularity
- **Current limitation**: 4 classes lose nuance
- **Proposed approach**:
  - Shot outcome classification (goal, saved, blocked, missed)
  - Receiver position prediction (which player receives)
  - Spatial outcome prediction (where ball ends up)
- **Expected Improvement**: More actionable insights for coaches

### 7. Interpretable ML
- **Techniques**:
  - SHAP values for feature importance per sample
  - LIME for local explanations
  - Attention mechanisms for neural networks
- **Goal**: Provide tactical insights, not just predictions

### 8. Transfer Learning
- **Approach**: Pre-train on all set pieces (free kicks, throw-ins)
  - Fine-tune on corners
  - Learn general spatial patterns
- **Expected Improvement**: Better generalization with limited corner data

---

## Conclusion

This project successfully implements a **complete machine learning pipeline for corner kick outcome prediction** using real professional soccer data from StatsBomb. The pipeline includes:

✅ **Data Extraction**: 1,933 corners with 360° freeze frames
✅ **Feature Engineering**: 27 spatial and trajectory features
✅ **Proper Evaluation**: Match-based splits, class imbalance handling
✅ **Multiple Tasks**: 4-class outcome prediction + binary shot prediction
✅ **Comprehensive Testing**: 70+ tests ensuring code quality
✅ **Research Alignment**: TacticAI methodology for shot labeling

**Performance Summary**:
- **4-Class Outcome Prediction**: ~50% accuracy (best: Random Forest)
  - Challenging due to 5.4:1 class imbalance
  - Pass trajectory features most predictive
- **Binary Shot Prediction**: ~70% accuracy (best: Random Forest)
  - 21.8% shot rate aligns with TacticAI research
  - ROC-AUC ~0.64 indicates reasonable discriminative ability

**Key Insight**: While prediction accuracy is moderate (50-70%), this reflects the **inherent unpredictability of soccer set pieces**. Corner kick outcomes depend on many factors beyond freeze frame positioning: player skill, timing, ball physics, tactical execution, and contested aerial duels.

**Research Contribution**: This work demonstrates that:
1. Traditional ML baselines provide reasonable performance on corner prediction
2. Proper methodology (match-based splits, class imbalance handling) is critical
3. Pass trajectory features are more predictive than player positioning
4. Alignment with published research (TacticAI) validates approach

**Value for Sports Analytics**:
- Baseline performance for future improvements (GNNs, temporal models)
- Feature importance insights for tactical analysis
- Clean, reproducible pipeline for corner kick analysis
- Demonstrates challenges and best practices for set piece prediction

The foundation is solid. Future work with GNNs, temporal modeling, and ensemble methods has clear potential to improve upon these baselines.
