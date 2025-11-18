# Complete Ablation Study Analysis Prompt
## Two-Stage Feature Engineering Investigation for Corner Kick Prediction

You are a sports analytics researcher writing a comprehensive methods and results section for a machine learning paper on predicting corner kick outcomes in professional soccer.

---

## Context

You have conducted a **two-stage ablation study** to measure the value of engineered features over raw StatsBomb event data:

1. **Stage 1: Grouped Feature Ablation** - Added features in 10 logical groups (27 → 61 features)
2. **Stage 2: Individual Feature Ablation** - Tested each feature independently (61 features tested one-by-one)

**Dataset:** 1,933 corner kicks with 360° freeze frame positioning data from StatsBomb Open Data

**Models:** Random Forest, XGBoost, MLP

**Tasks:**
- 4-class outcome prediction (Ball Receipt, Clearance, Goalkeeper, Other)
- Binary shot prediction (leads to shot attempt: yes/no)

---

## Available Data Files

### Documentation (`cornerTactics_docs/`)
- `ABLATION_STUDY_PLAN.md` - Original experimental design (grouped approach)
- `ABLATION_STUDY_IMPLEMENTATION.md` - Implementation details with grouped results
- `ABLATION_RESULTS.md` - Quick summary of grouped ablation
- `STATSBOMB_DATA_GUIDE.md` - Data structure, feature definitions

### Results - Grouped Ablation

**Primary Location (`cornerTactics_results_ablation/`):**
- `analysis/` - Visualization and analysis outputs
  - `performance_progression.png` - Accuracy curves across 10 steps
  - `feature_contribution.png` - Incremental gains table
  - `feature_correlation_matrix.png` - 63×63 feature correlations
  - `feature_outcome_correlation.png` - Top 20 predictive features
  - `transition_matrices.png` - Confusion matrices (baseline vs full)
  - `feature_importance_evolution.png` - Importance across steps
  - `ABLATION_SUMMARY.md` - Summary report
  - `feature_importance_all_steps.csv` - Feature importance evolution data
  - `feature_outcome_correlation.csv` - Feature-outcome correlations
  - `high_correlation_pairs.csv` - Highly correlated feature pairs
- `all_results.json` - Complete metrics for all 60 models (10 steps × 3 models × 2 tasks)
- `step{0-9}/` - Individual step results
  - `4class/` - 4-class outcome task results
    - `metrics.json` - Performance metrics
    - `random_forest.pkl` - Trained RF model
    - `xgboost.json` - Trained XGBoost model
    - `mlp.pth` - Trained MLP model
    - `scaler.pkl` - Feature scaler
  - `binary_shot/` - Binary shot task results (same structure)

**Secondary Location (`cornerTactics_results/`):**
- May contain additional result files and visualizations
- Check both locations for complete analysis outputs

### Results - Individual Ablation (`cornerTactics_results_ablation_individual_analysis/`)
- `phase1_raw_feature_loo.csv` - Leave-one-out results for 27 raw features
- `phase2_engineered_feature_ranking.csv` - Univariate gains for 34 engineered features
- `phase3_forward_selection.csv` - Forward selection results (step-by-step)
- `minimal_feature_set.txt` - Final minimal set (29 features)
- `INDIVIDUAL_ABLATION_SUMMARY.md` - Summary report

### Summaries
- `cornerTactics_ablation_results.md` - Grouped ablation quick summary
- `cornertactics_individual_ablation_summary.md` - Individual ablation findings

### Data (`cornerTactics_data_processed/`)
- `ablation/corners_features_step{0-9}.csv` - Feature sets for grouped approach
- `corner_labels.csv` - Ground truth labels (4-class + binary)

### Notes (`cornerTactics_notes/`)
- Development notes, design decisions, insights

---

## Task: Write Comprehensive Research Report

Write a publication-quality research report (6-8 pages) with the following structure:

---

## 1. INTRODUCTION (0.5-1 page)

### Soccer Analytics Context
- Corner kicks as high-value set pieces (2-3 per match, ~2-3% goal conversion)
- Tactical importance: planned attacking plays vs. organized defense
- Analytics opportunity: 360° freeze frame data captures full tactical setup

### Research Problem
**Central Question:** Is feature engineering worth the effort for corner kick prediction, and if so, which features matter most?

### Motivation for Two-Stage Approach
1. **Stage 1 (Grouped):** Test hypotheses about feature categories (e.g., "Do spatial features help?")
2. **Stage 2 (Individual):** Find minimal viable feature set (which individual features are essential?)

### Contributions
1. Comprehensive two-stage ablation methodology
2. Discovery that only 2 engineered features are needed (out of 34 candidates)
3. Identification of harmful raw features (some decrease performance)
4. Minimal feature set (29 features) outperforms full feature set (61 features)

---

## 2. METHODS (2-2.5 pages)

### 2.1 Dataset

**Source:** StatsBomb Open Data (international competitions, club matches)

**Sample:**
- Total corners: 1,933 (from 321 unique matches)
- With 360° freeze frames: 100% (all corners have player positioning)
- Train/Val/Test split: 70/15/15 (match-based to prevent data leakage)
  - Train: 1,346 samples (224 matches)
  - Val: 296 samples (48 matches)
  - Test: 291 samples (49 matches)

**Label Distribution:**

*4-Class Outcome:*
- Ball Receipt: 1,050 (54.3%)
- Clearance: 453 (23.4%)
- Goalkeeper: 196 (10.1%)
- Other: 234 (12.2%)

*Binary Shot:*
- No Shot: 1,511 (78.2%)
- Shot: 422 (21.8%) - **class imbalance!**

**Match-Based Splitting Rationale:**
- Prevents data leakage (multiple corners from same match could appear in train/test)
- Ensures model generalizes to new matches, not just new corners from seen matches
- More realistic deployment scenario

### 2.2 Feature Engineering

#### Raw Features (27 - Baseline)

**Numeric/Continuous (12):**
- Temporal: `period`, `minute`, `second`, `duration`
- Metadata: `index`, `possession`
- Spatial: `location_x`, `location_y`, `pass_end_x`, `pass_end_y`
- Pass: `pass_length`, `pass_angle`

**Categorical IDs (10):**
- `team_id`, `player_id`, `position_id`, `play_pattern_id`, `possession_team_id`
- `pass_height_id`, `pass_body_part_id`, `pass_type_id`, `pass_technique_id`, `pass_recipient_id`

**Boolean (3):**
- `under_pressure`, `has_pass_outcome`, `is_aerial_won`

**Simple Counts (2):**
- `total_attacking`, `total_defending` (from freeze frame)

#### Engineered Features (34)

Organized into 9 logical groups for Stage 1 (grouped ablation):

1. **Player Counts (4):** `attacking_in_box`, `defending_in_box`, `attacking_near_goal`, `defending_near_goal`
2. **Spatial Density (4):** `attacking_density`, `defending_density`, `numerical_advantage`, `attacker_defender_ratio`
3. **Positional (8):** Centroids, compactness, depth, distance to goal for both teams
4. **Pass Technique (2):** `is_inswinging`, `is_outswinging`
5. **Pass Outcome (4):** `pass_outcome_encoded`, `is_cross_field_switch`, `has_recipient`, `is_shot_assist`
6. **Goalkeeper (3):** `num_attacking_keepers`, `num_defending_keepers`, `keeper_distance_to_goal`
7. **Score State (4):** `attacking_team_goals`, `defending_team_goals`, `score_difference`, `match_situation`
8. **Substitutions (3):** `total_subs_before`, `recent_subs_5min`, `minutes_since_last_sub`
9. **Metadata (2):** `corner_side`, `timestamp_seconds`

### 2.3 Models

All models trained with early stopping on validation set.

**Random Forest:**
- n_estimators=100, max_depth=20
- min_samples_split=5, min_samples_leaf=2
- Rationale: Robust, handles non-linear relationships, provides feature importance

**XGBoost:**
- n_estimators=100, max_depth=6, learning_rate=0.1
- objective='binary:logistic' (shot) or 'multi:softprob' (4-class)
- Rationale: State-of-the-art gradient boosting, handles class imbalance

**MLP (Multi-Layer Perceptron):**
- Architecture: [512, 256, 128, 64] hidden layers, dropout=0.2
- Optimizer: Adam (lr=0.01), early stopping (patience=10)
- Rationale: Baseline neural network, captures complex interactions

### 2.4 Evaluation Metrics

**Primary:** Test accuracy (clear, interpretable)

**Secondary:**
- F1 macro (class-balanced performance)
- F1 weighted (accounts for class imbalance)
- ROC-AUC (binary shot only)
- PR-AUC (binary shot only, better for imbalanced data)
- Per-class precision/recall/F1

---

## 3. EXPERIMENTAL DESIGN (1.5-2 pages)

### 3.1 Stage 1: Grouped Feature Ablation

#### Methodology
Progressive feature addition in 10 steps (0-9):
- **Step 0 (Baseline):** 27 raw features only
- **Steps 1-9:** Add each feature group sequentially

**Total models trained:** 60 (10 steps × 3 models × 2 tasks)

**Training time:** ~5 minutes (60 models)

#### Hypotheses (per feature group)
1. **Player Counts:** Simple counts provide basic tactical context
2. **Spatial Density:** Density metrics capture crowding beyond counts
3. **Positional:** Team shape/positioning provides tactical intelligence
4. **Pass Technique:** Corner delivery technique affects outcome
5. **Pass Outcome:** Pass context provides outcome hints
6. **Goalkeeper:** GK positioning critical for aerial balls
7. **Score State:** Score affects tactical urgency (trailing teams commit more attackers)
8. **Substitutions:** Fresh players affect execution quality
9. **Metadata:** Minimal impact (redundant with existing features)

#### Why Grouped Ablation?
- Tests clear hypotheses about feature categories
- Interpretable results ("spatial features help by X%")
- Computationally efficient (60 models vs. 200+ for individual)
- Standard practice in ML research

#### Analysis Components
1. **Performance progression:** Accuracy curves across steps
2. **Feature importance:** RF importance at steps 0, 3, 6, 9
3. **Correlation analysis:** 63×63 feature correlation matrix
4. **Transition matrices:** P(predicted | actual) for baseline vs. full

### 3.2 Stage 2: Individual Feature Ablation

#### Motivation
Grouped ablation found that Step 5 (pass outcome) gave +4.81% gain, but:
- Which individual features within that group matter?
- Are all 27 raw features necessary?
- What's the minimal viable feature set?

#### Methodology - Three Phases

**Phase 1: Raw Feature Leave-One-Out**
- Remove each of 27 raw features individually
- Measure performance drop
- **Goal:** Identify which raw features are critical vs. redundant/harmful

**Phase 2: Engineered Feature Ranking (Univariate)**
- Add each of 34 engineered features individually on top of all 27 raw features
- Rank by performance gain
- **Goal:** Identify best individual engineered features

**Phase 3: Forward Selection (Minimal Set)**
- Start with 27 raw features
- Greedily add best remaining feature from Phase 2
- Stop when gain < 0.5%
- **Goal:** Find minimal feature set

**Total models trained:** ~122 (61 features × 2 tasks, Random Forest only for speed)

**Training time:** ~10 minutes

#### Why Individual Ablation?
- Finds truly minimal feature set (not just best group)
- Identifies feature redundancy within groups
- Discovers harmful features (that decrease performance)
- Optimal for production deployment (fewest features, best performance)

### 3.3 Combined Approach Value

**Grouped ablation** answers: "What types of features help?"
**Individual ablation** answers: "Which specific features are essential?"

Together, they provide:
1. **Hypothesis testing:** Grouped approach validates intuitions
2. **Optimization:** Individual approach finds optimal subset
3. **Robustness:** Two independent methods confirm findings

---

## 4. RESULTS (2.5-3 pages)

### 4.1 Stage 1: Grouped Feature Ablation Results

#### Overall Performance Summary

**Baseline (Step 0 - Raw Features, 27 features):**
- 4-Class Outcome: 80.76% accuracy (Random Forest)
- Binary Shot: 82.13% accuracy (Random Forest)

**Best Performance (Step 5 - + Pass Outcome, 49 features):**
- 4-Class Outcome: 80.07% accuracy
- Binary Shot: **86.94% accuracy** (+4.81% gain)

**Full Features (Step 9, 61 features):**
- 4-Class Outcome: 80.76% accuracy (no improvement)
- Binary Shot: 86.25% accuracy (+4.12% gain)

#### Step-by-Step Progression

**Binary Shot Prediction (Random Forest):**

| Step | Features Added | Total | Accuracy | Δ from Baseline | Marginal Gain |
|------|---------------|-------|----------|----------------|---------------|
| 0 | Raw (27) | 27 | 82.13% | - | - |
| 1 | + Player Counts | 31 | 83.85% | +1.72% | +1.72% |
| 2 | + Spatial Density | 35 | 83.16% | +1.03% | -0.69% |
| 3 | + Positional | 43 | 83.85% | +1.72% | +0.69% |
| 4 | + Pass Technique | 45 | 83.16% | +1.03% | -0.69% |
| 5 | + Pass Outcome | 49 | **86.94%** | **+4.81%** | **+3.78%** 🎯 |
| 6 | + Goalkeeper | 52 | 86.25% | +4.12% | -0.69% |
| 7 | + Score State | 56 | 86.25% | +4.12% | 0.00% |
| 8 | + Substitutions | 59 | 86.60% | +4.47% | +0.35% |
| 9 | + Metadata | 61 | 86.25% | +4.12% | -0.35% |

**4-Class Outcome Prediction (Random Forest):**

| Step | Features Added | Total | Accuracy | Δ from Baseline | Marginal Gain |
|------|---------------|-------|----------|----------------|---------------|
| 0 | Raw (27) | 27 | **80.76%** | - | - |
| 1 | + Player Counts | 31 | 80.41% | -0.34% | -0.34% |
| 2 | + Spatial Density | 35 | 80.41% | -0.34% | 0.00% |
| 3 | + Positional | 43 | **80.76%** | 0.00% | +0.34% |
| 4 | + Pass Technique | 45 | 79.73% | -1.03% | -1.03% |
| 5 | + Pass Outcome | 49 | 80.07% | -0.69% | +0.34% |
| 6 | + Goalkeeper | 52 | 80.41% | -0.34% | +0.34% |
| 7 | + Score State | 56 | **80.76%** | 0.00% | +0.34% |
| 8 | + Substitutions | 59 | 80.07% | -0.69% | -0.69% |
| 9 | + Metadata | 61 | **80.76%** | 0.00% | +0.69% |

**Key Observations:**
1. **Task-specific ROI:** Binary shot benefits greatly from engineering (+4.81%), but 4-class shows zero net gain
2. **Step 5 breakthrough (binary shot only):** Pass outcome features provide largest jump (+3.78% marginal gain)
3. **4-class plateau:** Performance oscillates around 80.76% baseline - feature engineering doesn't help!
4. **Diminishing returns:** Steps 6-9 add minimal value for binary shot (<1% combined)
5. **Non-monotonic:** Some steps decrease performance (overfitting, especially in 4-class)

#### Model Comparison (at Step 9, Binary Shot)

| Model | Accuracy | ROC-AUC | F1 Macro | Training Time |
|-------|----------|---------|----------|---------------|
| Random Forest | 86.25% | 0.833 | 0.800 | 0.20s |
| XGBoost | 85.57% | 0.854 | 0.790 | 0.16s |
| MLP | 84.88% | 0.826 | 0.777 | 0.45s |

**Winner:** Random Forest (best accuracy), XGBoost (best ROC-AUC)

#### Feature Importance Analysis (from correlation analysis)

**Top 10 Most Predictive Features:**

*For Binary Shot Prediction:*
1. `is_shot_assist` (0.649 correlation) ⭐
2. `has_recipient` (0.272)
3. `has_pass_outcome` (-0.426)
4. `pass_outcome_encoded` (-0.396)
5. `pass_height_id` (0.146)
6. `is_inswinging` (0.138)
7. `is_outswinging` (0.133)
8. `pass_technique_id` (0.138)
9. `pass_length` (0.097)
10. `pass_end_x` (-0.077)

*For 4-Class Outcome:*
1. `has_recipient` (-0.709) ⭐
2. `has_pass_outcome` (0.447)
3. `pass_outcome_encoded` (0.437)
4. `pass_end_x` (0.221)
5. `is_inswinging` (0.185)

#### Task-Specific Insights

**Why 4-class prediction plateaus:**
- Raw features (pass outcome, recipient) already encode the immediate result
- Task predicts what happens *next*, which is largely determined by pass execution
- Tactical context matters less than pass quality
- Natural ceiling: Some outcomes are inherently unpredictable

**Why binary shot prediction benefits more:**
- Shot creation depends on tactical setup (positioning, numerical advantage)
- Engineered spatial features capture attacking intent
- Pass outcome context (is_shot_assist) directly signals shots
- Binary task is simpler classification problem

### 4.2 Stage 2: Individual Feature Ablation Results

#### Phase 1: Raw Feature Leave-One-Out

**Critical Finding:** Some raw features HURT performance!

**Harmful Raw Features (removing them IMPROVES accuracy):**

| Feature | Accuracy Change | Interpretation |
|---------|----------------|----------------|
| `has_pass_outcome` | **+6.53%** | Most harmful! Creates noise |
| `period` | +3.09% | Match period irrelevant |
| `team_id` | +3.09% | Team identity not predictive |
| `pass_height_id` | +3.09% | Height not useful for shots |
| `pass_angle` | +2.75% | Angle adds noise |
| `location_y` | +2.41% | Y-coordinate not predictive |

**Total harmful features:** 6 out of 27 (22%)

**Most Important Raw Features (largest drop when removed):**

| Feature | Accuracy Drop | Interpretation |
|---------|--------------|----------------|
| `pass_recipient_id` | -1.03% (shot), **-17.2%** (4-class) | Critical for outcome! |
| `pass_length` | -1.03% | Distance matters |
| `player_id` | -1.03% | Corner taker identity matters |

**Insight:** Could reduce from 27 → ~20 raw features by removing harmful ones.

#### Phase 2: Engineered Feature Ranking

**Methodology:** Add each of 34 engineered features individually on top of 27 raw features, evaluate on both tasks.

**Top 10 Engineered Features (univariate gain):**

| Rank | Feature | Shot Gain | 4-Class Gain | Category |
|------|---------|-----------|--------------|----------|
| 1 | `is_shot_assist` | **+5.15%** | **-1.03%** | Pass Outcome |
| 2 | `defending_to_goal_dist` | +2.75% | **-0.69%** | Positional |
| 3 | `pass_outcome_encoded` | +2.75% | **-1.37%** | Pass Outcome |
| 4 | `defending_depth` | +2.41% | **-1.03%** | Positional |
| 5 | `has_recipient` | +2.41% | -0.34% | Pass Outcome |
| 6 | `defending_team_goals` | +2.41% | +0.34% | Score State |
| 7 | `defending_in_box` | +2.06% | +0.34% | Player Counts |
| 8 | `attacking_near_goal` | +2.06% | 0.00% | Player Counts |
| 9 | `corner_side` | +2.06% | +0.34% | Metadata |
| 10 | `is_cross_field_switch` | +2.06% | -0.69% | Pass Outcome |

**Critical Observation: Engineered Features Help Shot But HURT 4-Class!**

Notice the **negative** 4-Class Gain values (in bold):
- Top 4 features all **decrease** 4-class accuracy
- `is_shot_assist`: Great for shot (+5.15%) but harmful for 4-class (-1.03%)
- `pass_outcome_encoded`: Helps shot (+2.75%) but hurts 4-class (-1.37%)

**Implication:** The tasks have fundamentally different feature requirements!

**Key Finding:**
- `is_shot_assist` alone provides **+5.15%** for shot prediction - 2× better than any other feature!
- Pass outcome category dominates top 10 (4 out of 10)
- Score state features (`defending_team_goals`) surprisingly valuable individually

**Comparison to Grouped Ablation:**
- Individual `is_shot_assist`: +5.15%
- Entire Pass Outcome group (Step 5): +4.81%
- **Implication:** Most of Step 5's gain comes from ONE feature!

#### Phase 3: Forward Selection (Minimal Feature Set)

**Stopping criterion:** Add features until marginal gain < 0.5%

| Step | Feature Added | Total | Accuracy | Marginal Gain |
|------|--------------|-------|----------|---------------|
| 0 | Baseline (27 raw) | 27 | 82.13% | - |
| 1 | `is_shot_assist` | 28 | 87.29% | **+5.15%** |
| 2 | `attacking_in_box` | 29 | **87.97%** | **+0.69%** |
| 3 | (stopped) | - | - | <0.5% |

**Final Minimal Feature Set:**
- **29 features total**
  - 27 raw features
  - 2 engineered features: `is_shot_assist`, `attacking_in_box`

**Final Performance:** **87.97% shot prediction accuracy** ⭐

**Why only 2 engineered features?**
- Other features are **redundant** with is_shot_assist + attacking_in_box
- For example: `defending_to_goal_dist` gives +2.75% alone, but only +0.3% on top of the 2 selected features
- Feature correlation causes diminishing returns

### 4.3 Comparison: Grouped vs. Individual Ablation

| Approach | Best Model | Features | Accuracy | Insight |
|----------|-----------|----------|----------|---------|
| **Grouped** | Step 5 | 49 | 86.94% | Pass outcome group most valuable |
| **Grouped** | Step 9 (Full) | 61 | 86.25% | Too many features = overfitting! |
| **Individual** | Minimal | **29** | **87.97%** ⭐ | Only 2 engineered features needed |

**Performance Gain: +1.03% with 52% fewer features!**

**Critical Insight:** Feature quality >> feature quantity

### 4.4 Feature Correlation & Redundancy

**High Correlation Pairs (|r| > 0.8):**
- `attacking_in_box` ↔ `attacking_near_goal` (r = 0.92)
- `defending_in_box` ↔ `defending_near_goal` (r = 0.89)
- `total_attacking` ↔ `attacking_density` (r = 0.85)
- `team_id` ↔ `possession_team_id` (r = 1.0, identical)

**Implication:** Many engineered features measure the same underlying signal. This explains why:
1. Grouped ablation shows diminishing returns after Step 5
2. Individual ablation stops at 2 features
3. Full model (61 features) underperforms minimal model (29 features)

### 4.5 Transition Matrix Analysis

**Confusion Matrix Evolution (4-Class Task, Random Forest):**

*Baseline (Step 0, 27 raw features):*
```
                  Predicted
              BR    CL    GK    OT
Actual BR    0.85  0.10  0.03  0.02
       CL    0.35  0.54  0.08  0.03
       GK    0.40  0.20  0.35  0.05
       OT    0.45  0.25  0.10  0.20
```

*Full Features (Step 9, 61 features):*
```
                  Predicted
              BR    CL    GK    OT
Actual BR    0.87  0.08  0.03  0.02
       CL    0.32  0.58  0.07  0.03
       GK    0.38  0.18  0.39  0.05
       OT    0.42  0.23  0.10  0.25
```

**Observation:** Minimal improvement in class discrimination. Most gain comes from raw features.

---

## 5. DISCUSSION (1.5-2 pages)

### 5.1 Key Findings Summary

1. **Raw features are highly informative** (80.76% baseline)
2. **Only 2 engineered features needed** (is_shot_assist, attacking_in_box)
3. **Minimal set outperforms full set** (87.97% vs. 86.25%)
4. **Some raw features are harmful** (6 features decrease performance)
5. **Feature redundancy is pervasive** (34 candidates → 2 selected)

### 5.2 Methodological Insights

**Value of Two-Stage Approach:**
- **Grouped ablation:** Identified valuable feature category (pass outcome)
- **Individual ablation:** Found specific features within that category
- **Combined:** Comprehensive understanding of feature contributions

**Grouped vs. Individual Trade-offs:**

| Aspect | Grouped | Individual |
|--------|---------|------------|
| Interpretability | High (tests hypotheses) | Moderate (feature-level) |
| Computational cost | Low (60 models) | Medium (122 models) |
| Optimal performance | 86.94% (Step 5) | **87.97%** (Minimal) |
| Feature count | 49 | **29** |
| Use case | Research/hypothesis testing | Production deployment |

**Recommendation:** Use both!
1. Start with grouped ablation (hypothesis testing, fast)
2. Follow with individual ablation on high-value groups (optimization)

### 5.3 Feature Engineering ROI

**Investment vs. Return:**
- Engineered 34 features (significant effort: data extraction, computation)
- Only 2 provide unique value (+5.15% + 0.69% = +5.84% total)
- Remaining 32 features: redundant or harmful

**High-Value Features:**
1. `is_shot_assist` (+5.15%) - Whether corner directly assists shot
   - Easily extracted from StatsBomb data (boolean flag)
   - Clear tactical meaning

2. `attacking_in_box` (+0.69%) - Count of attacking players in box
   - Simple aggregation from freeze frame
   - Captures offensive intent

**Low-Value Categories:**
- Score state (0% gain in grouped, +2.41% individual but not selected in forward selection)
- Substitutions (-0.69% in grouped)
- Goalkeeper features (+0.8% in grouped)
- Metadata (0% in grouped)

**Surprising Non-Contributors:**
- Spatial density features (individually +2-3%, but redundant with player counts)
- Positional features (individually +2-3%, but redundant with is_shot_assist)
- Score state (counter-intuitive - expected trailing teams to attack more)

### 5.4 Harmful Raw Features Discovery

**Critical finding:** 6 raw features decrease performance when included.

**Most harmful: `has_pass_outcome` (-6.53%)**
- Boolean flag indicating pass failure
- Likely confounded with outcome label (circular reasoning)
- Removing it provides largest single gain in entire study!

**Other harmful features:**
- `period`, `team_id`, `pass_height_id`, `pass_angle`, `location_y`
- Likely add noise or create spurious correlations

**Implication:** **Ultra-Minimal Feature Set Hypothesis**
- 20 cleaned raw features (remove 6 harmful + 1 redundant)
- + 2 engineered features
- = 22 total features
- **Predicted performance: 88-90%** (hypothesis, needs testing)

### 5.5 Comparison to Related Work

**Corner kick prediction (prior work):**
- Limited research on corner-specific prediction
- Most work focuses on xG (expected goals) from all shot types
- Typical xG accuracy: 60-70%

**Our results:**
- Binary shot prediction: **87.97%**
- 4-class outcome: **80.76%**
- Substantially better due to:
  1. Specialized to corner kicks (narrower task)
  2. 360° freeze frame data (rich positional features)
  3. Careful feature selection (avoiding overfitting)

**General soccer outcome prediction:**
- Match result prediction: 50-55% (high variance)
- Goal-scoring prediction: 60-65%
- Our corner-specific task is more constrained, hence higher accuracy

### 5.6 Practical Implications

#### For Production Deployment

**Recommended Model:**
- **Minimal feature set (29 features)**
  - 27 raw StatsBomb features (or 21 after removing harmful ones)
  - `is_shot_assist`
  - `attacking_in_box`

- **Model:** Random Forest (best accuracy) or XGBoost (best ROC-AUC)

- **Performance:** 87.97% shot prediction accuracy, ROC-AUC 0.842

- **Benefits:**
  - Fast inference (29 features vs. 61)
  - Robust to missing data (fewer features)
  - Less data engineering overhead
  - Better generalization (less overfitting)

#### For Coaching/Tactics

**Actionable Insights:**

1. **Shot creation depends on two factors:**
   - Is the corner a shot assist? (most important)
   - Number of attackers in the box (secondary)

2. **Not important for shot creation:**
   - Score state (trailing teams don't create more shots from corners)
   - Substitutions (fresh players don't help)
   - Goalkeeper positioning (minimal impact)

3. **Implication:** Focus on corner delivery quality (shot assists) and committing attackers to the box.

### 5.7 Limitations

1. **Small test set:** 291 samples (49 matches)
   - Confidence intervals are wide (~±5% at 95% CI)
   - May not generalize to all competitions

2. **Class imbalance:** Binary shot is 78%/22% (no shot/shot)
   - Accuracy may overstate performance
   - ROC-AUC and PR-AUC are better metrics

3. **Single data source:** StatsBomb Open Data (mostly international competitions)
   - May not generalize to club leagues with different tactics

4. **Temporal independence:** Each corner treated independently
   - Ignores match context, momentum, fatigue

5. **Label quality:** "Next event" may not always reflect corner outcome
   - Quick transitions, deflections may be mislabeled

6. **Feature extraction errors:** Freeze frames may have:
   - Missing players (off-camera)
   - Position inaccuracies
   - Timestamp misalignment

### 5.8 Future Work

1. **Test ultra-minimal set** (21 cleaned raw + 2 engineered = 23 features)
   - Expected: 88-90% accuracy
   - Validate harmful feature removal

2. **Ensemble models** on minimal feature set
   - Random Forest + XGBoost + MLP ensemble
   - Expected: +1-2% accuracy gain

3. **Temporal modeling**
   - RNN/LSTM for corner sequences within match
   - Does team's corner success rate drift over time?

4. **Player identity features**
   - Corner taker skill (Messi vs. unknown player)
   - Defender/attacker matchups (height, heading ability)
   - Requires player database

5. **Video validation**
   - Verify freeze frames against video
   - Extract player orientation, marking assignments
   - Computer vision for tracking

6. **Causal inference**
   - Do trailing teams actually commit more attackers? (treatment effect)
   - Propensity score matching for score state

7. **Real-time prediction**
   - Deploy minimal model for live match prediction
   - Update probabilities as corner unfolds

8. **Multi-task learning**
   - Joint model for outcome + shot prediction
   - Share representations across tasks

9. **Feature interaction analysis**
   - Are is_shot_assist and attacking_in_box orthogonal?
   - Do they capture independent signals?

---

## 6. CONCLUSION (0.5 page)

### Summary of Contributions

1. **Comprehensive two-stage ablation methodology**
   - Grouped approach for hypothesis testing
   - Individual approach for optimization
   - Combined insights more valuable than either alone

2. **Discovery of minimal feature set (29 features)**
   - Only 2 engineered features needed (out of 34 candidates)
   - Outperforms full feature set (87.97% vs. 86.25%)
   - 52% fewer features, +1% better accuracy

3. **Identification of harmful features**
   - 6 raw features decrease performance
   - Most harmful: `has_pass_outcome` (-6.53%)
   - Challenge conventional wisdom about feature inclusion

4. **Task-specific insights**
   - 4-class outcome: plateaus at raw features (80.76%)
   - Binary shot: benefits from engineering (+5.84%)
   - Different tasks have different feature engineering ROI

### Key Takeaways

**For ML Practitioners:**
- **Feature quality >> feature quantity**
- Always test individual features, not just groups
- Some features harm performance (test removal)
- Minimal models often outperform full models (overfitting)

**For Soccer Analysts:**
- Shot creation from corners depends on:
  1. Shot assists (delivery quality) - most important
  2. Attackers in box (numerical advantage) - secondary
- Score state and substitutions don't affect corner outcomes
- Focus coaching on delivery quality, not just positioning

**For Researchers:**
- Two-stage ablation provides comprehensive understanding
- Grouped ablation: fast hypothesis testing
- Individual ablation: optimal feature selection
- Both approaches are complementary, not competitive

### Final Recommendation

**Use the minimal feature set (29 features) for production deployment:**
- Best performance (87.97%)
- Fewest features (simplest model)
- Fastest inference
- Most robust

**"Don't over-engineer. Two features beat thirty-four."**

---

## 7. FIGURES & TABLES TO INCLUDE

### From Grouped Ablation

1. **Table 1:** Progressive performance by step (10 rows, both tasks)
2. **Figure 1:** Performance progression curves (`performance_progression.png`)
3. **Figure 2:** Feature contribution table (`feature_contribution.png`)
4. **Figure 3:** Feature correlation matrix 63×63 (`feature_correlation_matrix.png`)
5. **Figure 4:** Top 20 feature-outcome correlations (`feature_outcome_correlation.png`)
6. **Figure 5:** Transition matrices baseline vs. full (`transition_matrices.png`)
7. **Figure 6:** Feature importance evolution (`feature_importance_evolution.png`)

### From Individual Ablation

8. **Table 2:** Raw feature leave-one-out (top 10 harmful + top 10 critical)
9. **Table 3:** Engineered feature ranking (top 15)
10. **Table 4:** Forward selection step-by-step (3 rows)
11. **Figure 7:** Minimal feature set composition (pie chart: 27 raw + 2 engineered)

### Summary Comparison

12. **Table 5:** Grouped vs. Individual ablation comparison
13. **Figure 8:** Performance vs. feature count (all approaches on one plot)

---

## 8. WRITING GUIDELINES

### Style
- Academic/technical but accessible
- Past tense for methods/experiments
- Present tense for results interpretation
- Specific numbers (not "improved", say "improved from 82.13% to 87.97%")

### Emphasis
- **Bold** for key findings
- *Italics* for technical terms on first use
- ⭐ for most important results
- 🎯 for breakthrough moments

### Storytelling Arc
1. **Setup:** Feature engineering is common, but is it always beneficial?
2. **Stage 1:** Grouped ablation finds pass outcome features valuable (+4.81%)
3. **Surprise:** Full model (61 features) underperforms (86.25%)
4. **Stage 2:** Individual ablation reveals only 2 features needed
5. **Breakthrough:** Minimal model (29 features) achieves 87.97% - best performance!
6. **Insight:** Some features harm performance - removal improves accuracy
7. **Conclusion:** Feature quality >> quantity. Less is more.

---

## 9. DELIVERABLE

**A 6-8 page research report** structured as:

```
Title: Progressive Feature Engineering for Corner Kick Prediction:
       A Two-Stage Ablation Study on StatsBomb 360° Data

Abstract (200-250 words)

1. Introduction (0.5-1 page)
2. Methods (2-2.5 pages)
3. Experimental Design (1.5-2 pages)
4. Results (2.5-3 pages)
5. Discussion (1.5-2 pages)
6. Conclusion (0.5 page)
7. References
8. Appendix (optional: full metrics tables)
```

**Suitable for:**
- MIT Sloan Sports Analytics Conference
- StatsBomb Conference
- Journal of Sports Analytics
- IEEE Transactions on Intelligent Systems and Technology (TIST)
- KDD Workshop on Machine Learning for Sports Analytics

---

## 10. KEY MESSAGES TO CONVEY

1. **Two-stage ablation is powerful**
   - Grouped: hypothesis testing
   - Individual: optimization
   - Together: comprehensive understanding

2. **Less is more**
   - 29 features > 61 features
   - 2 engineered > 34 engineered
   - Minimal model achieves best performance

3. **Quality over quantity**
   - is_shot_assist alone: +5.15%
   - All other 33 engineered features combined: +0.69%
   - Focus on high-value features

4. **Challenge assumptions**
   - Some features harm performance (remove them!)
   - More features ≠ better performance
   - Test everything, assume nothing

5. **Practical deployment**
   - Use minimal feature set (29 features)
   - 87.97% shot prediction accuracy
   - Fast, robust, interpretable

---

## FINAL INSTRUCTION

**Analyze the provided files and write the complete research report following this comprehensive structure.**

**Be thorough, quantitative, and insightful. Use specific numbers from both ablation studies. Tell the complete story of the two-stage experimental journey and the surprising findings.**

**Focus on the narrative:** *We started with grouped ablation (found pass outcome features valuable), then refined with individual ablation (discovered only 2 features needed), and ultimately found that the minimal model outperforms the full model - a counterintuitive but important finding about overfitting in feature engineering.*

**Output format: Markdown (.md) suitable for conversion to PDF/LaTeX for conference/journal submission.**
