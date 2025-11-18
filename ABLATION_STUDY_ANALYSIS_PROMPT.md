# Prompt for Analyzing CornerTactics Ablation Study

You are a sports analytics researcher writing a methods and results section for a machine learning paper on predicting corner kick outcomes in professional soccer.

## Context

You have conducted a comprehensive ablation study to measure the incremental value of engineered features over raw StatsBomb event data. The study used 1,933 corner kicks with 360° freeze frame positioning data, progressively adding features in 10 steps (27 → 61 features) while training 60 models across two prediction tasks.

## Available Data

### Documentation (`cornerTactics_docs/`)
- `ABLATION_STUDY_PLAN.md` - Experimental design, hypotheses, feature groups
- `ABLATION_STUDY_IMPLEMENTATION.md` - Implementation details, actual results
- `ABLATION_RESULTS.md` - Quick summary with performance tables
- `STATSBOMB_DATA_GUIDE.md` - Data structure, feature definitions

### Results (`cornerTactics_results/`)
- `ablation/analysis/` - Plots, correlation matrices, feature importance
- `ablation/all_results.json` - Complete metrics for all 60 models
- `ablation/step{0-9}/{4class,binary_shot}/metrics.json` - Per-step results

### Data (`cornerTactics_data_processed/`)
- `ablation/corners_features_step{0-9}.csv` - Feature sets for each step
- `corner_labels.csv` - Ground truth labels

### Notes (`cornerTactics_notes/`)
- Development notes, decisions, insights

## Task

Write a comprehensive research report with the following sections:

### 1. METHODS

#### Dataset Description
- Data source (StatsBomb Open Data)
- Sample size (1,933 corners with 360° freeze frames)
- Label distribution:
  - 4-class: Ball Receipt 54.3%, Clearance 23.4%, Goalkeeper 10.1%, Other 12.2%
  - Binary shot: 21.8% positive class
- Train/val/test split methodology (match-based 70/15/15)

#### Feature Engineering Strategy
- Describe the 10-step progressive addition approach
- For each step, explain what features were added and the hypothesis
- Explain why this ablation design is valuable

**Step 0 - Baseline (27 raw features):**
- 12 numeric/continuous: period, minute, second, duration, index, possession, location_x, location_y, pass_length, pass_angle, pass_end_x, pass_end_y
- 10 categorical IDs: team_id, player_id, position_id, play_pattern_id, possession_team_id, pass_height_id, pass_body_part_id, pass_type_id, pass_technique_id, pass_recipient_id
- 3 boolean: under_pressure, has_pass_outcome, is_aerial_won
- 2 simple counts: total_attacking, total_defending

**Step 1 - + Player Counts (6 features → 33 total):**
- attacking_in_box, defending_in_box, attacking_near_goal, defending_near_goal
- Hypothesis: Simple player counts provide basic tactical context

**Step 2 - + Spatial Density (4 features → 37 total):**
- attacking_density, defending_density, numerical_advantage, attacker_defender_ratio
- Hypothesis: Density metrics capture crowding effects beyond simple counts

**Step 3 - + Positional Features (8 features → 45 total):**
- attacking_centroid_x/y, defending_centroid_x/y, defending_compactness, defending_depth, attacking_to_goal_dist, defending_to_goal_dist
- Hypothesis: Team shape/positioning provides tactical intelligence

**Step 4 - + Pass Technique (2 features → 47 total):**
- is_inswinging, is_outswinging
- Hypothesis: Corner delivery technique affects outcome

**Step 5 - + Pass Outcome Context (4 features → 51 total):**
- pass_outcome_encoded, is_cross_field_switch, has_recipient, is_shot_assist
- Hypothesis: Pass context provides outcome hints

**Step 6 - + Goalkeeper Features (3 features → 54 total):**
- num_attacking_keepers, num_defending_keepers, keeper_distance_to_goal
- Hypothesis: Goalkeeper positioning critical for aerial balls

**Step 7 - + Score State (4 features → 58 total):**
- attacking_team_goals, defending_team_goals, score_difference, match_situation
- Hypothesis: Score affects tactical urgency (trailing teams commit more attackers)

**Step 8 - + Substitution Patterns (3 features → 61 total):**
- total_subs_before, recent_subs_5min, minutes_since_last_sub
- Hypothesis: Fresh players or tactical changes affect execution

**Step 9 - + Metadata (2 features → 63 total - FULL):**
- corner_side, timestamp_seconds
- Hypothesis: Minimal impact (redundant with existing features)

#### Models
- **Random Forest:**
  - n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2
  - Rationale: Robust to feature scaling, handles non-linear relationships, provides feature importance

- **XGBoost:**
  - n_estimators=100, max_depth=6, learning_rate=0.1
  - objective='binary:logistic' or 'multi:softprob'
  - Rationale: State-of-the-art gradient boosting, handles class imbalance well

- **MLP (Multi-Layer Perceptron):**
  - Architecture: [512, 256, 128, 64] hidden layers
  - Dropout=0.2, optimizer=Adam(lr=0.01), loss=CrossEntropy
  - Training: 50 epochs max, early stopping (patience=10)
  - Rationale: Captures complex feature interactions, baseline neural network approach

#### Evaluation Metrics
- **Primary:** Test accuracy
- **Secondary:** F1 macro/weighted, ROC-AUC (binary), PR-AUC (binary), precision/recall per class
- **Why match-based splitting:** Prevents data leakage (multiple corners from same match could appear in train/test)

### 2. EXPERIMENTAL DESIGN

#### Ablation Methodology
- Progressive feature addition: Start with raw data baseline, incrementally add feature groups
- Compare to standard approach: Training once with all features doesn't reveal which features matter
- Each step builds upon the previous, allowing measurement of marginal contribution

#### Why Start with Raw Features?
- Establishes true baseline performance
- Measures whether feature engineering effort is justified
- Reveals if raw StatsBomb data already captures predictive signal
- Avoids confounding effects of multiple feature groups added simultaneously

#### Research Question
**"What is the incremental value of each engineered feature group over raw StatsBomb data for predicting corner kick outcomes?"**

This design directly answers:
1. Which feature groups contribute most to prediction accuracy?
2. At what point do additional features show diminishing returns?
3. What is the minimal viable feature set for production deployment?
4. Do different prediction tasks (outcome vs. shot) benefit differently from feature engineering?

#### Analysis Components
1. **Performance Tracking:** Accuracy/F1 at each step
2. **Feature Importance:** Top features from Random Forest at steps 0, 3, 6, 9
3. **Correlation Analysis:**
   - Feature-feature correlation (63×63 matrix) - identifies redundancy
   - Feature-outcome correlation - identifies most predictive features
4. **Transition Matrices:** P(predicted | actual) for baseline vs. full features

### 3. RESULTS

#### Overall Performance Summary

**Baseline (Step 0 - Raw Features, 27 features):**
- 4-Class Outcome: 80.76% accuracy (Random Forest)
- Binary Shot Prediction: 82.13% accuracy (Random Forest)

**Best Performance (Step 5 - Pass Outcome Features, 49 features):**
- 4-Class Outcome: 80.07% accuracy
- Binary Shot Prediction: **86.94% accuracy** ⭐ (+4.81% gain from baseline)

**Full Features (Step 9, 61 features):**
- 4-Class Outcome: 80.76% accuracy (no improvement over baseline)
- Binary Shot Prediction: 86.25% accuracy (+4.12% gain from baseline)

**Model Comparison at Step 9 (Full Features):**
- Random Forest: 80.76% (4-class), 86.25% (binary shot)
- XGBoost: 83.16% (4-class), 85.57% (binary shot)
- MLP: 79.38% (4-class), 84.88% (binary shot)
- **Winner:** XGBoost for 4-class, Random Forest for binary shot

#### Progressive Feature Contribution

**Step-by-Step Results (Random Forest):**

| Step | Features Added | Total | 4-Class Acc | Binary Shot Acc | Shot Δ |
|------|---------------|-------|-------------|-----------------|---------|
| 0 | Raw (27) | 27 | 80.76% | 82.13% | - |
| 1 | + Player Counts | 31 | 80.41% | 83.85% | +1.72% |
| 2 | + Spatial Density | 35 | 80.41% | 83.16% | +1.03% |
| 3 | + Positional | 43 | 80.76% | 83.85% | +1.72% |
| 4 | + Pass Technique | 45 | 79.73% | 83.16% | +1.03% |
| 5 | + Pass Outcome | 49 | 80.07% | **86.94%** | **+4.81%** 🎯 |
| 6 | + Goalkeeper | 52 | 80.41% | 86.25% | +4.12% |
| 7 | + Score State | 56 | 80.76% | 86.25% | +4.12% |
| 8 | + Substitutions | 59 | 80.07% | 86.60% | +4.47% |
| 9 | + Metadata | 61 | 80.76% | 86.25% | +4.12% |

**Key Narrative Points:**
1. **Baseline is strong:** 80.76% for 4-class shows raw data is already informative
2. **Step 5 breakthrough:** Pass outcome features provide largest single jump (+3.09% from Step 4)
3. **4-class plateaus:** Remains around 80% throughout - feature engineering doesn't help
4. **Binary shot benefits:** Steady improvement from 82% → 87% with engineering
5. **Diminishing returns:** Steps 6-9 show minimal additional gains
6. **Noise in progression:** Step 4 actually decreases 4-class performance (79.73%)

#### Feature Analysis

**Most Predictive Features (by correlation with outcome):**

**For 4-Class Outcome:**
1. `has_recipient` (-0.709) - Whether pass reached a teammate
2. `has_pass_outcome` (0.447) - Pass was unsuccessful
3. `pass_outcome_encoded` (0.437) - Type of pass failure
4. `pass_end_x` (0.221) - Ball landing x-coordinate
5. `is_inswinging` (0.185) - Inswinging corner technique

**For Binary Shot:**
1. `is_shot_assist` (0.649) - Corner directly assisted a shot
2. `has_recipient` (0.272) - Pass reached a teammate
3. `has_pass_outcome` (-0.426) - Pass was unsuccessful (negative correlation)
4. `pass_outcome_encoded` (-0.396) - Type of pass failure
5. `pass_height_id` (0.146) - Height of the pass

**Feature Importance Evolution (Random Forest, top 10 at Step 9):**
1. `attacking_in_box` (0.125)
2. `defending_in_box` (0.118)
3. `pass_end_x` (0.095)
4. `keeper_distance_to_goal` (0.082)
5. `attacking_density` (0.071)
6. `defending_centroid_x` (0.065)
7. `score_difference` (0.058)
8. `numerical_advantage` (0.054)
9. `is_inswinging` (0.047)
10. `pass_length` (0.043)

**High Correlation Pairs (|r| > 0.8, potential redundancy):**
- `attacking_in_box` ↔ `attacking_near_goal` (0.92)
- `defending_in_box` ↔ `defending_near_goal` (0.89)
- `total_attacking` ↔ `attacking_density` (0.85)
- `team_id` ↔ `possession_team_id` (1.0 - identical)

#### Task-Specific Insights

**Why does 4-class prediction plateau at baseline?**
- Raw StatsBomb features (especially pass outcome, recipient) already encode the immediate result
- The 4-class task is about what happens *next*, which is largely determined by the pass itself
- Tactical context (positioning, score) matters less than pass execution
- Natural ceiling: Some outcomes are inherently unpredictable from pre-corner state

**Why does binary shot prediction benefit more from engineering?**
- Shot creation depends on *tactical setup* (player positioning, numerical advantage)
- Engineered spatial features (density, centroids) capture attacking intent
- Pass outcome context (is_shot_assist, has_recipient) directly signals shot attempts
- Binary task is simpler: "dangerous corner" vs. "not dangerous"

**What does this tell us about the predictive signal?**
- **Pass execution >> Tactical setup** for immediate outcomes
- **Spatial configuration matters** for shot creation
- **Goalkeeper positioning** has minimal impact (counter-intuitive!)
- **Score state & substitutions** add no value (surprising!)

### 4. DISCUSSION

#### Key Finding
**Raw StatsBomb data is already highly informative for corner kick prediction.** The baseline (27 raw features) achieves 80.76% accuracy for 4-class outcome and 82.13% for binary shot prediction. This suggests that StatsBomb's event data capture the essential predictive signal without manual feature engineering.

#### Best Feature Group: Pass Outcome Context (Step 5)
The largest performance gain (+4.81% for shot prediction) comes from Step 5 features:
- `is_shot_assist` - Whether the corner directly assisted a shot
- `has_recipient` - Whether a teammate received the pass
- `pass_outcome_encoded` - Type of pass completion/failure
- `is_cross_field_switch` - Cross-field delivery

These features encode the *immediate result* of the corner, which is the strongest signal for predicting shot creation.

#### Surprising Results: Tactical Features Add Minimal Value
Counter to common soccer analytics intuition:
- **Score state** (winning/losing): +0% gain (Step 7)
- **Substitutions** (fresh players): -0.69% change (Step 8)
- **Goalkeeper distance**: Minimal importance (0.082)

**Possible explanations:**
1. Sample size (1,933 corners) may be too small to capture subtle tactical effects
2. Score state may be *correlated* with raw features (trailing teams have more corners)
3. StatsBomb data may already capture tactical intent through play patterns
4. Corner kick outcomes are more *stochastic* than tactically determined

#### Practical Implications

**For Deployment:**
Recommend **Step 5 features (49 total)** as the optimal production model:
- Best shot prediction performance (86.94%)
- Good balance of accuracy vs. complexity
- Excludes low-value features that add computational cost
- Only 22 engineered features beyond raw data

**For Research:**
- 4-class outcome prediction: Use raw features (no engineering benefit)
- Binary shot prediction: Use Step 5 features (maximum ROI)
- Focus future work on *temporal modeling* (sequence of corners) rather than single-event features

#### Limitations

1. **Small test set:** 291 samples (49 matches) - confidence intervals are wide
2. **Class imbalance:** Shot outcome is only 21.8% positive - could bias metrics
3. **Single data source:** StatsBomb Open Data (mostly international competitions)
4. **Temporal independence:** Each corner treated independently (no match context)
5. **No video validation:** Can't verify if 360° freeze frames capture all players
6. **Label quality:** "Next event" may not always reflect corner outcome (e.g., quick transition)

#### Future Work

1. **Temporal modeling:**
   - RNN/LSTM to model corner sequences within a match
   - Does a team's corner success rate improve/decline over time?

2. **Player identity features:**
   - Corner taker skill (Messi vs. unknown player)
   - Defender/attacker matchups (height, heading ability)
   - Requires player statistics database

3. **Video validation:**
   - Verify 360° freeze frames against video
   - Extract player orientation, marking assignments
   - Computer vision for player tracking

4. **Causal inference:**
   - Do trailing teams *actually* commit more attackers? (treatment effect)
   - Propensity score matching for score state analysis

5. **Real-time prediction:**
   - Deploy Step 5 model for live match prediction
   - Update probabilities as corner taker approaches ball

6. **Multi-task learning:**
   - Joint model for outcome + shot prediction
   - Share representations across tasks

### 5. FIGURES & TABLES TO INCLUDE

#### Table 1: Progressive Performance by Step
(Already shown above in Results section)

#### Figure 1: Performance Progression Curves
**File:** `results/ablation/analysis/performance_progression.png`
- Two subplots: 4-class (left), Binary shot (right)
- Three lines per subplot: Random Forest, XGBoost, MLP
- X-axis: Feature step (0-9), Y-axis: Test accuracy
- Shows: RF dominates binary shot, XGBoost dominates 4-class

#### Table 2: Top 10 Most Predictive Features
(Already shown above in Feature Analysis)

#### Figure 2: Feature Importance Evolution
**File:** `results/ablation/analysis/feature_importance_evolution.png`
- Horizontal bar chart, top 15 features from Step 9
- Four bars per feature: Step 0, 3, 6, 9
- Shows: How importance changes as features are added
- Key insight: Raw features remain important throughout

#### Figure 3: Transition Matrices (Confusion Matrices)
**File:** `results/ablation/analysis/transition_matrices.png`
- Two heatmaps: Baseline (Step 0) vs. Full (Step 9)
- 4×4 matrix: Ball Receipt, Clearance, Goalkeeper, Other
- Shows: Classification improves slightly but remains similar

#### Table 3: Model Comparison at Key Steps

| Model | Step 0 (Raw) | Step 5 (Best) | Step 9 (Full) | Best Task |
|-------|-------------|---------------|---------------|-----------|
| **Random Forest** | 82.13% | **86.94%** | 86.25% | Binary Shot |
| **XGBoost** | 83.16% | 86.94% | 85.57% | 4-Class |
| **MLP** | 81.10% | 84.88% | 84.88% | - |

*(Binary shot accuracies shown; 4-class in parentheses if different)*

#### Figure 4: Feature Correlation Matrix
**File:** `results/ablation/analysis/feature_correlation_matrix.png`
- 63×63 heatmap (all features at Step 9)
- Red = positive correlation, Blue = negative
- Highlights: Highly correlated pairs (>0.8)

#### Figure 5: Feature-Outcome Correlation
**File:** `results/ablation/analysis/feature_outcome_correlation.png`
- Two bar charts: 4-class (left), Binary shot (right)
- Top 20 features by absolute correlation
- Shows: Different features matter for different tasks

### 6. WRITING STYLE GUIDELINES

- **Tense:**
  - Methods/Experiments: Past tense ("We trained 60 models...")
  - Results interpretation: Present tense ("This suggests that...")

- **Specificity:**
  - ✅ "Accuracy improved from 82.13% to 86.94% (+4.81 percentage points)"
  - ❌ "Accuracy improved significantly"

- **Statistical vs. Practical Significance:**
  - Acknowledge: 4.81% gain may not be statistically significant with n=291 test samples
  - But: Could be practically meaningful (1 extra corner in 20 predicted correctly)

- **Comparison to Related Work:**
  - Typical soccer prediction accuracy: 50-60% (match outcomes)
  - Expected goals (xG) models: 60-70% (shot outcomes)
  - Our 80-87% is strong for a specific event type (corners)

### 7. KEY QUESTIONS TO ANSWER

1. **Is feature engineering worth the effort for corner kick prediction?**
   - For 4-class: No (0% gain)
   - For binary shot: Yes, but only up to Step 5 (+4.81%)
   - Conclusion: Modest gains, focus on high-value features only

2. **Which types of features matter most?**
   - Pass outcome context (is_shot_assist, has_recipient)
   - Player positioning (in_box counts, density)
   - Pass execution (end_x, length, angle)

3. **Do tactical context features (score, subs) improve prediction?**
   - No, surprisingly minimal impact
   - Hypothesis: Already captured in raw data or too noisy

4. **What's the minimal viable feature set for production deployment?**
   - Step 5: 49 features (27 raw + 22 engineered)
   - 86.94% binary shot accuracy
   - Excludes: Goalkeeper, score state, substitutions, metadata

5. **Do different prediction tasks benefit differently from engineering?**
   - Yes, dramatically:
     - 4-class: 0% gain (plateaus at baseline)
     - Binary shot: +4.81% gain (steady improvement)
   - Implication: Task difficulty/complexity affects engineering ROI

### 8. DELIVERABLE FORMAT

**A 4-6 page research report structured as:**

```
Title: Progressive Feature Engineering for Corner Kick Outcome Prediction:
       An Ablation Study on StatsBomb 360° Data

Abstract (150-200 words)

1. Introduction (0.5 pages)
   - Soccer analytics context
   - Corner kicks as high-value events
   - Research question and contributions

2. Methods (1.5 pages)
   - Dataset description
   - Feature engineering strategy (10 steps)
   - Models and training
   - Evaluation metrics

3. Experimental Design (0.5 pages)
   - Ablation methodology
   - Why progressive addition?
   - Analysis components

4. Results (2 pages)
   - Overall performance summary
   - Progressive contribution (Table 1, Figure 1)
   - Feature analysis (Table 2, Figures 2-5)
   - Task-specific insights

5. Discussion (1 page)
   - Key findings interpretation
   - Surprising results
   - Practical implications
   - Limitations and future work

6. Conclusion (0.5 pages)
   - Summary of contributions
   - Recommendations

References
Appendix (optional: full metrics tables)
```

**Suitable for submission to:**
- MIT Sloan Sports Analytics Conference
- StatsBomb Conference
- Journal of Sports Analytics
- IEEE Transactions on Intelligent Systems and Technology (TIST)
- KDD Workshop on Machine Learning for Sports Analytics

### 9. STORYTELLING APPROACH

**Narrative Arc:**
1. **Setup:** Feature engineering is common in ML, but is it always necessary?
2. **Experiment:** We test this rigorously with progressive ablation on corner kicks
3. **Surprise:** Raw data is already excellent (80%+ accuracy)
4. **Discovery:** Only pass outcome features provide meaningful gains
5. **Insight:** Task complexity determines engineering ROI
6. **Recommendation:** Use Step 5 for production, focus future work on temporal modeling

**Key Message:**
*"Don't over-engineer. StatsBomb's raw event data is already highly predictive for corner kicks. Feature engineering provides modest gains for shot prediction (+5%) but zero gain for outcome classification. Focus your effort on the few features that matter: pass outcome context."*

---

## FINAL INSTRUCTION

**Analyze the provided files (`cornerTactics_docs/`, `cornerTactics_results/`, `cornerTactics_data_processed/`, `cornerTactics_notes/`) and write the complete research report following this structure.**

**Be thorough, quantitative, and insightful. Use specific numbers from the results. Tell the story of what this ablation study reveals about corner kick prediction.**

**Output format: Markdown (.md) suitable for conversion to PDF/LaTeX.**
