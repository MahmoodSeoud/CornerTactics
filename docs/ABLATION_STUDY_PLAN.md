# Ablation Study: Raw Data → Engineered Features

**Research Question**: What is the incremental value of each engineered feature over raw StatsBomb data?

---

## Experimental Design

### **Phase 1: Baseline (Raw Data Only)**

**Features (13 raw fields, no engineering):**
1. `period` - Match period (1 or 2)
2. `minute` - Minute of match
3. `second` - Second within minute
4. `duration` - Event duration
5. `corner_x` - Corner location x (≈120)
6. `corner_y` - Corner location y (0-80)
7. `pass_end_x` - Ball landing x
8. `pass_end_y` - Ball landing y
9. `pass_length` - Pass distance
10. `pass_angle` - Pass angle (radians)
11. `pass_height_id` - Height ID (1, 2, 3)
12. `pass_body_part_id` - Body part ID (38, 40)
13. `pass_technique_id` - Technique ID (104, 105, 106, null)

**No freeze frame data, no aggregations, no derived features**

---

### **Phase 2: Progressive Feature Addition**

#### **Step 1: Add Freeze Frame Counts (6 features)**
```
Baseline (13) + Player Counts (6) = 19 features
```

Added features:
- `total_attacking`
- `total_defending`
- `attacking_in_box`
- `defending_in_box`
- `attacking_near_goal`
- `defending_near_goal`

**Hypothesis**: Simple player counts provide basic tactical context

---

#### **Step 2: Add Spatial Density (4 features)**
```
Step 1 (19) + Spatial Density (4) = 23 features
```

Added features:
- `attacking_density`
- `defending_density`
- `numerical_advantage`
- `attacker_defender_ratio`

**Hypothesis**: Density metrics capture crowding effects beyond counts

---

#### **Step 3: Add Positional Features (8 features)**
```
Step 2 (23) + Positional (8) = 31 features
```

Added features:
- `attacking_centroid_x`
- `attacking_centroid_y`
- `defending_centroid_x`
- `defending_centroid_y`
- `defending_compactness`
- `defending_depth`
- `attacking_to_goal_dist`
- `defending_to_goal_dist`

**Hypothesis**: Team shape/positioning provides tactical intelligence

---

#### **Step 4: Add Pass Technique Encoding (2 features)**
```
Step 3 (31) + Technique (2) = 33 features
```

Added features:
- `is_inswinging`
- `is_outswinging`

**Hypothesis**: Corner delivery technique affects outcome (already have technique_id in raw)

---

#### **Step 5: Add Pass Outcome Context (4 features)**
```
Step 4 (33) + Outcome Context (4) = 37 features
```

Added features:
- `pass_outcome` (encoded)
- `is_cross_field_switch`
- `has_recipient`
- `is_shot_assist`

**Hypothesis**: Pass context provides outcome hints

---

#### **Step 6: Add Goalkeeper Features (3 features)**
```
Step 5 (37) + Goalkeeper (3) = 40 features
```

Added features:
- `num_attacking_keepers`
- `num_defending_keepers`
- `keeper_distance_to_goal`

**Hypothesis**: Goalkeeper positioning critical for aerial balls

---

#### **Step 7: Add Score State (4 features)**
```
Step 6 (40) + Score State (4) = 44 features
```

Added features:
- `attacking_team_goals`
- `defending_team_goals`
- `score_difference`
- `match_situation`

**Hypothesis**: Score affects tactical urgency (trailing teams commit more attackers)

---

#### **Step 8: Add Substitution Patterns (3 features)**
```
Step 7 (44) + Substitutions (3) = 47 features
```

Added features:
- `total_subs_before`
- `recent_subs_5min`
- `minutes_since_last_sub`

**Hypothesis**: Fresh players or tactical changes affect execution

---

#### **Step 9: Add Remaining Metadata (2 features)**
```
Step 8 (47) + Metadata (2) = 49 features (FULL)
```

Added features:
- `corner_side` (derived from corner_y)
- `timestamp_seconds` (derived from timestamp)

**Hypothesis**: Minimal impact (redundant with existing features)

---

## Metrics to Track Per Step

For each model configuration (13, 19, 23, 31, 33, 37, 40, 44, 47, 49 features):

### **Performance Metrics**
1. Test Accuracy
2. Test F1 Macro
3. Test F1 Weighted
4. ROC-AUC (for binary shot prediction)
5. PR-AUC (for binary shot prediction)
6. Per-class precision/recall/F1

### **Model Complexity**
7. Training time
8. Inference time (per sample)
9. Model size (bytes)
10. Number of parameters (for MLP)

### **Feature Importance**
11. Top 10 most important features (Random Forest)
12. Feature importance scores (all features)

---

## Correlation Analysis

### **Event Transition Matrix**

Compute **P(next_event | corner_features)** for each feature set:

```
                Ball Receipt  Clearance  Goalkeeper  Other
Baseline (raw)      0.543       0.234      0.101     0.121
+ Counts            0.XXX       0.XXX      0.XXX     0.XXX
+ Density           0.XXX       0.XXX      0.XXX     0.XXX
...
Full (49)           0.XXX       0.XXX      0.XXX     0.XXX
```

### **Feature-Outcome Correlation Matrix**

For each feature, compute correlation with outcome classes:

| Feature | Ball Receipt | Clearance | Goalkeeper | Other |
|---------|--------------|-----------|------------|-------|
| `attacking_in_box` | +0.XX | -0.XX | -0.XX | -0.XX |
| `defending_in_box` | -0.XX | +0.XX | +0.XX | -0.XX |
| `keeper_distance_to_goal` | -0.XX | -0.XX | +0.XX | -0.XX |
| `score_difference` | +0.XX | -0.XX | -0.XX | +0.XX |
| ... | ... | ... | ... | ... |

### **Feature-Feature Correlation Heatmap**

49×49 matrix showing redundancy:
- High correlation (>0.8): Potentially redundant features
- Negative correlation (<-0.5): Complementary features
- Near-zero correlation: Independent information

---

## Experimental Protocol

### **1. Data Preparation**

```python
# Extract raw features (baseline)
raw_features = extract_raw_features(corners)  # 13 features

# Extract progressive feature sets
step1_features = extract_features_step1(corners)  # 19 features
step2_features = extract_features_step2(corners)  # 23 features
...
step9_features = extract_all_features(corners)    # 49 features
```

### **2. Model Training**

For each feature set (9 configurations):
```python
for feature_set in [raw, step1, step2, ..., step9]:
    # Train 3 models
    rf_model = RandomForestClassifier().fit(X_train, y_train)
    xgb_model = XGBoostClassifier().fit(X_train, y_train)
    mlp_model = MLPClassifier().fit(X_train, y_train)

    # Evaluate on test set
    metrics = evaluate(model, X_test, y_test)

    # Save results
    save_ablation_results(feature_set, metrics)
```

### **3. Results Analysis**

```python
# Performance vs feature count
plot_performance_curve(steps, accuracies)

# Feature importance evolution
plot_feature_importance_evolution(steps)

# Correlation analysis
plot_correlation_matrices(feature_sets)

# Transition matrix analysis
compute_transition_matrices(feature_sets, outcomes)
```

---

## Expected Outputs

### **1. Performance Progression Plot**

```
Test Accuracy vs Feature Count
100% ┤
 90% ┤                                 ╭─────────
 80% ┤                       ╭─────────╯
 70% ┤              ╭────────╯
 60% ┤     ╭────────╯
 50% ┤─────╯
     └─────────────────────────────────────────
     13   19   23   31   33   37   40   44   49
         Number of Features
```

### **2. Feature Group Contribution**

| Step | Features Added | Accuracy | Δ Accuracy | Cumulative Gain |
|------|---------------|----------|------------|-----------------|
| 0 (Baseline) | Raw data (13) | 55.0% | - | - |
| 1 | Player counts (6) | 68.5% | +13.5% | +13.5% |
| 2 | Spatial density (4) | 72.1% | +3.6% | +17.1% |
| 3 | Positional (8) | 78.3% | +6.2% | +23.3% |
| 4 | Technique (2) | 79.1% | +0.8% | +24.1% |
| 5 | Outcome context (4) | 80.4% | +1.3% | +25.4% |
| 6 | Goalkeeper (3) | 81.2% | +0.8% | +26.2% |
| 7 | Score state (4) | 81.8% | +0.6% | +26.8% |
| 8 | Substitutions (3) | 81.8% | +0.0% | +26.8% |
| 9 | Metadata (2) | 81.8% | +0.0% | +26.8% |

### **3. Transition Matrix Evolution**

**Baseline (Raw Data):**
```
P(outcome | raw_features)
                Ball Receipt  Clearance  Goalkeeper  Other
Predicted BR         0.543       0.234      0.101     0.121
Predicted CL         0.235       0.543      0.111     0.111
Predicted GK         0.100       0.150      0.600     0.150
Predicted OT         0.120       0.130      0.120     0.630
```

**Full Features (49):**
```
P(outcome | all_features)
                Ball Receipt  Clearance  Goalkeeper  Other
Predicted BR         0.950       0.025      0.015     0.010
Predicted CL         0.030       0.850      0.080     0.040
Predicted GK         0.010       0.100      0.820     0.070
Predicted OT         0.100       0.200      0.050     0.650
```

### **4. Feature Importance Rankings**

**Top 10 Most Important Features (Random Forest):**
1. `attacking_in_box` (importance: 0.125)
2. `defending_in_box` (importance: 0.118)
3. `pass_end_x` (importance: 0.095)
4. `keeper_distance_to_goal` (importance: 0.082)
5. `attacking_density` (importance: 0.071)
6. `defending_centroid_x` (importance: 0.065)
7. `score_difference` (importance: 0.058)
8. `numerical_advantage` (importance: 0.054)
9. `is_inswinging` (importance: 0.047)
10. `pass_length` (importance: 0.043)

---

## Implementation Steps

### **Step 1: Create Raw Feature Extractor**

```python
# scripts/extract_raw_features.py
def extract_raw_features(corner):
    """Extract only raw fields, no engineering"""
    event = corner['event']
    pass_data = event['pass']

    return {
        'period': event['period'],
        'minute': event['minute'],
        'second': event['second'],
        'duration': event['duration'],
        'corner_x': event['location'][0],
        'corner_y': event['location'][1],
        'pass_end_x': pass_data['end_location'][0],
        'pass_end_y': pass_data['end_location'][1],
        'pass_length': pass_data['length'],
        'pass_angle': pass_data['angle'],
        'pass_height_id': pass_data['height']['id'],
        'pass_body_part_id': pass_data['body_part']['id'],
        'pass_technique_id': pass_data.get('technique', {}).get('id', -1)
    }
```

### **Step 2: Create Progressive Feature Extractors**

```python
# scripts/extract_features_progressive.py
def extract_features_step1(corner):
    """Baseline + Player Counts"""
    features = extract_raw_features(corner)
    features.update(extract_player_counts(corner['freeze_frame']))
    return features

def extract_features_step2(corner):
    """Step 1 + Spatial Density"""
    features = extract_features_step1(corner)
    features.update(extract_spatial_density(corner['freeze_frame']))
    return features

# ... step3 through step9
```

### **Step 3: Create Ablation Training Script**

```python
# scripts/09_ablation_study.py
# Train models on each feature set
# Track metrics at each step
# Generate comparison plots
```

### **Step 4: Create Analysis Script**

```python
# scripts/10_analyze_ablation.py
# Correlation matrices
# Transition matrices
# Feature importance evolution
# Performance visualization
```

---

## Timeline

1. **Create raw feature extractor** (1 hour)
2. **Create progressive extractors** (2 hours)
3. **Extract all feature sets** (30 min)
4. **Train models on all configurations** (3 hours - 9 configs × 3 models × 2 tasks)
5. **Correlation & transition analysis** (2 hours)
6. **Visualization & reporting** (2 hours)

**Total: ~10-12 hours** (mostly training time)

---

## Research Contribution

This ablation study will show:

1. ✅ **Which features matter most** for corner kick prediction
2. ✅ **Diminishing returns** of feature engineering
3. ✅ **Minimal viable feature set** for deployment
4. ✅ **Feature redundancy** through correlation analysis
5. ✅ **Transition dynamics** P(outcome | features)
6. ✅ **Engineering vs raw data** performance gap

This is **publication-quality** experimental design!
