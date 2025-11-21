# Data Leakage Findings: Corner Kick Prediction Models

**Date:** 2025-11-21
**Critical Discovery:** Previous models contained severe temporal data leakage

---

## Executive Summary

### The Problem
The "optimal" corner kick prediction model achieving 87.97% accuracy was fundamentally flawed. It used features that are only known AFTER the corner kick is completed, making it unsuitable for real-world prediction.

### Key Findings

1. **33% of features contained temporal leakage** (8 out of 24 features)
2. **The top-performing feature was pure leakage** (`is_shot_assist` provided +3.09% gain)
3. **True predictive accuracy is only 71%** (not 88%)
4. **AUC dropped to 0.52** (barely better than random guessing)

### Impact
- **16.65% performance drop** when leakage removed
- Models were essentially "predicting" things they already knew
- Previous results cannot be used for production or publications

---

## Leaked Features Identified

### Critical Leaks (Used in "Optimal" Model)

| Feature | Why It's Leaked | Impact |
|---------|-----------------|--------|
| `pass_end_x/y` | Actual ball landing position (outcome) | Used as if it were "intended target" |
| `is_shot_assist` | Whether corner led to shot (the thing we're predicting!) | +3.09% accuracy gain |
| `duration` | Event duration only known after completion | Time-based leak |
| `pass_length` | Calculated from actual trajectory | Distance leak |
| `pass_recipient_id` | Who actually received the ball | Outcome data |
| `has_pass_outcome` | Pass success flag | Outcome indicator |
| `is_aerial_won` | Aerial duel result | Post-corner event |
| `has_recipient` | Whether pass was successful | Outcome data |

### The Most Egregious Leak

**`is_shot_assist`** - This feature literally tells the model whether the corner led to a shot, which is what we're trying to predict! It provided the largest single performance gain (+3.09%) in the "optimal" model.

---

## Performance Comparison

### Models WITH Temporal Leakage

| Model | Test Accuracy | AUC | Features Used |
|-------|---------------|-----|---------------|
| Random Forest (optimal) | **87.97%** | 0.8486 | 24 (including leaks) |
| XGBoost | 85.22% | 0.8484 | 24 (including leaks) |
| MLP | 71.82% | 0.5499 | 24 (including leaks) |

### Models WITHOUT Temporal Leakage (Valid)

| Model | Test Accuracy | AUC | Features Used |
|-------|---------------|-----|---------------|
| MLP | **71.32%** | 0.5209 | 19 (all valid) |
| Random Forest | 63.57% | 0.5051 | 19 (all valid) |
| XGBoost | 60.47% | 0.5086 | 19 (all valid) |

### Performance Drop
- **Best accuracy dropped from 87.97% to 71.32%** (-16.65%)
- **AUC dropped from 0.85 to 0.52** (barely better than random)
- **Random Forest dropped most**: 87.97% → 63.57% (-24.4%)

---

## Valid Features for Prediction

### Features We CAN Use (Available at Corner Time)

**Event Data:**
- `second`, `minute`, `period` - Match timing
- `corner_x`, `corner_y` - Corner starting position
- `player_id`, `position_id` - Who's taking the corner
- `team_id`, `possession_team_id` - Team information

**360° Freeze Frame Data:**
- `total_attacking`, `total_defending` - Player counts
- `attacking_in_box`, `defending_in_box` - Box positioning
- `attacking_near_goal`, `defending_near_goal` - Near goal positioning
- `attacking_density`, `defending_density` - Spatial concentration
- `numerical_advantage` - Attacking vs defending players
- `defending_depth`, `defending_to_goal_dist` - Defensive positioning

---

## Empirical Evidence of Leakage

### Proof That pass_end_x/y Are Actual Outcomes

We verified this by examining the raw data:

```bash
# Compare corner kick data
Corner position:     [120.0, 80.0]  (corner arc)
pass_end_location:   [109.4, 35.5]   (from corner event)
pass_length:         45.7 units

# These match EXACTLY where the ball actually went:
Corner at [120.0, 80.0] → pass_end [109.4, 35.5] → pass_length 45.7
Corner at [120.0, 0.1] → pass_end [113.7, 30.8] → pass_length 31.3
```

**Conclusion:** `pass_end_x/y` are NOT "intended delivery zones" - they are the actual landing coordinates, only known after the kick is completed.

---

## Why This Happened

### 1. Misleading Feature Names
- `pass_end_x/y` sounds like "intended target" but is actually "where ball landed"
- `is_shot_assist` sounds predictive but is actually the outcome

### 2. Lack of Temporal Thinking
- Features were selected based on predictive power, not temporal availability
- Backward selection kept leaked features because they improved accuracy

### 3. Too-Good-To-Be-True Performance
- 88% accuracy should have been a red flag for this difficult task
- Real-world corner → shot prediction is inherently noisy

---

## Lessons Learned

### 1. Always Check Temporal Availability
Before using any feature, ask: "Is this known BEFORE the event I'm predicting?"

### 2. Suspicious Performance = Check for Leakage
If accuracy seems too good, it probably is. Corner kick outcomes are inherently unpredictable.

### 3. Feature Engineering Must Be Careful
When creating features from event data, clearly distinguish:
- **Pre-event state** (valid for prediction)
- **Event execution** (partially valid, needs care)
- **Post-event outcomes** (never valid for prediction)

### 4. Document Feature Definitions
Every feature should have clear documentation about:
- When it's measured
- What it represents
- Whether it's suitable for prediction

---

## Recommendations Going Forward

### 1. Immediate Actions
- ✅ **Stop using leaked models** immediately
- ✅ **Retrain all models** with valid features only
- ✅ **Update documentation** to reflect true performance
- ⚠️ **Audit any published results** that used leaked features

### 2. New Baseline Performance
- **Expected accuracy: 70-75%** for corner → shot prediction
- **Expected AUC: 0.55-0.60** (slight improvement possible with better features)
- **This is realistic** for this problem domain

### 3. Focus on Valid Feature Engineering
Since we lose high-value leaked features, invest in better valid features:
- Player skill ratings (historical shooting accuracy)
- Team tactical patterns (from previous corners)
- Match context (score, time remaining)
- Weather conditions (if available)
- Referee tendencies (card-happy refs change defensive aggression)

### 4. Consider Alternative Targets
Instead of binary shot prediction, consider:
- **Zone prediction**: Where will the ball land? (6-yard, penalty spot, far post)
- **Outcome type**: Clearance vs Reception vs Goalkeeper
- **Danger score**: Continuous measure of opportunity quality

---

## Technical Details

### Scripts Created
1. `scripts/14_extract_temporally_valid_features.py` - Removes leaked features
2. `scripts/15_retrain_without_leakage.py` - Retrains models properly

### Datasets
- **Original**: `corners_features_with_shot.csv` (contains leaks)
- **Cleaned**: `corners_features_temporal_valid.csv` (no leaks)

### Results Location
- `results/no_leakage/training_results_no_leakage.json`

---

## Conclusion

The discovery of temporal data leakage fundamentally changes our understanding of corner kick predictability. While the leaked models achieved impressive 88% accuracy, the true predictive performance is only 71% - barely better than predicting the majority class.

This is a sobering but important finding. It shows that:
1. **Corner kick outcomes are largely unpredictable** from pre-kick features alone
2. **The most predictive information comes from the execution** (where the ball actually goes)
3. **Future work should focus on execution prediction** rather than outcome prediction

The silver lining: We now have honest baselines and can focus on genuinely improving predictive performance through better feature engineering and novel data sources.

---

## Quote for Reflection

> "In data science, as in life, if something seems too good to be true, it probably is. The path from 88% to 71% accuracy is not a failure - it's a journey from self-deception to truth."

- The CornerTactics Team, after discovering the leakage