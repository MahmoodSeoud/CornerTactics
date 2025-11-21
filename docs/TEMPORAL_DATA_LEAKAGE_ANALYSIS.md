# Temporal Data Leakage Analysis: Corner Kick Features

**Date:** 2025-11-21
**Critical Finding:** Many optimal features contain temporal leakage - they're only known AFTER the corner is taken

---

## Executive Summary

**CRITICAL ISSUE DISCOVERED**: The current "optimal" model achieving 87.97% accuracy is fundamentally flawed due to temporal data leakage. Many features used for training are only available AFTER the corner kick outcome is known, making the model unsuitable for real-world prediction.

### Key Findings

1. **8 out of 24 "optimal" features contain definite leakage** (33% of features)
2. **The top performing feature `is_shot_assist` (+3.09% gain) is pure leakage**
3. **True predictive features likely achieve ~75-80% accuracy** (not 87.97%)
4. **Models must be retrained using only pre-kick features**

---

## Feature Temporal Analysis

### Categories

- ✅ **AVAILABLE BEFORE**: Can be used for prediction
- ❌ **LEAKED (AFTER)**: Only known after corner is taken
- ⚠️ **AMBIGUOUS**: Needs clarification from data source

---

## Raw Features Analysis (21 features)

### Temporal Features

| Feature | Available | Type | Reasoning |
|---------|-----------|------|-----------|
| `second` | ✅ BEFORE | Valid | Match time when corner is awarded is known |
| `duration` | ❌ **LEAKED** | After | Duration only measurable after event completes |
| `index` | ✅ BEFORE | Valid | Event sequence position known when corner awarded |
| `possession` | ✅ BEFORE | Valid | Possession count known at corner award |

**Analysis**:
- `duration` is a clear leak - you can't know how long a corner will take until it's finished
- `second` and `index` provide temporal context that's legitimately available
- `possession` tracks game flow up to the corner

### Spatial Features

| Feature | Available | Type | Reasoning |
|---------|-----------|------|-----------|
| `location_x` | ✅ BEFORE | Valid | Corner kick origin is fixed (corner arc) |
| `location_y` | ✅ BEFORE | Valid | Which side of goal (left/right corner) |
| `pass_length` | ❌ **LEAKED** | After | Actual pass distance only known after ball lands |
| `pass_end_x` | ❌ **LEAKED** | After | Where ball actually lands - outcome data |
| `pass_end_y` | ❌ **LEAKED** | After | Where ball actually lands - outcome data |

**Analysis**:
- Corner starting position (`location_x/y`) is valid - it's where the corner is taken from
- **CRITICAL LEAK**: `pass_end_x/y` are the actual landing positions - these ARE the outcome!
- `pass_length` calculated from actual trajectory is also leaked

### Player Context Features

| Feature | Available | Type | Reasoning |
|---------|-----------|------|-----------|
| `player_id` | ✅ BEFORE | Valid | Corner taker is decided before kick |
| `position_id` | ✅ BEFORE | Valid | Player's position/role is known |
| `play_pattern_id` | ✅ BEFORE | Valid | How corner was earned (e.g., from open play) |
| `possession_team_id` | ✅ BEFORE | Valid | Which team has the corner |

**Analysis**: All player context features are legitimately available before the kick.

### Pass Attribute Features

| Feature | Available | Type | Reasoning |
|---------|-----------|------|-----------|
| `pass_body_part_id` | ⚠️ AMBIGUOUS | Unclear | Could be intent OR actual execution |
| `pass_type_id` | ✅ BEFORE | Valid | "Corner" type is known beforehand |
| `pass_technique_id` | ⚠️ AMBIGUOUS | Unclear | Inswing/outswing - intent or actual? |
| `pass_recipient_id` | ❌ **LEAKED** | After | Who actually received the ball |

**Analysis**:
- `pass_recipient_id` is clearly leaked - you don't know who receives it until after
- Body part and technique need investigation - are these planned or actual?
- `pass_type_id` should just be "Corner" for all records

### Event Outcome Features

| Feature | Available | Type | Reasoning |
|---------|-----------|------|-----------|
| `under_pressure` | ⚠️ AMBIGUOUS | Unclear | When is pressure assessed? |
| `has_pass_outcome` | ❌ **LEAKED** | After | Outcome flag - definitionally after event |
| `is_aerial_won` | ❌ **LEAKED** | After | Aerial duel result happens after corner |

**Analysis**:
- All outcome-related features are leaks by definition
- `under_pressure` needs clarification - is this pressure on corner taker at kick time?

### Freeze Frame Features (360° data)

| Feature | Available | Type | Reasoning |
|---------|-----------|------|-----------|
| `total_attacking` | ✅ BEFORE | Valid | Player count from freeze frame at kick moment |
| `total_defending` | ✅ BEFORE | Valid | Player count from freeze frame at kick moment |

**Analysis**: Freeze frame features captured AT the moment of corner kick are valid.

---

## Engineered Features Analysis (10 top features examined)

### Features Used in Optimal Set (3 features)

| Feature | Available | Type | Reasoning |
|---------|-----------|------|-----------|
| `is_shot_assist` | ❌ **LEAKED** | After | Only known if next event is a shot |
| `has_recipient` | ❌ **LEAKED** | After | Success of pass only known after |
| `defending_to_goal_dist` | ✅ BEFORE | Valid | Calculated from freeze frame positions |

### Additional Top Engineered Features (7 more)

| Feature | Available | Type | Reasoning |
|---------|-----------|------|-----------|
| `pass_outcome_encoded` | ❌ **LEAKED** | After | Pass outcome only known after event |
| `defending_depth` | ✅ BEFORE | Valid | Defensive line position from freeze frame |
| `defending_team_goals` | ✅ BEFORE | Valid | Match state - goals conceded so far |
| `defending_in_box` | ✅ BEFORE | Valid | Defender count in box from freeze frame |
| `attacking_near_goal` | ✅ BEFORE | Valid | Attacker positions from freeze frame |
| `corner_side` | ✅ BEFORE | Valid | Left/right corner - known beforehand |
| `is_cross_field_switch` | ❌ **LEAKED** | After | Whether ball switched sides - outcome |

**Analysis**:
- **CRITICAL**: `is_shot_assist` - the best performing feature (+3.09%) - is pure leakage!
- `has_recipient` and `pass_outcome_encoded` are outcome data
- `is_cross_field_switch` requires knowing where ball ended up
- Good news: Several engineered features ARE valid (defending positions, match state)

---

## Summary Table: All 24 "Optimal" Features

### ✅ Valid for Prediction (16 features available)

**From Optimal Set (13 features):**
1. `second` - Match time
2. `index` - Event sequence
3. `possession` - Possession count
4. `location_x` - Corner origin X
5. `player_id` - Corner taker
6. `position_id` - Player position
7. `play_pattern_id` - How corner earned
8. `possession_team_id` - Team taking corner
9. `pass_type_id` - Should be "Corner" for all
10. `total_attacking` - From freeze frame
11. `total_defending` - From freeze frame
12. `defending_to_goal_dist` - From freeze frame

**Additional Valid Features (not in optimal but available):**
13. `location_y` - Corner origin Y (marked as "harmful" but actually valid)
14. `defending_depth` - Defensive line depth from freeze frame
15. `defending_team_goals` - Current score state
16. `defending_in_box` - Defenders in penalty box
17. `attacking_near_goal` - Attackers near goal
18. `corner_side` - Left/right side

### ❌ Leaked Features (8 features) - MUST REMOVE
1. **`duration`** - Event duration
2. **`pass_length`** - Actual pass distance
3. **`pass_end_x`** - Ball landing X
4. **`pass_end_y`** - Ball landing Y
5. **`pass_recipient_id`** - Who received ball
6. **`has_pass_outcome`** - Outcome flag
7. **`is_aerial_won`** - Aerial duel result
8. **`is_shot_assist`** - Whether corner led to shot

### ⚠️ Ambiguous - Need Investigation (3 features)
1. `pass_body_part_id` - Intent or execution?
2. `pass_technique_id` - Planned or actual?
3. `under_pressure` - When assessed?

---

## Impact Analysis

### Current "Optimal" Model Performance
- **Reported accuracy**: 87.97%
- **Features used**: 24 (including 8+ leaked)
- **Top feature**: `is_shot_assist` (+3.09% gain) - **LEAKED**

### Expected True Performance (without leaks)
- **Valid features**: 13-16 (depending on ambiguous)
- **Expected accuracy**: ~75-80% (rough estimate)
- **Performance drop**: -8-13%

### Why Leakage Wasn't Caught

1. **`pass_end_x/y` confusion**: These sound like "intended target" but are actually "where ball landed"
2. **Feature naming**: `is_shot_assist` sounds predictive but is actually outcome data
3. **High performance bias**: 88% accuracy should have raised red flags for this task
4. **Backward selection kept leaks**: The search optimized for accuracy, not validity

---

## Valid Features for Retraining

### Definitely Safe to Use (Pre-Corner Features)

#### From Event Data
```python
VALID_EVENT_FEATURES = [
    # Temporal context
    'second',           # When in match
    'index',            # Event sequence
    'possession',       # Possession number

    # Spatial context
    'location_x',       # Corner position X (should be ~120 or ~0)
    'location_y',       # Corner position Y (should be ~0 or ~80)

    # Team/Player context
    'player_id',        # Who's taking corner
    'position_id',      # Player's position
    'play_pattern_id',  # How corner was earned
    'possession_team_id', # Team taking corner
    'pass_type_id',     # Type (should always be 'Corner')
]
```

#### From Freeze Frame (360° Data)
```python
VALID_FREEZE_FRAME_FEATURES = [
    # Player counts
    'total_attacking',
    'total_defending',

    # Spatial distributions (must be calculated from freeze frame)
    'defending_to_goal_dist',
    'attacking_spread',        # Variance of attacking positions
    'defending_spread',        # Variance of defending positions
    'attacking_box_count',     # Players in penalty box
    'defending_box_count',
    'attacking_six_yard_count', # Players in 6-yard box
    'defending_six_yard_count',

    # Relative positioning
    'numerical_advantage',     # attacking - defending in key areas
    'space_control_metric',    # Voronoi-based space control
]
```

### Additional Valid Features to Engineer

Since we lose many features, we should engineer new VALID ones from freeze frames:

```python
POTENTIAL_NEW_FEATURES = [
    # Defensive shape
    'defensive_line_height',    # How high is defensive line
    'defensive_compactness',    # How tight is defensive shape
    'goalkeeper_position_x',    # GK positioning
    'goalkeeper_position_y',

    # Attacking positioning
    'attackers_near_post',      # Count near post
    'attackers_far_post',       # Count far post
    'attackers_penalty_spot',   # Count around penalty spot
    'tallest_attacker_marked',  # Is tallest player marked

    # Match context (if available)
    'score_differential',       # Current score difference
    'minutes_remaining',        # Time left in match
    'is_home_team',            # Home/away
    'yellow_cards_attacking',   # Disciplinary context
    'yellow_cards_defending',
]
```

---

## Recommendations

### 1. Immediate Actions

1. **STOP using current "optimal" model** - it contains severe data leakage
2. **Retrain immediately** using only valid features
3. **Document actual predictive performance** (likely 75-80%, not 88%)
4. **Rename misleading features** in codebase:
   - `pass_end_x/y` → `actual_landing_x/y` or `outcome_x/y`
   - `is_shot_assist` → `outcome_was_shot_assist`

### 2. Feature Engineering Focus

Since we lose high-value leaked features, focus on extracting more from freeze frames:

- **Marking assignments**: Who's marking whom
- **Space control**: Voronoi diagrams, pitch control
- **Formation shape**: Defensive block compactness
- **Key player tracking**: Tallest players, known headers
- **Movement vectors**: If available, player velocities/directions

### 3. Model Retraining Strategy

```python
# Proposed training pipeline
def create_valid_corner_features(corner_event, freeze_frame):
    """
    Extract ONLY features available at corner kick moment
    """
    features = {}

    # Valid event features
    features['second'] = corner_event['second']
    features['location_x'] = corner_event['location']['x']
    features['location_y'] = corner_event['location']['y']
    features['player_id'] = corner_event['player']['id']
    # ... other valid features

    # Freeze frame features
    if freeze_frame:
        features['total_attacking'] = len(freeze_frame['attacking_players'])
        features['total_defending'] = len(freeze_frame['defending_players'])
        # ... compute spatial metrics

    # EXPLICITLY EXCLUDE
    # - duration
    # - pass_end_x/y
    # - is_shot_assist
    # - has_pass_outcome
    # - is_aerial_won
    # - pass_recipient_id

    return features
```

### 4. Validation Requirements

1. **Temporal split validation**: Train on earlier matches, test on later
2. **Proper baselines**:
   - Random: ~50% for shot prediction
   - Class distribution: ~10% if 10% of corners lead to shots
3. **Sanity checks**: Model should NOT achieve >85% on this task
4. **Feature importance**: Top features should make soccer sense

### 5. Research Integrity

1. **Report true performance** in any publications
2. **Acknowledge the leakage** in documentation
3. **Compare against proper baselines**
4. **Focus on practical applicability** - can this predict in real-time?

---

## Conclusion

The current "optimal" model is **fundamentally flawed** due to temporal data leakage. The features `pass_end_x/y` are not "intended targets" but actual outcomes, and `is_shot_assist` is literally the thing we're trying to predict.

**Next steps:**
1. Retrain using only the 13 confirmed valid features
2. Engineer new features from freeze frames
3. Accept that true predictive accuracy is likely 75-80%
4. Focus on model interpretability and practical use cases

**Remember**: A 75% accurate model that works in real-time is infinitely more valuable than an 88% accurate model that requires knowing the future.

---

## Files to Update

1. `scripts/12_optimal_feature_search.py` - Remove leaked features
2. `scripts/13_interaction_pair_testing.py` - Rerun without leaked features
3. `docs/OPTIMAL_FEATURE_SELECTION.md` - Add warning about leakage
4. Create new script: `scripts/14_temporal_valid_feature_extraction.py`
5. Create new script: `scripts/15_retrain_without_leakage.py`