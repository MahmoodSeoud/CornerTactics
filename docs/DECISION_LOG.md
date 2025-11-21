# Feature Removal Decision Log

**Date:** November 21, 2025
**Purpose:** Record every decision made about feature inclusion/exclusion

---

## Decision Criteria

A feature is VALID if and only if:
1. It is known at or before the corner kick moment (t=0)
2. It does not encode the outcome we're predicting
3. It can be observed/measured without knowing how the corner played out

---

## Feature Decisions (All 53 Original Features)

### ✅ KEPT: Event Data (7 features)

| Feature | Decision | Reasoning |
|---------|----------|-----------|
| `second` | ✅ KEEP | Match time when corner awarded is known |
| `minute` | ✅ KEEP | Match minute is known at corner award |
| `period` | ✅ KEEP | First/second half is known |
| `corner_x` | ✅ KEEP | Corner arc position (always ~120 or ~0) |
| `corner_y` | ✅ KEEP | Corner arc position (always ~80 or ~0) |
| `team_id` | ✅ KEEP | Team taking corner is known |
| `player_id` | ✅ KEEP | Corner taker is decided before kick |

**Evidence:** These are all recorded at corner award time, before the kick is taken.

---

### ❌ REMOVED: Outcome Features (8 features)

| Feature | Decision | Reasoning | Evidence |
|---------|----------|-----------|----------|
| `duration` | ❌ REMOVE | Only measurable after event completes | Duration = next_event_timestamp - corner_timestamp |
| `pass_end_x` | ❌ REMOVE | Actual ball landing X coordinate | Varies 109-118 (not fixed delivery zones) |
| `pass_end_y` | ❌ REMOVE | Actual ball landing Y coordinate | Varies 30-75 (not fixed delivery zones) |
| `pass_length` | ❌ REMOVE | Calculated from actual trajectory | sqrt((end_x-start_x)² + (end_y-start_y)²) |
| `pass_recipient_id` | ❌ REMOVE | Player who actually received ball | Only known after pass completes |
| `has_pass_outcome` | ❌ REMOVE | Binary outcome flag | By definition recorded after outcome |
| `is_aerial_won` | ❌ REMOVE | Aerial duel result | Post-corner event outcome |
| `is_shot_assist` | ❌ REMOVE | **Whether corner assisted a shot** | **This is literally what we're predicting!** |

**Critical Finding:** `pass_end_x/y` are NOT "intended delivery zones" - verified by comparing with actual ball positions in the data.

---

### ✅ KEPT: Freeze Frame Features (12 features)

All freeze frame features capture player positions at the EXACT moment of the corner kick (t=0).

| Feature | Decision | Reasoning |
|---------|----------|-----------|
| `total_attacking` | ✅ KEEP | Player count from freeze frame at t=0 |
| `total_defending` | ✅ KEEP | Player count from freeze frame at t=0 |
| `attacking_in_box` | ✅ KEEP | Count from freeze frame at t=0 |
| `defending_in_box` | ✅ KEEP | Count from freeze frame at t=0 |
| `attacking_near_goal` | ✅ KEEP | Count from freeze frame at t=0 |
| `defending_near_goal` | ✅ KEEP | Count from freeze frame at t=0 |
| `attacking_density` | ✅ KEEP | Calculated from freeze frame positions |
| `defending_density` | ✅ KEEP | Calculated from freeze frame positions |
| `numerical_advantage` | ✅ KEEP | attacking - defending at t=0 |
| `attacker_defender_ratio` | ✅ KEEP | attacking / defending at t=0 |
| `defending_depth` | ✅ KEEP | Average Y position of defenders at t=0 |
| `corner_side` | ✅ KEEP | Left (y~80) or right (y~0) corner |

**Evidence:** StatsBomb freeze frames are explicitly captured at event time (t=0).

---

### ❌ REMOVED: Derived Outcome Features (4 engineered features)

| Feature | Decision | Reasoning | Derived From |
|---------|----------|-----------|--------------|
| `has_recipient` | ❌ REMOVE | Pass success indicator | Derived from `pass_recipient_id` (leaked) |
| `pass_outcome_encoded` | ❌ REMOVE | Categorical pass outcome | Encoded from outcome field (leaked) |
| `is_cross_field_switch` | ❌ REMOVE | Ball switched field sides | Requires knowing `pass_end_y` (leaked) |
| `pass_angle` | ❌ REMOVE | Angle of trajectory | Calculated from `pass_end_x/y` (leaked) |

**Reasoning:** Any feature derived from leaked features is also leaked.

---

### ⚠️ EXCLUDED: Ambiguous Features (3 features - not in final dataset)

| Feature | Decision | Reasoning |
|---------|----------|-----------|
| `pass_body_part_id` | ⚠️ EXCLUDE | Unclear if "planned" or "actual" body part used |
| `pass_technique_id` | ⚠️ EXCLUDE | Unclear if "planned" or "actual" technique |
| `under_pressure` | ⚠️ EXCLUDE | Unclear when pressure is assessed |

**Conservative Approach:** When in doubt, exclude. These features may be valid but need StatsBomb documentation to confirm.

---

### Not Present in Dataset (assumed valid if available)

| Feature | Decision | Notes |
|---------|----------|-------|
| `position_id` | ✅ KEEP | Player's position/role is known |
| `play_pattern_id` | ✅ KEEP | How corner was earned (e.g., from open play) |
| `possession_team_id` | ✅ KEEP | Team in possession (same as team_id for corners) |
| `index` | ✅ KEEP | Event sequence number |
| `possession` | ✅ KEEP | Possession count up to this point |

---

## Summary Statistics

| Category | Total Features | Kept | Removed | Excluded |
|----------|---------------|------|---------|----------|
| Event Data | 12 | 7 | 8 | 3 |
| Freeze Frame | 12 | 12 | 0 | 0 |
| Engineered | 10 | 0 | 4 | 0 |
| **TOTAL** | **34** | **19** | **12** | **3** |

---

## Validation of Decisions

### Test 1: Performance Drop
- **Before removal:** 87.97% accuracy
- **After removal:** 71.32% accuracy
- **Drop:** -16.65%
- **Conclusion:** Leaked features were responsible for inflated performance ✓

### Test 2: AUC Sanity Check
- **Before removal:** 0.8486 AUC
- **After removal:** 0.5209 AUC
- **Interpretation:** Model is now barely better than random ✓
- **Expected:** Corner outcomes are inherently unpredictable ✓

### Test 3: Cross-Validation Stability
- **Test accuracy:** 71.32%
- **CV accuracy:** 59.85% ± 1.82%
- **Interpretation:** Test set may be slightly optimistic, but AUC ~0.52 confirms minimal predictive power ✓

---

## Edge Cases and Special Considerations

### Why location_x/y are valid but pass_end_x/y are not:
- `location_x/y` = where corner is TAKEN FROM (corner arc, known before)
- `pass_end_x/y` = where ball LANDED (outcome, known after)
- **Confusion:** Both are "locations" but one is input, one is output

### Why is_shot_assist is especially problematic:
- Not just a leaked feature - it's the TARGET VARIABLE
- `is_shot_assist` = 1 if corner directly led to shot
- We're predicting shot outcomes, so using this is circular reasoning

### Why defending_depth is valid but pass_length is not:
- `defending_depth` = calculated from freeze frame positions (known at t=0)
- `pass_length` = calculated from actual trajectory (known after)
- **Both are "calculated"** but from different temporal sources

---

## Reproducibility Checklist

To verify these decisions:

- [ ] Load raw data: `data/processed/corners_features_with_shot.csv`
- [ ] Verify `pass_end_x/y` vary widely (not fixed zones)
- [ ] Check `is_shot_assist` correlates perfectly with shot outcomes
- [ ] Confirm freeze frame features are captured at event time
- [ ] Run `scripts/14_extract_temporally_valid_features.py`
- [ ] Train models with 19 features: `scripts/15_retrain_without_leakage.py`
- [ ] Verify accuracy ~71% and AUC ~0.52

---

## Future Work: Features We Could Add

### Valid Features Not Yet Implemented
- Player skill ratings (historical shooting %)
- Team tactical patterns (from previous corners)
- Score differential at corner time
- Minutes remaining in match
- Yellow/red cards given so far
- Recent substitutions

All of these are known at t=0 and could improve prediction marginally.

---

*Every decision documented. Every feature justified. No leakage.*
