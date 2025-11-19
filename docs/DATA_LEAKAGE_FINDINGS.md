# Data Leakage Analysis Findings

## Summary

Analysis of 43 features for temporal leakage in binary shot prediction from corner kicks (1,933 samples: 422 shots, 1,511 no shots).

## Critical Findings

### Features to Remove Immediately

| Feature | MCC | Why It's Leaked |
|---------|-----|-----------------|
| `is_shot_assist` | 0.649 | Literally encodes whether next event is a shot - only known AFTER outcome |
| `has_recipient` | 0.272 | Only labeled after pass completes/fails |

### Detailed Analysis: `is_shot_assist`

Despite high MCC (0.649), only adds ~3% accuracy because:

| is_shot_assist | No Shot | Shot | Total |
|----------------|---------|------|-------|
| 0 | 1,425 | 137 | 1,562 (81%) |
| 1 | 86 | 285 | 371 (19%) |

- **Coverage**: Only 19% of samples have `is_shot_assist=1`
- **Precision**: 76.8% of shot assists lead to shots (not perfect)
- **Recall**: Only 67.5% of shots have `is_shot_assist=1`

**Key insight**: Leakage ≠ high accuracy. The feature is still invalid because information isn't available at t=0.

## Features CONFIRMED as Leaked

| Feature | MCC | Evidence |
|---------|-----|---------|
| `pass_end_x` | -0.137 | **LEAKED** - Actual landing location (matches next event location) |
| `pass_end_y` | ~0 | **LEAKED** - Actual landing location (matches next event location) |
| `pass_length` | 0.126 | **LEAKED** - Computed from actual landing |
| `pass_angle` | -0.085 | **LEAKED** - Computed from actual landing |

**Verified via StatsBomb event analysis**:
- Corner with outcome="Incomplete": pass.end_location doesn't match next event
- Corner with recipient: pass.end_location EXACTLY matches Ball Receipt location
- This proves pass_end is the actual outcome location, not intended target

## Safe Features (MCC < 0.1)

All 40 freeze-frame and match-state features are legitimate:

### Freeze-Frame Derived (t=0)
- `attacking_in_box`, `defending_in_box`
- `attacking_near_goal`, `defending_near_goal`
- `attacking_density`, `defending_density`
- `numerical_advantage`, `attacker_defender_ratio`
- `attacking_centroid_x/y`, `defending_centroid_x/y`
- `defending_compactness`, `defending_depth`
- `attacking_to_goal_dist`, `defending_to_goal_dist`
- `keeper_distance_to_goal`

### Pass Intent (t=0)
- `is_inswinging`, `is_outswinging`
- `is_cross_field_switch`

### Match State (t=0)
- `score_difference`, `match_situation`
- `attacking_team_goals`, `defending_team_goals`
- `minute`, `second`, `period`

### Other
- `corner_side`, `total_subs_before`, `recent_subs_5min`

## Expected Impact After Removing Leaked Features

- **Accuracy drop**: Expect ~3% decrease (from `is_shot_assist` removal)
- **Baseline performance**: MCC ~0.1-0.15 is realistic for legitimate features
- **Model validity**: Results will reflect true predictive signal from t=0 information

## Recommendations

### Immediate Actions
1. Remove `is_shot_assist` from feature set
2. Remove `has_recipient` from feature set
3. Verify `pass_end_x/y` represents intended target (check StatsBomb docs)

### For Paper
1. Clearly state which features are available at t=0
2. Document why certain features were excluded
3. Report honest baseline performance without leaked features

### Alternative Features to Engineer
Since leaked features must be removed, consider:
1. Historical corner success rate (team/player)
2. Defensive formation patterns from freeze-frame
3. Spatial entropy of player positions
4. Distance-based interaction features

## Files Generated

- `reports/data_leakage_report.md` - Full report
- `reports/leakage_analysis_results.json` - Raw metrics
- `reports/figures/leakage_heatmap.png` - All features vs metrics
- `reports/figures/mcc_vs_importance.png` - Scatter plot
- `reports/figures/confusion_matrices.png` - Top suspicious features
- `reports/figures/timeline_diagram.png` - Temporal availability

## Data Files

- `data/processed/corners_features_with_shot.csv` - Merged features + shot labels (use this)
- `data/processed/corner_labels.csv` - Shot labels only

---
*Generated: 2025-11-19*
