# Data Leakage Analysis

## Purpose
Analyze 61 features for temporal leakage in shot outcome prediction from corner kicks.

## Key Insight
Features must be available at t=0 (when corner is taken), NOT after ball lands/outcome occurs.

## Feature Classification Summary

### CRITICAL LEAKAGE (8 features - MUST REMOVE)
1. `pass_outcome_id` - Directly encodes pass success/failure
2. `pass_outcome_encoded` - Re-encoded version of above
3. `has_pass_outcome` - Boolean for pass completion status
4. `is_shot_assist` - Only true if NEXT event is shot (target leakage)
5. `has_recipient` - Only labeled after ball reaches player
6. `is_aerial_won` - Aerial duel outcome during ball flight
7. `pass_recipient_id` - Only assigned after ball arrives
8. `duration` - Event duration until next event

### SUSPICIOUS (4 features - VERIFY)
1. `pass_end_x` - Could be intended OR actual landing
2. `pass_end_y` - Could be intended OR actual landing
3. `pass_length` - Depends on how computed
4. `pass_angle` - Depends on how computed

### SAFE (49 features)
All freeze-frame derived features, match state, temporal metadata, taker identity.

## Implementation Notes
- MCC > 0.7 indicates critical leakage
- MCC 0.3-0.7 indicates strong suspicion
- Cross-reference MCC with Random Forest feature importance

## Files Created
- `scripts/analyze_data_leakage.py` - Main analysis script
- `tests/test_leakage_analysis.py` - Test suite (21 tests)
- `scripts/run_leakage_analysis.sbatch` - SLURM submission
- `reports/data_leakage_report.md` - Final report

## Usage

```bash
# Run with specific data file
python scripts/analyze_data_leakage.py --data data/processed/my_features.csv --target shot_outcome

# Submit to SLURM
sbatch scripts/run_leakage_analysis.sbatch
```

The script expects:
- A DataFrame with a binary target column (shot_outcome: 0=No Shot, 1=Shot)
- Feature columns matching the 61 expected features
- Auto-detects target if named: shot_outcome, outcome, target, or label

## Note on Current Data
The existing `corners_with_features.csv` has outcome types (Ball Receipt, Clearance, etc.), not shot prediction. You need to create a dataset where:
- Target = 1 if corner resulted in shot
- Target = 0 otherwise
