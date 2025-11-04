# Multi-Class Outcome Baselines Feature Notes

## Goal
Implement Day 6.5 from TACTICAI_IMPLEMENTATION_PLAN.md: Multi-Class Outcome Baselines

## Requirements
- Add multi-class outcome prediction (4-class: Goal/Shot/Clearance/Possession)
- Extend receiver_data_loader.py to add outcome_class_label field
- Extend baselines.py with 3 new outcome classifiers:
  - RandomOutcomeBaseline: uniform distribution over 4 classes (25% accuracy expected)
  - XGBoostOutcomeBaseline: 50-60% accuracy expected
  - MLPOutcomeBaseline: 55-65% accuracy expected
- Create train_outcome_baselines.py training script
- Metrics: Accuracy, Macro F1, Per-class F1, Confusion Matrix

## Class Mapping
Actual distribution from data (5,814 graphs with outcome labels):
- Clearance: 3,021 (52.0%)
- Goal: 74 (1.3%)
- Loss: 1,128 (19.4%)
- Possession: 609 (10.5%)
- Shot: 982 (16.9%)

**Mapping to 4 classes** (merge Loss into Possession):
- 0: Goal (1.3% - rare)
- 1: Shot (16.9% - minority)
- 2: Clearance (52.0% - common)
- 3: Possession (29.9% - Loss + Possession merged)

## Success Criteria
- Random baseline: 25% accuracy (uniform)
- XGBoost baseline: 50-60% accuracy, Macro F1 > 0.45
- MLP baseline: 55-65% accuracy, Macro F1 > 0.50

## Implementation Notes
- Following TDD approach (Red-Green-Refactor)
- Need to understand existing receiver_data_loader.py and baselines.py first
- Outcome mapping from existing outcome_category field
