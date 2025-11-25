# CornerTactics Documentation

**Last Updated:** November 25, 2025

---

## Quick Start: What You Need to Know

After discovering temporal data leakage and fixing methodology issues, all previous results have been invalidated and corrected.

**Key Facts:**
- **Split**: 60/20/20 (train/val/test) with match-based stratification
- **Features**: 22 temporally valid features
- **Binary shot prediction**: ~70% accuracy (baseline: 72%) - no predictive power
- **Multi-class outcome**: ~51% accuracy (baseline: 53%) - no predictive power
- **Conclusion**: Corner kick outcomes are essentially unpredictable from pre-kick features

---

## Essential Reading (Start Here)

### 1. Current Valid Results
**File:** `CURRENT_VALID_RESULTS.md`

The single source of truth for model performance:
- Proper 60/20/20 match-based splits
- Binary shot prediction results
- Multi-class outcome prediction results (4 classes)
- 22 temporally valid features
- What can and cannot be predicted

### 2. Why Previous Results Were Wrong
**File:** `DATA_LEAKAGE_FINDINGS.md`

Executive summary of the data leakage discovery:
- What went wrong (87% → 70% accuracy)
- Which features were leaked
- Empirical evidence
- Lessons learned

### 3. How We Fixed It
**File:** `FEATURE_REMOVAL_METHODOLOGY.md`

Systematic process for identifying and removing leaked features:
- Classification methodology (BEFORE/AFTER)
- Evidence collection process
- Feature-by-feature decisions
- Validation of results

### 4. Every Decision Documented
**File:** `DECISION_LOG.md`

Complete record of every feature inclusion/exclusion decision:
- 22 features kept (with reasoning)
- Features removed (with evidence)
- Edge cases explained

---

## Current Results Summary

### Binary Shot Prediction

| Model | Train | Val | Test | Test AUC |
|-------|-------|-----|------|----------|
| MLP | 70.48% | 72.78% | 70.52% | 0.4324 |
| XGBoost | 99.39% | 63.61% | 60.44% | 0.5095 |
| Random Forest | 92.29% | 66.85% | 59.95% | 0.4526 |
| **Baseline** | - | - | **71.74%** | 0.5000 |

**All models at or below baseline. AUC ~0.5 = no predictive power.**

### Multi-Class Outcome Prediction

Classes: Ball Receipt (53%), Clearance (24%), Goalkeeper (11%), Other (12%)

| Model | Train | Val | Test | Test F1 |
|-------|-------|-----|------|---------|
| MLP | 57.49% | 53.37% | 50.86% | 0.1734 |
| XGBoost | 98.53% | 49.06% | 49.39% | 0.2237 |
| Random Forest | 88.48% | 37.74% | 43.00% | 0.2819 |
| **Baseline** | - | - | **53.07%** | - |

**All models at or below baseline. Models just predict majority class.**

---

## Detailed Technical Documentation

### Feature Analysis
**File:** `TEMPORAL_DATA_LEAKAGE_ANALYSIS.md`

Comprehensive feature-by-feature temporal analysis:
- All original features categorized
- Detailed reasoning for each classification
- Valid vs leaked vs ambiguous

### Data Guide
**File:** `STATSBOMB_DATA_GUIDE.md`

Technical reference for StatsBomb data structure:
- Event data format
- Freeze frame (360°) data
- Coordinate system
- Corner kick definitions

---

## File Organization Summary

```
docs/
├── README.md (this file)                           # Documentation index
│
├── CURRENT_VALID_RESULTS.md                        # ⭐ START HERE - Valid results
├── DATA_LEAKAGE_FINDINGS.md                        # ⭐ Why results changed
├── FEATURE_REMOVAL_METHODOLOGY.md                  # ⭐ How we fixed it
├── DECISION_LOG.md                                 # ⭐ Every feature decision
│
├── TEMPORAL_DATA_LEAKAGE_ANALYSIS.md              # Detailed feature analysis
├── STATSBOMB_DATA_GUIDE.md                        # Technical data reference
│
├── CLEAN_BASELINE_RESULTS.md [DEPRECATED]         # Historical - now invalid
├── PAPER_METHODS_AND_RESULTS.md                   # Methods overview (needs update)
├── SHOT_LABEL_VERIFICATION.md                     # Shot label validation
└── VERIFICATION_SUMMARY.md                        # Verification notes
```

---

## Key Findings Summary

### The Problems Fixed
1. **Data Leakage**: 8+ features contained information about the outcome
2. **Improper Splits**: Was using 80/20 instead of 60/20/20
3. **No Validation Set**: Model selection done on test set
4. **No Match-Based Splits**: Same match corners in train and test
5. **Single Task Only**: Only binary prediction, no multi-class

### The Fixes Applied
1. Removed all temporally leaked features (kept 22)
2. Implemented 60/20/20 match-based stratified splits
3. Added proper validation set for model selection
4. Ensured no match overlap between sets
5. Added multi-class outcome prediction task

### The Reality
- **Binary shot prediction**: 70.52% accuracy (baseline: 71.74%)
- **Multi-class outcome**: 50.86% accuracy (baseline: 53.07%)
- **AUC values**: 0.43-0.51 (random = 0.50)
- **Conclusion**: Corner outcomes are unpredictable from pre-kick features

---

## Scripts Reference

### Valid Scripts (Use These)
- `scripts/14_extract_temporally_valid_features.py` - Extract clean features
- `scripts/15_retrain_without_leakage.py` - Train valid models (both tasks)

### Dataset Files
- `data/processed/corners_features_temporal_valid.csv` - Clean dataset (22 features)
- `data/processed/temporal_valid_features.json` - Feature metadata
- `data/processed/train_indices.csv` - Training set indices (60%)
- `data/processed/val_indices.csv` - Validation set indices (20%)
- `data/processed/test_indices.csv` - Test set indices (20%)

### Results
- `results/no_leakage/training_results.json` - Full results JSON

---

## For Researchers

### Citing This Work
If you use this dataset or methodology, please note:
1. Only results from November 25, 2025 onward are valid
2. Any results showing >75% accuracy for corner prediction should be scrutinized for leakage
3. Use match-based splits to prevent leakage
4. Report both binary and multi-class results

### Common Pitfalls to Avoid
1. ❌ Using `pass_end_x/y` as "intended target" (they're actual outcomes)
2. ❌ Using `is_shot_assist` as a feature (it's the target variable)
3. ❌ Including any `has_*`, `is_*_outcome`, or `duration` features
4. ❌ Trusting feature names without validating temporal availability
5. ❌ Using random splits instead of match-based splits
6. ❌ Evaluating on test set without a separate validation set

---

## Questions?

### Q: Why is performance so low (~70%)?
**A:** Because corner kick outcomes are inherently unpredictable from pre-kick features alone. The models perform at or below the baseline of always predicting the majority class.

### Q: Can we improve beyond baseline?
**A:** Unlikely with current features. Possible approaches:
- Video/trajectory data (post-kick information)
- Player skill ratings (external data)
- Team tactical patterns (historical data)

Don't expect significant improvements from pre-kick positioning alone.

### Q: What's the difference between binary and multi-class tasks?
**A:**
- **Binary**: Did the corner lead to a shot? (Yes/No)
- **Multi-class**: What happened immediately after? (Ball Receipt, Clearance, Goalkeeper, Other)

Both tasks show no predictive power above baseline.

### Q: Why use 60/20/20 instead of 80/20?
**A:** The validation set is essential for:
- Hyperparameter tuning
- Model selection
- Early stopping
- Preventing overfitting to test set

---

*Last updated: November 25, 2025*
*Previous results invalidated due to temporal data leakage and methodology issues*
