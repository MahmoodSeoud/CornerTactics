# CornerTactics Documentation

**Last Updated:** November 21, 2025

---

## Quick Start: What You Need to Know

After discovering temporal data leakage in November 2025, all previous results have been invalidated and corrected. **True performance is 71% accuracy with 22 valid features** (not 88% with leaked features).

---

## Essential Reading (Start Here)

### 1. Current Valid Results ⭐
**File:** `CURRENT_VALID_RESULTS.md`

The single source of truth for model performance:
- Valid performance: 71.06% accuracy, 0.556 AUC
- 22 temporally valid features
- What can and cannot be predicted

### 2. Why Previous Results Were Wrong
**File:** `DATA_LEAKAGE_FINDINGS.md`

Executive summary of the data leakage discovery:
- What went wrong (87% → 71% accuracy)
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
- 19 features kept (with reasoning)
- 12 features removed (with evidence)
- 3 features excluded (ambiguous)
- Edge cases explained

---

## Detailed Technical Documentation

### Feature Analysis
**File:** `TEMPORAL_DATA_LEAKAGE_ANALYSIS.md`

Comprehensive feature-by-feature temporal analysis:
- All 53 original features categorized
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

## Archived Documents

### Partially Valid (Historical Context)
**File:** `CLEAN_BASELINE_RESULTS.md` [ARCHIVED]

Intermediate analysis from November 19, 2025 that removed 7 leaked features but missed additional leakage. Preserved for historical context only.

### Paper Methods Summary
**File:** `PAPER_METHODS_AND_RESULTS.md`

Overview of methodology and results. Updated to reflect valid results but some sections may reference older intermediate work.

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
├── CLEAN_BASELINE_RESULTS.md [ARCHIVED]           # Historical intermediate work
└── PAPER_METHODS_AND_RESULTS.md                   # Methods overview
```

---

## Key Findings Summary

### The Problem
- Initial models: 87.97% accuracy (INVALID)
- Cause: 8 features contained temporal leakage
- Most egregious: `is_shot_assist` (literally the prediction target)

### The Fix
- Removed 9 leaked features systematically
- Kept 22 temporally valid features
- Retrained all models from scratch

### The Reality
- True performance: 71.06% accuracy
- AUC: 0.556 (limited predictive power)
- Conclusion: Corner outcomes are largely unpredictable from pre-kick features

---

## Scripts Reference

### Valid Scripts (Use These)
- `scripts/14_extract_temporally_valid_features.py` - Extract clean features
- `scripts/15_retrain_without_leakage.py` - Train valid models

### Dataset Files
- `data/processed/corners_features_temporal_valid.csv` - Clean dataset (22 features)
- `data/processed/temporal_valid_features.json` - Feature metadata (22 features)

---

## For Researchers

### Citing This Work
If you use this dataset or methodology, please note:
1. Only results from November 21, 2025 onward are valid
2. Any results showing >75% accuracy for corner prediction should be scrutinized for leakage
3. See `FEATURE_REMOVAL_METHODOLOGY.md` for reproducible removal process

### Common Pitfalls to Avoid
1. ❌ Using `pass_end_x/y` as "intended target" (they're actual outcomes)
2. ❌ Using `is_shot_assist` as a feature (it's the target variable)
3. ❌ Including any `has_*`, `is_*_outcome`, or `duration` features
4. ❌ Trusting feature names without validating temporal availability

---

## Questions?

### Q: Why is performance so low (71%)?
**A:** Because corner kick outcomes are inherently unpredictable from pre-kick features alone. Most of the "predictability" comes from execution (how the kick is taken), not setup (player positioning).

### Q: Can we improve beyond 71%?
**A:** Marginally. Possible approaches:
- Better feature engineering from freeze frames
- Player skill ratings (historical data)
- Team tactical patterns
- Match context (score, time)

Expect improvements of 2-5%, not 15%.

### Q: What happened to the optimal feature selection results?
**A:** Deleted. They were based on leaked features and are completely invalid.

### Q: Why does the documentation mention different feature counts (19, 22, 36)?

**Answer: Use 22 features only. This is the complete, corrected valid feature set.**

- **22 features** = Correct, complete temporally valid features (71% accuracy)
- **19 features** = OLD - was missing `attacking_team_goals`, `attacking_to_goal_dist`, `keeper_distance_to_goal`
- **36 features** = INVALID, includes leaked features like `is_cross_field_switch`

Always use the dataset at `data/processed/corners_features_temporal_valid.csv` with 22 features.

---

*Last updated: November 21, 2025*
*Previous results invalidated due to temporal data leakage*