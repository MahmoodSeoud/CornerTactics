# DEPRECATED - Clean Baseline Results

> **This document is deprecated. See [CURRENT_VALID_RESULTS.md](CURRENT_VALID_RESULTS.md) for valid results.**

---

## Why This Document is Invalid

This document contained results that were affected by:

1. **Data Leakage**: Features like `is_cross_field_switch`, `is_outswinging`, etc. contained information about the outcome
2. **Improper Splits**: Used 80/20 train/test split instead of proper 60/20/20 with validation
3. **No Match-Based Splitting**: Same match corners could appear in both train and test

---

## Current Valid Results (November 25, 2025)

### Experimental Setup
- **Split**: 60/20/20 (train/val/test) with match-based stratification
- **Features**: 22 temporally valid features only
- **Tasks**: Binary shot prediction + Multi-class outcome prediction

### Binary Shot Prediction

| Model | Test Acc | Test AUC | Baseline |
|-------|----------|----------|----------|
| MLP | 70.52% | 0.4324 | 71.74% |
| XGBoost | 60.44% | 0.5095 | 71.74% |
| Random Forest | 59.95% | 0.4526 | 71.74% |

**All models perform at or below baseline.**

### Multi-Class Outcome Prediction (4 classes)

| Model | Test Acc | Test F1 | Baseline |
|-------|----------|---------|----------|
| MLP | 50.86% | 0.1734 | 53.07% |
| XGBoost | 49.39% | 0.2237 | 53.07% |
| Random Forest | 43.00% | 0.2819 | 53.07% |

**All models perform at or below baseline.**

---

## Conclusion

Corner kick outcomes are **essentially unpredictable** using only pre-kick information. The AUC values around 0.45-0.51 indicate no better than random guessing.

See [CURRENT_VALID_RESULTS.md](CURRENT_VALID_RESULTS.md) for full details.

---

*Updated: November 25, 2025*
