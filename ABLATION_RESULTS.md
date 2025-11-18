# Ablation Study Results Summary

**Date:** November 18, 2025
**Study:** Progressive Feature Addition for Corner Kick Prediction

---

## Quick Results

### Best Model Performance
- **Binary Shot Prediction:** 86.94% (Step 5, Random Forest)
- **4-Class Outcome:** 80.76% (Step 0/3/7/9, Random Forest)

### Key Finding
**Raw StatsBomb features are already highly predictive.** Feature engineering provides modest gains for shot prediction (+4.81%) but minimal benefit for outcome classification.

---

## Progressive Results (10 Steps, 27→61 Features)

| Step | Feature Group | Total Features | Shot Accuracy | Δ from Baseline |
|------|--------------|----------------|---------------|-----------------|
| 0 | **Baseline (Raw)** | 27 | 82.13% | - |
| 1 | + Player Counts | 31 | 83.85% | +1.72% |
| 2 | + Spatial Density | 35 | 83.16% | +1.03% |
| 3 | + Positional | 43 | 83.85% | +1.72% |
| 4 | + Pass Technique | 45 | 83.16% | +1.03% |
| **5** | **+ Pass Outcome** ⭐ | **49** | **86.94%** | **+4.81%** |
| 6 | + Goalkeeper | 52 | 86.25% | +4.12% |
| 7 | + Score State | 56 | 86.25% | +4.12% |
| 8 | + Substitutions | 59 | 86.60% | +4.47% |
| 9 | + Metadata | 61 | 86.25% | +4.12% |

---

## Most Important Features

### Top 5 Predictors (by correlation)

1. **is_shot_assist** (0.649) - Corner directly assisted a shot
2. **has_recipient** (-0.709) - Pass reached a teammate
3. **has_pass_outcome** (0.447) - Pass was unsuccessful
4. **pass_outcome_encoded** (0.437) - Type of pass failure
5. **pass_end_x** (0.221) - Ball landing position

---

## Recommendations

### For Deployment
**Use Step 5 features (49 total):**
- Best shot prediction performance (86.94%)
- Good balance of accuracy vs complexity
- Excludes low-value features (goalkeeper position, score state, substitutions)

### For Research
- **4-class outcome prediction** doesn't benefit from engineering - use raw features
- **Pass outcome features** are the most valuable engineered features
- **Tactical context** (score, subs) surprisingly adds minimal value

---

## Documentation

- **Detailed Implementation:** `docs/ABLATION_STUDY_IMPLEMENTATION.md`
- **Experimental Plan:** `docs/ABLATION_STUDY_PLAN.md`
- **All Analysis Plots:** `results/ablation/analysis/`
- **Raw Results:** `results/ablation/all_results.json`

---

## Methodology

**Models Trained:** 60 total (10 steps × 3 models × 2 tasks)
- Random Forest (n=100 trees, depth=20)
- XGBoost (n=100 trees, depth=6)
- MLP (4 hidden layers: 512-256-128-64)

**Tasks:**
1. 4-Class Outcome: Ball Receipt / Clearance / Goalkeeper / Other
2. Binary Shot: Led to shot attempt (Yes/No)

**Data Split:** Match-based 70/15/15 (train/val/test)
**Dataset:** 1,933 corners with 360° freeze frames from StatsBomb

---

## Citation

If using this methodology or results:

```
CornerTactics Ablation Study (2025)
Progressive Feature Engineering for Corner Kick Outcome Prediction
Dataset: StatsBomb Open Data (1,933 corners, 360° freeze frames)
```
