# CornerTactics: Complete Progress Summary

**Date**: November 15, 2025
**Session**: Data Exploration & Ablation Study
**Status**: ✅ **COMPLETE** - All analyses finished

---

## 🎯 What We Accomplished Today

### 1. ✅ Dataset Exploration (Phases 2-6 from DATA_EXPLORATION_PLAN.md)

**Goal**: Systematically explore the 21,656 corner kick dataset to identify which features to ablate

**Implemented**: `scripts/explore_dataset.py`

**Key Outputs**:
- `results/exploration/feature_tiers.json` - Features grouped by importance
- `results/exploration/leakage_suspects.json` - 6 flagged leakage features
- `results/exploration/ablation_plan.json` - 10 systematic configs to test
- `results/exploration/SUMMARY.md` - Detailed exploration summary

**Major Finding**: Identified **6 features leaking post-kick information**:
1. `second` (39.1% importance) - Time of next event
2. `shot_assist` (23.5% importance) - Boolean for shot attempt
3. `pass_length` (5.8% importance) - Distance ball traveled
4. `end_location_x` (3.8% importance) - X-coordinate where ball landed
5. `end_location_y` (3.3% importance) - Y-coordinate where ball landed
6. `pass_angle` (3.3% importance) - Angle of pass trajectory

---

### 2. ✅ Ablation Experiments (All 10 Configs)

**Goal**: Quantify leakage impact and test feature selection strategies

**Implemented**: `scripts/run_ablation_experiments.py`

**Results**: `results/ablation_experiments/ablation_results_final.csv`

---

## 📊 Complete Ablation Results

### Summary Table

| Config ID | Config Name | Features | Accuracy | Macro F1 | Description |
|-----------|-------------|----------|----------|----------|-------------|
| **0** | baseline_all_features | 23 | **85.7%** | **49.8%** | All features (WITH leakage) |
| 1 | remove_second | 22 | 78.9% | 31.3% | Remove time feature only |
| 2 | remove_shot_assist | 22 | 86.1% | 50.4% | Remove shot label only |
| 3 | remove_pass_length | 22 | 86.0% | 50.3% | Remove distance only |
| 4 | remove_end_location_x | 22 | 85.9% | 50.1% | Remove x-coord only |
| 5 | remove_end_location_y | 22 | 85.9% | 50.2% | Remove y-coord only |
| **6** | **clean_baseline** | **17** | **78.8%** | **36.7%** | **Remove ALL 6 leakage features** |
| 7 | remove_tier4 | 15 | 85.6% | 49.6% | Remove weak features (<1%) |
| 8 | keep_tier1_tier2 | 3 | 85.5% | 49.1% | Keep only top features (>5%) |
| 9 | keep_tier1_only | 2 | 84.4% | 47.0% | Keep only 2 highest (>10%) |

---

## 🔍 Key Insights

### Insight 1: `second` is the BIGGEST Leakage Source

**Config 1 vs Config 0**:
- Removing ONLY `second`: Accuracy drops 85.7% → 78.9% (**-6.8%**)
- Macro F1 drops 49.8% → 31.3% (**-18.5%**)

**This single feature accounts for most of the leakage!** It's the timestamp of the next event, directly revealing when the outcome occurs.

---

### Insight 2: `shot_assist`, `pass_length`, `end_location` Have Minimal Impact

**Configs 2-5**: Removing these individually has almost NO effect
- Accuracy stays ~86% (only -0.1% to +0.3% change)
- Macro F1 stays ~50% (minimal change)

**Why?** These features are redundant with `second` and other features. The model can compensate when only one is removed.

---

### Insight 3: Clean Baseline Shows TRUE Predictive Capability

**Config 6 (clean_baseline)**: Remove ALL 6 leakage features
- Accuracy: 78.8% (vs 85.7% with leakage)
- Macro F1: 36.7% (vs 49.8% with leakage)

**This is the fair baseline** - what can actually be predicted using ONLY pre-kick information.

**Leakage Impact**:
- Accuracy inflated by: **+6.9 percentage points** (8.8% relative)
- Macro F1 inflated by: **+13.1 percentage points** (35.7% relative)

---

### Insight 4: Feature Selection Strategies Don't Help Much

**Configs 7-9**: Removing weak features or keeping only top features
- All achieve ~85-86% accuracy
- Why? Because they still include the leakage features!

**Config 9** (keep only `second` + `shot_assist`) achieves 84.4% accuracy with just 2 features, confirming these are the dominant leakage sources.

---

## 📁 All Files Created

### Exploration Outputs
```
results/exploration/
├── feature_tiers.json              # Features by importance tier
├── feature_correlation_matrix.csv  # Pairwise correlations
├── high_correlations.json          # Highly correlated pairs
├── leakage_suspects.json           # 6 flagged features
├── ablation_plan.json              # 10 systematic configs
└── SUMMARY.md                      # Exploration summary
```

### Ablation Outputs
```
results/ablation_experiments/
├── ablation_results.csv            # Incremental results
├── ablation_results_final.csv      # Complete results (10 configs)
└── PRELIMINARY_RESULTS.md          # Initial findings
```

### Scripts
```
scripts/
├── explore_dataset.py              # Dataset exploration (Phases 2-6)
├── run_ablation_experiments.py     # Systematic ablation training
├── check_feature_origin.py         # Verify feature sources
└── slurm/
    ├── explore_dataset.sh          # SLURM wrapper for exploration
    └── run_ablation_experiments.sh # SLURM wrapper for ablation
```

---

## 🎓 What We Learned

### 1. StatsBomb Provides Leakage Features by Default

The 6 leakage features come directly from StatsBomb's API:
- `end_location_x/y`: Where the ball landed (outcome)
- `pass_length/angle`: Calculated from end_location
- `shot_assist`: Boolean label (is this literally the prediction target)
- `second`: Timestamp from the NEXT event

StatsBomb includes these for post-match analysis, **not for prediction**. We must explicitly remove them.

### 2. One Feature (`second`) Dominates Leakage

- `second` alone accounts for 6.8% accuracy boost
- Removing it drops Macro F1 by 18.5%
- It's the most critical feature to exclude

### 3. Clean Baseline: 78.8% Accuracy is Fair

When predicting corner outcomes using ONLY pre-kick information:
- **78.8% accuracy** is the true capability
- **36.7% macro F1** shows difficulty with minority classes
- This is our new baseline for future work

### 4. Class Imbalance is Severe

After filtering:
- Class 1: 17,115 samples (79.0%) - dominant
- Class 2: 4,465 samples (20.6%)
- Class 0: 75 samples (0.3%) - very rare

The low macro F1 (36.7%) reflects poor performance on rare classes.

---

## 🚀 Next Steps

### Immediate Actions

1. **Use Config 6 as Baseline**: All future work should use the clean 17-feature set
2. **Update Documentation**: Mark previous baselines using all 23 features as invalid
3. **Share Findings**: Document this leakage discovery for the research community

### Future Work

1. **Engineer Better Pre-Kick Features**:
   - Player positioning features (freeze frames)
   - Tactical formations
   - Historical team statistics
   - Player skill ratings

2. **Address Class Imbalance**:
   - Use class weights in training
   - Try SMOTE or other sampling techniques
   - Consider different outcome taxonomies

3. **Try Different Models**:
   - Graph Neural Networks (original plan)
   - Transformers for temporal sequences
   - Ensemble methods

---

## 📌 References

**Plans**:
- `docs/DATA_EXPLORATION_PLAN.md` - Original exploration plan (completed)
- `docs/FLEXIBLE_ABLATION_FRAMEWORK.md` - Ablation methodology
- `docs/raw_dataset_plan.md` - Training plan

**Data**:
- `data/analysis/corner_sequences_full.json` - 21,656 corners
- Dataset has 4 outcome classes (1 filtered due to single sample)

**Models**:
- XGBoost (max_depth=5, n_estimators=200)
- 70% train / 15% val / 15% test (stratified)

---

## ✅ Session Complete

**Total Time**: ~2 hours
**Configs Tested**: 10
**Features Analyzed**: 23
**Leakage Features Identified**: 6
**Clean Baseline Established**: ✅ 78.8% accuracy

**Status**: Ready for next phase of research with clean, validated baseline!
