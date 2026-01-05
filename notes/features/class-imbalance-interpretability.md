# Feature: Class Imbalance Handling & Spatial Attention Analysis

## Overview

Implementation of Tasks 4 and 5 from plan2.md:
- Task 4: Improved Class Imbalance Handling
- Task 5: Spatial Attention Analysis

## Task 4: Improved Class Imbalance Handling

### Goals
- Implement Focal Loss instead of weighted cross-entropy
- Implement SMOTE/ADASYN on graph embeddings for minority classes
- Implement Hierarchical Classification (SHOT vs NO_SHOT vs PROCEDURAL)
- Class Grouping Experiments (merge rare classes)

### Class Distribution (from plan2.md)
- NOT_DANGEROUS: 40.1%
- CLEARED: 23.5%
- SHOT_OFF_TARGET: 14.7%
- SHOT_ON_TARGET: 8.0%
- FOUL: 7.9%
- GOAL: 3.6%
- OFFSIDE: 1.6%
- CORNER_WON: 0.5%

### Implementation Details

#### Focal Loss
- Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
- Default: gamma=2.0, alpha=balanced class weights
- Down-weights well-classified examples, focuses on hard examples

#### SMOTE/ADASYN on Embeddings
- Extract graph-level embeddings from trained GNN
- Apply SMOTE/ADASYN to oversample minority class embeddings
- Train classifier on augmented embedding space

#### Hierarchical Classification
- Level 1: SHOT vs NO_SHOT vs PROCEDURAL
  - SHOT: GOAL, SHOT_ON_TARGET, SHOT_OFF_TARGET
  - NO_SHOT: NOT_DANGEROUS, CLEARED
  - PROCEDURAL: FOUL, OFFSIDE, CORNER_WON
- Level 2: Within-group predictions

#### Class Grouping
- Binary: SHOT vs NO_SHOT (merge into 2 classes)
- Ternary: SHOT vs NO_SHOT vs PROCEDURAL (3 classes)
- Compare with 8-class original

## Task 5: Spatial Attention Analysis

### Goals
- Attention Visualization for GAT models
- SHAP/Integrated Gradients for feature importance

### Implementation Details

#### Attention Visualization
- Extract attention weights from GATConv layers
- Visualize which player relationships model attends to
- Compare attention patterns: shot vs no-shot corners

#### SHAP/Integrated Gradients
- Use captum library for integrated gradients
- Node-level feature importance
- Edge-level importance (for MPNN)

## Code Structure
```
experiments/
├── class_imbalance/
│   ├── __init__.py
│   ├── focal_loss.py        (FocalLoss, FocalLossMulticlass)
│   ├── oversampling.py      (EmbeddingExtractor, SMOTEOversampler, ADASYNOversampler)
│   ├── hierarchical_classifier.py (HierarchicalClassifier, HierarchicalLoss, HierarchicalTrainer)
│   └── class_grouping.py    (ClassGroupingExperiment, run_all_grouping_experiments)
└── interpretability/
    ├── __init__.py
    ├── attention_viz.py     (AttentionExtractor, plot_attention_on_pitch)
    └── shap_analysis.py     (IntegratedGradientsExplainer, analyze_feature_importance)
```

## Test Structure
```
tests/
├── class_imbalance/
│   ├── __init__.py
│   ├── test_focal_loss.py            (16 tests)
│   ├── test_oversampling.py          (11 tests)
│   ├── test_hierarchical_classifier.py (15 tests)
│   └── test_class_grouping.py        (13 tests)
└── interpretability/
    ├── __init__.py
    ├── test_attention_viz.py         (8 tests)
    └── test_shap_analysis.py         (10 tests)
```

**Total: 73 tests passing**

## Progress Log
- [2026-01-05] Created feature branch and notes file
- [2026-01-05] Implemented Focal Loss with TDD (16 tests)
- [2026-01-05] Implemented SMOTE/ADASYN oversampling (11 tests)
- [2026-01-05] Implemented Hierarchical Classification (15 tests)
- [2026-01-05] Implemented Class Grouping Experiments (13 tests)
- [2026-01-05] Implemented Attention Visualization (8 tests)
- [2026-01-05] Implemented SHAP/Integrated Gradients (10 tests)
- [2026-01-05] All 73 tests passing, ready for PR
