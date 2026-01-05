# Statistical Rigor Module

## Purpose

Implement Task 3 from plan2.md: Add statistical rigor to all experiment results.

## Requirements from plan2.md

### 1. Confidence Intervals
- Bootstrap 95% CIs for all AUC and mAP scores
- Report whether results are statistically significantly different from random baseline

### 2. Significance Testing
- Permutation tests comparing model performance to random
- McNemar's test comparing different model predictions

### 3. Effect Size
- Cohen's d or similar for any differences found

## Current State

The GNN baseline branch has basic implementations:
- `bootstrap_ci()` - only supports AUC
- `permutation_test()` - only supports AUC
- `Evaluator` class - wraps above functions

Missing:
- mAP support in bootstrap CI
- McNemar's test for model comparison
- Cohen's d effect size
- Proper module structure under `experiments/statistical_tests/`

## Design Decisions

### Module Structure
```
experiments/
└── statistical_tests/
    ├── __init__.py          # Exports all public functions
    ├── bootstrap_ci.py      # Bootstrap confidence intervals
    ├── permutation_tests.py # Permutation-based significance tests
    ├── mcnemar.py           # McNemar's test for paired comparisons
    ├── effect_size.py       # Cohen's d and related metrics
    └── significance.py      # Unified interface
```

### API Design

```python
# Bootstrap CI - supports multiple metrics
from experiments.statistical_tests import bootstrap_ci

ci = bootstrap_ci(y_true, y_pred, metric='auc')  # Returns (lower, upper)
ci = bootstrap_ci(y_true, y_pred, metric='average_precision')

# Permutation test
from experiments.statistical_tests import permutation_test

p_value = permutation_test(y_true, y_pred, metric='auc')

# McNemar's test - compares two models
from experiments.statistical_tests import mcnemar_test

result = mcnemar_test(y_true, y_pred_model_a, y_pred_model_b, threshold=0.5)
# Returns: chi2_statistic, p_value, contingency_table

# Effect size
from experiments.statistical_tests import cohens_d

d = cohens_d(group_a_scores, group_b_scores)
```

## Test Plan

1. Test bootstrap_ci with known distributions
2. Test permutation_test returns p < 0.05 for perfect classifier
3. Test permutation_test returns p > 0.05 for random classifier
4. Test mcnemar correctly identifies when models differ
5. Test cohens_d matches expected values for known inputs

## Results Expected

After implementation, we should be able to:
1. Add statistical rigor to existing GNN results
2. Compare models (e.g., GAT vs GraphSAGE) using McNemar's test
3. Report effect sizes for any performance differences
4. Apply to classical ML results as well
