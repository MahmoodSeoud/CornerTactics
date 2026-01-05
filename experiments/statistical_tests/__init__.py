"""Statistical tests module for rigorous evaluation.

Provides:
- Bootstrap confidence intervals for AUC and mAP
- Permutation tests for significance testing
- McNemar's test for model comparison
- Cohen's d effect size calculation
"""

from experiments.statistical_tests.bootstrap_ci import bootstrap_ci
from experiments.statistical_tests.permutation_tests import permutation_test
from experiments.statistical_tests.mcnemar import mcnemar_test
from experiments.statistical_tests.effect_size import cohens_d
from experiments.statistical_tests.significance import (
    comprehensive_evaluation,
    compare_models,
    format_results,
)

__all__ = [
    'bootstrap_ci',
    'permutation_test',
    'mcnemar_test',
    'cohens_d',
    'comprehensive_evaluation',
    'compare_models',
    'format_results',
]
