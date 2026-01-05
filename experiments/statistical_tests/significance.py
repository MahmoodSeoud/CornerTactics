"""Unified significance analysis module.

Provides high-level functions for comprehensive statistical evaluation
of model performance and model comparison.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.statistical_tests.bootstrap_ci import bootstrap_ci
from experiments.statistical_tests.effect_size import cohens_d, interpret_cohens_d
from experiments.statistical_tests.mcnemar import mcnemar_test
from experiments.statistical_tests.permutation_tests import permutation_test


def comprehensive_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'auc',
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform comprehensive statistical evaluation of model performance.

    Computes:
    - Point estimate of the metric
    - 95% confidence interval via bootstrap
    - p-value via permutation test
    - Significance determination

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        metric: Metric to compute ('auc' or 'average_precision')
        n_bootstrap: Number of bootstrap iterations
        n_permutations: Number of permutations
        alpha: Significance level (default 0.05)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with:
        - point_estimate: Observed metric value
        - ci_lower, ci_upper: 95% confidence interval
        - p_value: Significance vs random baseline
        - is_significant: Whether p < alpha
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Compute point estimate
    try:
        if metric == 'auc':
            point_estimate = roc_auc_score(y_true, y_pred)
        elif metric == 'average_precision':
            point_estimate = average_precision_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    except ValueError:
        point_estimate = 0.5 if metric == 'auc' else 0.0

    # Bootstrap confidence interval
    ci_lower, ci_upper = bootstrap_ci(
        y_true, y_pred,
        metric=metric,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    # Permutation test for significance
    p_value = permutation_test(
        y_true, y_pred,
        metric=metric,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    return {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'is_significant': p_value < alpha,
    }


def compare_models(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compare two models using McNemar's test.

    Tests whether two models have significantly different error rates
    on the same test set.

    Args:
        y_true: True binary labels
        y_pred_a: Predictions from model A (probabilities)
        y_pred_b: Predictions from model B (probabilities)
        threshold: Threshold for converting to binary predictions
        alpha: Significance level

    Returns:
        Dictionary with:
        - mcnemar_statistic: McNemar chi-squared statistic
        - mcnemar_p_value: p-value from McNemar's test
        - models_significantly_different: Whether p < alpha
        - contingency_table: 2x2 table of correct/incorrect
    """
    result = mcnemar_test(y_true, y_pred_a, y_pred_b, threshold=threshold)

    return {
        'mcnemar_statistic': result['statistic'],
        'mcnemar_p_value': result['p_value'],
        'models_significantly_different': result['p_value'] < alpha,
        'contingency_table': result['contingency_table'],
    }


def format_results(
    result: Dict[str, Any],
    metric: str = 'auc',
    model_name: str = 'Model',
) -> str:
    """Format evaluation results for thesis presentation.

    Args:
        result: Dictionary from comprehensive_evaluation
        metric: Name of the metric
        model_name: Name of the model

    Returns:
        Formatted string suitable for thesis
    """
    lines = []

    metric_name = 'AUC' if metric == 'auc' else 'Average Precision'

    point = result.get('point_estimate', 0.5)
    ci_lower = result.get('ci_lower', 0.0)
    ci_upper = result.get('ci_upper', 1.0)
    p_value = result.get('p_value', 1.0)
    is_sig = result.get('is_significant', False)

    lines.append(f"{model_name}:")
    lines.append(f"  {metric_name} = {point:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
    lines.append(f"  p-value = {p_value:.4f}")

    if is_sig:
        lines.append("  Result: Statistically significant (p < 0.05)")
        lines.append("  Interpretation: Evidence of predictive signal above random baseline")
    else:
        lines.append("  Result: Not statistically significant (p >= 0.05)")
        lines.append("  Interpretation: No evidence of predictive signal above random baseline")

    return '\n'.join(lines)
