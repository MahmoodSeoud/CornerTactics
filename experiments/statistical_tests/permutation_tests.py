"""Permutation test for significance testing.

Tests whether a model performs significantly better than random
by comparing observed performance to a null distribution generated
by permuting the labels.
"""

from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'auc',
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> float:
    """Compute p-value using permutation test.

    Tests whether the model performs significantly better than random
    by permuting labels and comparing to observed score.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities
        metric: Metric to compute ('auc' or 'average_precision')
        n_permutations: Number of permutations to generate
        random_state: Random seed for reproducibility

    Returns:
        p-value: Probability of observing this score by chance
                 (one-sided test: observed >= null)

    Raises:
        ValueError: If metric is not supported
    """
    if metric not in ('auc', 'average_precision'):
        raise ValueError(
            f"Unknown metric: '{metric}'. "
            f"Supported metrics: 'auc', 'average_precision'"
        )

    rng = np.random.RandomState(random_state)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Compute observed score
    try:
        if metric == 'auc':
            observed_score = roc_auc_score(y_true, y_pred)
        elif metric == 'average_precision':
            observed_score = average_precision_score(y_true, y_pred)
    except ValueError:
        return 1.0  # Can't compute with single class

    # Generate null distribution by permuting labels
    null_scores = []
    for _ in range(n_permutations):
        y_true_perm = rng.permutation(y_true)

        if len(np.unique(y_true_perm)) < 2:
            continue

        try:
            if metric == 'auc':
                score = roc_auc_score(y_true_perm, y_pred)
            elif metric == 'average_precision':
                score = average_precision_score(y_true_perm, y_pred)
            null_scores.append(score)
        except ValueError:
            continue

    if len(null_scores) == 0:
        return 1.0

    # Compute p-value (one-sided: observed >= null)
    # This tests if our model is significantly BETTER than random
    null_scores = np.array(null_scores)
    p_value = np.mean(null_scores >= observed_score)

    return float(p_value)
