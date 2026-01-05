"""Bootstrap confidence interval calculation.

Supports multiple metrics:
- AUC (Area Under ROC Curve)
- Average Precision (mAP for binary classification)
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'auc',
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities
        metric: Metric to compute ('auc' or 'average_precision')
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (default 95%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval

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
    n_samples = len(y_true)

    scores = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            if metric == 'auc':
                score = roc_auc_score(y_true_boot, y_pred_boot)
            elif metric == 'average_precision':
                score = average_precision_score(y_true_boot, y_pred_boot)
            scores.append(score)
        except ValueError:
            # Skip samples where metric computation fails
            continue

    if len(scores) == 0:
        return 0.0, 1.0

    # Compute percentile confidence interval
    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return float(lower), float(upper)
