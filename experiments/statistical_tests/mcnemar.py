"""McNemar's test for comparing two classifiers.

McNemar's test compares paired predictions from two classifiers
to determine if they have significantly different error rates.
It focuses on cases where the classifiers disagree.
"""

from typing import Dict, Optional, Union

import numpy as np
from scipy import stats


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """Perform McNemar's test to compare two classifiers.

    McNemar's test examines the 2x2 contingency table of:
    - Both correct
    - A correct, B wrong
    - A wrong, B correct
    - Both wrong

    It tests whether the off-diagonal elements (discordant pairs)
    differ significantly, indicating models have different error patterns.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_a: Predictions from model A (binary or probabilities)
        y_pred_b: Predictions from model B (binary or probabilities)
        threshold: Threshold for converting probabilities to binary.
                   If None, predictions are assumed to already be binary.

    Returns:
        Dictionary with:
        - 'statistic': McNemar chi-squared statistic
        - 'p_value': Two-sided p-value
        - 'contingency_table': 2x2 contingency table
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    # Convert probabilities to binary predictions if threshold given
    if threshold is not None:
        y_pred_a = (y_pred_a >= threshold).astype(int)
        y_pred_b = (y_pred_b >= threshold).astype(int)
    else:
        # Ensure binary
        y_pred_a = y_pred_a.astype(int)
        y_pred_b = y_pred_b.astype(int)

    # Compute correctness
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)

    # Build contingency table
    # Rows: A correct (1) vs wrong (0)
    # Cols: B correct (1) vs wrong (0)
    both_correct = np.sum((correct_a == 1) & (correct_b == 1))
    a_correct_b_wrong = np.sum((correct_a == 1) & (correct_b == 0))
    a_wrong_b_correct = np.sum((correct_a == 0) & (correct_b == 1))
    both_wrong = np.sum((correct_a == 0) & (correct_b == 0))

    contingency_table = np.array([
        [both_correct, a_correct_b_wrong],
        [a_wrong_b_correct, both_wrong]
    ])

    # Get the discordant counts
    b = a_correct_b_wrong  # A right, B wrong
    c = a_wrong_b_correct  # A wrong, B right

    # McNemar's test with continuity correction
    if b + c == 0:
        # No discordant pairs - models make identical predictions
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'contingency_table': contingency_table
        }

    # Use exact binomial test for small samples
    if b + c < 25:
        # Exact binomial test (using newer API)
        result = stats.binomtest(b, b + c, 0.5, alternative='two-sided')
        p_value = result.pvalue
        statistic = (b - c) ** 2 / (b + c)
    else:
        # Chi-squared with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'contingency_table': contingency_table
    }
