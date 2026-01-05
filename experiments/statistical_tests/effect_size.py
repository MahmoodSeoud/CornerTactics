"""Effect size calculations.

Cohen's d is the most common measure of effect size for comparing
two groups. It expresses the difference between means in terms of
standard deviations.

Interpretation (Cohen, 1988):
- |d| < 0.2: negligible
- 0.2 <= |d| < 0.5: small
- 0.5 <= |d| < 0.8: medium
- |d| >= 0.8: large
"""

import numpy as np


def cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """Calculate Cohen's d effect size.

    Cohen's d measures the standardized difference between two means.
    A positive d indicates group_b has a higher mean than group_a.

    Uses pooled standard deviation (assumes equal variances).

    Args:
        group_a: First group of observations
        group_b: Second group of observations

    Returns:
        Cohen's d effect size (group_b - group_a) / pooled_std
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    n_a = len(group_a)
    n_b = len(group_b)

    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)

    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        # Both groups have zero variance
        if mean_a == mean_b:
            return 0.0
        else:
            # Means differ but no variance - undefined, return large effect
            return np.inf if mean_b > mean_a else -np.inf

    d = (mean_b - mean_a) / pooled_std
    return float(d)


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        String interpretation of effect size magnitude
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
