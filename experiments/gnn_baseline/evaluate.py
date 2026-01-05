"""Evaluation module with statistical rigor.

Provides:
- Bootstrap confidence intervals
- Permutation tests for significance
- Comprehensive model evaluation
- Results formatting for thesis
"""

from typing import Dict, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


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
        y_true: True binary labels
        y_pred: Predicted probabilities
        metric: Metric to compute ('auc')
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (default 95%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        if metric == 'auc':
            score = roc_auc_score(y_true_boot, y_pred_boot)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    if len(scores) == 0:
        return 0.0, 1.0

    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'auc',
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> float:
    """Compute p-value using permutation test.

    Tests whether the model performs significantly better than random.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        metric: Metric to compute ('auc')
        n_permutations: Number of permutations
        random_state: Random seed for reproducibility

    Returns:
        p-value (probability of observing this score by chance)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Compute observed score
    try:
        if metric == 'auc':
            observed_score = roc_auc_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    except ValueError:
        return 1.0  # Can't compute AUC with single class

    # Generate null distribution by permuting labels
    null_scores = []
    for _ in range(n_permutations):
        y_true_perm = np.random.permutation(y_true)

        if len(np.unique(y_true_perm)) < 2:
            continue

        try:
            if metric == 'auc':
                score = roc_auc_score(y_true_perm, y_pred)
            null_scores.append(score)
        except ValueError:
            continue

    if len(null_scores) == 0:
        return 1.0

    # Compute p-value (one-sided: observed >= null)
    null_scores = np.array(null_scores)
    p_value = np.mean(null_scores >= observed_score)

    return float(p_value)


class Evaluator:
    """Comprehensive evaluator with statistical rigor."""

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """Initialize evaluator.

        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)

    def evaluate(
        self,
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ) -> Dict:
        """Compute comprehensive evaluation metrics.

        Args:
            n_bootstrap: Bootstrap iterations for CI
            n_permutations: Permutations for p-value
            alpha: Significance level
            random_state: Random seed

        Returns:
            Dictionary with:
                - auc: Point estimate
                - auc_ci_lower, auc_ci_upper: 95% CI
                - p_value: Significance vs random
                - is_significant: Whether p < alpha
        """
        results = {}

        # Point estimate
        try:
            results['auc'] = roc_auc_score(self.y_true, self.y_pred)
        except ValueError:
            results['auc'] = 0.5

        # Bootstrap CI
        lower, upper = bootstrap_ci(
            self.y_true, self.y_pred,
            metric='auc',
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        results['auc_ci_lower'] = lower
        results['auc_ci_upper'] = upper

        # Permutation test
        p_value = permutation_test(
            self.y_true, self.y_pred,
            metric='auc',
            n_permutations=n_permutations,
            random_state=random_state,
        )
        results['p_value'] = p_value
        results['is_significant'] = p_value < alpha

        return results


def evaluate_model(
    trainer,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
) -> Dict:
    """Evaluate trained model on test set with statistical rigor.

    Args:
        trainer: Trained Trainer instance
        n_bootstrap: Bootstrap iterations for CI
        n_permutations: Permutations for p-value

    Returns:
        Dictionary with test set evaluation results
    """
    trainer.model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in trainer.test_loader:
            batch = batch.to(trainer.device)

            if trainer.model_name == 'mpnn':
                out = trainer.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out = trainer.model(batch.x, batch.edge_index, batch.batch)

            all_preds.extend(out.squeeze().cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    evaluator = Evaluator(y_true, y_pred)
    eval_results = evaluator.evaluate(
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
    )

    # Rename keys with test_ prefix
    results = {
        'test_auc': eval_results['auc'],
        'test_auc_ci_lower': eval_results['auc_ci_lower'],
        'test_auc_ci_upper': eval_results['auc_ci_upper'],
        'test_p_value': eval_results['p_value'],
        'test_is_significant': eval_results['is_significant'],
    }

    return results


def format_results(results: Dict, model_name: str = 'Model') -> str:
    """Format results as thesis-ready string.

    Args:
        results: Evaluation results dictionary
        model_name: Name of the model

    Returns:
        Formatted string for thesis
    """
    lines = []

    # Get AUC info (handle both with and without test_ prefix)
    auc = results.get('auc', results.get('test_auc', 0.5))
    ci_lower = results.get('auc_ci_lower', results.get('test_auc_ci_lower', 0.0))
    ci_upper = results.get('auc_ci_upper', results.get('test_auc_ci_upper', 1.0))
    p_value = results.get('p_value', results.get('test_p_value', 1.0))
    is_sig = results.get('is_significant', results.get('test_is_significant', False))

    lines.append(f"{model_name}:")
    lines.append(f"  AUC = {auc:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
    lines.append(f"  p-value = {p_value:.4f}")

    if is_sig:
        lines.append("  Result: Statistically significant (p < 0.05)")
    else:
        lines.append("  Result: Not statistically significant (p >= 0.05)")

    # Interpretation
    if auc > 0.55 and is_sig:
        lines.append("  Interpretation: Evidence of predictive signal above random baseline")
    elif auc <= 0.55 or not is_sig:
        lines.append("  Interpretation: No evidence of predictive signal above random baseline")

    return '\n'.join(lines)
