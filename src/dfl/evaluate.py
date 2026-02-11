"""Evaluation utilities for corner kick ST-GNN.

This module provides:
- AUC and F1 metric computation
- Cross-validation result aggregation
- Statistical significance testing (paired t-test)
- Ablation experiment analysis
- Results formatting for reporting
"""

from typing import List, Dict, Any
import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score, f1_score


def compute_auc(y_true: List[int], y_pred: List[float]) -> float:
    """Compute Area Under the ROC Curve (AUC).

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities

    Returns:
        AUC score as float between 0 and 1
    """
    return float(roc_auc_score(y_true, y_pred))


def compute_f1(y_true: List[int], y_pred: List[int]) -> float:
    """Compute F1 score for binary classification.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels (not probabilities)

    Returns:
        F1 score as float between 0 and 1
    """
    return float(f1_score(y_true, y_pred))


def aggregate_fold_results(
    fold_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate cross-validation results across folds.

    Computes AUC for each fold and returns mean/std statistics.

    Args:
        fold_results: List of dicts with 'predictions' and 'labels' keys

    Returns:
        Dict with:
            - fold_aucs: List of AUC scores per fold
            - mean_auc: Mean AUC across folds
            - std_auc: Standard deviation of AUC across folds
    """
    fold_aucs = []

    for fold in fold_results:
        preds = fold["predictions"]
        labels = fold["labels"]

        # Skip folds with only one class
        if len(set(labels)) < 2:
            continue

        auc = compute_auc(labels, preds)
        fold_aucs.append(auc)

    if not fold_aucs:
        return {
            "fold_aucs": [],
            "mean_auc": 0.0,
            "std_auc": 0.0,
        }

    return {
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
    }


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, float]:
    """Perform paired t-test between two sets of scores.

    Used to test if the difference between two conditions is statistically
    significant. Assumes paired samples (e.g., same folds, different features).

    Args:
        scores_a: Scores from condition A
        scores_b: Scores from condition B (same length as scores_a)

    Returns:
        Dict with:
            - t_stat: T-statistic
            - p_value: Two-tailed p-value
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have the same length")

    t_stat, p_value = ttest_rel(scores_a, scores_b)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def analyze_ablation_results(
    ablation_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Analyze velocity ablation experiment results.

    Compares position-only vs position+velocity conditions,
    computing mean AUCs and statistical significance.

    Args:
        ablation_results: Dict with 'position_only' and 'position_velocity' keys,
            each containing list of fold results

    Returns:
        Dict with analysis metrics including:
            - position_only_mean_auc
            - position_only_std_auc
            - position_velocity_mean_auc
            - position_velocity_std_auc
            - delta_auc
            - p_value
    """
    pos_only_results = aggregate_fold_results(ablation_results["position_only"])
    pos_vel_results = aggregate_fold_results(ablation_results["position_velocity"])

    analysis = {
        "position_only_mean_auc": pos_only_results["mean_auc"],
        "position_only_std_auc": pos_only_results["std_auc"],
        "position_only_fold_aucs": pos_only_results["fold_aucs"],
        "position_velocity_mean_auc": pos_vel_results["mean_auc"],
        "position_velocity_std_auc": pos_vel_results["std_auc"],
        "position_velocity_fold_aucs": pos_vel_results["fold_aucs"],
        "delta_auc": pos_vel_results["mean_auc"] - pos_only_results["mean_auc"],
    }

    # Statistical test if we have enough folds
    if (
        len(pos_only_results["fold_aucs"]) >= 2
        and len(pos_vel_results["fold_aucs"]) >= 2
        and len(pos_only_results["fold_aucs"]) == len(pos_vel_results["fold_aucs"])
    ):
        t_test = paired_t_test(
            pos_only_results["fold_aucs"],
            pos_vel_results["fold_aucs"],
        )
        analysis["t_stat"] = t_test["t_stat"]
        analysis["p_value"] = t_test["p_value"]
    else:
        analysis["t_stat"] = float("nan")
        analysis["p_value"] = float("nan")

    return analysis


def format_ablation_report(analysis: Dict[str, Any]) -> str:
    """Format ablation analysis results for reporting.

    Creates a human-readable summary of the velocity ablation experiment.

    Args:
        analysis: Output of analyze_ablation_results

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 50,
        "VELOCITY ABLATION EXPERIMENT RESULTS",
        "=" * 50,
        "",
        "Condition A: Position-only (vx=0, vy=0)",
        f"  Mean AUC: {analysis['position_only_mean_auc']:.3f} "
        f"+/- {analysis.get('position_only_std_auc', 0):.3f}",
        "",
        "Condition B: Position + Velocity",
        f"  Mean AUC: {analysis['position_velocity_mean_auc']:.3f} "
        f"+/- {analysis.get('position_velocity_std_auc', 0):.3f}",
        "",
        "-" * 50,
        f"Delta AUC: {analysis['delta_auc']:.3f}",
        "",
    ]

    if "p_value" in analysis and not np.isnan(analysis["p_value"]):
        lines.append(f"Paired t-test p-value: {analysis['p_value']:.4f}")

        if analysis["p_value"] < 0.05:
            lines.append("Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
        elif analysis["p_value"] < 0.10:
            lines.append("Result: Marginally significant (p < 0.10)")
        else:
            lines.append("Result: Not statistically significant (p >= 0.10)")
    else:
        lines.append("Statistical test: N/A (insufficient folds)")

    lines.append("=" * 50)

    return "\n".join(lines)
