#!/usr/bin/env python3
"""
Balanced metrics for imbalanced classification.
"""
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def find_optimal_threshold(y_true, y_scores, metric='f1'):
    """
    Find optimal decision threshold for binary classification.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        metric: 'f1', 'balanced_acc', or 'mcc'

    Returns:
        optimal_threshold, best_score
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'balanced_acc':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]


def evaluate_with_multiple_thresholds(y_true, y_scores):
    """
    Evaluate model at multiple thresholds.

    Returns dict with metrics at different thresholds.
    """
    results = {}

    # Default threshold (0.5)
    y_pred_default = (y_scores >= 0.5).astype(int)

    # Calculate confusion matrix for default threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_default).ravel()

    results['default_0.5'] = {
        'threshold': 0.5,
        'f1': f1_score(y_true, y_pred_default),
        'precision': precision_score(y_true, y_pred_default, zero_division=0),
        'recall': recall_score(y_true, y_pred_default),
        'balanced_acc': balanced_accuracy_score(y_true, y_pred_default),
        'mcc': matthews_corrcoef(y_true, y_pred_default),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }

    # Optimal thresholds for different metrics
    for metric in ['f1', 'balanced_acc', 'mcc']:
        opt_threshold, best_score = find_optimal_threshold(
            y_true, y_scores, metric=metric
        )
        y_pred_opt = (y_scores >= opt_threshold).astype(int)

        # Calculate confusion matrix for optimal threshold
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_opt).ravel()

        results[f'optimal_{metric}'] = {
            'threshold': opt_threshold,
            'f1': f1_score(y_true, y_pred_opt),
            'precision': precision_score(y_true, y_pred_opt, zero_division=0),
            'recall': recall_score(y_true, y_pred_opt),
            'balanced_acc': balanced_accuracy_score(y_true, y_pred_opt),
            'mcc': matthews_corrcoef(y_true, y_pred_opt),
            f'best_{metric}': best_score,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }

    # Average precision (threshold-independent)
    results['average_precision'] = average_precision_score(y_true, y_scores)

    return results


def print_metrics_comparison(results):
    """Pretty print metrics comparison"""
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*60)

    for name, metrics in results.items():
        if name == 'average_precision':
            print(f"\nAverage Precision (threshold-independent): {metrics:.4f}")
        else:
            print(f"\n{name}:")
            print(f"  Threshold: {metrics['threshold']:.3f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Balanced Accuracy: {metrics['balanced_acc']:.4f}")
            print(f"  Matthews Correlation: {metrics['mcc']:.4f}")

            # Print confusion matrix details
            if 'tp' in metrics:
                print(f"  Confusion Matrix:")
                print(f"    TP: {metrics['tp']}, FP: {metrics['fp']}")
                print(f"    FN: {metrics['fn']}, TN: {metrics['tn']}")


def calculate_class_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate comprehensive metrics for binary classification.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        threshold: Decision threshold

    Returns:
        Dictionary with all metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'average_precision': average_precision_score(y_true, y_pred_proba),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'threshold': threshold
    }

    return metrics