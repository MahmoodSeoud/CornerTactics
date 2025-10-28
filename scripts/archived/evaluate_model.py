#!/usr/bin/env python3
"""
Model Evaluation Script for Corner Kick GNN

Implements Phase 3.6: Comprehensive model evaluation and analysis.
Evaluates trained models, generates predictions, and produces visualizations.

Usage:
    python scripts/evaluate_model.py --model-path models/corner_gnn_gcn_goal/best_model.pth
    python scripts/evaluate_model.py --experiment-dir models/corner_gnn_gcn_goal_20241023

Author: mseo
Date: October 2024
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    calibration_curve
)
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gnn_model import create_model
from src.data_loader import load_corner_dataset
from src.train_utils import MetricsComputer

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained GNN model")

    # Model arguments
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--experiment-dir', type=str, default=None,
                       help='Path to experiment directory')
    parser.add_argument('--model-type', type=str, default='gcn',
                       choices=['gcn', 'gat'],
                       help='Model type')

    # Data arguments
    parser.add_argument('--graph-path', type=str,
                       default='data/graphs/adjacency_team/statsbomb_graphs.pkl',
                       help='Path to graph dataset')
    parser.add_argument('--outcome-type', type=str, default='goal',
                       choices=['goal', 'shot', 'multi'],
                       help='Outcome type')

    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions to CSV')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate plots')

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, model_type: str,
                               device: torch.device) -> nn.Module:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model (gcn/gat)
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"Loading model from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = create_model(model_type=model_type)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"Model from epoch: {checkpoint['epoch'] + 1}")
    if 'metrics' in checkpoint:
        val_metrics = checkpoint['metrics']
        if 'auc_roc' in val_metrics:
            print(f"Validation AUC: {val_metrics['auc_roc']:.4f}")

    return model


@torch.no_grad()
def get_predictions(model: nn.Module, loader, device: torch.device) -> Dict:
    """
    Get model predictions on a dataset.

    Args:
        model: Trained model
        loader: Data loader
        device: Device

    Returns:
        Dictionary with predictions, probabilities, labels, etc.
    """
    model.eval()

    all_probs = []
    all_labels = []
    all_corner_ids = []
    all_features = []

    for batch in tqdm(loader, desc="Getting predictions"):
        batch = batch.to(device)

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.sigmoid(out.squeeze())

        # Store results
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

        # Store corner IDs if available
        if hasattr(batch, 'corner_id'):
            all_corner_ids.extend(batch.corner_id)

        # Store node features for later analysis
        # Average pool node features per graph for simplicity
        num_graphs = batch.batch.max().item() + 1
        for i in range(num_graphs):
            mask = batch.batch == i
            graph_features = batch.x[mask].mean(dim=0).cpu().numpy()
            all_features.append(graph_features)

    return {
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels).squeeze(),
        'corner_ids': all_corner_ids,
        'features': np.array(all_features) if all_features else None
    }


def compute_metrics(predictions: Dict, threshold: float = 0.5) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        predictions: Dictionary with predictions and labels
        threshold: Decision threshold

    Returns:
        Dictionary of metrics
    """
    probs = predictions['probabilities']
    labels = predictions['labels']

    # Binary predictions
    preds = (probs > threshold).astype(int)

    # Basic metrics
    metrics = {
        'num_samples': len(labels),
        'num_positive': int(labels.sum()),
        'num_negative': int(len(labels) - labels.sum()),
        'positive_rate': float(labels.mean())
    }

    # Classification metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics.update({
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0)
    })

    # Probabilistic metrics
    try:
        metrics['auc_roc'] = roc_auc_score(labels, probs)
        metrics['avg_precision'] = average_precision_score(labels, probs)
    except ValueError:
        metrics['auc_roc'] = 0.0
        metrics['avg_precision'] = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics.update({
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0
    })

    # Calibration metrics
    fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=10,
                                                strategy='uniform')
    calibration_error = np.mean(np.abs(fraction_pos - mean_pred))
    metrics['calibration_error'] = float(calibration_error)

    return metrics


def plot_roc_curve(predictions: Dict, output_path: Path):
    """Plot ROC curve."""
    labels = predictions['labels']
    probs = predictions['probabilities']

    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Corner Kick Goal Prediction')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {output_path / 'roc_curve.png'}")


def plot_precision_recall_curve(predictions: Dict, output_path: Path):
    """Plot precision-recall curve."""
    labels = predictions['labels']
    probs = predictions['probabilities']

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2,
            label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Corner Kick Goal Prediction')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Mark baseline (random classifier)
    baseline = labels.mean()
    plt.axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
               label=f'Baseline ({baseline:.3f})')

    plt.savefig(output_path / 'precision_recall_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve to {output_path / 'precision_recall_curve.png'}")


def plot_calibration_curve(predictions: Dict, output_path: Path):
    """Plot calibration curve."""
    labels = predictions['labels']
    probs = predictions['probabilities']

    fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=10,
                                               strategy='uniform')

    plt.figure(figsize=(8, 6))
    plt.plot(mean_pred, fraction_pos, marker='o', linewidth=2,
            label='Model', markersize=8)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray',
            label='Perfectly calibrated')

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve - Corner Kick Goal Prediction')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path / 'calibration_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration curve to {output_path / 'calibration_curve.png'}")


def plot_confusion_matrix(predictions: Dict, threshold: float, output_path: Path):
    """Plot confusion matrix."""
    labels = predictions['labels']
    probs = predictions['probabilities']
    preds = (probs > threshold).astype(int)

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Goal', 'Goal'],
                yticklabels=['No Goal', 'Goal'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Corner Kick Goal Prediction')

    plt.savefig(output_path / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path / 'confusion_matrix.png'}")


def plot_probability_distribution(predictions: Dict, output_path: Path):
    """Plot distribution of predicted probabilities."""
    probs = predictions['probabilities']
    labels = predictions['labels']

    plt.figure(figsize=(10, 6))

    # Separate by actual outcome
    goal_probs = probs[labels == 1]
    no_goal_probs = probs[labels == 0]

    plt.hist(no_goal_probs, bins=50, alpha=0.5, label='No Goal', color='blue',
            density=True)
    plt.hist(goal_probs, bins=50, alpha=0.5, label='Goal', color='red',
            density=True)

    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path / 'probability_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved probability distribution to {output_path / 'probability_distribution.png'}")


def analyze_feature_importance(model: nn.Module, loader, device: torch.device,
                              output_path: Path):
    """
    Analyze feature importance using gradient-based attribution.

    Simple implementation - can be enhanced with more sophisticated methods.
    """
    model.eval()
    feature_gradients = []
    feature_dim = 14  # Number of input features

    for batch in tqdm(loader, desc="Computing feature importance", total=min(10, len(loader))):
        if len(feature_gradients) >= 10:  # Sample a few batches
            break

        batch = batch.to(device)
        batch.x.requires_grad = True

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = out.mean()  # Simple aggregation

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Store gradients
        if batch.x.grad is not None:
            grads = batch.x.grad.abs().mean(dim=0).cpu().numpy()
            feature_gradients.append(grads)

    # Average gradients
    avg_importance = np.mean(feature_gradients, axis=0)

    # Feature names (from Phase 2.1)
    feature_names = [
        'x', 'y', 'distance_to_goal', 'distance_to_ball_target',
        'vx', 'vy', 'velocity_magnitude', 'velocity_angle',
        'angle_to_goal', 'angle_to_ball', 'team_flag', 'in_penalty_box',
        'num_players_within_5m', 'local_density_score'
    ]

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    indices = np.argsort(avg_importance)[::-1]
    plt.bar(range(feature_dim), avg_importance[indices])
    plt.xticks(range(feature_dim), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance (Gradient Magnitude)')
    plt.title('Feature Importance Analysis')
    plt.tight_layout()

    plt.savefig(output_path / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance to {output_path / 'feature_importance.png'}")

    return dict(zip(feature_names, avg_importance))


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find model path
    if args.model_path:
        model_path = Path(args.model_path)
    elif args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        model_path = exp_dir / 'best_model.pth'
        if not model_path.exists():
            model_path = exp_dir / 'latest_checkpoint.pth'
    else:
        # Find most recent model
        models_dir = Path('models')
        if models_dir.exists():
            exp_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
            if exp_dirs:
                model_path = exp_dirs[-1] / 'best_model.pth'
            else:
                print("No model found. Please specify --model-path or --experiment-dir")
                return
        else:
            print("No models directory found")
            return

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 60)
    print("Corner Kick GNN Model Evaluation")
    print("=" * 60)

    # Load model
    model = load_model_from_checkpoint(str(model_path), args.model_type, device)

    # Load data
    print("\nLoading data...")
    dataset, train_loader, val_loader, test_loader = load_corner_dataset(
        graph_path=args.graph_path,
        outcome_type=args.outcome_type,
        batch_size=args.batch_size,
        random_state=42
    )

    # Get predictions on test set
    print("\nGenerating predictions on test set...")
    test_predictions = get_predictions(model, test_loader, device)

    # Compute metrics
    print("\nComputing metrics...")
    test_metrics = compute_metrics(test_predictions, args.threshold)

    # Print metrics
    print("\nTest Set Performance:")
    print("-" * 40)
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        plot_roc_curve(test_predictions, output_dir)
        plot_precision_recall_curve(test_predictions, output_dir)
        plot_calibration_curve(test_predictions, output_dir)
        plot_confusion_matrix(test_predictions, args.threshold, output_dir)
        plot_probability_distribution(test_predictions, output_dir)

    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_importance = analyze_feature_importance(model, test_loader, device, output_dir)

    # Save predictions if requested
    if args.save_predictions:
        predictions_df = pd.DataFrame({
            'corner_id': test_predictions['corner_ids'],
            'true_label': test_predictions['labels'],
            'predicted_prob': test_predictions['probabilities'],
            'predicted_label': (test_predictions['probabilities'] > args.threshold).astype(int)
        })
        predictions_path = output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Saved predictions to {predictions_path}")

    # Save evaluation results
    results = {
        'model_path': str(model_path),
        'test_metrics': test_metrics,
        'feature_importance': feature_importance,
        'threshold': args.threshold,
        'args': vars(args)
    }

    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()