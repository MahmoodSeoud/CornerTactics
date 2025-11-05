#!/usr/bin/env python3
"""
Train Multi-Class Outcome Prediction Baselines

Implements Day 6.5: Multi-Class Outcome Baselines
Trains 3 baseline models for 4-class outcome prediction:
- RandomOutcomeBaseline: Uniform random (25% accuracy)
- XGBoostOutcomeBaseline: Graph-level features (50-60% accuracy)
- MLPOutcomeBaseline: Flattened MLP (55-65% accuracy)

Based on TacticAI Implementation Plan:
- Class distribution: Goal (1.3%), Shot (15.8%), Clearance (39.5%), Possession (43.4%)
- Expected Macro F1: Random (0.25), XGBoost (> 0.45), MLP (> 0.50)
- Weighted loss for class imbalance

Author: mseo
Date: November 2024
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.receiver_data_loader import load_receiver_dataset, OUTCOME_CLASS_MAPPING
from src.models.baselines import (
    RandomOutcomeBaseline, XGBoostOutcomeBaseline, MLPOutcomeBaseline,
    evaluate_outcome_baseline, train_mlp_outcome
)


def save_feature_importance(model, output_dir: Path):
    """
    Save XGBoost feature importance to JSON and create basic plot.

    Args:
        model: Trained XGBoostOutcomeBaseline model
        output_dir: Directory to save outputs
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get feature importance
    try:
        booster = model.model
        importance_dict = booster.get_score(importance_type='gain')

        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        # Save to JSON
        json_path = output_dir / "feature_importance.json"
        with open(json_path, 'w') as f:
            json.dump(sorted_importance, f, indent=2)
        print(f"✓ Saved feature importance to {json_path}")

        # Create basic plot
        top_n = min(20, len(sorted_importance))
        top_features = list(sorted_importance.items())[:top_n]
        features = [f[0] for f in top_features]
        scores = [f[1] for f in top_features]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(features)), scores, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Feature Importance (Gain)', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} XGBoost Features', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "feature_importance_basic.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved basic importance plot to {plot_path}")
        plt.close()

        # Print top 10
        print("\nTop 10 Most Important Features:")
        for i, (feature, score) in enumerate(list(sorted_importance.items())[:10], 1):
            print(f"  {i:2d}. {feature:40s}: {score:8.2f}")

        print(f"\nRun visualization script to create pitch-based visualizations:")
        print(f"  python scripts/visualization/visualize_feature_importance.py \\")
        print(f"      --importance-path {json_path} \\")
        print(f"      --output-dir {output_dir / 'visualizations'}")

    except Exception as e:
        print(f"⚠️  Could not extract feature importance: {e}")


def print_dataset_statistics(dataset, train_loader, val_loader, test_loader):
    """Print comprehensive dataset statistics."""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS (3-CLASS)")
    print("=" * 80)

    # Outcome class distribution
    all_outcomes = [data.outcome_class_label.item() for data in dataset.data_list]
    class_names = ['Shot', 'Clearance', 'Possession']  # 3-class (Goal+Shot merged)

    print("\nOutcome class distribution:")
    for class_id, class_name in enumerate(class_names):
        count = sum(1 for o in all_outcomes if o == class_id)
        pct = count / len(all_outcomes) * 100
        print(f"  {class_id}. {class_name:12s}: {count:4d} ({pct:5.1f}%)")

    # Train/val/test statistics
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_loader.dataset):4d} samples")
    print(f"  Val:   {len(val_loader.dataset):4d} samples")
    print(f"  Test:  {len(test_loader.dataset):4d} samples")

    print("=" * 80 + "\n")


def train_random_baseline(test_loader, device='cpu'):
    """Evaluate random baseline (no training needed)."""
    print("\n" + "=" * 80)
    print("RANDOM OUTCOME BASELINE (3-CLASS)")
    print("=" * 80)
    print("Expected: 33% accuracy (uniform random over 3 classes)")
    print("=" * 80 + "\n")

    model = RandomOutcomeBaseline(num_classes=3)
    model.to(device)

    # Evaluate on test set
    metrics = evaluate_outcome_baseline(model, test_loader, device)

    print(f"\nTest Results:")
    print(f"  Accuracy:        {metrics['accuracy']*100:.1f}%")
    print(f"  Macro F1:        {metrics['macro_f1']:.3f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.3f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.3f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.3f}")

    print(f"\nPer-class F1 scores:")
    class_names = ['Shot', 'Clearance', 'Possession']  # 3-class
    for name in class_names:
        f1_key = f'{name.lower()}_f1'
        print(f"  {name:12s}: {metrics[f1_key]:.3f}")

    print(f"\nConfusion Matrix:")
    conf_matrix = np.array(metrics['confusion_matrix'])
    print("  Predicted:  Shot  Clear  Poss")
    for i, actual in enumerate(class_names):
        row_str = "  " + actual[:6].ljust(10) + ": "
        row_str += "  ".join(f"{conf_matrix[i, j]:5d}" for j in range(3))
        print(row_str)

    return metrics


def train_xgboost_baseline(train_loader, val_loader, test_loader, device='cpu'):
    """Train XGBoost baseline with graph-level features."""
    print("\n" + "=" * 80)
    print("XGBOOST OUTCOME BASELINE (3-CLASS)")
    print("=" * 80)
    print("Expected: 55-65% accuracy, Macro F1 > 0.50")
    print("Graph-level features: position stats, team balance, zonal density")
    print("=" * 80 + "\n")

    # Prepare training data
    print("Extracting graph-level features...")
    train_x_list = []
    train_batch_list = []
    train_labels = []

    for batch in train_loader:
        batch_size = batch.batch.max().item() + 1
        for i in range(batch_size):
            mask = (batch.batch == i)
            graph_x = batch.x[mask]
            train_x_list.append(graph_x)
            train_batch_list.append(batch.batch[mask])
            train_labels.append(batch.outcome_class_label[i].item())

    print(f"Extracted {len(train_x_list)} training graphs")

    # Train XGBoost
    print("\nTraining XGBoost classifier (500 trees, max_depth=6)...")
    model = XGBoostOutcomeBaseline(
        max_depth=6,
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    )
    model.train(train_x_list, train_batch_list, train_labels)
    print("✓ Training complete")

    # Evaluate on validation set
    print("\nValidation Results:")
    val_metrics = evaluate_outcome_baseline(model, val_loader, device)
    print(f"  Accuracy:    {val_metrics['accuracy']*100:.1f}%")
    print(f"  Macro F1:    {val_metrics['macro_f1']:.3f}")
    print(f"  Weighted F1: {val_metrics['weighted_f1']:.3f}")

    # Evaluate on test set
    print("\nTest Results:")
    test_metrics = evaluate_outcome_baseline(model, test_loader, device)
    print(f"  Accuracy:        {test_metrics['accuracy']*100:.1f}%")
    print(f"  Macro F1:        {test_metrics['macro_f1']:.3f}")
    print(f"  Macro Precision: {test_metrics['macro_precision']:.3f}")
    print(f"  Macro Recall:    {test_metrics['macro_recall']:.3f}")
    print(f"  Weighted F1:     {test_metrics['weighted_f1']:.3f}")

    print(f"\nPer-class F1 scores:")
    class_names = ['Shot', 'Clearance', 'Possession']
    for name in class_names:
        f1_key = f'{name.lower()}_f1'
        print(f"  {name:12s}: {test_metrics[f1_key]:.3f}")

    print(f"\nConfusion Matrix:")
    conf_matrix = np.array(test_metrics['confusion_matrix'])
    print("  Predicted:  Shot  Clear  Poss")
    for i, actual in enumerate(class_names):
        row_str = "  " + actual[:5].ljust(9) + ": "
        row_str += "  ".join(f"{conf_matrix[i, j]:4d}" for j in range(3))
        print(row_str)

    return model, test_metrics


def train_mlp_baseline(train_loader, val_loader, test_loader, device='cuda', num_steps=15000):
    """Train MLP baseline with flattened player positions."""
    print("\n" + "=" * 80)
    print("MLP OUTCOME BASELINE (3-CLASS)")
    print("=" * 80)
    print("Expected: 60-70% accuracy, Macro F1 > 0.55")
    print("Architecture: 308 → 512 → 256 → 128 → 3")
    print(f"Training for {num_steps} steps with class-weighted loss")
    print("=" * 80 + "\n")

    # Create model
    model = MLPOutcomeBaseline(
        num_features=14,
        num_players=22,
        hidden_dim1=512,
        hidden_dim2=256,
        hidden_dim3=128,
        num_classes=3,  # 3-class: Shot, Clearance, Possession
        dropout=0.25
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    history = train_mlp_outcome(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_steps=num_steps,
        lr=5e-4,
        weight_decay=1e-4,
        device=device,
        eval_every=500,
        verbose=True,
        use_class_weights=True
    )

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80 + "\n")

    test_metrics = evaluate_outcome_baseline(model, test_loader, device)
    print(f"Test Results:")
    print(f"  Accuracy:        {test_metrics['accuracy']*100:.1f}%")
    print(f"  Macro F1:        {test_metrics['macro_f1']:.3f}")
    print(f"  Macro Precision: {test_metrics['macro_precision']:.3f}")
    print(f"  Macro Recall:    {test_metrics['macro_recall']:.3f}")
    print(f"  Weighted F1:     {test_metrics['weighted_f1']:.3f}")

    print(f"\nPer-class F1 scores:")
    class_names = ['Shot', 'Clearance', 'Possession']
    for name in class_names:
        f1_key = f'{name.lower()}_f1'
        print(f"  {name:12s}: {test_metrics[f1_key]:.3f}")

    print(f"\nConfusion Matrix:")
    conf_matrix = np.array(test_metrics['confusion_matrix'])
    print("  Predicted:  Shot  Clear  Poss")
    for i, actual in enumerate(class_names):
        row_str = "  " + actual[:5].ljust(9) + ": "
        row_str += "  ".join(f"{conf_matrix[i, j]:4d}" for j in range(3))
        print(row_str)

    return test_metrics, history, model


def save_results(results: Dict, output_dir: Path):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, metrics in results.items():
        output_file = output_dir / f"outcome_{model_name}_results.json"

        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, list)):
                metrics_json[key] = value if isinstance(value, list) else value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metrics_json[key] = float(value)
            else:
                metrics_json[key] = value

        with open(output_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)

        print(f"✓ Saved {model_name} results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Train multi-class outcome baselines')
    parser.add_argument('--graph-path', type=str,
                       default='data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl',
                       help='Path to graph pickle file with receiver labels')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num-steps', type=int, default=15000,
                       help='Number of training steps for MLP (default: 15000)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu, default: cuda)')
    parser.add_argument('--models', type=str, default='all',
                       help='Models to train: all/random/xgboost/mlp (default: all)')
    parser.add_argument('--output-dir', type=str, default='results/baselines',
                       help='Output directory for results (default: results/baselines)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    print("\n" + "=" * 80)
    print("MULTI-CLASS OUTCOME BASELINE TRAINING")
    print("=" * 80)
    print(f"Graph path: {args.graph_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Models: {args.models}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 80)

    # Load dataset
    print("\nLoading dataset...")
    dataset, train_loader, val_loader, test_loader = load_receiver_dataset(
        graph_path=args.graph_path,
        batch_size=args.batch_size,
        test_size=0.15,
        val_size=0.15,
        random_state=args.random_seed,
        mask_velocities=True
    )

    # Print statistics
    print_dataset_statistics(dataset, train_loader, val_loader, test_loader)

    # Train models
    results = {}

    if args.models in ['all', 'random']:
        results['random'] = train_random_baseline(test_loader, args.device)

    if args.models in ['all', 'xgboost']:
        xgboost_model, xgboost_metrics = train_xgboost_baseline(train_loader, val_loader, test_loader, args.device)
        results['xgboost'] = xgboost_metrics

        # Save feature importance
        print("\nExtracting and saving feature importance...")
        save_feature_importance(xgboost_model, output_dir=Path(args.output_dir))

    if args.models in ['all', 'mlp']:
        test_metrics, history, model = train_mlp_baseline(
            train_loader, val_loader, test_loader, args.device, args.num_steps
        )
        results['mlp'] = test_metrics

        # Save MLP model
        model_path = Path(args.output_dir) / 'outcome_mlp_model.pt'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'test_metrics': test_metrics
        }, model_path)
        print(f"\n✓ Saved MLP model to {model_path}")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80 + "\n")
    save_results(results, Path(args.output_dir))

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY (3-CLASS: DANGEROUS/CLEARANCE/POSSESSION)")
    print("=" * 80)
    print(f"\nModel Performance (Test Set):")
    print(f"{'Model':<15} {'Accuracy':<12} {'Macro F1':<12} {'Shot F1':<12} {'Clear F1':<12}")
    print("-" * 67)

    for model_name, metrics in results.items():
        print(f"{model_name.capitalize():<15} "
              f"{metrics['accuracy']*100:>6.1f}%      "
              f"{metrics['macro_f1']:>6.3f}       "
              f"{metrics.get('shot_f1', 0.0):>6.3f}       "
              f"{metrics.get('clearance_f1', 0.0):>6.3f}")

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
