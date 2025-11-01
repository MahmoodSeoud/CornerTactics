#!/usr/bin/env python3
"""
Training Script for TacticAI Baseline Models

Implements Days 5-6: Train and evaluate baseline models:
- RandomReceiverBaseline (sanity check)
- XGBoostReceiverBaseline (engineered features)
- MLPReceiverBaseline (simple neural network)

Success Criteria:
- Random: top-1=4.5%, top-3=13.6%
- XGBoost: top-1 > 25%, top-3 > 42%
- MLP: top-1 > 22%, top-3 > 45%
- If MLP top-3 < 40%: STOP and debug data pipeline

Author: mseo
Date: October 2024
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.baselines import (
    RandomReceiverBaseline,
    XGBoostReceiverBaseline,
    MLPReceiverBaseline,
    evaluate_baseline,
    train_mlp_baseline
)
from src.data.receiver_data_loader import load_receiver_dataset


def evaluate_random_baseline(data_loader, device='cpu'):
    """Evaluate random baseline."""
    print("\n" + "="*80)
    print("EVALUATING RANDOM BASELINE")
    print("="*80)

    model = RandomReceiverBaseline(num_players=22)
    metrics = evaluate_baseline(model, data_loader, device=device)

    print(f"\nRandom Baseline Results:")
    print(f"  Top-1: {metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-3: {metrics['top3_accuracy']*100:.2f}%")
    print(f"  Top-5: {metrics['top5_accuracy']*100:.2f}%")
    print(f"  Loss:  {metrics['loss']:.4f}")

    # Sanity check
    assert 3.0 < metrics['top1_accuracy']*100 < 6.0, \
        f"Random top-1 should be ~4.5%, got {metrics['top1_accuracy']*100:.1f}%"
    assert 11.0 < metrics['top3_accuracy']*100 < 16.0, \
        f"Random top-3 should be ~13.6%, got {metrics['top3_accuracy']*100:.1f}%"

    print("\n✅ Random baseline sanity check passed!")

    return {
        'model': 'Random',
        'top1_accuracy': float(metrics['top1_accuracy']),
        'top3_accuracy': float(metrics['top3_accuracy']),
        'top5_accuracy': float(metrics['top5_accuracy']),
        'loss': float(metrics['loss'])
    }


def train_and_evaluate_xgboost(train_loader, val_loader, test_loader, device='cpu'):
    """Train and evaluate XGBoost baseline."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST BASELINE")
    print("="*80)

    model = XGBoostReceiverBaseline(
        max_depth=6,
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    )

    # Collect training data
    print("\nCollecting training data...")
    train_x_list = []
    train_batch_list = []
    train_labels = []

    for batch in train_loader:
        batch_size = batch.batch.max().item() + 1
        for i in range(batch_size):
            mask = (batch.batch == i)
            graph_x = batch.x[mask]
            batch_tensor = torch.full((graph_x.size(0),), i, dtype=torch.long)

            train_x_list.append(graph_x)
            train_batch_list.append(batch_tensor)
            train_labels.append(batch.receiver_label[i].item())

    print(f"Collected {len(train_labels)} training corners")

    # Train XGBoost
    print("\nTraining XGBoost (500 trees, max_depth=6, lr=0.05)...")
    model.train(train_x_list, train_batch_list, train_labels)
    print("✅ Training complete!")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_baseline(model, val_loader, device=device)

    print(f"\nXGBoost Validation Results:")
    print(f"  Top-1: {val_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-3: {val_metrics['top3_accuracy']*100:.2f}%")
    print(f"  Top-5: {val_metrics['top5_accuracy']*100:.2f}%")
    print(f"  Loss:  {val_metrics['loss']:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_baseline(model, test_loader, device=device)

    print(f"\nXGBoost Test Results:")
    print(f"  Top-1: {test_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-3: {test_metrics['top3_accuracy']*100:.2f}%")
    print(f"  Top-5: {test_metrics['top5_accuracy']*100:.2f}%")
    print(f"  Loss:  {test_metrics['loss']:.4f}")

    # Success criteria check
    if test_metrics['top3_accuracy'] >= 0.42:
        print("\n✅ XGBoost baseline meets success criteria (top-3 > 42%)!")
    else:
        print(f"\n⚠️  XGBoost baseline below target (top-3 = {test_metrics['top3_accuracy']*100:.1f}% < 42%)")

    return {
        'model': 'XGBoost',
        'hyperparameters': {
            'max_depth': 6,
            'n_estimators': 500,
            'learning_rate': 0.05
        },
        'val_top1_accuracy': float(val_metrics['top1_accuracy']),
        'val_top3_accuracy': float(val_metrics['top3_accuracy']),
        'val_top5_accuracy': float(val_metrics['top5_accuracy']),
        'val_loss': float(val_metrics['loss']),
        'test_top1_accuracy': float(test_metrics['top1_accuracy']),
        'test_top3_accuracy': float(test_metrics['top3_accuracy']),
        'test_top5_accuracy': float(test_metrics['top5_accuracy']),
        'test_loss': float(test_metrics['loss'])
    }


def train_and_evaluate_mlp(train_loader, val_loader, test_loader, device='cuda'):
    """Train and evaluate MLP baseline."""
    print("\n" + "="*80)
    print("TRAINING MLP BASELINE")
    print("="*80)

    model = MLPReceiverBaseline(
        num_features=14,
        num_players=22,
        hidden_dim1=256,
        hidden_dim2=128,
        dropout=0.3
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nMLP Architecture:")
    print(f"  Input: 308 (22 players × 14 features)")
    print(f"  Hidden: 308 → 256 → 128 → 22")
    print(f"  Parameters: {num_params:,}")
    print(f"  Dropout: 0.3")

    # Train MLP
    print(f"\nTraining MLP for 10,000 steps...")
    history = train_mlp_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_steps=10000,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        eval_every=1000,
        verbose=True
    )

    print("\n✅ Training complete!")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_baseline(model, test_loader, device=device)

    print(f"\nMLP Test Results:")
    print(f"  Top-1: {test_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-3: {test_metrics['top3_accuracy']*100:.2f}%")
    print(f"  Top-5: {test_metrics['top5_accuracy']*100:.2f}%")
    print(f"  Loss:  {test_metrics['loss']:.4f}")

    # Critical success criteria check
    if test_metrics['top3_accuracy'] < 0.40:
        print("\n❌ CRITICAL: MLP top-3 < 40% - DEBUG DATA PIPELINE!")
        print("This indicates a potential issue with receiver labels or data quality.")
    elif test_metrics['top3_accuracy'] >= 0.45:
        print("\n✅ MLP baseline meets success criteria (top-3 > 45%)!")
    else:
        print(f"\n⚠️  MLP baseline marginal (40% < top-3 = {test_metrics['top3_accuracy']*100:.1f}% < 45%)")

    return {
        'model': 'MLP',
        'architecture': {
            'input_dim': 308,
            'hidden_dims': [256, 128],
            'output_dim': 22,
            'dropout': 0.3,
            'num_parameters': num_params
        },
        'training': {
            'num_steps': 10000,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'best_val_top3': float(history['best_val_top3'])
        },
        'val_top1_accuracy': float(history['val_top1'][-1]),
        'val_top3_accuracy': float(history['val_top3'][-1]),
        'val_top5_accuracy': float(history['val_top5'][-1]),
        'test_top1_accuracy': float(test_metrics['top1_accuracy']),
        'test_top3_accuracy': float(test_metrics['top3_accuracy']),
        'test_top5_accuracy': float(test_metrics['top5_accuracy']),
        'test_loss': float(test_metrics['loss'])
    }


def main():
    """Main training script."""
    print("="*80)
    print("TACTICAI BASELINE MODELS TRAINING")
    print("Days 5-6: Baseline Receiver Prediction Models")
    print("="*80)

    # Configuration
    graph_path = "data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl"
    batch_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Graph file: {graph_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    dataset, train_loader, val_loader, test_loader = load_receiver_dataset(
        graph_path=graph_path,
        batch_size=batch_size,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        mask_velocities=True
    )

    # Train and evaluate all baselines
    results = {
        'date': datetime.now().isoformat(),
        'dataset': {
            'graph_path': graph_path,
            'total_corners': len(dataset.data_list),
            'num_features': dataset.num_features,
            'batch_size': batch_size
        },
        'baselines': []
    }

    # 1. Random Baseline
    random_results = evaluate_random_baseline(test_loader, device=device)
    results['baselines'].append(random_results)

    # 2. XGBoost Baseline
    xgboost_results = train_and_evaluate_xgboost(
        train_loader, val_loader, test_loader, device=device
    )
    results['baselines'].append(xgboost_results)

    # 3. MLP Baseline
    mlp_results = train_and_evaluate_mlp(
        train_loader, val_loader, test_loader, device=device
    )
    results['baselines'].append(mlp_results)

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nModel Comparison (Test Set):")
    print(f"{'Model':<15} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8}")
    print("-" * 45)

    for baseline in results['baselines']:
        model_name = baseline['model']
        top1 = baseline['test_top1_accuracy'] if 'test_top1_accuracy' in baseline else baseline['top1_accuracy']
        top3 = baseline['test_top3_accuracy'] if 'test_top3_accuracy' in baseline else baseline['top3_accuracy']
        top5 = baseline['test_top5_accuracy'] if 'test_top5_accuracy' in baseline else baseline['top5_accuracy']

        print(f"{model_name:<15} {top1*100:>7.2f}% {top3*100:>7.2f}% {top5*100:>7.2f}%")

    print(f"\n✅ Results saved to: {results_file}")
    print("\n" + "="*80)
    print("BASELINE TRAINING COMPLETE")
    print("="*80)

    # Final decision point
    mlp_top3 = mlp_results['test_top3_accuracy']
    if mlp_top3 >= 0.45:
        print("\n✅ SUCCESS: Proceed to Phase 2 (GATv2 with D2)")
        print(f"MLP top-3 accuracy: {mlp_top3*100:.1f}% > 45%")
    elif mlp_top3 >= 0.40:
        print("\n⚠️  MARGINAL: MLP top-3 acceptable but below target")
        print(f"MLP top-3 accuracy: {mlp_top3*100:.1f}% (40% < x < 45%)")
    else:
        print("\n❌ FAILURE: MLP top-3 < 40% - DEBUG DATA PIPELINE BEFORE PROCEEDING")
        print(f"MLP top-3 accuracy: {mlp_top3*100:.1f}%")


if __name__ == "__main__":
    main()
