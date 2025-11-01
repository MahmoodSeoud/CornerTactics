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
    evaluate_shot_prediction,
    train_mlp_baseline
)
from src.data.receiver_data_loader import load_receiver_dataset


def evaluate_random_baseline(data_loader, device='cpu'):
    """Evaluate random baseline."""
    print("\n" + "="*80)
    print("EVALUATING RANDOM BASELINE")
    print("="*80)

    model = RandomReceiverBaseline(num_players=22)

    # Evaluate receiver prediction
    receiver_metrics = evaluate_baseline(model, data_loader, device=device)

    # Evaluate shot prediction
    shot_metrics = evaluate_shot_prediction(model, data_loader, device=device)

    print(f"\nRandom Baseline Results (Receiver):")
    print(f"  Top-1: {receiver_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-3: {receiver_metrics['top3_accuracy']*100:.2f}%")
    print(f"  Top-5: {receiver_metrics['top5_accuracy']*100:.2f}%")

    print(f"\nRandom Baseline Results (Shot):")
    print(f"  Accuracy: {shot_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score: {shot_metrics['f1_score']:.4f}")
    print(f"  AUROC: {shot_metrics['auroc']:.4f}")
    print(f"  AUPRC: {shot_metrics['auprc']:.4f}")

    # Sanity check (relaxed bounds for variable number of players)
    assert 3.0 < receiver_metrics['top1_accuracy']*100 < 10.0, \
        f"Random top-1 should be 3-10%, got {receiver_metrics['top1_accuracy']*100:.1f}%"
    assert 10.0 < receiver_metrics['top3_accuracy']*100 < 20.0, \
        f"Random top-3 should be 10-20%, got {receiver_metrics['top3_accuracy']*100:.1f}%"

    print("\n✅ Random baseline sanity check passed!")

    return {
        'model': 'Random',
        'receiver': {
            'top1_accuracy': float(receiver_metrics['top1_accuracy']),
            'top3_accuracy': float(receiver_metrics['top3_accuracy']),
            'top5_accuracy': float(receiver_metrics['top5_accuracy']),
            'loss': float(receiver_metrics['loss'])
        },
        'shot': {
            'accuracy': float(shot_metrics['accuracy']),
            'f1_score': float(shot_metrics['f1_score']),
            'precision': float(shot_metrics['precision']),
            'recall': float(shot_metrics['recall']),
            'auroc': float(shot_metrics['auroc']),
            'auprc': float(shot_metrics['auprc'])
        }
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
    shot_labels = []

    for batch in train_loader:
        batch_size = batch.batch.max().item() + 1
        for i in range(batch_size):
            mask = (batch.batch == i)
            graph_x = batch.x[mask]
            batch_tensor = torch.full((graph_x.size(0),), i, dtype=torch.long)

            train_x_list.append(graph_x)
            train_batch_list.append(batch_tensor)
            train_labels.append(batch.receiver_label[i].item())
            shot_labels.append(batch.shot_label[i].item())

    print(f"Collected {len(train_labels)} training corners")

    # Train XGBoost
    print("\nTraining XGBoost (500 trees, max_depth=6, lr=0.05)...")
    print("Training receiver prediction...")
    model.train(train_x_list, train_batch_list, train_labels)
    print("Training shot prediction...")
    model.train_shot(train_x_list, train_batch_list, shot_labels)
    print("✅ Training complete!")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_receiver_metrics = evaluate_baseline(model, val_loader, device=device)
    val_shot_metrics = evaluate_shot_prediction(model, val_loader, device=device)

    print(f"\nXGBoost Validation Results (Receiver):")
    print(f"  Top-1: {val_receiver_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-3: {val_receiver_metrics['top3_accuracy']*100:.2f}%")
    print(f"  Top-5: {val_receiver_metrics['top5_accuracy']*100:.2f}%")
    print(f"  Loss:  {val_receiver_metrics['loss']:.4f}")

    print(f"\nXGBoost Validation Results (Shot):")
    print(f"  Accuracy: {val_shot_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score: {val_shot_metrics['f1_score']:.4f}")
    print(f"  AUROC: {val_shot_metrics['auroc']:.4f}")
    print(f"  AUPRC: {val_shot_metrics['auprc']:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_receiver_metrics = evaluate_baseline(model, test_loader, device=device)
    test_shot_metrics = evaluate_shot_prediction(model, test_loader, device=device)

    print(f"\nXGBoost Test Results (Receiver):")
    print(f"  Top-1: {test_receiver_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-3: {test_receiver_metrics['top3_accuracy']*100:.2f}%")
    print(f"  Top-5: {test_receiver_metrics['top5_accuracy']*100:.2f}%")
    print(f"  Loss:  {test_receiver_metrics['loss']:.4f}")

    print(f"\nXGBoost Test Results (Shot):")
    print(f"  Accuracy: {test_shot_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score: {test_shot_metrics['f1_score']:.4f}")
    print(f"  AUROC: {test_shot_metrics['auroc']:.4f}")
    print(f"  AUPRC: {test_shot_metrics['auprc']:.4f}")

    # Success criteria check
    if test_receiver_metrics['top3_accuracy'] >= 0.42:
        print("\n✅ XGBoost baseline meets success criteria (top-3 > 42%)!")
    else:
        print(f"\n⚠️  XGBoost baseline below target (top-3 = {test_receiver_metrics['top3_accuracy']*100:.1f}% < 42%)")

    return {
        'model': 'XGBoost',
        'hyperparameters': {
            'max_depth': 6,
            'n_estimators': 500,
            'learning_rate': 0.05
        },
        'receiver': {
            'val_top1_accuracy': float(val_receiver_metrics['top1_accuracy']),
            'val_top3_accuracy': float(val_receiver_metrics['top3_accuracy']),
            'val_top5_accuracy': float(val_receiver_metrics['top5_accuracy']),
            'val_loss': float(val_receiver_metrics['loss']),
            'test_top1_accuracy': float(test_receiver_metrics['top1_accuracy']),
            'test_top3_accuracy': float(test_receiver_metrics['top3_accuracy']),
            'test_top5_accuracy': float(test_receiver_metrics['top5_accuracy']),
            'test_loss': float(test_receiver_metrics['loss'])
        },
        'shot': {
            'val_accuracy': float(val_shot_metrics['accuracy']),
            'val_f1_score': float(val_shot_metrics['f1_score']),
            'val_precision': float(val_shot_metrics['precision']),
            'val_recall': float(val_shot_metrics['recall']),
            'val_auroc': float(val_shot_metrics['auroc']),
            'val_auprc': float(val_shot_metrics['auprc']),
            'test_accuracy': float(test_shot_metrics['accuracy']),
            'test_f1_score': float(test_shot_metrics['f1_score']),
            'test_precision': float(test_shot_metrics['precision']),
            'test_recall': float(test_shot_metrics['recall']),
            'test_auroc': float(test_shot_metrics['auroc']),
            'test_auprc': float(test_shot_metrics['auprc'])
        }
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

    # Train MLP with dual-task learning
    print(f"\nTraining MLP for 10,000 steps with dual-task learning...")
    history = train_mlp_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_steps=10000,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        eval_every=1000,
        verbose=True,
        dual_task=True,  # Enable dual-task training
        shot_weight=1.0  # Equal weight for both tasks
    )

    print("\n✅ Training complete!")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_receiver_metrics = evaluate_baseline(model, test_loader, device=device)
    test_shot_metrics = evaluate_shot_prediction(model, test_loader, device=device)

    print(f"\nMLP Test Results (Receiver):")
    print(f"  Top-1: {test_receiver_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-3: {test_receiver_metrics['top3_accuracy']*100:.2f}%")
    print(f"  Top-5: {test_receiver_metrics['top5_accuracy']*100:.2f}%")
    print(f"  Loss:  {test_receiver_metrics['loss']:.4f}")

    print(f"\nMLP Test Results (Shot):")
    print(f"  Accuracy: {test_shot_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score: {test_shot_metrics['f1_score']:.4f}")
    print(f"  AUROC: {test_shot_metrics['auroc']:.4f}")
    print(f"  AUPRC: {test_shot_metrics['auprc']:.4f}")

    # Critical success criteria check
    if test_receiver_metrics['top3_accuracy'] < 0.40:
        print("\n❌ CRITICAL: MLP top-3 < 40% - DEBUG DATA PIPELINE!")
        print("This indicates a potential issue with receiver labels or data quality.")
    elif test_receiver_metrics['top3_accuracy'] >= 0.45:
        print("\n✅ MLP baseline meets success criteria (top-3 > 45%)!")
    else:
        print(f"\n⚠️  MLP baseline marginal (40% < top-3 = {test_receiver_metrics['top3_accuracy']*100:.1f}% < 45%)")

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
        'receiver': {
            'val_top1_accuracy': float(history['val_top1'][-1]),
            'val_top3_accuracy': float(history['val_top3'][-1]),
            'val_top5_accuracy': float(history['val_top5'][-1]),
            'test_top1_accuracy': float(test_receiver_metrics['top1_accuracy']),
            'test_top3_accuracy': float(test_receiver_metrics['top3_accuracy']),
            'test_top5_accuracy': float(test_receiver_metrics['top5_accuracy']),
            'test_loss': float(test_receiver_metrics['loss'])
        },
        'shot': {
            'test_accuracy': float(test_shot_metrics['accuracy']),
            'test_f1_score': float(test_shot_metrics['f1_score']),
            'test_precision': float(test_shot_metrics['precision']),
            'test_recall': float(test_shot_metrics['recall']),
            'test_auroc': float(test_shot_metrics['auroc']),
            'test_auprc': float(test_shot_metrics['auprc'])
        }
    }


def main():
    """Main training script."""
    print("="*80)
    print("TACTICAI BASELINE MODELS TRAINING")
    print("Days 5-6: Baseline Receiver Prediction Models")
    print("="*80)

    # Configuration
    graph_path = "data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl"
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

    # Receiver Prediction Results
    print(f"\nReceiver Prediction (Test Set):")
    print(f"{'Model':<15} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8}")
    print("-" * 45)

    for baseline in results['baselines']:
        model_name = baseline['model']
        # Handle both receiver-nested and flat structures
        if 'receiver' in baseline:
            # XGBoost and MLP use nested structure
            receiver = baseline['receiver']
            top1 = receiver.get('test_top1_accuracy', receiver.get('top1_accuracy', 0))
            top3 = receiver.get('test_top3_accuracy', receiver.get('top3_accuracy', 0))
            top5 = receiver.get('test_top5_accuracy', receiver.get('top5_accuracy', 0))
        else:
            # Random baseline uses flat structure
            top1 = baseline.get('top1_accuracy', 0)
            top3 = baseline.get('top3_accuracy', 0)
            top5 = baseline.get('top5_accuracy', 0)

        print(f"{model_name:<15} {top1*100:>7.2f}% {top3*100:>7.2f}% {top5*100:>7.2f}%")

    # Shot Prediction Results
    print(f"\nShot Prediction (Test Set):")
    print(f"{'Model':<15} {'F1':>8} {'AUROC':>8} {'AUPRC':>8}")
    print("-" * 45)

    for baseline in results['baselines']:
        model_name = baseline['model']
        if 'shot' in baseline:
            f1 = baseline['shot']['test_f1_score']
            auroc = baseline['shot']['test_auroc']
            auprc = baseline['shot']['test_auprc']
            print(f"{model_name:<15} {f1:>8.4f} {auroc:>8.4f} {auprc:>8.4f}")

    print(f"\n✅ Results saved to: {results_file}")
    print("\n" + "="*80)
    print("BASELINE TRAINING COMPLETE")
    print("="*80)

    # Final decision point
    mlp_top3 = mlp_results['receiver']['test_top3_accuracy']
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
