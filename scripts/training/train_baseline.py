#!/usr/bin/env python3
"""
Baseline Model Training Script for TacticAI-Style Receiver Prediction

Implements Day 5-6: Baseline Models Training
- Train RandomReceiverBaseline (sanity check)
- Train MLPReceiverBaseline for 10k steps
- Compute top-1, top-3, top-5 accuracy
- Save results to results/baseline_mlp.json

Success Criteria:
- Random baseline: top-1=4.5%, top-3=13.6% (sanity check)
- MLP baseline: top-1 > 20%, top-3 > 45%
- If MLP top-3 < 40%: STOP and debug data pipeline

Author: mseo
Date: October 2024
"""

import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.receiver_data_loader import load_receiver_dataset
from src.models.baselines import (
    RandomReceiverBaseline,
    MLPReceiverBaseline,
    evaluate_baseline,
    train_mlp_baseline
)


def main():
    parser = argparse.ArgumentParser(description="Train baseline receiver prediction models")
    parser.add_argument('--graph-path', type=str,
                       default='data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl',
                       help='Path to graph pickle file with receiver labels')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num-steps', type=int, default=10000,
                       help='Number of training steps for MLP baseline')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Set random seeds
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_state)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("TACTICAI BASELINE MODELS TRAINING")
    print("="*70)
    print(f"Graph file: {args.graph_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Training steps: {args.num_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Random seed: {args.random_state}")
    print("="*70)

    # Load dataset
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASET")
    print("="*70)

    dataset, train_loader, val_loader, test_loader = load_receiver_dataset(
        graph_path=args.graph_path,
        batch_size=args.batch_size,
        test_size=0.15,
        val_size=0.15,
        random_state=args.random_state,
        mask_velocities=True
    )

    print(f"\n✅ Dataset loaded successfully")
    print(f"   Total corners: {len(dataset.data_list)}")
    print(f"   Node features: {dataset.num_features} dimensions (velocities masked)")

    # ========================================================================
    # STEP 2: RANDOM BASELINE (SANITY CHECK)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: RANDOM BASELINE (SANITY CHECK)")
    print("="*70)
    print("Expected performance:")
    print("  Top-1: 4.5% (1/22 players)")
    print("  Top-3: 13.6% (3/22 players)")
    print("  Top-5: 22.7% (5/22 players)")
    print()

    random_model = RandomReceiverBaseline(num_players=22)

    # Evaluate on validation set
    print("Evaluating on validation set...")
    random_val_metrics = evaluate_baseline(random_model, val_loader, device)

    print(f"\nRandom Baseline Results (Val Set):")
    print(f"  Top-1: {random_val_metrics['top1_accuracy']*100:.1f}%")
    print(f"  Top-3: {random_val_metrics['top3_accuracy']*100:.1f}%")
    print(f"  Top-5: {random_val_metrics['top5_accuracy']*100:.1f}%")
    print(f"  Loss:  {random_val_metrics['loss']:.4f}")

    # Sanity check: Should be close to random chance
    if abs(random_val_metrics['top1_accuracy'] - 0.045) > 0.02:
        print("\n⚠️  WARNING: Random baseline top-1 deviates from expected 4.5%")
        print("    This may indicate an issue with the evaluation pipeline")
    else:
        print("\n✅ Random baseline passes sanity check!")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    random_test_metrics = evaluate_baseline(random_model, test_loader, device)

    print(f"\nRandom Baseline Results (Test Set):")
    print(f"  Top-1: {random_test_metrics['top1_accuracy']*100:.1f}%")
    print(f"  Top-3: {random_test_metrics['top3_accuracy']*100:.1f}%")
    print(f"  Top-5: {random_test_metrics['top5_accuracy']*100:.1f}%")

    # ========================================================================
    # STEP 3: MLP BASELINE TRAINING
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: MLP BASELINE TRAINING")
    print("="*70)
    print("Expected performance:")
    print("  Top-1: > 20%")
    print("  Top-3: > 45%")
    print("  ❌ If Top-3 < 40%: STOP and debug data pipeline")
    print()

    mlp_model = MLPReceiverBaseline(
        num_features=14,
        num_players=22,
        hidden_dim1=256,
        hidden_dim2=128,
        dropout=0.3
    )

    # Count parameters
    num_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"MLP Model: {num_params:,} parameters")

    # Train model
    history = train_mlp_baseline(
        model=mlp_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_steps=args.num_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        eval_every=500,
        verbose=True
    )

    # ========================================================================
    # STEP 4: FINAL EVALUATION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: FINAL EVALUATION")
    print("="*70)

    # Evaluate on validation set
    print("\nEvaluating best MLP model on validation set...")
    mlp_val_metrics = evaluate_baseline(mlp_model, val_loader, device)

    print(f"\nMLP Baseline Results (Val Set):")
    print(f"  Top-1: {mlp_val_metrics['top1_accuracy']*100:.1f}%")
    print(f"  Top-3: {mlp_val_metrics['top3_accuracy']*100:.1f}%")
    print(f"  Top-5: {mlp_val_metrics['top5_accuracy']*100:.1f}%")
    print(f"  Loss:  {mlp_val_metrics['loss']:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    mlp_test_metrics = evaluate_baseline(mlp_model, test_loader, device)

    print(f"\nMLP Baseline Results (Test Set):")
    print(f"  Top-1: {mlp_test_metrics['top1_accuracy']*100:.1f}%")
    print(f"  Top-3: {mlp_test_metrics['top3_accuracy']*100:.1f}%")
    print(f"  Top-5: {mlp_test_metrics['top5_accuracy']*100:.1f}%")
    print(f"  Loss:  {mlp_test_metrics['loss']:.4f}")

    # ========================================================================
    # STEP 5: SUCCESS CRITERIA CHECK
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: SUCCESS CRITERIA CHECK")
    print("="*70)

    success = True

    # Check MLP top-3 accuracy
    val_top3 = mlp_val_metrics['top3_accuracy']
    test_top3 = mlp_test_metrics['top3_accuracy']

    print(f"\nMLP Top-3 Accuracy:")
    print(f"  Val:  {val_top3*100:.1f}%")
    print(f"  Test: {test_top3*100:.1f}%")
    print()

    if val_top3 > 0.45:
        print("✅ SUCCESS: MLP top-3 > 45% - Proceed to Phase 2 (GATv2)")
    elif val_top3 > 0.40:
        print("⚠️  MARGINAL: MLP top-3 between 40-45% - Consider debugging")
        print("   But still acceptable to proceed to Phase 2")
    else:
        print("❌ FAILURE: MLP top-3 < 40% - STOP and debug data pipeline!")
        print("   Check receiver label extraction in:")
        print("   scripts/preprocessing/add_receiver_labels.py")
        success = False

    # Check MLP top-1 accuracy
    val_top1 = mlp_val_metrics['top1_accuracy']
    if val_top1 > 0.20:
        print(f"✅ SUCCESS: MLP top-1 > 20% ({val_top1*100:.1f}%)")
    else:
        print(f"⚠️  WARNING: MLP top-1 < 20% ({val_top1*100:.1f}%)")

    # ========================================================================
    # STEP 6: SAVE RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: SAVING RESULTS")
    print("="*70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'graph_path': args.graph_path,
            'batch_size': args.batch_size,
            'num_steps': args.num_steps,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'random_state': args.random_state,
            'device': device,
        },
        'dataset': {
            'num_corners': len(dataset.data_list),
            'num_features': dataset.num_features,
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'test_size': len(test_loader.dataset),
        },
        'random_baseline': {
            'val': {
                'top1': random_val_metrics['top1_accuracy'],
                'top3': random_val_metrics['top3_accuracy'],
                'top5': random_val_metrics['top5_accuracy'],
                'loss': random_val_metrics['loss'],
            },
            'test': {
                'top1': random_test_metrics['top1_accuracy'],
                'top3': random_test_metrics['top3_accuracy'],
                'top5': random_test_metrics['top5_accuracy'],
                'loss': random_test_metrics['loss'],
            }
        },
        'mlp_baseline': {
            'num_parameters': num_params,
            'val': {
                'top1': mlp_val_metrics['top1_accuracy'],
                'top3': mlp_val_metrics['top3_accuracy'],
                'top5': mlp_val_metrics['top5_accuracy'],
                'loss': mlp_val_metrics['loss'],
            },
            'test': {
                'top1': mlp_test_metrics['top1_accuracy'],
                'top3': mlp_test_metrics['top3_accuracy'],
                'top5': mlp_test_metrics['top5_accuracy'],
                'loss': mlp_test_metrics['loss'],
            },
            'training_history': history,
        },
        'success_criteria': {
            'passed': success,
            'mlp_top3_val': val_top3,
            'mlp_top1_val': val_top1,
            'proceed_to_phase2': val_top3 >= 0.40,
        }
    }

    # Save to JSON
    output_file = output_dir / 'baseline_mlp.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\nRandom Baseline (Sanity Check):")
    print(f"  Val Top-3:  {random_val_metrics['top3_accuracy']*100:.1f}% (expected 13.6%)")
    print(f"  Test Top-3: {random_test_metrics['top3_accuracy']*100:.1f}%")

    print(f"\nMLP Baseline (Main Result):")
    print(f"  Val Top-1:  {mlp_val_metrics['top1_accuracy']*100:.1f}% (target > 20%)")
    print(f"  Val Top-3:  {mlp_val_metrics['top3_accuracy']*100:.1f}% (target > 45%)")
    print(f"  Val Top-5:  {mlp_val_metrics['top5_accuracy']*100:.1f}%")
    print(f"  Test Top-3: {mlp_test_metrics['top3_accuracy']*100:.1f}%")

    print(f"\nDecision:")
    if success and val_top3 >= 0.45:
        print("  ✅ PROCEED TO PHASE 2: GATv2 Implementation")
        print("     MLP baseline performance is strong enough to validate data pipeline")
    elif val_top3 >= 0.40:
        print("  ⚠️  PROCEED WITH CAUTION: Performance is marginal")
        print("     Consider investigating data quality before Phase 2")
    else:
        print("  ❌ STOP: Debug data pipeline before proceeding")
        print("     Check receiver label extraction and data quality")

    print("="*70)


if __name__ == "__main__":
    main()
