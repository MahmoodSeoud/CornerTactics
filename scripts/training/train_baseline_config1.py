#!/usr/bin/env python3
"""
Config 1: Reduced shot weight (0.3)
Hypothesis: Lower weight prevents shot task from disrupting receiver learning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.train_baseline import main

if __name__ == "__main__":
    # Override shot weight to 0.3
    import scripts.training.train_baseline as tb

    # Monkey-patch the train_and_evaluate_mlp function
    original_train_mlp = tb.train_and_evaluate_mlp

    def custom_train_mlp(train_loader, val_loader, test_loader, device='cuda'):
        """Custom MLP training with shot_weight=0.3"""
        print("\n" + "="*80)
        print("CONFIG 1: REDUCED SHOT WEIGHT (0.3)")
        print("="*80)

        from src.models.baselines import MLPReceiverBaseline, train_mlp_baseline, evaluate_baseline, evaluate_shot_prediction

        model = MLPReceiverBaseline(
            num_features=14,
            num_players=22,
            hidden_dim1=256,
            hidden_dim2=128,
            dropout=0.3
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nMLP Architecture:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Shot weight: 0.3 (reduced from 1.0)")

        # Train with reduced shot weight
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
            dual_task=True,
            shot_weight=0.3  # REDUCED WEIGHT
        )

        print("\n✅ Training complete!")

        # Evaluate
        test_receiver_metrics = evaluate_baseline(model, test_loader, device=device)
        test_shot_metrics = evaluate_shot_prediction(model, test_loader, device=device)

        print(f"\nMLP Test Results (Receiver):")
        print(f"  Top-1: {test_receiver_metrics['top1_accuracy']*100:.2f}%")
        print(f"  Top-3: {test_receiver_metrics['top3_accuracy']*100:.2f}%")

        print(f"\nMLP Test Results (Shot):")
        print(f"  F1 Score: {test_shot_metrics['f1_score']:.4f}")
        print(f"  AUROC: {test_shot_metrics['auroc']:.4f}")

        return {
            'model': 'MLP-Config1',
            'config': 'shot_weight=0.3',
            'receiver': {
                'test_top3_accuracy': float(test_receiver_metrics['top3_accuracy']),
            },
            'shot': {
                'test_f1_score': float(test_shot_metrics['f1_score']),
                'test_auroc': float(test_shot_metrics['auroc']),
            }
        }

    tb.train_and_evaluate_mlp = custom_train_mlp
    main()