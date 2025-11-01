#!/usr/bin/env python3
"""
Config 3: Combined approach - Reduced weight (0.5) + Weighted BCE
Hypothesis: Combination of both fixes provides best balance
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.train_baseline import main

if __name__ == "__main__":
    # Override with combined approach
    import scripts.training.train_baseline as tb
    import src.models.baselines as baselines

    # Create custom training function with both fixes
    def combined_train_mlp_baseline(model, train_loader, val_loader,
                                   num_steps=10000, lr=1e-3, weight_decay=1e-4,
                                   device='cuda', eval_every=500, verbose=True):
        """Train with both reduced weight AND weighted BCE loss"""
        import numpy as np
        from sklearn.metrics import f1_score, roc_auc_score

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        receiver_criterion = nn.CrossEntropyLoss()

        # WEIGHTED BCE LOSS (positive class weight = 2.0, slightly less aggressive)
        pos_weight = torch.tensor([2.0]).to(device)
        shot_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # REDUCED SHOT WEIGHT
        shot_weight = 0.5

        history = {'train_loss': [], 'val_top3': [], 'val_shot_auroc': [], 'steps': []}

        step = 0
        best_combined_score = 0.0
        best_model_state = None

        if verbose:
            print(f"\nTraining MLP with COMBINED APPROACH")
            print(f"Shot weight: 0.5 (reduced from 1.0)")
            print(f"Positive class weight: 2.0")
            print(f"Evaluating every {eval_every} steps\n")

        train_iter = iter(train_loader)

        while step < num_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = batch.to(device)
            model.train()
            optimizer.zero_grad()

            # Receiver prediction
            logits = model(batch.x, batch.batch)
            receiver_targets = batch.receiver_label.squeeze()
            receiver_loss = receiver_criterion(logits, receiver_targets)

            # Shot prediction with weighted loss
            batch_size = batch.batch.max().item() + 1
            flattened = model._flatten_batch(batch.x, batch.batch, batch_size)
            h1 = model.relu(model.fc1(flattened))
            h1 = model.dropout(h1)
            h2 = model.relu(model.fc2(h1))
            h2 = model.dropout(h2)
            shot_logits = model.fc3_shot(h2)
            shot_targets = batch.shot_label.float().unsqueeze(1)
            shot_loss = shot_criterion(shot_logits, shot_targets)

            # Combined loss with reduced weight
            total_loss = receiver_loss + shot_weight * shot_loss

            total_loss.backward()
            optimizer.step()
            step += 1

            # Evaluate
            if step % eval_every == 0 or step == num_steps:
                val_metrics = baselines.evaluate_baseline(model, val_loader, device)

                # Evaluate shot
                all_shot_preds = []
                all_shot_labels = []
                model.eval()
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        shot_probs = model.predict_shot(val_batch.x, val_batch.batch)
                        all_shot_preds.extend(shot_probs.cpu().numpy().flatten())
                        all_shot_labels.extend(val_batch.shot_label.cpu().numpy())

                shot_preds_binary = (np.array(all_shot_preds) > 0.5).astype(int)
                val_shot_f1 = f1_score(all_shot_labels, shot_preds_binary)
                val_shot_auroc = roc_auc_score(all_shot_labels, all_shot_preds)

                if verbose:
                    print(f"Step {step:5d}/{num_steps} | "
                          f"Loss: {total_loss.item():.4f} (R:{receiver_loss.item():.3f}, S:{shot_loss.item():.3f}) | "
                          f"Val Top-3: {val_metrics['top3_accuracy']*100:.1f}% | "
                          f"Shot F1: {val_shot_f1:.3f} | "
                          f"Shot AUROC: {val_shot_auroc:.3f}")

                # Save based on COMBINED score (prioritize good performance on both tasks)
                combined_score = val_metrics['top3_accuracy'] + 0.5 * val_shot_auroc
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_model_state = model.state_dict().copy()

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return {'best_combined_score': best_combined_score}

    # Custom MLP training function
    original_train_mlp = tb.train_and_evaluate_mlp

    def custom_train_mlp(train_loader, val_loader, test_loader, device='cuda'):
        """Custom MLP training with combined approach"""
        print("\n" + "="*80)
        print("CONFIG 3: COMBINED APPROACH")
        print("="*80)

        from src.models.baselines import MLPReceiverBaseline, evaluate_baseline, evaluate_shot_prediction

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
        print(f"  Shot weight: 0.5")
        print(f"  Loss: BCEWithLogitsLoss(pos_weight=2.0)")

        # Train with combined approach
        history = combined_train_mlp_baseline(
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
            'model': 'MLP-Config3',
            'config': 'shot_weight=0.5 + BCE_pos_weight=2.0',
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