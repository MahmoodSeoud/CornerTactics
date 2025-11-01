#!/usr/bin/env python3
"""
Train baseline models for TacticAI-style receiver and shot prediction.

Implements training for three baseline models:
1. Random baseline (sanity check)
2. XGBoost with engineered features
3. MLP with flattened features

Expected performance targets:
- Random: top-3 = 13.6%
- XGBoost: top-3 > 42%
- MLP: top-3 > 45%
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.receiver_data_loader import load_receiver_dataset
from src.models.baselines import (
    RandomReceiverBaseline,
    XGBoostReceiverBaseline,
    MLPReceiverBaseline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """Compute top-k accuracy."""
    import torch.nn.functional as F

    # Ensure both tensors are on the same device
    labels = labels.to(logits.device)

    probs = F.softmax(logits, dim=1)
    results = {}

    for k in k_values:
        if k == 1:
            pred = probs.argmax(dim=1)
            acc = (pred == labels).float().mean().item()
        else:
            topk = probs.topk(k, dim=1)[1]
            acc = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f'top{k}'] = acc

    return results


def compute_shot_metrics(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute shot prediction metrics."""
    # Ensure both tensors are on the same device
    labels = labels.to(probs.device)

    preds = (probs >= threshold).float()
    labels = labels.float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}


def prepare_data_for_xgboost(data_loader) -> Tuple[List, List, List]:
    """Extract features for XGBoost training."""
    x_list = []
    receiver_labels = []
    shot_labels = []

    for batch in data_loader:
        batch_size = batch.batch.max().item() + 1

        for i in range(batch_size):
            mask = (batch.batch == i)
            graph_x = batch.x[mask]

            x_list.append(graph_x)

            # Extract labels
            if hasattr(batch, 'receiver_label'):
                receiver_labels.append(batch.receiver_label[i].item())
            else:
                receiver_labels.append(0)

            if hasattr(batch, 'shot_label'):
                shot_labels.append(batch.shot_label[i].item())
            else:
                shot_labels.append(0)

    return x_list, receiver_labels, shot_labels


def evaluate_receiver_model(model, data_loader, device='cuda') -> Dict:
    """Evaluate receiver prediction model."""
    model.eval() if hasattr(model, 'eval') else None

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device) if hasattr(batch, 'to') else batch

            # Get predictions
            if hasattr(model, 'forward'):
                logits = model(batch.x, batch.batch)
            else:
                logits = model.predict(batch.x, batch.batch)

            all_logits.append(logits)
            all_labels.append(batch.receiver_label)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).squeeze()

    # Compute top-k accuracy
    metrics = compute_topk_accuracy(all_logits, all_labels, k_values=[1, 3, 5])

    return metrics


def evaluate_shot_model(model, data_loader, device='cuda', threshold=0.3) -> Dict:
    """Evaluate shot prediction model."""
    model.eval() if hasattr(model, 'eval') else None

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device) if hasattr(batch, 'to') else batch

            # Get predictions
            if hasattr(model, 'predict_shot'):
                probs = model.predict_shot(batch.x, batch.batch)
            else:
                # Fallback for models without shot prediction
                probs = torch.rand(batch.shot_label.shape)

            all_probs.append(probs)
            all_labels.append(batch.shot_label)

    all_probs = torch.cat(all_probs, dim=0).squeeze()
    all_labels = torch.cat(all_labels, dim=0).squeeze()

    # Compute metrics
    metrics = compute_shot_metrics(all_probs, all_labels, threshold=threshold)

    # Add AUROC if sklearn available
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        metrics['auroc'] = roc_auc_score(all_labels.cpu(), all_probs.cpu())
        metrics['auprc'] = average_precision_score(all_labels.cpu(), all_probs.cpu())
    except:
        pass

    return metrics


def train_mlp_baseline(model, train_loader, val_loader, config, device='cuda'):
    """Train MLP baseline model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    receiver_criterion = nn.CrossEntropyLoss()
    shot_criterion = nn.BCELoss()

    best_val_top3 = 0.0
    best_model_state = None
    history = []

    logger.info(f"Training MLP for {config['num_steps']} steps...")

    train_iter = iter(train_loader)
    for step in range(config['num_steps']):
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(device)
        model.train()

        # Forward pass - receiver prediction
        optimizer.zero_grad()
        logits = model(batch.x, batch.batch)
        receiver_loss = receiver_criterion(logits, batch.receiver_label.squeeze())

        # Shot prediction
        shot_probs = model.predict_shot(batch.x, batch.batch)
        shot_loss = shot_criterion(shot_probs, batch.shot_label.unsqueeze(1))

        # Combined loss
        total_loss = receiver_loss + config['shot_weight'] * shot_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Evaluate periodically
        if (step + 1) % config['eval_every'] == 0:
            val_receiver_metrics = evaluate_receiver_model(model, val_loader, device)
            val_shot_metrics = evaluate_shot_model(model, val_loader, device)

            logger.info(
                f"Step {step+1}/{config['num_steps']} | "
                f"Loss: {total_loss.item():.4f} | "
                f"Val Top-3: {val_receiver_metrics['top3']*100:.1f}% | "
                f"Shot F1: {val_shot_metrics['f1']:.3f}"
            )

            # Save best model
            if val_receiver_metrics['top3'] > best_val_top3:
                best_val_top3 = val_receiver_metrics['top3']
                best_model_state = model.state_dict().copy()

            history.append({
                'step': step + 1,
                'train_loss': total_loss.item(),
                'val_top3': val_receiver_metrics['top3'],
                'val_shot_f1': val_shot_metrics['f1']
            })

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def main(args):
    """Main training function."""
    logger.info("="*70)
    logger.info(f"Training {args.model} baseline model")
    logger.info("="*70)

    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading data...")
    # Load dataset with correct parameters (no num_workers, no require_receiver)
    graph_path = str(Path(args.data_path)) if args.data_path else "data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl"
    dataset, train_loader, val_loader, test_loader = load_receiver_dataset(
        graph_path=graph_path,
        batch_size=args.batch_size,
        mask_velocities=True  # Always mask velocities for StatsBomb data
    )

    # Create dataset_info dict for compatibility
    dataset_info = {
        'train_graphs': len(train_loader.dataset),
        'val_graphs': len(val_loader.dataset),
        'test_graphs': len(test_loader.dataset),
        'num_players': 22,
        'num_features': 14
    }

    logger.info(f"Dataset: {dataset_info['train_graphs']} train, "
                f"{dataset_info['val_graphs']} val, "
                f"{dataset_info['test_graphs']} test")

    # Initialize model
    logger.info(f"Initializing {args.model} model...")

    if args.model == 'random':
        model = RandomReceiverBaseline(num_players=22)

    elif args.model == 'xgboost':
        model = XGBoostReceiverBaseline(
            max_depth=args.xgb_max_depth,
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate
        )

        # Prepare data for XGBoost
        logger.info("Preparing data for XGBoost...")
        train_x, train_receiver, train_shot = prepare_data_for_xgboost(train_loader)
        val_x, val_receiver, val_shot = prepare_data_for_xgboost(val_loader)

        # Train receiver model
        logger.info("Training XGBoost receiver model...")
        model.train(train_x, None, train_receiver)

        # Train shot model
        logger.info("Training XGBoost shot model...")
        model.train_shot(train_x, None, train_shot)

    elif args.model == 'mlp':
        model = MLPReceiverBaseline(
            num_features=14,
            num_players=22,
            hidden_dim1=args.mlp_hidden1,
            hidden_dim2=args.mlp_hidden2,
            dropout=args.mlp_dropout
        )

        # Train MLP
        config = {
            'num_steps': args.mlp_steps,
            'lr': args.mlp_lr,
            'weight_decay': args.mlp_weight_decay,
            'eval_every': args.eval_every,
            'shot_weight': args.shot_weight
        }
        model, history = train_mlp_baseline(model, train_loader, val_loader, config, device)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")

    # Receiver prediction
    test_receiver_metrics = evaluate_receiver_model(model, test_loader, device)
    logger.info("Receiver Prediction Results:")
    logger.info(f"  Top-1: {test_receiver_metrics['top1']*100:.2f}%")
    logger.info(f"  Top-3: {test_receiver_metrics['top3']*100:.2f}%")
    logger.info(f"  Top-5: {test_receiver_metrics['top5']*100:.2f}%")

    # Shot prediction
    test_shot_metrics = evaluate_shot_model(model, test_loader, device)
    logger.info("Shot Prediction Results:")
    logger.info(f"  F1: {test_shot_metrics['f1']:.3f}")
    logger.info(f"  Precision: {test_shot_metrics['precision']:.3f}")
    logger.info(f"  Recall: {test_shot_metrics['recall']:.3f}")
    if 'auroc' in test_shot_metrics:
        logger.info(f"  AUROC: {test_shot_metrics['auroc']:.3f}")
        logger.info(f"  AUPRC: {test_shot_metrics['auprc']:.3f}")

    # Check success criteria
    logger.info("\n" + "="*70)
    logger.info("SUCCESS CRITERIA CHECK")
    logger.info("="*70)

    if args.model == 'random':
        expected_top3 = 0.136
        if abs(test_receiver_metrics['top3'] - expected_top3) < 0.02:
            logger.info("✅ Random baseline performs as expected (~13.6%)")
        else:
            logger.warning(f"⚠️ Random baseline unexpected: {test_receiver_metrics['top3']*100:.1f}% (expected ~13.6%)")

    elif args.model == 'xgboost':
        if test_receiver_metrics['top3'] > 0.42:
            logger.info(f"✅ XGBoost exceeds target (>42%): {test_receiver_metrics['top3']*100:.1f}%")
        else:
            logger.warning(f"⚠️ XGBoost below target (<42%): {test_receiver_metrics['top3']*100:.1f}%")

    elif args.model == 'mlp':
        if test_receiver_metrics['top3'] > 0.45:
            logger.info(f"✅ MLP exceeds target (>45%): {test_receiver_metrics['top3']*100:.1f}%")
        elif test_receiver_metrics['top3'] > 0.40:
            logger.info(f"⚠️ MLP acceptable (>40%): {test_receiver_metrics['top3']*100:.1f}%")
        else:
            logger.error(f"❌ MLP FAILURE (<40%): {test_receiver_metrics['top3']*100:.1f}%")
            logger.error("    Check data pipeline - receiver labels may be incorrect!")

    # Save results
    results = {
        'model': args.model,
        'timestamp': datetime.now().isoformat(),
        'dataset_info': dataset_info,
        'test_receiver_metrics': test_receiver_metrics,
        'test_shot_metrics': test_shot_metrics,
        'args': vars(args)
    }

    output_dir = Path('results/baselines')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.model}_results_{args.gpu_type}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    # Save model
    if args.save_model and args.model in ['mlp']:
        model_file = output_dir / f"{args.model}_model_{args.gpu_type}.pt"
        torch.save(model.state_dict(), model_file)
        logger.info(f"Model saved to: {model_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TacticAI baseline models")

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['random', 'xgboost', 'mlp'],
                       help='Baseline model to train')

    # Data
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to graph pickle file')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')

    # XGBoost hyperparameters
    parser.add_argument('--xgb-max-depth', type=int, default=6,
                       help='XGBoost max depth')
    parser.add_argument('--xgb-n-estimators', type=int, default=500,
                       help='XGBoost number of estimators')
    parser.add_argument('--xgb-learning-rate', type=float, default=0.05,
                       help='XGBoost learning rate')

    # MLP hyperparameters
    parser.add_argument('--mlp-hidden1', type=int, default=256,
                       help='MLP first hidden dimension')
    parser.add_argument('--mlp-hidden2', type=int, default=128,
                       help='MLP second hidden dimension')
    parser.add_argument('--mlp-dropout', type=float, default=0.3,
                       help='MLP dropout rate')
    parser.add_argument('--mlp-lr', type=float, default=1e-3,
                       help='MLP learning rate')
    parser.add_argument('--mlp-weight-decay', type=float, default=1e-4,
                       help='MLP weight decay')
    parser.add_argument('--mlp-steps', type=int, default=10000,
                       help='MLP training steps')
    parser.add_argument('--eval-every', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--shot-weight', type=float, default=1.0,
                       help='Weight for shot loss')

    # GPU settings
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--gpu-type', type=str, default='v100',
                       choices=['v100', 'a100', 'h100'],
                       help='GPU type for results tracking')

    # Other
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)