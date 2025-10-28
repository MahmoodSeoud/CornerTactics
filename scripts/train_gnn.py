#!/usr/bin/env python3
"""
Training Script for Corner Kick GNN

Implements Phase 3.4: Main training pipeline for corner kick outcome prediction.
Trains a Graph Neural Network on corner kick data to predict goal probability.

Usage:
    python scripts/train_gnn.py --epochs 100 --batch-size 32
    python scripts/train_gnn.py --model gat --lr 0.001 --patience 15

Author: mseo
Date: October 2024
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gnn_model import create_model, count_parameters
from src.data_loader import load_corner_dataset
from src.train_utils import (
    EarlyStopping, MetricsComputer,
    train_epoch, validate_epoch,
    save_checkpoint, load_checkpoint,
    print_metrics, compute_class_weights,
    focal_loss
)

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GNN for corner kick prediction")

    # Data arguments
    parser.add_argument('--graph-path', type=str,
                       default='data/graphs/adjacency_team/combined_temporal_graphs.pkl',
                       help='Path to graph dataset (combined StatsBomb + SkillCorner)')
    parser.add_argument('--outcome-type', type=str, default='shot',
                       choices=['goal', 'shot', 'multi'],
                       help='Outcome to predict (shot = dangerous situation: shot OR goal)')

    # Model arguments
    parser.add_argument('--model', type=str, default='gcn',
                       choices=['gcn', 'gcn_edge', 'gat'],
                       help='Model type (GCN, GCN with edge features, or GAT)')
    parser.add_argument('--use-edge-features', action='store_true', default=False,
                       help='Use 6-dim edge features (distance, velocity, angle)')
    parser.add_argument('--hidden-dim1', type=int, default=64,
                       help='First hidden dimension')
    parser.add_argument('--hidden-dim2', type=int, default=128,
                       help='Second hidden dimension')
    parser.add_argument('--hidden-dim3', type=int, default=64,
                       help='Third hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for GCN layers')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--loss', type=str, default='bce',
                       choices=['bce', 'focal', 'weighted'],
                       help='Loss function')

    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=0.0001,
                       help='Minimum change for early stopping')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='runs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create unique experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"corner_gnn_{args.model}_{args.outcome_type}_{timestamp}"
    experiment_dir = save_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    # Save training configuration
    config_path = experiment_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to {config_path}")

    # Setup tensorboard
    log_dir = Path(args.log_dir) / experiment_name
    writer = SummaryWriter(log_dir)

    print("\n" + "=" * 60)
    print("Corner Kick GNN Training")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    dataset, train_loader, val_loader, test_loader = load_corner_dataset(
        graph_path=args.graph_path,
        outcome_type=args.outcome_type,
        batch_size=args.batch_size,
        random_state=args.seed
    )

    # Create model
    use_edges = args.use_edge_features or args.model == 'gcn_edge'
    model_name = args.model.upper() + (" (with edge features)" if use_edges else "")
    print(f"\nCreating {model_name} model...")
    model = create_model(
        model_type=args.model,
        use_edge_features=use_edges,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        hidden_dim3=args.hidden_dim3,
        dropout_rate=args.dropout
    )
    model = model.to(device)
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    if use_edges:
        print("✓ Using 6-dimensional edge features (distance, velocity, angle)")

    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Setup loss function
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'weighted':
        # Compute class weights for imbalanced dataset
        pos_weight = compute_class_weights(train_loader)[1]
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    elif args.loss == 'focal':
        criterion = lambda preds, targets: focal_loss(preds, targets)
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    # Setup scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5,
                                     factor=0.5, verbose=True)
    else:
        scheduler = None

    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='max',
        verbose=True
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_auc = 0
    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint.get('best_val_auc', 0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    training_history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        # Training
        train_start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer,
                                   criterion, device)
        train_time = time.time() - train_start

        # Validation
        val_start = time.time()
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        val_time = time.time() - val_start

        # Print metrics
        print(f"Time: Train {train_time:.1f}s, Val {val_time:.1f}s")
        print_metrics(train_metrics, "Train")
        print_metrics(val_metrics, "Val")

        # Update learning rate
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['auc_roc'])
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('AUC/train', train_metrics.get('auc_roc', 0), epoch)
        writer.add_scalar('AUC/val', val_metrics.get('auc_roc', 0), epoch)
        writer.add_scalar('AP/train', train_metrics.get('avg_precision', 0), epoch)
        writer.add_scalar('AP/val', val_metrics.get('avg_precision', 0), epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)

        # Save history
        history_keys = {
            'loss': 'loss',
            'accuracy': 'acc',
            'auc_roc': 'auc'
        }
        for key, short_key in history_keys.items():
            if key in train_metrics:
                training_history[f'train_{short_key}'].append(train_metrics[key])
            if key in val_metrics:
                training_history[f'val_{short_key}'].append(val_metrics[key])

        # Save best model (use Average Precision for imbalanced data)
        val_ap = val_metrics.get('avg_precision', 0)
        val_auc = val_metrics.get('auc_roc', 0)
        # Use AP as primary metric if available, fall back to AUC
        val_score = val_ap if val_ap > 0 else val_auc
        if val_score > best_val_auc:
            best_val_auc = val_score
            best_model_path = experiment_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, epoch, val_metrics, best_model_path)
            metric_name = "AP" if val_ap > 0 else "AUC"
            print(f"New best model! {metric_name}: {best_val_auc:.4f}")

        # Save latest checkpoint
        latest_path = experiment_dir / 'latest_checkpoint.pth'
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
            'best_val_auc': best_val_auc
        }
        torch.save(checkpoint_dict, latest_path)

        # Early stopping (use same metric as model selection)
        if early_stopping(val_score):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    best_model_path = experiment_dir / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    # Evaluate on test set
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    print_metrics(test_metrics, "Test")

    # Save results
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy/torch types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results = {
        'args': vars(args),
        'best_val_auc': float(best_val_auc),
        'test_metrics': convert_to_native(test_metrics),
        'training_history': convert_to_native(training_history)
    }

    results_path = experiment_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Close tensorboard writer
    writer.close()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Best Val Score: {best_val_auc:.4f}")
    print(f"Test AP: {test_metrics.get('avg_precision', 0):.4f}")
    print(f"Test AUC: {test_metrics.get('auc_roc', 0):.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics.get('f1', 0):.4f}")
    print(f"Models saved in: {experiment_dir}")


if __name__ == "__main__":
    main()