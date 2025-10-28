#!/usr/bin/env python3
"""
Balanced training script for corner kick GNN with class imbalance fixes.
Implements: weighted loss, balanced sampling, focal loss, proper metrics.
"""
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_corner_dataset
from src.gnn_model import CornerGNN
from src.focal_loss import WeightedFocalLoss, FocalLoss
from src.balanced_sampler import BalancedBatchSampler
from src.balanced_metrics import evaluate_with_multiple_thresholds, print_metrics_comparison, calculate_class_metrics
from torch_geometric.loader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN with balanced techniques')

    # Data arguments
    parser.add_argument('--use-smote', action='store_true', default=False,
                        help='Use SMOTE-augmented dataset')
    parser.add_argument('--outcome-type', type=str, default='shot',
                        choices=['goal', 'shot', 'dangerous'],
                        help='Outcome type to predict')

    # Loss function arguments
    parser.add_argument('--loss-type', type=str, default='focal',
                        choices=['bce', 'weighted_bce', 'focal'],
                        help='Loss function type')
    parser.add_argument('--pos-weight', type=float, default=6.0,
                        help='Weight for positive class')

    # Sampling arguments
    parser.add_argument('--use-balanced-sampling', action='store_true', default=True,
                        help='Use balanced batch sampling')

    # Model arguments
    parser.add_argument('--model-type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'graphsage'],
                        help='Type of GNN model')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--early-stopping', type=int, default=15,
                        help='Early stopping patience')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--threshold-optimize', action='store_true', default=True,
                        help='Optimize decision threshold')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_balanced(args):
    """
    Train GNN with balanced techniques.
    """
    set_seed(args.seed)

    # Create experiment directory
    experiment_name = f"balanced_{args.model_type}_{args.loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = Path('models') / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("BALANCED GNN TRAINING")
    print("="*60)
    print(f"Experiment: {experiment_name}")

    # Load data
    if args.use_smote:
        data_path = "data/graphs/adjacency_team/combined_temporal_graphs_smote.pkl"
    else:
        data_path = "data/graphs/adjacency_team/combined_temporal_graphs.pkl"

    print(f"\nLoading data from: {data_path}")
    dataset, train_loader, val_loader, test_loader = load_corner_dataset(
        graph_path=data_path,
        outcome_type=args.outcome_type,
        batch_size=args.batch_size
    )

    # Get data statistics
    total_samples = len(dataset.data_list)
    positive_samples = sum(1 for data in dataset.data_list if data.y.item() == 1)
    positive_rate = positive_samples / total_samples

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Positive samples: {positive_samples} ({positive_rate*100:.1f}%)")
    print(f"  Negative samples: {total_samples - positive_samples} ({(1-positive_rate)*100:.1f}%)")
    print(f"  Imbalance ratio: {(1-positive_rate)/positive_rate:.1f}:1")

    # Override with balanced sampling if requested
    if args.use_balanced_sampling:
        print("\nUsing balanced batch sampling...")

        # Get train data and labels
        train_indices = dataset.get_split_indices()['train']
        train_data = [dataset.data_list[i] for i in train_indices]
        train_labels = [data.y.item() for data in train_data]

        # Create balanced sampler
        balanced_sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=args.batch_size,
            oversample=True
        )

        # Create new train loader with balanced sampling
        train_loader = DataLoader(
            train_data,
            batch_sampler=balanced_sampler,
            num_workers=4,
            pin_memory=True
        )

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = CornerGNN(
        input_dim=dataset.num_features,
        hidden_dim1=args.hidden_dim,
        hidden_dim2=args.hidden_dim * 2,
        hidden_dim3=args.hidden_dim,
        fc_dim1=args.hidden_dim,
        fc_dim2=args.hidden_dim // 2,
        output_dim=1,
        dropout_rate=args.dropout,
        fc_dropout_rate=args.dropout
    ).to(device)

    print(f"\nModel: {args.model_type.upper()}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize loss function
    if args.loss_type == 'focal':
        print(f"\nUsing Focal Loss with pos_weight={args.pos_weight}")
        criterion = WeightedFocalLoss(pos_weight=args.pos_weight, gamma=2.0)
    elif args.loss_type == 'weighted_bce':
        print(f"\nUsing Weighted BCE with pos_weight={args.pos_weight}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))
    else:
        print("\nUsing standard BCE Loss")
        criterion = nn.BCEWithLogitsLoss()

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': []}

    print("\nStarting training...")
    print("-"*60)

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False):
            batch = batch.to(device)

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.squeeze(), batch.y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(out).cpu().detach().numpy())
            train_labels.extend(batch.y.cpu().numpy())

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y)

                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(out).cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())

        # Calculate metrics with threshold optimization
        train_metrics = evaluate_with_multiple_thresholds(
            np.array(train_labels), np.array(train_preds)
        )
        val_metrics = evaluate_with_multiple_thresholds(
            np.array(val_labels), np.array(val_preds)
        )

        # Use F1-optimized threshold metrics
        train_f1 = train_metrics['optimal_f1']['f1']
        val_f1 = val_metrics['optimal_f1']['f1']
        val_recall = val_metrics['optimal_f1']['recall']
        val_precision = val_metrics['optimal_f1']['precision']

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'optimal_threshold': val_metrics['optimal_f1']['threshold'],
                'args': vars(args)
            }, experiment_dir / 'best_model.pth')
        else:
            patience_counter += 1

        # Print progress every 10 epochs or when best model found
        if epoch % 10 == 0 or patience_counter == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"  Train F1 (optimized): {train_f1:.4f}")
            print(f"  Val F1 (optimized): {val_f1:.4f}")
            print(f"  Val Recall: {val_recall:.4f}, Precision: {val_precision:.4f}")
            print(f"  Best Val F1: {best_val_f1:.4f} (epoch {best_epoch+1})")
            print(f"  Optimal threshold: {val_metrics['optimal_f1']['threshold']:.3f}")

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)

    # Load best model
    checkpoint = torch.load(experiment_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimal_threshold = checkpoint['optimal_threshold']

    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    print(f"Using optimal threshold: {optimal_threshold:.3f}")

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            test_preds.extend(torch.sigmoid(out).cpu().numpy())
            test_labels.extend(batch.y.cpu().numpy())

    # Evaluate with multiple thresholds
    test_metrics = evaluate_with_multiple_thresholds(
        np.array(test_labels), np.array(test_preds)
    )

    print_metrics_comparison(test_metrics)

    # Save results
    results = {
        'experiment_name': experiment_name,
        'args': vars(args),
        'dataset_stats': {
            'total_samples': total_samples,
            'positive_samples': positive_samples,
            'positive_rate': positive_rate
        },
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch + 1,
        'optimal_threshold': optimal_threshold,
        'test_metrics': test_metrics,
        'history': {
            'train': [m for m in history['train']],  # Convert to serializable format
            'val': [m for m in history['val']]
        }
    }

    # Save with proper JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    # Recursively convert numpy types
    results = json.loads(json.dumps(results, default=convert_to_native))

    with open(experiment_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Training complete! Results saved to {experiment_dir}")

    # Print final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Test Performance (with optimal threshold):")
    optimal_test = test_metrics['optimal_f1']
    print(f"  F1 Score: {optimal_test['f1']:.4f}")
    print(f"  Recall: {optimal_test['recall']:.4f}")
    print(f"  Precision: {optimal_test['precision']:.4f}")
    print(f"  Balanced Accuracy: {optimal_test['balanced_acc']:.4f}")

    return results


if __name__ == "__main__":
    args = parse_args()
    train_balanced(args)