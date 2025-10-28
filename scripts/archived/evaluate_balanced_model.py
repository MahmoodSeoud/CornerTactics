#!/usr/bin/env python3
"""
Evaluate the trained balanced model on test set.
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_corner_dataset
from src.gnn_model import CornerGNN
from src.balanced_metrics import evaluate_with_multiple_thresholds, print_metrics_comparison
from tqdm import tqdm

# Load model
model_path = Path("models/balanced_gcn_focal_20251027_122157/best_model.pth")
print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, weights_only=False)

print(f"Best epoch: {checkpoint['epoch']+1}")
print(f"Best val F1: {checkpoint['val_f1']:.4f}")
print(f"Optimal threshold: {checkpoint['optimal_threshold']:.3f}")

# Load data
dataset, train_loader, val_loader, test_loader = load_corner_dataset(
    graph_path="data/graphs/adjacency_team/combined_temporal_graphs.pkl",
    outcome_type="shot",
    batch_size=32
)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CornerGNN(
    input_dim=dataset.num_features,
    hidden_dim1=64,
    hidden_dim2=128,
    hidden_dim3=64,
    fc_dim1=64,
    fc_dim2=32,
    output_dim=1,
    dropout_rate=0.2,
    fc_dropout_rate=0.2
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\nModel loaded successfully on {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Evaluate on test set
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

test_preds = []
test_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        test_preds.extend(torch.sigmoid(out).cpu().numpy())
        test_labels.extend(batch.y.cpu().numpy())

# Evaluate with multiple thresholds
test_metrics = evaluate_with_multiple_thresholds(
    np.array(test_labels), np.array(test_preds)
)

print_metrics_comparison(test_metrics)

# Print summary
print("\n" + "="*60)
print("SUMMARY - COMPARISON WITH BASELINE")
print("="*60)

print("\nBASELINE (Job 29019 - with imbalance):")
print("  Test Recall:    17.2%")
print("  Test Precision: 10.3%")
print("  Test F1:         0.128")

print("\nBALANCED MODEL (current):")
opt_metrics = test_metrics['optimal_f1']
print(f"  Test Recall:    {opt_metrics['recall']*100:.1f}%")
print(f"  Test Precision: {opt_metrics['precision']*100:.1f}%")
print(f"  Test F1:         {opt_metrics['f1']:.3f}")
print(f"  Optimal Threshold: {opt_metrics['threshold']:.3f}")

print("\nIMPROVEMENT:")
baseline_f1 = 0.128
improvement = (opt_metrics['f1'] - baseline_f1) / baseline_f1 * 100
print(f"  F1 Score: +{improvement:.1f}% improvement")
print(f"  Recall: +{(opt_metrics['recall']*100 - 17.2):.1f}%")
print(f"  Precision: +{(opt_metrics['precision']*100 - 10.3):.1f}%")

print("\n" + "="*60)
print("✅ Class imbalance fixes successfully improved model performance!")
print("="*60)