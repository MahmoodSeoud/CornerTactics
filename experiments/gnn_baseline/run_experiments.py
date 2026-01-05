#!/usr/bin/env python3
"""Run GNN baseline experiments on StatsBomb data.

This script trains and evaluates GNN models for corner kick outcome prediction.
Results are reported with 95% confidence intervals and p-values.

Usage:
    python experiments/gnn_baseline/run_experiments.py
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

from experiments.gnn_baseline.train import Trainer, set_seed
from experiments.gnn_baseline.evaluate import evaluate_model, format_results


# Default paths
DATA_PATH = Path("data/processed/corners_with_shot_labels.json")
TRAIN_INDICES_PATH = Path("data/processed/train_indices.csv")
VAL_INDICES_PATH = Path("data/processed/val_indices.csv")
TEST_INDICES_PATH = Path("data/processed/test_indices.csv")
RESULTS_DIR = Path("experiments/gnn_baseline/results")


def load_data(data_path: Path):
    """Load corner data and split indices."""
    with open(data_path, 'r') as f:
        corners = json.load(f)

    train_df = pd.read_csv(TRAIN_INDICES_PATH)
    val_df = pd.read_csv(VAL_INDICES_PATH)
    test_df = pd.read_csv(TEST_INDICES_PATH)

    train_indices = train_df['index'].tolist()
    val_indices = val_df['index'].tolist()
    test_indices = test_df['index'].tolist()

    return corners, train_indices, val_indices, test_indices


def run_experiment(
    corners,
    train_indices,
    val_indices,
    test_indices,
    model_name: str,
    edge_type: str = 'knn',
    k: int = 5,
    hidden_channels: int = 64,
    num_layers: int = 2,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 15,
    seed: int = 42,
    verbose: bool = True,
):
    """Run a single experiment configuration."""
    set_seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {model_name.upper()}")
        print(f"Edge type: {edge_type}, k={k}")
        print(f"Hidden: {hidden_channels}, Layers: {num_layers}")
        print(f"{'='*60}")

    trainer = Trainer(
        corners=corners,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        model_name=model_name,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        batch_size=batch_size,
        patience=patience,
        edge_type=edge_type,
        k=k,
        use_class_weights=True,
    )

    history = trainer.fit(epochs=epochs, verbose=verbose)

    # Evaluate on test set
    results = evaluate_model(trainer, n_bootstrap=1000, n_permutations=1000)

    # Add training info
    results['model'] = model_name
    results['edge_type'] = edge_type
    results['k'] = k
    results['hidden_channels'] = hidden_channels
    results['num_layers'] = num_layers
    results['epochs_trained'] = len(history['train_loss'])
    results['best_val_auc'] = max(history['val_auc'])

    return results, history


def run_all_experiments(corners, train_indices, val_indices, test_indices, args):
    """Run all experiment configurations."""
    all_results = []

    # Model configurations
    models = ['graphsage', 'gat']  # Skip MPNN for now (slower)
    edge_types = ['knn', 'full']
    k_values = [3, 5, 7]

    # Main experiments: Different models with k-NN edges
    for model_name in models:
        for k in k_values:
            results, _ = run_experiment(
                corners, train_indices, val_indices, test_indices,
                model_name=model_name,
                edge_type='knn',
                k=k,
                hidden_channels=args.hidden,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                verbose=args.verbose,
            )
            all_results.append(results)

    # Full connectivity ablation
    for model_name in models:
        results, _ = run_experiment(
            corners, train_indices, val_indices, test_indices,
            model_name=model_name,
            edge_type='full',
            k=0,
            hidden_channels=args.hidden,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            verbose=args.verbose,
        )
        all_results.append(results)

    return all_results


def save_results(results, output_dir):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save as CSV
    csv_path = output_dir / f"results_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    summary_cols = ['model', 'edge_type', 'k', 'test_auc', 'test_auc_ci_lower',
                    'test_auc_ci_upper', 'test_p_value', 'test_is_significant']
    summary_df = df[summary_cols].copy()
    summary_df['test_auc'] = summary_df['test_auc'].apply(lambda x: f"{x:.3f}")
    summary_df['CI'] = summary_df.apply(
        lambda r: f"({r['test_auc_ci_lower']:.3f}-{r['test_auc_ci_upper']:.3f})", axis=1
    )
    summary_df['test_p_value'] = summary_df['test_p_value'].apply(lambda x: f"{x:.4f}")

    print(summary_df[['model', 'edge_type', 'k', 'test_auc', 'CI', 'test_p_value', 'test_is_significant']].to_string(index=False))

    # Best result
    best_idx = df['test_auc'].idxmax()
    best = df.loc[best_idx]
    print("\n" + "-"*80)
    print("BEST RESULT:")
    print(format_results({
        'auc': best['test_auc'],
        'auc_ci_lower': best['test_auc_ci_lower'],
        'auc_ci_upper': best['test_auc_ci_upper'],
        'p_value': best['test_p_value'],
        'is_significant': best['test_is_significant'],
    }, model_name=f"{best['model']} ({best['edge_type']}, k={best['k']})"))

    print(f"\nResults saved to: {json_path}")
    print(f"Results saved to: {csv_path}")

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Run GNN baseline experiments")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to corners data")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channels")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer configs)")
    args = parser.parse_args()

    print("Loading data...")
    corners, train_indices, val_indices, test_indices = load_data(args.data)
    print(f"Loaded {len(corners)} corners")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Count class distribution
    train_labels = [corners[i]['shot_outcome'] for i in train_indices]
    shot_count = sum(train_labels)
    print(f"Training set: {shot_count} shots ({100*shot_count/len(train_labels):.1f}%), "
          f"{len(train_labels)-shot_count} no-shots")

    if args.quick:
        # Quick test with single configuration
        print("\nRunning quick test...")
        results, _ = run_experiment(
            corners, train_indices, val_indices, test_indices,
            model_name='graphsage',
            edge_type='knn',
            k=5,
            hidden_channels=32,
            epochs=10,
            patience=5,
            seed=args.seed,
            verbose=args.verbose,
        )
        print("\n" + format_results({
            'auc': results['test_auc'],
            'auc_ci_lower': results['test_auc_ci_lower'],
            'auc_ci_upper': results['test_auc_ci_upper'],
            'p_value': results['test_p_value'],
            'is_significant': results['test_is_significant'],
        }, model_name='GraphSAGE'))
    else:
        # Full experiments
        print("\nRunning full experiments...")
        results = run_all_experiments(corners, train_indices, val_indices, test_indices, args)
        save_results(results, args.output)


if __name__ == "__main__":
    main()
