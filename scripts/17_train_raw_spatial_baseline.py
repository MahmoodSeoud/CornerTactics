#!/usr/bin/env python3
"""
Train and compare models using different feature sets.

Compares:
1. Original 22 aggregate features (current approach)
2. Raw coordinates only (46 features)
3. Pairwise distances only (18 features)
4. Spatial structure only (10 features)
5. All raw spatial features combined (74 features)

Author: CornerTactics Team
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def load_data_with_splits(data_path: Path, splits_dir: Path):
    """Load data and apply pre-computed match-based splits."""
    df = pd.read_csv(data_path)

    # Load indices
    train_idx = pd.read_csv(splits_dir / 'train_indices.csv')['index'].values
    val_idx = pd.read_csv(splits_dir / 'val_indices.csv')['index'].values
    test_idx = pd.read_csv(splits_dir / 'test_indices.csv')['index'].values

    return df, train_idx, val_idx, test_idx


def prepare_features(df: pd.DataFrame, feature_cols: list,
                     train_idx, val_idx, test_idx):
    """Prepare feature matrices for training."""
    # Handle missing values
    X = df[feature_cols].fillna(-1)

    X_train = X.iloc[train_idx].values
    X_val = X.iloc[val_idx].values
    X_test = X.iloc[test_idx].values

    y_train = df.iloc[train_idx]['shot'].values
    y_val = df.iloc[val_idx]['shot'].values
    y_test = df.iloc[test_idx]['shot'].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler
    }


def train_and_evaluate(data: dict, feature_set_name: str):
    """Train RF, XGBoost, MLP and return results."""
    results = {}

    # Class weight for imbalance
    scale_pos_weight = len(data['y_train'][data['y_train']==0]) / max(1, len(data['y_train'][data['y_train']==1]))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(data['X_train_scaled'], data['y_train'])

    rf_val_prob = rf.predict_proba(data['X_val_scaled'])[:, 1]
    rf_test_prob = rf.predict_proba(data['X_test_scaled'])[:, 1]
    rf_test_pred = rf.predict(data['X_test_scaled'])

    results['RandomForest'] = {
        'val_auc': roc_auc_score(data['y_val'], rf_val_prob),
        'test_auc': roc_auc_score(data['y_test'], rf_test_prob),
        'test_acc': accuracy_score(data['y_test'], rf_test_pred),
        'test_f1': f1_score(data['y_test'], rf_test_pred)
    }

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(data['X_train'], data['y_train'])

    xgb_val_prob = xgb_model.predict_proba(data['X_val'])[:, 1]
    xgb_test_prob = xgb_model.predict_proba(data['X_test'])[:, 1]
    xgb_test_pred = xgb_model.predict(data['X_test'])

    results['XGBoost'] = {
        'val_auc': roc_auc_score(data['y_val'], xgb_val_prob),
        'test_auc': roc_auc_score(data['y_test'], xgb_test_prob),
        'test_acc': accuracy_score(data['y_test'], xgb_test_pred),
        'test_f1': f1_score(data['y_test'], xgb_test_pred)
    }

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(data['X_train_scaled'], data['y_train'])

    mlp_val_prob = mlp.predict_proba(data['X_val_scaled'])[:, 1]
    mlp_test_prob = mlp.predict_proba(data['X_test_scaled'])[:, 1]
    mlp_test_pred = mlp.predict(data['X_test_scaled'])

    results['MLP'] = {
        'val_auc': roc_auc_score(data['y_val'], mlp_val_prob),
        'test_auc': roc_auc_score(data['y_test'], mlp_test_prob),
        'test_acc': accuracy_score(data['y_test'], mlp_test_pred),
        'test_f1': f1_score(data['y_test'], mlp_test_pred)
    }

    return results


def print_comparison_table(all_results: dict):
    """Print formatted comparison table."""
    print("\n" + "="*90)
    print("FEATURE SET COMPARISON - Binary Shot Prediction")
    print("="*90)

    print(f"\n{'Feature Set':<25} {'#Feat':>6} {'Model':<12} {'Val AUC':>10} {'Test AUC':>10} {'Test Acc':>10} {'Test F1':>10}")
    print("-"*90)

    for feature_set, data in all_results.items():
        n_features = data['n_features']
        for i, (model, metrics) in enumerate(data['results'].items()):
            if i == 0:
                print(f"{feature_set:<25} {n_features:>6} {model:<12} {metrics['val_auc']:>10.4f} {metrics['test_auc']:>10.4f} {metrics['test_acc']*100:>9.1f}% {metrics['test_f1']:>10.4f}")
            else:
                print(f"{'':<25} {'':<6} {model:<12} {metrics['val_auc']:>10.4f} {metrics['test_auc']:>10.4f} {metrics['test_acc']*100:>9.1f}% {metrics['test_f1']:>10.4f}")
        print()

    # Baseline
    baseline_acc = 71.0  # Always predict "no shot"
    print(f"{'Baseline (majority)':<25} {'-':>6} {'-':<12} {'0.5000':>10} {'0.5000':>10} {baseline_acc:>9.1f}% {'-':>10}")
    print("="*90)


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    results_dir = base_dir / 'results' / 'raw_spatial_baseline'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    # Load raw spatial features
    df_raw = pd.read_csv(data_dir / 'corners_raw_spatial_features.csv')

    # Load original aggregate features
    df_agg = pd.read_csv(data_dir / 'corners_features_temporal_valid.csv')

    # Load splits
    train_idx = pd.read_csv(data_dir / 'train_indices.csv')['index'].values
    val_idx = pd.read_csv(data_dir / 'val_indices.csv')['index'].values
    test_idx = pd.read_csv(data_dir / 'test_indices.csv')['index'].values

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Define feature sets
    raw_coord_cols = [c for c in df_raw.columns if
                      (c.startswith('att_') and (c.endswith('_x') or c.endswith('_y'))) or
                      (c.startswith('def_') and (c.endswith('_x') or c.endswith('_y'))) or
                      c.startswith('gk_')]
    raw_coord_cols = [c for c in raw_coord_cols if 'dist' not in c]

    pairwise_cols = [c for c in df_raw.columns if 'dist' in c or 'unmarked' in c]

    structure_cols = [c for c in df_raw.columns if 'spread' in c or 'range' in c or
                      'centroid' in c or c in ['n_attackers', 'n_defenders', 'has_goalkeeper']]

    all_raw_cols = raw_coord_cols + pairwise_cols + structure_cols

    aggregate_cols = [c for c in df_agg.columns if c not in
                      ['match_id', 'event_id', 'outcome', 'shot']]

    feature_sets = {
        '1. Aggregates (current)': (df_agg, aggregate_cols),
        '2. Raw Coordinates': (df_raw, raw_coord_cols),
        '3. Pairwise Distances': (df_raw, pairwise_cols),
        '4. Spatial Structure': (df_raw, structure_cols),
        '5. All Raw Spatial': (df_raw, all_raw_cols),
        '6. Aggregates + Raw': (None, None),  # Special case
    }

    all_results = {}

    for name, (df, cols) in feature_sets.items():
        if name == '6. Aggregates + Raw':
            # Merge aggregate and raw features
            df_combined = df_agg.merge(
                df_raw.drop(columns=['outcome', 'shot']),
                on=['match_id', 'event_id'],
                how='inner'
            )
            cols = aggregate_cols + all_raw_cols
            df = df_combined

        print(f"\n{'='*60}")
        print(f"Training: {name} ({len(cols)} features)")
        print(f"{'='*60}")

        data = prepare_features(df, cols, train_idx, val_idx, test_idx)
        results = train_and_evaluate(data, name)

        all_results[name] = {
            'n_features': len(cols),
            'features': cols,
            'results': results
        }

        # Print individual results
        for model, metrics in results.items():
            print(f"  {model}: Val AUC={metrics['val_auc']:.4f}, Test AUC={metrics['test_auc']:.4f}, Test Acc={metrics['test_acc']*100:.1f}%")

    # Print comparison table
    print_comparison_table(all_results)

    # Save results
    results_to_save = {}
    for name, data in all_results.items():
        results_to_save[name] = {
            'n_features': data['n_features'],
            'results': data['results']
        }

    with open(results_dir / 'comparison_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to {results_dir / 'comparison_results.json'}")

    # Summary interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    # Find best performing feature set
    best_auc = 0
    best_set = ""
    for name, data in all_results.items():
        for model, metrics in data['results'].items():
            if metrics['test_auc'] > best_auc:
                best_auc = metrics['test_auc']
                best_set = f"{name} + {model}"

    print(f"\nBest Test AUC: {best_auc:.4f} ({best_set})")

    agg_best = max(m['test_auc'] for m in all_results['1. Aggregates (current)']['results'].values())
    raw_best = max(m['test_auc'] for m in all_results['2. Raw Coordinates']['results'].values())

    if raw_best > agg_best + 0.02:
        print("\n→ Raw coordinates OUTPERFORM aggregates")
        print("  Aggregates lose information. Consider using raw coords or richer features.")
    elif agg_best > raw_best + 0.02:
        print("\n→ Aggregates OUTPERFORM raw coordinates")
        print("  Current approach is justified. Raw coords add noise.")
    else:
        print("\n→ Raw coordinates ≈ Aggregates (within 0.02 AUC)")
        print("  Both approaches capture similar signal. Aggregates are simpler.")


if __name__ == '__main__':
    main()
