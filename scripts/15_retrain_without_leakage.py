#!/usr/bin/env python3
"""
Retrain models using only temporally valid features (no data leakage).

This script trains multiple ML models on corner kick data using ONLY
features available at the time of the corner kick, removing all
temporal leakage that inflated previous results.

Uses pre-computed 60/20/20 match-based splits to prevent data leakage
from same-match corners appearing in different sets.

Trains two types of models:
1. Binary shot prediction (did the corner lead to a shot?)
2. Multi-class outcome prediction (Ball Receipt, Clearance, Other, Goalkeeper)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def load_data_with_splits(data_path: Path, splits_dir: Path):
    """
    Load data and apply pre-computed match-based splits.

    Args:
        data_path: Path to CSV with temporally valid features
        splits_dir: Directory containing train/val/test index files

    Returns:
        df, train_idx, val_idx, test_idx
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Load pre-computed indices
    train_idx = pd.read_csv(splits_dir / 'train_indices.csv')['index'].values
    val_idx = pd.read_csv(splits_dir / 'val_indices.csv')['index'].values
    test_idx = pd.read_csv(splits_dir / 'test_indices.csv')['index'].values

    print(f"Dataset: {len(df)} samples")
    print(f"Train: {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")

    # Verify no overlap
    train_matches = set(df.iloc[train_idx]['match_id'].unique())
    val_matches = set(df.iloc[val_idx]['match_id'].unique())
    test_matches = set(df.iloc[test_idx]['match_id'].unique())

    assert len(train_matches & val_matches) == 0, "Train-Val overlap!"
    assert len(train_matches & test_matches) == 0, "Train-Test overlap!"
    assert len(val_matches & test_matches) == 0, "Val-Test overlap!"
    print("Match-based splits verified (no overlap)")

    return df, train_idx, val_idx, test_idx


def prepare_features(df: pd.DataFrame, train_idx, val_idx, test_idx):
    """
    Prepare feature matrices and targets for both tasks.

    Returns:
        Dictionary with X_train, X_val, X_test, y_train_shot, etc.
    """
    exclude_cols = ['match_id', 'event_id', 'outcome', 'shot']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Extract features
    X_train = df.iloc[train_idx][feature_cols].values
    X_val = df.iloc[val_idx][feature_cols].values
    X_test = df.iloc[test_idx][feature_cols].values

    # Binary shot targets
    y_train_shot = df.iloc[train_idx]['shot'].values
    y_val_shot = df.iloc[val_idx]['shot'].values
    y_test_shot = df.iloc[test_idx]['shot'].values

    # Multi-class outcome targets
    label_encoder = LabelEncoder()
    all_outcomes = df['outcome'].values
    label_encoder.fit(all_outcomes)

    y_train_outcome = label_encoder.transform(df.iloc[train_idx]['outcome'].values)
    y_val_outcome = label_encoder.transform(df.iloc[val_idx]['outcome'].values)
    y_test_outcome = label_encoder.transform(df.iloc[test_idx]['outcome'].values)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Shot distribution - Train: {y_train_shot.mean()*100:.1f}% positive")
    print(f"Outcome classes: {label_encoder.classes_}")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train_shot': y_train_shot,
        'y_val_shot': y_val_shot,
        'y_test_shot': y_test_shot,
        'y_train_outcome': y_train_outcome,
        'y_val_outcome': y_val_outcome,
        'y_test_outcome': y_test_outcome,
        'feature_cols': feature_cols,
        'label_encoder': label_encoder,
        'scaler': scaler
    }


def train_binary_models(data: dict):
    """
    Train models for binary shot prediction.

    Returns:
        Dictionary of results for each model
    """
    print("\n" + "="*70)
    print("BINARY SHOT PREDICTION")
    print("="*70)

    results = {}

    # Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(data['X_train_scaled'], data['y_train_shot'])

    rf_train_pred = rf.predict(data['X_train_scaled'])
    rf_val_pred = rf.predict(data['X_val_scaled'])
    rf_test_pred = rf.predict(data['X_test_scaled'])
    rf_val_prob = rf.predict_proba(data['X_val_scaled'])[:, 1]
    rf_test_prob = rf.predict_proba(data['X_test_scaled'])[:, 1]

    results['RandomForest'] = {
        'train_acc': accuracy_score(data['y_train_shot'], rf_train_pred),
        'val_acc': accuracy_score(data['y_val_shot'], rf_val_pred),
        'test_acc': accuracy_score(data['y_test_shot'], rf_test_pred),
        'val_auc': roc_auc_score(data['y_val_shot'], rf_val_prob),
        'test_auc': roc_auc_score(data['y_test_shot'], rf_test_prob),
        'model': rf
    }
    print(f"Train Acc: {results['RandomForest']['train_acc']*100:.2f}%")
    print(f"Val Acc: {results['RandomForest']['val_acc']*100:.2f}%  AUC: {results['RandomForest']['val_auc']:.4f}")
    print(f"Test Acc: {results['RandomForest']['test_acc']*100:.2f}%  AUC: {results['RandomForest']['test_auc']:.4f}")

    # XGBoost
    print("\n--- XGBoost ---")
    scale_pos_weight = len(data['y_train_shot'][data['y_train_shot']==0]) / max(1, len(data['y_train_shot'][data['y_train_shot']==1]))
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    xgb_model.fit(data['X_train'], data['y_train_shot'])

    xgb_train_pred = xgb_model.predict(data['X_train'])
    xgb_val_pred = xgb_model.predict(data['X_val'])
    xgb_test_pred = xgb_model.predict(data['X_test'])
    xgb_val_prob = xgb_model.predict_proba(data['X_val'])[:, 1]
    xgb_test_prob = xgb_model.predict_proba(data['X_test'])[:, 1]

    results['XGBoost'] = {
        'train_acc': accuracy_score(data['y_train_shot'], xgb_train_pred),
        'val_acc': accuracy_score(data['y_val_shot'], xgb_val_pred),
        'test_acc': accuracy_score(data['y_test_shot'], xgb_test_pred),
        'val_auc': roc_auc_score(data['y_val_shot'], xgb_val_prob),
        'test_auc': roc_auc_score(data['y_test_shot'], xgb_test_prob),
        'model': xgb_model
    }
    print(f"Train Acc: {results['XGBoost']['train_acc']*100:.2f}%")
    print(f"Val Acc: {results['XGBoost']['val_acc']*100:.2f}%  AUC: {results['XGBoost']['val_auc']:.4f}")
    print(f"Test Acc: {results['XGBoost']['test_acc']*100:.2f}%  AUC: {results['XGBoost']['test_auc']:.4f}")

    # MLP
    print("\n--- MLP ---")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(data['X_train_scaled'], data['y_train_shot'])

    mlp_train_pred = mlp.predict(data['X_train_scaled'])
    mlp_val_pred = mlp.predict(data['X_val_scaled'])
    mlp_test_pred = mlp.predict(data['X_test_scaled'])
    mlp_val_prob = mlp.predict_proba(data['X_val_scaled'])[:, 1]
    mlp_test_prob = mlp.predict_proba(data['X_test_scaled'])[:, 1]

    results['MLP'] = {
        'train_acc': accuracy_score(data['y_train_shot'], mlp_train_pred),
        'val_acc': accuracy_score(data['y_val_shot'], mlp_val_pred),
        'test_acc': accuracy_score(data['y_test_shot'], mlp_test_pred),
        'val_auc': roc_auc_score(data['y_val_shot'], mlp_val_prob),
        'test_auc': roc_auc_score(data['y_test_shot'], mlp_test_prob),
        'model': mlp
    }
    print(f"Train Acc: {results['MLP']['train_acc']*100:.2f}%")
    print(f"Val Acc: {results['MLP']['val_acc']*100:.2f}%  AUC: {results['MLP']['val_auc']:.4f}")
    print(f"Test Acc: {results['MLP']['test_acc']*100:.2f}%  AUC: {results['MLP']['test_auc']:.4f}")

    return results


def train_multiclass_models(data: dict):
    """
    Train models for multi-class outcome prediction.

    Returns:
        Dictionary of results for each model
    """
    print("\n" + "="*70)
    print("MULTI-CLASS OUTCOME PREDICTION")
    print("="*70)

    n_classes = len(data['label_encoder'].classes_)
    print(f"Classes: {list(data['label_encoder'].classes_)}")

    results = {}

    # Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(data['X_train_scaled'], data['y_train_outcome'])

    rf_train_pred = rf.predict(data['X_train_scaled'])
    rf_val_pred = rf.predict(data['X_val_scaled'])
    rf_test_pred = rf.predict(data['X_test_scaled'])
    rf_val_prob = rf.predict_proba(data['X_val_scaled'])
    rf_test_prob = rf.predict_proba(data['X_test_scaled'])

    results['RandomForest'] = {
        'train_acc': accuracy_score(data['y_train_outcome'], rf_train_pred),
        'val_acc': accuracy_score(data['y_val_outcome'], rf_val_pred),
        'test_acc': accuracy_score(data['y_test_outcome'], rf_test_pred),
        'val_f1_macro': f1_score(data['y_val_outcome'], rf_val_pred, average='macro'),
        'test_f1_macro': f1_score(data['y_test_outcome'], rf_test_pred, average='macro'),
        'val_auc_ovr': roc_auc_score(data['y_val_outcome'], rf_val_prob, multi_class='ovr'),
        'test_auc_ovr': roc_auc_score(data['y_test_outcome'], rf_test_prob, multi_class='ovr'),
        'model': rf
    }
    print(f"Train Acc: {results['RandomForest']['train_acc']*100:.2f}%")
    print(f"Val Acc: {results['RandomForest']['val_acc']*100:.2f}%  F1: {results['RandomForest']['val_f1_macro']:.4f}  AUC: {results['RandomForest']['val_auc_ovr']:.4f}")
    print(f"Test Acc: {results['RandomForest']['test_acc']*100:.2f}%  F1: {results['RandomForest']['test_f1_macro']:.4f}  AUC: {results['RandomForest']['test_auc_ovr']:.4f}")

    # XGBoost
    print("\n--- XGBoost ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softprob',
        num_class=n_classes
    )
    xgb_model.fit(data['X_train'], data['y_train_outcome'])

    xgb_train_pred = xgb_model.predict(data['X_train'])
    xgb_val_pred = xgb_model.predict(data['X_val'])
    xgb_test_pred = xgb_model.predict(data['X_test'])
    xgb_val_prob = xgb_model.predict_proba(data['X_val'])
    xgb_test_prob = xgb_model.predict_proba(data['X_test'])

    results['XGBoost'] = {
        'train_acc': accuracy_score(data['y_train_outcome'], xgb_train_pred),
        'val_acc': accuracy_score(data['y_val_outcome'], xgb_val_pred),
        'test_acc': accuracy_score(data['y_test_outcome'], xgb_test_pred),
        'val_f1_macro': f1_score(data['y_val_outcome'], xgb_val_pred, average='macro'),
        'test_f1_macro': f1_score(data['y_test_outcome'], xgb_test_pred, average='macro'),
        'val_auc_ovr': roc_auc_score(data['y_val_outcome'], xgb_val_prob, multi_class='ovr'),
        'test_auc_ovr': roc_auc_score(data['y_test_outcome'], xgb_test_prob, multi_class='ovr'),
        'model': xgb_model
    }
    print(f"Train Acc: {results['XGBoost']['train_acc']*100:.2f}%")
    print(f"Val Acc: {results['XGBoost']['val_acc']*100:.2f}%  F1: {results['XGBoost']['val_f1_macro']:.4f}  AUC: {results['XGBoost']['val_auc_ovr']:.4f}")
    print(f"Test Acc: {results['XGBoost']['test_acc']*100:.2f}%  F1: {results['XGBoost']['test_f1_macro']:.4f}  AUC: {results['XGBoost']['test_auc_ovr']:.4f}")

    # MLP
    print("\n--- MLP ---")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=1000,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(data['X_train_scaled'], data['y_train_outcome'])

    mlp_train_pred = mlp.predict(data['X_train_scaled'])
    mlp_val_pred = mlp.predict(data['X_val_scaled'])
    mlp_test_pred = mlp.predict(data['X_test_scaled'])
    mlp_val_prob = mlp.predict_proba(data['X_val_scaled'])
    mlp_test_prob = mlp.predict_proba(data['X_test_scaled'])

    results['MLP'] = {
        'train_acc': accuracy_score(data['y_train_outcome'], mlp_train_pred),
        'val_acc': accuracy_score(data['y_val_outcome'], mlp_val_pred),
        'test_acc': accuracy_score(data['y_test_outcome'], mlp_test_pred),
        'val_f1_macro': f1_score(data['y_val_outcome'], mlp_val_pred, average='macro'),
        'test_f1_macro': f1_score(data['y_test_outcome'], mlp_test_pred, average='macro'),
        'val_auc_ovr': roc_auc_score(data['y_val_outcome'], mlp_val_prob, multi_class='ovr'),
        'test_auc_ovr': roc_auc_score(data['y_test_outcome'], mlp_test_prob, multi_class='ovr'),
        'model': mlp
    }
    print(f"Train Acc: {results['MLP']['train_acc']*100:.2f}%")
    print(f"Val Acc: {results['MLP']['val_acc']*100:.2f}%  F1: {results['MLP']['val_f1_macro']:.4f}  AUC: {results['MLP']['val_auc_ovr']:.4f}")
    print(f"Test Acc: {results['MLP']['test_acc']*100:.2f}%  F1: {results['MLP']['test_f1_macro']:.4f}  AUC: {results['MLP']['test_auc_ovr']:.4f}")

    # Print detailed classification report for best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['val_acc'])
    best_model = results[best_model_name]['model']

    print(f"\n--- Classification Report ({best_model_name}) ---")
    if best_model_name == 'XGBoost':
        test_pred = best_model.predict(data['X_test'])
    else:
        test_pred = best_model.predict(data['X_test_scaled'])

    print(classification_report(
        data['y_test_outcome'],
        test_pred,
        target_names=data['label_encoder'].classes_
    ))

    return results


def print_summary(binary_results: dict, multiclass_results: dict, data: dict):
    """Print summary of all results."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\n### Binary Shot Prediction ###")
    print(f"{'Model':<15} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Val AUC':>10} {'Test AUC':>10}")
    print("-"*70)
    for name, r in binary_results.items():
        print(f"{name:<15} {r['train_acc']*100:>9.2f}% {r['val_acc']*100:>9.2f}% {r['test_acc']*100:>9.2f}% {r['val_auc']:>10.4f} {r['test_auc']:>10.4f}")

    # Baseline for binary
    baseline_acc = 1 - data['y_test_shot'].mean()
    print(f"{'Baseline':<15} {'-':>10} {'-':>10} {baseline_acc*100:>9.2f}% {'-':>10} {'0.5000':>10}")

    print("\n### Multi-Class Outcome Prediction ###")
    print(f"{'Model':<15} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Val F1':>10} {'Test F1':>10}")
    print("-"*70)
    for name, r in multiclass_results.items():
        print(f"{name:<15} {r['train_acc']*100:>9.2f}% {r['val_acc']*100:>9.2f}% {r['test_acc']*100:>9.2f}% {r['val_f1_macro']:>10.4f} {r['test_f1_macro']:>10.4f}")

    # Baseline for multiclass (majority class)
    from collections import Counter
    majority_class = Counter(data['y_test_outcome']).most_common(1)[0][0]
    baseline_acc = (data['y_test_outcome'] == majority_class).mean()
    print(f"{'Baseline':<15} {'-':>10} {'-':>10} {baseline_acc*100:>9.2f}% {'-':>10} {'-':>10}")


def main():
    """Main execution function."""
    base_dir = Path('/home/mseo/CornerTactics')
    data_path = base_dir / 'data/processed/corners_features_temporal_valid.csv'
    splits_dir = base_dir / 'data/processed'
    results_dir = base_dir / 'results/no_leakage'
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please run 14_extract_temporally_valid_features.py first")
        return

    # Load data with match-based splits
    df, train_idx, val_idx, test_idx = load_data_with_splits(data_path, splits_dir)

    # Prepare features
    data = prepare_features(df, train_idx, val_idx, test_idx)

    # Train binary shot prediction models
    binary_results = train_binary_models(data)

    # Train multi-class outcome prediction models
    multiclass_results = train_multiclass_models(data)

    # Print summary
    print_summary(binary_results, multiclass_results, data)

    # Save results
    results_summary = {
        'binary_shot_prediction': {
            name: {k: v for k, v in r.items() if k != 'model'}
            for name, r in binary_results.items()
        },
        'multiclass_outcome_prediction': {
            name: {k: v for k, v in r.items() if k != 'model'}
            for name, r in multiclass_results.items()
        },
        'split_sizes': {
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx)
        },
        'num_features': len(data['feature_cols']),
        'features_used': data['feature_cols'],
        'outcome_classes': list(data['label_encoder'].classes_)
    }

    results_path = results_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
