#!/usr/bin/env python3
"""
Retrain models using only temporally valid features (no data leakage).

This script trains multiple ML models on corner kick data using ONLY
features available at the time of the corner kick, removing all
temporal leakage that inflated previous results.

Author: CornerTactics Team
Date: 2025-11-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def prepare_data(data_path):
    """
    Load and prepare data for training.

    Args:
        data_path: Path to CSV with temporally valid features

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Separate features and target
    exclude_cols = ['match_id', 'event_id', 'outcome', 'shot']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y = df['shot'].values

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {np.mean(y)*100:.1f}% positive")

    # Split data (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test, feature_cols


def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Train Random Forest classifier."""
    print("\n" + "="*60)
    print("TRAINING: Random Forest")
    print("="*60)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  # Reduced to prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_prob = model.predict_proba(X_train_scaled)[:, 1]
    test_prob = model.predict_proba(X_test_scaled)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    train_auc = roc_auc_score(y_train, train_prob)
    test_auc = roc_auc_score(y_test, test_prob)

    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    for idx, row in importance.head(10).iterrows():
        print(f"  {row['feature']:25s} {row['importance']:.4f}")

    return model, test_acc, test_auc


def train_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """Train XGBoost classifier."""
    print("\n" + "="*60)
    print("TRAINING: XGBoost")
    print("="*60)

    # No scaling needed for tree-based models
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Handle imbalance
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    train_auc = roc_auc_score(y_train, train_prob)
    test_auc = roc_auc_score(y_test, test_prob)

    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    return model, test_acc, test_auc


def train_mlp(X_train, X_test, y_train, y_test, feature_names):
    """Train MLP (neural network) classifier."""
    print("\n" + "="*60)
    print("TRAINING: MLP (Neural Network)")
    print("="*60)

    # Scale features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Smaller network for small dataset
        max_iter=1000,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_prob = model.predict_proba(X_train_scaled)[:, 1]
    test_prob = model.predict_proba(X_test_scaled)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    train_auc = roc_auc_score(y_train, train_prob)
    test_auc = roc_auc_score(y_test, test_prob)

    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    return model, test_acc, test_auc


def cross_validate_best_model(X, y, model_class, model_params, feature_names):
    """Perform cross-validation on the best model."""
    print("\n" + "="*60)
    print("CROSS-VALIDATION: Best Model")
    print("="*60)

    # Create model
    if model_class == 'RandomForest':
        model = RandomForestClassifier(**model_params)
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_use = X_scaled
    elif model_class == 'XGBoost':
        # XGBoost requires modified params
        xgb_params = model_params.copy()
        xgb_params['scale_pos_weight'] = len(y[y==0]) / len(y[y==1])
        model = xgb.XGBClassifier(**xgb_params)
        X_use = X
    elif model_class == 'MLP':
        model = MLPClassifier(**model_params)
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_use = X_scaled
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(model, X_use, y, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(model, X_use, y, cv=cv, scoring='roc_auc')

    print(f"CV Accuracy: {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%")
    print(f"CV AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")

    print("\nIndividual Fold Scores:")
    for i, (acc, auc) in enumerate(zip(cv_scores, cv_auc), 1):
        print(f"  Fold {i}: Acc={acc*100:.2f}%, AUC={auc:.4f}")

    return np.mean(cv_scores), np.mean(cv_auc)


def main():
    """Main execution function."""
    # Set paths
    data_path = Path('/home/mseo/CornerTactics/data/processed/corners_features_temporal_valid.csv')
    results_dir = Path('/home/mseo/CornerTactics/results/no_leakage')
    results_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run 14_extract_temporally_valid_features.py first")
        return

    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(data_path)

    # Train models
    results = {}

    # Random Forest
    rf_model, rf_acc, rf_auc = train_random_forest(
        X_train, X_test, y_train, y_test, feature_names
    )
    results['RandomForest'] = {
        'accuracy': rf_acc,
        'auc': rf_auc,
        'model_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    }

    # XGBoost
    xgb_model, xgb_acc, xgb_auc = train_xgboost(
        X_train, X_test, y_train, y_test, feature_names
    )
    results['XGBoost'] = {
        'accuracy': xgb_acc,
        'auc': xgb_auc,
        'model_params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    }

    # MLP
    mlp_model, mlp_acc, mlp_auc = train_mlp(
        X_train, X_test, y_train, y_test, feature_names
    )
    results['MLP'] = {
        'accuracy': mlp_acc,
        'auc': mlp_auc,
        'model_params': {
            'hidden_layer_sizes': (64, 32),
            'max_iter': 1000,
            'learning_rate_init': 0.001,
            'random_state': 42
        }
    }

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_name = best_model[0]
    best_acc = best_model[1]['accuracy']

    print("\n" + "="*60)
    print("RESULTS SUMMARY (NO TEMPORAL LEAKAGE)")
    print("="*60)

    for model_name, metrics in results.items():
        marker = "👑" if model_name == best_name else "  "
        print(f"{marker} {model_name:15s} Acc: {metrics['accuracy']*100:.2f}%  AUC: {metrics['auc']:.4f}")

    # Cross-validate best model
    print(f"\nBest Model: {best_name}")

    # Prepare full dataset
    df = pd.read_csv(data_path)
    exclude_cols = ['match_id', 'event_id', 'outcome', 'shot']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    y = df['shot'].values

    cv_acc, cv_auc = cross_validate_best_model(
        X, y,
        best_name,
        results[best_name]['model_params'],
        feature_cols
    )

    # Save results
    results_summary = {
        'models': results,
        'best_model': best_name,
        'best_accuracy': best_acc,
        'cv_accuracy': cv_acc,
        'cv_auc': cv_auc,
        'num_features': len(feature_names),
        'features_used': feature_names
    }

    results_path = results_dir / 'training_results_no_leakage.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n📊 Results saved to {results_path}")

    # Final comparison
    print("\n" + "="*60)
    print("⚠️ COMPARISON TO LEAKED MODELS")
    print("="*60)
    print("Previous 'optimal' model (WITH leakage):")
    print("  - Accuracy: 87.97%")
    print("  - Features: 24 (including leaked)")
    print("\nNew valid model (NO leakage):")
    print(f"  - Accuracy: {best_acc*100:.2f}%")
    print(f"  - Features: {len(feature_names)} (all valid)")
    print(f"  - Performance drop: -{(87.97 - best_acc*100):.2f}%")
    print("\n✅ This is the REAL predictive performance!")


if __name__ == "__main__":
    main()