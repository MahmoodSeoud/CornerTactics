#!/usr/bin/env python3
"""
Train baseline models for corner kick outcome prediction.

This script trains three baseline models:
1. Random Forest
2. XGBoost
3. MLP (Neural Network)

All models use balanced class weights to handle the 5.4:1 imbalance.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import pickle


def load_data():
    """Load features and split indices."""
    # Load features
    features_path = Path("data/processed/corners_with_features.csv")
    df = pd.read_csv(features_path)

    # Load split indices
    train_indices = pd.read_csv("data/processed/train_indices.csv")["index"].values
    val_indices = pd.read_csv("data/processed/val_indices.csv")["index"].values
    test_indices = pd.read_csv("data/processed/test_indices.csv")["index"].values

    print(f"Loaded {len(df)} total samples")
    print(f"Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    return df, train_indices, val_indices, test_indices


def prepare_features(df):
    """Prepare features and labels."""
    # Feature columns (exclude metadata)
    feature_cols = [col for col in df.columns
                    if col not in ['match_id', 'event_id', 'outcome']]

    X = df[feature_cols].values
    y = df['outcome'].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\nFeature shape: {X.shape}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Class distribution:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"  {class_name}: {count} ({count/len(y_encoded)*100:.1f}%)")

    return X, y_encoded, label_encoder, feature_cols


def calculate_class_weights(y_train):
    """Calculate scale_pos_weight for XGBoost (for multi-class, returns weights dict)."""
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = {}

    # For multi-class XGBoost, we'll use sample_weight instead
    # Calculate balanced weights
    n_samples = len(y_train)
    n_classes = len(unique)

    for cls, count in zip(unique, counts):
        class_weights[cls] = n_samples / (n_classes * count)

    return class_weights


def train_random_forest(X_train, y_train, X_val, y_val, n_classes):
    """Train Random Forest classifier."""
    print("\n" + "="*50)
    print("Training Random Forest")
    print("="*50)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rf.fit(X_train, y_train)

    # Evaluate
    train_preds = rf.predict(X_train)
    val_preds = rf.predict(X_val)

    train_f1_macro = f1_score(y_train, train_preds, average='macro')
    train_f1_weighted = f1_score(y_train, train_preds, average='weighted')
    val_f1_macro = f1_score(y_val, val_preds, average='macro')
    val_f1_weighted = f1_score(y_val, val_preds, average='weighted')

    print(f"\nTrain - Macro F1: {train_f1_macro:.4f} | Weighted F1: {train_f1_weighted:.4f}")
    print(f"Val   - Macro F1: {val_f1_macro:.4f} | Weighted F1: {val_f1_weighted:.4f}")

    return rf, {
        'train_f1_macro': train_f1_macro,
        'train_f1_weighted': train_f1_weighted,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted,
    }


def train_xgboost(X_train, y_train, X_val, y_val, n_classes):
    """Train XGBoost classifier."""
    print("\n" + "="*50)
    print("Training XGBoost")
    print("="*50)

    # Calculate sample weights for handling class imbalance
    class_weights = calculate_class_weights(y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        objective='multi:softmax',  # Multi-class classification
        num_class=n_classes
    )

    xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate
    train_preds = xgb.predict(X_train)
    val_preds = xgb.predict(X_val)

    train_f1_macro = f1_score(y_train, train_preds, average='macro')
    train_f1_weighted = f1_score(y_train, train_preds, average='weighted')
    val_f1_macro = f1_score(y_val, val_preds, average='macro')
    val_f1_weighted = f1_score(y_val, val_preds, average='weighted')

    print(f"\nTrain - Macro F1: {train_f1_macro:.4f} | Weighted F1: {train_f1_weighted:.4f}")
    print(f"Val   - Macro F1: {val_f1_macro:.4f} | Weighted F1: {val_f1_weighted:.4f}")

    return xgb, {
        'train_f1_macro': train_f1_macro,
        'train_f1_weighted': train_f1_weighted,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted,
    }


def train_mlp(X_train, y_train, X_val, y_val, n_classes):
    """Train MLP classifier with feature scaling."""
    print("\n" + "="*50)
    print("Training MLP (Neural Network)")
    print("="*50)

    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.001,  # L2 regularization
        max_iter=500,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )

    mlp.fit(X_train_scaled, y_train)

    # Evaluate
    train_preds = mlp.predict(X_train_scaled)
    val_preds = mlp.predict(X_val_scaled)

    train_f1_macro = f1_score(y_train, train_preds, average='macro')
    train_f1_weighted = f1_score(y_train, train_preds, average='weighted')
    val_f1_macro = f1_score(y_val, val_preds, average='macro')
    val_f1_weighted = f1_score(y_val, val_preds, average='weighted')

    print(f"\nTrain - Macro F1: {train_f1_macro:.4f} | Weighted F1: {train_f1_weighted:.4f}")
    print(f"Val   - Macro F1: {val_f1_macro:.4f} | Weighted F1: {val_f1_weighted:.4f}")

    return mlp, scaler, {
        'train_f1_macro': train_f1_macro,
        'train_f1_weighted': train_f1_weighted,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted,
    }


def evaluate_on_test(models, X_test, y_test, label_encoder, scaler=None):
    """Evaluate all models on test set and generate detailed metrics."""
    print("\n" + "="*50)
    print("Test Set Evaluation")
    print("="*50)

    results = {}

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 50)

        # Prepare test data
        if model_name == 'MLP' and scaler is not None:
            X_test_input = scaler.transform(X_test)
        else:
            X_test_input = X_test

        # Predictions
        y_pred = model.predict(X_test_input)

        # Metrics
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        print(f"\nAccuracy: {report['accuracy']:.4f}")
        print("\nPer-class metrics:")
        for class_name in label_encoder.classes_:
            metrics = report[class_name]
            print(f"  {class_name:15s} - P: {metrics['precision']:.3f} | "
                  f"R: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f} | "
                  f"Support: {int(metrics['support'])}")

        print(f"\nConfusion Matrix:")
        print("Predicted →")
        print("Actual ↓")
        header = "".join([f"{cls[:12]:>12s}" for cls in label_encoder.classes_])
        print(f"{'':15s}{header}")
        for i, class_name in enumerate(label_encoder.classes_):
            row = "".join([f"{cm[i][j]:12d}" for j in range(len(label_encoder.classes_))])
            print(f"{class_name[:15]:15s}{row}")

        # Store results
        results[model_name] = {
            'test_f1_macro': f1_macro,
            'test_f1_weighted': f1_weighted,
            'test_accuracy': report['accuracy'],
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

    return results


def save_models(models, label_encoder, scaler, results):
    """Save trained models and results."""
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Save models
    print("\nSaving models...")

    # Model name to filename mapping
    model_filenames = {
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'MLP': 'mlp.pkl'
    }

    # Save each model that exists
    for model_name, filename in model_filenames.items():
        if model_name in models:
            with open(f"models/{filename}", "wb") as f:
                pickle.dump(models[model_name], f)
            print(f"✓ Saved models/{filename}")

    # Save label encoder
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print("✓ Saved models/label_encoder.pkl")

    # Save scaler if provided
    if scaler is not None:
        with open("models/feature_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("✓ Saved models/feature_scaler.pkl")

    # Save results
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✓ Saved results/baseline_metrics.json")


def print_comparison_table(all_results):
    """Print comparison table of all models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    print(f"\n{'Model':<20s} {'Val Macro F1':>12s} {'Val Weighted F1':>15s} "
          f"{'Test Macro F1':>13s} {'Test Weighted F1':>16s}")
    print("-" * 80)

    for model_name in ['Random Forest', 'XGBoost', 'MLP']:
        val_macro = all_results[model_name]['val_f1_macro']
        val_weighted = all_results[model_name]['val_f1_weighted']
        test_macro = all_results[model_name]['test_f1_macro']
        test_weighted = all_results[model_name]['test_f1_weighted']

        print(f"{model_name:<20s} {val_macro:>12.4f} {val_weighted:>15.4f} "
              f"{test_macro:>13.4f} {test_weighted:>16.4f}")

    print("="*80)


def main():
    """Main training pipeline."""
    print("="*80)
    print("CORNER KICK OUTCOME PREDICTION - BASELINE MODELS")
    print("="*80)

    # Load data
    df, train_indices, val_indices, test_indices = load_data()

    # Prepare features
    X, y, label_encoder, feature_cols = prepare_features(df)

    # Split data
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    n_classes = len(label_encoder.classes_)

    # Train models
    all_results = {}

    # 1. Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val, n_classes)
    all_results['Random Forest'] = rf_metrics

    # 2. XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val, n_classes)
    all_results['XGBoost'] = xgb_metrics

    # 3. MLP
    mlp_model, scaler, mlp_metrics = train_mlp(X_train, y_train, X_val, y_val, n_classes)
    all_results['MLP'] = mlp_metrics

    # Collect models
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'MLP': mlp_model
    }

    # Evaluate on test set
    test_results = evaluate_on_test(models, X_test, y_test, label_encoder, scaler)

    # Merge results
    for model_name in models.keys():
        all_results[model_name].update(test_results[model_name])

    # Print comparison table
    print_comparison_table(all_results)

    # Save everything
    save_models(models, label_encoder, scaler, all_results)

    print("\n✓ Training complete!")
    print(f"\nModels saved to: models/")
    print(f"Results saved to: results/baseline_metrics.json")


if __name__ == "__main__":
    main()
