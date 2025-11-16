"""
Task 8: Binary Shot Classification Models

Train binary classifiers (Random Forest, XGBoost, MLP) to predict whether
a corner kick leads to a shot within the next 5 events.

Usage:
    python scripts/08_train_binary_models.py
"""
import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    auc
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """
    Load and merge shot labels with features.

    Returns:
        pd.DataFrame: Merged data with features and shot_outcome
    """
    print("Loading shot labels...")
    shot_labels_path = Path("data/processed/corners_with_shot_labels.json")
    with open(shot_labels_path) as f:
        shot_data = json.load(f)

    # Extract shot labels
    shot_labels = []
    for corner in shot_data:
        event_id = corner['event']['id']
        shot_outcome = corner['shot_outcome']
        shot_labels.append({'event_id': event_id, 'shot_outcome': shot_outcome})

    shot_df = pd.DataFrame(shot_labels)
    print(f"Loaded {len(shot_df)} shot labels")

    # Load features
    print("\nLoading features...")
    features_path = Path("data/processed/corners_with_features.csv")
    features_df = pd.read_csv(features_path)
    print(f"Loaded {len(features_df)} feature rows")

    # Merge
    print("\nMerging shot labels with features...")
    merged = pd.merge(features_df, shot_df, on='event_id', how='inner')
    print(f"Merged data: {len(merged)} rows")

    # Verify binary labels
    assert merged['shot_outcome'].isin([0, 1]).all(), "Shot outcomes must be binary"

    return merged


def load_splits():
    """
    Load train/val/test splits from Task 4.

    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    print("\nLoading train/val/test splits...")
    train_indices = pd.read_csv("data/processed/train_indices.csv")['index'].values
    val_indices = pd.read_csv("data/processed/val_indices.csv")['index'].values
    test_indices = pd.read_csv("data/processed/test_indices.csv")['index'].values

    print(f"Train: {len(train_indices)} samples")
    print(f"Val:   {len(val_indices)} samples")
    print(f"Test:  {len(test_indices)} samples")

    return train_indices, val_indices, test_indices


def prepare_datasets(df, train_indices, val_indices, test_indices):
    """
    Prepare train/val/test datasets.

    Args:
        df: Full dataset with features and labels
        train_indices: Training indices
        val_indices: Validation indices
        test_indices: Test indices

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\nPreparing datasets...")

    # Define feature columns (exclude metadata and labels)
    exclude_cols = ['match_id', 'event_id', 'outcome', 'shot_outcome']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Feature columns: {len(feature_cols)}")

    # Split data
    X_train = df.iloc[train_indices][feature_cols].values
    X_val = df.iloc[val_indices][feature_cols].values
    X_test = df.iloc[test_indices][feature_cols].values

    y_train = df.iloc[train_indices]['shot_outcome'].values
    y_val = df.iloc[val_indices]['shot_outcome'].values
    y_test = df.iloc[test_indices]['shot_outcome'].values

    # Print class distributions
    print("\nClass distribution:")
    print(f"Train - Shot: {y_train.sum()}/{len(y_train)} ({100*y_train.mean():.1f}%)")
    print(f"Val   - Shot: {y_val.sum()}/{len(y_val)} ({100*y_val.mean():.1f}%)")
    print(f"Test  - Shot: {y_test.sum()}/{len(y_test)} ({100*y_test.mean():.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest binary classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        RandomForestClassifier: Trained model
    """
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        class_weight='balanced',  # Handle 2.45:1 imbalance
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # Validation performance
    y_val_pred = rf.predict(X_val)
    y_val_proba = rf.predict_proba(X_val)[:, 1]

    print("\nValidation Performance:")
    print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_val_pred):.4f}")
    print(f"F1:        {f1_score(y_val, y_val_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_val, y_val_proba):.4f}")

    return rf


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost binary classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        XGBClassifier: Trained model
    """
    print("\n" + "="*60)
    print("Training XGBoost")
    print("="*60)

    # Calculate scale_pos_weight (ratio of negative to positive)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    xgb_model.fit(X_train, y_train)

    # Validation performance
    y_val_pred = xgb_model.predict(X_val)
    y_val_proba = xgb_model.predict_proba(X_val)[:, 1]

    print("\nValidation Performance:")
    print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_val_pred):.4f}")
    print(f"F1:        {f1_score(y_val, y_val_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_val, y_val_proba):.4f}")

    return xgb_model


def train_mlp(X_train, y_train, X_val, y_val):
    """
    Train MLP binary classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        tuple: (MLPClassifier, StandardScaler) - Trained model and scaler
    """
    print("\n" + "="*60)
    print("Training MLP (Neural Network)")
    print("="*60)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.001,
        max_iter=500,
        random_state=42,
        verbose=True
    )

    mlp.fit(X_train_scaled, y_train)

    # Validation performance
    y_val_pred = mlp.predict(X_val_scaled)
    y_val_proba = mlp.predict_proba(X_val_scaled)[:, 1]

    print("\nValidation Performance:")
    print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_val_pred):.4f}")
    print(f"F1:        {f1_score(y_val, y_val_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_val, y_val_proba):.4f}")

    return mlp, scaler


def evaluate_model(model, X_test, y_test, model_name, scaler=None):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        scaler: StandardScaler for MLP (optional)

    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on Test Set")
    print(f"{'='*60}")

    # Scale features if using MLP
    X_test_input = scaler.transform(X_test) if scaler else X_test

    # Predictions
    y_pred = model.predict(X_test_input)
    y_proba = model.predict_proba(X_test_input)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)

    print(f"\nAccuracy:           {accuracy:.4f}")
    print(f"Precision (Shot):   {precision:.4f}")
    print(f"Recall (Shot):      {recall:.4f}")
    print(f"F1 (Shot):          {f1:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"PR-AUC:             {pr_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Shot', 'Shot']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_proba.tolist()
    }


def save_models(rf, xgb_model, mlp, scaler):
    """
    Save trained models to disk.

    Args:
        rf: Random Forest model
        xgb_model: XGBoost model
        mlp: MLP model
        scaler: StandardScaler for MLP
    """
    print("\nSaving models...")
    models_dir = Path("models/binary")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save Random Forest
    with open(models_dir / "random_forest.pkl", 'wb') as f:
        pickle.dump(rf, f)
    print(f"Saved: {models_dir / 'random_forest.pkl'}")

    # Save XGBoost
    xgb_model.save_model(str(models_dir / "xgboost.json"))
    print(f"Saved: {models_dir / 'xgboost.json'}")

    # Save MLP
    with open(models_dir / "mlp.pkl", 'wb') as f:
        pickle.dump(mlp, f)
    print(f"Saved: {models_dir / 'mlp.pkl'}")

    # Save scaler
    with open(models_dir / "feature_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved: {models_dir / 'feature_scaler.pkl'}")


def save_metrics(rf_metrics, xgb_metrics, mlp_metrics):
    """
    Save evaluation metrics to JSON.

    Args:
        rf_metrics: Random Forest metrics
        xgb_metrics: XGBoost metrics
        mlp_metrics: MLP metrics
    """
    print("\nSaving metrics...")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Remove non-serializable fields
    for metrics in [rf_metrics, xgb_metrics, mlp_metrics]:
        metrics.pop('y_pred', None)
        metrics.pop('y_proba', None)

    metrics = {
        'random_forest': rf_metrics,
        'xgboost': xgb_metrics,
        'mlp': mlp_metrics
    }

    with open(results_dir / "binary_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved: {results_dir / 'binary_metrics.json'}")


def plot_confusion_matrices(rf_metrics, xgb_metrics, mlp_metrics):
    """
    Plot confusion matrices for all models.

    Args:
        rf_metrics: Random Forest metrics
        xgb_metrics: XGBoost metrics
        mlp_metrics: MLP metrics
    """
    print("\nGenerating confusion matrices...")
    results_dir = Path("results/confusion_matrices_binary")
    results_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ('Random Forest', rf_metrics['confusion_matrix']),
        ('XGBoost', xgb_metrics['confusion_matrix']),
        ('MLP', mlp_metrics['confusion_matrix'])
    ]

    for model_name, cm in models:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Shot', 'Shot'],
                    yticklabels=['No Shot', 'Shot'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        filename = results_dir / f"{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("Task 8: Binary Shot Classification Models")
    print("="*60)

    # Load data
    df = load_data()

    # Load splits
    train_indices, val_indices, test_indices = load_splits()

    # Prepare datasets
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_datasets(
        df, train_indices, val_indices, test_indices
    )

    # Train models
    rf = train_random_forest(X_train, y_train, X_val, y_val)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    mlp, scaler = train_mlp(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    mlp_metrics = evaluate_model(mlp, X_test, y_test, "MLP", scaler=scaler)

    # Save models
    save_models(rf, xgb_model, mlp, scaler)

    # Save metrics
    save_metrics(rf_metrics, xgb_metrics, mlp_metrics)

    # Plot confusion matrices
    plot_confusion_matrices(rf_metrics, xgb_metrics, mlp_metrics)

    print("\n" + "="*60)
    print("✓ Task 8 complete!")
    print("="*60)
    print(f"\nModels saved to: models/binary/")
    print(f"Metrics saved to: results/binary_metrics.json")
    print(f"Confusion matrices saved to: results/confusion_matrices_binary/")


if __name__ == "__main__":
    main()
