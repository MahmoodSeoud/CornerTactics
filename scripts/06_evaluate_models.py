#!/usr/bin/env python3
"""
Task 6: Model Evaluation and Analysis

Loads trained baseline models and generates comprehensive evaluation:
- Confusion matrices for all models
- Feature importance plots (for RF and XGBoost)
- Per-class F1 comparison charts
- Error analysis and misclassification patterns
- Detailed evaluation report in markdown
"""

import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


def load_models(models_dir="models"):
    """Load all trained models from disk.

    Args:
        models_dir: Directory containing saved model files

    Returns:
        dict: Dictionary mapping model names to loaded model objects
    """
    models = {}

    # Load Random Forest
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            models['Random Forest'] = pickle.load(f)

    # Load XGBoost
    xgb_path = os.path.join(models_dir, "xgboost.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            models['XGBoost'] = pickle.load(f)

    # Load MLP
    mlp_path = os.path.join(models_dir, "mlp.pkl")
    if os.path.exists(mlp_path):
        with open(mlp_path, "rb") as f:
            models['MLP'] = pickle.load(f)

    return models


def load_label_encoder(models_dir="models"):
    """Load the label encoder.

    Args:
        models_dir: Directory containing saved label encoder

    Returns:
        LabelEncoder: Fitted label encoder
    """
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    with open(encoder_path, "rb") as f:
        return pickle.load(f)


def load_feature_scaler(models_dir="models"):
    """Load the feature scaler (for MLP).

    Args:
        models_dir: Directory containing saved scaler

    Returns:
        StandardScaler: Fitted scaler or None if not found
    """
    scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None


def generate_confusion_matrices(models, X_test, y_test, label_encoder, output_dir):
    """Generate and save confusion matrix plots for all models.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        label_encoder: Fitted label encoder
        output_dir: Directory to save plots

    Returns:
        dict: Dictionary mapping model names to confusion matrices
    """
    os.makedirs(output_dir, exist_ok=True)

    confusion_matrices = {}
    class_names = label_encoder.classes_

    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[model_name] = cm

        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # Save plot
        filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()

    return confusion_matrices


def extract_feature_importance(models, feature_names):
    """Extract feature importance from tree-based models.

    Args:
        models: Dictionary of trained models
        feature_names: List of feature names

    Returns:
        dict: Dictionary mapping model names to feature importance arrays
    """
    importance_dict = {}

    for model_name, model in models.items():
        # Only tree-based models have feature_importances_
        if hasattr(model, 'feature_importances_'):
            importance_dict[model_name] = model.feature_importances_

    return importance_dict


def plot_feature_importance(importance_dict, output_path, top_n=15):
    """Create feature importance comparison plot.

    Args:
        importance_dict: Dictionary of feature importances
        output_path: Path to save the plot
        top_n: Number of top features to show
    """
    if not importance_dict:
        return

    # Get feature names (assuming all models have same features)
    n_features = len(next(iter(importance_dict.values())))
    feature_names = [f'feature_{i}' for i in range(n_features)]

    # Create subplots for each model
    n_models = len(importance_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, importances) in zip(axes, importance_dict.items()):
        # Get top N features
        indices = np.argsort(importances)[-top_n:]

        # Plot
        ax.barh(range(top_n), importances[indices])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name} - Top {top_n} Features')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def find_most_confused_pairs(models, X_test, y_test, label_encoder, top_n=3):
    """Find the most commonly confused class pairs.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        label_encoder: Fitted label encoder
        top_n: Number of top confused pairs to return

    Returns:
        dict: Dictionary mapping model names to list of confused pairs
    """
    confused_pairs = {}
    class_names = label_encoder.classes_

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Find off-diagonal elements (misclassifications)
        pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    pairs.append((class_names[i], class_names[j], cm[i, j]))

        # Sort by count and take top N
        pairs.sort(key=lambda x: x[2], reverse=True)
        confused_pairs[model_name] = pairs[:top_n]

    return confused_pairs


def analyze_misclassifications(models, X_test, y_test, label_encoder, feature_names):
    """Analyze patterns in misclassified samples.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        label_encoder: Fitted label encoder
        feature_names: List of feature names

    Returns:
        dict: Dictionary mapping model names to misclassification analysis
    """
    analysis = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        # Find misclassified indices
        misclassified = y_pred != y_test
        n_misclassified = misclassified.sum()

        analysis[model_name] = {
            'total_misclassified': int(n_misclassified),
            'misclassification_rate': float(n_misclassified / len(y_test))
        }

    return analysis


def plot_per_class_f1(models, X_test, y_test, label_encoder, output_path):
    """Create per-class F1 score comparison bar chart.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        label_encoder: Fitted label encoder
        output_path: Path to save the plot
    """
    class_names = label_encoder.classes_
    n_classes = len(class_names)
    n_models = len(models)

    # Compute F1 scores for each model
    f1_scores = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        f1_scores[model_name] = f1_per_class

    # Create grouped bar chart
    x = np.arange(n_classes)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model_name, scores) in enumerate(f1_scores.items()):
        offset = (i - n_models / 2) * width + width / 2
        ax.bar(x + offset, scores, width, label=model_name)

    ax.set_xlabel('Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_evaluation_report(models, X_test, y_test, label_encoder, feature_names, output_path):
    """Generate comprehensive evaluation report in markdown.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        label_encoder: Fitted label encoder
        feature_names: List of feature names
        output_path: Path to save the report
    """
    class_names = label_encoder.classes_

    with open(output_path, 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"**Test Set Size**: {len(y_test)} samples\n\n")
        f.write("---\n\n")

        # Test Set Performance
        f.write("## Test Set Performance\n\n")

        for model_name, model in models.items():
            y_pred = model.predict(X_test)

            f.write(f"### {model_name}\n\n")

            # Classification report
            report = classification_report(y_test, y_pred, target_names=class_names)
            f.write("```\n")
            f.write(report)
            f.write("\n```\n\n")

        f.write("---\n\n")

        # Confusion Matrices
        f.write("## Confusion Matrices\n\n")
        f.write("See confusion matrix plots in `results/confusion_matrices/`\n\n")

        confused_pairs = find_most_confused_pairs(models, X_test, y_test, label_encoder)
        f.write("### Most Confused Pairs\n\n")

        for model_name, pairs in confused_pairs.items():
            f.write(f"**{model_name}**:\n")
            for true_class, pred_class, count in pairs:
                f.write(f"- {true_class} → {pred_class}: {count} times\n")
            f.write("\n")

        f.write("---\n\n")

        # Feature Importance
        f.write("## Feature Importance\n\n")

        # Extract from the underlying models, not wrapped ones
        actual_models = {}
        for model_name, wrapped_model in models.items():
            if hasattr(wrapped_model, 'model'):
                actual_models[model_name] = wrapped_model.model
            else:
                actual_models[model_name] = wrapped_model

        importance_dict = extract_feature_importance(actual_models, feature_names)
        if importance_dict:
            f.write("See feature importance plot in `results/feature_importance.png`\n\n")

            for model_name, importances in importance_dict.items():
                indices = np.argsort(importances)[-5:][::-1]
                f.write(f"**{model_name} - Top 5 Features**:\n")
                for idx in indices:
                    f.write(f"- {feature_names[idx]}: {importances[idx]:.4f}\n")
                f.write("\n")
        else:
            f.write("No tree-based models to extract feature importance.\n\n")

        f.write("---\n\n")

        # Error Analysis
        f.write("## Error Analysis\n\n")

        analysis = analyze_misclassifications(models, X_test, y_test, label_encoder, feature_names)
        for model_name, metrics in analysis.items():
            f.write(f"**{model_name}**:\n")
            f.write(f"- Total misclassified: {metrics['total_misclassified']}\n")
            f.write(f"- Misclassification rate: {metrics['misclassification_rate']:.2%}\n")
            f.write("\n")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Task 6: Model Evaluation and Analysis")
    print("=" * 80)

    # Setup paths
    data_dir = "data/processed"
    models_dir = "models"
    results_dir = "results"

    print("\n[1/7] Loading test data...")

    # Load test data
    features_df = pd.read_csv(os.path.join(data_dir, "corners_with_features.csv"))
    test_indices_df = pd.read_csv(os.path.join(data_dir, "test_indices.csv"))
    test_indices = test_indices_df['index'].values

    # Extract features and labels
    feature_cols = [col for col in features_df.columns if col not in ['match_id', 'event_id', 'outcome']]
    X = features_df[feature_cols].values

    # Load label encoder
    label_encoder = load_label_encoder(models_dir)
    y = label_encoder.transform(features_df['outcome'])

    # Get test set
    X_test = X[test_indices]
    y_test = y[test_indices]

    print(f"   Test set size: {len(y_test)} samples")
    print(f"   Number of features: {X_test.shape[1]}")

    print("\n[2/7] Loading trained models...")

    models = load_models(models_dir)
    scaler = load_feature_scaler(models_dir)

    print(f"   Loaded {len(models)} models: {list(models.keys())}")

    print("\n[3/7] Generating confusion matrices...")

    cm_dir = os.path.join(results_dir, "confusion_matrices")

    # Apply scaling for MLP if needed
    X_test_for_models = {}
    for model_name in models.keys():
        if model_name == 'MLP' and scaler is not None:
            X_test_for_models[model_name] = scaler.transform(X_test)
        else:
            X_test_for_models[model_name] = X_test

    # Generate confusion matrices (need to handle scaling per model)
    os.makedirs(cm_dir, exist_ok=True)
    for model_name, model in models.items():
        X_for_model = X_test_for_models[model_name]
        y_pred = model.predict(X_for_model)
        cm = confusion_matrix(y_test, y_pred)

        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(os.path.join(cm_dir, filename), dpi=150)
        plt.close()

        print(f"   Saved {filename}")

    print("\n[4/7] Extracting feature importance...")

    importance_dict = extract_feature_importance(models, feature_cols)

    if importance_dict:
        print(f"   Extracted importance for {len(importance_dict)} models")
        plot_feature_importance(importance_dict, os.path.join(results_dir, "feature_importance.png"))
        print("   Saved feature_importance.png")
    else:
        print("   No tree-based models found")

    print("\n[5/7] Creating per-class F1 comparison...")

    # Use non-scaled data for tree models, scaled for MLP
    models_for_f1 = {}
    for model_name, model in models.items():
        models_for_f1[model_name] = model

    # Create a wrapper to handle scaling
    class ModelWrapper:
        def __init__(self, model, scaler=None):
            self.model = model
            self.scaler = scaler

        def predict(self, X):
            if self.scaler is not None:
                X = self.scaler.transform(X)
            return self.model.predict(X)

    wrapped_models = {}
    for model_name, model in models.items():
        if model_name == 'MLP' and scaler is not None:
            wrapped_models[model_name] = ModelWrapper(model, scaler)
        else:
            wrapped_models[model_name] = ModelWrapper(model, None)

    plot_per_class_f1(wrapped_models, X_test, y_test, label_encoder,
                     os.path.join(results_dir, "per_class_f1.png"))
    print("   Saved per_class_f1.png")

    print("\n[6/7] Performing error analysis...")

    analysis = analyze_misclassifications(wrapped_models, X_test, y_test, label_encoder, feature_cols)
    for model_name, metrics in analysis.items():
        print(f"   {model_name}: {metrics['total_misclassified']} misclassified ({metrics['misclassification_rate']:.2%})")

    print("\n[7/7] Generating evaluation report...")

    generate_evaluation_report(wrapped_models, X_test, y_test, label_encoder, feature_cols,
                              os.path.join(results_dir, "evaluation_report.md"))
    print("   Saved evaluation_report.md")

    print("\n" + "=" * 80)
    print("Task 6 Complete!")
    print("=" * 80)
    print(f"\nOutputs saved to:")
    print(f"  - {cm_dir}/")
    print(f"  - {results_dir}/feature_importance.png")
    print(f"  - {results_dir}/per_class_f1.png")
    print(f"  - {results_dir}/evaluation_report.md")
    print()


if __name__ == "__main__":
    main()
