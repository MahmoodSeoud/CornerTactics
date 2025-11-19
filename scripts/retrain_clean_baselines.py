#!/usr/bin/env python3
"""
Retrain All Baseline Models Without Leaked Features

This script:
1. Removes all leaked features
2. Retrains MLP, XGBoost, and Random Forest
3. Performs proper evaluation with cross-validation
4. Saves clean models and results
"""

import numpy as np
import pandas as pd
import json
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available, will skip XGBoost model")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Leaked features to remove
LEAKED_FEATURES = [
    'is_shot_assist',      # Directly encodes target
    'has_recipient',       # Only known after pass completes
    'duration',            # Time until next event
    'pass_end_x',          # Actual landing location
    'pass_end_y',          # Actual landing location
    'pass_length',         # Computed from actual landing
    'pass_angle',          # Computed from actual landing
    'pass_outcome_id',     # If exists
    'pass_outcome_encoded',# If exists
    'has_pass_outcome',    # If exists
    'is_aerial_won',       # If exists
    'pass_recipient_id',   # If exists
]

# Non-features to exclude
NON_FEATURES = [
    'match_id', 'event_id', 'outcome', 'leads_to_shot',
    'pass_outcome', 'pass_height', 'pass_body_part',
    'pass_technique', 'corner_x', 'corner_y'
]

# Random seed for reproducibility
RANDOM_STATE = 42

# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_clean_data(data_path: str):
    """Load data and remove leaked features."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"Original shape: {df.shape}")
    print(f"Target distribution:\n{df['leads_to_shot'].value_counts()}")
    print(f"Target percentage: {df['leads_to_shot'].mean()*100:.1f}% shots")

    # Get clean features
    all_cols = df.columns.tolist()
    features_to_keep = []
    removed_features = []

    for col in all_cols:
        if col in NON_FEATURES or col == 'leads_to_shot':
            continue
        elif col in LEAKED_FEATURES:
            if col in all_cols:  # Only if it exists
                removed_features.append(col)
        else:
            features_to_keep.append(col)

    print(f"\nRemoved {len(removed_features)} leaked features: {removed_features}")
    print(f"Keeping {len(features_to_keep)} clean features")

    return df, features_to_keep

def prepare_data(df, features, test_size=0.2):
    """Prepare train/test split with scaling."""
    X = df[features].fillna(df[features].median())
    y = df['leads_to_shot'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTrain set: {X_train_scaled.shape}, {y_train.mean()*100:.1f}% positive")
    print(f"Test set: {X_test_scaled.shape}, {y_test.mean()*100:.1f}% positive")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# =============================================================================
# MODEL TRAINING
# =============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Comprehensive model evaluation."""
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
    )
    metrics['cv_auc_mean'] = cv_scores.mean()
    metrics['cv_auc_std'] = cv_scores.std()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.3f}")
    print(f"MCC:       {metrics['mcc']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    print(f"CV AUC:    {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

    return metrics, cm

def train_mlp(X_train, y_train, X_test, y_test):
    """Train Multi-Layer Perceptron."""
    print("\nTraining MLP...")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )

    model.fit(X_train, y_train)

    metrics, cm = evaluate_model(model, X_train, y_train, X_test, y_test, "MLP")

    return model, metrics, cm

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest."""
    print("\nTraining Random Forest...")

    # Calculate class weight
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    metrics, cm = evaluate_model(model, X_train, y_train, X_test, y_test, "Random Forest")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        print(f"\nTop 10 Features:")
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        for idx in np.argsort(importance)[-10:][::-1]:
            print(f"  {feature_names[idx]}: {importance[idx]:.4f}")

    return model, metrics, cm

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost."""
    if not HAS_XGBOOST:
        print("\nSkipping XGBoost (not installed)")
        return None, None, None

    print("\nTraining XGBoost...")

    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    )

    # Train with early stopping
    eval_set = [(X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=10,
        verbose=False
    )

    metrics, cm = evaluate_model(model, X_train, y_train, X_test, y_test, "XGBoost")

    return model, metrics, cm

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_confusion_matrices(cms, model_names, output_path):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, len(cms), figsize=(5*len(cms), 4))

    if len(cms) == 1:
        axes = [axes]

    for ax, cm, name in zip(axes, cms, model_names):
        if cm is None:
            continue

        # Normalize for percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'})

        # Add percentages
        for i in range(2):
            for j in range(2):
                text = ax.text(j+0.5, i+0.7, f'({cm_norm[i,j]:.1f}%)',
                             ha='center', va='center', fontsize=8)

        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['No Shot', 'Shot'])
        ax.set_yticklabels(['No Shot', 'Shot'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved confusion matrices to {output_path}")

def plot_model_comparison(all_metrics, output_path):
    """Create bar plot comparing model performances."""
    metrics_to_plot = ['accuracy', 'auc_roc', 'mcc', 'f1']
    model_names = list(all_metrics.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for i, model in enumerate(model_names):
        if all_metrics[model] is not None:
            values = [all_metrics[model][m] for m in metrics_to_plot]
            ax.bar(x + i*width, values, width, label=model)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Clean Baseline Model Comparison (No Leaked Features)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics_to_plot])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'processed' / 'corners_features_with_shot.csv'
    output_dir = project_root / 'models' / 'clean_baselines'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df, features = load_and_clean_data(data_path)

    # Save feature list
    with open(output_dir / 'clean_features.json', 'w') as f:
        json.dump({
            'features': features,
            'num_features': len(features),
            'removed_features': LEAKED_FEATURES
        }, f, indent=2)

    X_train, X_test, y_train, y_test, scaler = prepare_data(df, features)

    # Save scaler
    joblib.dump(scaler, output_dir / 'feature_scaler.pkl')
    print(f"\nSaved feature scaler to {output_dir / 'feature_scaler.pkl'}")

    # Train all models
    print("\n" + "="*60)
    print("TRAINING CLEAN BASELINE MODELS")
    print("="*60)

    # MLP
    mlp_model, mlp_metrics, mlp_cm = train_mlp(X_train, y_train, X_test, y_test)
    joblib.dump(mlp_model, output_dir / 'mlp_clean.pkl')

    # Random Forest
    rf_model, rf_metrics, rf_cm = train_random_forest(X_train, y_train, X_test, y_test)
    joblib.dump(rf_model, output_dir / 'random_forest_clean.pkl')

    # XGBoost
    xgb_model, xgb_metrics, xgb_cm = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_model is not None:
        joblib.dump(xgb_model, output_dir / 'xgboost_clean.pkl')

    # Compile results
    all_metrics = {
        'MLP': mlp_metrics,
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics
    }

    all_cms = [mlp_cm, rf_cm, xgb_cm]
    model_names = ['MLP', 'Random Forest', 'XGBoost']

    # Create visualizations
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    plot_confusion_matrices(all_cms, model_names, figures_dir / 'confusion_matrices.png')
    plot_model_comparison(all_metrics, figures_dir / 'model_comparison.png')

    # Save all results
    results = {
        'data_info': {
            'total_samples': len(df),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'shot_percentage': float(df['leads_to_shot'].mean()),
            'num_features': len(features)
        },
        'model_results': all_metrics,
        'feature_info': {
            'clean_features': features,
            'removed_features': LEAKED_FEATURES
        }
    }

    with open(output_dir / 'clean_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate summary report
    print("\n" + "="*60)
    print("CLEAN BASELINE SUMMARY")
    print("="*60)
    print(f"Data: {len(df)} samples, {len(features)} clean features")
    print(f"Target: {df['leads_to_shot'].mean()*100:.1f}% shots")
    print(f"\nBest Performance:")

    best_model = None
    best_auc = 0
    for model_name, metrics in all_metrics.items():
        if metrics is not None and metrics['auc_roc'] > best_auc:
            best_auc = metrics['auc_roc']
            best_model = model_name

    if best_model:
        print(f"  Model: {best_model}")
        print(f"  AUC-ROC: {all_metrics[best_model]['auc_roc']:.3f}")
        print(f"  Accuracy: {all_metrics[best_model]['accuracy']:.3f}")
        print(f"  MCC: {all_metrics[best_model]['mcc']:.3f}")

    print(f"\nModels saved to: {output_dir}")
    print(f"Results saved to: {output_dir / 'clean_baseline_results.json'}")
    print("="*60)

if __name__ == '__main__':
    main()