"""
Ablation Study: Progressive Feature Addition

Trains models on each feature configuration (Step 0-9) and tracks performance metrics.

For each step:
1. Load feature CSV
2. Create train/val/test splits
3. Train 3 models: Random Forest, XGBoost, MLP
4. Evaluate on test set
5. Save metrics and model checkpoints
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle


# ============ NEURAL NETWORK DEFINITION ============

class MLP(nn.Module):
    """Multi-layer perceptron for classification."""

    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============ DATA LOADING ============

def load_feature_data(feature_file, labels_file):
    """Load features and labels."""
    print(f"Loading features from: {feature_file}")
    features_df = pd.read_csv(feature_file)

    print(f"Loading labels from: {labels_file}")
    labels_df = pd.read_csv(labels_file)

    # Merge on event_id
    merged = features_df.merge(labels_df, on='event_id', how='inner')

    print(f"Merged dataset shape: {merged.shape}")
    return merged


def prepare_splits(df, random_seed=42):
    """Create match-based train/val/test splits."""
    # Get unique match IDs
    unique_matches = df['match_id'].unique()
    print(f"Total unique matches: {len(unique_matches)}")

    # Split matches 70/15/15
    train_matches, temp_matches = train_test_split(
        unique_matches, test_size=0.3, random_state=random_seed
    )
    val_matches, test_matches = train_test_split(
        temp_matches, test_size=0.5, random_state=random_seed
    )

    # Create splits
    train_df = df[df['match_id'].isin(train_matches)]
    val_df = df[df['match_id'].isin(val_matches)]
    test_df = df[df['match_id'].isin(test_matches)]

    print(f"Train: {len(train_df)} samples from {len(train_matches)} matches")
    print(f"Val: {len(val_df)} samples from {len(val_matches)} matches")
    print(f"Test: {len(test_df)} samples from {len(test_matches)} matches")

    return train_df, val_df, test_df


def prepare_features_labels(df, feature_cols, label_col):
    """Prepare feature matrix and label vector."""
    X = df[feature_cols].values
    y = df[label_col].values
    return X, y


# ============ MODEL TRAINING ============

def train_random_forest(X_train, y_train, X_val, y_val, random_seed=42):
    """Train Random Forest classifier."""
    print("Training Random Forest...")
    start_time = time.time()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")

    return model, train_time


def train_xgboost(X_train, y_train, X_val, y_val, random_seed=42):
    """Train XGBoost classifier."""
    print("Training XGBoost...")
    start_time = time.time()

    # Determine task type
    num_classes = len(np.unique(y_train))

    if num_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'logloss'
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss'

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective=objective,
        eval_metric=eval_metric,
        random_state=random_seed,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")

    return model, train_time


def train_mlp(X_train, y_train, X_val, y_val, random_seed=42):
    """Train MLP classifier."""
    print("Training MLP...")
    start_time = time.time()

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Determine task type
    num_classes = len(np.unique(y_train))

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]
    hidden_dims = [512, 256, 128, 64]
    model = MLP(input_dim, hidden_dims, num_classes, dropout=0.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(best_model_state)

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")

    return (model, scaler), train_time


# ============ MODEL EVALUATION ============

def evaluate_model(model, X_test, y_test, model_type='rf'):
    """Evaluate model on test set."""
    print(f"Evaluating {model_type.upper()}...")

    # Predictions
    if model_type == 'mlp':
        mlp_model, scaler = model
        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlp_model = mlp_model.to(device)
        mlp_model.eval()

        with torch.no_grad():
            outputs = mlp_model(X_test_tensor.to(device))
            probas = torch.softmax(outputs, dim=1).cpu().numpy()
            y_pred = outputs.argmax(dim=1).cpu().numpy()
    else:
        y_pred = model.predict(X_test)
        probas = model.predict_proba(X_test)

    # Metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')

    # Binary task metrics
    num_classes = len(np.unique(y_test))
    if num_classes == 2:
        metrics['roc_auc'] = roc_auc_score(y_test, probas[:, 1])
        metrics['pr_auc'] = average_precision_score(y_test, probas[:, 1])

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

    # Per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics['per_class'] = report

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    if num_classes == 2:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics


# ============ MAIN ABLATION LOOP ============

def run_ablation_study(step, task='4class'):
    """
    Run ablation study for a single step.

    Args:
        step: Feature step (0-9)
        task: '4class' or 'binary_shot'
    """
    print(f"\n{'='*60}")
    print(f"STEP {step} - {task.upper()}")
    print(f"{'='*60}")

    # Paths
    project_root = Path(__file__).parent.parent
    feature_file = project_root / 'data' / 'processed' / 'ablation' / f'corners_features_step{step}.csv'
    labels_file = project_root / 'data' / 'processed' / 'corner_labels.csv'

    output_dir = project_root / 'results' / 'ablation' / f'step{step}' / task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_feature_data(feature_file, labels_file)

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)

    # Get feature columns (exclude metadata and labels)
    metadata_cols = ['match_id', 'event_id', 'event_timestamp']
    label_cols = ['next_event_type', 'leads_to_shot', 'next_event_name']
    feature_cols = [col for col in df.columns if col not in metadata_cols + label_cols]

    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols[:5]}...")

    # Determine label column
    if task == '4class':
        label_col = 'next_event_type'
    else:
        label_col = 'leads_to_shot'

    # Prepare features and labels
    X_train, y_train = prepare_features_labels(train_df, feature_cols, label_col)
    X_val, y_val = prepare_features_labels(val_df, feature_cols, label_col)
    X_test, y_test = prepare_features_labels(test_df, feature_cols, label_col)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train distribution: {np.bincount(y_train)}")

    # Train models
    results = {}

    # Random Forest
    rf_model, rf_train_time = train_random_forest(X_train, y_train, X_val, y_val)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, model_type='rf')
    rf_metrics['train_time'] = rf_train_time
    results['random_forest'] = rf_metrics

    # Save RF model
    rf_model_path = output_dir / 'random_forest.pkl'
    with open(rf_model_path, 'wb') as f:
        pickle.dump(rf_model, f)

    # XGBoost
    xgb_model, xgb_train_time = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, model_type='xgb')
    xgb_metrics['train_time'] = xgb_train_time
    results['xgboost'] = xgb_metrics

    # Save XGBoost model
    xgb_model_path = output_dir / 'xgboost.json'
    xgb_model.save_model(xgb_model_path)

    # MLP
    mlp_model, mlp_train_time = train_mlp(X_train, y_train, X_val, y_val)
    mlp_metrics = evaluate_model(mlp_model, X_test, y_test, model_type='mlp')
    mlp_metrics['train_time'] = mlp_train_time
    results['mlp'] = mlp_metrics

    # Save MLP model
    mlp_model_path = output_dir / 'mlp.pth'
    torch.save(mlp_model[0].state_dict(), mlp_model_path)

    # Save scaler
    scaler_path = output_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(mlp_model[1], f)

    # Save results
    results_path = output_dir / 'metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    return results


def main():
    """Run ablation study for all steps and tasks."""
    steps = list(range(10))  # 0-9
    tasks = ['4class', 'binary_shot']

    all_results = {}

    for step in steps:
        for task in tasks:
            key = f'step{step}_{task}'
            results = run_ablation_study(step, task)
            all_results[key] = results

    # Save combined results
    project_root = Path(__file__).parent.parent
    combined_results_path = project_root / 'results' / 'ablation' / 'all_results.json'
    combined_results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ All results saved to: {combined_results_path}")


if __name__ == '__main__':
    main()
