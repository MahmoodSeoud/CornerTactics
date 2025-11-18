"""
Generate Predictions for Confusion Matrix

Trains optimal 24-feature model and saves predictions for confusion matrix visualization.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(features_file, labels_file):
    """Load features and labels."""
    features_df = pd.read_csv(features_file)
    labels_df = pd.read_csv(labels_file)
    merged = features_df.merge(labels_df, on='event_id', how='inner')
    return merged


def prepare_splits(df, random_seed=42):
    """Create match-based train/val/test splits."""
    unique_matches = df['match_id'].unique()

    train_matches, temp_matches = train_test_split(
        unique_matches, test_size=0.3, random_state=random_seed
    )
    val_matches, test_matches = train_test_split(
        temp_matches, test_size=0.5, random_state=random_seed
    )

    train_df = df[df['match_id'].isin(train_matches)]
    val_df = df[df['match_id'].isin(val_matches)]
    test_df = df[df['match_id'].isin(test_matches)]

    return train_df, val_df, test_df


def plot_confusion_matrix(cm, class_names, output_path, title='Confusion Matrix'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved confusion matrix to: {output_path}")


def plot_normalized_confusion_matrix(cm, class_names, output_path, title='Normalized Confusion Matrix'):
    """Plot and save normalized confusion matrix (percentages)."""
    # Normalize by row (true label)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))

    # Create heatmap with percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved normalized confusion matrix to: {output_path}")


def main():
    """Generate predictions and confusion matrices."""
    project_root = Path(__file__).parent.parent

    # Load optimal feature set
    optimal_sets_file = project_root / 'results' / 'optimal_search' / 'optimal_feature_sets.json'

    if not optimal_sets_file.exists():
        print(f"ERROR: Optimal feature sets not found at {optimal_sets_file}")
        print("Run scripts/12_optimal_feature_search.py first!")
        return

    with open(optimal_sets_file, 'r') as f:
        optimal_sets = json.load(f)

    # Use bidirectional search result (best)
    optimal_features = optimal_sets['bidirectional_search']

    print(f"Loaded optimal feature set: {len(optimal_features)} features")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(project_root / 'data' / 'processed' / 'ablation' / 'corners_features_step9.csv')
    labels_df = pd.read_csv(project_root / 'data' / 'processed' / 'corner_labels.csv')
    df = df.merge(labels_df, on='event_id', how='inner')

    print(f"✓ Loaded {len(df)} corners")

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} corners")
    print(f"  Val:   {len(val_df)} corners")
    print(f"  Test:  {len(test_df)} corners")

    # Output directory
    output_dir = project_root / 'results' / 'optimal_search' / 'confusion_matrices'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # BINARY SHOT PREDICTION (leads_to_shot)
    # ============================================================

    print("\n" + "="*70)
    print("BINARY SHOT PREDICTION (leads_to_shot)")
    print("="*70)

    # Prepare data
    X_train = train_df[optimal_features].values
    y_train = train_df['leads_to_shot'].values

    X_test = test_df[optimal_features].values
    y_test = test_df['leads_to_shot'].values

    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix (Binary Shot):")
    print(cm)

    # Class names
    class_names_binary = ['No Shot', 'Shot']

    # Save predictions
    predictions_df = pd.DataFrame({
        'event_id': test_df['event_id'].values,
        'match_id': test_df['match_id'].values,
        'true_label': y_test,
        'predicted_label': y_pred,
        'probability_no_shot': y_proba[:, 0],
        'probability_shot': y_proba[:, 1],
        'correct': (y_test == y_pred).astype(int)
    })

    predictions_file = output_dir / 'binary_shot_predictions.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"\n✓ Saved predictions to: {predictions_file}")

    # Save confusion matrix data
    cm_df = pd.DataFrame(cm,
                         index=class_names_binary,
                         columns=class_names_binary)
    cm_file = output_dir / 'binary_shot_confusion_matrix.csv'
    cm_df.to_csv(cm_file)
    print(f"✓ Saved confusion matrix data to: {cm_file}")

    # Plot confusion matrices
    plot_confusion_matrix(
        cm,
        class_names_binary,
        output_dir / 'binary_shot_confusion_matrix.png',
        title='Binary Shot Prediction - Confusion Matrix\n(Optimal 24 Features, RF, Test Set)'
    )

    plot_normalized_confusion_matrix(
        cm,
        class_names_binary,
        output_dir / 'binary_shot_confusion_matrix_normalized.png',
        title='Binary Shot Prediction - Normalized Confusion Matrix\n(Optimal 24 Features, RF, Test Set)'
    )

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names_binary)
    print("\nClassification Report (Binary Shot):")
    print(report)

    report_file = output_dir / 'binary_shot_classification_report.txt'
    with open(report_file, 'w') as f:
        f.write("Binary Shot Prediction - Classification Report\n")
        f.write("="*70 + "\n\n")
        f.write(report)
    print(f"✓ Saved classification report to: {report_file}")

    # ============================================================
    # 4-CLASS OUTCOME PREDICTION (next_event_type)
    # ============================================================

    print("\n" + "="*70)
    print("4-CLASS OUTCOME PREDICTION (next_event_type)")
    print("="*70)

    # Prepare data
    y_train_4class = train_df['next_event_type'].values
    y_test_4class = test_df['next_event_type'].values

    # Train model
    print("\nTraining Random Forest...")
    model_4class = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model_4class.fit(X_train, y_train_4class)

    # Get predictions
    y_pred_4class = model_4class.predict(X_test)
    y_proba_4class = model_4class.predict_proba(X_test)

    # Calculate confusion matrix
    cm_4class = confusion_matrix(y_test_4class, y_pred_4class)

    print("\nConfusion Matrix (4-Class):")
    print(cm_4class)

    # Get class names from data
    class_names_4class = sorted(df['next_event_type'].unique())
    class_names_4class_str = [str(c) for c in class_names_4class]  # Convert to strings for sklearn

    print(f"\nClass mapping: {class_names_4class}")

    # Save predictions
    predictions_4class_df = pd.DataFrame({
        'event_id': test_df['event_id'].values,
        'match_id': test_df['match_id'].values,
        'true_label': y_test_4class,
        'predicted_label': y_pred_4class,
        'correct': (y_test_4class == y_pred_4class).astype(int)
    })

    # Add probability columns for each class
    for idx, class_name in enumerate(class_names_4class_str):
        predictions_4class_df[f'probability_class_{class_name}'] = y_proba_4class[:, idx]

    predictions_4class_file = output_dir / '4class_outcome_predictions.csv'
    predictions_4class_df.to_csv(predictions_4class_file, index=False)
    print(f"\n✓ Saved predictions to: {predictions_4class_file}")

    # Save confusion matrix data
    cm_4class_df = pd.DataFrame(cm_4class,
                                 index=class_names_4class_str,
                                 columns=class_names_4class_str)
    cm_4class_file = output_dir / '4class_outcome_confusion_matrix.csv'
    cm_4class_df.to_csv(cm_4class_file)
    print(f"✓ Saved confusion matrix data to: {cm_4class_file}")

    # Plot confusion matrices
    plot_confusion_matrix(
        cm_4class,
        class_names_4class_str,
        output_dir / '4class_outcome_confusion_matrix.png',
        title='4-Class Outcome Prediction - Confusion Matrix\n(Optimal 24 Features, RF, Test Set)'
    )

    plot_normalized_confusion_matrix(
        cm_4class,
        class_names_4class_str,
        output_dir / '4class_outcome_confusion_matrix_normalized.png',
        title='4-Class Outcome Prediction - Normalized Confusion Matrix\n(Optimal 24 Features, RF, Test Set)'
    )

    # Classification report
    report_4class = classification_report(y_test_4class, y_pred_4class, target_names=class_names_4class_str)
    print("\nClassification Report (4-Class):")
    print(report_4class)

    report_4class_file = output_dir / '4class_outcome_classification_report.txt'
    with open(report_4class_file, 'w') as f:
        f.write("4-Class Outcome Prediction - Classification Report\n")
        f.write("="*70 + "\n\n")
        f.write(report_4class)
    print(f"✓ Saved classification report to: {report_4class_file}")

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  Binary Shot Prediction:")
    print("    - binary_shot_predictions.csv")
    print("    - binary_shot_confusion_matrix.csv")
    print("    - binary_shot_confusion_matrix.png")
    print("    - binary_shot_confusion_matrix_normalized.png")
    print("    - binary_shot_classification_report.txt")
    print("\n  4-Class Outcome Prediction:")
    print("    - 4class_outcome_predictions.csv")
    print("    - 4class_outcome_confusion_matrix.csv")
    print("    - 4class_outcome_confusion_matrix.png")
    print("    - 4class_outcome_confusion_matrix_normalized.png")
    print("    - 4class_outcome_classification_report.txt")

    print("\n" + "="*70)
    print("Use the CSV files to create custom confusion matrix visualizations!")
    print("="*70)


if __name__ == '__main__':
    main()
