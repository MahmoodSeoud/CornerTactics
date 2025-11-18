"""
Ablation Study Analysis

Generates:
1. Performance progression plots (accuracy vs feature count)
2. Feature group contribution tables
3. Correlation matrices (feature-feature, feature-outcome)
4. Transition matrices P(outcome | features)
5. Feature importance evolution
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import pickle


# ============ DATA LOADING ============

def load_all_results(results_dir):
    """Load all ablation results."""
    results_file = results_dir / 'all_results.json'
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def load_feature_data(step, data_dir):
    """Load feature CSV for a specific step."""
    feature_file = data_dir / 'ablation' / f'corners_features_step{step}.csv'
    return pd.read_csv(feature_file)


# ============ PERFORMANCE PROGRESSION ============

def plot_performance_progression(results, output_dir):
    """Plot accuracy vs feature count for all models."""
    steps = list(range(10))
    tasks = ['4class', 'binary_shot']
    models = ['random_forest', 'xgboost', 'mlp']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for task_idx, task in enumerate(tasks):
        ax = axes[task_idx]

        for model in models:
            accuracies = []
            for step in steps:
                key = f'step{step}_{task}'
                accuracy = results[key][model]['accuracy']
                accuracies.append(accuracy)

            ax.plot(steps, accuracies, marker='o', label=model.replace('_', ' ').title())

        ax.set_xlabel('Feature Step')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{task.replace("_", " ").title()} - Accuracy vs Feature Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(steps)

    plt.tight_layout()
    output_path = output_dir / 'performance_progression.png'
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved performance progression plot to: {output_path}")
    plt.close()


def plot_feature_contribution(results, output_dir):
    """Plot incremental feature contribution."""
    steps = list(range(10))
    task = '4class'  # Focus on 4-class task
    model = 'random_forest'  # Focus on best model

    accuracies = []
    for step in steps:
        key = f'step{step}_{task}'
        accuracy = results[key][model]['accuracy']
        accuracies.append(accuracy)

    # Calculate deltas
    deltas = [0] + [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]

    # Feature groups
    feature_groups = [
        'Raw (27)',
        '+ Player Counts (6)',
        '+ Spatial Density (4)',
        '+ Positional (8)',
        '+ Pass Technique (2)',
        '+ Pass Outcome (4)',
        '+ Goalkeeper (3)',
        '+ Score State (4)',
        '+ Substitutions (3)',
        '+ Metadata (2)'
    ]

    # Create table
    table_data = []
    cumulative_gain = 0
    for i, (step, acc, delta, group) in enumerate(zip(steps, accuracies, deltas, feature_groups)):
        if i > 0:
            cumulative_gain += delta
        table_data.append({
            'Step': step,
            'Features Added': group,
            'Accuracy': f'{acc:.4f}',
            'Δ Accuracy': f'{delta:+.4f}',
            'Cumulative Gain': f'{cumulative_gain:+.4f}'
        })

    df = pd.DataFrame(table_data)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.3, 0.15, 0.15, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color code by contribution
    for i in range(1, len(table_data) + 1):
        delta_val = float(table_data[i-1]['Δ Accuracy'])
        if delta_val > 0.05:
            color = '#d4f1d4'  # Light green for high contribution
        elif delta_val > 0.02:
            color = '#fff3cd'  # Light yellow for medium
        else:
            color = '#f8d7da'  # Light red for low

        table[(i, 3)].set_facecolor(color)

    plt.title(f'Feature Group Contribution ({task.upper()}, {model.replace("_", " ").title()})', fontsize=14, pad=20)
    plt.tight_layout()

    output_path = output_dir / 'feature_contribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature contribution table to: {output_path}")
    plt.close()

    return df


# ============ CORRELATION ANALYSIS ============

def compute_feature_correlation(data_dir, output_dir):
    """Compute and visualize feature-feature correlation matrix."""
    # Load full feature set (step 9)
    df = load_feature_data(9, data_dir)

    # Get feature columns
    metadata_cols = ['match_id', 'event_id', 'event_timestamp']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        corr_matrix,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title('Feature-Feature Correlation Matrix (Full Feature Set)', fontsize=16)
    plt.tight_layout()

    output_path = output_dir / 'feature_correlation_matrix.png'
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved feature correlation matrix to: {output_path}")
    plt.close()

    # Find highly correlated pairs (>0.8 or <-0.5)
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })

    high_corr_df = pd.DataFrame(high_corr)
    high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)

    # Save to CSV
    high_corr_path = output_dir / 'high_correlation_pairs.csv'
    high_corr_df.to_csv(high_corr_path, index=False)
    print(f"✓ Saved high correlation pairs to: {high_corr_path}")

    return corr_matrix


def compute_feature_outcome_correlation(data_dir, labels_file, output_dir):
    """Compute correlation between features and outcomes."""
    # Load full feature set (step 9)
    df_features = load_feature_data(9, data_dir)

    # Load labels
    df_labels = pd.read_csv(labels_file)

    # Merge
    df = df_features.merge(df_labels, on='event_id', how='inner')

    # Get feature columns
    metadata_cols = ['match_id', 'event_id', 'event_timestamp', 'next_event_type', 'leads_to_shot', 'next_event_name']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Compute correlation with outcome
    outcome_corr = {}
    for feature in feature_cols:
        # 4-class outcome correlation
        corr_4class = df[[feature, 'next_event_type']].corr().iloc[0, 1]

        # Binary shot correlation
        corr_shot = df[[feature, 'leads_to_shot']].corr().iloc[0, 1]

        outcome_corr[feature] = {
            '4class': corr_4class,
            'binary_shot': corr_shot
        }

    # Convert to DataFrame
    corr_df = pd.DataFrame(outcome_corr).T
    corr_df = corr_df.sort_values('4class', key=abs, ascending=False)

    # Save to CSV
    corr_path = output_dir / 'feature_outcome_correlation.csv'
    corr_df.to_csv(corr_path)
    print(f"✓ Saved feature-outcome correlation to: {corr_path}")

    # Plot top 20 features
    top_features = corr_df.head(20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # 4-class
    ax = axes[0]
    top_features['4class'].plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Correlation with 4-Class Outcome')
    ax.set_title('Top 20 Features - 4-Class Outcome Correlation')
    ax.grid(True, alpha=0.3)

    # Binary shot
    ax = axes[1]
    top_features['binary_shot'].plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Correlation with Binary Shot Outcome')
    ax.set_title('Top 20 Features - Binary Shot Outcome Correlation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'feature_outcome_correlation.png'
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved feature-outcome correlation plot to: {output_path}")
    plt.close()

    return corr_df


# ============ TRANSITION MATRICES ============

def compute_transition_matrix(results_dir, step, task='4class'):
    """
    Compute transition matrix P(predicted | actual) for a given step.

    This shows how well the model discriminates between outcome classes.
    """
    # Load model
    model_dir = results_dir / f'step{step}' / task
    model_path = model_dir / 'random_forest.pkl'

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load test data (we need to re-create splits - simplified here)
    # For now, just load confusion matrix from metrics
    metrics_path = model_dir / 'metrics.json'
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    cm = np.array(metrics['random_forest']['confusion_matrix'])

    # Normalize to get P(predicted | actual)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    return cm_normalized


def plot_transition_matrices(results_dir, output_dir):
    """Plot transition matrices for baseline and full feature sets."""
    steps = [0, 9]  # Baseline and full
    task = '4class'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    class_names = ['Ball Receipt', 'Clearance', 'Goalkeeper', 'Other']

    for idx, step in enumerate(steps):
        ax = axes[idx]
        cm = compute_transition_matrix(results_dir.parent / 'ablation', step, task)

        sns.heatmap(
            cm,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            vmin=0,
            vmax=1
        )

        title = f'Step {step} (' + ('Raw Features' if step == 0 else 'Full Features') + ')'
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    plt.tight_layout()
    output_path = output_dir / 'transition_matrices.png'
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved transition matrices to: {output_path}")
    plt.close()


# ============ FEATURE IMPORTANCE ============

def plot_feature_importance_evolution(results_dir, data_dir, output_dir):
    """Plot how feature importance changes across steps."""
    # Load Random Forest models from key steps
    steps = [0, 3, 6, 9]  # Baseline, mid, late, full
    task = '4class'

    importance_data = {}

    for step in steps:
        # Load model
        model_path = results_dir / f'step{step}' / task / 'random_forest.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Get feature names
        df = load_feature_data(step, data_dir)
        metadata_cols = ['match_id', 'event_id', 'event_timestamp']
        feature_cols = [col for col in df.columns if col not in metadata_cols]

        # Get feature importance
        importances = model.feature_importances_

        importance_data[f'Step {step}'] = dict(zip(feature_cols, importances))

    # Convert to DataFrame
    importance_df = pd.DataFrame(importance_data).fillna(0)

    # Plot top 15 features from full model
    top_features = importance_df['Step 9'].nlargest(15).index

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(top_features))
    width = 0.2

    for i, step_key in enumerate(['Step 0', 'Step 3', 'Step 6', 'Step 9']):
        values = [importance_df.loc[feat, step_key] if feat in importance_df.index else 0
                  for feat in top_features]
        ax.barh(x + i*width, values, width, label=step_key)

    ax.set_yticks(x + width * 1.5)
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance Evolution (Top 15 Features)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = output_dir / 'feature_importance_evolution.png'
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved feature importance evolution to: {output_path}")
    plt.close()

    # Save importance data
    importance_path = output_dir / 'feature_importance_all_steps.csv'
    importance_df.to_csv(importance_path)
    print(f"✓ Saved feature importance data to: {importance_path}")


# ============ SUMMARY REPORT ============

def generate_summary_report(results, contribution_df, output_dir):
    """Generate markdown summary report."""
    report = []
    report.append("# Ablation Study Results\n")
    report.append(f"Generated: {pd.Timestamp.now()}\n\n")

    # Overall findings
    report.append("## Key Findings\n\n")

    # Best performance
    step9_4class_rf = results['step9_4class']['random_forest']
    step9_shot_rf = results['step9_binary_shot']['random_forest']

    report.append(f"### Best Performance (Step 9 - Full Features)\n\n")
    report.append(f"- **4-Class Prediction** (Random Forest): {step9_4class_rf['accuracy']:.4f} accuracy\n")
    report.append(f"- **Binary Shot Prediction** (Random Forest): {step9_shot_rf['accuracy']:.4f} accuracy\n\n")

    # Baseline performance
    step0_4class_rf = results['step0_4class']['random_forest']
    step0_shot_rf = results['step0_binary_shot']['random_forest']

    report.append(f"### Baseline Performance (Step 0 - Raw Features)\n\n")
    report.append(f"- **4-Class Prediction** (Random Forest): {step0_4class_rf['accuracy']:.4f} accuracy\n")
    report.append(f"- **Binary Shot Prediction** (Random Forest): {step0_shot_rf['accuracy']:.4f} accuracy\n\n")

    # Improvement
    improvement_4class = step9_4class_rf['accuracy'] - step0_4class_rf['accuracy']
    improvement_shot = step9_shot_rf['accuracy'] - step0_shot_rf['accuracy']

    report.append(f"### Overall Improvement\n\n")
    report.append(f"- **4-Class**: +{improvement_4class:.4f} ({improvement_4class*100:.2f}% absolute gain)\n")
    report.append(f"- **Binary Shot**: +{improvement_shot:.4f} ({improvement_shot*100:.2f}% absolute gain)\n\n")

    # Feature contribution table
    report.append("## Feature Group Contribution\n\n")
    report.append(contribution_df.to_markdown(index=False))
    report.append("\n\n")

    # Save report
    report_path = output_dir / 'ABLATION_SUMMARY.md'
    with open(report_path, 'w') as f:
        f.write(''.join(report))

    print(f"✓ Saved summary report to: {report_path}")


# ============ MAIN ============

def main():
    """Run complete ablation analysis."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results' / 'ablation'
    data_dir = project_root / 'data' / 'processed'
    labels_file = data_dir / 'corner_labels.csv'
    output_dir = project_root / 'results' / 'ablation' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Ablation Study Analysis ===\n")

    # Load results
    print("Loading results...")
    results = load_all_results(results_dir)

    # Performance progression
    print("\nGenerating performance progression plots...")
    plot_performance_progression(results, output_dir)

    # Feature contribution
    print("\nGenerating feature contribution analysis...")
    contribution_df = plot_feature_contribution(results, output_dir)

    # Correlation analysis
    print("\nComputing correlation matrices...")
    compute_feature_correlation(data_dir, output_dir)
    compute_feature_outcome_correlation(data_dir, labels_file, output_dir)

    # Transition matrices
    print("\nComputing transition matrices...")
    plot_transition_matrices(results_dir, output_dir)

    # Feature importance
    print("\nAnalyzing feature importance evolution...")
    plot_feature_importance_evolution(results_dir, data_dir, output_dir)

    # Summary report
    print("\nGenerating summary report...")
    generate_summary_report(results, contribution_df, output_dir)

    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
