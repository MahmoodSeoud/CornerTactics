"""
Individual Feature Ablation Study

Phase 1: Raw Feature Analysis (Leave-One-Out)
  - Remove each of 27 raw features individually
  - Measure performance drop
  - Identifies which raw features are necessary

Phase 2: Engineered Feature Ranking (Univariate)
  - Add each of 34 engineered features individually (on top of all raw)
  - Rank by performance gain
  - Identifies best individual engineered features

Phase 3: Minimal Feature Set (Forward Selection)
  - Start with best feature from Phase 2
  - Greedily add next best feature
  - Stop when improvement < 0.5%
  - Finds minimal viable feature set
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import pickle


# ============ HELPER FUNCTIONS ============

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


def train_evaluate_rf(X_train, y_train, X_test, y_test, random_seed=42):
    """Train Random Forest and return test metrics."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted')
    }

    # Add ROC-AUC for binary tasks
    if len(np.unique(y_test)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])

    return metrics, model


# ============ PHASE 1: RAW FEATURE LEAVE-ONE-OUT ============

def phase1_raw_feature_loo(df, output_dir):
    """
    Leave-one-out ablation on raw features.

    Tests: What happens if we remove each raw feature?
    """
    print("\n" + "="*60)
    print("PHASE 1: RAW FEATURE LEAVE-ONE-OUT ANALYSIS")
    print("="*60)

    # Get raw features (from Step 0)
    metadata_cols = ['match_id', 'event_id', 'event_timestamp']
    label_cols = ['next_event_name', 'next_event_type', 'leads_to_shot']

    all_cols = df.columns.tolist()
    raw_features = [col for col in all_cols if col not in metadata_cols + label_cols][:27]

    print(f"\nRaw features to test: {len(raw_features)}")

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)

    results = []

    # Baseline: All raw features
    X_train = train_df[raw_features].values
    y_train_4class = train_df['next_event_type'].values
    y_train_shot = train_df['leads_to_shot'].values

    X_test = test_df[raw_features].values
    y_test_4class = test_df['next_event_type'].values
    y_test_shot = test_df['leads_to_shot'].values

    print("\n--- Baseline (all 27 raw features) ---")
    baseline_4class, _ = train_evaluate_rf(X_train, y_train_4class, X_test, y_test_4class)
    baseline_shot, _ = train_evaluate_rf(X_train, y_train_shot, X_test, y_test_shot)

    print(f"4-Class: {baseline_4class['accuracy']:.4f}")
    print(f"Binary Shot: {baseline_shot['accuracy']:.4f}")

    results.append({
        'feature_removed': 'NONE (baseline)',
        'num_features': len(raw_features),
        '4class_accuracy': baseline_4class['accuracy'],
        '4class_f1': baseline_4class['f1_macro'],
        'shot_accuracy': baseline_shot['accuracy'],
        'shot_roc_auc': baseline_shot['roc_auc'],
        '4class_drop': 0.0,
        'shot_drop': 0.0
    })

    # Leave-one-out
    for feature in tqdm(raw_features, desc="Testing raw features"):
        # Remove this feature
        remaining_features = [f for f in raw_features if f != feature]

        X_train_loo = train_df[remaining_features].values
        X_test_loo = test_df[remaining_features].values

        # Train and evaluate
        metrics_4class, _ = train_evaluate_rf(X_train_loo, y_train_4class, X_test_loo, y_test_4class)
        metrics_shot, _ = train_evaluate_rf(X_train_loo, y_train_shot, X_test_loo, y_test_shot)

        results.append({
            'feature_removed': feature,
            'num_features': len(remaining_features),
            '4class_accuracy': metrics_4class['accuracy'],
            '4class_f1': metrics_4class['f1_macro'],
            'shot_accuracy': metrics_shot['accuracy'],
            'shot_roc_auc': metrics_shot['roc_auc'],
            '4class_drop': baseline_4class['accuracy'] - metrics_4class['accuracy'],
            'shot_drop': baseline_shot['accuracy'] - metrics_shot['accuracy']
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('shot_drop', ascending=False)

    output_file = output_dir / 'phase1_raw_feature_loo.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")

    # Print top 10 most important raw features
    print("\n=== Top 10 Most Important Raw Features (by shot prediction drop) ===")
    top10 = results_df[results_df['feature_removed'] != 'NONE (baseline)'].head(10)
    for _, row in top10.iterrows():
        print(f"{row['feature_removed']:30s} | Drop: {row['shot_drop']:+.4f} (shot), {row['4class_drop']:+.4f} (4class)")

    return results_df


# ============ PHASE 2: ENGINEERED FEATURE RANKING ============

def phase2_engineered_feature_ranking(df_raw, df_full, output_dir):
    """
    Univariate evaluation of engineered features.

    Tests: Add each engineered feature individually on top of raw features.
    """
    print("\n" + "="*60)
    print("PHASE 2: ENGINEERED FEATURE RANKING (UNIVARIATE)")
    print("="*60)

    # Get feature sets
    metadata_cols = ['match_id', 'event_id', 'event_timestamp']
    label_cols = ['next_event_name', 'next_event_type', 'leads_to_shot']

    raw_features = [col for col in df_raw.columns if col not in metadata_cols + label_cols][:27]
    all_features = [col for col in df_full.columns if col not in metadata_cols + label_cols]
    engineered_features = [f for f in all_features if f not in raw_features]

    print(f"\nRaw features: {len(raw_features)}")
    print(f"Engineered features to test: {len(engineered_features)}")

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df_full)

    # Baseline: Raw features only
    X_train_raw = train_df[raw_features].values
    X_test_raw = test_df[raw_features].values
    y_train_4class = train_df['next_event_type'].values
    y_train_shot = train_df['leads_to_shot'].values
    y_test_4class = test_df['next_event_type'].values
    y_test_shot = test_df['leads_to_shot'].values

    print("\n--- Baseline (27 raw features only) ---")
    baseline_4class, _ = train_evaluate_rf(X_train_raw, y_train_4class, X_test_raw, y_test_4class)
    baseline_shot, _ = train_evaluate_rf(X_train_raw, y_train_shot, X_test_raw, y_test_shot)

    print(f"4-Class: {baseline_4class['accuracy']:.4f}")
    print(f"Binary Shot: {baseline_shot['accuracy']:.4f}")

    results = []

    results.append({
        'feature_added': 'NONE (baseline)',
        'num_features': len(raw_features),
        '4class_accuracy': baseline_4class['accuracy'],
        '4class_f1': baseline_4class['f1_macro'],
        'shot_accuracy': baseline_shot['accuracy'],
        'shot_roc_auc': baseline_shot['roc_auc'],
        '4class_gain': 0.0,
        'shot_gain': 0.0
    })

    # Test each engineered feature individually
    for feature in tqdm(engineered_features, desc="Testing engineered features"):
        # Add this feature to raw features
        features_to_use = raw_features + [feature]

        X_train_eng = train_df[features_to_use].values
        X_test_eng = test_df[features_to_use].values

        # Train and evaluate
        metrics_4class, _ = train_evaluate_rf(X_train_eng, y_train_4class, X_test_eng, y_test_4class)
        metrics_shot, _ = train_evaluate_rf(X_train_eng, y_train_shot, X_test_eng, y_test_shot)

        results.append({
            'feature_added': feature,
            'num_features': len(features_to_use),
            '4class_accuracy': metrics_4class['accuracy'],
            '4class_f1': metrics_4class['f1_macro'],
            'shot_accuracy': metrics_shot['accuracy'],
            'shot_roc_auc': metrics_shot['roc_auc'],
            '4class_gain': metrics_4class['accuracy'] - baseline_4class['accuracy'],
            'shot_gain': metrics_shot['accuracy'] - baseline_shot['accuracy']
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('shot_gain', ascending=False)

    output_file = output_dir / 'phase2_engineered_feature_ranking.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")

    # Print top 10 best engineered features
    print("\n=== Top 10 Best Engineered Features (by shot prediction gain) ===")
    top10 = results_df[results_df['feature_added'] != 'NONE (baseline)'].head(10)
    for _, row in top10.iterrows():
        print(f"{row['feature_added']:30s} | Gain: {row['shot_gain']:+.4f} (shot), {row['4class_gain']:+.4f} (4class)")

    return results_df


# ============ PHASE 3: MINIMAL FEATURE SET (FORWARD SELECTION) ============

def phase3_forward_selection(df, engineered_ranking, output_dir, threshold=0.005):
    """
    Forward selection to find minimal feature set.

    Start with best feature, greedily add next best until gain < threshold.
    """
    print("\n" + "="*60)
    print("PHASE 3: MINIMAL FEATURE SET (FORWARD SELECTION)")
    print("="*60)
    print(f"Stopping criterion: gain < {threshold:.3f}\n")

    # Get feature candidates (sorted by shot prediction gain)
    candidates = engineered_ranking[engineered_ranking['feature_added'] != 'NONE (baseline)']
    candidate_features = candidates['feature_added'].tolist()

    # Get raw features
    metadata_cols = ['match_id', 'event_id', 'event_timestamp']
    label_cols = ['next_event_name', 'next_event_type', 'leads_to_shot']
    raw_features = [col for col in df.columns if col not in metadata_cols + label_cols][:27]

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)

    y_train_shot = train_df['leads_to_shot'].values
    y_test_shot = test_df['leads_to_shot'].values

    # Start with raw features
    selected_features = raw_features.copy()
    remaining_candidates = candidate_features.copy()

    # Baseline
    X_train = train_df[selected_features].values
    X_test = test_df[selected_features].values
    baseline_metrics, _ = train_evaluate_rf(X_train, y_train_shot, X_test, y_test_shot)

    current_accuracy = baseline_metrics['accuracy']
    print(f"Baseline (27 raw): {current_accuracy:.4f}")

    results = []
    results.append({
        'step': 0,
        'feature_added': 'RAW FEATURES',
        'num_features': len(selected_features),
        'accuracy': current_accuracy,
        'roc_auc': baseline_metrics['roc_auc'],
        'gain': 0.0
    })

    # Forward selection
    step = 1
    while remaining_candidates:
        best_feature = None
        best_accuracy = current_accuracy
        best_metrics = None

        # Try adding each remaining candidate
        for candidate in remaining_candidates:
            test_features = selected_features + [candidate]

            X_train_test = train_df[test_features].values
            X_test_test = test_df[test_features].values

            metrics, _ = train_evaluate_rf(X_train_test, y_train_shot, X_test_test, y_test_shot)

            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_feature = candidate
                best_metrics = metrics

        # Check stopping criterion
        gain = best_accuracy - current_accuracy

        if gain < threshold:
            print(f"\nStopping: Best remaining feature '{best_feature}' only improves by {gain:.4f} < {threshold:.3f}")
            break

        # Add best feature
        selected_features.append(best_feature)
        remaining_candidates.remove(best_feature)
        current_accuracy = best_accuracy

        print(f"Step {step}: +{best_feature:30s} | Accuracy: {current_accuracy:.4f} (+{gain:.4f})")

        results.append({
            'step': step,
            'feature_added': best_feature,
            'num_features': len(selected_features),
            'accuracy': current_accuracy,
            'roc_auc': best_metrics['roc_auc'],
            'gain': gain
        })

        step += 1

        # Safety: Stop after 20 features
        if len(selected_features) >= 47:  # 27 raw + 20 engineered
            print("\nStopping: Reached 47 features (safety limit)")
            break

    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'phase3_forward_selection.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")

    # Save minimal feature set
    minimal_features_file = output_dir / 'minimal_feature_set.txt'
    with open(minimal_features_file, 'w') as f:
        f.write(f"# Minimal Feature Set ({len(selected_features)} features)\n")
        f.write(f"# Shot Prediction Accuracy: {current_accuracy:.4f}\n\n")
        f.write("## Raw Features (27)\n")
        for feat in raw_features:
            f.write(f"{feat}\n")
        f.write(f"\n## Engineered Features ({len(selected_features) - 27})\n")
        for feat in selected_features[27:]:
            f.write(f"{feat}\n")

    print(f"✓ Saved minimal feature set to: {minimal_features_file}")
    print(f"\n=== FINAL MINIMAL FEATURE SET ===")
    print(f"Total features: {len(selected_features)}")
    print(f"  - Raw: 27")
    print(f"  - Engineered: {len(selected_features) - 27}")
    print(f"Shot prediction accuracy: {current_accuracy:.4f}")

    return results_df, selected_features


# ============ MAIN ============

def main():
    """Run all three phases of individual feature ablation."""
    project_root = Path(__file__).parent.parent

    # Load data
    df_raw = pd.read_csv(project_root / 'data' / 'processed' / 'ablation' / 'corners_features_step0.csv')
    df_full = pd.read_csv(project_root / 'data' / 'processed' / 'ablation' / 'corners_features_step9.csv')
    labels_df = pd.read_csv(project_root / 'data' / 'processed' / 'corner_labels.csv')

    # Merge labels
    df_raw = df_raw.merge(labels_df, on='event_id', how='inner')
    df_full = df_full.merge(labels_df, on='event_id', how='inner')

    # Output directory
    output_dir = project_root / 'results' / 'ablation' / 'individual_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Raw feature leave-one-out
    raw_loo_results = phase1_raw_feature_loo(df_raw, output_dir)

    # Phase 2: Engineered feature ranking
    engineered_ranking = phase2_engineered_feature_ranking(df_raw, df_full, output_dir)

    # Phase 3: Forward selection for minimal set
    forward_results, minimal_features = phase3_forward_selection(df_full, engineered_ranking, output_dir)

    # Generate summary report
    summary_file = output_dir / 'INDIVIDUAL_ABLATION_SUMMARY.md'
    with open(summary_file, 'w') as f:
        f.write("# Individual Feature Ablation Study Results\n\n")

        f.write("## Phase 1: Raw Feature Analysis\n\n")
        f.write("**Top 5 Most Critical Raw Features** (largest performance drop when removed):\n\n")
        top5_raw = raw_loo_results[raw_loo_results['feature_removed'] != 'NONE (baseline)'].head(5)
        for i, row in top5_raw.iterrows():
            f.write(f"{i+1}. `{row['feature_removed']}` - Drop: {row['shot_drop']:.4f} (shot), {row['4class_drop']:.4f} (4class)\n")

        f.write("\n## Phase 2: Engineered Feature Ranking\n\n")
        f.write("**Top 10 Best Engineered Features** (largest performance gain when added):\n\n")
        top10_eng = engineered_ranking[engineered_ranking['feature_added'] != 'NONE (baseline)'].head(10)
        for i, row in top10_eng.iterrows():
            f.write(f"{i+1}. `{row['feature_added']}` - Gain: {row['shot_gain']:.4f} (shot), {row['4class_gain']:.4f} (4class)\n")

        f.write("\n## Phase 3: Minimal Feature Set\n\n")
        f.write(f"**Total features in minimal set:** {len(minimal_features)}\n")
        f.write(f"- Raw features: 27\n")
        f.write(f"- Engineered features: {len(minimal_features) - 27}\n\n")
        f.write(f"**Final accuracy:** {forward_results.iloc[-1]['accuracy']:.4f}\n\n")
        f.write("**Engineered features selected:**\n\n")
        for feat in minimal_features[27:]:
            f.write(f"- {feat}\n")

    print(f"\n✓ Summary report saved to: {summary_file}")
    print(f"\n{'='*60}")
    print("INDIVIDUAL FEATURE ABLATION COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
