"""
Feature Interaction Pair Testing

Tests explicit feature interactions for highly correlated pairs:
1. Test adding/removing correlated pairs together
2. Measure synergy: Is pair better than sum of individual contributions?
3. Harmful feature inclusion test: Do harmful features help in certain contexts?
4. Cross-validation stability check
5. Model comparison across architectures
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import pickle


# ============ FEATURE DEFINITIONS ============

# From previous analysis
HARMFUL_FEATURES = [
    'period', 'team_id', 'pass_height_id',
    'pass_angle', 'location_y', 'minute'
]

# High-correlation pairs
INTERACTION_PAIRS = [
    ('location_y', 'corner_side'),                      # r = 1.00
    ('has_pass_outcome', 'pass_outcome_encoded'),       # r = 0.91
    ('total_attacking', 'attacking_density'),           # r = 1.00
    ('team_id', 'possession_team_id'),                  # r = 1.00
    ('numerical_advantage', 'attacker_defender_ratio'), # r = 0.95
    ('minute', 'index'),                                # r = 0.97
    ('pass_height_id', 'pass_technique_id'),            # r = 0.96
    ('is_inswinging', 'is_outswinging')                 # one-hot pair
]

METADATA_COLS = ['match_id', 'event_id', 'event_timestamp']
LABEL_COLS = ['next_event_name', 'next_event_type', 'leads_to_shot']


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
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba[:, 1])
    }

    return metrics, model


def validate_features(df, feature_list):
    """Validate that features exist in dataframe."""
    all_cols = df.columns.tolist()
    all_features = [col for col in all_cols if col not in METADATA_COLS + LABEL_COLS]
    valid = [f for f in feature_list if f in all_features]
    missing = [f for f in feature_list if f not in all_features]
    if missing:
        print(f"  WARNING: {len(missing)} features not found: {missing}")
    return valid


# ============ PHASE 6: INTERACTION PAIR TESTING ============

def phase6_interaction_pairs(df, optimal_features, output_dir):
    """
    Test feature interaction pairs.

    For each pair:
    1. Test with both members
    2. Test with only first member
    3. Test with only second member
    4. Test with neither member
    5. Calculate synergy score
    """
    print("\n" + "="*70)
    print("PHASE 6: INTERACTION PAIR TESTING")
    print("="*70)

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)
    y_train = train_df['leads_to_shot'].values
    y_test = test_df['leads_to_shot'].values

    results = []

    for pair_idx, (feat1, feat2) in enumerate(INTERACTION_PAIRS):
        print(f"\n--- Pair {pair_idx + 1}/{len(INTERACTION_PAIRS)}: ({feat1}, {feat2}) ---")

        # Check if features exist
        if feat1 not in df.columns or feat2 not in df.columns:
            print(f"  SKIP: One or both features not in data")
            continue

        # Base features (optimal set without this pair)
        base_features = [f for f in optimal_features if f not in [feat1, feat2]]

        # Test 4 configurations
        configs = {
            'neither': base_features,
            'feat1_only': base_features + [feat1],
            'feat2_only': base_features + [feat2],
            'both': base_features + [feat1, feat2]
        }

        config_results = {}

        for config_name, features in configs.items():
            X_train = train_df[features].values
            X_test = test_df[features].values

            metrics, _ = train_evaluate_rf(X_train, y_train, X_test, y_test)
            config_results[config_name] = metrics['accuracy']

            print(f"  {config_name:12s}: {metrics['accuracy']:.4f}")

        # Calculate synergy
        # Synergy = Acc(both) - [Acc(feat1_only) + Acc(feat2_only) - Acc(neither)]
        # Positive synergy: pair together is better than sum of individual contributions
        expected_joint = config_results['feat1_only'] + config_results['feat2_only'] - config_results['neither']
        synergy = config_results['both'] - expected_joint

        print(f"  Synergy: {synergy:+.4f} ({'positive' if synergy > 0 else 'negative'})")

        results.append({
            'feat1': feat1,
            'feat2': feat2,
            'acc_neither': config_results['neither'],
            'acc_feat1_only': config_results['feat1_only'],
            'acc_feat2_only': config_results['feat2_only'],
            'acc_both': config_results['both'],
            'synergy': synergy,
            'feat1_contribution': config_results['feat1_only'] - config_results['neither'],
            'feat2_contribution': config_results['feat2_only'] - config_results['neither'],
            'pair_contribution': config_results['both'] - config_results['neither']
        })

    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'phase6_interaction_pairs.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n✓ Saved interaction pair results to: {output_file}")

    # Print summary
    print("\n=== INTERACTION PAIR SUMMARY ===")
    results_df_sorted = results_df.sort_values('synergy', ascending=False)
    for _, row in results_df_sorted.iterrows():
        print(f"({row['feat1']}, {row['feat2']})")
        print(f"  Synergy: {row['synergy']:+.4f} | Pair contrib: {row['pair_contribution']:+.4f}")

    return results_df


# ============ PHASE 7: HARMFUL FEATURE INCLUSION TEST ============

def phase7_harmful_feature_test(df, optimal_features, harmful_features, output_dir):
    """
    Test if harmful features help in context of optimal feature set.

    Tests:
    1. Add each harmful feature individually
    2. Add all harmful features together
    3. Test with harmful features but without their correlated pairs
    """
    print("\n" + "="*70)
    print("PHASE 7: HARMFUL FEATURE INCLUSION TEST")
    print("="*70)

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)
    y_train = train_df['leads_to_shot'].values
    y_test = test_df['leads_to_shot'].values

    # Baseline: Optimal features without harmful
    valid_harmful = validate_features(df, harmful_features)
    valid_optimal = validate_features(df, optimal_features)

    print(f"\nOptimal features: {len(valid_optimal)}")
    print(f"Harmful features to test: {len(valid_harmful)}")

    X_train_baseline = train_df[valid_optimal].values
    X_test_baseline = test_df[valid_optimal].values
    baseline_metrics, _ = train_evaluate_rf(X_train_baseline, y_train, X_test_baseline, y_test)
    baseline_acc = baseline_metrics['accuracy']

    print(f"\nBaseline (optimal without harmful): {baseline_acc:.4f}")

    results = [{
        'config': 'baseline_optimal',
        'features_added': 'none',
        'num_features': len(valid_optimal),
        'accuracy': baseline_acc,
        'roc_auc': baseline_metrics['roc_auc'],
        'change': 0.0
    }]

    # Test adding each harmful feature individually
    print("\n--- Testing individual harmful features ---")
    for harmful in valid_harmful:
        test_features = valid_optimal + [harmful]
        X_train = train_df[test_features].values
        X_test = test_df[test_features].values

        metrics, _ = train_evaluate_rf(X_train, y_train, X_test, y_test)
        change = metrics['accuracy'] - baseline_acc

        print(f"+ {harmful:20s}: {metrics['accuracy']:.4f} (change: {change:+.4f})")

        results.append({
            'config': 'add_single_harmful',
            'features_added': harmful,
            'num_features': len(test_features),
            'accuracy': metrics['accuracy'],
            'roc_auc': metrics['roc_auc'],
            'change': change
        })

    # Test adding all harmful features
    print("\n--- Testing all harmful features together ---")
    test_features = valid_optimal + valid_harmful
    X_train = train_df[test_features].values
    X_test = test_df[test_features].values

    metrics, _ = train_evaluate_rf(X_train, y_train, X_test, y_test)
    change = metrics['accuracy'] - baseline_acc

    print(f"+ All harmful ({len(valid_harmful)}): {metrics['accuracy']:.4f} (change: {change:+.4f})")

    results.append({
        'config': 'add_all_harmful',
        'features_added': ', '.join(valid_harmful),
        'num_features': len(test_features),
        'accuracy': metrics['accuracy'],
        'roc_auc': metrics['roc_auc'],
        'change': change
    })

    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'phase7_harmful_feature_test.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n✓ Saved harmful feature test results to: {output_file}")

    # Summary
    print("\n=== HARMFUL FEATURE TEST SUMMARY ===")
    print(f"Baseline: {baseline_acc:.4f}")
    harmful_results = results_df[results_df['config'] != 'baseline_optimal']
    best = harmful_results.loc[harmful_results['accuracy'].idxmax()]
    worst = harmful_results.loc[harmful_results['accuracy'].idxmin()]
    print(f"Best addition: {best['features_added']} ({best['change']:+.4f})")
    print(f"Worst addition: {worst['features_added']} ({worst['change']:+.4f})")

    return results_df


# ============ PHASE 8: CROSS-VALIDATION STABILITY ============

def phase8_cross_validation(df, feature_sets, output_dir, n_folds=5):
    """
    Test stability of top feature sets using cross-validation.

    Args:
        feature_sets: dict of {name: feature_list}
    """
    print("\n" + "="*70)
    print("PHASE 8: CROSS-VALIDATION STABILITY CHECK")
    print("="*70)
    print(f"Folds: {n_folds}\n")

    # Prepare data
    all_cols = df.columns.tolist()
    all_features = [col for col in all_cols if col not in METADATA_COLS + LABEL_COLS]

    y = df['leads_to_shot'].values

    results = []

    for set_name, features in feature_sets.items():
        print(f"\n--- {set_name} ({len(features)} features) ---")

        # Validate features
        valid_features = [f for f in features if f in all_features]
        if len(valid_features) < len(features):
            print(f"  WARNING: {len(features) - len(valid_features)} features missing")

        X = df[valid_features].values

        # Cross-validation
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        cv_scores = cross_val_score(model, X, y, cv=n_folds, scoring='accuracy', n_jobs=-1)

        mean_acc = cv_scores.mean()
        std_acc = cv_scores.std()

        print(f"  CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Fold scores: {[f'{score:.4f}' for score in cv_scores]}")

        results.append({
            'feature_set': set_name,
            'num_features': len(valid_features),
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'min_accuracy': cv_scores.min(),
            'max_accuracy': cv_scores.max(),
            'fold_scores': cv_scores.tolist()
        })

    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'phase8_cross_validation.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n✓ Saved cross-validation results to: {output_file}")

    # Summary
    print("\n=== CROSS-VALIDATION SUMMARY ===")
    results_df_sorted = results_df.sort_values('mean_accuracy', ascending=False)
    for _, row in results_df_sorted.iterrows():
        print(f"{row['feature_set']:30s}: {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}")

    return results_df


# ============ PHASE 9: MODEL COMPARISON ============

def phase9_model_comparison(df, optimal_features, output_dir):
    """
    Test optimal feature set across multiple model architectures.

    Models:
    1. Random Forest (current best)
    2. Gradient Boosting
    3. MLP Neural Network
    """
    print("\n" + "="*70)
    print("PHASE 9: MODEL COMPARISON")
    print("="*70)

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)
    valid_features = validate_features(df, optimal_features)

    X_train = train_df[valid_features].values
    X_test = test_df[valid_features].values
    y_train = train_df['leads_to_shot'].values
    y_test = test_df['leads_to_shot'].values

    print(f"\nOptimal features: {len(valid_features)}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    results = []

    # Model 1: Random Forest
    print("\n--- Model 1: Random Forest ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)

    results.append({
        'model': 'RandomForest',
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1_macro': f1_score(y_test, y_pred_rf, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred_rf, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba_rf[:, 1])
    })
    print(f"  Accuracy: {results[-1]['accuracy']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")

    # Model 2: Gradient Boosting
    print("\n--- Model 2: Gradient Boosting ---")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    y_proba_gb = gb_model.predict_proba(X_test)

    results.append({
        'model': 'GradientBoosting',
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'f1_macro': f1_score(y_test, y_pred_gb, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred_gb, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba_gb[:, 1])
    })
    print(f"  Accuracy: {results[-1]['accuracy']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")

    # Model 3: MLP
    print("\n--- Model 3: Multi-Layer Perceptron ---")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_model.fit(X_train, y_train)
    y_pred_mlp = mlp_model.predict(X_test)
    y_proba_mlp = mlp_model.predict_proba(X_test)

    results.append({
        'model': 'MLP',
        'accuracy': accuracy_score(y_test, y_pred_mlp),
        'f1_macro': f1_score(y_test, y_pred_mlp, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred_mlp, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba_mlp[:, 1])
    })
    print(f"  Accuracy: {results[-1]['accuracy']:.4f} | AUC: {results[-1]['roc_auc']:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'phase9_model_comparison.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n✓ Saved model comparison results to: {output_file}")

    # Summary
    print("\n=== MODEL COMPARISON SUMMARY ===")
    results_df_sorted = results_df.sort_values('accuracy', ascending=False)
    for _, row in results_df_sorted.iterrows():
        print(f"{row['model']:20s}: {row['accuracy']:.4f} (AUC: {row['roc_auc']:.4f})")

    return results_df


# ============ MAIN ============

def main():
    """Run interaction and validation phases."""
    project_root = Path(__file__).parent.parent

    # Load data
    print("Loading data...")
    df = pd.read_csv(project_root / 'data' / 'processed' / 'ablation' / 'corners_features_step9.csv')
    labels_df = pd.read_csv(project_root / 'data' / 'processed' / 'corner_labels.csv')
    df = df.merge(labels_df, on='event_id', how='inner')

    print(f"✓ Loaded {len(df)} corners")

    # Output directory
    output_dir = project_root / 'results' / 'optimal_search'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load optimal feature sets from Phase 5
    optimal_sets_file = output_dir / 'optimal_feature_sets.json'

    if not optimal_sets_file.exists():
        print(f"\nERROR: Run script 12 first to generate optimal feature sets!")
        print(f"Expected file: {optimal_sets_file}")
        return

    with open(optimal_sets_file, 'r') as f:
        optimal_sets = json.load(f)

    # Use bidirectional search result as primary optimal
    optimal_features = optimal_sets['bidirectional_search']
    print(f"\n✓ Loaded optimal feature set: {len(optimal_features)} features")

    # Phase 6: Interaction pair testing
    interaction_results = phase6_interaction_pairs(df, optimal_features, output_dir)

    # Phase 7: Harmful feature inclusion test
    harmful_results = phase7_harmful_feature_test(df, optimal_features, HARMFUL_FEATURES, output_dir)

    # Phase 8: Cross-validation stability
    feature_sets_to_test = {
        'bidirectional_optimal': optimal_sets['bidirectional_search'],
        'forward_selection': optimal_sets['forward_selection'],
        'backward_elimination': optimal_sets['backward_elimination']
    }
    cv_results = phase8_cross_validation(df, feature_sets_to_test, output_dir, n_folds=5)

    # Phase 9: Model comparison
    model_results = phase9_model_comparison(df, optimal_features, output_dir)

    print(f"\n{'='*70}")
    print("INTERACTION & VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == '__main__':
    main()
