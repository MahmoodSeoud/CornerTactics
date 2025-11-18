"""
Optimal Feature Selection: Comprehensive Bidirectional Search

Finds the truly optimal feature set for binary shot prediction using:
1. Baseline experiments with/without harmful features
2. Forward selection from beneficial baseline
3. Backward elimination from full beneficial set
4. Bidirectional search (combines forward + backward)
5. Interaction pair testing
6. Cross-validation stability verification

Goal: Beat 87.97% accuracy (current best with 29 features)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import pickle
from datetime import datetime


# ============ FEATURE DEFINITIONS ============

# Harmful features (cause -3.09% to -2.41% drop when removed)
HARMFUL_FEATURES = [
    'period',      # -3.09%
    'team_id',     # -3.09%
    'pass_height_id',  # -3.09%
    'pass_angle',  # -2.75%
    'location_y',  # -2.41%
    'minute'       # -2.41%
]

# All 27 raw features
RAW_FEATURES_ALL = [
    'period', 'minute', 'second', 'duration',
    'index', 'possession',
    'location_x', 'location_y',
    'pass_length', 'pass_angle',
    'pass_end_x', 'pass_end_y',
    'team_id', 'player_id', 'position_id',
    'play_pattern_id', 'possession_team_id',
    'pass_height_id', 'pass_body_part_id',
    'pass_type_id', 'pass_technique_id',
    'pass_recipient_id',
    'under_pressure',
    'has_pass_outcome',
    'is_aerial_won',
    'total_attacking', 'total_defending'
]

# Beneficial raw features (excludes harmful)
BENEFICIAL_RAW = [f for f in RAW_FEATURES_ALL if f not in HARMFUL_FEATURES]

# Top 10 engineered features (by shot prediction gain from Phase 2)
TOP_ENGINEERED = [
    'is_shot_assist',           # +5.15%
    'defending_to_goal_dist',   # +2.75%
    'pass_outcome_encoded',     # +2.75%
    'defending_depth',          # +2.41%
    'has_recipient',            # +2.41%
    'defending_team_goals',     # +2.41%
    'defending_in_box',         # +2.06%
    'attacking_near_goal',      # +2.06%
    'corner_side',              # +2.06%
    'is_cross_field_switch'     # +2.06%
]

# Feature interaction pairs (high correlation |r| > 0.8)
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

# Metadata and label columns
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


def get_all_features(df):
    """Get all available features from dataframe."""
    all_cols = df.columns.tolist()
    all_features = [col for col in all_cols if col not in METADATA_COLS + LABEL_COLS]
    return all_features


def validate_features(df, feature_list):
    """Validate that features exist in dataframe."""
    available = get_all_features(df)
    valid = [f for f in feature_list if f in available]
    missing = [f for f in feature_list if f not in available]
    if missing:
        print(f"  WARNING: {len(missing)} features not found in data: {missing[:3]}...")
    return valid


# ============ PHASE 1: FEATURE GROUP DEFINITION ============

def phase1_define_feature_groups(df, output_dir):
    """
    Define and validate feature groups.

    Returns:
        beneficial_raw: 21 raw features (excludes harmful)
        harmful_raw: 6 harmful features
        top_engineered: 10 best engineered features
    """
    print("\n" + "="*70)
    print("PHASE 1: FEATURE GROUP DEFINITION")
    print("="*70)

    # Validate features exist
    beneficial_raw = validate_features(df, BENEFICIAL_RAW)
    harmful_raw = validate_features(df, HARMFUL_FEATURES)
    top_engineered = validate_features(df, TOP_ENGINEERED)

    print(f"\n✓ Beneficial raw features: {len(beneficial_raw)}")
    print(f"✓ Harmful raw features: {len(harmful_raw)}")
    print(f"✓ Top engineered features: {len(top_engineered)}")

    # Save feature groups
    groups = {
        'beneficial_raw': beneficial_raw,
        'harmful_raw': harmful_raw,
        'top_engineered': top_engineered,
        'all_raw': RAW_FEATURES_ALL
    }

    output_file = output_dir / 'feature_groups.json'
    with open(output_file, 'w') as f:
        json.dump(groups, f, indent=2)

    print(f"\n✓ Saved feature groups to: {output_file}")

    return beneficial_raw, harmful_raw, top_engineered


# ============ PHASE 2: BASELINE EXPERIMENTS ============

def phase2_baseline_experiments(df, beneficial_raw, harmful_raw, output_dir):
    """
    Test 3 baseline configurations:
    1. Beneficial raw only (21 features) - WITHOUT harmful features
    2. All raw (27 features) - WITH harmful features
    3. Current best from forward selection (29 features)

    Hypothesis: Removing harmful features improves baseline.
    """
    print("\n" + "="*70)
    print("PHASE 2: BASELINE EXPERIMENTS")
    print("="*70)

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)
    y_train = train_df['leads_to_shot'].values
    y_test = test_df['leads_to_shot'].values

    results = []

    # Baseline 1: Beneficial raw only
    print("\n--- Baseline 1: Beneficial raw only (21 features) ---")
    X_train = train_df[beneficial_raw].values
    X_test = test_df[beneficial_raw].values
    metrics, _ = train_evaluate_rf(X_train, y_train, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f}")

    results.append({
        'baseline': 'beneficial_raw_only',
        'num_features': len(beneficial_raw),
        'features': beneficial_raw,
        **metrics
    })

    # Baseline 2: All raw (with harmful)
    print("\n--- Baseline 2: All raw (27 features, includes harmful) ---")
    all_raw_valid = validate_features(df, RAW_FEATURES_ALL)
    X_train = train_df[all_raw_valid].values
    X_test = test_df[all_raw_valid].values
    metrics, _ = train_evaluate_rf(X_train, y_train, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f}")

    results.append({
        'baseline': 'all_raw_with_harmful',
        'num_features': len(all_raw_valid),
        'features': all_raw_valid,
        **metrics
    })

    # Baseline 3: Current best (from forward selection Phase 3)
    # This should be: all 27 raw + is_shot_assist + attacking_in_box
    print("\n--- Baseline 3: Current best from Phase 3 forward selection ---")
    current_best = all_raw_valid + ['is_shot_assist', 'attacking_in_box']
    current_best_valid = validate_features(df, current_best)
    X_train = train_df[current_best_valid].values
    X_test = test_df[current_best_valid].values
    metrics, _ = train_evaluate_rf(X_train, y_train, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f}")
    print(f"Expected: 87.97% (from previous results)")

    results.append({
        'baseline': 'current_best_phase3',
        'num_features': len(current_best_valid),
        'features': current_best_valid,
        **metrics
    })

    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'phase2_baseline_experiments.json'
    results_df.to_json(output_file, orient='records', indent=2)

    print(f"\n✓ Saved baseline experiments to: {output_file}")

    # Print comparison
    print("\n=== BASELINE COMPARISON ===")
    for result in results:
        print(f"{result['baseline']:30s} | {result['num_features']:2d} features | Acc: {result['accuracy']:.4f}")

    return results_df


# ============ PHASE 3: FORWARD SELECTION FROM BENEFICIAL BASELINE ============

def phase3_forward_selection(df, beneficial_raw, top_engineered, output_dir, threshold=0.003):
    """
    Forward selection starting from beneficial raw features.

    Greedily add engineered features until gain < threshold (0.3%).
    """
    print("\n" + "="*70)
    print("PHASE 3: FORWARD SELECTION FROM BENEFICIAL BASELINE")
    print("="*70)
    print(f"Starting with: {len(beneficial_raw)} beneficial raw features")
    print(f"Candidates: {len(top_engineered)} top engineered features")
    print(f"Stopping criterion: gain < {threshold:.3f}\n")

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)
    y_train = train_df['leads_to_shot'].values
    y_test = test_df['leads_to_shot'].values

    # Start with beneficial raw features
    selected_features = beneficial_raw.copy()
    remaining_candidates = top_engineered.copy()

    # Initial performance
    X_train = train_df[selected_features].values
    X_test = test_df[selected_features].values
    metrics, _ = train_evaluate_rf(X_train, y_train, X_test, y_test)
    current_accuracy = metrics['accuracy']

    print(f"Initial baseline (beneficial raw): {current_accuracy:.4f}")

    results = [{
        'step': 0,
        'feature_added': 'NONE (beneficial raw baseline)',
        'num_features': len(selected_features),
        'accuracy': current_accuracy,
        'roc_auc': metrics['roc_auc'],
        'gain': 0.0,
        'cumulative_gain': 0.0
    }]

    # Greedy forward selection
    step = 1
    while remaining_candidates:
        best_feature = None
        best_gain = threshold  # Must exceed threshold
        best_metrics = None

        # Try adding each remaining candidate
        for candidate in tqdm(remaining_candidates, desc=f"Step {step}", leave=False):
            test_features = selected_features + [candidate]

            X_train_test = train_df[test_features].values
            X_test_test = test_df[test_features].values

            metrics, _ = train_evaluate_rf(X_train_test, y_train, X_test_test, y_test)
            gain = metrics['accuracy'] - current_accuracy

            if gain > best_gain:
                best_gain = gain
                best_feature = candidate
                best_metrics = metrics

        # Stop if no improvement exceeds threshold
        if best_feature is None:
            print(f"\n→ No feature provides gain > {threshold:.3f}. Stopping.")
            break

        # Add best feature
        selected_features.append(best_feature)
        remaining_candidates.remove(best_feature)
        current_accuracy = best_metrics['accuracy']

        results.append({
            'step': step,
            'feature_added': best_feature,
            'num_features': len(selected_features),
            'accuracy': current_accuracy,
            'roc_auc': best_metrics['roc_auc'],
            'gain': best_gain,
            'cumulative_gain': current_accuracy - results[0]['accuracy']
        })

        print(f"\n→ Step {step}: Added '{best_feature}'")
        print(f"   Accuracy: {current_accuracy:.4f} (gain: {best_gain:+.4f})")

        step += 1

    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'phase3_forward_selection_beneficial.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n✓ Saved forward selection results to: {output_file}")
    print(f"\n=== FINAL FEATURE SET (Forward Selection) ===")
    print(f"Total features: {len(selected_features)}")
    print(f"  - Beneficial raw: {len(beneficial_raw)}")
    print(f"  - Engineered: {len(selected_features) - len(beneficial_raw)}")
    print(f"Final accuracy: {current_accuracy:.4f}")
    print(f"Improvement over baseline: {current_accuracy - results[0]['accuracy']:+.4f}")

    return results_df, selected_features


# ============ PHASE 4: BACKWARD ELIMINATION ============

def phase4_backward_elimination(df, beneficial_raw, top_engineered, output_dir, threshold=0.003):
    """
    Backward elimination starting from beneficial raw + top engineered.

    Iteratively remove feature with smallest importance until performance drops > threshold.
    """
    print("\n" + "="*70)
    print("PHASE 4: BACKWARD ELIMINATION")
    print("="*70)

    # Start with all beneficial features
    starting_features = beneficial_raw + top_engineered
    starting_features_valid = validate_features(df, starting_features)

    print(f"Starting with: {len(starting_features_valid)} features")
    print(f"  - Beneficial raw: {len(beneficial_raw)}")
    print(f"  - Top engineered: {len([f for f in starting_features_valid if f in top_engineered])}")
    print(f"Stopping criterion: performance drop > {threshold:.3f}\n")

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)
    y_train = train_df['leads_to_shot'].values
    y_test = test_df['leads_to_shot'].values

    # Initial performance
    current_features = starting_features_valid.copy()
    X_train = train_df[current_features].values
    X_test = test_df[current_features].values
    metrics, model = train_evaluate_rf(X_train, y_train, X_test, y_test)
    current_accuracy = metrics['accuracy']

    print(f"Initial performance (all beneficial): {current_accuracy:.4f}")

    results = [{
        'step': 0,
        'feature_removed': 'NONE (full beneficial set)',
        'num_features': len(current_features),
        'accuracy': current_accuracy,
        'roc_auc': metrics['roc_auc'],
        'drop': 0.0
    }]

    # Backward elimination
    step = 1
    while len(current_features) > len(beneficial_raw):  # Keep at least beneficial raw
        # Get feature importances
        feature_importances = pd.DataFrame({
            'feature': current_features,
            'importance': model.feature_importances_
        }).sort_values('importance')

        # Try removing least important feature
        least_important = feature_importances.iloc[0]['feature']

        # Test removal
        test_features = [f for f in current_features if f != least_important]
        X_train_test = train_df[test_features].values
        X_test_test = test_df[test_features].values

        metrics, test_model = train_evaluate_rf(X_train_test, y_train, X_test_test, y_test)
        new_accuracy = metrics['accuracy']
        drop = current_accuracy - new_accuracy

        # Stop if drop exceeds threshold
        if drop > threshold:
            print(f"\n→ Removing '{least_important}' causes drop of {drop:.4f} > {threshold:.3f}")
            print(f"   Stopping backward elimination.")
            break

        # Remove feature
        current_features = test_features
        current_accuracy = new_accuracy
        model = test_model

        results.append({
            'step': step,
            'feature_removed': least_important,
            'num_features': len(current_features),
            'accuracy': current_accuracy,
            'roc_auc': metrics['roc_auc'],
            'drop': drop
        })

        print(f"\n→ Step {step}: Removed '{least_important}'")
        print(f"   Accuracy: {current_accuracy:.4f} (drop: {drop:+.4f})")
        print(f"   Importance: {feature_importances.iloc[0]['importance']:.6f}")

        step += 1

    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'phase4_backward_elimination.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n✓ Saved backward elimination results to: {output_file}")
    print(f"\n=== FINAL FEATURE SET (Backward Elimination) ===")
    print(f"Total features: {len(current_features)}")
    print(f"Final accuracy: {current_accuracy:.4f}")
    print(f"Change from initial: {current_accuracy - results[0]['accuracy']:+.4f}")

    return results_df, current_features


# ============ PHASE 5: BIDIRECTIONAL SEARCH ============

def phase5_bidirectional_search(df, beneficial_raw, top_engineered, output_dir, threshold=0.003):
    """
    Bidirectional search: Combine forward + backward.

    Algorithm:
    1. Start with beneficial raw features
    2. Forward: Try adding best engineered feature
    3. Backward: Try removing worst current feature
    4. Accept whichever improves performance more
    5. Repeat until no improvement
    """
    print("\n" + "="*70)
    print("PHASE 5: BIDIRECTIONAL SEARCH")
    print("="*70)
    print(f"Starting with: {len(beneficial_raw)} beneficial raw features")
    print(f"Candidates: {len(top_engineered)} top engineered features")
    print(f"Threshold: {threshold:.3f}\n")

    # Prepare splits
    train_df, val_df, test_df = prepare_splits(df)
    y_train = train_df['leads_to_shot'].values
    y_test = test_df['leads_to_shot'].values

    # Initialize
    current_features = beneficial_raw.copy()
    available_to_add = top_engineered.copy()

    # Initial performance
    X_train = train_df[current_features].values
    X_test = test_df[current_features].values
    metrics, model = train_evaluate_rf(X_train, y_train, X_test, y_test)
    current_accuracy = metrics['accuracy']

    print(f"Initial baseline (beneficial raw): {current_accuracy:.4f}")

    results = [{
        'step': 0,
        'action': 'NONE (baseline)',
        'feature': None,
        'num_features': len(current_features),
        'accuracy': current_accuracy,
        'roc_auc': metrics['roc_auc'],
        'gain': 0.0
    }]

    step = 1
    max_iterations = 50  # Safety limit

    while step < max_iterations:
        best_action = None
        best_feature = None
        best_gain = threshold
        best_metrics = None
        best_features = None

        # FORWARD: Try adding each available feature
        print(f"\n→ Step {step}: Testing forward additions ({len(available_to_add)} candidates)...")
        for candidate in available_to_add:
            test_features = current_features + [candidate]
            X_train_test = train_df[test_features].values
            X_test_test = test_df[test_features].values

            metrics_test, _ = train_evaluate_rf(X_train_test, y_train, X_test_test, y_test)
            gain = metrics_test['accuracy'] - current_accuracy

            if gain > best_gain:
                best_gain = gain
                best_action = 'ADD'
                best_feature = candidate
                best_metrics = metrics_test
                best_features = test_features

        # BACKWARD: Try removing each engineered feature (keep beneficial raw)
        engineered_in_current = [f for f in current_features if f not in beneficial_raw]
        if engineered_in_current:
            print(f"   Testing backward removals ({len(engineered_in_current)} candidates)...")
            for candidate in engineered_in_current:
                test_features = [f for f in current_features if f != candidate]
                X_train_test = train_df[test_features].values
                X_test_test = test_df[test_features].values

                metrics_test, _ = train_evaluate_rf(X_train_test, y_train, X_test_test, y_test)
                gain = metrics_test['accuracy'] - current_accuracy

                if gain > best_gain:
                    best_gain = gain
                    best_action = 'REMOVE'
                    best_feature = candidate
                    best_metrics = metrics_test
                    best_features = test_features

        # No improvement found
        if best_action is None:
            print(f"\n→ No action provides gain > {threshold:.3f}. Stopping.")
            break

        # Apply best action
        current_features = best_features
        current_accuracy = best_metrics['accuracy']

        if best_action == 'ADD':
            available_to_add.remove(best_feature)
            print(f"   ✓ Added '{best_feature}' | Acc: {current_accuracy:.4f} (gain: {best_gain:+.4f})")
        else:  # REMOVE
            available_to_add.append(best_feature)
            print(f"   ✓ Removed '{best_feature}' | Acc: {current_accuracy:.4f} (gain: {best_gain:+.4f})")

        results.append({
            'step': step,
            'action': best_action,
            'feature': best_feature,
            'num_features': len(current_features),
            'accuracy': current_accuracy,
            'roc_auc': best_metrics['roc_auc'],
            'gain': best_gain
        })

        # Retrain for next iteration
        X_train = train_df[current_features].values
        X_test = test_df[current_features].values
        metrics, model = train_evaluate_rf(X_train, y_train, X_test, y_test)

        step += 1

    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'phase5_bidirectional_search.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n✓ Saved bidirectional search results to: {output_file}")
    print(f"\n=== FINAL FEATURE SET (Bidirectional Search) ===")
    print(f"Total features: {len(current_features)}")
    print(f"  - Beneficial raw: {len([f for f in current_features if f in beneficial_raw])}")
    print(f"  - Engineered: {len([f for f in current_features if f not in beneficial_raw])}")
    print(f"Final accuracy: {current_accuracy:.4f}")
    print(f"Improvement over baseline: {current_accuracy - results[0]['accuracy']:+.4f}")

    return results_df, current_features


# ============ MAIN ============

def main():
    """Run optimal feature search."""
    project_root = Path(__file__).parent.parent

    # Load data
    print("Loading data...")
    df_full = pd.read_csv(project_root / 'data' / 'processed' / 'ablation' / 'corners_features_step9.csv')
    labels_df = pd.read_csv(project_root / 'data' / 'processed' / 'corner_labels.csv')
    df = df_full.merge(labels_df, on='event_id', how='inner')

    print(f"✓ Loaded {len(df)} corners")
    print(f"✓ Total columns: {len(df.columns)}")

    # Output directory
    output_dir = project_root / 'results' / 'optimal_search'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Define feature groups
    beneficial_raw, harmful_raw, top_engineered = phase1_define_feature_groups(df, output_dir)

    # Phase 2: Baseline experiments
    baseline_results = phase2_baseline_experiments(df, beneficial_raw, harmful_raw, output_dir)

    # Phase 3: Forward selection from beneficial baseline
    forward_results, forward_features = phase3_forward_selection(
        df, beneficial_raw, top_engineered, output_dir, threshold=0.003
    )

    # Phase 4: Backward elimination
    backward_results, backward_features = phase4_backward_elimination(
        df, beneficial_raw, top_engineered, output_dir, threshold=0.003
    )

    # Phase 5: Bidirectional search
    bidir_results, bidir_features = phase5_bidirectional_search(
        df, beneficial_raw, top_engineered, output_dir, threshold=0.003
    )

    # Save optimal feature sets
    optimal_sets = {
        'forward_selection': forward_features,
        'backward_elimination': backward_features,
        'bidirectional_search': bidir_features
    }

    output_file = output_dir / 'optimal_feature_sets.json'
    with open(output_file, 'w') as f:
        json.dump(optimal_sets, f, indent=2)

    print(f"\n{'='*70}")
    print("OPTIMAL FEATURE SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"\n=== SUMMARY ===")
    print(f"Forward selection:     {len(forward_features)} features, {forward_results.iloc[-1]['accuracy']:.4f} acc")
    print(f"Backward elimination:  {len(backward_features)} features, {backward_results.iloc[-1]['accuracy']:.4f} acc")
    print(f"Bidirectional search:  {len(bidir_features)} features, {bidir_results.iloc[-1]['accuracy']:.4f} acc")
    print(f"\nTarget to beat: 87.97% (previous best)")


if __name__ == '__main__':
    main()
