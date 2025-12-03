#!/usr/bin/env python3
"""
Feature Ablation Study and Permutation Test for Corner Kick Prediction.

Task 1: Feature Ablation Study
- Configuration A (Minimal): 4 statistically significant features
- Configuration B (Reduced): 18 features (removing redundant pairs)
- Configuration C (Full): 22 aggregate features

Task 2: Permutation Test
- Train XGBoost on shuffled labels (N=100 iterations)
- Compare to real-label performance
- Establishes if there's any learnable signal

Author: CornerTactics Team
Date: 2025-12-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# FEATURE CONFIGURATIONS
# =============================================================================

# Configuration A: Minimal (4 features)
# Only statistically significant predictors (p < 0.05 or borderline)
MINIMAL_FEATURES = [
    'corner_y',
    'defending_to_goal_dist',
    'defending_near_goal',
    'defending_depth'
]

# Configuration B: Reduced (18 features)
# Remove redundant pairs (|r| > 0.9):
# - attacking_density (redundant with attacking_in_box)
# - defending_density (redundant with defending_in_box)
# - corner_side (redundant with corner_y)
# - attacker_defender_ratio (derived from counts already included)
REDUNDANT_FEATURES = [
    'attacking_density',
    'defending_density',
    'corner_side',
    'attacker_defender_ratio'
]

# Configuration C: Full (22 features)
ALL_FEATURES = [
    # Event Metadata (7)
    'minute', 'second', 'period',
    'corner_x', 'corner_y',
    'attacking_team_goals', 'defending_team_goals',
    # Freeze Frame (15)
    'total_attacking', 'total_defending',
    'attacking_in_box', 'defending_in_box',
    'attacking_near_goal', 'defending_near_goal',
    'attacking_density', 'defending_density',
    'numerical_advantage', 'attacker_defender_ratio',
    'defending_depth',
    'attacking_to_goal_dist', 'defending_to_goal_dist',
    'keeper_distance_to_goal',
    'corner_side'
]

# Derive reduced features (18 = 22 - 4 redundant)
REDUCED_FEATURES = [f for f in ALL_FEATURES if f not in REDUNDANT_FEATURES]


def load_data(base_dir: Path):
    """Load data and pre-computed match-based splits."""
    data_dir = base_dir / 'data' / 'processed'

    # Load features
    df = pd.read_csv(data_dir / 'corners_features_temporal_valid.csv')

    # Load splits
    train_idx = pd.read_csv(data_dir / 'train_indices.csv')['index'].values
    val_idx = pd.read_csv(data_dir / 'val_indices.csv')['index'].values
    test_idx = pd.read_csv(data_dir / 'test_indices.csv')['index'].values

    print(f"Dataset: {len(df)} samples")
    print(f"Train: {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")

    # Verify no overlap
    train_matches = set(df.iloc[train_idx]['match_id'].unique())
    val_matches = set(df.iloc[val_idx]['match_id'].unique())
    test_matches = set(df.iloc[test_idx]['match_id'].unique())

    assert len(train_matches & val_matches) == 0, "Train-Val overlap!"
    assert len(train_matches & test_matches) == 0, "Train-Test overlap!"
    assert len(val_matches & test_matches) == 0, "Val-Test overlap!"
    print("Match-based splits verified (no overlap)\n")

    return df, train_idx, val_idx, test_idx


def prepare_features(df: pd.DataFrame, feature_cols: list,
                     train_idx, val_idx, test_idx):
    """Prepare feature matrices and targets."""
    X = df[feature_cols].fillna(-1)

    X_train = X.iloc[train_idx].values
    X_val = X.iloc[val_idx].values
    X_test = X.iloc[test_idx].values

    y_train = df.iloc[train_idx]['shot'].values
    y_val = df.iloc[val_idx]['shot'].values
    y_test = df.iloc[test_idx]['shot'].values

    # Scale features using training set statistics only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler
    }


def train_random_forest(data: dict):
    """Train Random Forest with specified hyperparameters."""
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,  # No max depth as specified
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(data['X_train_scaled'], data['y_train'])

    test_prob = rf.predict_proba(data['X_test_scaled'])[:, 1]
    test_pred = rf.predict(data['X_test_scaled'])

    return {
        'test_auc': roc_auc_score(data['y_test'], test_prob),
        'test_f1': f1_score(data['y_test'], test_pred),
        'model': rf
    }


def train_xgboost(data: dict):
    """Train XGBoost with specified hyperparameters."""
    scale_pos_weight = len(data['y_train'][data['y_train']==0]) / max(1, len(data['y_train'][data['y_train']==1]))

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(data['X_train'], data['y_train'])

    test_prob = xgb_model.predict_proba(data['X_test'])[:, 1]
    test_pred = xgb_model.predict(data['X_test'])

    return {
        'test_auc': roc_auc_score(data['y_test'], test_prob),
        'test_f1': f1_score(data['y_test'], test_pred),
        'model': xgb_model
    }


def train_mlp(data: dict):
    """Train MLP with specified hyperparameters."""
    # Architecture: 22→64→32→1 (adjust input based on feature count)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(data['X_train_scaled'], data['y_train'])

    test_prob = mlp.predict_proba(data['X_test_scaled'])[:, 1]
    test_pred = mlp.predict(data['X_test_scaled'])

    return {
        'test_auc': roc_auc_score(data['y_test'], test_prob),
        'test_f1': f1_score(data['y_test'], test_pred),
        'model': mlp
    }


def run_ablation_study(df, train_idx, val_idx, test_idx):
    """
    Task 1: Feature Ablation Study

    Test three feature configurations:
    A. Minimal (4 features)
    B. Reduced (18 features)
    C. Full (22 features)
    """
    print("="*70)
    print("TASK 1: FEATURE ABLATION STUDY")
    print("="*70)

    configurations = {
        'A: Minimal': MINIMAL_FEATURES,
        'B: Reduced': REDUCED_FEATURES,
        'C: Full': ALL_FEATURES
    }

    results = []

    for config_name, features in configurations.items():
        print(f"\n--- {config_name} ({len(features)} features) ---")
        print(f"Features: {features[:5]}{'...' if len(features) > 5 else ''}")

        # Prepare data for this configuration
        data = prepare_features(df, features, train_idx, val_idx, test_idx)

        # Train all three models
        for model_name, train_fn in [
            ('RandomForest', train_random_forest),
            ('XGBoost', train_xgboost),
            ('MLP', train_mlp)
        ]:
            model_results = train_fn(data)

            results.append({
                'config': config_name,
                'n_features': len(features),
                'model': model_name,
                'test_auc': model_results['test_auc'],
                'test_f1': model_results['test_f1']
            })

            print(f"  {model_name}: AUC={model_results['test_auc']:.3f}, F1={model_results['test_f1']:.3f}")

    return results


def run_permutation_test(df, train_idx, val_idx, test_idx, n_permutations=100):
    """
    Task 2: Permutation Test

    Train XGBoost on shuffled labels to establish baseline.
    If real-label AUC falls within shuffled distribution, features contain no signal.
    """
    print("\n" + "="*70)
    print("TASK 2: PERMUTATION TEST")
    print("="*70)

    # Prepare data with full features
    data = prepare_features(df, ALL_FEATURES, train_idx, val_idx, test_idx)

    # Train on real labels first
    print("\nTraining on real labels...")
    real_results = train_xgboost(data)
    real_auc = real_results['test_auc']
    print(f"Real-label Test AUC: {real_auc:.4f}")

    # Permutation test
    print(f"\nRunning {n_permutations} permutations...")
    shuffled_aucs = []

    for i in tqdm(range(n_permutations), desc="Permutations"):
        # Shuffle training labels
        y_train_shuffled = np.random.permutation(data['y_train'])

        # Create modified data dict
        data_shuffled = data.copy()
        data_shuffled['y_train'] = y_train_shuffled

        # Train XGBoost on shuffled data
        scale_pos_weight = len(y_train_shuffled[y_train_shuffled==0]) / max(1, len(y_train_shuffled[y_train_shuffled==1]))

        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42 + i,  # Vary seed for each permutation
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            verbosity=0
        )
        xgb_model.fit(data_shuffled['X_train'], y_train_shuffled)

        # Evaluate on UNSHUFFLED test set
        test_prob = xgb_model.predict_proba(data_shuffled['X_test'])[:, 1]
        shuffled_auc = roc_auc_score(data_shuffled['y_test'], test_prob)
        shuffled_aucs.append(shuffled_auc)

    shuffled_aucs = np.array(shuffled_aucs)
    mean_shuffled = shuffled_aucs.mean()
    std_shuffled = shuffled_aucs.std()

    print(f"\nShuffled-label AUC: {mean_shuffled:.4f} ± {std_shuffled:.4f}")
    print(f"Real-label AUC: {real_auc:.4f}")

    # Statistical interpretation
    z_score = (real_auc - mean_shuffled) / std_shuffled if std_shuffled > 0 else 0
    p_value_approx = np.mean(shuffled_aucs >= real_auc)

    return {
        'real_auc': real_auc,
        'shuffled_mean': mean_shuffled,
        'shuffled_std': std_shuffled,
        'shuffled_aucs': shuffled_aucs.tolist(),
        'z_score': z_score,
        'p_value_approx': p_value_approx
    }


def print_ablation_table(results):
    """Print ablation results in LaTeX-friendly format."""
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS (Table X)")
    print("="*70)

    print(f"\n{'Config':<15} {'#Feat':>6} {'Model':<15} {'Test AUC':>10} {'Test F1':>10}")
    print("-"*60)

    for r in results:
        print(f"{r['config']:<15} {r['n_features']:>6} {r['model']:<15} {r['test_auc']:>10.3f} {r['test_f1']:>10.3f}")

    # Add baseline
    print(f"{'Baseline':<15} {'–':>6} {'Random':>15} {'0.500':>10} {'–':>10}")

    # LaTeX format
    print("\n--- LaTeX Table ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Feature Ablation Study Results}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{llcrr}")
    print(r"\toprule")
    print(r"Config & \#Feat & Model & Test AUC & Test F1 \\")
    print(r"\midrule")

    prev_config = None
    for r in results:
        config_str = r['config'] if r['config'] != prev_config else ""
        n_feat_str = str(r['n_features']) if r['config'] != prev_config else ""
        print(f"{config_str} & {n_feat_str} & {r['model']} & {r['test_auc']:.3f} & {r['test_f1']:.3f} \\\\")
        prev_config = r['config']

    print(r"\midrule")
    print(r"Baseline & -- & Random & 0.500 & -- \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_permutation_table(perm_results):
    """Print permutation test results."""
    print("\n" + "="*70)
    print("PERMUTATION TEST RESULTS")
    print("="*70)

    print(f"\n{'Condition':<30} {'AUC (mean ± std)':>20}")
    print("-"*55)
    print(f"{'Real labels':<30} {perm_results['real_auc']:.3f}")
    print(f"{'Shuffled labels (N=100)':<30} {perm_results['shuffled_mean']:.3f} ± {perm_results['shuffled_std']:.3f}")

    # LaTeX format
    print("\n--- LaTeX Table ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Permutation Test Results}")
    print(r"\label{tab:permutation}")
    print(r"\begin{tabular}{lr}")
    print(r"\toprule")
    print(r"Condition & AUC (mean $\pm$ std) \\")
    print(r"\midrule")
    print(f"Real labels & {perm_results['real_auc']:.3f} \\\\")
    print(f"Shuffled labels (N=100) & {perm_results['shuffled_mean']:.3f} $\\pm$ {perm_results['shuffled_std']:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_interpretation(ablation_results, perm_results):
    """Print interpretation of results."""
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    # Find best ablation result
    best_result = max(ablation_results, key=lambda x: x['test_auc'])

    # Check if any config beats random
    max_auc = best_result['test_auc']

    # Permutation interpretation
    real_auc = perm_results['real_auc']
    shuffled_mean = perm_results['shuffled_mean']
    shuffled_std = perm_results['shuffled_std']

    # Check if real AUC is within 2 std of shuffled mean
    lower_bound = shuffled_mean - 2 * shuffled_std
    upper_bound = shuffled_mean + 2 * shuffled_std
    within_noise = lower_bound <= real_auc <= upper_bound

    print(f"\n1. ABLATION STUDY:")
    print(f"   Best configuration: {best_result['config']} + {best_result['model']}")
    print(f"   Best Test AUC: {best_result['test_auc']:.3f}")

    if max_auc < 0.52:
        print(f"   → All configurations achieve AUC ≈ 0.50 (random chance)")
        print(f"   → Feature selection does not improve predictive performance")

    print(f"\n2. PERMUTATION TEST:")
    print(f"   Real-label AUC: {real_auc:.3f}")
    print(f"   Shuffled 95% CI: [{lower_bound:.3f}, {upper_bound:.3f}]")

    if within_noise:
        print(f"   → Real-label AUC falls WITHIN shuffled distribution")
        print(f"   → Features contain NO predictive signal distinguishable from noise")
        print(f"   → The prediction task may be fundamentally unpredictable from static positioning")
    else:
        print(f"   → Real-label AUC is OUTSIDE shuffled distribution")
        print(f"   → Some signal exists in features (p ≈ {perm_results['p_value_approx']:.3f})")


def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results' / 'ablation_permutation'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, train_idx, val_idx, test_idx = load_data(base_dir)

    # Verify feature columns exist
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"WARNING: Missing features: {missing}")
        return

    # Task 1: Ablation Study
    ablation_results = run_ablation_study(df, train_idx, val_idx, test_idx)

    # Task 2: Permutation Test
    perm_results = run_permutation_test(df, train_idx, val_idx, test_idx, n_permutations=100)

    # Print formatted results
    print_ablation_table(ablation_results)
    print_permutation_table(perm_results)
    print_interpretation(ablation_results, perm_results)

    # Save results
    all_results = {
        'ablation_study': ablation_results,
        'permutation_test': {
            'real_auc': perm_results['real_auc'],
            'shuffled_mean': perm_results['shuffled_mean'],
            'shuffled_std': perm_results['shuffled_std'],
            'z_score': perm_results['z_score'],
            'p_value_approx': perm_results['p_value_approx'],
            'shuffled_aucs': perm_results['shuffled_aucs']
        },
        'feature_configurations': {
            'minimal': MINIMAL_FEATURES,
            'reduced': REDUCED_FEATURES,
            'full': ALL_FEATURES,
            'redundant_removed': REDUNDANT_FEATURES
        }
    }

    results_path = results_dir / 'ablation_permutation_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
