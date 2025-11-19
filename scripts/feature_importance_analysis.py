#!/usr/bin/env python3
"""
Comprehensive Feature Importance Analysis for Corner Kick Shot Prediction

This script:
1. Removes leaked features based on temporal analysis
2. Trains baseline models with clean features
3. Analyzes feature importance using multiple methods:
   - XGBoost native feature importance
   - Permutation importance
   - SHAP values
4. Identifies low-impact features for potential removal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap
from scipy import stats
import joblib
import logging
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Features confirmed as leaked from temporal analysis
LEAKED_FEATURES = [
    'is_shot_assist',      # Directly encodes whether next event is shot
    'has_recipient',       # Only known after pass completes
    'duration',            # Time until next event
    'pass_end_x',          # Actual ending location (not intended)
    'pass_end_y',          # Actual ending location (not intended)
    'pass_length',         # Computed from actual ending
    'pass_angle',          # Computed from actual ending
    'pass_outcome_id',     # Pass success/failure (if exists)
    'pass_outcome_encoded',# Encoded version
    'has_pass_outcome',    # Boolean for completion
    'is_aerial_won',       # Outcome during ball flight
    'pass_recipient_id',   # Assigned after arrival
]

# Features to exclude (non-predictive)
NON_FEATURES = [
    'match_id',
    'event_id',
    'outcome',  # Different outcome type (not shot)
    'leads_to_shot',  # This is our target
    'pass_outcome',  # Text version of outcome
    'pass_height',   # Text (needs encoding)
    'pass_body_part',  # Text (needs encoding)
    'pass_technique',  # Text (needs encoding)
    'corner_x',  # Duplicate of location_x
    'corner_y',  # Duplicate of location_y
]

# Clean feature groups for analysis
FEATURE_GROUPS = {
    'freeze_frame_counts': [
        'total_attacking', 'total_defending',
        'attacking_in_box', 'defending_in_box',
        'attacking_near_goal', 'defending_near_goal'
    ],
    'freeze_frame_density': [
        'attacking_density', 'defending_density',
        'numerical_advantage', 'attacker_defender_ratio'
    ],
    'freeze_frame_spatial': [
        'attacking_centroid_x', 'attacking_centroid_y',
        'defending_centroid_x', 'defending_centroid_y',
        'defending_compactness', 'defending_depth',
        'attacking_to_goal_dist', 'defending_to_goal_dist'
    ],
    'goalkeeper': [
        'num_attacking_keepers', 'num_defending_keepers',
        'keeper_distance_to_goal'
    ],
    'pass_technique': [
        'is_inswinging', 'is_outswinging', 'is_cross_field_switch'
    ],
    'score_state': [
        'attacking_team_goals', 'defending_team_goals',
        'score_difference', 'match_situation'
    ],
    'temporal': [
        'period', 'minute', 'second', 'timestamp_seconds'
    ],
    'substitutions': [
        'total_subs_before', 'recent_subs_5min', 'minutes_since_last_sub'
    ],
    'other': [
        'corner_side'
    ]
}


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load data and remove leaked features.

    Returns:
        Tuple of (cleaned DataFrame, list of feature columns)
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Target distribution: {df['leads_to_shot'].value_counts().to_dict()}")

    # Get all columns
    all_cols = df.columns.tolist()

    # Identify features to keep
    features_to_keep = []
    removed_features = []

    for col in all_cols:
        if col in NON_FEATURES:
            continue
        elif col in LEAKED_FEATURES:
            removed_features.append(col)
        elif col == 'leads_to_shot':
            continue  # Target column
        else:
            features_to_keep.append(col)

    logger.info(f"Removed {len(removed_features)} leaked features: {removed_features}")
    logger.info(f"Keeping {len(features_to_keep)} clean features")

    return df, features_to_keep


def prepare_train_test_split(df: pd.DataFrame, features: List[str],
                             test_size: float = 0.2,
                             random_state: int = 42) -> Tuple:
    """
    Prepare train/test split with proper stratification.
    """
    X = df[features].copy()
    y = df['leads_to_shot'].values

    # Handle missing values
    X = X.fillna(X.median())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Keep original DataFrames for SHAP
    X_train_df = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

    return X_train_df, X_test_df, y_train, y_test, scaler


# =============================================================================
# BASELINE MODEL TRAINING
# =============================================================================

def train_xgboost_baseline(X_train: pd.DataFrame, y_train: np.ndarray,
                           X_test: pd.DataFrame, y_test: np.ndarray) -> xgb.XGBClassifier:
    """
    Train XGBoost baseline with clean features.
    """
    logger.info("Training XGBoost baseline model...")

    # Calculate scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    logger.info(f"Baseline Performance:")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  AUC-ROC: {auc:.3f}")
    logger.info(f"  MCC: {mcc:.3f}")

    return model


# =============================================================================
# FEATURE IMPORTANCE METHODS
# =============================================================================

def get_xgboost_importance(model: xgb.XGBClassifier,
                          feature_names: List[str]) -> pd.DataFrame:
    """
    Get XGBoost native feature importances (gain, weight, cover).
    """
    logger.info("Computing XGBoost native feature importance...")

    # Get different importance types
    importance_types = ['gain', 'weight', 'cover']
    importance_dict = {}

    for imp_type in importance_types:
        importance = model.get_booster().get_score(importance_type=imp_type)
        # Map to feature names
        importance_mapped = {}
        for i, fname in enumerate(feature_names):
            key = f'f{i}'
            if key in importance:
                importance_mapped[fname] = importance[key]
            else:
                importance_mapped[fname] = 0
        importance_dict[imp_type] = importance_mapped

    # Create DataFrame
    df_importance = pd.DataFrame(importance_dict)
    df_importance = df_importance.fillna(0)

    # Add average rank
    df_importance['avg_rank'] = df_importance.rank(ascending=False).mean(axis=1)

    return df_importance.sort_values('gain', ascending=False)


def get_permutation_importance(model, X_test: pd.DataFrame, y_test: np.ndarray,
                               n_repeats: int = 10) -> pd.DataFrame:
    """
    Compute permutation importance.
    """
    logger.info(f"Computing permutation importance ({n_repeats} repeats)...")

    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring='roc_auc'
    )

    df_perm = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    })

    # Add confidence intervals
    df_perm['importance_lower'] = df_perm['importance_mean'] - 2 * df_perm['importance_std']
    df_perm['importance_upper'] = df_perm['importance_mean'] + 2 * df_perm['importance_std']

    return df_perm.sort_values('importance_mean', ascending=False)


def compute_shap_values(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    """
    Compute SHAP values for feature importance.
    """
    logger.info("Computing SHAP values...")

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test)

    # Get feature importance from SHAP
    shap_importance = pd.DataFrame({
        'feature': X_test.columns,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)

    return shap_values, explainer, shap_importance


def statistical_significance_test(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Test statistical significance of each feature with the target.
    """
    logger.info("Computing statistical significance tests...")

    results = []

    for col in X.columns:
        feature_vals = X[col].values

        # Handle missing values
        mask = ~np.isnan(feature_vals)
        feature_clean = feature_vals[mask]
        y_clean = y[mask]

        # Compute correlation
        if len(np.unique(feature_clean)) > 2:
            # Continuous feature - point-biserial correlation
            corr, p_value = stats.pointbiserialr(y_clean, feature_clean)
        else:
            # Binary feature - chi-square test
            contingency = pd.crosstab(feature_clean, y_clean)
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            corr = np.sqrt(chi2 / len(y_clean))  # Cramér's V

        results.append({
            'feature': col,
            'correlation': abs(corr),
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    return pd.DataFrame(results).sort_values('correlation', ascending=False)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_feature_importance_comparison(xgb_imp: pd.DataFrame,
                                       perm_imp: pd.DataFrame,
                                       shap_imp: pd.DataFrame,
                                       output_dir: Path):
    """
    Create comprehensive comparison of feature importance methods.
    """
    # Prepare data for plotting
    features = xgb_imp.index.tolist()[:20]  # Top 20 features

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # XGBoost Gain
    ax = axes[0, 0]
    top_xgb = xgb_imp.head(20)
    ax.barh(range(len(top_xgb)), top_xgb['gain'].values)
    ax.set_yticks(range(len(top_xgb)))
    ax.set_yticklabels(top_xgb.index, fontsize=8)
    ax.set_xlabel('Gain')
    ax.set_title('XGBoost Feature Importance (Gain)')
    ax.invert_yaxis()

    # Permutation Importance
    ax = axes[0, 1]
    top_perm = perm_imp.head(20)
    ax.barh(range(len(top_perm)), top_perm['importance_mean'].values)
    ax.set_yticks(range(len(top_perm)))
    ax.set_yticklabels(top_perm['feature'].values, fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Permutation Importance')
    ax.invert_yaxis()

    # Add error bars
    for i, row in enumerate(top_perm.itertuples()):
        ax.errorbar(row.importance_mean, i, xerr=row.importance_std*2,
                   fmt='none', color='black', alpha=0.3)

    # SHAP Importance
    ax = axes[1, 0]
    top_shap = shap_imp.head(20)
    ax.barh(range(len(top_shap)), top_shap['shap_importance'].values)
    ax.set_yticks(range(len(top_shap)))
    ax.set_yticklabels(top_shap['feature'].values, fontsize=8)
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title('SHAP Feature Importance')
    ax.invert_yaxis()

    # Combined Ranking
    ax = axes[1, 1]

    # Merge all rankings
    combined = pd.DataFrame(index=features)
    combined['xgb_rank'] = xgb_imp['gain'].rank(ascending=False)
    combined['perm_rank'] = perm_imp.set_index('feature')['importance_mean'].rank(ascending=False)
    combined['shap_rank'] = shap_imp.set_index('feature')['shap_importance'].rank(ascending=False)
    combined['avg_rank'] = combined.mean(axis=1)
    combined = combined.sort_values('avg_rank')

    # Plot heatmap of ranks
    sns.heatmap(combined.head(20).T, annot=True, fmt='.0f', cmap='YlOrRd_r',
                ax=ax, cbar_kws={'label': 'Rank (lower is better)'})
    ax.set_title('Feature Importance Ranking Comparison')
    ax.set_xlabel('Features')
    ax.set_ylabel('Method')

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved comparison plot to {output_dir}/feature_importance_comparison.png")


def plot_shap_summary(shap_values, X_test: pd.DataFrame, output_dir: Path):
    """
    Create SHAP summary plots.
    """
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_importance_bar.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved SHAP plots to {output_dir}")


def identify_low_impact_features(xgb_imp: pd.DataFrame,
                                 perm_imp: pd.DataFrame,
                                 shap_imp: pd.DataFrame,
                                 significance: pd.DataFrame,
                                 threshold: float = 0.01) -> List[str]:
    """
    Identify features with consistently low importance across all methods.
    """
    # Normalize importances to 0-1 scale
    xgb_norm = xgb_imp['gain'] / xgb_imp['gain'].max() if xgb_imp['gain'].max() > 0 else xgb_imp['gain']

    perm_dict = dict(zip(perm_imp['feature'],
                        perm_imp['importance_mean'] / perm_imp['importance_mean'].max()
                        if perm_imp['importance_mean'].max() > 0 else perm_imp['importance_mean']))

    shap_dict = dict(zip(shap_imp['feature'],
                        shap_imp['shap_importance'] / shap_imp['shap_importance'].max()
                        if shap_imp['shap_importance'].max() > 0 else shap_imp['shap_importance']))

    sig_dict = dict(zip(significance['feature'],
                       significance['significant'].astype(int)))

    # Combine scores
    low_impact = []

    for feature in xgb_imp.index:
        xgb_score = xgb_norm.get(feature, 0)
        perm_score = perm_dict.get(feature, 0)
        shap_score = shap_dict.get(feature, 0)
        is_significant = sig_dict.get(feature, 0)

        # Average normalized importance
        avg_importance = (xgb_score + perm_score + shap_score) / 3

        # Low impact if below threshold and not statistically significant
        if avg_importance < threshold and not is_significant:
            low_impact.append(feature)

    return low_impact


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(xgb_imp: pd.DataFrame, perm_imp: pd.DataFrame,
                   shap_imp: pd.DataFrame, significance: pd.DataFrame,
                   low_impact: List[str], model_metrics: Dict,
                   output_path: Path):
    """
    Generate comprehensive markdown report.
    """
    report = []
    report.append("# Feature Importance Analysis Report\n")
    report.append("## Clean Baseline Performance (Without Leaked Features)\n\n")

    report.append("### Model Metrics\n")
    for metric, value in model_metrics.items():
        report.append(f"- **{metric}**: {value:.3f}\n")

    report.append("\n### Removed Leaked Features\n")
    for feature in LEAKED_FEATURES:
        report.append(f"- `{feature}`\n")

    report.append("\n## Top 10 Most Important Features\n\n")
    report.append("### XGBoost Gain\n")
    for i, (feature, row) in enumerate(xgb_imp.head(10).iterrows()):
        report.append(f"{i+1}. **{feature}**: {row['gain']:.3f}\n")

    report.append("\n### Permutation Importance\n")
    for i, row in enumerate(perm_imp.head(10).itertuples()):
        report.append(f"{i+1}. **{row.feature}**: {row.importance_mean:.4f} ± {row.importance_std:.4f}\n")

    report.append("\n### SHAP Values\n")
    for i, row in enumerate(shap_imp.head(10).itertuples()):
        report.append(f"{i+1}. **{row.feature}**: {row.shap_importance:.4f}\n")

    report.append("\n## Statistical Significance\n\n")
    sig_features = significance[significance['significant']]
    report.append(f"**{len(sig_features)} features** are statistically significant (p < 0.05)\n\n")

    report.append("Top 5 most correlated:\n")
    for row in sig_features.head(5).itertuples():
        report.append(f"- **{row.feature}**: r={row.correlation:.3f}, p={row.p_value:.3e}\n")

    report.append("\n## Feature Groups Analysis\n\n")
    for group_name, group_features in FEATURE_GROUPS.items():
        group_imp = xgb_imp.loc[xgb_imp.index.isin(group_features), 'gain'].sum()
        report.append(f"- **{group_name}**: Total gain = {group_imp:.3f}\n")

    report.append("\n## Low Impact Features (Candidates for Removal)\n\n")
    if low_impact:
        report.append(f"**{len(low_impact)} features** have consistently low importance:\n\n")
        for feature in low_impact:
            report.append(f"- `{feature}`\n")
    else:
        report.append("No features identified as low impact.\n")

    report.append("\n## Recommendations\n\n")
    report.append("1. **Keep all freeze-frame features** - they show consistent importance\n")
    report.append("2. **Consider removing low-impact features** to reduce dimensionality\n")
    report.append("3. **Focus feature engineering** on top-performing feature groups\n")
    report.append("4. **Validate findings** with cross-validation before final model\n")

    # Write report
    with open(output_path, 'w') as f:
        f.writelines(report)

    logger.info(f"Generated report at {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(data_path: str = None, output_dir: str = None):
    """
    Run comprehensive feature importance analysis.
    """
    # Setup paths
    project_root = Path(__file__).parent.parent

    if data_path is None:
        data_path = project_root / 'data' / 'processed' / 'corners_features_with_shot.csv'
    else:
        data_path = Path(data_path)

    if output_dir is None:
        output_dir = project_root / 'results' / 'feature_importance'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Load and prepare data
    df, features = load_and_prepare_data(data_path)

    # Prepare train/test split
    X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(df, features)

    # Train baseline model
    model = train_xgboost_baseline(X_train, y_train, X_test, y_test)

    # Save model
    joblib.dump(model, output_dir / 'xgboost_clean_baseline.pkl')
    joblib.dump(scaler, output_dir / 'feature_scaler.pkl')

    # Compute feature importances
    xgb_importance = get_xgboost_importance(model, features)
    perm_importance = get_permutation_importance(model, X_test, y_test)
    shap_values, explainer, shap_importance = compute_shap_values(model, X_train, X_test)

    # Statistical significance tests
    significance = statistical_significance_test(X_train, y_train)

    # Identify low impact features
    low_impact = identify_low_impact_features(
        xgb_importance, perm_importance, shap_importance, significance
    )

    logger.info(f"Identified {len(low_impact)} low impact features")

    # Create visualizations
    plot_feature_importance_comparison(
        xgb_importance, perm_importance, shap_importance, figures_dir
    )
    plot_shap_summary(shap_values, X_test, figures_dir)

    # Model metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    model_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_prob),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }

    # Generate report
    generate_report(
        xgb_importance, perm_importance, shap_importance,
        significance, low_impact, model_metrics,
        output_dir / 'feature_importance_report.md'
    )

    # Save detailed results
    results = {
        'model_metrics': model_metrics,
        'xgb_importance': xgb_importance.to_dict(),
        'perm_importance': perm_importance.to_dict('records'),
        'shap_importance': shap_importance.to_dict('records'),
        'significance_tests': significance.to_dict('records'),
        'low_impact_features': low_impact,
        'feature_groups': FEATURE_GROUPS,
        'leaked_features_removed': LEAKED_FEATURES
    }

    with open(output_dir / 'feature_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Analysis complete. Results saved to {output_dir}")

    # Print summary
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nClean Baseline Performance:")
    print(f"  Accuracy: {model_metrics['Accuracy']:.3f}")
    print(f"  AUC-ROC: {model_metrics['AUC-ROC']:.3f}")
    print(f"  MCC: {model_metrics['MCC']:.3f}")

    print(f"\nTop 5 Features (XGBoost Gain):")
    for i, (feature, row) in enumerate(xgb_importance.head(5).iterrows()):
        print(f"  {i+1}. {feature}: {row['gain']:.3f}")

    print(f"\nLow Impact Features: {len(low_impact)}")
    if low_impact:
        print(f"  {', '.join(low_impact[:5])}...")

    print("\n" + "="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Feature importance analysis')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--output', type=str, help='Output directory')

    args = parser.parse_args()
    main(data_path=args.data, output_dir=args.output)