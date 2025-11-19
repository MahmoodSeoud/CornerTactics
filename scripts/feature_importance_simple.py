#!/usr/bin/env python3
"""
Simple Feature Importance Analysis using RandomForest
(No XGBoost/SHAP dependencies)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy import stats
import joblib

warnings.filterwarnings('ignore')

# Leaked features to remove
LEAKED_FEATURES = [
    'is_shot_assist', 'has_recipient', 'duration',
    'pass_end_x', 'pass_end_y', 'pass_length', 'pass_angle'
]

# Non-features
NON_FEATURES = [
    'match_id', 'event_id', 'outcome', 'leads_to_shot',
    'pass_outcome', 'pass_height', 'pass_body_part',
    'pass_technique', 'corner_x', 'corner_y'
]

print("Loading data...")
df = pd.read_csv('data/processed/corners_features_with_shot.csv')
print(f"Original shape: {df.shape}")
print(f"Target distribution:\n{df['leads_to_shot'].value_counts()}")

# Get clean features
all_cols = df.columns.tolist()
features_to_keep = []
for col in all_cols:
    if col not in NON_FEATURES and col not in LEAKED_FEATURES and col != 'leads_to_shot':
        features_to_keep.append(col)

print(f"\nKeeping {len(features_to_keep)} clean features")

# Prepare data
X = df[features_to_keep].fillna(df[features_to_keep].median())
y = df['leads_to_shot'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Random Forest...")
# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)[:, 1]

print(f"\nClean Baseline Performance:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"  MCC: {matthews_corrcoef(y_test, y_pred):.3f}")

# Feature importance
print("\n" + "="*60)
print("TOP 20 FEATURES (Random Forest Importance)")
print("="*60)

importance_df = pd.DataFrame({
    'feature': features_to_keep,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance_df.head(20).iterrows():
    print(f"{row['feature']:30s}: {row['importance']:.4f}")

# Permutation importance
print("\n" + "="*60)
print("PERMUTATION IMPORTANCE (Top 20)")
print("="*60)

perm_imp = permutation_importance(
    rf, X_test_scaled, y_test,
    n_repeats=10, random_state=42, scoring='roc_auc'
)

perm_df = pd.DataFrame({
    'feature': features_to_keep,
    'importance': perm_imp.importances_mean,
    'std': perm_imp.importances_std
}).sort_values('importance', ascending=False)

for _, row in perm_df.head(20).iterrows():
    print(f"{row['feature']:30s}: {row['importance']:.4f} ± {row['std']:.4f}")

# Statistical significance
print("\n" + "="*60)
print("STATISTICAL SIGNIFICANCE (Top 10 Correlated)")
print("="*60)

sig_results = []
for col in features_to_keep:
    if X[col].nunique() > 2:
        corr, p_val = stats.pointbiserialr(y, X[col].fillna(X[col].median()))
    else:
        # Binary feature
        contingency = pd.crosstab(X[col].fillna(0), y)
        chi2, p_val, _, _ = stats.chi2_contingency(contingency)
        corr = np.sqrt(chi2 / len(y))

    sig_results.append({
        'feature': col,
        'correlation': abs(corr),
        'p_value': p_val,
        'significant': p_val < 0.05
    })

sig_df = pd.DataFrame(sig_results).sort_values('correlation', ascending=False)

for _, row in sig_df.head(10).iterrows():
    sig_marker = "*" if row['significant'] else ""
    print(f"{row['feature']:30s}: r={row['correlation']:.3f}, p={row['p_value']:.3e} {sig_marker}")

# Low impact features
print("\n" + "="*60)
print("LOW IMPACT FEATURES (Candidates for Removal)")
print("="*60)

# Features with low importance and not significant
low_impact = []
for _, row in importance_df.iterrows():
    if row['importance'] < 0.005:
        sig_row = sig_df[sig_df['feature'] == row['feature']].iloc[0]
        if not sig_row['significant']:
            low_impact.append(row['feature'])

print(f"Found {len(low_impact)} low-impact features:")
for feat in low_impact:
    imp = importance_df[importance_df['feature'] == feat]['importance'].iloc[0]
    print(f"  - {feat}: importance={imp:.4f}")

# Save results
results = {
    'model_performance': {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'mcc': matthews_corrcoef(y_test, y_pred)
    },
    'feature_importance': importance_df.to_dict('records'),
    'permutation_importance': perm_df.to_dict('records'),
    'statistical_significance': sig_df.to_dict('records'),
    'low_impact_features': low_impact,
    'leaked_features_removed': LEAKED_FEATURES,
    'clean_features_count': len(features_to_keep)
}

output_dir = Path('results/feature_importance')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'simple_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save model
joblib.dump(rf, output_dir / 'random_forest_clean.pkl')
joblib.dump(scaler, output_dir / 'feature_scaler.pkl')

print(f"\n✓ Results saved to {output_dir}")

# Create simple visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Feature importance bar plot
ax = axes[0]
top_20 = importance_df.head(20)
ax.barh(range(len(top_20)), top_20['importance'].values)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'].values, fontsize=8)
ax.set_xlabel('Importance')
ax.set_title('Random Forest Feature Importance (Top 20)')
ax.invert_yaxis()

# Permutation importance with error bars
ax = axes[1]
top_20_perm = perm_df.head(20)
ax.barh(range(len(top_20_perm)), top_20_perm['importance'].values)
ax.set_yticks(range(len(top_20_perm)))
ax.set_yticklabels(top_20_perm['feature'].values, fontsize=8)
ax.set_xlabel('Importance')
ax.set_title('Permutation Importance (Top 20)')
ax.invert_yaxis()

# Add error bars
for i, row in enumerate(top_20_perm.itertuples()):
    ax.errorbar(row.importance, i, xerr=row.std*2,
               fmt='none', color='black', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance_simple.png', dpi=150)
plt.close()

print(f"✓ Plot saved to {output_dir}/feature_importance_simple.png")