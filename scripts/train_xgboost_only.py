#!/usr/bin/env python3
"""Quick XGBoost training only"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    print("Installing XGBoost...")
    import subprocess
    subprocess.run(['pip', 'install', 'xgboost==1.7.6'], check=True)
    import xgboost as xgb

# Load saved features list
project_root = Path(__file__).parent.parent
with open(project_root / 'models/clean_baselines/clean_features.json') as f:
    feature_info = json.load(f)
features = feature_info['features']

# Load data
df = pd.read_csv(project_root / 'data/processed/corners_features_with_shot.csv')
X = df[features].fillna(df[features].median())
y = df['leads_to_shot'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load scaler and transform
scaler = joblib.load(project_root / 'models/clean_baselines/feature_scaler.pkl')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training XGBoost with clean features...")
print(f"Features: {len(features)}")
print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

# Train XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# Train with early stopping
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Cross-validation with compatibility fix
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
cv_scores = []
for train_idx, val_idx in skf.split(X_train_scaled, y_train):
    X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

    cv_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    cv_model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)],
                 early_stopping_rounds=10, verbose=False)

    cv_pred_prob = cv_model.predict_proba(X_cv_val)[:, 1]
    cv_auc = roc_auc_score(y_cv_val, cv_pred_prob)
    cv_scores.append(cv_auc)

cv_scores = np.array(cv_scores)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print("XGBoost Clean Baseline Results")
print("="*50)
print(f"Accuracy:  {acc:.3f}")
print(f"AUC-ROC:   {auc:.3f}")
print(f"MCC:       {mcc:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"CV AUC:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"\nConfusion Matrix:")
print(f"TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
print(f"FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

# Save model
output_path = project_root / 'models/clean_baselines/xgboost_clean.pkl'
joblib.dump(model, output_path)
print(f"\nModel saved to: {output_path}")

# Update results file
results_path = project_root / 'models/clean_baselines/clean_baseline_results.json'
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
else:
    results = {'model_results': {}}

results['model_results']['XGBoost'] = {
    'accuracy': float(acc),
    'auc_roc': float(auc),
    'mcc': float(mcc),
    'f1': float(f1),
    'cv_auc_mean': float(cv_scores.mean()),
    'cv_auc_std': float(cv_scores.std())
}

with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results updated in: {results_path}")