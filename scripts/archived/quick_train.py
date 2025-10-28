#!/usr/bin/env python3
"""
Quick and dirty training script - just get something running!
"""

import pickle
import numpy as np
from pathlib import Path

print("Quick training script - bypassing all the broken dependencies")
print("=" * 60)

# Load the graphs
graph_path = Path("data/graphs/adjacency_team/combined_temporal_graphs.pkl")
print(f"Loading graphs from {graph_path}")

try:
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs successfully")

    # Count positives
    positives = sum(1 for g in graphs if (hasattr(g, 'outcome_label') and g.outcome_label == "Shot") or (hasattr(g, 'goal_scored') and g.goal_scored))
    print(f"Positive samples: {positives} ({positives/len(graphs)*100:.1f}%)")

    # Basic statistics
    print("\nGraph statistics:")
    print(f"  Average nodes: {np.mean([g.node_features.shape[0] for g in graphs]):.1f}")
    print(f"  Average edges: {np.mean([g.edge_index.shape[1] if hasattr(g, 'edge_index') and g.edge_index is not None else 0 for g in graphs]):.1f}")

    # Try simple sklearn model as baseline
    print("\nTrying simple Random Forest as baseline...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

    # Extract simple features (just average of node features)
    X = []
    y = []

    for g in graphs:
        # Simple feature: just average all node features
        avg_features = np.mean(g.node_features, axis=0)
        X.append(avg_features)

        # Label
        is_dangerous = (hasattr(g, 'outcome_label') and g.outcome_label == "Shot") or (hasattr(g, 'goal_scored') and g.goal_scored)
        y.append(1 if is_dangerous else 0)

    X = np.array(X)
    y = np.array(y)

    print(f"\nFeature shape: {X.shape}")
    print(f"Label distribution: {np.mean(y)*100:.1f}% positive")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]

    print("\nResults on test set:")
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print(f"  Average Precision: {average_precision_score(y_test, y_pred_proba):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    print("\nTop 5 most important features:")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-5:][::-1]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. Feature {idx}: {importances[idx]:.3f}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Quick test complete!")
print("This gives us a baseline to compare GNN against")