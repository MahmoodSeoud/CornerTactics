#!/usr/bin/env python3
"""
Save XGBoost Feature Importance

Extracts feature importance from a trained XGBoost model and saves to JSON.
Also creates a simple matplotlib bar chart using XGBoost's built-in plotting.

This script should be run after training XGBoost baselines:
    python scripts/training/train_outcome_baselines.py

Usage:
    python scripts/analysis/save_xgboost_importance.py \
        --model-path data/models/xgboost_outcome_baseline.pkl \
        --output-json data/results/feature_importance.json \
        --output-plot data/results/feature_importance_basic.png

Author: mseo
Date: November 2024
"""

import sys
import json
import pickle
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def extract_feature_importance_from_model(model_path: str) -> Dict[str, float]:
    """
    Load trained XGBoost model and extract feature importance.

    Args:
        model_path: Path to pickled XGBoostOutcomeBaseline model

    Returns:
        Dict mapping feature names to importance scores
    """
    with open(model_path, 'rb') as f:
        model_wrapper = pickle.load(f)

    # Get XGBoost booster
    booster = model_wrapper.model

    # Get feature importance as dict
    importance_dict = booster.get_score(importance_type='gain')  # 'gain', 'weight', or 'cover'

    # Sort by importance
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    return sorted_importance


def save_importance_json(importance: Dict[str, float], output_path: str):
    """
    Save feature importance to JSON file.

    Args:
        importance: Feature name -> importance score
        output_path: Where to save JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(importance, f, indent=2)

    print(f"✓ Saved feature importance to {output_path}")


def plot_basic_importance(importance: Dict[str, float], output_path: str, top_n: int = 20):
    """
    Create basic matplotlib bar chart of feature importance.

    Args:
        importance: Feature name -> importance score
        output_path: Where to save plot
        top_n: Number of top features to show
    """
    # Get top N features
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [f[0] for f in sorted_features]
    scores = [f[1] for f in sorted_features]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(features)), scores, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} XGBoost Features', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved basic importance plot to {output_path}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract and save XGBoost feature importance'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='data/models/xgboost_outcome_baseline.pkl',
        help='Path to trained XGBoost model (pickle file)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='data/results/feature_importance.json',
        help='Path to save feature importance JSON'
    )
    parser.add_argument(
        '--output-plot',
        type=str,
        default='data/results/feature_importance_basic.png',
        help='Path to save basic importance plot'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top features to plot'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("EXTRACTING XGBOOST FEATURE IMPORTANCE")
    print("=" * 80)
    print(f"Model: {args.model_path}\n")

    # Extract importance
    print("Loading model and extracting feature importance...")
    importance = extract_feature_importance_from_model(args.model_path)
    print(f"✓ Extracted {len(importance)} features\n")

    # Print top 10 features
    print("Top 10 Most Important Features:")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feature, score) in enumerate(sorted_features, 1):
        print(f"  {i:2d}. {feature:40s}: {score:8.2f}")

    print("\n" + "-" * 80)

    # Save JSON
    print("\nSaving outputs...")
    save_importance_json(importance, args.output_json)

    # Save basic plot
    plot_basic_importance(importance, args.output_plot, top_n=args.top_n)

    print("\n" + "=" * 80)
    print("✓ FEATURE IMPORTANCE EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Visualize with pitch context")
    print(f"  python scripts/visualization/visualize_feature_importance.py \\")
    print(f"      --importance-path {args.output_json} \\")
    print(f"      --output-dir data/results/feature_importance")
    print("\n")


if __name__ == "__main__":
    main()
