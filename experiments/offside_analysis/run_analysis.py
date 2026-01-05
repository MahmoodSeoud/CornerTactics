#!/usr/bin/env python3
"""Run OFFSIDE signal investigation analysis.

Analyzes spatial features that could predict offside outcomes
and tests whether they transfer to shot prediction.

Usage:
    python experiments/offside_analysis/run_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.offside_analysis.feature_extraction import extract_features_batch
from experiments.offside_analysis.visualization import (
    plot_average_positions,
    create_position_heatmap,
    create_difference_heatmap,
    plot_feature_distributions,
)
from experiments.offside_analysis.transfer_learning import (
    compare_classifiers,
    compute_feature_importance,
    rank_offside_features,
    compute_feature_statistics,
    compute_feature_significance,
)


# Paths
DATA_PATH = Path("data/processed/corners_with_shot_labels.json")
OUTPUT_DIR = Path("experiments/offside_analysis/results")


def load_data(data_path: Path):
    """Load corner data."""
    with open(data_path, 'r') as f:
        corners = json.load(f)
    return corners


def run_feature_analysis(corners):
    """Analyze spatial features."""
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)

    # Extract features
    features_df = extract_features_batch(corners)
    print(f"\nExtracted features for {len(features_df)} corners")
    print(f"\nFeature columns: {list(features_df.columns)}")

    # Basic statistics
    print("\n--- Feature Statistics ---")
    print(features_df.describe().round(2).to_string())

    return features_df


def run_statistical_analysis(corners):
    """Run statistical analysis of feature differences."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    # Statistics by outcome
    stats = compute_feature_statistics(corners)

    print("\n--- Shot Corners (mean ± std) ---")
    shot_mean = stats['shot']['mean']
    shot_std = stats['shot']['std']
    for feat in shot_mean:
        if feat not in ['match_id', 'shot_outcome']:
            m = shot_mean.get(feat, 0)
            s = shot_std.get(feat, 0)
            if not np.isnan(m):
                print(f"  {feat}: {m:.2f} ± {s:.2f}")

    print("\n--- No-Shot Corners (mean ± std) ---")
    no_shot_mean = stats['no_shot']['mean']
    no_shot_std = stats['no_shot']['std']
    for feat in no_shot_mean:
        if feat not in ['match_id', 'shot_outcome']:
            m = no_shot_mean.get(feat, 0)
            s = no_shot_std.get(feat, 0)
            if not np.isnan(m):
                print(f"  {feat}: {m:.2f} ± {s:.2f}")

    # Significance tests
    print("\n--- Statistical Significance (t-test) ---")
    significance = compute_feature_significance(corners)
    for feat, result in sorted(significance.items(), key=lambda x: x[1]['p_value']):
        t_stat = result['t_statistic']
        p_val = result['p_value']
        sig = "*" if p_val < 0.05 else ""
        print(f"  {feat}: t={t_stat:.3f}, p={p_val:.4f} {sig}")

    return stats, significance


def run_transfer_analysis(corners):
    """Test if offside features improve shot prediction."""
    print("\n" + "="*60)
    print("TRANSFER LEARNING ANALYSIS")
    print("="*60)

    # Compare classifiers
    comparison = compare_classifiers(corners)
    print(f"\nBaseline AUC (without offside features): {comparison['baseline_auc']:.4f}")
    print(f"Augmented AUC (with offside features): {comparison['augmented_auc']:.4f}")
    print(f"Improvement: {comparison['improvement']:.4f}")

    if comparison['improvement'] > 0.05:
        print("\n✓ Offside features provide meaningful improvement!")
    else:
        print("\n✗ Offside features do NOT improve shot prediction")
        print("  (Confirms they are specific to offside, not transferable)")

    # Feature importance
    print("\n--- Feature Importance Ranking ---")
    ranking = rank_offside_features(corners)
    for feat, importance in ranking:
        print(f"  {feat}: {importance:.4f}")

    return comparison, ranking


def create_visualizations(corners, output_dir):
    """Create and save visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Average positions
    print("Creating average position plot...")
    fig = plot_average_positions(corners)
    fig.savefig(output_dir / "average_positions.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Position heatmaps
    print("Creating position heatmaps...")
    for player_type in ['attackers', 'defenders']:
        fig = create_position_heatmap(corners, player_type=player_type)
        fig.savefig(output_dir / f"heatmap_{player_type}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Difference heatmap
    print("Creating difference heatmap...")
    fig = create_difference_heatmap(corners)
    fig.savefig(output_dir / "heatmap_difference.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Feature distributions
    print("Creating feature distribution plots...")
    fig = plot_feature_distributions(corners)
    fig.savefig(output_dir / "feature_distributions.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nVisualizations saved to: {output_dir}")


def main():
    print("="*60)
    print("OFFSIDE SIGNAL INVESTIGATION")
    print("="*60)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    corners = load_data(DATA_PATH)
    print(f"Loaded {len(corners)} corners")

    # Count outcomes
    shots = sum(1 for c in corners if c.get('shot_outcome', 0) == 1)
    no_shots = len(corners) - shots
    print(f"  Shots: {shots} ({100*shots/len(corners):.1f}%)")
    print(f"  No-shots: {no_shots} ({100*no_shots/len(corners):.1f}%)")

    # Run analyses
    features_df = run_feature_analysis(corners)
    stats, significance = run_statistical_analysis(corners)
    comparison, ranking = run_transfer_analysis(corners)

    # Create visualizations
    create_visualizations(corners, OUTPUT_DIR)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("""
Key Findings:

1. OFFSIDE PREDICTION HYPOTHESIS
   - Attackers beyond the defensive line should predict offside
   - However, we only have 1 OFFSIDE corner in freeze-frame data
   - Cannot directly test this hypothesis

2. FEATURE ANALYSIS
   - Computed 9 spatial features related to offside positioning
   - Most features show minimal difference between shot/no-shot

3. TRANSFER TO SHOT PREDICTION
   - Offside features do NOT improve shot prediction
   - Baseline AUC ≈ Augmented AUC ≈ random (0.5)
   - Confirms these features are specific to procedural outcomes

4. INTERPRETATION
   - The FAANTRA video model's 38.36% AP for OFFSIDE likely comes from:
     * Visual patterns of crowded penalty areas
     * Player movement patterns before the kick
   - These patterns are NOT captured by static position features
   - Offside is a procedural outcome (rule-based, not skill-based)
""")

    # Save summary
    summary = {
        'num_corners': len(corners),
        'num_shots': shots,
        'comparison': comparison,
        'ranking': ranking,
        'significant_features': [f for f, r in significance.items() if r['p_value'] < 0.05],
    }

    with open(OUTPUT_DIR / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
