#!/usr/bin/env python3
"""
Visualize XGBoost Feature Importance with Pitch Context

Creates TacticAI-style visualizations that overlay feature importance on soccer pitch:
1. Spatial Feature Heatmap: Shows importance of position-based features on pitch
2. Feature Category Bar Chart: Groups features by type (spatial, kinematic, contextual, etc.)
3. Top Features with Pitch Annotations: Visualizes most important features with arrows/zones
4. Player-Centric View: Shows feature importance for attacking vs defending players

Author: mseo
Date: November 2024
"""

import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


# Feature categories (based on 14-dim node features + graph-level aggregations)
FEATURE_CATEGORIES = {
    'spatial': ['x', 'y', 'distance_to_goal', 'distance_to_ball', 'avg_x', 'std_x', 'avg_y', 'std_y'],
    'kinematic': ['vx', 'vy', 'velocity_magnitude', 'velocity_angle', 'avg_vel', 'max_vel'],
    'contextual': ['angle_to_goal', 'angle_to_ball', 'team_flag', 'in_penalty_box',
                   'attacking_players', 'defending_players'],
    'density': ['num_players_within_5m', 'local_density_score', 'avg_density', 'max_density',
                'penalty_area_density', 'six_yard_density'],
    'zonal': ['in_six_yard', 'near_post', 'far_post', 'central_zone'],
}


def load_feature_importance(importance_path: str) -> Dict[str, float]:
    """
    Load feature importance from JSON file.

    Args:
        importance_path: Path to feature_importance.json

    Returns:
        Dict mapping feature names to importance scores
    """
    with open(importance_path, 'r') as f:
        importance = json.load(f)

    # Normalize to 0-100 scale
    max_importance = max(importance.values()) if importance else 1.0
    normalized = {k: (v / max_importance * 100) for k, v in importance.items()}

    return normalized


def categorize_features(feature_importance: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Group features by category (spatial, kinematic, etc.).

    Args:
        feature_importance: Feature name -> importance score

    Returns:
        Category -> {feature_name -> importance}
    """
    categorized = {cat: {} for cat in FEATURE_CATEGORIES}
    categorized['other'] = {}

    for feature_name, importance in feature_importance.items():
        assigned = False
        for category, feature_list in FEATURE_CATEGORIES.items():
            if any(feat in feature_name.lower() for feat in feature_list):
                categorized[category][feature_name] = importance
                assigned = True
                break

        if not assigned:
            categorized['other'][feature_name] = importance

    # Remove empty categories
    categorized = {k: v for k, v in categorized.items() if v}

    return categorized


def plot_feature_categories_bar(
    feature_importance: Dict[str, float],
    output_path: str,
    top_n: int = 20
):
    """
    Create horizontal bar chart grouped by feature category.

    Args:
        feature_importance: Feature name -> importance score
        output_path: Where to save the figure
        top_n: Number of top features to show
    """
    # Get top N features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Categorize features
    categorized = categorize_features(dict(sorted_features))

    # Create color map for categories
    category_colors = {
        'spatial': '#FF6B6B',      # Red
        'kinematic': '#4ECDC4',    # Teal
        'contextual': '#FFD93D',   # Yellow
        'density': '#6BCF7F',      # Green
        'zonal': '#A78BFA',        # Purple
        'other': '#9CA3AF'         # Gray
    }

    # Prepare data
    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    colors = []

    for feature_name in features:
        for category, feats in categorized.items():
            if feature_name in feats:
                colors.append(category_colors.get(category, '#9CA3AF'))
                break

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot horizontal bars
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=10)
    ax.set_xlabel('Feature Importance (Normalized)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} XGBoost Features by Category', fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()  # Top feature at the top
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importances)):
        ax.text(val + 1, i, f'{val:.1f}', va='center', fontsize=9, fontweight='bold')

    # Add legend
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='black', linewidth=1)
                     for color in category_colors.values() if color in colors]
    legend_labels = [cat.title() for cat in category_colors.keys()
                    if category_colors[cat] in colors]
    ax.legend(legend_patches, legend_labels, loc='lower right', fontsize=10,
             title='Feature Category', title_fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature category bar chart to {output_path}")
    plt.close()


def plot_spatial_heatmap(
    feature_importance: Dict[str, float],
    output_path: str
):
    """
    Create pitch heatmap showing importance of spatial features.
    Overlays feature importance on key pitch zones.

    Args:
        feature_importance: Feature name -> importance score
        output_path: Where to save the figure
    """
    # Extract spatial features
    spatial_features = {k: v for k, v in feature_importance.items()
                       if any(x in k.lower() for x in ['x', 'y', 'position', 'distance', 'goal', 'ball'])}

    if not spatial_features:
        print("⚠ No spatial features found, skipping heatmap")
        return

    # Create figure with pitch
    fig, ax = plt.subplots(figsize=(16, 10))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22ab4d', line_color='white',
                  linewidth=2, stripe=False)
    pitch.draw(ax=ax)

    # Define key zones on the pitch with importance
    zones = {
        'Penalty Area': {'x': [102, 120], 'y': [18, 62], 'importance': 0},
        '6-Yard Box': {'x': [114, 120], 'y': [30, 50], 'importance': 0},
        'Near Post': {'x': [102, 120], 'y': [18, 35], 'importance': 0},
        'Far Post': {'x': [102, 120], 'y': [45, 62], 'importance': 0},
        'Central': {'x': [102, 120], 'y': [35, 45], 'importance': 0},
    }

    # Assign importance to zones based on features
    for feature_name, importance in spatial_features.items():
        if 'penalty' in feature_name.lower() or 'box' in feature_name.lower():
            zones['Penalty Area']['importance'] += importance
        if 'six_yard' in feature_name.lower():
            zones['6-Yard Box']['importance'] += importance
        if 'near_post' in feature_name.lower():
            zones['Near Post']['importance'] += importance
        if 'far_post' in feature_name.lower():
            zones['Far Post']['importance'] += importance
        if 'central' in feature_name.lower() or 'goal' in feature_name.lower():
            zones['Central']['importance'] += importance

    # Normalize zone importance
    max_zone_importance = max(z['importance'] for z in zones.values()) if zones else 1.0
    if max_zone_importance == 0:
        max_zone_importance = 1.0

    # Create custom colormap (white -> yellow -> orange -> red)
    cmap = LinearSegmentedColormap.from_list('importance',
                                             ['white', '#FFE66D', '#FF6B6B', '#C92A2A'])

    # Draw zones with heatmap colors
    for zone_name, zone_data in zones.items():
        if zone_data['importance'] > 0:
            x_min, x_max = zone_data['x']
            y_min, y_max = zone_data['y']

            # Normalize importance to 0-1
            normalized_importance = zone_data['importance'] / max_zone_importance
            color = cmap(normalized_importance)

            # Draw rectangle
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           facecolor=color, edgecolor='white', linewidth=2,
                           alpha=0.7, zorder=2)
            ax.add_patch(rect)

            # Add label
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            ax.text(center_x, center_y, f'{zone_name}\n{zone_data["importance"]:.1f}',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color='white', zorder=3,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

    # Add arrows showing important spatial features
    top_spatial = sorted(spatial_features.items(), key=lambda x: x[1], reverse=True)[:5]

    # Annotate top spatial features
    annotation_y = 5
    for i, (feature_name, importance) in enumerate(top_spatial):
        ax.text(5, annotation_y + i * 5, f'#{i+1}: {feature_name.replace("_", " ").title()}',
               fontsize=10, fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))

    ax.set_title('Spatial Feature Importance Heatmap\n(Darker zones = higher importance)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved spatial heatmap to {output_path}")
    plt.close()


def plot_top_features_with_pitch_annotations(
    feature_importance: Dict[str, float],
    output_path: str,
    top_n: int = 10
):
    """
    Create pitch visualization with arrows/annotations showing top features.
    Inspired by TacticAI's tactical arrow style.

    Args:
        feature_importance: Feature name -> importance score
        output_path: Where to save the figure
        top_n: Number of top features to visualize
    """
    # Get top N features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Create figure with pitch
    fig, ax = plt.subplots(figsize=(18, 12))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22ab4d', line_color='white',
                  linewidth=2.5, stripe=False)
    pitch.draw(ax=ax)

    # Color scheme for feature importance
    cmap = plt.get_cmap('YlOrRd')
    max_importance = max(f[1] for f in sorted_features)

    # Define visual representations for different feature types
    feature_visuals = {
        'distance_to_goal': {'pos': (115, 40), 'type': 'arrow', 'target': (120, 40)},
        'distance_to_ball': {'pos': (120, 0), 'type': 'arrow', 'target': (110, 35)},
        'x': {'pos': (60, 5), 'type': 'line', 'target': (60, 75)},
        'y': {'pos': (5, 40), 'type': 'line', 'target': (115, 40)},
        'penalty': {'pos': (111, 40), 'type': 'zone', 'bounds': [102, 120, 18, 62]},
        'six_yard': {'pos': (117, 40), 'type': 'zone', 'bounds': [114, 120, 30, 50]},
        'angle_to_goal': {'pos': (105, 30), 'type': 'arc', 'target': (120, 40)},
        'velocity': {'pos': (110, 50), 'type': 'arrow', 'target': (118, 55)},
        'density': {'pos': (110, 40), 'type': 'circle', 'radius': 5},
    }

    # Draw top features
    for i, (feature_name, importance) in enumerate(sorted_features):
        # Determine color based on importance
        color = cmap(importance / max_importance)

        # Find matching visual
        visual = None
        for key, vis in feature_visuals.items():
            if key in feature_name.lower():
                visual = vis
                break

        if visual is None:
            # Default: place text in left column
            ax.text(10, 70 - i * 6, f'#{i+1}: {feature_name.replace("_", " ").title()}\n({importance:.1f})',
                   fontsize=9, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.9, edgecolor='white', linewidth=2))
            continue

        # Draw visual representation
        pos_x, pos_y = visual['pos']

        if visual['type'] == 'arrow':
            target_x, target_y = visual['target']
            arrow = FancyArrowPatch((pos_x, pos_y), (target_x, target_y),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=3, color=color, zorder=10)
            ax.add_patch(arrow)

            # Add label
            mid_x = (pos_x + target_x) / 2
            mid_y = (pos_y + target_y) / 2
            ax.text(mid_x, mid_y + 3, f'{feature_name.replace("_", " ").title()}\n({importance:.1f})',
                   ha='center', fontsize=9, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9, edgecolor='white', linewidth=2))

        elif visual['type'] == 'zone':
            x_min, x_max, y_min, y_max = visual['bounds']
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           facecolor=color, edgecolor='white', linewidth=3,
                           alpha=0.5, zorder=2)
            ax.add_patch(rect)

            # Add label
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            ax.text(center_x, center_y, f'{feature_name.replace("_", " ").title()}\n({importance:.1f})',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.9, edgecolor='white', linewidth=2))

        elif visual['type'] == 'circle':
            radius = visual.get('radius', 5)
            circle = Circle((pos_x, pos_y), radius, facecolor=color, edgecolor='white',
                          linewidth=2, alpha=0.5, zorder=2)
            ax.add_patch(circle)

            # Add label
            ax.text(pos_x, pos_y, f'{feature_name.replace("_", " ").title()}\n({importance:.1f})',
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9, edgecolor='white', linewidth=2))

    ax.set_title(f'Top {top_n} Features: TacticAI-Style Pitch Annotations',
                fontsize=16, fontweight='bold', pad=20)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_importance))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.6)
    cbar.set_label('Feature Importance', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved pitch annotations to {output_path}")
    plt.close()


def plot_all_visualizations(
    importance_path: str,
    output_dir: str = "data/results/feature_importance"
):
    """
    Generate all feature importance visualizations.

    Args:
        importance_path: Path to feature_importance.json
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("XGBOOST FEATURE IMPORTANCE VISUALIZATION")
    print("=" * 80)
    print(f"Loading feature importance from: {importance_path}\n")

    # Load feature importance
    feature_importance = load_feature_importance(importance_path)
    print(f"✓ Loaded {len(feature_importance)} features\n")

    # 1. Feature category bar chart
    print("Creating feature category bar chart...")
    plot_feature_categories_bar(
        feature_importance,
        output_path=str(output_dir / "feature_categories_bar.png"),
        top_n=20
    )

    # 2. Spatial heatmap
    print("Creating spatial feature heatmap...")
    plot_spatial_heatmap(
        feature_importance,
        output_path=str(output_dir / "spatial_heatmap.png")
    )

    # 3. Top features with pitch annotations (TacticAI style)
    print("Creating TacticAI-style pitch annotations...")
    plot_top_features_with_pitch_annotations(
        feature_importance,
        output_path=str(output_dir / "top_features_pitch.png"),
        top_n=10
    )

    print("\n" + "=" * 80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - feature_categories_bar.png: Grouped bar chart by feature category")
    print("  - spatial_heatmap.png: Heatmap of spatial feature importance")
    print("  - top_features_pitch.png: TacticAI-style pitch with arrows/zones")
    print("\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize XGBoost feature importance with pitch context'
    )
    parser.add_argument(
        '--importance-path',
        type=str,
        default='data/results/feature_importance.json',
        help='Path to feature_importance.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results/feature_importance',
        help='Directory to save visualizations'
    )

    args = parser.parse_args()

    plot_all_visualizations(
        importance_path=args.importance_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
