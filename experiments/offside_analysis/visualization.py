"""Visualization module for offside signal investigation.

Creates visualizations of player positions and feature distributions
to understand spatial patterns that may predict offside outcomes.
"""

from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from experiments.offside_analysis.feature_extraction import (
    get_attackers, get_defenders, find_last_defender_x, extract_offside_features
)


# Pitch dimensions (StatsBomb)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def draw_pitch(ax: plt.Axes, color: str = 'black', linewidth: float = 1.0) -> plt.Axes:
    """Draw a football pitch on the given axes.

    Args:
        ax: Matplotlib axes
        color: Line color
        linewidth: Line width

    Returns:
        Axes with pitch drawn
    """
    # Pitch outline
    ax.plot([0, 0], [0, PITCH_WIDTH], color=color, linewidth=linewidth)
    ax.plot([0, PITCH_LENGTH], [PITCH_WIDTH, PITCH_WIDTH], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH, PITCH_LENGTH], [PITCH_WIDTH, 0], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH, 0], [0, 0], color=color, linewidth=linewidth)

    # Center line
    ax.plot([PITCH_LENGTH/2, PITCH_LENGTH/2], [0, PITCH_WIDTH], color=color, linewidth=linewidth)

    # Center circle
    center_circle = plt.Circle((PITCH_LENGTH/2, PITCH_WIDTH/2), 9.15,
                                color=color, fill=False, linewidth=linewidth)
    ax.add_patch(center_circle)

    # Penalty areas (18-yard box)
    # Left
    ax.plot([0, 16.5], [62, 62], color=color, linewidth=linewidth)
    ax.plot([16.5, 16.5], [62, 18], color=color, linewidth=linewidth)
    ax.plot([16.5, 0], [18, 18], color=color, linewidth=linewidth)

    # Right
    ax.plot([PITCH_LENGTH, PITCH_LENGTH-16.5], [62, 62], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH-16.5, PITCH_LENGTH-16.5], [62, 18], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH-16.5, PITCH_LENGTH], [18, 18], color=color, linewidth=linewidth)

    # 6-yard boxes
    # Left
    ax.plot([0, 5.5], [54.8, 54.8], color=color, linewidth=linewidth)
    ax.plot([5.5, 5.5], [54.8, 25.2], color=color, linewidth=linewidth)
    ax.plot([5.5, 0], [25.2, 25.2], color=color, linewidth=linewidth)

    # Right
    ax.plot([PITCH_LENGTH, PITCH_LENGTH-5.5], [54.8, 54.8], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH-5.5, PITCH_LENGTH-5.5], [54.8, 25.2], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH-5.5, PITCH_LENGTH], [25.2, 25.2], color=color, linewidth=linewidth)

    # Goals
    ax.plot([0, -2], [44, 44], color=color, linewidth=linewidth*2)
    ax.plot([-2, -2], [44, 36], color=color, linewidth=linewidth*2)
    ax.plot([-2, 0], [36, 36], color=color, linewidth=linewidth*2)

    ax.plot([PITCH_LENGTH, PITCH_LENGTH+2], [44, 44], color=color, linewidth=linewidth*2)
    ax.plot([PITCH_LENGTH+2, PITCH_LENGTH+2], [44, 36], color=color, linewidth=linewidth*2)
    ax.plot([PITCH_LENGTH+2, PITCH_LENGTH], [36, 36], color=color, linewidth=linewidth*2)

    ax.set_xlim(-5, PITCH_LENGTH + 5)
    ax.set_ylim(-5, PITCH_WIDTH + 5)
    ax.set_aspect('equal')
    ax.axis('off')

    return ax


def draw_half_pitch(ax: plt.Axes, color: str = 'black', linewidth: float = 1.0) -> plt.Axes:
    """Draw the attacking half of the pitch.

    Args:
        ax: Matplotlib axes
        color: Line color
        linewidth: Line width

    Returns:
        Axes with half pitch drawn
    """
    half_start = PITCH_LENGTH / 2

    # Pitch outline (right half only)
    ax.plot([half_start, half_start], [0, PITCH_WIDTH], color=color, linewidth=linewidth)
    ax.plot([half_start, PITCH_LENGTH], [PITCH_WIDTH, PITCH_WIDTH], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH, PITCH_LENGTH], [PITCH_WIDTH, 0], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH, half_start], [0, 0], color=color, linewidth=linewidth)

    # Penalty area
    ax.plot([PITCH_LENGTH, PITCH_LENGTH-16.5], [62, 62], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH-16.5, PITCH_LENGTH-16.5], [62, 18], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH-16.5, PITCH_LENGTH], [18, 18], color=color, linewidth=linewidth)

    # 6-yard box
    ax.plot([PITCH_LENGTH, PITCH_LENGTH-5.5], [54.8, 54.8], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH-5.5, PITCH_LENGTH-5.5], [54.8, 25.2], color=color, linewidth=linewidth)
    ax.plot([PITCH_LENGTH-5.5, PITCH_LENGTH], [25.2, 25.2], color=color, linewidth=linewidth)

    # Goal
    ax.plot([PITCH_LENGTH, PITCH_LENGTH+2], [44, 44], color=color, linewidth=linewidth*2)
    ax.plot([PITCH_LENGTH+2, PITCH_LENGTH+2], [44, 36], color=color, linewidth=linewidth*2)
    ax.plot([PITCH_LENGTH+2, PITCH_LENGTH], [36, 36], color=color, linewidth=linewidth*2)

    ax.set_xlim(half_start - 5, PITCH_LENGTH + 5)
    ax.set_ylim(-5, PITCH_WIDTH + 5)
    ax.set_aspect('equal')
    ax.axis('off')

    return ax


def compute_average_positions(corners: List[Dict[str, Any]]) -> Dict:
    """Compute average player positions grouped by outcome.

    Args:
        corners: List of corner dictionaries

    Returns:
        Dictionary with structure:
        {
            'shot': {'attackers': np.array, 'defenders': np.array},
            'no_shot': {'attackers': np.array, 'defenders': np.array}
        }
    """
    shot_attackers = []
    shot_defenders = []
    no_shot_attackers = []
    no_shot_defenders = []

    for corner in corners:
        freeze_frame = corner.get('freeze_frame', [])
        outcome = corner.get('shot_outcome', 0)

        attackers = get_attackers(freeze_frame, exclude_corner_taker=True)
        defenders = get_defenders(freeze_frame, exclude_goalkeeper=True)

        attacker_positions = [a['location'] for a in attackers]
        defender_positions = [d['location'] for d in defenders]

        if outcome == 1:
            shot_attackers.extend(attacker_positions)
            shot_defenders.extend(defender_positions)
        else:
            no_shot_attackers.extend(attacker_positions)
            no_shot_defenders.extend(defender_positions)

    return {
        'shot': {
            'attackers': np.array(shot_attackers) if shot_attackers else np.empty((0, 2)),
            'defenders': np.array(shot_defenders) if shot_defenders else np.empty((0, 2)),
        },
        'no_shot': {
            'attackers': np.array(no_shot_attackers) if no_shot_attackers else np.empty((0, 2)),
            'defenders': np.array(no_shot_defenders) if no_shot_defenders else np.empty((0, 2)),
        }
    }


def plot_average_positions(corners: List[Dict[str, Any]], figsize: tuple = (14, 6)) -> plt.Figure:
    """Plot average player positions for shot vs no-shot corners.

    Args:
        corners: List of corner dictionaries
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    avg_positions = compute_average_positions(corners)

    for idx, (outcome, label) in enumerate([('shot', 'Shot Corners'), ('no_shot', 'No-Shot Corners')]):
        ax = axes[idx]
        draw_half_pitch(ax)

        attackers = avg_positions[outcome]['attackers']
        defenders = avg_positions[outcome]['defenders']

        if len(attackers) > 0:
            ax.scatter(attackers[:, 0], attackers[:, 1], c='red', s=30, alpha=0.3, label='Attackers')
            # Mean position
            ax.scatter(attackers[:, 0].mean(), attackers[:, 1].mean(),
                      c='red', s=200, marker='X', edgecolors='black', label='Mean Attacker')

        if len(defenders) > 0:
            ax.scatter(defenders[:, 0], defenders[:, 1], c='blue', s=30, alpha=0.3, label='Defenders')
            ax.scatter(defenders[:, 0].mean(), defenders[:, 1].mean(),
                      c='blue', s=200, marker='X', edgecolors='black', label='Mean Defender')

        ax.set_title(label)
        ax.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    return fig


def create_position_heatmap(
    corners: List[Dict[str, Any]],
    player_type: str = 'attackers',
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Create position density heatmap.

    Args:
        corners: List of corner dictionaries
        player_type: 'attackers' or 'defenders'
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for idx, outcome in enumerate([1, 0]):
        ax = axes[idx]
        draw_half_pitch(ax)

        # Collect positions
        positions = []
        for corner in corners:
            if corner.get('shot_outcome', 0) != outcome:
                continue
            freeze_frame = corner.get('freeze_frame', [])
            if player_type == 'attackers':
                players = get_attackers(freeze_frame, exclude_corner_taker=True)
            else:
                players = get_defenders(freeze_frame, exclude_goalkeeper=True)
            positions.extend([p['location'] for p in players])

        if positions:
            positions = np.array(positions)
            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(
                positions[:, 0], positions[:, 1],
                bins=[20, 16],
                range=[[60, 120], [0, 80]]
            )
            extent = [60, 120, 0, 80]
            ax.imshow(heatmap.T, extent=extent, origin='lower',
                     cmap='Reds' if player_type == 'attackers' else 'Blues',
                     alpha=0.6, aspect='auto')

        title = f"{'Shot' if outcome == 1 else 'No-Shot'} - {player_type.title()}"
        ax.set_title(title)

    plt.tight_layout()
    return fig


def create_difference_heatmap(
    corners: List[Dict[str, Any]],
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Create heatmap showing positional differences between outcomes.

    Args:
        corners: List of corner dictionaries
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    draw_half_pitch(ax)

    # Collect positions by outcome
    shot_positions = []
    no_shot_positions = []

    for corner in corners:
        freeze_frame = corner.get('freeze_frame', [])
        attackers = get_attackers(freeze_frame, exclude_corner_taker=True)
        positions = [a['location'] for a in attackers]

        if corner.get('shot_outcome', 0) == 1:
            shot_positions.extend(positions)
        else:
            no_shot_positions.extend(positions)

    bins_x = np.linspace(60, 120, 21)
    bins_y = np.linspace(0, 80, 17)

    # Create histograms
    if shot_positions and no_shot_positions:
        shot_positions = np.array(shot_positions)
        no_shot_positions = np.array(no_shot_positions)

        shot_hist, _, _ = np.histogram2d(
            shot_positions[:, 0], shot_positions[:, 1],
            bins=[bins_x, bins_y]
        )
        no_shot_hist, _, _ = np.histogram2d(
            no_shot_positions[:, 0], no_shot_positions[:, 1],
            bins=[bins_x, bins_y]
        )

        # Normalize and compute difference
        shot_hist = shot_hist / (shot_hist.sum() + 1e-8)
        no_shot_hist = no_shot_hist / (no_shot_hist.sum() + 1e-8)
        diff = shot_hist - no_shot_hist

        # Custom colormap: blue (no-shot more) -> white -> red (shot more)
        cmap = LinearSegmentedColormap.from_list('diff', ['blue', 'white', 'red'])

        extent = [60, 120, 0, 80]
        im = ax.imshow(diff.T, extent=extent, origin='lower',
                      cmap=cmap, alpha=0.7, aspect='auto',
                      vmin=-0.03, vmax=0.03)
        plt.colorbar(im, ax=ax, label='Shot - No Shot Density')

    ax.set_title('Attacker Position Difference\n(Red = More common in shots)')

    plt.tight_layout()
    return fig


def plot_feature_distributions(
    corners: List[Dict[str, Any]],
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot distributions of offside-related features by outcome.

    Args:
        corners: List of corner dictionaries
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Extract features
    features_list = []
    for corner in corners:
        feats = extract_offside_features(corner)
        feats['shot_outcome'] = corner.get('shot_outcome', 0)
        features_list.append(feats)

    # Key features to plot
    feature_names = [
        'attackers_beyond_defender',
        'attacker_defender_gap',
        'last_defender_x',
        'defensive_line_spread',
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, feature_name in enumerate(feature_names):
        ax = axes[idx]

        shot_vals = [f[feature_name] for f in features_list if f['shot_outcome'] == 1 and not np.isnan(f[feature_name])]
        no_shot_vals = [f[feature_name] for f in features_list if f['shot_outcome'] == 0 and not np.isnan(f[feature_name])]

        if shot_vals and no_shot_vals:
            ax.hist(shot_vals, bins=20, alpha=0.5, label='Shot', color='red', density=True)
            ax.hist(no_shot_vals, bins=20, alpha=0.5, label='No Shot', color='blue', density=True)
            ax.legend()

        ax.set_xlabel(feature_name.replace('_', ' ').title())
        ax.set_ylabel('Density')

    plt.tight_layout()
    return fig


def plot_single_feature(
    corners: List[Dict[str, Any]],
    feature_name: str,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Plot distribution of a single feature.

    Args:
        corners: List of corner dictionaries
        feature_name: Name of feature to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    shot_vals = []
    no_shot_vals = []

    for corner in corners:
        feats = extract_offside_features(corner)
        val = feats.get(feature_name)
        if val is not None and not np.isnan(val):
            if corner.get('shot_outcome', 0) == 1:
                shot_vals.append(val)
            else:
                no_shot_vals.append(val)

    if shot_vals and no_shot_vals:
        ax.hist(shot_vals, bins=20, alpha=0.5, label='Shot', color='red', density=True)
        ax.hist(no_shot_vals, bins=20, alpha=0.5, label='No Shot', color='blue', density=True)
        ax.legend()

    ax.set_xlabel(feature_name.replace('_', ' ').title())
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {feature_name.replace("_", " ").title()}')

    return fig


def plot_single_corner(
    corner: Dict[str, Any],
    show_offside_line: bool = False,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Visualize a single corner freeze frame.

    Args:
        corner: Corner dictionary
        show_offside_line: Whether to draw the offside line
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    draw_half_pitch(ax)

    freeze_frame = corner.get('freeze_frame', [])

    # Plot players
    for player in freeze_frame:
        x, y = player['location']
        color = 'red' if player['teammate'] else 'blue'
        marker = '*' if player.get('keeper', False) else 'o'
        size = 150 if player.get('actor', False) else 100

        ax.scatter(x, y, c=color, s=size, marker=marker, edgecolors='black', linewidths=1)

    # Show offside line
    if show_offside_line:
        last_def_x = find_last_defender_x(freeze_frame)
        if last_def_x is not None:
            ax.axvline(x=last_def_x, color='green', linestyle='--',
                      linewidth=2, label=f'Offside Line (x={last_def_x:.1f})')
            ax.legend()

    # Title with outcome
    outcome = 'Shot' if corner.get('shot_outcome', 0) == 1 else 'No Shot'
    ax.set_title(f'Corner Freeze Frame - {outcome}')

    return fig
