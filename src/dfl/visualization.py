"""Visualization utilities for corner kick tracking data."""

from pathlib import Path
from typing import Dict, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_corner_frame(
    frame,
    velocities: Dict[str, Dict[str, float]],
    corner_event,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    velocity_scale: float = 0.3,
) -> plt.Figure:
    """
    Plot a single frame with player positions and velocity arrows.

    Args:
        frame: A kloppy tracking frame
        velocities: Dict mapping player_id -> {"vx": float, "vy": float}
        corner_event: The corner kick event (for identifying attacking team)
        title: Optional title for the plot
        save_path: Optional path to save the figure
        pitch_length: Length of pitch in meters (default 105)
        pitch_width: Width of pitch in meters (default 68)
        velocity_scale: Scale factor for velocity arrows

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    # Draw pitch
    _draw_pitch(ax, pitch_length, pitch_width)

    # Get attacking team
    attacking_team = corner_event.team if hasattr(corner_event, "team") else None

    # Plot players
    home_players_x, home_players_y = [], []
    away_players_x, away_players_y = [], []

    for player_id, pdata in frame.players_data.items():
        if pdata.coordinates is None:
            continue

        x = pdata.coordinates.x
        y = pdata.coordinates.y

        # Determine team
        player_team = None
        if hasattr(pdata, "team"):
            player_team = pdata.team

        # Determine if attacking or defending
        is_attacking = False
        if attacking_team and player_team:
            is_attacking = player_team == attacking_team

        # Color by team
        if is_attacking:
            color = "red"
            home_players_x.append(x)
            home_players_y.append(y)
        else:
            color = "blue"
            away_players_x.append(x)
            away_players_y.append(y)

        # Plot player
        ax.scatter(x, y, c=color, s=120, zorder=5, edgecolors="black", linewidths=1)

        # Draw velocity arrow
        if player_id in velocities:
            vx = velocities[player_id].get("vx", 0)
            vy = velocities[player_id].get("vy", 0)
            speed = (vx**2 + vy**2) ** 0.5

            if speed > 0.5:  # Only draw if moving meaningfully
                ax.arrow(
                    x,
                    y,
                    vx * velocity_scale,
                    vy * velocity_scale,
                    head_width=1.0,
                    head_length=0.5,
                    fc=color,
                    ec=color,
                    alpha=0.7,
                    zorder=4,
                )

    # Plot ball
    if hasattr(frame, "ball_coordinates") and frame.ball_coordinates:
        ball_x = frame.ball_coordinates.x
        ball_y = frame.ball_coordinates.y
        ax.scatter(
            ball_x,
            ball_y,
            c="yellow",
            s=200,
            edgecolors="black",
            linewidths=2,
            zorder=10,
            marker="o",
        )

    # Set title
    if title is None:
        title = f"Corner Kick - {corner_event.timestamp}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Attacking",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="Defending",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="yellow",
            markersize=10,
            markeredgecolor="black",
            label="Ball",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _draw_pitch(ax: plt.Axes, length: float = 105.0, width: float = 68.0):
    """Draw a soccer pitch on the given axes."""
    # Set limits with padding
    ax.set_xlim(-5, length + 5)
    ax.set_ylim(-5, width + 5)

    # Pitch outline
    ax.plot([0, 0, length, length, 0], [0, width, width, 0, 0], "k-", linewidth=2)

    # Halfway line
    ax.plot([length / 2, length / 2], [0, width], "k-", linewidth=1)

    # Center circle
    center_circle = plt.Circle(
        (length / 2, width / 2), 9.15, fill=False, color="black", linewidth=1
    )
    ax.add_patch(center_circle)

    # Center spot
    ax.scatter(length / 2, width / 2, c="black", s=20, zorder=5)

    # Penalty areas (left)
    ax.plot(
        [0, 16.5, 16.5, 0],
        [width / 2 - 20.15, width / 2 - 20.15, width / 2 + 20.15, width / 2 + 20.15],
        "k-",
        linewidth=1,
    )

    # Penalty areas (right)
    ax.plot(
        [length, length - 16.5, length - 16.5, length],
        [width / 2 - 20.15, width / 2 - 20.15, width / 2 + 20.15, width / 2 + 20.15],
        "k-",
        linewidth=1,
    )

    # Six-yard boxes (left)
    ax.plot(
        [0, 5.5, 5.5, 0],
        [width / 2 - 9.15, width / 2 - 9.15, width / 2 + 9.15, width / 2 + 9.15],
        "k-",
        linewidth=1,
    )

    # Six-yard boxes (right)
    ax.plot(
        [length, length - 5.5, length - 5.5, length],
        [width / 2 - 9.15, width / 2 - 9.15, width / 2 + 9.15, width / 2 + 9.15],
        "k-",
        linewidth=1,
    )

    # Goals
    goal_width = 7.32
    ax.plot(
        [-2, -2, 0, 0],
        [
            width / 2 - goal_width / 2,
            width / 2 + goal_width / 2,
            width / 2 + goal_width / 2,
            width / 2 - goal_width / 2,
        ],
        "k-",
        linewidth=2,
    )
    ax.plot(
        [length, length + 2, length + 2, length],
        [
            width / 2 - goal_width / 2,
            width / 2 - goal_width / 2,
            width / 2 + goal_width / 2,
            width / 2 + goal_width / 2,
        ],
        "k-",
        linewidth=2,
    )

    # Penalty spots
    ax.scatter(11, width / 2, c="black", s=20, zorder=5)
    ax.scatter(length - 11, width / 2, c="black", s=20, zorder=5)

    # Penalty arcs
    left_arc = patches.Arc(
        (11, width / 2), 18.3, 18.3, angle=0, theta1=308, theta2=52, color="black"
    )
    right_arc = patches.Arc(
        (length - 11, width / 2),
        18.3,
        18.3,
        angle=0,
        theta1=128,
        theta2=232,
        color="black",
    )
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    # Corner arcs
    for corner_x, corner_y in [(0, 0), (0, width), (length, 0), (length, width)]:
        theta1, theta2 = 0, 90
        if corner_x == 0 and corner_y == width:
            theta1, theta2 = 270, 360
        elif corner_x == length and corner_y == 0:
            theta1, theta2 = 90, 180
        elif corner_x == length and corner_y == width:
            theta1, theta2 = 180, 270
        corner_arc = patches.Arc(
            (corner_x, corner_y), 2, 2, angle=0, theta1=theta1, theta2=theta2
        )
        ax.add_patch(corner_arc)

    # Grass color
    ax.set_facecolor("#90EE90")
    ax.set_aspect("equal")

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
