"""Quality validation for extracted tracking data.

Computes per-corner quality metrics, dataset-level statistics, and
sanity checks for coordinate/velocity ranges.
"""

import math
import logging
from collections import Counter
from typing import Dict, List, Any

from .core import CornerTrackingData, Frame, PITCH_LENGTH, PITCH_WIDTH

logger = logging.getLogger(__name__)


def compute_quality_metrics(corner: CornerTrackingData) -> Dict[str, Any]:
    """Compute quality metrics for a single corner.

    Returns:
        Dict with:
            n_frames: total frames
            mean_players_per_frame: average detected players
            min_players_per_frame: worst frame
            pct_frames_with_20plus: % frames with >=20 players
            max_position_jump_m: largest position change between consecutive frames
            pct_visible: % of player-frames where is_visible=True
            has_ball_pct: % of frames with ball position
            velocity_mean_mps: mean absolute speed across all player-frames
            velocity_max_mps: max absolute speed across all player-frames
    """
    if not corner.frames:
        return {
            "n_frames": 0,
            "mean_players_per_frame": 0.0,
            "min_players_per_frame": 0,
            "pct_frames_with_20plus": 0.0,
            "max_position_jump_m": 0.0,
            "pct_visible": 0.0,
            "has_ball_pct": 0.0,
            "velocity_mean_mps": 0.0,
            "velocity_max_mps": 0.0,
        }

    n_frames = len(corner.frames)
    player_counts = [len(f.players) for f in corner.frames]
    mean_players = sum(player_counts) / n_frames
    min_players = min(player_counts) if player_counts else 0
    pct_20plus = sum(1 for c in player_counts if c >= 20) / n_frames * 100.0

    # Visibility
    total_pf = 0
    visible_pf = 0
    for f in corner.frames:
        for p in f.players:
            total_pf += 1
            if p.is_visible:
                visible_pf += 1
    pct_visible = (visible_pf / total_pf * 100.0) if total_pf > 0 else 0.0

    # Ball presence
    has_ball = sum(1 for f in corner.frames if f.ball_x is not None)
    has_ball_pct = has_ball / n_frames * 100.0

    # Velocity stats
    speeds = []
    for f in corner.frames:
        for p in f.players:
            if p.vx is not None and p.vy is not None:
                speed = math.sqrt(p.vx ** 2 + p.vy ** 2)
                speeds.append(speed)

    velocity_mean = sum(speeds) / len(speeds) if speeds else 0.0
    velocity_max = max(speeds) if speeds else 0.0

    # Max position jump between consecutive frames
    max_jump = 0.0
    prev_positions: Dict[str, tuple] = {}
    for f in corner.frames:
        for p in f.players:
            if p.player_id in prev_positions:
                px, py = prev_positions[p.player_id]
                dx = p.x - px
                dy = p.y - py
                jump = math.sqrt(dx * dx + dy * dy)
                if jump > max_jump:
                    max_jump = jump
            prev_positions[p.player_id] = (p.x, p.y)

    return {
        "n_frames": n_frames,
        "mean_players_per_frame": round(mean_players, 1),
        "min_players_per_frame": min_players,
        "pct_frames_with_20plus": round(pct_20plus, 1),
        "max_position_jump_m": round(max_jump, 2),
        "pct_visible": round(pct_visible, 1),
        "has_ball_pct": round(has_ball_pct, 1),
        "velocity_mean_mps": round(velocity_mean, 2),
        "velocity_max_mps": round(velocity_max, 2),
    }


def validate_corner(corner: CornerTrackingData) -> List[str]:
    """Run sanity checks on a single corner. Returns list of warnings."""
    warnings = []

    if not corner.frames:
        warnings.append("No frames")
        return warnings

    # Check coordinate bounds
    for f in corner.frames:
        for p in f.players:
            if p.x < -1.0 or p.x > PITCH_LENGTH + 1.0:
                warnings.append(
                    f"Frame {f.frame_idx}: player {p.player_id} x={p.x:.1f} out of bounds"
                )
                break
            if p.y < -1.0 or p.y > PITCH_WIDTH + 1.0:
                warnings.append(
                    f"Frame {f.frame_idx}: player {p.player_id} y={p.y:.1f} out of bounds"
                )
                break

        if f.ball_x is not None:
            if f.ball_x < -5.0 or f.ball_x > PITCH_LENGTH + 5.0:
                warnings.append(f"Frame {f.frame_idx}: ball x={f.ball_x:.1f} out of bounds")
            if f.ball_y is not None and (f.ball_y < -5.0 or f.ball_y > PITCH_WIDTH + 5.0):
                warnings.append(f"Frame {f.frame_idx}: ball y={f.ball_y:.1f} out of bounds")

    # Check velocities are reasonable (< 15 m/s sprint max)
    for f in corner.frames:
        for p in f.players:
            if p.vx is not None and p.vy is not None:
                speed = math.sqrt(p.vx ** 2 + p.vy ** 2)
                if speed > 15.0:
                    warnings.append(
                        f"Frame {f.frame_idx}: player {p.player_id} speed={speed:.1f} m/s (too fast)"
                    )
                    break

    # Check delivery frame is within bounds
    if corner.delivery_frame < 0 or corner.delivery_frame >= len(corner.frames):
        warnings.append(
            f"delivery_frame={corner.delivery_frame} out of range [0, {len(corner.frames)-1}]"
        )

    # Check minimum player count
    metrics = compute_quality_metrics(corner)
    if metrics["mean_players_per_frame"] < 15:
        warnings.append(
            f"Low player count: {metrics['mean_players_per_frame']:.1f} avg (expect >=15)"
        )

    return warnings


def print_dataset_summary(corners: List[CornerTrackingData]) -> None:
    """Print summary statistics for a tracking dataset."""
    if not corners:
        print("Empty dataset")
        return

    print(f"\n{'='*60}")
    print(f"Dataset Summary: {len(corners)} corners")
    print(f"{'='*60}")

    # By source
    source_counts = Counter(c.source for c in corners)
    print(f"\nBy source:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    # Outcome distribution
    outcome_counts = Counter(c.outcome for c in corners)
    print(f"\nOutcome distribution:")
    for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        pct = count / len(corners) * 100
        print(f"  {outcome or 'None'}: {count} ({pct:.1f}%)")

    # Quality metrics
    all_metrics = [compute_quality_metrics(c) for c in corners]

    print(f"\nQuality metrics (mean across corners):")
    metric_keys = [
        ("n_frames", "Frames per corner"),
        ("mean_players_per_frame", "Players per frame"),
        ("pct_frames_with_20plus", "% frames with 20+ players"),
        ("has_ball_pct", "% frames with ball"),
        ("velocity_mean_mps", "Mean speed (m/s)"),
        ("velocity_max_mps", "Max speed (m/s)"),
        ("max_position_jump_m", "Max position jump (m)"),
    ]

    for key, label in metric_keys:
        vals = [m[key] for m in all_metrics]
        mean_val = sum(vals) / len(vals)
        min_val = min(vals)
        max_val = max(vals)
        print(f"  {label}: mean={mean_val:.1f}, min={min_val:.1f}, max={max_val:.1f}")

    # Validation warnings
    total_warnings = 0
    corners_with_warnings = 0
    for c in corners:
        w = validate_corner(c)
        if w:
            total_warnings += len(w)
            corners_with_warnings += 1

    print(f"\nValidation:")
    print(f"  Corners with warnings: {corners_with_warnings}/{len(corners)}")
    print(f"  Total warnings: {total_warnings}")
    print(f"{'='*60}\n")
