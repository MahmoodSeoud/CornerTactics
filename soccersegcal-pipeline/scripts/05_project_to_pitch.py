#!/usr/bin/env python3
"""
Step 5: Project player foot positions to pitch coordinates.

Uses camera calibration (homography) to transform image coordinates
to world (pitch) coordinates in meters.

Standard pitch dimensions: 105m x 68m
Origin at center of pitch:
- X: -52.5 to 52.5 (along length)
- Y: -34 to 34 (along width)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard pitch dimensions (meters)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def apply_homography(
    image_point: list,
    homography: np.ndarray,
    image_size: tuple = None
) -> list:
    """
    Apply homography to transform image point to pitch coordinates.

    Args:
        image_point: [x, y] in image pixels
        homography: 3x3 homography matrix
        image_size: Optional (height, width) for normalization

    Returns:
        [x, y] in pitch meters or None if invalid
    """
    # Normalize image coordinates if image_size provided
    if image_size:
        x_norm = image_point[0] / image_size[1]  # x / width
        y_norm = image_point[1] / image_size[0]  # y / height
        point = np.array([x_norm, y_norm, 1.0])
    else:
        point = np.array([image_point[0], image_point[1], 1.0])

    # Apply homography
    homography = np.array(homography)
    projected = homography @ point

    # Normalize by w
    if abs(projected[2]) < 1e-10:
        return None

    x_pitch = projected[0] / projected[2]
    y_pitch = projected[1] / projected[2]

    return [float(x_pitch), float(y_pitch)]


def validate_pitch_position(position: list, margin: float = 10.0) -> bool:
    """
    Check if position is within valid pitch bounds.

    Args:
        position: [x, y] in meters
        margin: Extra margin beyond pitch bounds

    Returns:
        True if valid position
    """
    if position is None:
        return False

    x, y = position

    # Check bounds with margin
    if abs(x) > (PITCH_LENGTH / 2 + margin):
        return False
    if abs(y) > (PITCH_WIDTH / 2 + margin):
        return False

    return True


def project_frame_detections(
    detections: list,
    calibration: dict,
    image_size: tuple = None
) -> list:
    """
    Project all detections in a frame to pitch coordinates.

    Args:
        detections: List of player detections
        calibration: Calibration result with homography
        image_size: Image dimensions (height, width)

    Returns:
        List of detections with pitch positions
    """
    if not calibration.get('success', False):
        return []

    homography = calibration.get('homography')
    if homography is None:
        return []

    # Use original size if available
    if 'original_size' in calibration:
        image_size = tuple(calibration['original_size'])

    projected = []

    for det in detections:
        foot_pos = det['foot_position']

        # Project foot position
        pitch_pos = apply_homography(foot_pos, homography, image_size)

        if validate_pitch_position(pitch_pos):
            projected.append({
                'image_position': foot_pos,
                'pitch_position': pitch_pos,
                'confidence': det['confidence'],
                'bbox': det['bbox']
            })

    return projected


def project_all_frames(
    detections_file: str,
    calibrations_file: str,
    output_file: str,
    min_players: int = 10
) -> list:
    """
    Project all player detections to pitch coordinates.

    Args:
        detections_file: JSON file with player detections
        calibrations_file: JSON file with camera calibrations
        output_file: Output file for projected positions
        min_players: Minimum players required per frame

    Returns:
        List of projected corner data
    """

    # Load data
    with open(detections_file) as f:
        all_detections = json.load(f)

    with open(calibrations_file) as f:
        all_calibrations = json.load(f)

    # Index calibrations by frame path
    calib_by_frame = {c['frame_path']: c for c in all_calibrations}

    projected_corners = []

    for det_result in tqdm(all_detections, desc="Projecting to pitch"):
        frame_path = det_result['frame_path']
        corner_id = det_result['corner_id']
        offset_ms = det_result['offset_ms']
        detections = det_result['detections']

        # Get calibration
        calib = calib_by_frame.get(frame_path)
        if calib is None:
            continue

        # Project detections
        players = project_frame_detections(detections, calib)

        # Filter frames with enough players
        if len(players) >= min_players:
            projected_corners.append({
                'corner_id': corner_id,
                'offset_ms': offset_ms,
                'frame_path': frame_path,
                'num_players': len(players),
                'players': players
            })

    # Save results
    with open(output_file, 'w') as f:
        json.dump(projected_corners, f, indent=2)

    # Summary
    print(f"\n=== Projection Summary ===")
    print(f"Total frames with {min_players}+ players: {len(projected_corners)}")

    if projected_corners:
        unique_corners = len(set(p['corner_id'] for p in projected_corners))
        print(f"Unique corners: {unique_corners}")

        player_counts = [p['num_players'] for p in projected_corners]
        print(f"Players per frame: min={min(player_counts)}, max={max(player_counts)}, avg={np.mean(player_counts):.1f}")

    return projected_corners


def visualize_pitch_positions(
    projected: dict,
    output_path: str
):
    """
    Visualize player positions on pitch diagram.

    Args:
        projected: Single frame projection result
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw pitch
    pitch_length = PITCH_LENGTH
    pitch_width = PITCH_WIDTH

    # Pitch outline
    ax.plot([-pitch_length/2, pitch_length/2], [-pitch_width/2, -pitch_width/2], 'g-', linewidth=2)
    ax.plot([-pitch_length/2, pitch_length/2], [pitch_width/2, pitch_width/2], 'g-', linewidth=2)
    ax.plot([-pitch_length/2, -pitch_length/2], [-pitch_width/2, pitch_width/2], 'g-', linewidth=2)
    ax.plot([pitch_length/2, pitch_length/2], [-pitch_width/2, pitch_width/2], 'g-', linewidth=2)

    # Center line
    ax.plot([0, 0], [-pitch_width/2, pitch_width/2], 'g-', linewidth=1)

    # Center circle
    circle = plt.Circle((0, 0), 9.15, fill=False, color='green', linewidth=1)
    ax.add_patch(circle)

    # Penalty areas (16.5m from goal line, 40.32m wide)
    for sign in [-1, 1]:
        x = sign * pitch_length/2
        rect = patches.Rectangle(
            (x - sign*16.5 if sign == 1 else x, -20.16),
            16.5, 40.32, fill=False, color='green', linewidth=1
        )
        ax.add_patch(rect)

    # Plot players
    for player in projected['players']:
        x, y = player['pitch_position']
        conf = player['confidence']
        ax.scatter(x, y, c='blue', s=100, alpha=0.7)
        ax.annotate(f'{conf:.2f}', (x, y), fontsize=8)

    ax.set_xlim(-60, 60)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.set_title(f"Corner {projected['corner_id']} - Offset {projected['offset_ms']}ms - {projected['num_players']} players")

    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Project player positions to pitch coordinates')
    parser.add_argument('--detections',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/player_detections.json',
                        help='Path to player detections JSON')
    parser.add_argument('--calibrations',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/camera_calibrations.json',
                        help='Path to camera calibrations JSON')
    parser.add_argument('--output',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/corner_positions.json',
                        help='Output positions file')
    parser.add_argument('--min-players',
                        type=int,
                        default=10,
                        help='Minimum players per frame')
    parser.add_argument('--visualize',
                        action='store_true',
                        help='Save pitch visualizations')
    parser.add_argument('--vis-dir',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/outputs/pitch_views',
                        help='Directory for visualizations')
    args = parser.parse_args()

    # Project positions
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    projected = project_all_frames(
        args.detections,
        args.calibrations,
        args.output,
        args.min_players
    )

    # Optionally visualize
    if args.visualize and projected:
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Saving pitch visualizations...")
        for proj in tqdm(projected[:10], desc="Visualizing"):  # First 10
            vis_path = Path(args.vis_dir) / f"pitch_{proj['corner_id']:04d}_{proj['offset_ms']}.png"
            visualize_pitch_positions(proj, str(vis_path))

    print(f"\nPositions saved to {args.output}")


if __name__ == "__main__":
    main()
