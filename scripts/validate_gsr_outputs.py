#!/usr/bin/env python3
"""Validate GSR outputs and check tracking quality.

Phase 6: Validation & Quality Check
- Check for failed/missing state files
- Validate tracking quality per corner
- Generate summary statistics
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from postprocess_gsr import parse_state_file


def validate_state_file(state_path: Path) -> dict:
    """Validate a single state file and return quality metrics.

    Returns:
        dict with metrics: num_tracks, num_frames, players_at_t0,
        x_range, y_range, is_valid
    """
    try:
        df = parse_state_file(state_path)

        if df.empty:
            return {
                'corner_id': state_path.stem,
                'is_valid': False,
                'error': 'Empty DataFrame'
            }

        # Basic stats
        num_tracks = df['track_id'].nunique()
        num_frames = df['frame'].nunique()

        # Players at t=0 (frame 50)
        frame_50 = df[df['frame'] == 50]
        players_at_t0 = len(frame_50)

        # Coordinate ranges
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()

        # Check for issues
        # Note: Standard pitch is 105m x 68m, so coordinates go to +/-52.5m x +/-34m
        # Allow extra margin for camera calibration errors and players outside pitch
        issues = []
        if players_at_t0 < 5:
            issues.append(f'Low players at t=0: {players_at_t0}')
        if x_min < -65 or x_max > 65:
            issues.append(f'X out of range: [{x_min:.1f}, {x_max:.1f}]')
        if y_min < -50 or y_max > 50:
            issues.append(f'Y out of range: [{y_min:.1f}, {y_max:.1f}]')

        return {
            'corner_id': state_path.stem,
            'is_valid': len(issues) == 0,
            'num_tracks': num_tracks,
            'num_frames': num_frames,
            'players_at_t0': players_at_t0,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'issues': '; '.join(issues) if issues else None
        }

    except Exception as e:
        return {
            'corner_id': state_path.stem,
            'is_valid': False,
            'error': str(e)
        }


def check_missing_files(clips_dir: Path, states_dir: Path) -> list:
    """Find corner clips that don't have corresponding state files."""
    clips = set(p.stem.replace('corner_', 'CORNER-') for p in clips_dir.glob('corner_*.mp4'))
    states = set(p.stem for p in states_dir.glob('CORNER-*.pklz'))

    missing = clips - states
    return sorted(missing)


def main():
    parser = argparse.ArgumentParser(description='Validate GSR outputs')
    parser.add_argument('--states-dir', type=str,
                        default='/home/mseo/CornerTactics/outputs/states',
                        help='Directory containing CORNER-*.pklz state files')
    parser.add_argument('--clips-dir', type=str,
                        default='/home/mseo/CornerTactics/data/corner_clips',
                        help='Directory containing corner clips')
    parser.add_argument('--output', type=str,
                        default='/home/mseo/CornerTactics/outputs/validation_report.csv',
                        help='Output CSV path for validation report')
    args = parser.parse_args()

    states_dir = Path(args.states_dir)
    clips_dir = Path(args.clips_dir)

    print("=" * 60)
    print("GSR Output Validation")
    print("=" * 60)

    # Check for missing files
    print("\n1. Checking for missing state files...")
    missing = check_missing_files(clips_dir, states_dir)
    total_clips = len(list(clips_dir.glob('corner_*.mp4')))
    total_states = len(list(states_dir.glob('CORNER-*.pklz')))

    print(f"   Total clips: {total_clips}")
    print(f"   State files: {total_states}")
    print(f"   Missing: {len(missing)}")

    if missing and len(missing) <= 20:
        print(f"   Missing IDs: {missing}")

    # Validate each state file
    print("\n2. Validating state files...")
    state_files = sorted(states_dir.glob('CORNER-*.pklz'))
    results = []

    for state_file in tqdm(state_files, desc="Validating"):
        result = validate_state_file(state_file)
        results.append(result)

    # Create summary
    df = pd.DataFrame(results)

    valid_count = df['is_valid'].sum()
    invalid_count = (~df['is_valid']).sum()

    print(f"\n3. Validation Summary")
    print(f"   Valid: {valid_count} ({100*valid_count/len(df):.1f}%)")
    print(f"   Invalid: {invalid_count}")

    if 'players_at_t0' in df.columns:
        low_players = (df['players_at_t0'] < 5).sum()
        print(f"   Low players at t=0 (<5): {low_players}")
        print(f"   Avg players at t=0: {df['players_at_t0'].mean():.1f}")

    if 'num_tracks' in df.columns:
        print(f"   Avg tracks per corner: {df['num_tracks'].mean():.1f}")

    if 'num_frames' in df.columns:
        print(f"   Avg frames per corner: {df['num_frames'].mean():.1f}")

    # Save report
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"\n4. Saved validation report to: {output_path}")

    # List issues if any
    issues_df = df[df['issues'].notna()]
    if len(issues_df) > 0:
        print(f"\n5. Corners with issues ({len(issues_df)}):")
        for _, row in issues_df.head(10).iterrows():
            print(f"   {row['corner_id']}: {row['issues']}")
        if len(issues_df) > 10:
            print(f"   ... and {len(issues_df) - 10} more")


if __name__ == '__main__':
    main()
