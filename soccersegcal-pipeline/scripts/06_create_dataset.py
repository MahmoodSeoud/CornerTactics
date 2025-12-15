#!/usr/bin/env python3
"""
Step 6: Create final labeled dataset for training.

Combines:
- Projected player positions
- Corner metadata
- Outcome labels (from existing data or inferred)

Output format suitable for GNN-based outcome prediction.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Outcome classes
OUTCOMES = {
    'goal': 0,
    'shot_on_target': 1,
    'shot_off_target': 2,
    'header_won': 3,
    'clearance': 4,
    'foul': 5,
    'out_of_play': 6,
    'possession_retained': 7,
    'unknown': 8
}


def load_existing_labels(
    labels_file: str = "/home/mseo/CornerTactics/data/processed/corners_with_shot_labels.json"
) -> dict:
    """
    Load existing outcome labels if available.

    Returns:
        Dict mapping corner_id to outcome
    """
    try:
        with open(labels_file) as f:
            data = json.load(f)

        labels = {}
        for item in data:
            corner_id = item.get('corner_id')
            if corner_id is not None:
                # Map existing labels to our outcomes
                if item.get('resulted_in_shot', False):
                    labels[corner_id] = 'shot_on_target'
                elif item.get('resulted_in_goal', False):
                    labels[corner_id] = 'goal'
                else:
                    labels[corner_id] = 'clearance'  # Default

        return labels

    except FileNotFoundError:
        logger.warning(f"Labels file not found: {labels_file}")
        return {}
    except Exception as e:
        logger.warning(f"Error loading labels: {e}")
        return {}


def create_dataset(
    positions_file: str,
    corners_index_file: str,
    output_file: str,
    existing_labels: dict = None
) -> list:
    """
    Create final labeled dataset.

    Args:
        positions_file: JSON with projected positions
        corners_index_file: CSV with corner metadata
        output_file: Output dataset file
        existing_labels: Optional dict of corner_id -> outcome

    Returns:
        List of dataset samples
    """

    # Load data
    with open(positions_file) as f:
        all_positions = json.load(f)

    corners_df = pd.read_csv(corners_index_file)
    corner_info = corners_df.set_index('corner_id').to_dict('index')

    existing_labels = existing_labels or {}

    dataset = []

    # Group positions by corner_id
    positions_by_corner = defaultdict(list)
    for pos in all_positions:
        positions_by_corner[pos['corner_id']].append(pos)

    for corner_id, frames in tqdm(positions_by_corner.items(), desc="Creating dataset"):
        # Get corner metadata
        info = corner_info.get(corner_id, {})

        # Get outcome label
        outcome = existing_labels.get(corner_id, 'unknown')
        outcome_label = OUTCOMES.get(outcome, OUTCOMES['unknown'])

        # Use frame at offset 0 (corner moment) as primary
        primary_frame = None
        delivery_frame = None
        outcome_frame = None

        for frame in frames:
            offset = frame['offset_ms']
            if offset == 0:
                primary_frame = frame
            elif offset == 2000:
                delivery_frame = frame
            elif offset == 5000:
                outcome_frame = frame

        if primary_frame is None:
            continue

        # Extract player positions as numpy array
        players = primary_frame['players']
        positions = np.array([p['pitch_position'] for p in players])

        # Create sample
        sample = {
            'corner_id': corner_id,
            'game_path': info.get('game_path', ''),
            'timestamp_seconds': info.get('timestamp_seconds', 0),
            'team': info.get('team', 'unknown'),
            'outcome': outcome,
            'outcome_label': outcome_label,
            'num_players': len(players),
            'player_positions': positions.tolist(),
            'player_confidences': [p['confidence'] for p in players],
            # Add temporal data if available
            'has_delivery_frame': delivery_frame is not None,
            'has_outcome_frame': outcome_frame is not None
        }

        if delivery_frame:
            sample['delivery_positions'] = [p['pitch_position'] for p in delivery_frame['players']]

        if outcome_frame:
            sample['outcome_positions'] = [p['pitch_position'] for p in outcome_frame['players']]

        dataset.append(sample)

    # Save dataset
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Also save as parquet for efficient loading
    parquet_file = output_file.replace('.json', '.parquet')
    save_as_parquet(dataset, parquet_file)

    # Summary
    print_dataset_summary(dataset)

    return dataset


def save_as_parquet(dataset: list, output_file: str):
    """Save dataset in parquet format for efficient loading."""

    # Flatten for parquet
    records = []
    for sample in dataset:
        record = {
            'corner_id': sample['corner_id'],
            'game_path': sample['game_path'],
            'timestamp_seconds': sample['timestamp_seconds'],
            'team': sample['team'],
            'outcome': sample['outcome'],
            'outcome_label': sample['outcome_label'],
            'num_players': sample['num_players'],
            'has_delivery_frame': sample['has_delivery_frame'],
            'has_outcome_frame': sample['has_outcome_frame']
        }

        # Store positions as JSON strings (parquet doesn't handle nested arrays well)
        record['player_positions_json'] = json.dumps(sample['player_positions'])
        record['player_confidences_json'] = json.dumps(sample['player_confidences'])

        if 'delivery_positions' in sample:
            record['delivery_positions_json'] = json.dumps(sample['delivery_positions'])
        if 'outcome_positions' in sample:
            record['outcome_positions_json'] = json.dumps(sample['outcome_positions'])

        records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(output_file, index=False)
    print(f"Saved parquet to {output_file}")


def print_dataset_summary(dataset: list):
    """Print dataset statistics."""

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(dataset)}")

    # Outcome distribution
    outcome_counts = defaultdict(int)
    for sample in dataset:
        outcome_counts[sample['outcome']] += 1

    print("\nOutcome distribution:")
    for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(dataset)
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    # Player count stats
    player_counts = [s['num_players'] for s in dataset]
    print(f"\nPlayers per sample: min={min(player_counts)}, max={max(player_counts)}, avg={np.mean(player_counts):.1f}")

    # Team distribution
    team_counts = defaultdict(int)
    for sample in dataset:
        team_counts[sample['team']] += 1

    print("\nTeam distribution:")
    for team, count in sorted(team_counts.items(), key=lambda x: -x[1]):
        print(f"  {team}: {count}")

    # Temporal data availability
    with_delivery = sum(1 for s in dataset if s['has_delivery_frame'])
    with_outcome = sum(1 for s in dataset if s['has_outcome_frame'])
    print(f"\nSamples with delivery frame: {with_delivery} ({100*with_delivery/len(dataset):.1f}%)")
    print(f"Samples with outcome frame: {with_outcome} ({100*with_outcome/len(dataset):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Create final training dataset')
    parser.add_argument('--positions',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/corner_positions.json',
                        help='Path to projected positions JSON')
    parser.add_argument('--corners-index',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/corner_index.csv',
                        help='Path to corners index CSV')
    parser.add_argument('--labels',
                        default='/home/mseo/CornerTactics/data/processed/corners_with_shot_labels.json',
                        help='Path to existing outcome labels')
    parser.add_argument('--output',
                        default='/home/mseo/CornerTactics/soccersegcal-pipeline/data/corner_dataset.json',
                        help='Output dataset file')
    args = parser.parse_args()

    # Load existing labels if available
    existing_labels = load_existing_labels(args.labels)
    print(f"Loaded {len(existing_labels)} existing outcome labels")

    # Create dataset
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset(
        args.positions,
        args.corners_index,
        args.output,
        existing_labels
    )

    print(f"\nDataset saved to {args.output}")


if __name__ == "__main__":
    main()
