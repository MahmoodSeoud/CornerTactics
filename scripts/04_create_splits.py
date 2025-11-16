#!/usr/bin/env python3
"""
Task 4: Create Train/Test Splits

Creates match-based stratified train/test splits (80/20) to prevent
data leakage while maintaining similar outcome class distributions.

Input:  data/processed/corners_with_features.csv
Output: data/processed/train_indices.csv
        data/processed/test_indices.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def _extract_group_info(group_df: pd.DataFrame, stratify_col: str) -> dict:
    """
    Extract information about a group.

    Args:
        group_df: DataFrame for a single group
        stratify_col: Column name for stratification

    Returns:
        Dictionary with group information
    """
    return {
        'indices': group_df.index.tolist(),
        'dominant_class': group_df[stratify_col].mode()[0]
    }


def stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stratified split on grouped data.

    Splits data by groups (e.g., matches) while maintaining stratification
    on a target variable (e.g., outcome classes). This prevents data leakage
    by ensuring entire groups stay together in either train or test.

    Args:
        df: DataFrame to split
        group_col: Column name for grouping (e.g., 'match_id')
        stratify_col: Column name for stratification (e.g., 'outcome')
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, test_indices) as numpy arrays
    """
    np.random.seed(random_state)

    # Extract group information with dominant class for stratification
    groups = df.groupby(group_col)
    group_data = [_extract_group_info(group_df, stratify_col)
                  for _, group_df in groups]

    # Organize groups by their dominant class
    groups_by_class = {}
    for group in group_data:
        dominant_class = group['dominant_class']
        if dominant_class not in groups_by_class:
            groups_by_class[dominant_class] = []
        groups_by_class[dominant_class].append(group)

    # Split each class proportionally
    train_indices = []
    test_indices = []

    for class_groups in groups_by_class.values():
        # Shuffle to randomize group assignment
        np.random.shuffle(class_groups)

        # Calculate split point for this class
        n_test_groups = int(len(class_groups) * test_size)

        # Assign groups to test and train
        for i, group in enumerate(class_groups):
            if i < n_test_groups:
                test_indices.extend(group['indices'])
            else:
                train_indices.extend(group['indices'])

    return np.array(train_indices), np.array(test_indices)


def print_split_statistics(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray
) -> None:
    """
    Print comprehensive statistics about the train/test split.

    Args:
        df: Full DataFrame
        train_indices: Row indices for training set
        test_indices: Row indices for test set
    """
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    # Print header
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT STATISTICS")
    print("="*60)

    # Sample counts
    total_samples = len(df)
    train_pct = len(train_indices) / total_samples * 100
    test_pct = len(test_indices) / total_samples * 100

    print(f"\nTotal samples: {total_samples}")
    print(f"Train samples: {len(train_indices)} ({train_pct:.1f}%)")
    print(f"Test samples:  {len(test_indices)} ({test_pct:.1f}%)")

    # Match overlap verification
    train_matches = set(train_df['match_id'].unique())
    test_matches = set(test_df['match_id'].unique())
    overlap = train_matches & test_matches

    print(f"\nTrain matches: {len(train_matches)}")
    print(f"Test matches:  {len(test_matches)}")
    print(f"Match overlap: {len(overlap)} (should be 0)")

    # Class distribution comparison
    print("\n" + "-"*60)
    print("CLASS DISTRIBUTION")
    print("-"*60)

    overall_dist = df['outcome'].value_counts(normalize=True).sort_index()
    train_dist = train_df['outcome'].value_counts(normalize=True).sort_index()
    test_dist = test_df['outcome'].value_counts(normalize=True).sort_index()

    print(f"\n{'Class':<20} {'Overall':>10} {'Train':>10} {'Test':>10} {'Diff':>10}")
    print("-"*60)

    for outcome_class in overall_dist.index:
        overall_pct = overall_dist[outcome_class] * 100
        train_pct = train_dist.get(outcome_class, 0) * 100
        test_pct = test_dist.get(outcome_class, 0) * 100
        diff = abs(train_pct - overall_pct)

        print(f"{outcome_class:<20} {overall_pct:>9.1f}% "
              f"{train_pct:>9.1f}% {test_pct:>9.1f}% {diff:>9.1f}%")

    print("="*60)


def main() -> None:
    """
    Main execution function.

    Loads corner features, creates match-based stratified splits,
    and saves train/test indices to CSV files.
    """
    # Define file paths
    base_dir = Path(__file__).parent.parent
    features_file = base_dir / 'data/processed/corners_with_features.csv'
    train_output = base_dir / 'data/processed/train_indices.csv'
    test_output = base_dir / 'data/processed/test_indices.csv'

    # Load features
    print(f"Loading features from {features_file}...")
    df = pd.read_csv(features_file)
    print(f"Loaded {len(df)} corners from {df['match_id'].nunique()} matches")

    # Create stratified group split
    print("\nCreating match-based stratified split...")
    train_indices, test_indices = stratified_group_split(
        df,
        group_col='match_id',
        stratify_col='outcome',
        test_size=0.2,
        random_state=42
    )

    # Display split statistics
    print_split_statistics(df, train_indices, test_indices)

    # Save train indices
    print(f"\nSaving train indices to {train_output}...")
    pd.DataFrame({'index': train_indices}).to_csv(train_output, index=False)

    # Save test indices
    print(f"Saving test indices to {test_output}...")
    pd.DataFrame({'index': test_indices}).to_csv(test_output, index=False)

    print("\nDone! Train/test splits created successfully.")


if __name__ == '__main__':
    main()
