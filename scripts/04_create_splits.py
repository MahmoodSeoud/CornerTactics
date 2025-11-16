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
from collections import Counter


def stratified_group_split(df, group_col, stratify_col, test_size=0.2, random_state=42):
    """
    Perform stratified split on grouped data.

    Splits data by groups (e.g., matches) while maintaining stratification
    on a target variable (e.g., outcome classes).

    Args:
        df: DataFrame to split
        group_col: Column name for grouping (e.g., 'match_id')
        stratify_col: Column name for stratification (e.g., 'outcome')
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility

    Returns:
        train_indices, test_indices: Arrays of row indices for train and test sets
    """
    np.random.seed(random_state)

    # Get unique groups and their class distributions
    groups = df.groupby(group_col)
    group_data = []

    for group_id, group_df in groups:
        # Count classes in this group
        class_counts = group_df[stratify_col].value_counts().to_dict()
        group_data.append({
            'group_id': group_id,
            'indices': group_df.index.tolist(),
            'size': len(group_df),
            'class_counts': class_counts,
            'dominant_class': group_df[stratify_col].mode()[0]  # Most common class
        })

    # Sort groups by dominant class for stratification
    group_data.sort(key=lambda x: x['dominant_class'])

    # Calculate target test size
    total_samples = len(df)
    target_test_samples = int(total_samples * test_size)

    # Distribute groups to train and test sets
    train_indices = []
    test_indices = []
    test_count = 0

    # Use stratified sampling: iterate through groups and assign to test
    # until we reach target test size, ensuring class balance
    groups_by_class = {}
    for group in group_data:
        cls = group['dominant_class']
        if cls not in groups_by_class:
            groups_by_class[cls] = []
        groups_by_class[cls].append(group)

    # For each class, take approximately test_size fraction
    for cls, cls_groups in groups_by_class.items():
        # Shuffle groups within class
        np.random.shuffle(cls_groups)

        # Calculate how many from this class should go to test
        cls_test_count = int(len(cls_groups) * test_size)

        # Assign groups to test and train
        for i, group in enumerate(cls_groups):
            if i < cls_test_count:
                test_indices.extend(group['indices'])
            else:
                train_indices.extend(group['indices'])

    return np.array(train_indices), np.array(test_indices)


def print_split_statistics(df, train_indices, test_indices):
    """Print statistics about the train/test split."""
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT STATISTICS")
    print("="*60)

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    print(f"\nTotal samples: {len(df)}")
    print(f"Train samples: {len(train_indices)} ({len(train_indices)/len(df)*100:.1f}%)")
    print(f"Test samples:  {len(test_indices)} ({len(test_indices)/len(df)*100:.1f}%)")

    # Check match overlap
    train_matches = set(train_df['match_id'].unique())
    test_matches = set(test_df['match_id'].unique())
    overlap = train_matches & test_matches

    print(f"\nTrain matches: {len(train_matches)}")
    print(f"Test matches:  {len(test_matches)}")
    print(f"Match overlap: {len(overlap)} (should be 0)")

    # Class distributions
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

        print(f"{outcome_class:<20} {overall_pct:>9.1f}% {train_pct:>9.1f}% {test_pct:>9.1f}% {diff:>9.1f}%")

    print("="*60)


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    features_file = base_dir / 'data/processed/corners_with_features.csv'
    train_output = base_dir / 'data/processed/train_indices.csv'
    test_output = base_dir / 'data/processed/test_indices.csv'

    print(f"Loading features from {features_file}...")

    # Load features
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

    # Print statistics
    print_split_statistics(df, train_indices, test_indices)

    # Save indices
    print(f"\nSaving train indices to {train_output}...")
    train_df = pd.DataFrame({'index': train_indices})
    train_df.to_csv(train_output, index=False)

    print(f"Saving test indices to {test_output}...")
    test_df = pd.DataFrame({'index': test_indices})
    test_df.to_csv(test_output, index=False)

    print("\nDone! Train/test splits created successfully.")


if __name__ == '__main__':
    main()
