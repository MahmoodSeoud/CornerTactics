#!/usr/bin/env python3
"""Analyze dataset statistics to answer data quality questions."""

import pandas as pd
import pickle
from pathlib import Path

def main():
    # Load the unified dataset
    df = pd.read_parquet('data/processed/unified_corners_dataset.parquet')

    print('=' * 80)
    print('FINAL STATISTICS FOR YOUR QUESTIONS')
    print('=' * 80)

    # Question 1: Number of corners with 360 freeze frames
    corners_with_360 = df['has_player_positions'].sum()
    print(f'\n1. Corners with complete 360 freeze frames: {corners_with_360}')

    if corners_with_360 >= 1000:
        assessment = 'GOOD for deep learning (>1000)'
    elif corners_with_360 >= 500:
        assessment = 'MARGINAL - will need heavy regularization (<1000)'
    else:
        assessment = 'TOO SMALL for deep learning (<500)'

    print(f'   Assessment: {assessment}')
    print(f'   Comparison to TacticAI: {corners_with_360}/7176 = {corners_with_360/7176*100:.1f}% of their dataset')

    # Question 2: Velocity/Orientation data
    print(f'\n2. Velocity/Orientation data in freeze frames: NO')
    print(f'   Assessment: StatsBomb 360 only provides (x,y) positions at snapshot time')
    print(f'   Impact: Will underperform TacticAI unless we compute synthetic velocities')

    # Question 3: Shot rate
    total_corners = len(df)
    shots = df['outcome_category'].isin(['Shot', 'Goal']).sum()
    shot_rate = shots / total_corners * 100

    print(f'\n3. Shot rate (shots OR goals): {shots}/{total_corners} = {shot_rate:.1f}%')

    if shot_rate >= 15:
        assessment = 'REASONABLE positive class'
    else:
        assessment = 'LOW - will need class weighting or oversampling'

    print(f'   Assessment: {assessment}')

    # Question 4: Validation set shots
    val_split = 0.2
    test_split = 0.2
    train_corners = corners_with_360 * (1 - val_split - test_split)
    val_corners = corners_with_360 * val_split
    test_corners = corners_with_360 * test_split

    val_shots = val_corners * (shot_rate / 100)
    test_shots = test_corners * (shot_rate / 100)

    print(f'\n4. Shots in validation/test sets (assuming 20% each):')
    print(f'   Training: {int(train_corners)} corners')
    print(f'   Validation: {int(val_corners)} corners → ~{int(val_shots)} shots')
    print(f'   Test: {int(test_corners)} corners → ~{int(test_shots)} shots')

    if test_shots >= 30:
        assessment = 'SUFFICIENT for evaluation (>30 shots)'
    elif test_shots >= 20:
        assessment = 'MARGINAL (20-30 shots)'
    else:
        assessment = 'TOO FEW for reliable metrics (<20 shots)'

    print(f'   Assessment: {assessment}')

    print('\n' + '=' * 80)
    print('AUGMENTED DATASET (WITH TEMPORAL FRAMES)')
    print('=' * 80)

    # Check if augmented graphs exist
    graph_path = Path('data/graphs/adjacency_team/combined_temporal_graphs.pkl')

    if graph_path.exists():
        with open(graph_path, 'rb') as f:
            graphs = pickle.load(f)

        print(f'\nTotal augmented graphs: {len(graphs)}')

        # Count dangerous situations
        dangerous_count = sum(1 for g in graphs if g.get('dangerous_situation', False))
        print(f'Dangerous situations: {dangerous_count} ({dangerous_count/len(graphs)*100:.1f}%)')

        # Augmentation factor
        aug_factor = len(graphs) / corners_with_360
        print(f'Augmentation factor: {aug_factor:.1f}x')

        # Recalculate for augmented dataset
        val_graphs = len(graphs) * val_split
        test_graphs = len(graphs) * test_split
        dangerous_rate = dangerous_count / len(graphs)

        val_dangerous = val_graphs * dangerous_rate
        test_dangerous = test_graphs * dangerous_rate

        print(f'\nWith augmentation:')
        print(f'   Validation: {int(val_graphs)} graphs → ~{int(val_dangerous)} dangerous situations')
        print(f'   Test: {int(test_graphs)} graphs → ~{int(test_dangerous)} dangerous situations')

        if test_dangerous >= 100:
            assessment = 'EXCELLENT for evaluation'
        elif test_dangerous >= 50:
            assessment = 'GOOD for evaluation'
        else:
            assessment = 'ADEQUATE'

        print(f'   Assessment: {assessment}')

    else:
        print('\nAugmented dataset not found at data/graphs/adjacency_team/combined_temporal_graphs.pkl')

    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'\nBase dataset: {corners_with_360} corners with 360 data')
    print(f'Shot rate: {shot_rate:.1f}%')
    print(f'Velocity data: NO (only static positions)')
    print(f'Test set shots: ~{int(test_shots)} (without augmentation)')

    if graph_path.exists():
        print(f'\nAugmented dataset: {len(graphs)} graphs ({aug_factor:.1f}x increase)')
        print(f'Dangerous situations: {dangerous_count} ({dangerous_count/len(graphs)*100:.1f}%)')
        print(f'Test set dangerous: ~{int(test_dangerous)}')

if __name__ == '__main__':
    main()
