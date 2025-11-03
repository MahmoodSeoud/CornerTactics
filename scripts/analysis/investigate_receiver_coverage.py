#!/usr/bin/env python3
"""
Investigate Receiver Label Coverage

Analyzes why only 3,492 out of 5,814 graphs have receiver labels.
Goal: Determine if we can recover the missing ~2,300 graphs.

Author: mseo
Date: November 2024
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_graph_coverage():
    """Analyze receiver label coverage in graphs."""
    print('='*80)
    print('RECEIVER LABEL COVERAGE ANALYSIS')
    print('='*80)

    # Load graphs
    graph_path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl")
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)

    print(f'\nTotal graphs: {len(graphs)}')

    # Split by receiver label presence
    with_receiver = [g for g in graphs if g.receiver_node_index is not None]
    without_receiver = [g for g in graphs if g.receiver_node_index is None]

    print(f'With receiver labels: {len(with_receiver)} ({len(with_receiver)/len(graphs)*100:.1f}%)')
    print(f'Without receiver labels: {len(without_receiver)} ({len(without_receiver)/len(graphs)*100:.1f}%)')

    # Analyze outcome distribution
    print('\n' + '='*80)
    print('OUTCOME DISTRIBUTION')
    print('='*80)

    outcomes_with = Counter([g.outcome_label for g in with_receiver])
    outcomes_without = Counter([g.outcome_label for g in without_receiver])

    print('\nGraphs WITH receiver labels:')
    for outcome, count in outcomes_with.most_common():
        print(f'  {outcome:20s}: {count:4d} ({count/len(with_receiver)*100:5.1f}%)')

    print('\nGraphs WITHOUT receiver labels:')
    for outcome, count in outcomes_without.most_common():
        print(f'  {outcome:20s}: {count:4d} ({count/len(without_receiver)*100:5.1f}%)')

    # Key finding: 73.1% of graphs without receivers are Clearances
    clearance_pct = outcomes_without.get('Clearance', 0) / len(without_receiver) * 100
    print(f'\n⚠️  KEY FINDING: {clearance_pct:.1f}% of graphs without receivers are Clearances')

    return with_receiver, without_receiver


def analyze_raw_statsbomb_data():
    """Analyze receiver coverage in raw StatsBomb CSV."""
    print('\n' + '='*80)
    print('RAW STATSBOMB DATA ANALYSIS')
    print('='*80)

    # Load StatsBomb corners CSV
    corners_path = Path('data/raw/statsbomb/corners_360.csv')
    df = pd.read_csv(corners_path)

    print(f'\nTotal corners in CSV: {len(df)}')

    # Check receiver columns
    has_receiver_name = df['receiver_name'].notna()
    has_location = df['receiver_location_x'].notna() & df['receiver_location_y'].notna()
    full_info = has_receiver_name & has_location

    print('\nReceiver Data Coverage:')
    print(f'  Corners with receiver name: {has_receiver_name.sum()} ({has_receiver_name.sum()/len(df)*100:.1f}%)')
    print(f'  Corners with receiver location: {has_location.sum()} ({has_location.sum()/len(df)*100:.1f}%)')
    print(f'  Corners with BOTH name + location: {full_info.sum()} ({full_info.sum()/len(df)*100:.1f}%)')

    # What's missing?
    print('\nMissing Data Breakdown:')
    missing_both = (~has_receiver_name) & (~has_location)
    print(f'  Missing both name and location: {missing_both.sum()}')

    # Analyze outcome for corners without receiver
    # Check available columns
    outcome_cols = [col for col in df.columns if 'outcome' in col.lower()]
    print(f'\nAvailable outcome columns: {outcome_cols}')

    if outcome_cols:
        outcome_col = outcome_cols[0]
        print(f'\nOutcome distribution for corners WITHOUT receiver info:')
        no_receiver_df = df[~full_info]
        outcome_dist = no_receiver_df[outcome_col].value_counts()
        for outcome, count in outcome_dist.items():
            print(f'  {outcome:20s}: {count:4d} ({count/len(no_receiver_df)*100:5.1f}%)')

    return df, full_info


def analyze_clearance_corners(graphs):
    """Analyze clearance corners in detail."""
    print('\n' + '='*80)
    print('CLEARANCE CORNER ANALYSIS')
    print('='*80)

    clearances = [g for g in graphs if g.outcome_label == 'Clearance']
    clearances_with_receiver = [g for g in clearances if g.receiver_node_index is not None]
    clearances_without_receiver = [g for g in clearances if g.receiver_node_index is None]

    print(f'\nTotal Clearance corners: {len(clearances)}')
    print(f'  With receiver: {len(clearances_with_receiver)} ({len(clearances_with_receiver)/len(clearances)*100:.1f}%)')
    print(f'  Without receiver: {len(clearances_without_receiver)} ({len(clearances_without_receiver)/len(clearances)*100:.1f}%)')

    # Check if they have player_names available
    if clearances_without_receiver:
        sample = clearances_without_receiver[0]
        print(f'\nSample clearance without receiver:')
        print(f'  Corner ID: {sample.corner_id}')
        print(f'  Num nodes: {sample.num_nodes}')
        print(f'  Has player_ids: {hasattr(sample, "player_ids") and sample.player_ids is not None}')
        print(f'  Has teams: {hasattr(sample, "teams") and sample.teams is not None}')
        print(f'  Receiver player name: {sample.receiver_player_name}')
        print(f'  Receiver location: {sample.receiver_location}')


def check_potential_recovery():
    """Check if we can recover missing receivers."""
    print('\n' + '='*80)
    print('RECOVERY POTENTIAL ANALYSIS')
    print('='*80)

    # Load graphs
    graph_path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl")
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)

    without_receiver = [g for g in graphs if g.receiver_node_index is None]

    # Count how many have player_ids but no receiver_node_index
    has_player_ids = sum(1 for g in without_receiver if hasattr(g, 'player_ids') and g.player_ids)
    has_teams = sum(1 for g in without_receiver if hasattr(g, 'teams') and g.teams)

    print(f'\nGraphs without receiver_node_index: {len(without_receiver)}')
    print(f'  Have player_ids: {has_player_ids}')
    print(f'  Have teams: {has_teams}')

    # Check if receiver_player_name exists but receiver_node_index is None
    has_name_no_index = [g for g in without_receiver if g.receiver_player_name is not None]
    print(f'  Have receiver_player_name but no node_index: {len(has_name_no_index)}')

    if has_name_no_index:
        print('\n⚠️  FOUND POTENTIAL RECOVERY: Corners with receiver name but no node mapping!')
        print(f'  Could potentially recover {len(has_name_no_index)} graphs')
        sample = has_name_no_index[0]
        print(f'\n  Example: {sample.corner_id}')
        print(f'    Receiver name: {sample.receiver_player_name}')
        print(f'    Has player_ids: {hasattr(sample, "player_ids")}')

    # MAIN QUESTION: Can we use alternative labeling strategies?
    print('\n' + '='*80)
    print('ALTERNATIVE LABELING STRATEGIES')
    print('='*80)

    print('\nCurrent approach:')
    print('  ✓ Uses Ball Receipt location from StatsBomb events')
    print('  ✗ Only works when Ball Receipt event exists (~60% coverage)')

    print('\nPotential alternatives for Clearance corners:')
    print('  1. Label the DEFENDING player who clears (instead of attacking receiver)')
    print('  2. Use "first touch after corner" regardless of team')
    print('  3. Use "ball destination" (even if cleared)')
    print('  4. Skip receiver prediction, focus only on shot prediction')

    print('\nTrade-offs:')
    print('  Option 1-3: Increases data but changes task definition')
    print('  Option 4: Keeps task definition but reduces dataset size')


def generate_recommendations():
    """Generate recommendations for increasing coverage."""
    print('\n' + '='*80)
    print('RECOMMENDATIONS')
    print('='*80)

    print('\n1. EASIEST: Accept 60% coverage (3,492 corners)')
    print('   ✓ Cleanest task definition (attacking receiver only)')
    print('   ✓ Matches TacticAI methodology')
    print('   ✗ Loses 40% of data')

    print('\n2. MODERATE: Expand to "first touch" (any team)')
    print('   ✓ Could reach ~85-90% coverage')
    print('   ✓ Still semantically meaningful')
    print('   ⚠️ Changes task: now predicting ANY first touch, not just attacking receiver')
    print('   ⚠️ Requires re-labeling from StatsBomb events')

    print('\n3. ADVANCED: Multi-task with clearance prediction')
    print('   ✓ Uses all 5,814 corners')
    print('   ✓ Predicts: (1) outcome type, (2) receiver/clearer node')
    print('   ⚠️ More complex model architecture')
    print('   ⚠️ Requires additional labeling')

    print('\n4. HYBRID: Train on 60%, test on missing 40%')
    print('   ✓ Uses receiver labels where available')
    print('   ✓ Tests generalization to clearance scenarios')
    print('   ⚠️ Test set would be different distribution')


def main():
    """Main analysis function."""
    print('\n')
    print('╔' + '='*78 + '╗')
    print('║' + ' '*78 + '║')
    print('║' + 'INVESTIGATING RECEIVER LABEL COVERAGE'.center(78) + '║')
    print('║' + 'Goal: Recover missing ~2,300 graphs from 5,814 total'.center(78) + '║')
    print('║' + ' '*78 + '║')
    print('╚' + '='*78 + '╝')

    # Run analyses
    with_receiver, without_receiver = analyze_graph_coverage()
    df, full_info = analyze_raw_statsbomb_data()
    analyze_clearance_corners(with_receiver + without_receiver)
    check_potential_recovery()
    generate_recommendations()

    # Summary
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)
    print('\nCurrent Status:')
    print(f'  ✓ 3,492 graphs with receiver labels (60.1%)')
    print(f'  ✗ 2,322 graphs without receiver labels (39.9%)')
    print(f'\n  ⚠️  73% of missing graphs are Clearances (no Ball Receipt event)')

    print('\nRoot Cause:')
    print('  The receiver labeling depends on StatsBomb "Ball Receipt" events.')
    print('  Clearances often lack this event (ball goes out of play immediately).')

    print('\nNext Steps:')
    print('  1. Decide on labeling strategy (see recommendations above)')
    print('  2. Re-run labeling if expanding coverage')
    print('  3. Update data loader to handle new labels')

    # Save results
    results = {
        'total_graphs': len(with_receiver) + len(without_receiver),
        'with_receiver': len(with_receiver),
        'without_receiver': len(without_receiver),
        'coverage_pct': len(with_receiver) / (len(with_receiver) + len(without_receiver)) * 100,
        'clearance_pct_of_missing': 73.1,
        'recommendations': [
            'Accept 60% coverage (cleanest)',
            'Expand to first touch (any team)',
            'Multi-task with clearance prediction',
            'Hybrid approach'
        ]
    }

    output_path = Path('results/analysis/receiver_coverage_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n✓ Results saved to: {output_path}')


if __name__ == "__main__":
    main()
