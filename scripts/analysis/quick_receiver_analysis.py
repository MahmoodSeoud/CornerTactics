#!/usr/bin/env python3
"""Quick receiver coverage analysis without pandas dependency"""

import pickle
import json
from pathlib import Path
from collections import Counter

# Load graphs
graph_path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl")
with open(graph_path, 'rb') as f:
    graphs = pickle.load(f)

print('='*80)
print('RECEIVER LABEL COVERAGE ANALYSIS')
print('='*80)
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

# Key finding
clearance_pct = outcomes_without.get('Clearance', 0) / len(without_receiver) * 100
print(f'\n⚠️  KEY FINDING: {clearance_pct:.1f}% of graphs without receivers are Clearances')

# Check unique base corners
print('\n' + '='*80)
print('BASE CORNER ANALYSIS')
print('='*80)

base_corners = set()
base_corners_with_receiver = set()

for g in graphs:
    base_id = g.corner_id.split('_t')[0].split('_mirror')[0]
    base_corners.add(base_id)
    if g.receiver_node_index is not None:
        base_corners_with_receiver.add(base_id)

print(f'\nUnique base corners: {len(base_corners)}')
print(f'Base corners with at least one receiver: {len(base_corners_with_receiver)}')
print(f'Coverage: {len(base_corners_with_receiver)/len(base_corners)*100:.1f}%')

# Recommendations
print('\n' + '='*80)
print('RECOMMENDATIONS')
print('='*80)

print('\n1. CURRENT APPROACH: Accept 60% coverage (3,492 corners)')
print('   ✓ Cleanest task definition (attacking receiver only)')
print('   ✓ Matches TacticAI methodology')
print('   ✗ Loses 40% of data (mostly clearances)')

print('\n2. ALTERNATIVE: Expand to "first touch" (any team)')
print('   ✓ Could reach ~85-90% coverage')
print('   ✓ Still semantically meaningful')
print('   ⚠️  Changes task: predicting ANY first touch, not just attacking receiver')
print('   ⚠️  Requires re-labeling from StatsBomb events')

print('\n3. HYBRID: Multi-task with clearance prediction')
print('   ✓ Uses all 5,814 corners')
print('   ✓ Predicts: (1) outcome type, (2) receiver/clearer node')
print('   ⚠️  More complex model architecture')

# Save results
results = {
    'total_graphs': len(graphs),
    'with_receiver': len(with_receiver),
    'without_receiver': len(without_receiver),
    'coverage_pct': len(with_receiver) / len(graphs) * 100,
    'clearance_pct_of_missing': clearance_pct,
    'unique_base_corners': len(base_corners),
    'base_corners_with_receiver': len(base_corners_with_receiver),
    'outcome_distribution_with': dict(outcomes_with),
    'outcome_distribution_without': dict(outcomes_without)
}

output_path = Path('results/analysis/receiver_coverage_analysis.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✓ Results saved to: {output_path}')
