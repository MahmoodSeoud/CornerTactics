#!/usr/bin/env python3
"""
Debug receiver labeling v2 to find why coverage decreased.
"""

import pickle
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load graphs
print("Loading graphs...")
with open("data/graphs/adjacency_team/statsbomb_temporal_augmented.pkl", 'rb') as f:
    graphs = pickle.load(f)

print(f"Total graphs: {len(graphs)}")

# Load corners CSV
print("\nLoading corners CSV...")
corners_df = pd.read_csv("data/raw/statsbomb/corners_360.csv")
print(f"Total corners in CSV: {len(corners_df)}")

# Check corner_id formats
print("\n=== Sample Corner IDs ===")
print("From graphs (first 10):")
for i in range(min(10, len(graphs))):
    print(f"  {graphs[i].corner_id}")

print("\nFrom CSV (first 10):")
print(corners_df['corner_id'].head(10).tolist())

# Check if base corner_id extraction works
print("\n=== Testing Base Corner ID Extraction ===")
test_ids = [
    "c88df499-4e73-4ed0-bf2d-d224cd82b3f3",  # Original
    "c88df499-4e73-4ed0-bf2d-d224cd82b3f3_t0",  # Temporal
    "c88df499-4e73-4ed0-bf2d-d224cd82b3f3_t1_mirror",  # Temporal + mirror
]

for corner_id in test_ids:
    # Extract base corner_id (remove temporal/mirror suffixes)
    base_corner_id = corner_id.split('_t')[0].split('_mirror')[0]

    # Check if it exists in CSV
    match_row = corners_df[corners_df['corner_id'] == base_corner_id]

    print(f"\nCorner ID: {corner_id}")
    print(f"  Base ID: {base_corner_id}")
    print(f"  Found in CSV: {len(match_row) > 0}")
    if len(match_row) > 0:
        print(f"  Match ID: {match_row.iloc[0]['match_id']}")

# Count how many graph corner_ids have matches in CSV
print("\n=== Checking All Graph Corner IDs ===")
matched_count = 0
unmatched_examples = []

for graph in graphs[:100]:  # Check first 100
    base_corner_id = graph.corner_id.split('_t')[0].split('_mirror')[0]
    match_row = corners_df[corners_df['corner_id'] == base_corner_id]

    if len(match_row) > 0:
        matched_count += 1
    else:
        unmatched_examples.append((graph.corner_id, base_corner_id))

print(f"Matched: {matched_count}/100")
print(f"Unmatched: {len(unmatched_examples)}/100")

if unmatched_examples:
    print("\nUnmatched examples (first 5):")
    for orig_id, base_id in unmatched_examples[:5]:
        print(f"  Original: {orig_id}")
        print(f"  Base: {base_id}")

# Check if corner_id column actually exists and what it contains
print("\n=== CSV Column Check ===")
print(f"Columns in CSV: {corners_df.columns.tolist()}")
print(f"Corner ID dtype: {corners_df['corner_id'].dtype}")
print(f"Sample corner IDs from CSV:")
print(corners_df['corner_id'].head(20))
