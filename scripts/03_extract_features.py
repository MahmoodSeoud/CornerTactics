#!/usr/bin/env python3
"""
Task 3: Extract Features from Corner Freeze Frames

Reads corners_with_labels.json and extracts 49 features for each corner,
saving the result as a CSV file.

Input:  data/processed/corners_with_labels.json
Output: data/processed/corners_with_features.csv

Features (49 total):
- Basic metadata (5)
- Temporal (3)
- Player counts (6)
- Spatial density (4)
- Positional (8)
- Pass trajectory (4)
- Pass technique & body part (5)
- Pass outcome & context (4)
- Goalkeeper (3)
- Score state (4)
- Substitution patterns (3)
"""

import json
import pandas as pd
from pathlib import Path
from extract_features import extract_all_features


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data/processed/corners_with_labels.json'
    output_file = base_dir / 'data/processed/corners_with_features.csv'

    print(f"Loading labeled corners from {input_file}...")

    # Load corners with labels
    with open(input_file, 'r') as f:
        corners = json.load(f)

    print(f"Loaded {len(corners)} corners")
    print(f"\nExtracting features...")
    print("Note: Match context features (score state, substitutions) require loading match event files.")
    print("      This may take some time...")

    # Extract features for each corner
    rows = []
    for i, corner in enumerate(corners, 1):
        if i % 100 == 0:
            print(f"Processing corner {i}/{len(corners)}...")

        # Extract all 49 features
        features = extract_all_features(corner)

        # Add metadata
        row = {
            'match_id': corner['match_id'],
            'event_id': corner['event']['id'],
            'outcome': corner['outcome']
        }

        # Add all features
        row.update(features)

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    print(f"\nSaving features to {output_file}...")
    df.to_csv(output_file, index=False)

    print(f"\n=== Feature Extraction Summary ===")
    print(f"Total corners processed: {len(df)}")
    print(f"Total features per corner: {len(df.columns) - 3}")  # Exclude match_id, event_id, outcome
    print(f"Output shape: {df.shape}")
    print(f"\nColumn names:")
    print(df.columns.tolist())

    print(f"\n✓ Task 3 complete!")
    print(f"Output saved to: {output_file}")


if __name__ == '__main__':
    main()
