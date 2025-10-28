#!/usr/bin/env python3
"""
Inspect actual column structure of StatsBomb corner events.
"""

from statsbombpy import sb
import pandas as pd

match_id = 3795506

events_df = sb.events(match_id=match_id, fmt='dataframe')

# Filter for corners
corner_events = events_df[
    (events_df['type'] == 'Pass') &
    (events_df['pass_type'].fillna('').str.contains('Corner', case=False, na=False))
].copy()

print(f"Found {len(corner_events)} corner kicks")
print(f"\nAll columns in corner events:")
print(list(corner_events.columns))

# Show first corner
if len(corner_events) > 0:
    print(f"\nFirst corner event:")
    first_corner = corner_events.iloc[0]

    # Show all pass-related columns
    pass_cols = [col for col in first_corner.index if 'pass' in col.lower() or 'recipient' in col.lower()]
    print(f"\nPass-related columns:")
    for col in pass_cols:
        print(f"  {col}: {first_corner[col]}")
