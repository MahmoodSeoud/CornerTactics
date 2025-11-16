# Task 3: Feature Engineering

## Goal
Extract 27 features from each corner's freeze frame data and save to CSV format.

## Feature Categories
1. **Basic corner metadata** (5 features): side, period, minute, x, y
2. **Player count features** (6 features): attacking/defending counts in different zones
3. **Spatial density features** (4 features): density metrics and numerical advantage
4. **Positional features** (8 features): centroids, compactness, distances
5. **Pass trajectory features** (4 features): end location, length, height

Total: 27 features

## Input
- `data/processed/corners_with_labels.json` (1,933 corners with freeze frames and labels)

## Output
- `data/processed/corners_with_features.csv` (1,933 rows × ~30 columns: match_id, event_id, outcome, 27 features)

## StatsBomb Coordinate System
- Pitch dimensions: 120 x 80
- Goal center: x=120, y=40
- Penalty box: x > 102, 18 < y < 62
- Near goal area: x > 108, 30 < y < 50

## Implementation Notes
- Freeze frame contains player positions with 'teammate' boolean field
- Corner location in event['location']
- Pass end location in event['pass']['end_location']
- Pass height in event['pass']['height']['name']
