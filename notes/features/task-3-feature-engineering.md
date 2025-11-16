# Task 3: Feature Engineering

## Status: ✅ COMPLETED

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
- `data/processed/corners_with_features.csv` (1,933 rows × 30 columns: match_id, event_id, outcome, 27 features)

## Implementation Details

### Files Created
- `scripts/extract_features.py` - Feature extraction functions (5 functions, one per category)
- `scripts/03_extract_features.py` - Main script to process all corners
- `tests/test_extract_features.py` - Comprehensive test suite (19 tests)

### Test Coverage
All 19 tests passing:
- Basic metadata: 3 tests
- Player counts: 3 tests
- Spatial density: 4 tests
- Positional features: 4 tests
- Pass trajectory: 3 tests
- Integration: 2 tests

### TDD Process Followed
1. RED: Wrote comprehensive tests first (all failing due to missing module)
2. GREEN: Implemented simplest code to make tests pass
3. REFACTOR: Code is clean and well-documented (minimal refactoring needed)

### Feature Statistics
- Total corners: 1,933
- Outcome distribution: Ball Receipt (54.3%), Clearance (23.4%), Goalkeeper (10.1%), Other (12.1%)
- All features are numeric (int or float)
- No missing values

### Commits
- b8edc07: Implement Task 3: Extract 27 features from corners
- 344a661: Update PLAN.md: Mark Task 3 as complete

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
- Handled edge cases: division by zero in ratio calculation, empty player lists
