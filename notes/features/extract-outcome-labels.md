# Task 2: Extract Outcome Labels

## Objective
Create `scripts/02_extract_outcome_labels.py` to label corner kick outcomes based on the immediate next event.

## Requirements
1. Load corners from Task 1 output (`data/processed/corners_with_freeze_frames.json`)
2. For each corner:
   - Load full match events from `data/statsbomb/events/<match_id>.json`
   - Find corner in event sequence by UUID
   - Get IMMEDIATE next event (index + 1)
   - Map event type to 4 classes
3. Add `"outcome"` field to each corner
4. Save to `data/processed/corners_with_labels.json`

## Outcome Mapping
```python
OUTCOME_MAPPING = {
    "Ball Receipt*": "Ball Receipt",
    "Clearance": "Clearance",
    "Goal Keeper": "Goalkeeper",
    # Everything else → "Other"
    "Duel": "Other",
    "Pressure": "Other",
    "Pass": "Other",
    "Foul Committed": "Other",
    "Ball Recovery": "Other",
    "Block": "Other",
    "Interception": "Other",
    "Dispossessed": "Other",
    "Shot": "Other"
}
```

## Expected Distribution
- Ball Receipt: ~54%
- Clearance: ~23%
- Goalkeeper: ~10%
- Other: ~12%

## TDD Progress
- [x] Test: Load corners_with_freeze_frames.json
- [x] Test: Load match events file
- [x] Test: Find corner event by UUID in event sequence
- [x] Test: Get next event after corner
- [x] Test: Map event type to outcome class
- [x] Test: Handle edge cases (last event in match, missing next event)
- [x] Test: Validate class distribution
- [x] Test: Save output file

## Implementation Complete

**Output**: `data/processed/corners_with_labels.json` (9.2MB, 1,933 samples)

**Actual Distribution**:
- Ball Receipt: 1,050 (54.3%)
- Clearance: 453 (23.4%)
- Goalkeeper: 196 (10.1%)
- Other: 234 (12.1%)

**Tests**: 14 tests passing

**Script**: `scripts/02_extract_outcome_labels.py` (working with TDD approach)
