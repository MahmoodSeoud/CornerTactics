# Task 7: Shot Label Extraction

## Goal
Create `scripts/07_extract_shot_labels.py` to extract binary shot labels for corners.

## Requirements
1. Load `corners_with_freeze_frames.json`
2. For each corner:
   - Find corner in match event sequence by UUID
   - Look ahead at next N events (window size: 5 events, following TacticAI)
   - Check if any subsequent event has type="Shot"
   - Assign binary label: 1 (Shot) or 0 (No Shot)
3. Save to `data/processed/corners_with_shot_labels.json`
4. Print class distribution and imbalance analysis

## Output Format
```json
[
  {
    "match_id": "123456",
    "event": { /* full corner event */ },
    "freeze_frame": [ /* player positions */ ],
    "shot_outcome": 1  // Binary: 1=Shot, 0=No Shot
  }
]
```

## Expected Distribution
- Shot: ~15% (10-20% range)
- No Shot: ~85%
- Class imbalance: ~5.7:1

## Validation Criteria
- Shot conversion rate should be 10-20%
- If <10%: increase lookahead window
- If >25%: decrease lookahead window

## Implementation Notes
- Window size: 5 events (as per TacticAI paper)
- Based on script 02 structure
- Need to check event type="Shot" in lookahead window
