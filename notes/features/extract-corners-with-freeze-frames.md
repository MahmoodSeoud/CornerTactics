# Feature: Extract Corners with Freeze Frames

## Goal
Implement TASK 1 from PLAN.md: Create a data extraction pipeline that matches corner kick events with their corresponding freeze frame data.

## Requirements
1. Load all match event JSONs from `data/statsbomb/events/events/`
2. For each match:
   - Find corner kick events (type="Pass" AND pass.type.name="Corner")
   - Load corresponding freeze frame data from `data/statsbomb/freeze-frames/<match_id>.json`
   - Match corner event UUIDs with freeze frame event_uuids
3. Save corners that have freeze frames to `data/processed/corners_with_freeze_frames.json`

## Data Structure Insights

### Corner Event Structure
- Located in: `data/statsbomb/events/events/<match_id>.json`
- Event ID field: `"id"` (UUID)
- Identification: `event['type']['name'] == 'Pass'` AND `event['pass']['type']['name'] == 'Corner'`
- Key fields:
  - `id`: Event UUID
  - `location`: [x, y] coordinates
  - `pass.end_location`: [x, y] where pass ended
  - `pass.height`: Ground/Low/High
  - All other event metadata

### Freeze Frame Structure
- Located in: `data/statsbomb/freeze-frames/<match_id>.json`
- File format: Array of freeze frame objects
- Key fields:
  - `event_uuid`: UUID matching corner event `id`
  - `freeze_frame`: Array of player positions
    - Each player: `{teammate: bool, actor: bool, keeper: bool, location: [x, y]}`
  - `visible_area`: Polygon defining visible area

## Expected Output
- File: `data/processed/corners_with_freeze_frames.json`
- Format: Array of objects with:
  ```json
  {
    "match_id": "123456",
    "event": { /* full corner event object */ },
    "freeze_frame": [ /* array of player positions */ ]
  }
  ```
- Expected count: ~1,933 corners with freeze frames

## Dataset Statistics
- Total event files: 3,464 matches
- Total freeze frame files: 323 matches
- Sample match (15946): 8 corners in 3,762 events
- Sample freeze frame file: ~3,370 freeze frames per match

## Implementation Plan
1. Create test file: `tests/test_extract_corners.py`
2. Create script: `scripts/01_extract_corners_with_freeze_frames.py`
3. Follow TDD approach:
   - Red: Write failing test
   - Green: Implement minimal working code
   - Refactor: Optimize and improve

## Notes
- Not all corners will have freeze frames (only ~5.7% based on docs)
- Need to handle missing freeze frame files gracefully
- Match freeze frames by UUID: `freeze_frame['event_uuid'] == corner_event['id']`
