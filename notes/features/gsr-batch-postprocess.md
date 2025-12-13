# GSR Batch Processing and Post-Processing Feature

## Overview
Implement batch GSR inference on corner clips and post-process the state files to extract player tracking data.

## Data Status (as of 2025-12-13)
- Visible corners: 4,229
- Source videos available: 4,229 (100% - all downloaded)
- Clips extracted: 2,565 (extraction in progress for remaining ~1,664)
- GSR processed: 1 (CORNER-0000 as test)

## Key Findings

### State File Format (.pklz)
- GSR outputs `.pklz` files (ZIP archives), NOT JSON files
- Archive contains:
  - `summary.json` - metadata
  - `0.pkl` - Detection DataFrame with 25 columns
  - `0_image.pkl` - Per-frame camera parameters

### Detection DataFrame Columns
- `image_id`: Frame number (float)
- `track_id`: Unique player ID across frames (float)
- `bbox_pitch`: Dict with pitch coordinates in meters:
  - `x_bottom_middle`, `y_bottom_middle` - Player position
- `role`: player / goalkeeper / referee
- `team`: left / right / nan
- `jersey_number`: Detected number

### Sample Data (CORNER-0000)
- 2,349 detections across 244 frames
- 53 unique tracks
- All rows have pitch coordinates

### Coordinate System
- Origin: Center of pitch (0, 0)
- X: -52.5m (left goal) to +52.5m (right goal)
- Y: -34m (bottom) to +34m (top)

## Bug: Existing postprocess_gsr.py reads JSON instead of .pklz

The current implementation expects JSON files but GSR outputs .pklz pickle archives.
Need to fix this to read the actual output format.

## Test Strategy
Using TDD approach:
1. Write tests for each post-processing function
2. Implement minimal code to pass tests
3. Refactor for clarity and efficiency
