# GSR Batch Processing and Post-Processing Feature

## Overview
Implement batch GSR inference on corner clips and post-process the state files to extract player tracking data.

## Status: COMPLETE

All implementation tasks completed:
1. Extracted all 4,229 corner clips
2. Fixed postprocess_gsr.py to read .pklz files
3. Updated batch SLURM script for HPC execution
4. Added validation script for quality checks
5. Created comprehensive test suite (16 tests)

## Data Status (as of 2025-12-13)
- Visible corners: 4,229
- Source videos available: 4,229 (100% - all downloaded)
- Clips extracted: 4,229 (100% complete)
- GSR processed: 1 (CORNER-0000 as test)
- Remaining: ~4,228 clips to process via SLURM

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
- 11 players at t=0 (frame 50)
- All rows have pitch coordinates

### Coordinate System
- Origin: Center of pitch (0, 0)
- X: -52.5m (left goal) to +52.5m (right goal)
- Y: -34m (bottom) to +34m (top)

## Scripts Created/Updated

### postprocess_gsr.py
- `parse_state_file()`: Read .pklz files
- `compute_velocities()`: Calculate vx, vy, speed
- `extract_snapshot()`: Get player positions at t=0
- `process_all_corners()`: Batch process all state files

### run_gsr.sbatch
- Fixed to use venv (not conda)
- Load correct modules (GCC/12.3.0, CUDA/12.1.1)
- Process clips directly with tracklab
- Skip already processed clips
- Array job for parallel processing

### validate_gsr_outputs.py
- Check for missing state files
- Validate tracking quality
- Generate CSV validation report

## Test Coverage (16 tests)
- SLURM script configuration
- State file parsing
- Velocity computation
- Snapshot extraction
- Pipeline integration
- Validation quality checks

## Next Steps (for user)
1. Submit batch SLURM job: `sbatch scripts/run_gsr.sbatch`
2. Monitor progress: `squeue -u $USER`
3. Run post-processing: `python scripts/postprocess_gsr.py`
4. Validate outputs: `python scripts/validate_gsr_outputs.py`
