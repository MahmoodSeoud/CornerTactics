# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

CornerTactics extracts **tracking data** (player x, y coordinates and velocities) from SoccerNet broadcast videos using the sn-gamestate pipeline. Data will be used for GNN-based corner kick outcome prediction.

**Goal**: Extract ~4,000 corner kicks with per-frame player pitch coordinates from 550 SoccerNet games.

## Current Status

**Phase 4 Complete**: GSR pipeline successfully tested on single video.

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Environment Setup | Done | Python 3.9 venv with all dependencies |
| 2. Corner Extraction | Done | 2,566 corner clips extracted |
| 3. Format for GSR | Done | MySoccerNetGS dataset formatted |
| 4. GSR Inference | In Progress | Single video tested successfully |
| 5. Post-Processing | Pending | Convert to parquet format |
| 6. Analysis | Pending | GNN model training |

## Development Environment

**Virtual Environment**: `/home/mseo/CornerTactics/sn-gamestate/.venv`
```bash
source /home/mseo/CornerTactics/sn-gamestate/.venv/bin/activate
```

**SLURM Modules** (required for GPU jobs):
```bash
module purge
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1
```

**Key Dependencies**:
- Python 3.9 (exact)
- PyTorch 2.5.1 + CUDA 12.1
- mmcv 2.1.0 (compiled from source with CUDA ops)
- mmdet 3.1.0
- mmocr 1.0.1
- tracklab (installed from sn-gamestate)

## Project Structure

```
CornerTactics/
├── sn-gamestate/              # SoccerNet game state reconstruction
│   ├── .venv/                 # Python virtual environment
│   ├── sn_gamestate/          # GSR modules
│   │   └── calibration/       # Camera calibration (nbjw_calib.py)
│   ├── pretrained_models/     # Model weights
│   └── outputs/               # Run outputs and visualizations
├── scripts/
│   ├── extract_soccernet_corners.py  # Phase 2.1: Parse Labels-v2.json
│   ├── extract_corner_clips.py       # Phase 2.2: Extract video clips
│   ├── create_corner_metadata.py     # Phase 2.3: Corner metadata JSON
│   ├── format_for_gsr.py             # Phase 3: Format for GSR pipeline
│   ├── test_gsr_single.sbatch        # Phase 4: Single video test
│   ├── run_gsr.sbatch                # Phase 4: Batch SLURM script
│   └── postprocess_gsr.py            # Phase 5: Post-processing
├── data/
│   ├── misc/soccernet/videos/        # SoccerNet video data (downloading)
│   ├── corner_clips/                 # Extracted 10-sec corner clips
│   ├── MySoccerNetGS/                # Formatted for GSR pipeline
│   └── processed/                    # Corner metadata JSON/CSV
├── outputs/
│   ├── states/                       # Tracker state .pklz files
│   ├── json/                         # GSR JSON predictions
│   ├── demo_corner_0000.mp4          # Visualization demo
│   └── processed/                    # Final parquet files
├── logs/
│   └── slurm/                        # SLURM job logs
├── plan.md                           # Full 6-phase implementation plan
├── CLAUDE.md                         # This file
└── README.md
```

## Running GSR Pipeline

### Single Video Test
```bash
cd /home/mseo/CornerTactics/scripts
sbatch test_gsr_single.sbatch
```

### Check Job Status
```bash
squeue -u $USER
tail -f logs/slurm/gsr_test_*.out
```

### View Results
```bash
# Visualization video
ls -la /home/mseo/CornerTactics/outputs/demo_corner_0000.mp4

# Tracking state file
ls -la /home/mseo/CornerTactics/outputs/states/CORNER-0000.pklz
```

## GSR Output Format

The `.pklz` state files are ZIP archives containing:
- `summary.json` - Video metadata
- `0.pkl` - Detection DataFrame with columns:
  - `track_id` - Unique player ID across frames
  - `image_id` - Frame number
  - `bbox_ltwh` - Bounding box (left, top, width, height)
  - `team` - Team assignment (left/right)
  - `role` - Player role (player/goalkeeper/referee)
  - `jersey_number` - Detected jersey number
  - `bbox_pitch` - Pitch coordinates in meters:
    - `x_bottom_middle`, `y_bottom_middle` - Player position
- `0_image.pkl` - Per-frame camera parameters

### Loading State Files
```python
import zipfile
import pickle

with zipfile.ZipFile('outputs/states/CORNER-0000.pklz', 'r') as zf:
    with zf.open('0.pkl') as f:
        detections = pickle.load(f)

# Sample pitch coordinates
for _, row in detections[detections['bbox_pitch'].notna()].head().iterrows():
    x = row['bbox_pitch']['x_bottom_middle']
    y = row['bbox_pitch']['y_bottom_middle']
    print(f"Track {row['track_id']}: ({x:.1f}m, {y:.1f}m)")
```

## Dataset Statistics

- **SoccerNet games**: 550 (6 leagues, 1.1TB total)
  - Spain La Liga: 125
  - UEFA Champions League: 108
  - Italy Serie A: 105
  - England EPL: 104
  - Germany Bundesliga: 61
  - France Ligue 1: 47
- **Total corners**: 4,836
- **Visible corners**: 4,229
- **Extracted clips**: 2,566 (from available videos)
- **Video format**: 720p MP4, 10-sec clips

### GSR Test Results (CORNER-0000)
- **Detections**: 2,349
- **Frames**: 244
- **Unique tracks**: 53
- **State file size**: 3.2MB

## Bug Fixes Applied

### nbjw_calib.py KeyError Fix
**File**: `sn-gamestate/sn_gamestate/calibration/nbjw_calib.py:179`

**Problem**: Accessing DataFrame by label `[0]` instead of position `.iloc[0]`

**Fix**:
```python
# Before (broken):
predictions = metadatas["keypoints"][0]

# After (fixed):
predictions = metadatas["keypoints"].iloc[0]
```

### mmdet Version Compatibility
**File**: `.venv/lib/python3.9/site-packages/mmdet/__init__.py`

**Fix**: Changed `mmcv_maximum_version = '2.2.0'` (was '2.1.0')

## Code Philosophy

- Straightforward, data-oriented code
- Efficient batch processing with SLURM
- Clear variable names
- Think like John Carmack: fix problems, don't work around them

## Important Notes

1. **Virtual Environment**: Use sn-gamestate/.venv (Python 3.9)
2. **SLURM Modules**: Load GCC/12.3.0 and CUDA/12.1.1 before running
3. **Data Directory**: All data in `data/` (gitignored, ~100GB when complete)
4. **Pitch Coordinates**: Output is in meters (origin at center, 105m x 68m pitch)
5. **mmcv**: Must be compiled from source for CUDA ops to work

## Git Workflow

- Never commit data files (gitignored)
- Keep commit messages concise
- Main plan reference: `plan.md`
