# Corner Kick Tracking Data Extraction Plan

## Project Overview
Extract pitch coordinates (x, y in meters) and velocity data for ~4,000 corner kicks from SoccerNet's 500 broadcast games using the sn-gamestate pipeline. Data will be used for GNN-based corner kick outcome prediction.

---

## Phase 1: Environment Setup

### Task 1.1: Clone repositories
```bash
# Clone sn-gamestate
git clone https://github.com/SoccerNet/sn-gamestate.git

# Clone tracklab as local dependency
git clone https://github.com/TrackingLaboratory/tracklab.git
```

### Task 1.2: Create conda environment for SLURM
```bash
# Python 3.9
# PyTorch with CUDA support (match cluster CUDA version)
# Install sn-gamestate dependencies
# Install mmcv==2.0.1 via mim
```

### Task 1.3: Verify GPU access
```bash
# Test that pipeline runs on single video from GSR validation set
# Confirm output JSON contains bbox_pitch coordinates
```

---

## Phase 2: Corner Clip Extraction

### Task 2.1: Parse SoccerNet action spotting labels
```
Input: SoccerNet Labels-v2.json files from all 500 games
Output: CSV with columns [game_path, half, timestamp_seconds, corner_id]
Filter for label == "Corner"
Expected: ~4,000 corners
```

### Task 2.2: Extract video clips with ffmpeg
```bash
# For each corner: extract 10-second clip (2 sec before kick, 8 sec after)
# Input: {game_path}/{half}_720p.mkv
# Output: /path/to/corner_clips/corner_{id:04d}.mp4
# Use libx264, no audio
# Create SLURM array job for parallel extraction (optional)
```

### Task 2.3: Create corner metadata JSON
```
- Map each corner_id to original game, half, timestamp
- Include match teams, competition, season for later analysis
- Save as corner_metadata.json
```

---

## Phase 3: Format Data for GSR Pipeline

### Task 3.1: Create SoccerNetGS directory structure
```
MySoccerNetGS/
└── custom/
    ├── CORNER-0000/
    │   ├── video.mp4
    │   └── Labels-GameState.json (minimal, empty annotations)
    ├── CORNER-0001/
    │   ├── video.mp4
    │   └── Labels-GameState.json
    └── ...
```

### Task 3.2: Generate minimal Labels-GameState.json per video
```json
{
  "info": {"version": "1.3"},
  "images": [],
  "annotations": []
}
```

### Task 3.3: Create video index file
```
- List all video IDs for SLURM array indexing
- Save as video_list.txt (one ID per line)
```

---

## Phase 4: SLURM Job Configuration

### Task 4.1: Create SLURM batch script for GSR inference
```
- Job name: gsr_corners
- Array job: 0-3999 (or chunked, e.g., 0-39 with 100 videos each)
- GPU: 1x A100 (40GB or 80GB)
- Time: 10 minutes per video (buffer for safety)
- Memory: 32GB RAM
- Load conda environment
- Run tracklab with video index from SLURM_ARRAY_TASK_ID
```

### Task 4.2: Create wrapper script
```bash
# Read video ID from array index
# Set CUDA_VISIBLE_DEVICES
# Run: uv run tracklab -cn soccernet \
#     dataset_path=/path/to/MySoccerNetGS \
#     eval_set=custom \
#     nvid=${VIDEO_INDEX} \
#     evaluator=null \
#     state.save_file=outputs/states/corner_${VIDEO_ID}.pklz
# Save JSON output to outputs/json/corner_${VIDEO_ID}.json
```

### Task 4.3: Create job dependency chain (optional)
```
- Phase 2 jobs → Phase 3 jobs → Phase 4 jobs
- Use SLURM --dependency flag
```

---

## Phase 5: Post-Processing

### Task 5.1: Parse GSR JSON outputs
```
For each corner JSON file:
  - Extract all predictions per frame
  - Fields needed: image_id (frame), track_id, bbox_pitch.x_bottom_middle, bbox_pitch.y_bottom_middle
Output: Parquet file per corner with columns [frame, track_id, x, y, role, team, jersey]
```

### Task 5.2: Compute velocity from positions
```python
# Group by track_id
# Sort by frame
# Compute: vx = (x_t - x_{t-1}) / dt, vy = (y_t - y_{t-1}) / dt
# dt = 1/25 (assuming 25 fps)
# Add columns: vx, vy, speed
```

### Task 5.3: Extract corner kick moment (t=0)
```
- t=0 is approximately frame 50 (2 seconds into 10-sec clip at 25fps)
- For each corner, extract snapshot at t=0 with all player positions and velocities
- Output: corner_snapshots.parquet with columns [corner_id, track_id, x, y, vx, vy, speed, role, team]
```

### Task 5.4: Merge with corner metadata
```
- Join corner_snapshots with corner_metadata
- Add outcome labels from StatsBomb (if available) or SoccerNet events
- Final output: corners_with_tracking.parquet
```

---

## Phase 6: Validation & Quality Check

### Task 6.1: Check for failed jobs
```
- Scan SLURM output logs for errors
- List corners with missing JSON outputs
- Resubmit failed jobs
```

### Task 6.2: Validate tracking quality
```
- Check: number of tracked players per frame (expect 10-20 in penalty box area)
- Flag corners with < 5 players tracked (likely calibration failure)
- Compute statistics: mean players tracked, calibration success rate
```

### Task 6.3: Visualize sample outputs
```
- Generate minimap visualization for 10 random corners
- Verify player positions look reasonable
- Check velocity vectors make sense
```

---

## Directory Structure

```
project/
├── sn-gamestate/              # Cloned repo
├── tracklab/                  # Cloned repo (local install)
├── data/
│   ├── SoccerNet/            # Original SoccerNet data
│   ├── corner_clips/         # Extracted 10-sec clips
│   ├── MySoccerNetGS/        # Formatted for GSR pipeline
│   │   └── custom/
│   └── corner_metadata.json
├── outputs/
│   ├── states/               # Tracker state .pklz files
│   ├── json/                 # GSR JSON predictions
│   └── processed/            # Final parquet files
├── scripts/
│   ├── extract_corners.py
│   ├── format_for_gsr.py
│   ├── run_gsr.sh           # SLURM batch script
│   ├── run_single.sh        # Single video wrapper
│   ├── postprocess.py
│   └── validate.py
└── logs/
    └── slurm/               # SLURM output logs
```

---

## Key Configuration Values

```yaml
# sn-gamestate config overrides
dataset_path: /path/to/MySoccerNetGS
eval_set: custom
nvid: -1  # or specific index for array jobs
evaluator: null  # disable evaluation (no ground truth)
save_videos: false  # disable visualization to save time
state.save_file: outputs/states/corner_${ID}.pklz
```

---

## Expected Outputs

1. **~4,000 corner clips** (10 sec each, ~100GB total)
2. **~4,000 JSON files** with per-frame pitch coordinates
3. **corners_with_tracking.parquet** (~50MB) with:
   - corner_id, game_id, timestamp
   - Per-player: x, y, vx, vy, speed, role, team
   - Ready for GNN input

---

## Estimated Runtime

| Phase | Time |
|-------|------|
| Phase 1: Setup | 1 hour |
| Phase 2: Clip extraction | 2-4 hours (parallel) |
| Phase 3: Formatting | 30 min |
| Phase 4: GSR inference | 3-4 days (4 GPUs) |
| Phase 5: Post-processing | 1-2 hours |
| Phase 6: Validation | 1 hour |

**Total: ~4 days**

---

## Execution Notes

Feed this to Claude Code phase by phase. Start with Phase 1, verify it works, then proceed.
