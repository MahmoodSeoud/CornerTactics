# Corner Kick Tracking Data Extraction Plan

## Project Overview
Extract pitch coordinates (x, y in meters) and velocity data for ~4,000 corner kicks from SoccerNet's 550 broadcast games using the sn-gamestate pipeline. Data will be used for GNN-based corner kick outcome prediction.

---

## Current Status Summary

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Environment Setup | **DONE** | 100% |
| Phase 2: Corner Clip Extraction | **DONE** | 2,566 clips extracted |
| Phase 3: Format for GSR | **DONE** | MySoccerNetGS ready |
| Phase 4: GSR Inference | **IN PROGRESS** | 1/2,566 tested |
| Phase 5: Post-Processing | Pending | - |
| Phase 6: Validation | Pending | - |

**Last Updated**: 2025-12-13

---

## Phase 1: Environment Setup [DONE]

### Task 1.1: Clone repositories [DONE]
```bash
# Clone sn-gamestate
git clone https://github.com/SoccerNet/sn-gamestate.git

# Clone tracklab as local dependency
git clone https://github.com/TrackingLaboratory/tracklab.git
```

### Task 1.2: Create virtual environment [DONE]
```bash
# Using venv (not conda) with Python 3.9
cd sn-gamestate
python3.9 -m venv .venv
source .venv/bin/activate

# Install sn-gamestate and tracklab
pip install -e .
pip install -e ../tracklab

# Install mmcv with CUDA ops (compiled from source)
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1
MMCV_WITH_OPS=1 pip install mmcv==2.1.0 --no-binary :all:

# Install mmdet and mmocr
pip install mmdet==3.1.0 mmocr==1.0.1
```

**Environment Details**:
- Python 3.9
- PyTorch 2.5.1 + CUDA 12.1
- mmcv 2.1.0 (compiled with CUDA ops)
- mmdet 3.1.0
- mmocr 1.0.1
- Virtual env: `/home/mseo/CornerTactics/sn-gamestate/.venv`

### Task 1.3: Verify GPU access [DONE]
```bash
# Successfully tested on CORNER-0000
# Output: 2,349 detections, 53 tracks, 244 frames
# State file: outputs/states/CORNER-0000.pklz (3.2MB)
# Visualization: outputs/demo_corner_0000.mp4 (11MB)
```

---

## Phase 2: Corner Clip Extraction [DONE]

### Task 2.1: Parse SoccerNet action spotting labels [DONE]
```
Script: scripts/extract_soccernet_corners.py
Input: SoccerNet Labels-v2.json files from all 550 games
Output: data/processed/all_corners.csv

Results:
- Total corners found: 4,836
- Visible corners: 4,229
```

### Task 2.2: Extract video clips with ffmpeg [DONE]
```bash
# Script: scripts/extract_corner_clips.py --visible-only
# Input: SoccerNet MKV videos
# Output: data/corner_clips/corner_{id:04d}.mp4

Results:
- Extracted clips: 2,566 (from available videos)
- Format: 720p MP4, 10 seconds each
- Note: Some source videos still downloading
```

### Task 2.3: Create corner metadata JSON [DONE]
```
Script: scripts/create_corner_metadata.py
Output: data/processed/corner_metadata.json

Contents: game path, half, timestamp, teams, competition, season
```

---

## Phase 3: Format Data for GSR Pipeline [DONE]

### Task 3.1: Create SoccerNetGS directory structure [DONE]
```
Script: scripts/format_for_gsr.py
Output: data/MySoccerNetGS/

MySoccerNetGS/
└── custom/
    ├── CORNER-0000/
    │   ├── video.mp4 -> symlink to corner_clips/corner_0000.mp4
    │   └── Labels-GameState.json
    ├── CORNER-0001/
    │   └── ...
    └── ... (2,566 directories)
```

### Task 3.2: Generate minimal Labels-GameState.json [DONE]
```json
{
  "info": {"version": "1.3"},
  "images": [],
  "annotations": []
}
```

### Task 3.3: Create video index file [DONE]
```
Output: data/MySoccerNetGS/video_list.txt
Contents: One video ID per line (CORNER-0000 to CORNER-2565)
```

---

## Phase 4: SLURM Job Configuration [IN PROGRESS]

### Task 4.1: Create SLURM batch script [DONE]
```bash
# Single video test script: scripts/test_gsr_single.sbatch
# Batch processing script: scripts/run_gsr.sbatch (pending)

#SBATCH --partition=acltr
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
```

### Task 4.2: Test single video [DONE]
```bash
# Command:
tracklab -cn soccernet \
    dataset=video \
    dataset.video_path=/path/to/corner_0000.mp4 \
    dataset.eval_set=val \
    eval_tracking=false \
    state.save_file=outputs/states/CORNER-0000.pklz

# Results:
# - Runtime: ~6 minutes
# - Detections: 2,349
# - Unique tracks: 53
# - Frames: 244
# - Output size: 3.2MB
```

### Task 4.3: Bug fixes applied [DONE]
1. **nbjw_calib.py KeyError fix** (`sn_gamestate/calibration/nbjw_calib.py:179`):
   ```python
   # Changed from:
   predictions = metadatas["keypoints"][0]
   # To:
   predictions = metadatas["keypoints"].iloc[0]
   ```

2. **mmdet version compatibility** (`.venv/lib/python3.9/site-packages/mmdet/__init__.py`):
   ```python
   mmcv_maximum_version = '2.2.0'  # was '2.1.0'
   ```

### Task 4.4: Batch processing [PENDING]
```
TODO:
- Create SLURM array job for all 2,566 clips
- Estimate: ~6 min/clip × 2,566 clips = 256 hours
- With 4 GPUs: ~64 hours (2.7 days)
```

---

## Phase 5: Post-Processing [PENDING]

### Task 5.1: Parse GSR state files
```
Script: scripts/postprocess_gsr.py (to be created)
Input: outputs/states/*.pklz
Output: outputs/processed/corners_tracking.parquet

Columns needed:
- corner_id, frame, track_id
- x, y (pitch coordinates in meters)
- role, team, jersey_number
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
- Output: corner_snapshots.parquet
```

### Task 5.4: Merge with corner metadata
```
- Join corner_snapshots with corner_metadata
- Add outcome labels from SoccerNet events
- Final output: corners_with_tracking.parquet
```

---

## Phase 6: Validation & Quality Check [PENDING]

### Task 6.1: Check for failed jobs
```
- Scan SLURM output logs for errors
- List corners with missing state files
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

## Directory Structure (Current)

```
CornerTactics/
├── sn-gamestate/              # Cloned repo with .venv
│   ├── .venv/                 # Python virtual environment
│   ├── sn_gamestate/          # GSR modules (nbjw_calib.py fixed)
│   └── pretrained_models/     # Model weights
├── tracklab/                  # Cloned repo (local install)
├── data/
│   ├── misc/soccernet/videos/ # Original SoccerNet data (downloading)
│   ├── corner_clips/          # 2,566 extracted 10-sec clips
│   ├── MySoccerNetGS/         # Formatted for GSR pipeline
│   │   └── custom/            # 2,566 video directories
│   └── processed/
│       ├── all_corners.csv
│       └── corner_metadata.json
├── outputs/
│   ├── states/                # Tracker state .pklz files
│   │   └── CORNER-0000.pklz   # Test output (3.2MB)
│   ├── json/                  # GSR JSON predictions
│   ├── demo_corner_0000.mp4   # Visualization (11MB)
│   └── processed/             # Final parquet files (pending)
├── scripts/
│   ├── extract_soccernet_corners.py  # Phase 2.1
│   ├── extract_corner_clips.py       # Phase 2.2
│   ├── create_corner_metadata.py     # Phase 2.3
│   ├── format_for_gsr.py             # Phase 3
│   ├── test_gsr_single.sbatch        # Phase 4 (single test)
│   ├── run_gsr.sbatch                # Phase 4 (batch - pending)
│   └── postprocess_gsr.py            # Phase 5 (pending)
├── logs/
│   └── slurm/                 # SLURM output logs
├── plan.md                    # This file
├── CLAUDE.md                  # Claude Code instructions
└── README.md                  # Project documentation
```

---

## Key Configuration Values

```yaml
# tracklab command (working)
tracklab -cn soccernet \
    dataset=video \
    dataset.video_path=/path/to/video.mp4 \
    dataset.eval_set=val \
    eval_tracking=false \
    state.save_file=outputs/states/CORNER-XXXX.pklz
```

**SLURM Modules Required**:
```bash
module purge
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1
```

---

## GSR Output Format

State files (`.pklz`) are ZIP archives containing:
- `summary.json` - Video metadata
- `0.pkl` - Detection DataFrame
- `0_image.pkl` - Camera parameters per frame

**Detection columns**:
| Column | Description |
|--------|-------------|
| track_id | Unique player ID across frames |
| image_id | Frame number |
| team | left / right |
| role | player / goalkeeper / referee |
| jersey_number | Detected number |
| bbox_pitch | {x_bottom_middle, y_bottom_middle, ...} in meters |

**Coordinate system**:
- Origin: Center of pitch (0, 0)
- X: -52.5m (left goal) to +52.5m (right goal)
- Y: -34m (bottom) to +34m (top)

---

## Expected Outputs

1. **2,566 corner clips** (10 sec each) - DONE
2. **2,566 state files** with per-frame pitch coordinates - IN PROGRESS
3. **corners_with_tracking.parquet** (~50MB) with:
   - corner_id, game_id, timestamp
   - Per-player: x, y, vx, vy, speed, role, team
   - Ready for GNN input

---

## Estimated Runtime

| Phase | Estimated | Actual |
|-------|-----------|--------|
| Phase 1: Setup | 1 hour | ~4 hours (dependency issues) |
| Phase 2: Clip extraction | 2-4 hours | Done |
| Phase 3: Formatting | 30 min | Done |
| Phase 4: GSR inference | 3-4 days (4 GPUs) | ~6 min/clip tested |
| Phase 5: Post-processing | 1-2 hours | Pending |
| Phase 6: Validation | 1 hour | Pending |

**Remaining**: ~3 days for batch GSR + post-processing

---

## Next Steps

1. **Create batch SLURM script** for processing all 2,566 clips
2. **Submit array job** and monitor progress
3. **Implement post-processing** to extract tracking data to parquet
4. **Validate** tracking quality across all corners
