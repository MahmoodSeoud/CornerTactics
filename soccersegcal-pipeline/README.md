# SoccerSegCal Pipeline

Alternative pipeline for extracting player pitch positions from corner kick videos using Spiideo's soccersegcal for camera calibration.

## Pipeline Overview

```
Corner clips → Extract frames → Camera calibration → Player detection → Pitch projection → Dataset
```

1. **Load Corners** (`01_load_corners.py`) - Index corner metadata and video clips
2. **Extract Frames** (`02_extract_frames.py`) - Extract frames at 0ms, 2000ms, 5000ms offsets
3. **Camera Calibration** (`03_calibrate_cameras.py`) - Use soccersegcal to get homography matrices
4. **Player Detection** (`04_detect_players.py`) - YOLOv8 to detect players and get foot positions
5. **Pitch Projection** (`05_project_to_pitch.py`) - Project image coords to pitch coords (meters)
6. **Create Dataset** (`06_create_dataset.py`) - Combine with outcome labels for training

## Setup

```bash
# Run setup script (creates venv, installs dependencies, downloads model)
cd /home/mseo/CornerTactics/soccersegcal-pipeline
bash scripts/setup_environment.sh
```

## Quick Test

Run on 10 corners locally to verify everything works:

```bash
source venv/bin/activate
python scripts/quick_test.py
```

Or submit as SLURM job:

```bash
sbatch scripts/quick_test.sbatch
```

## Full Run

Submit full pipeline to SLURM:

```bash
sbatch scripts/run_pipeline.sbatch
```

## Output

The pipeline produces:
- `data/corner_index.csv` - Index of corners with clip paths
- `data/frames/` - Extracted frames (JPEG)
- `data/frames_index.csv` - Index of extracted frames
- `data/camera_calibrations.json` - Homography matrices per frame
- `data/player_detections.json` - YOLOv8 bounding boxes
- `data/corner_positions.json` - Player pitch coordinates
- `data/corner_dataset.json` - Final dataset with outcome labels
- `data/corner_dataset.parquet` - Parquet format for efficient loading

## Comparison with sn-gamestate

| Feature | sn-gamestate | soccersegcal |
|---------|-------------|--------------|
| Camera calibration | nbjw_calib | soccersegcal |
| Player detection | Built-in tracker | YOLOv8 |
| Tracking | Yes (track_id) | No |
| Team assignment | Yes | No |
| Jersey detection | Yes | No |
| Processing speed | Slower | Faster |
| Dependencies | Complex | Simpler |

Use sn-gamestate if you need tracking/team info. Use soccersegcal for simpler position extraction.

## Directory Structure

```
soccersegcal-pipeline/
├── scripts/
│   ├── setup_environment.sh   # Environment setup
│   ├── 01_load_corners.py     # Load corner metadata
│   ├── 02_extract_frames.py   # Extract frames from clips
│   ├── 03_calibrate_cameras.py # Camera calibration
│   ├── 04_detect_players.py   # YOLOv8 detection
│   ├── 05_project_to_pitch.py # Pitch projection
│   ├── 06_create_dataset.py   # Create training dataset
│   ├── quick_test.py          # Local quick test
│   ├── quick_test.sbatch      # SLURM quick test
│   └── run_pipeline.sbatch    # SLURM full run
├── venv/                      # Virtual environment
├── soccersegcal/              # Cloned repo
├── sskit/                     # Cloned repo
├── models/                    # Pretrained models
├── data/                      # Pipeline outputs
└── outputs/                   # Visualizations
```
