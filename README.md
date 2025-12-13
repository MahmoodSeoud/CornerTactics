# CornerTactics

Extract player tracking data from soccer broadcast videos for corner kick analysis using Graph Neural Networks.

## Overview

CornerTactics extracts per-frame player positions (x, y coordinates in meters) from SoccerNet broadcast videos using the [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) pipeline. The extracted tracking data will be used for GNN-based corner kick outcome prediction.

**Pipeline**: Video -> Object Detection -> Re-ID -> Tracking -> Camera Calibration -> Pitch Projection

## Features

- Extract corner kick clips from SoccerNet game videos
- Run Game State Reconstruction (GSR) pipeline for player tracking
- Project player bounding boxes to pitch coordinates (meters)
- Team assignment and jersey number detection
- SLURM-based batch processing for HPC clusters

## Quick Start

### Prerequisites

- Python 3.9
- CUDA 12.1
- Access to SoccerNet dataset
- SLURM cluster (for batch processing)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd CornerTactics

# Create virtual environment
cd sn-gamestate
python3.9 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install -e ../tracklab

# Install mmcv with CUDA ops (requires compilation)
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1
MMCV_WITH_OPS=1 pip install mmcv==2.1.0 --no-binary :all:

# Install remaining dependencies
pip install mmdet==3.1.0 mmocr==1.0.1
```

### Run GSR on a Single Video

```bash
cd scripts
sbatch test_gsr_single.sbatch
```

### Check Results

```bash
# View tracking state
ls -la outputs/states/CORNER-0000.pklz

# View visualization video
ls -la outputs/demo_corner_0000.mp4
```

## Project Structure

```
CornerTactics/
├── sn-gamestate/           # Game State Reconstruction pipeline
│   ├── .venv/              # Python virtual environment
│   └── sn_gamestate/       # GSR modules
├── tracklab/               # Tracking framework
├── scripts/                # Processing scripts
│   ├── extract_corner_clips.py
│   ├── test_gsr_single.sbatch
│   └── run_gsr.sbatch
├── data/
│   ├── corner_clips/       # Extracted 10-sec clips
│   └── MySoccerNetGS/      # Formatted for GSR
├── outputs/
│   └── states/             # Tracking results (.pklz)
└── logs/slurm/             # Job logs
```

## Pipeline Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | - | Environment setup |
| 2 | `extract_corner_clips.py` | Extract 10-sec corner clips |
| 3 | `format_for_gsr.py` | Format for GSR pipeline |
| 4 | `run_gsr.sbatch` | Run GSR inference |
| 5 | `postprocess_gsr.py` | Convert to parquet |
| 6 | - | GNN training |

## Output Format

### Tracking State Files (.pklz)

ZIP archives containing pandas DataFrames with:

| Column | Description |
|--------|-------------|
| `track_id` | Unique player ID across frames |
| `image_id` | Frame number |
| `team` | Team assignment (left/right) |
| `role` | player / goalkeeper / referee |
| `jersey_number` | Detected jersey number |
| `bbox_pitch` | Pitch coordinates (meters) |

### Loading Results

```python
import zipfile
import pickle

with zipfile.ZipFile('outputs/states/CORNER-0000.pklz', 'r') as zf:
    with zf.open('0.pkl') as f:
        detections = pickle.load(f)

# Get player positions
for _, row in detections.iterrows():
    if row['bbox_pitch']:
        x = row['bbox_pitch']['x_bottom_middle']
        y = row['bbox_pitch']['y_bottom_middle']
        print(f"Track {row['track_id']}: ({x:.1f}m, {y:.1f}m)")
```

## Coordinate System

- **Origin**: Center of pitch (0, 0)
- **X-axis**: -52.5m (left goal) to +52.5m (right goal)
- **Y-axis**: -34m (bottom touchline) to +34m (top touchline)
- **Pitch size**: 105m x 68m (standard)

## Dataset Statistics

| Metric | Value |
|--------|-------|
| SoccerNet games | 550 (1.1TB) |
| Total corners | 4,836 |
| Visible corners | 4,229 |
| Extracted clips | 2,566 |
| Video duration | 10 sec each |

**Games per league**:
- Spain La Liga: 125
- UEFA Champions League: 108
- Italy Serie A: 105
- England EPL: 104
- Germany Bundesliga: 61
- France Ligue 1: 47

### GSR Performance (per clip)

| Metric | Value |
|--------|-------|
| Detections | ~2,300 |
| Frames | ~244 |
| Unique tracks | ~50 |
| Processing time | ~6 min |

## Requirements

See `sn-gamestate/pyproject.toml` for full dependencies.

Key packages:
- PyTorch 2.5.1
- mmcv 2.1.0 (compiled with CUDA)
- mmdet 3.1.0
- mmocr 1.0.1
- tracklab

## SLURM Configuration

```bash
#SBATCH --partition=acltr
#SBATCH --account=students
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
```

Required modules:
```bash
module load GCC/12.3.0 CUDA/12.1.1
export CUDA_HOME=/opt/itu/easybuild/software/CUDA/12.1.1
```

## Troubleshooting

### mmcv._ext not found
Compile mmcv from source with CUDA:
```bash
MMCV_WITH_OPS=1 pip install mmcv==2.1.0 --no-binary :all:
```

### KeyError: 0 in NBJW_Calib
Fixed in `sn_gamestate/calibration/nbjw_calib.py:179`:
```python
# Use .iloc[0] instead of [0]
predictions = metadatas["keypoints"].iloc[0]
```

## License

This project uses:
- [SoccerNet](https://www.soccer-net.org/) data (research use)
- [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) (Apache 2.0)

## Acknowledgments

- SoccerNet team for the dataset and GSR pipeline
- TrackingLaboratory for the tracklab framework
