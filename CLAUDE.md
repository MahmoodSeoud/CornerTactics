# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

CornerTactics extracts **tracking data** (player x, y coordinates and velocities) from SoccerNet broadcast videos using the sn-gamestate pipeline. Data will be used for GNN-based corner kick outcome prediction.

**Goal**: Extract ~4,000 corner kicks with per-frame player pitch coordinates from 500 SoccerNet games.

## Development Environment

**Conda Environment**: `sn-gamestate`
```bash
conda activate sn-gamestate
```

**Requirements**:
- Python 3.9 (exact)
- PyTorch 1.13.1
- CUDA 11.7
- mmcv 2.0.1

## Project Structure

```
CornerTactics/
├── sn-gamestate/              # Cloned: SoccerNet game state reconstruction
├── tracklab/                  # Cloned: Tracking laboratory framework
├── scripts/
│   ├── extract_soccernet_corners.py  # Phase 2.1: Parse Labels-v2.json
│   ├── extract_corner_clips.py       # Phase 2.2: Extract video clips
│   ├── create_corner_metadata.py     # Phase 2.3: Corner metadata JSON
│   ├── format_for_gsr.py             # Phase 3: Format for GSR pipeline
│   ├── run_gsr.sbatch                # Phase 4: SLURM batch script
│   └── postprocess_gsr.py            # Phase 5: Post-processing
├── data/
│   ├── misc/soccernet/videos/        # SoccerNet video data (500 games)
│   ├── corner_clips/                 # Extracted 10-sec corner clips
│   ├── MySoccerNetGS/                # Formatted for GSR pipeline
│   └── processed/                    # Corner metadata JSON/CSV
├── outputs/
│   ├── states/                       # Tracker state .pklz files
│   ├── json/                         # GSR JSON predictions
│   └── processed/                    # Final parquet files
├── logs/
│   └── slurm/                        # SLURM job logs
├── plan.md                           # Full 6-phase implementation plan
├── CLAUDE.md                         # This file
└── README.md
```

## Pipeline Phases

### Phase 1: Environment Setup
```bash
# Clone repos (done)
git clone https://github.com/SoccerNet/sn-gamestate.git
git clone https://github.com/TrackingLaboratory/tracklab.git

# Create conda env with exact versions
conda create -n sn-gamestate pip python=3.9 pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda activate sn-gamestate
pip install -e sn-gamestate/
mim install mmcv==2.0.1
```

### Phase 2: Corner Clip Extraction
```bash
# Extract corners from Labels-v2.json
python scripts/extract_soccernet_corners.py

# Create corner metadata
python scripts/create_corner_metadata.py

# Extract video clips (10-sec each)
python scripts/extract_corner_clips.py --visible-only
```

### Phase 3: Format for GSR Pipeline
```bash
python scripts/format_for_gsr.py
```

### Phase 4: SLURM Inference
```bash
sbatch scripts/run_gsr.sbatch
```

### Phase 5: Post-Processing
```bash
python scripts/postprocess_gsr.py
```

## Dataset Statistics

- **SoccerNet games**: 500 (6 leagues)
- **Total corners**: 4,836
- **Visible corners**: 4,229
- **Video format**: 720p MKV, 10-sec clips

## Code Philosophy

- Straightforward, data-oriented code
- Efficient batch processing with SLURM
- Clear variable names
- Think like John Carmack: fix problems, don't work around them

## Important Notes

1. **Conda Environment**: Use `sn-gamestate` (Python 3.9 + PyTorch 1.13.1)
2. **Data Directory**: All data in `data/` (gitignored, ~100GB when complete)
3. **SLURM**: GSR inference requires GPU nodes via SLURM
4. **Pitch Coordinates**: Output is in meters (105m x 68m standard pitch)

## Git Workflow

- Never commit data files (gitignored)
- Keep commit messages concise
- Main plan reference: `plan.md`
