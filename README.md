# CornerTactics

Predict corner kick outcomes from soccer broadcast videos using deep learning.

## Overview

CornerTactics uses the FAANTRA (Football Action ANticipation TRAnsformer) architecture to predict corner kick outcomes from SoccerNet broadcast videos. The system observes video leading up to a corner kick and anticipates whether it will result in a goal, shot, clearance, or other outcome.

## Status

Finished (2026). What came out of it:

- A dataset of 4,836 corner-kick clips from SoccerNet broadcasts, each labelled with its outcome (table below).
- FAANTRA trained on those clips reaches 12.6% mAP on the 8-class outcome task. On the binary shot / no-shot task it is at chance (50% mAP).
- A classical ML baseline on StatsBomb freeze frames is also at chance (AUC 0.43) once data leakage is removed.

So with this data, whether a corner leads to a shot could not be predicted from what happens before it. Full numbers are in [docs/RESULTS.md](docs/RESULTS.md). A later graph-network attempt on tracked corners lives in `corner_prediction/`.

## Dataset

### Corner Kick Video Clips

- **Total clips**: 4,836
- **Clip duration**: 30 seconds (25s observation + 5s anticipation)
- **Resolution**: 720p MP4
- **Total size**: 114GB

### Outcome Classes

| Outcome | Count | % |
|---------|-------|---|
| NOT_DANGEROUS | 1,939 | 40.1% |
| CLEARED | 1,138 | 23.5% |
| SHOT_OFF_TARGET | 713 | 14.7% |
| SHOT_ON_TARGET | 387 | 8.0% |
| FOUL | 384 | 7.9% |
| GOAL | 172 | 3.6% |
| OFFSIDE | 77 | 1.6% |
| CORNER_WON | 26 | 0.5% |

### Source Data

SoccerNet videos from 6 leagues:
- England Premier League, Spain La Liga, Italy Serie A
- Germany Bundesliga, France Ligue 1, UEFA Champions League

## Project Structure

```
CornerTactics/
├── scripts/                        # Data pipeline scripts
│   ├── 01_build_corner_dataset.py  # Step 1: Build metadata from SoccerNet
│   ├── 02_extract_video_clips.py   # Step 2: Extract 30s video clips
│   ├── 03_prepare_faantra_data.py  # Step 3: Create splits + extract frames
│   └── slurm/                      # SLURM batch scripts
│       └── extract_frames.sbatch
├── FAANTRA/                        # FAANTRA model (forked)
│   ├── data/
│   │   ├── corners/                # Corner dataset
│   │   │   ├── corner_dataset.json # Metadata for all corners
│   │   │   └── clips/              # 4,836 video clips (114GB)
│   │   └── corner_anticipation/    # FAANTRA-formatted data
│   │       ├── train/              # Training frames
│   │       ├── valid/              # Validation frames
│   │       └── test/               # Test frames
│   ├── main.py                     # Training script
│   ├── test.py                     # Evaluation script
│   └── venv/                       # Python environment
├── data/misc/soccernet/videos/     # Source SoccerNet videos (1.1TB)
├── plan.md                         # Detailed project plan
└── CLAUDE.md                       # Development context
```

## Quick Start

### Prerequisites

- Python 3.9+
- FFmpeg 6.0 (via module system)
- decord (for frame extraction)
- SLURM cluster access

### Setup

```bash
cd CornerTactics

# Activate FAANTRA environment
source FAANTRA/venv/bin/activate

# Load FFmpeg (for video processing)
module load GCCcore/12.3.0 FFmpeg/6.0-GCCcore-12.3.0
```

### Data Pipeline

Run from CornerTactics root:

```bash
# Step 1: Build corner dataset (already done)
python scripts/01_build_corner_dataset.py

# Step 2: Extract video clips (already done)
python scripts/02_extract_video_clips.py

# Step 2b: Verify and repair corrupt clips
python scripts/02_extract_video_clips.py --verify --repair

# Step 3a: Create train/val/test splits
python scripts/03_prepare_faantra_data.py --splits-only

# Step 3b: Extract frames (use SLURM for parallel processing)
sbatch scripts/slurm/extract_frames.sbatch                    # Train split
SPLIT=valid CHUNKS=5 sbatch scripts/slurm/extract_frames.sbatch  # Valid split
SPLIT=test CHUNKS=5 sbatch scripts/slurm/extract_frames.sbatch   # Test split
```

### Monitor Progress

```bash
# Check SLURM jobs
squeue -u $USER

# Count extracted frames
ls FAANTRA/data/corner_anticipation/train/clip_*/ 2>/dev/null | wc -l
```

### Train Model

```bash
cd FAANTRA
source venv/bin/activate
python main.py config/corner_config.json corner_model
```

### Evaluate

```bash
python test.py config/corner_config.json checkpoints/best.pth corner_model -s test
```

## Pipeline Details

### Step 1: Build Corner Dataset

Scans SoccerNet Labels-v2.json files to find corner kick events. For each corner:
- Extracts timing information
- Classifies outcome based on events within 15s window
- Creates `corner_dataset.json` with all metadata

### Step 2: Extract Video Clips

Uses FFmpeg to extract 30-second clips from match videos:
- **Observation window**: 25 seconds before corner (what model sees)
- **Anticipation window**: 5 seconds after corner (what model predicts)
- Includes integrity verification and corrupt clip repair

### Step 3: Prepare FAANTRA Data

Formats data for FAANTRA training:
- Creates 80/10/10 train/val/test splits
- Extracts individual frames using decord (750 frames per clip)
- Creates Labels-ball.json with outcome annotations
- Supports chunked parallel extraction via SLURM

## References

- [FAANTRA Paper (CVPR 2025 Workshop)](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Dalal_Action_Anticipation_from_SoccerNet_Football_Video_Broadcasts_CVPRW_2025_paper.html)
- [FAANTRA Repository](https://github.com/MohamadDalal/FAANTRA)
- [SoccerNet](https://www.soccer-net.org/)

## License

This project uses:
- [SoccerNet](https://www.soccer-net.org/) data (research use, requires NDA)
- [FAANTRA](https://github.com/MohamadDalal/FAANTRA) code (see FAANTRA/LICENSE)
