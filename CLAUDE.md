# CLAUDE.md

Context for Claude Code when working with this repository.

## Project Overview

CornerTactics predicts corner kick outcomes from SoccerNet broadcast videos using FAANTRA (Football Action ANticipation TRAnsformer).

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Build Dataset | **Complete** | Corner metadata from SoccerNet labels |
| 2. Video Clip Extraction | **Complete** | 4,836 clips extracted (114GB) |
| 3. Frame Extraction | **Complete** | Frames extracted for all splits |
| 4. Model Training | **Complete** | FAANTRA trained on corner outcomes |
| 5. Evaluation | **Complete** | 12.6% mAP@∞ (8-class), 50% mAP@∞ (binary) |

## Results Summary

| Task | Model | Result |
|------|-------|--------|
| 8-class outcome | FAANTRA | mAP@∞ = 12.6% |
| Binary shot/no-shot | FAANTRA | mAP@∞ = 50% (random) |
| Binary shot (StatsBomb) | Classical ML | AUC = 0.43 |
| Ball actions (baseline) | FAANTRA | mAP@∞ = 18.48% |

**Key Finding**: Shot prediction from pre-corner observation achieves random performance (~50%), confirming outcome depends on post-corner events.

## Project Structure

```
CornerTactics/
├── scripts/                           # Data pipeline (run from project root)
│   ├── 01_build_corner_dataset.py     # Build metadata from SoccerNet labels
│   ├── 02_extract_video_clips.py      # Extract 30s video clips
│   ├── 03_prepare_faantra_data.py     # Create splits + extract frames
│   └── slurm/
│       └── extract_frames.sbatch      # Parallel frame extraction
├── FAANTRA/                           # FAANTRA model code
│   ├── data/
│   │   ├── corners/
│   │   │   ├── corner_dataset.json    # Metadata for 4,836 corners
│   │   │   └── clips/                 # Video clips (114GB)
│   │   └── corner_anticipation/       # FAANTRA-formatted frames
│   │       ├── train/, valid/, test/  # Split directories
│   │       └── *.json, class.txt      # Labels and config
│   ├── main.py, train.py, test.py     # Training/evaluation
│   └── venv/                          # Python environment
├── data/misc/soccernet/videos/        # Source SoccerNet videos (1.1TB)
├── plan.md                            # Detailed project plan
└── CLAUDE.md                          # This file
```

## Data Pipeline

All scripts run from CornerTactics root directory:

```bash
cd /home/mseo/CornerTactics
source FAANTRA/venv/bin/activate

# Step 1: Build corner dataset (already complete)
python scripts/01_build_corner_dataset.py

# Step 2: Extract video clips (already complete)
python scripts/02_extract_video_clips.py

# Step 2b: Verify and repair corrupt clips
python scripts/02_extract_video_clips.py --verify --repair

# Step 3a: Create train/val/test splits
python scripts/03_prepare_faantra_data.py --splits-only

# Step 3b: Extract frames (parallel via SLURM)
sbatch scripts/slurm/extract_frames.sbatch                        # Train
SPLIT=valid CHUNKS=5 sbatch scripts/slurm/extract_frames.sbatch   # Valid
SPLIT=test CHUNKS=5 sbatch scripts/slurm/extract_frames.sbatch    # Test
```

## Dataset

### Video Clips
- **Location**: `FAANTRA/data/corners/clips/corner_XXXX/720p.mp4`
- **Count**: 4,836 clips
- **Duration**: 30 seconds (25s observation + 5s anticipation)
- **Size**: 114GB total

### Outcome Distribution
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

### Train/Val/Test Splits
- Train: 3,868 clips (80%)
- Valid: 483 clips (10%)
- Test: 485 clips (10%)

## Environment Setup

```bash
# Activate Python environment
source FAANTRA/venv/bin/activate

# Load FFmpeg (for video processing)
module load GCCcore/12.3.0 FFmpeg/6.0-GCCcore-12.3.0
```

## Common Commands

```bash
# Check video clip extraction status
ls FAANTRA/data/corners/clips/ | wc -l  # Should be 4836

# Check frame extraction progress
ls -d FAANTRA/data/corner_anticipation/train/clip_*/ 2>/dev/null | wc -l

# Verify corrupt clips
python scripts/02_extract_video_clips.py --verify

# Monitor SLURM jobs
squeue -u $USER
```

## Training (after frame extraction)

```bash
cd FAANTRA
source venv/bin/activate
python main.py config/corner_config.json corner_model
```

## Important Notes

1. **Run scripts from CornerTactics root** - All paths are relative to project root
2. **FFmpeg module required** - Load before video processing
3. **Large data** - Clips are 114GB, source videos are 1.1TB (gitignored)
4. **Class imbalance** - 40% NOT_DANGEROUS vs 0.5% CORNER_WON - use weighted loss
5. **SLURM for parallelism** - Use array jobs for frame extraction
