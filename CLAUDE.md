# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CornerTactics is a focused soccer corner kick analysis pipeline that processes SoccerNet broadcast videos to extract and analyze corner kick events across hundreds of matches.

## Core Pipeline Architecture

The system follows a simple 3-step flow:
1. **Data Loading** (`src/data_loader.py`) - Loads SoccerNet annotations and lists available games
2. **Video Extraction** (`src/corner_extractor.py`) - Extracts 30-second clips around corner events  
3. **Analysis** (`main.py`) - Orchestrates the pipeline and generates CSV output

All processing is batch-oriented - the system always processes ALL games in the dataset (no single-game option).

## Data Structure

```
data/
├── datasets/soccernet/
│   ├── soccernet_videos/        # 720p broadcast videos (551 games)
│   │   ├── england_epl/         # EPL matches
│   │   ├── europe_uefa-champions-league/
│   │   └── france_ligue-1/
│   └── soccernet_tracking/      # Player tracking data (ZIP files extract here)
│       ├── train.zip            # Training split tracking data
│       ├── test.zip             # Test split tracking data
│       └── challenge.zip        # Challenge split tracking data
└── insights/                    # Analysis results (CSV files)
```

## Common Commands

### Full Pipeline
```bash
# Process all 551 games with video extraction
python main.py --data-dir data/datasets/soccernet/soccernet_videos --output data/insights/corners.csv

# Analysis only (no video clips) - faster
python main.py --no-clips --data-dir data/datasets/soccernet/soccernet_videos
```

### SLURM Jobs (HPC cluster)
```bash
# Download SoccerNet data (one-time)
sbatch scripts/slurm/download_videos.sh

# Extract corner clips with GPU
sbatch scripts/slurm/extract_corners.sh  

# Analysis only (lightweight)
sbatch scripts/slurm/analyze_corners.sh
```

### Quick Checks
```bash
# Count total games
find data/datasets/soccernet/soccernet_videos -mindepth 3 -maxdepth 3 -type d | wc -l

# List available games  
from src.data_loader import SoccerNetDataLoader
loader = SoccerNetDataLoader('data/datasets/soccernet/soccernet_videos')
games = loader.list_games()
```

## Key Technical Details

- **Video Format**: MKV files (1_720p.mkv for first half, 2_720p.mkv for second half)
- **Annotations**: Labels-v2.json contains corner events with gameTime and team
- **Corner Detection**: Finds events where `label == "Corner"` in annotations
- **Clip Extraction**: Default 30 seconds (10s before, 20s after corner)
- **Output Format**: CSV with game, half, time, team, visibility columns
- **Current Status**: 373/550 games have videos downloaded (3,564 corners found)

## Important Constraints

- Pipeline ALWAYS processes entire dataset (no partial processing)
- SoccerNet API downloads complete splits (~100GB per split)
- Each game directory must contain both video files and Labels-v2.json
- Video quality is 720p (high quality) for analysis

## File Responsibilities

- `main.py` - Entry point, orchestrates full pipeline
- `src/data_loader.py` - Game discovery and annotation parsing
- `src/corner_extractor.py` - Video clip extraction using ffmpeg
- `src/download_soccernet.py` - SoccerNet dataset downloads
- `scripts/slurm/*.sh` - HPC cluster job scripts (all fixed with conda activation)

## SLURM Scripts Status

All SLURM scripts have been fixed and verified:
- ✅ Proper conda environment activation (`conda activate robo`)
- ✅ No module load errors
- ✅ Correct data directory paths
- ✅ All scripts tested and working