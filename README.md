# CornerTactics

Soccer corner kick analysis pipeline that processes SoccerNet broadcast videos to extract and analyze corner kick events across hundreds of matches.

## Project Status

- ✅ **Data Structure**: Organized SoccerNet dataset (373 games with videos, 550 total directories)
- ✅ **Analysis Pipeline**: Extract corner events from 373 games (3564 corners found)
- ✅ **SLURM Integration**: All cluster job scripts fixed and working
- ⚠️ **Video Downloads**: Only 373/550 games have video files (need to complete download)

## Quick Start

### HPC Cluster (Recommended)

```bash
# Download missing video files
sbatch scripts/slurm/download_videos.sh

# Analyze corners (no video clips) - fast
sbatch scripts/slurm/analyze_corners.sh

# Extract corner clips + analyze - slow  
sbatch scripts/slurm/extract_corners.sh
```

### Local Development

```bash
# Analyze only (uses existing data)
python main.py --no-clips --data-dir data/datasets/soccernet/soccernet_videos

# Full pipeline with video extraction
python main.py --data-dir data/datasets/soccernet/soccernet_videos --output data/insights/corners.csv
```

## Data Structure

```
data/
├── datasets/soccernet/
│   ├── soccernet_videos/        # 720p broadcast videos
│   │   ├── england_epl/         # EPL matches  
│   │   ├── europe_uefa-champions-league/
│   │   └── france_ligue-1/
│   └── soccernet_tracking/      # Player tracking data
│       ├── train.zip            # Training split
│       ├── test.zip             # Test split
│       └── challenge.zip        # Challenge split
└── insights/                    # Analysis results (CSV files)
```

## Current Dataset Status

- **Total Games**: 550 directories found
- **Games with Videos**: 373 (67.8% complete)
- **Games with Labels**: 500 annotation files
- **Total Corners**: 3,564 corner events analyzed
- **Video Quality**: 720p broadcast footage

## Pipeline Architecture

1. **Data Loading** (`src/data_loader.py`) - Loads SoccerNet annotations and lists available games
2. **Video Extraction** (`src/corner_extractor.py`) - Extracts 30-second clips around corner events  
3. **Analysis** (`main.py`) - Orchestrates the pipeline and generates CSV output

## SLURM Job Scripts

All scripts properly configured with conda environment activation:

- `scripts/slurm/analyze_corners.sh` - Fast analysis (no video clips)
- `scripts/slurm/extract_corners.sh` - Full pipeline with clips
- `scripts/slurm/download_videos.sh` - Download missing videos
- `scripts/slurm/extract_tracklets.sh` - Extract tracking data

## Output

- **Analysis CSV**: `data/insights/corners_analysis.csv` - All corner events with metadata
- **Clips CSV**: `data/insights/corners_with_clips.csv` - With video clip paths
- **Corner Statistics**: Home vs away, half distribution, team analysis

## Requirements

- Python 3.11+ (conda environment: `robo`)
- ffmpeg (for video processing)
- SoccerNet dataset access

## File Responsibilities

- `main.py` - Entry point, orchestrates full pipeline
- `src/data_loader.py` - Game discovery and annotation parsing  
- `src/corner_extractor.py` - Video clip extraction using ffmpeg
- `src/download_soccernet.py` - SoccerNet dataset downloads
- `scripts/slurm/*.sh` - HPC cluster job scripts