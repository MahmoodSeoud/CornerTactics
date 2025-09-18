# CornerTactics

Soccer corner kick analysis pipeline that processes SoccerNet broadcast videos to extract and analyze corner kick events across hundreds of matches.

## Project Status

- ✅ **Data Structure**: Organized SoccerNet dataset (373 games with videos, 550 total directories)
- ✅ **Analysis Pipeline**: Extract corner events from 373 games (3564 corners found)
- ✅ **SLURM Integration**: All cluster job scripts fixed and working
- ⚠️ **Video Downloads**: Only 373/550 games have video files (need to complete download)

## Quick Start

### Phase 1: Corner Frame Extraction (NEW!)

Extract single frames at the exact moment each corner kick is taken:

```bash
# Extract frames from all corners (requires Labels-v3.json + videos)
python src/cli.py --data-dir ./data

# Custom output location
python src/cli.py --data-dir ./data --output ./my_corner_frames.csv
```

**Output**: ~4,836 corner frames (~1GB) + metadata CSV for machine learning analysis.

### HPC Cluster (Full Pipeline)

```bash
# Download SoccerNet data
sbatch scripts/slurm/01_fetch_soccernet_dataset.sh

# Extract corner frames
sbatch scripts/slurm/05_extract_corner_clips.sh

# Analysis only (legacy)
sbatch scripts/slurm/analyze_corners.sh
```

### Local Development (Legacy)

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

### Phase 1: Corner Frame Extraction (Current)
1. **Data Loading** (`src/data_loader.py`) - Loads SoccerNet annotations and lists available games
2. **Frame Extraction** (`src/frame_extractor.py`) - Extracts single frames at corner moments using ffmpeg
3. **Batch Pipeline** (`src/corner_frame_pipeline.py`) - Processes all games and generates metadata CSV
4. **CLI Interface** (`src/cli.py`) - Command-line tool for easy frame extraction

### Phase 2: Video Analysis (Legacy/Future)
1. **Video Extraction** (`src/corner_extractor.py`) - Extracts 30-second clips around corner events
2. **Analysis** (`main.py`) - Orchestrates the pipeline and generates CSV output

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

### Core Pipeline (Phase 1)
- `src/cli.py` - Command-line interface for corner frame extraction
- `src/corner_frame_pipeline.py` - Batch processing pipeline for all games
- `src/frame_extractor.py` - Single frame extraction using ffmpeg at corner timestamps
- `src/data_loader.py` - Game discovery and annotation parsing (Labels-v2/v3)

### Legacy/Utilities
- `main.py` - Entry point for legacy full pipeline (30s clips)
- `src/corner_extractor.py` - Video clip extraction using ffmpeg
- `src/download_soccernet.py` - SoccerNet dataset downloads
- `scripts/slurm/*.sh` - HPC cluster job scripts