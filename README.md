# CornerTactics

Soccer corner kick analysis pipeline that processes SoccerNet broadcast videos to extract and analyze corner kick events across hundreds of matches.

## Project Status

- ✅ **Data Structure**: Organized SoccerNet dataset (500 games with both videos and labels)
- ✅ **Corner Extraction**: **4,826 corner frames extracted** from 500 games (100% success rate)
- ✅ **High Quality Data**: 4,221 visible corners (87.5%) ready for player analysis
- ✅ **SLURM Integration**: Clean, working scripts for HPC processing
- ✅ **Refactored Codebase**: Simple, maintainable architecture

## Quick Start

### Corner Frame Extraction ✅ COMPLETED

Single frames extracted at the exact moment each corner kick is taken:

```bash
# Extract frames from all corners (requires Labels-v2.json/v3.json + videos)
python extract_corners.py --data-dir ./data

# Custom output location
python extract_corners.py --data-dir ./data --output ./my_corner_frames.csv
```

**Current Status**: ✅ **4,826 corner frames extracted** (856MB) + metadata CSV ready for ML analysis.

### HPC Cluster Scripts

```bash
# Download SoccerNet data (Labels-v2.json + Labels-v3.json + videos + tracking)
sbatch scripts/slurm/download_data.sh

# Extract corner frames ✅ COMPLETED
sbatch scripts/slurm/extract_corner_frames.sh
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

## Current Dataset Status ✅ EXTRACTION COMPLETE

- **Total Games**: 500 games with both videos and labels
- **Corner Frames**: **4,826 extracted** (100% success rate)
- **Visible Corners**: **4,221** (87.5%) - High quality for player analysis
- **Not Shown Corners**: 605 (12.5%) - Filter out for ML
- **Storage**: 856MB of corner frame data
- **Video Quality**: 720p broadcast footage
- **Labels**: Both Labels-v2.json (ALL corners) and Labels-v3.json (spatial)

## Pipeline Architecture

### Phase 1: Corner Frame Extraction ✅ COMPLETED
1. **Data Loading** (`src/data_loader.py`) - Loads both Labels-v2.json and Labels-v3.json
2. **Frame Extraction** (`src/frame_extractor.py`) - Extracts single frames at corner moments using ffmpeg
3. **Batch Pipeline** (`src/corner_frame_pipeline.py`) - Processes all games and generates metadata CSV
4. **CLI Interface** (`extract_corners.py`) - Simple entry point for frame extraction

### Phase 2: Video Analysis (Legacy/Future)
1. **Video Extraction** (`src/corner_extractor.py`) - Extracts 30-second clips around corner events
2. **Analysis** (`main.py`) - Orchestrates the pipeline and generates CSV output

## SLURM Job Scripts

All scripts properly configured with conda environment activation:

- `scripts/slurm/analyze_corners.sh` - Fast analysis (no video clips)
- `scripts/slurm/extract_corners.sh` - Full pipeline with clips
- `scripts/slurm/download_videos.sh` - Download missing videos
- `scripts/slurm/extract_tracklets.sh` - Extract tracking data

## Output ✅ READY

- **Corner Frames**: `data/datasets/soccernet/corner_frames/*.jpg` (4,826 frames, 856MB)
- **Metadata CSV**: `data/insights/corner_frames_metadata.csv` - Complete corner data
- **Visibility Filter**: Use `visibility == "visible"` for 4,221 high-quality frames
- **Ready for**: YOLOv8 player detection and geometric deep learning

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