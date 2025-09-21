# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CornerTactics is a comprehensive soccer corner kick analysis pipeline that successfully extracted **4,826 corner kick frames** from 500 SoccerNet broadcast videos. The system achieved 100% extraction success and provides **4,221 high-quality visible corners** ready for state-of-the-art computer vision analysis (YOLOv8 + player detection) and geometric deep learning models for tactical analysis and outcome prediction.

## Core Pipeline Architecture ✅ COMPLETED

The system follows a clean, refactored architecture:
1. **Data Loading** (`src/data_loader.py`) - Loads both Labels-v2.json (ALL corners) and Labels-v3.json (spatial)
2. **Frame Extraction** (`src/frame_extractor.py`) - Extracts single frames at exact corner moments
3. **Batch Processing** (`src/corner_frame_pipeline.py`) - Processes all 500 games efficiently
4. **Entry Point** (`extract_corners.py`) - Simple command-line interface

The system successfully extracted **4,826 corner frames** with **100% success rate** and **856MB of data** ready for player position analysis.

## Data Structure

```
data/
├── datasets/soccernet/
│   ├── videos/                  # 720p broadcast videos (500 games with both videos + labels)
│   │   ├── england_epl/         # EPL matches with Labels-v2.json + Labels-v3.json
│   │   ├── europe_uefa-champions-league/
│   │   └── france_ligue-1/
│   ├── corner_frames/           # ✅ 4,826 extracted corner frames (856MB)
│   └── tracking/                # SNMOT tracking sequences for validation
│       ├── train/               # 6 corner sequences with ground truth
│       ├── test/
│       └── challenge/
└── insights/                    # ✅ corner_frames_metadata.csv ready
```

### Current Status & Next Steps
- **Extraction**: ✅ COMPLETED - 4,826 corner frames extracted (100% success rate)
- **Data Quality**: ✅ 4,221 visible corners (87.5%) ready for analysis
- **Storage**: ✅ 856MB of corner frame data organized and accessible
- **Labels**: ✅ Both Labels-v2.json (ALL corners) and Labels-v3.json available
- **Next Phase**: Player position detection using YOLOv8 on corner frames
- **Objects to detect**: Players (both teams), goalkeepers, referees, ball
- **Validation**: 6 SNMOT corner sequences with ground truth available

## Common Commands

### Corner Frame Extraction ✅ COMPLETED
```bash
# Extract corner frames (COMPLETED - 4,826 frames extracted)
python extract_corners.py --data-dir data

# Filter visible corners for analysis
python -c "
import pandas as pd
df = pd.read_csv('data/insights/corner_frames_metadata.csv')
visible = df[df.visibility == 'visible']
print(f'Visible corners: {len(visible)} of {len(df)}')
"
```

### Next Phase: Player Detection (TODO)
```bash
# Run YOLOv8 on corner frames (TO BE IMPLEMENTED)
python src/detect_players.py --frames data/datasets/soccernet/corner_frames/

# Train geometric deep learning model (TO BE IMPLEMENTED)
python src/train_model.py --features corner_features.h5
```

### SLURM Jobs (HPC cluster)
```bash
# Download SoccerNet data (Labels-v2.json + Labels-v3.json + videos + tracking)
sbatch scripts/slurm/download_data.sh

# Extract corner frames ✅ COMPLETED (4,826 frames extracted)
sbatch scripts/slurm/extract_corner_frames.sh

# Next: Player detection with GPU (TO BE IMPLEMENTED)
sbatch scripts/slurm/detect_players.sh
```

### Quick Checks
```bash
# Check extracted corner frames ✅
ls data/datasets/soccernet/corner_frames/ | wc -l  # Should show 4826

# Check corner metadata ✅
head data/insights/corner_frames_metadata.csv

# Find corner sequences with ground truth
find data/datasets/soccernet/tracking -name "gameinfo.ini" -exec grep -l "Corner" {} \;

# Check visibility distribution
python -c "
import pandas as pd
df = pd.read_csv('data/insights/corner_frames_metadata.csv')
print(df.visibility.value_counts())
print(f'Success rate: {len(df)} corners extracted')
"
```

## Key Technical Details

- **Video Format**: MKV files (1_720p.mkv for first half, 2_720p.mkv for second half)
- **Labels**: Labels-v2.json (ALL corners) + Labels-v3.json (spatial annotations)
- **Corner Detection**: Extracts from both label sources, removes duplicates
- **Frame Extraction**: Single frame at exact corner moment using ffmpeg
- **Output Format**: CSV with game, half, time, team, visibility, frame_path columns
- **Dataset Stats**: 4,826 total corners, 4,221 visible (87.5% usable)
- **Quality Filter**: Use `visibility == "visible"` for high-quality analysis
- **Storage**: 856MB total, ~200KB per frame (JPEG)
- **Processing Time**: ~23 minutes for 4,826 corners (100% success rate)
- **Success**: 26x improvement from 180 to 4,826 corners!

## Important Constraints

- Pipeline processes entire dataset (4,826 corners from 500 games)
- Requires both Labels-v2.json (comprehensive) and video files
- Frame extraction needs ffmpeg for video processing
- Use `visibility == "visible"` filter for ML (4,221 high-quality corners)
- Video quality is 720p (high quality) for detailed player analysis

## File Responsibilities

- `extract_corners.py` - ✅ Main entry point for corner frame extraction
- `src/data_loader.py` - ✅ Game discovery and Labels-v2/v3 parsing
- `src/frame_extractor.py` - ✅ Single frame extraction using ffmpeg
- `src/corner_frame_pipeline.py` - ✅ Batch processing pipeline
- `src/download_soccernet.py` - ✅ SoccerNet dataset downloads (both label types)
- `scripts/slurm/*.sh` - ✅ Clean HPC cluster job scripts

## SLURM Scripts Status ✅ COMPLETED

Clean, working SLURM scripts:
- ✅ `scripts/slurm/download_data.sh` - Downloads all SoccerNet data
- ✅ `scripts/slurm/extract_corner_frames.sh` - Extracts corner frames (COMPLETED)
- ✅ Proper conda environment activation (`conda activate robo`)
- ✅ Correct directory paths (`corner_frames/` not `soccernet_corner_frames/`)
- ✅ 100% success rate achieved (4,826/4,826 corners)


# Football Corner Prediction ML Project

## Geometric Deep Learning Context

### Core Principles
Geometric Deep Learning provides a unified framework for neural networks by incorporating geometric structure and symmetries. Key principles:

- **Geometric Priors**: Leverage inherent symmetries and structure in data (e.g., spatial relationships on football pitch)
- **Invariance & Equivariance**: Models should be invariant to irrelevant transformations (rotation, translation) but equivariant to meaningful ones
- **Graph Neural Networks**: Represent players and ball as nodes with edges encoding relationships/distances

### Application to Football Corner Kicks
Following TacticAI's approach:

- **Player Positions as Graphs**: Each player is a node, edges represent spatial relationships
- **Temporal Dynamics**: Track how positions evolve during corner kick sequence  
- **Geometric Features**: Distance matrices, angles, formation shapes as input features
- **Data Efficiency**: GDL principles help with limited football data by encoding domain knowledge

### Key Technical Components
1. **Graph Representation**: Convert player coordinates to graph structure
2. **Message Passing**: Information flow between connected players
3. **Pooling Operations**: Aggregate local patterns into global tactical understanding
4. **Geometric Invariances**: Ensure model works regardless of pitch orientation/camera angle

### References
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478) - Foundational framework
- [TacticAI: an AI assistant for football tactics](https://arxiv.org/abs/2310.10553) - Direct application to corner kicks
- [Geometric Deep Learning Guide](https://geometricdeeplearning.com/) - Comprehensive resource
- [YouTube Playlist](https://youtube.com/playlist?list=PLn2-dEmQeTfQ8YVuHBOvAhUlnIPYxkeu3&si=xBYpgKYo3szmUOHM) - Video tutorials

### Implementation Notes for Claude Code
When implementing:
- Use PyTorch Geometric for graph neural networks
- Represent each corner kick as a heterogeneous graph (players, ball, goal posts as different node types)
- Apply geometric transformations to augment limited training data
- Focus on interpretable geometric features that coaches can understand
