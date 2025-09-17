# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CornerTactics is a comprehensive soccer corner kick analysis pipeline that uses state-of-the-art computer vision (YOLOv8 + ByteTrack) to track players in **4,229 visible corner kicks** from SoccerNet broadcast videos, enabling robust machine learning models for outcome prediction and tactical analysis.

## Core Pipeline Architecture

The system follows a simple 3-step flow:
1. **Data Loading** (`src/data_loader.py`) - Loads SoccerNet annotations and lists available games
2. **Video Extraction** (`src/corner_extractor.py`) - Extracts 30-second clips around corner events  
3. **Analysis** (`main.py`) - Orchestrates the pipeline and generates CSV output

The system focuses exclusively on the 12 corner sequences with tracking data for deep learning model development.

## Data Structure

```
data/
├── datasets/soccernet/
│   ├── soccernet_videos/        # 720p broadcast videos (550 games total)
│   │   ├── england_epl/         # EPL matches
│   │   ├── europe_uefa-champions-league/
│   │   └── france_ligue-1/
│   └── soccernet_tracking/      # SNMOT tracking sequences (30-second clips)
│       ├── train/               # 57 sequences (30s each)
│       ├── test/                # 49 sequences (30s each)
│       └── challenge/           # 58 sequences + 1 full half-time (45 min)
└── insights/                    # Analysis results (CSV files)
```

### Tracking Approach
- **Detection Model**: YOLOv8 (state-of-the-art object detection)
- **Tracking Algorithm**: ByteTrack (handles occlusions in crowded scenes)
- **Corner Coverage**: 4,229 visible corners from 500 games
- **Processing**: 30-second clips at 25 FPS
- **Objects tracked**: Players (both teams), goalkeepers, referees, ball
- **Validation**: 12 SNMOT sequences with ground truth for quality checks

## Common Commands

### ML Model Development
```bash
# Extract features from 12 corner sequences
python src/feature_extractor.py --tracking-data data/datasets/soccernet/soccernet_tracking

# Train geometric deep learning model
python src/train_model.py --sequences data/corner_sequences.h5

# Evaluate on test split
python src/evaluate.py --model models/corner_gnn.pt
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
# List corner sequences
find data/datasets/soccernet/soccernet_tracking -name "gameinfo.ini" -exec grep -l "Corner" {} \;

# Load tracking data for a sequence
from src.tracking_loader import SNMOTLoader
loader = SNMOTLoader('data/datasets/soccernet/soccernet_tracking/train/SNMOT-067')
positions = loader.get_player_positions()
```

## Key Technical Details

- **Video Format**: MKV files (1_720p.mkv for first half, 2_720p.mkv for second half)
- **Annotations**: Labels-v3.json contains corner events in actions dict with imageMetadata
- **Corner Detection**: Finds events where `imageMetadata.label == "Corner"` in actions
- **Clip Extraction**: Default 30 seconds (10s before, 20s after corner)
- **Output Format**: CSV with game, half, time, team, visibility columns
- **Primary Dataset**: 4,229 visible corner kicks for tracking
- **Training Split**: ~3,000 corners for model development
- **Validation Split**: ~600 corners for hyperparameter tuning
- **Test Split**: ~629 corners for final evaluation
- **Processing Time**: ~12 hours on single GPU for all corners

## Important Constraints

- Pipeline ALWAYS processes entire dataset (no partial processing)
- SoccerNet API downloads complete splits (~100GB per split)
- Each game directory must contain both video files and Labels-v3.json
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
