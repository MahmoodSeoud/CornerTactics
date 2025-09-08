# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CornerTactics is a football/soccer analytics tool that analyzes defensive positioning during corner kicks using SoccerNet data. The system processes tracking data to identify defensive formations, calculate metrics, and generate tactical recommendations.

## Core Architecture

The codebase is organized into four main modules in `corner_tactics/`:

- `data_loader.py` - Handles SoccerNet dataset downloading and parsing corner kick events from Labels-v2.json files
- `tracking_processor.py` - Processes player tracking data, normalizes positions, and classifies defensive formations  
- `formation_analyzer.py` - Performs statistical analysis on formations using clustering (DBSCAN) and generates recommendations
- `visualizer.py` - Creates matplotlib/plotly visualizations of formations, heatmaps, and pitch diagrams

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Running Analysis
```bash
# Basic analysis with existing data
python main.py --data-path ./data/soccernet --split train

# Download SoccerNet data (requires password)
python main.py --download --password YOUR_PASSWORD --data-path ./data/soccernet

# Generate visualizations
python main.py --visualize --output ./results
```

### Data Structure
- SoccerNet data stored in `data/soccernet/league/season/match/` format
- Each match contains `Labels-v2.json` with timestamped events
- Tracking data expected in same directory structure
- Results output to `./results/` by default

## Key Data Flows

1. **Corner Detection**: Loads Labels-v2.json files to extract corner kick timestamps
2. **Tracking Processing**: Normalizes player positions from tracking data to pitch coordinates (68x105m)
3. **Formation Analysis**: Clusters defensive positions using DBSCAN, calculates compactness metrics
4. **Visualization**: Generates formation plots and heatmaps using matplotlib

## Important Constants

- Pitch dimensions: 68m x 105m (width x height)
- Defensive third threshold: pitch_width / 3
- Formation clustering eps parameter: 3.0 meters
- Minimum tracking frames: 150 frames per sequence