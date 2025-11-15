# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CornerTactics is a **machine learning research project** for predicting corner kick outcomes in professional soccer using StatsBomb's open event data and 360-degree freeze frame player positioning data.

**Purpose**: Build ML models to predict corner kick outcomes and player positioning using graph neural networks and traditional ML methods

**Project Stage**: Mid-to-advanced with trained models and comprehensive data (1,933 labeled corners with 360° positioning)

## Development Environment

**Conda Environment**: `robo`
```bash
conda activate robo
```

**Python Version**: 3.x with deep learning dependencies

## Project Structure

```
CornerTactics/
├── scripts/                                    # Data download scripts
│   ├── download_statsbomb_events.py           # Downloads all StatsBomb event data
│   └── download_statsbomb_360_freeze_frames.py # Downloads 360 freeze frame data
├── src/                                        # Core modules
│   └── statsbomb_loader.py                    # StatsBomb data loader class
├── data/                                       # Data directory (gitignored, ~11.3GB)
│   ├── statsbomb/
│   │   ├── events/                            # 3,464 match event JSON files
│   │   │   ├── events/*.json
│   │   │   ├── competitions.json
│   │   │   ├── master_event_sequence.json     # Combined sequence (264MB)
│   │   │   ├── match_index.csv
│   │   │   └── matches_with_corners.csv
│   │   └── freeze-frames/                     # 323 freeze frame JSON files
│   ├── processed/
│   │   └── corner_outcome_distribution.json   # Outcome analysis
│   └── misc/                                  # Additional data sources
│       ├── soccernet/                         # SoccerNet corner clips
│       ├── soccersynth/                       # Synthetic soccer data
│       └── ussf_data_sample.pkl               # USSF data sample
├── models/                                     # Trained models (gitignored)
│   ├── final/                                 # Production-ready models
│   │   ├── mlp_event_best.pth                 # MLP for event prediction (728KB)
│   │   ├── mlp_receiver_best.pth              # MLP for receiver prediction (755KB)
│   │   ├── xgboost_event_best.json            # XGBoost event model (1.6MB)
│   │   └── xgboost_receiver_best.json         # XGBoost receiver model (13MB)
│   ├── hyperparameter_search/                 # Hyperparameter optimization results
│   ├── receiver_prediction/                   # Receiver prediction task
│   └── corner_gnn_*/                          # 12 GNN model checkpoints
├── runs/                                       # TensorBoard logs (gitignored)
├── docs/
│   └── STATSBOMB_DATA_GUIDE.md                # Comprehensive data documentation (16KB)
├── logs/                                       # Log files (gitignored)
├── requirements.txt                            # Python dependencies
├── PLAN.md                                     # Implementation roadmap
├── CLAUDE.md                                   # This file
└── README.md                                   # Project overview
```

## Dataset Statistics

```
Total Competitions:     75
Total Matches:          3,464
Total Events:           12,188,949
Total Corners:          34,049
Corners with 360 Data:  1,933 (5.7%)
Context Events:         366,281
Data Volume:            ~11.3GB
```

## Key Files

### Data Download Scripts

#### `scripts/download_statsbomb_events.py`
Downloads complete StatsBomb event data from GitHub open-data repository.

**Features**:
- Fetches all competitions, matches, and event data (3,464 matches)
- Creates match index and corner kick index
- Generates master event sequence file
- Retry logic with exponential backoff
- Rate limiting (0.1s delay between requests)

**Output**: `data/statsbomb/events/`

#### `scripts/download_statsbomb_360_freeze_frames.py`
Downloads StatsBomb 360-degree freeze frame data for set pieces.

**Features**:
- Identifies competitions with 360 data available
- Downloads freeze frame files for matches (323 files)
- Matches corner event UUIDs with freeze frames
- Tracks coverage statistics (1,933 corner freeze frames)

**Output**: `data/statsbomb/freeze-frames/`

### Core Module

#### `src/statsbomb_loader.py`
Data loader class for StatsBomb corner kick analysis.

**Methods**:
- `get_available_competitions()` - Fetch competition list via statsbombpy API
- `fetch_competition_events()` - Get all events for a competition
- `build_corner_dataset()` - Build complete corner dataset with outcomes
- `get_next_action()` - Find outcome event following corner kick
- `save_dataset()` - Save processed data to CSV

**Note**: Uses statsbombpy API; may need updates to work with downloaded JSON files directly.

### Documentation

#### `docs/STATSBOMB_DATA_GUIDE.md`
Comprehensive documentation of StatsBomb data structure and usage (16KB).

**Contents**:
- Dataset statistics and corner outcome distribution
- File format specifications (events, competitions, freeze frames)
- StatsBomb coordinate system (120x80 pitch)
- Event type reference
- Python code examples for data loading and analysis
- Competition coverage information

**Corner Outcome Distribution** (1,933 corners with 360 data):
- Ball Receipt: 1,050 (54.3%)
- Clearance: 453 (23.4%)
- Goal Keeper: 196 (10.1%)
- Other: 234 (12.2%)

#### `PLAN.md`
Implementation roadmap and project status assessment.

**Key Insights**:
- Identifies gaps in current implementation
- Provides detailed 6-task implementation plan for baseline models
- Outlines expected baseline performance metrics
- Emphasizes proper data handling (match-based splits, class imbalance)

## Machine Learning Models

### Graph Neural Networks (GNN)

**12 trained GNN checkpoints** using Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT).

**Architecture Example**:
- Model: GCN/GAT
- Hidden layers: [64, 128, 64]
- Dropout: 0.3
- Loss: Weighted cross-entropy / Focal loss
- Optimizer: Adam with cosine annealing

**Tasks**:
- Shot prediction (binary classification)
- Goal prediction (binary classification)

**Performance** (shot prediction):
- Best validation AUC: 0.636
- Test accuracy: 68.5%
- Test AUC: 0.565

### Production Models

Located in `models/final/`:

1. **MLP Event Predictor** (728KB) - Neural network for event outcome classification
2. **MLP Receiver Predictor** (755KB) - Neural network for receiver identification
3. **XGBoost Event Model** (1.6MB) - Gradient boosting for event outcome
4. **XGBoost Receiver Model** (13MB) - Gradient boosting for receiver identification

### Hyperparameter Optimization

Systematic search results in `models/hyperparameter_search/`:
- MLP event prediction (best validation: 0.526)
- MLP receiver prediction
- XGBoost event prediction
- XGBoost receiver prediction

**Best MLP Architecture**: [512, 256, 128, 64] with LR=0.01, batch_size=128, dropout=0.2

### Receiver Prediction Task

Novel task: predicting which player will receive the corner kick.

**Performance**:
- Test top-1 accuracy: 2.0%
- Test top-3 accuracy: 14.0%
- Test top-5 accuracy: 32.7%

This is a challenging multi-class problem (predicting 1 player out of ~22).

## Dependencies

See `requirements.txt`:
- `pandas>=2.0.0` - Data processing
- `numpy>=1.24.0` - Numerical operations
- `statsbombpy>=1.11.0` - StatsBomb API client
- `statsbomb>=0.10.0` - Alternative StatsBomb library
- `matplotlib>=3.7.0` - Visualization
- `mplsoccer>=1.3.0` - Soccer pitch visualization
- `tqdm>=4.65.0` - Progress bars
- `requests>=2.31.0` - HTTP requests

Additional ML dependencies (check for PyTorch, XGBoost, scikit-learn in environment).

## Code Philosophy

- Straightforward, data-oriented code
- Efficient pandas operations
- Clear variable names and comments
- Minimal abstraction
- Think like John Carmack: fix problems, don't work around them

## Important Notes

1. **Data Directory**: All downloaded data goes to `data/statsbomb/` (gitignored, ~11.3GB)
2. **Coordinate System**: StatsBomb uses 120x80 pitch (x: 0-120, y: 0-80)
3. **360 Freeze Frames**: Player positions at the exact moment of corner kick (1,933 available)
4. **Model Checkpoints**: Stored in `models/` (gitignored)
5. **TensorBoard Logs**: Training runs logged in `runs/` for visualization
6. **Match-Based Splits**: Use match-based train/test splits to avoid data leakage

## Known Gaps (from PLAN.md)

1. **Missing extraction pipeline** - No scripts for extracting corners from downloaded JSONs
2. **No feature engineering code** - Only documented in PLAN.md
3. **Incomplete baseline implementation** - GNN models exist but traditional baselines may need work
4. **StatsBombLoader outdated** - Uses API instead of local JSON files

## Usage Examples

### Download StatsBomb Data

```bash
conda activate robo
python scripts/download_statsbomb_events.py
python scripts/download_statsbomb_360_freeze_frames.py
```

### Use StatsBomb Loader (API-based)

```python
from src.statsbomb_loader import StatsBombCornerLoader

loader = StatsBombCornerLoader(output_dir="data/statsbomb")
competitions = loader.get_available_competitions()
df = loader.build_corner_dataset(
    country="England",
    division="Premier League",
    season="2019/2020"
)
loader.save_dataset(df, "corners_epl_2019.csv")
```

### Load Event Data Directly (JSON-based)

```python
import json
import pandas as pd

# Load match events
with open('data/statsbomb/events/events/15946.json') as f:
    events = json.load(f)

# Filter corner kicks
corners = [e for e in events if e.get('type', {}).get('name') == 'Pass'
           and e.get('pass', {}).get('type', {}).get('name') == 'Corner']

# Load freeze frames
with open('data/statsbomb/freeze-frames/15946.json') as f:
    freeze_frames = json.load(f)
```

See `docs/STATSBOMB_DATA_GUIDE.md` for detailed API reference.

## Git Workflow

- Never commit data files (covered by `.gitignore`)
- Never commit model checkpoints (covered by `.gitignore`)
- Keep commit messages concise and descriptive
- Repository is initialized (`.git/` present)
