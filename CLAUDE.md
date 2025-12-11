# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CornerTactics is a **data extraction and processing project** for corner kick data in professional soccer using StatsBomb's open event data and 360-degree freeze frame player positioning data.

**Purpose**: Download, extract, and process corner kick data with player positions for soccer analytics research

**Project Stage**: Complete data pipeline with comprehensive data (1,933 labeled corners with 360 positioning)

## Development Environment

**Conda Environment**: `robo`
```bash
conda activate robo
```

**Python Version**: 3.x

## Project Structure

```
CornerTactics/
├── scripts/                                    # Data processing scripts
│   ├── download_statsbomb_events.py           # Downloads all StatsBomb event data
│   ├── download_statsbomb_360_freeze_frames.py # Downloads 360 freeze frame data
│   ├── 01_extract_corners_with_freeze_frames.py # Extract corners with freeze frames
│   ├── 02_extract_outcome_labels.py           # Extract outcome labels
│   ├── 03_extract_features.py                 # Feature extraction
│   ├── 04_create_splits.py                    # Create train/val/test splits
│   ├── 07_extract_shot_labels.py              # Extract shot labels
│   ├── 14_extract_temporally_valid_features.py # Temporally valid features
│   └── 16_extract_raw_spatial_features.py     # Raw spatial features
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
│   └── processed/                             # Processed corner data
│       ├── corners_with_freeze_frames.json
│       ├── corners_with_labels.json
│       ├── corners_with_features.csv
│       └── train/val/test_indices.csv
├── tests/                                      # Test suite for data scripts
├── docs/
│   ├── STATSBOMB_DATA_GUIDE.md                # Comprehensive data documentation
│   ├── DATASET_STATISTICS.json                # Dataset statistics
│   └── SHOT_LABEL_VERIFICATION.md             # Shot label verification
├── notes/features/                             # Feature extraction notes
├── corner_clips/                               # Sample corner video clips
├── requirements.txt                            # Python dependencies
├── PLAN.md                                     # Data processing plan
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

### Data Processing Scripts

#### `scripts/01_extract_corners_with_freeze_frames.py`
Extracts corner kicks that have 360 freeze frame data.

**Output**: `data/processed/corners_with_freeze_frames.json` (~1,933 samples)

#### `scripts/02_extract_outcome_labels.py`
Labels corners with outcome classes (Ball Receipt, Clearance, Goalkeeper, Other).

**Output**: `data/processed/corners_with_labels.json`

#### `scripts/03_extract_features.py`
Extracts 49 features from corner events and freeze frames.

**Output**: `data/processed/corners_with_features.csv`

#### `scripts/04_create_splits.py`
Creates match-based stratified train/val/test splits (60/20/20).

**Output**: `data/processed/train_indices.csv`, `val_indices.csv`, `test_indices.csv`

#### `scripts/07_extract_shot_labels.py`
Extracts binary shot labels following TacticAI methodology.

**Output**: `data/processed/corners_with_shot_labels.json`

#### `scripts/14_extract_temporally_valid_features.py`
Extracts only features available at corner kick time (no temporal leakage).

**Output**: `data/processed/corners_features_temporal_valid.csv`

#### `scripts/16_extract_raw_spatial_features.py`
Extracts raw player coordinates and pairwise distances.

**Output**: `data/processed/corners_raw_spatial_features.csv`

### Core Module

#### `src/statsbomb_loader.py`
Data loader class for StatsBomb corner kick analysis.

**Methods**:
- `get_available_competitions()` - Fetch competition list via statsbombpy API
- `fetch_competition_events()` - Get all events for a competition
- `build_corner_dataset()` - Build complete corner dataset with outcomes
- `get_next_action()` - Find outcome event following corner kick
- `save_dataset()` - Save processed data to CSV

### Documentation

#### `docs/STATSBOMB_DATA_GUIDE.md`
Comprehensive documentation of StatsBomb data structure and usage.

**Contents**:
- Dataset statistics and corner outcome distribution
- File format specifications (events, competitions, freeze frames)
- StatsBomb coordinate system (120x80 pitch)
- Event type reference
- Python code examples for data loading and analysis

**Corner Outcome Distribution** (1,933 corners with 360 data):
- Ball Receipt: 1,050 (54.3%)
- Clearance: 453 (23.4%)
- Goal Keeper: 196 (10.1%)
- Other: 234 (12.2%)

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
4. **Match-Based Splits**: Use match-based train/test splits to avoid data leakage

## Usage Examples

### Download StatsBomb Data

```bash
conda activate robo
python scripts/download_statsbomb_events.py
python scripts/download_statsbomb_360_freeze_frames.py
```

### Run Data Processing Pipeline

```bash
python scripts/01_extract_corners_with_freeze_frames.py
python scripts/02_extract_outcome_labels.py
python scripts/03_extract_features.py
python scripts/04_create_splits.py
python scripts/07_extract_shot_labels.py
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
- Keep commit messages concise and descriptive
- Repository is initialized (`.git/` present)
