# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a minimal utility for downloading StatsBomb corner kick data with 360-degree player freeze frames from the StatsBomb open data API.

**Purpose**: Download raw StatsBomb corner kick event data and 360 freeze frame player positions

## Development Environment

**Conda Environment**: `robo`
```bash
conda activate robo
```

## Project Structure

```
CornerTactics/
├── scripts/
│   └── download_statsbomb_raw_jsons.py    # Main download script
├── src/
│   └── statsbomb_loader.py                # StatsBomb data loader module
├── data/                                   # Output directory (gitignored)
│   └── statsbomb/
│       └── raw/
│           ├── competitions.json
│           ├── matches/
│           └── events/
├── logs/                                   # Log files (gitignored)
├── requirements.txt                        # Python dependencies
├── CLAUDE.md                              # This file
└── README.md                              # Project README
```

## Key Files

### `scripts/download_statsbomb_raw_jsons.py`
Main script for downloading StatsBomb corner data. Fetches:
- All available competitions from StatsBomb open data
- Matches for each competition
- Event data (focusing on corner kicks with 360 freeze frames)

**Output**: Raw JSON files saved to `data/statsbomb/raw/`

### `src/statsbomb_loader.py`
Module for loading and processing StatsBomb data. Provides `StatsBombCornerLoader` class with methods:
- `get_available_competitions()`: Fetch available competitions
- `fetch_competition_events()`: Get events for a competition
- `build_corner_dataset()`: Build dataset with corner events
- `save_dataset()`: Save data to CSV

## Usage

### Download StatsBomb Data

```bash
python scripts/download_statsbomb_raw_jsons.py
```

### Use StatsBomb Loader

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

## Dependencies

See `requirements.txt`:
- `pandas`: Data processing
- `numpy`: Numerical operations
- `statsbombpy`: StatsBomb API client
- `statsbomb`: Alternative StatsBomb library
- `tqdm`: Progress bars
- `requests`: HTTP requests

## Code Philosophy

- Straightforward, data-oriented code
- Efficient pandas operations
- Clear variable names and comments
- Minimal abstraction

## Important Notes

1. **Data Directory**: All downloaded data goes to `data/statsbomb/raw/` (gitignored)
2. **Coordinate System**: StatsBomb uses 120x80 pitch (x: 0-120, y: 0-80)
3. **360 Freeze Frames**: Player positions at the exact moment of corner kick
4. **Professional Men's Only**: Script filters for professional men's competitions

## Git Workflow

- Never commit data files (already in `.gitignore`)
- Keep commit messages concise and descriptive
