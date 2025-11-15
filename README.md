# StatsBomb Corner Data Downloader

A simple utility for downloading StatsBomb corner kick data with 360-degree player positioning from their open dataset.

## Overview

This project downloads corner kick events and freeze frame data (360-degree player positions) from StatsBomb's open data API, focusing on professional men's soccer competitions.

## Features

- Download corner kicks from all available StatsBomb open data competitions
- Extract 360-degree freeze frame player positions at the moment of corner kicks
- Filter for professional men's competitions only
- Priority-based download order (Champions League, La Liga, Premier League, etc.)
- Save data in CSV format for further analysis

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CornerTactics
```

2. Create and activate conda environment:
```bash
conda create -n robo python=3.10
conda activate robo
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Download StatsBomb Corner Data

```bash
python scripts/download_statsbomb_raw_jsons.py
```

This will:
- Fetch all available competitions from StatsBomb open data
- Filter for professional men's competitions
- Download corner kick events with 360 player freeze frame data
- Save raw JSON files to `data/statsbomb/raw/`

### Using the StatsBomb Loader

```python
from src.statsbomb_loader import StatsBombCornerLoader

# Initialize loader
loader = StatsBombCornerLoader(output_dir="data/statsbomb")

# Get available competitions
competitions = loader.get_available_competitions()
print(competitions)

# Build corner dataset for specific competition
df = loader.build_corner_dataset(
    country="England",
    division="Premier League",
    season="2019/2020",
    gender="male"
)

# Save dataset
loader.save_dataset(df, "corners_epl_2019.csv")
```

## Data Structure

### Output Directory

```
data/
└── statsbomb/
    └── raw/
        ├── competitions.json
        ├── matches/
        │   └── <competition_id>_<season_id>.json
        └── events/
            └── <match_id>.json
```

### Corner Event Data

Each corner kick includes:
- **Match Info**: competition, season, home team, away team, date
- **Corner Event**: minute, second, team, player
- **Location**: x, y coordinates (StatsBomb 120x80 pitch)
- **360 Freeze Frame**: All player positions at the moment of the corner kick

## Coordinate System

**StatsBomb Pitch**: 120 units wide × 80 units tall
- X-axis: 0 (defensive goal) to 120 (attacking goal)
- Y-axis: 0 (bottom sideline) to 80 (top sideline)
- Corners typically at (120, 0) or (120, 80)

## Dependencies

- `pandas`: Data processing
- `statsbombpy`: StatsBomb API client
- `tqdm`: Progress bars
- `requests`: HTTP requests

See `requirements.txt` for complete list with versions.

## Data Sources

### StatsBomb Open Data

Free access to event data and 360 freeze frames for select competitions:
- Champions League
- Premier League
- La Liga
- World Cup
- And more...

Documentation: https://github.com/statsbomb/open-data

## License

This project uses StatsBomb's open data, which is freely available for research and non-commercial use. Please review StatsBomb's [license terms](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf) before use.

## Acknowledgments

- **StatsBomb** for providing open event and 360 data
