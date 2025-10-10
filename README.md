# CornerTactics

A soccer analytics research project for analyzing and predicting corner kick outcomes using StatsBomb open data.

## Overview

CornerTactics analyzes corner kick outcomes using StatsBomb's open data, combining event data with 360-degree player positioning freeze frames to understand what happens after corner kicks in professional soccer.

**Key Data**: StatsBomb Open Data provides event data with 360 freeze frames showing exact player positions at the moment of corner kicks

## Features

- Download and process StatsBomb corner kick data with 360 player positions
- Extract corner kick outcomes (shots, clearances, goals, etc.)
- Visualize player positioning during corner kicks
- SLURM cluster integration for large-scale data processing

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)
- Access to SLURM cluster (optional, for large-scale processing)

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

## Quick Start

### Download StatsBomb Corner Data

**Local execution:**
```bash
python scripts/download_statsbomb_corners.py
```

**SLURM cluster:**
```bash
sbatch scripts/slurm/download_statsbomb_corners.sh
```

This downloads all professional men's corner kicks with 360 player position data from StatsBomb's open dataset. Output is saved to `data/statsbomb/corners_360.csv`.

### Visualize Corner Kicks

```bash
python scripts/visualize_corners_with_players.py
```

Creates a 2x2 grid visualization showing corner kicks with player positions (attacking team in blue, defending team in orange).

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

### StatsBomb Corner Dataset

The corner dataset (`corners_360.csv`) includes:

- **Match Info**: `match_id`, `competition`, `season`, `home_team`, `away_team`, `match_date`
- **Corner Event**: `minute`, `second`, `team`, `player`
- **Location**: `location_x`, `location_y`, `end_x`, `end_y` (StatsBomb 120x80 pitch)
- **Player Positions**: `num_attacking_players`, `num_defending_players`, `attacking_positions`, `defending_positions` (JSON format)
- **Outcome**: Analysis of what happened after the corner (shot, clearance, etc.)

### Directory Structure

```
CornerTactics/
├── scripts/
│   ├── download_statsbomb_corners.py    # Download StatsBomb 360 data
│   ├── visualize_corners_with_players.py # Create visualizations
│   └── slurm/                            # SLURM job scripts
│       ├── download_statsbomb_corners.sh
│       └── visualize_corners_players.sh
├── src/
│   └── statsbomb_loader.py               # StatsBomb data loader
├── data/                                  # Data directory (gitignored)
│   ├── statsbomb/                        # StatsBomb data
│   └── datasets/                         # Other datasets
├── requirements.txt
├── CLAUDE.md                             # Development guide for Claude Code
└── README.md
```

## SLURM Cluster Usage

For large-scale data processing, submit jobs to the SLURM cluster:

```bash
# Submit job
sbatch scripts/slurm/<script_name>.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/<job_name>_<job_id>.out
```

All SLURM scripts automatically:
- Activate the `robo` conda environment
- Install required dependencies
- Set up Python path
- Save logs to `logs/` at project root

## Data Sources

### StatsBomb Open Data

Free access to event data and 360 freeze frames for select competitions:
- Champions League
- Premier League
- La Liga
- World Cup
- And more...

Documentation: https://github.com/statsbomb/open-data

## Development

### Code Philosophy

This project follows a data-oriented approach:
- Efficient pandas operations over loops
- Batch processing for large datasets
- Clear, straightforward code
- Minimal abstraction

See `CLAUDE.md` for detailed development guidelines.

### Adding New Features

1. Test locally with small data samples
2. Create corresponding SLURM script for cluster execution
3. Add documentation to `CLAUDE.md`
4. Never commit large data files (already in `.gitignore`)

## Coordinate Systems

**StatsBomb Pitch**: 120 units wide × 80 units tall
- X-axis: 0 (defensive goal) to 120 (attacking goal)
- Y-axis: 0 (bottom sideline) to 80 (top sideline)
- Corners typically at (120, 0) or (120, 80)

## Dependencies

Core dependencies:
- `pandas`: Data processing and analysis
- `statsbombpy`: StatsBomb API client
- `matplotlib`: Plotting
- `mplsoccer`: Soccer-specific visualizations
- `tqdm`: Progress bars
- `requests`: HTTP requests

See `requirements.txt` for complete list with versions.

## License

This project uses StatsBomb's open data, which is freely available for research and non-commercial use. Please review StatsBomb's [license terms](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf) before use.

## Contributing

This is a research project. For questions or collaboration inquiries, please open an issue.

## Acknowledgments

- **StatsBomb** for providing open event and 360 data
- **SoccerNet** for video and tracking datasets
- **mplsoccer** for soccer pitch visualization tools
