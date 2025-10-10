# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CornerTactics is a soccer analytics research project focused on analyzing corner kick outcomes using StatsBomb open data (event data + 360 player positioning freeze frames). The project aims to predict corner kick outcomes by combining event data with detailed player positioning at the moment of the corner kick.

## Development Environment

### SLURM Cluster Setup
This project runs on an HPC cluster using SLURM for job scheduling. All major data processing tasks should be submitted as SLURM jobs.

**Conda Environment**: `robo`
```bash
conda activate robo
```

**Common SLURM Parameters**:
- Partition: `dgpu` (for GPU jobs) or standard CPU partition
- Account: `researchers`
- Typical resources: 4-8 CPUs, 8-16GB RAM, 2-4 hour time limits

### Running SLURM Jobs
```bash
# Submit a job
sbatch scripts/slurm/<script_name>.sh

# Check job status
squeue -u $USER

# View logs (created in logs/ at project root)
tail -f logs/<job_name>_<job_id>.out
tail -f logs/<job_name>_<job_id>.err
```

## Data Architecture

### Data Sources

**StatsBomb Open Data** (`data/statsbomb/`)
- Event data with corner kick events
- 360 freeze frame data with player positions at corner kick moments
- Download script: `scripts/download_statsbomb_corners.py`
- SLURM job: `scripts/slurm/download_statsbomb_corners.sh`
- Output: `data/statsbomb/corners_360.csv`

### Data Directory Structure
```
data/
└── statsbomb/              # StatsBomb data
    └── corners_360.csv     # Corner kicks with player positions
```

**Important**: All data directories are gitignored. Large datasets (videos, CSVs, JSONs) are never committed to git.

## Core Modules

### `src/statsbomb_loader.py`
Main module for loading and processing StatsBomb data.

**Key Classes**:
- `StatsBombCornerLoader`: Loads corner kick events and 360 player positioning data
  - `get_available_competitions()`: Fetch available competitions
  - `fetch_competition_events()`: Get all events for a competition
  - `build_corner_dataset()`: Build dataset with corner events and outcomes
  - `get_next_action()`: Analyze outcome after corner kick (shot, clearance, etc.)

**Usage Example**:
```python
from src.statsbomb_loader import StatsBombCornerLoader

loader = StatsBombCornerLoader(output_dir="data/statsbomb")
df = loader.build_corner_dataset(
    country="England",
    division="Premier League",
    season="2019/2020"
)
loader.save_dataset(df, "corners_epl_2019.csv")
```

## Key Scripts

### Data Download Scripts

**`scripts/download_statsbomb_corners.py`**
- Fast pandas-based StatsBomb 360 downloader
- Filters for professional men's competitions only (excludes youth/women's)
- Priority order: Champions League, La Liga, Premier League, etc.
- Extracts corners with 360 player position data
- Output: `data/statsbomb/corners_360.csv` with player positions (JSON format)
- Run via: `scripts/slurm/download_statsbomb_corners.sh`

**`scripts/visualize_corners_with_players.py`**
- Creates 2x2 grid visualization of corner kicks with player positions
- Blue = attacking team, Orange = defending team, Red star = corner kick
- Uses mplsoccer for pitch visualization
- Output: `data/statsbomb/corners_with_players_2x2.png`

**`scripts/visualize_single_corner.py`**
- Individual corner visualization script
- Professional broadcast style: grass pitch, red attacking, blue defending
- Cropped to attacking half (right side of pitch) for focused view
- Transparent players (70% alpha) for overlap visibility
- Dotted line trajectory with heat spot for ball landing
- Useful for quick testing or analyzing specific corners
- Output: `data/statsbomb/single_corner_<corner_id>.png`
- Run via: `scripts/slurm/visualize_single_corner.sh`

**`scripts/visualize_all_corners.py`**
- Batch generation of all corner kick visualizations
- Processes entire dataset (~1,118 corners)
- Same broadcast-style presentation as test script
- Output: `data/statsbomb/corner_images/corner_<corner_id>.png`
- Progress bar via tqdm
- Run via: `scripts/slurm/visualize_all_corners.sh`

### SLURM Scripts (`scripts/slurm/`)

All SLURM scripts follow this pattern:
```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/<name>_%j.out
#SBATCH --error=logs/<name>_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate robo
cd /home/mseo/CornerTactics
export PYTHONPATH="${PYTHONPATH}:/home/mseo/CornerTactics"

# Install dependencies
pip install <dependencies> --quiet

# Run script
python scripts/<script_name>.py
```

**Available SLURM scripts**:
- `download_statsbomb_corners.sh`: Download StatsBomb 360 corner data
- `visualize_corners_players.sh`: Create corner visualizations (2x2 grid)
- `visualize_single_corner.sh`: Generate single corner visualization with cropped view
- `visualize_all_corners.sh`: Batch generate all 1,118 corner visualizations (4 hours, 16GB RAM)
- `cleanup_archives.sh`: Clean up downloaded archive files

## Dependencies

Core dependencies (install as needed):
```bash
pip install statsbombpy pandas tqdm matplotlib mplsoccer requests
```

- **statsbombpy**: StatsBomb data access
- **pandas**: Data processing
- **tqdm**: Progress bars
- **matplotlib**: Plotting
- **mplsoccer**: Soccer pitch visualizations
- **requests**: HTTP requests for data download

## Development Workflow

### Adding New Data Sources

1. Create download script in `scripts/`
2. Create corresponding SLURM script in `scripts/slurm/`
3. Test locally first with small subset
4. Submit SLURM job for full download
5. Add output paths to `.gitignore` if needed

### Working with Corner Data

The main data structure for corners includes:
- Match metadata (competition, season, teams, date)
- Corner event (minute, second, team, player)
- Corner location (x, y coordinates on 120x80 StatsBomb pitch)
- Pass end location (where ball was aimed)
- Player positions (attacking/defending positions in JSON format)
- Outcome analysis (next action: shot, clearance, interception, etc.)

### Coordinate Systems

**StatsBomb Pitch**: 120x80 units
- X: 0 (defensive end) to 120 (attacking end)
- Y: 0 (bottom) to 80 (top)
- Corners typically at (120, 0) or (120, 80)

## Code Philosophy

Think like John Carmack when writing code:
- Prefer data-oriented design with pandas operations over loops
- Use efficient batch processing for large datasets
- Minimize memory allocation and data copying
- Write clear, straightforward code without unnecessary abstractions
- Comment complex logic, especially event matching and coordinate transformations

## Git Workflow

**Main branch**: `main`

**Current branch**: `gsr-corner-clip-poc`

**Commit message style** (based on recent commits):
- Descriptive, action-oriented messages
- Examples: "Add StatsBomb integration for corner kick outcome prediction"
- Keep messages concise but informative

## Important Notes

1. **Large Files**: Never commit videos (*.mp4), large CSVs, or model files. All data paths are gitignored.

2. **SLURM Jobs**: Always test scripts locally with small data samples before submitting long-running SLURM jobs.

3. **Data Paths**: Use absolute paths in SLURM scripts (`/home/mseo/CornerTactics`), but relative paths in Python code for portability.

4. **Player Position Data**: StatsBomb 360 freeze frames are stored as JSON strings in the CSV. Parse with `json.loads()` when needed.

5. **Conda Environment**: The `robo` environment should be activated for all work. Dependencies are installed dynamically in SLURM scripts to ensure consistency.

6. **Log Files**: SLURM logs are created in `logs/` at the project root (e.g., `#SBATCH --output=logs/sb_corners_%j.out` creates `/home/mseo/CornerTactics/logs/sb_corners_<job_id>.out`). Always check both .out and .err files when debugging.
