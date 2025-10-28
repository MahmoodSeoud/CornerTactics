# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CornerTactics is a soccer analytics research project focused on predicting corner kick outcomes using Graph Neural Networks (GNNs). The project implements the methodology from Bekkers & Sahasrabudhe (2024) "A Graph Neural Network Deep-Dive into Successful Counterattacks", applying it to corner kick scenarios with StatsBomb 360 freeze frames, SkillCorner tracking data, and SoccerNet videos.

**Current Status**: Phase 2 Complete + Temporal Augmentation - Ready for Phase 3 Training

**Dataset**: 7,369 temporally augmented graphs (6.6× increase from original 1,118 corners)

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

### Project File Tree
```
CornerTactics/
├── CLAUDE.md                          # This file - project guide
├── README.md                          # Project README
├── requirements.txt                   # Python dependencies
├── CORNER_GNN_PLAN.md                # Master implementation plan (MOVED TO docs/)
│
├── docs/                              # 📚 Documentation
│   ├── CORNER_GNN_PLAN.md            # Master GNN implementation plan
│   ├── DATA_FEASIBILITY_ANALYSIS.md  # Data source analysis
│   └── PROJECT_STATUS.md             # Current project status
│
├── src/                               # 🔧 Core Modules
│   ├── statsbomb_loader.py           # StatsBomb data loading
│   ├── outcome_labeler.py            # Corner outcome labeling (Phase 1.2)
│   ├── feature_engineering.py        # Node feature extraction (Phase 2.1)
│   └── graph_builder.py              # Adjacency matrix construction (Phase 2.2)
│
├── scripts/                           # 📜 Execution Scripts
│   ├── download_statsbomb_corners.py      # Download StatsBomb 360 data
│   ├── extract_skillcorner_corners.py     # Extract SkillCorner corners
│   ├── extract_soccernet_corners.py       # Extract SoccerNet corners
│   ├── integrate_corner_datasets.py       # Unify all datasets (Phase 1.1)
│   ├── label_statsbomb_outcomes.py        # Label StatsBomb outcomes
│   ├── label_skillcorner_outcomes.py      # Label SkillCorner outcomes
│   ├── label_soccernet_outcomes.py        # Label SoccerNet outcomes
│   ├── extract_corner_features.py         # Extract node features (Phase 2.1)
│   ├── build_graph_dataset.py             # Build graphs (Phase 2.2)
│   ├── extract_skillcorner_temporal.py    # Extract temporal SkillCorner features (Phase 2.3)
│   ├── augment_statsbomb_temporal.py      # Temporal augmentation for StatsBomb (Phase 2.4)
│   ├── train_gnn.py                       # Train GNN model (Phase 3)
│   ├── visualize_graph_structure.py       # Visualize adjacency matrices
│   ├── test_feature_extraction.py         # Testing utilities
│   │
│   ├── slurm/                             # SLURM Job Scripts
│   │   ├── phase1_1_complete.sh          # Phase 1.1: Data integration
│   │   ├── phase1_2_label_outcomes.sh    # Phase 1.2: Outcome labeling
│   │   ├── phase2_1_extract_features.sh  # Phase 2.1: Node features
│   │   ├── phase2_2_build_graphs.sh      # Phase 2.2: Graph construction
│   │   ├── phase2_3_skillcorner_temporal.sh  # Phase 2.3: SkillCorner temporal
│   │   ├── phase2_4_statsbomb_augment.sh     # Phase 2.4: StatsBomb augmentation
│   │   └── phase3_train_gnn.sh           # Phase 3: GNN training
│   │
│   └── visualization/                     # Visualization Scripts
│       ├── visualize_corners_with_players.py
│       ├── visualize_single_corner.py
│       ├── visualize_all_corners.py
│       └── *.sh (SLURM wrappers)
│
├── data/                              # 💾 Data Directory (gitignored)
│   ├── raw/                          # Original downloaded data
│   │   ├── statsbomb/               # StatsBomb 360 freeze frames
│   │   ├── skillcorner/             # SkillCorner tracking data
│   │   ├── soccernet/               # SoccerNet videos
│   │   └── soccersynth/             # Synthetic data
│   │
│   ├── processed/                    # Unified datasets
│   │   ├── unified_corners_dataset.csv
│   │   └── unified_corners_dataset.parquet
│   │
│   ├── features/                     # Extracted features
│   │   ├── node_features/            # Phase 2.1: Single frame features
│   │   │   ├── statsbomb_player_features.parquet
│   │   │   └── statsbomb_player_features.csv
│   │   └── temporal/                 # Phase 2.3: Temporal features
│   │       ├── skillcorner_temporal_features.parquet
│   │       └── skillcorner_temporal_features.csv
│   │
│   ├── graphs/                       # Graph datasets
│   │   └── adjacency_team/           # Team-based adjacency strategy
│   │       ├── statsbomb_graphs.pkl  # Original single-frame (Phase 2.2)
│   │       ├── statsbomb_temporal_augmented.pkl  # Temporal augmented (Phase 2.4)
│   │       └── graph_statistics.json
│   │
│   └── results/                      # Analysis outputs
│       ├── statsbomb/               # StatsBomb visualizations
│       ├── skillcorner/             # SkillCorner results
│       ├── graphs/                  # Graph structure visualizations
│       └── unified/                 # Unified analysis
│
└── logs/                              # 📋 SLURM Job Logs
    ├── phase1_1_*.out/err
    ├── phase1_2_*.out/err
    ├── phase2_1_*.out/err
    └── phase2_2_*.out/err
```

**Important**: All data directories are gitignored. Large datasets (videos, CSVs, JSONs) are never committed to git.

### Data Sources

**StatsBomb Open Data** (`data/raw/statsbomb/`)
- Event data with corner kick events
- 360 freeze frame data with player positions at corner kick moments
- Download script: `scripts/download_statsbomb_corners.py`
- SLURM job: `scripts/slurm/download_statsbomb_corners.sh`
- Output: `data/raw/statsbomb/corners_360.csv`

**SkillCorner Data** (`data/raw/skillcorner/`)
- Player tracking data with x,y coordinates
- Output: Processed tracking data files

**SoccerNet Data** (`data/raw/soccernet/`)
- Video footage and event labels
- Corner clip extraction and frame processing
- Size: ~1.1TB of video data

**Unified Dataset** (`data/processed/`)
- Combined corner kick data from all sources
- Files: `unified_corners_dataset.csv`, `unified_corners_dataset.parquet`

## Core Modules (`src/`)

### `src/statsbomb_loader.py`
**Phase**: Data loading foundation
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

loader = StatsBombCornerLoader(output_dir="data/raw/statsbomb")
df = loader.build_corner_dataset(
    country="England",
    division="Premier League",
    season="2019/2020"
)
loader.save_dataset(df, "corners_epl_2019.csv")
```

### `src/outcome_labeler.py`
**Phase 1.2**: Outcome labeling
Labels corner kick outcomes by analyzing subsequent events.

**Key Classes**:
- `OutcomeLabeler`: Analyzes events following corner kicks
  - `label_corner_outcome()`: Classify outcome (goal/shot/clearance/possession)
  - `calculate_xthreat()`: Compute expected threat value
  - `get_temporal_features()`: Extract time/events to outcome

**Outcome Categories**:
- `goal`: Goal scored within 20 seconds
- `shot`: Shot attempt (no goal)
- `clearance`: Defensive clearance
- `second_corner`: Another corner awarded
- `possession`: Attacking team retains possession
- `opposition_possession`: Defending team gains possession

**Usage Example**:
```python
from src.outcome_labeler import OutcomeLabeler

labeler = OutcomeLabeler()
outcome = labeler.label_corner_outcome(events_df, corner_event_id)
print(f"Outcome: {outcome['category']}, Goal: {outcome['goal_scored']}")
```

### `src/feature_engineering.py`
**Phase 2.1**: Node feature engineering
Extracts 14-dimensional feature vectors for each player in corner scenarios.

**Key Classes**:
- `FeatureEngineer`: Main feature extraction class
- `PlayerFeatures`: Dataclass for 14-dimensional feature vectors

**Feature Dimensions (14 total)**:
1. **Spatial (4)**: x, y, distance_to_goal, distance_to_ball_target
2. **Kinematic (4)**: vx, vy, velocity_magnitude, velocity_angle
3. **Contextual (4)**: angle_to_goal, angle_to_ball, team_flag, in_penalty_box
4. **Density (2)**: num_players_within_5m, local_density_score

**Usage Example**:
```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.extract_features_from_statsbomb_corner(corner_row)
features_df = engineer.features_to_dataframe(features)
```

### `src/graph_builder.py`
**Phase 2.2**: Adjacency matrix construction
Converts player features into graph representations with various connectivity strategies.

**Key Classes**:
- `GraphBuilder`: Main graph construction class
- `CornerGraph`: Complete graph representation dataclass
- `EdgeFeatures`: 6-dimensional edge feature vectors

**Adjacency Strategies (5 types)**:
1. **team**: Connect teammates only (paper baseline)
2. **distance**: Connect players within 10m
3. **delaunay**: Spatial triangulation
4. **ball_centric**: Focus on ball landing zone
5. **zone**: Tactical zone-based

**Edge Features (6 dimensions)**:
- Normalized distance between players
- Relative velocity (x, y components)
- Relative velocity magnitude
- Angle between players (sine, cosine)

**Usage Example**:
```python
from src.graph_builder import GraphBuilder

builder = GraphBuilder(adjacency_strategy='team')
graph = builder.build_graph_from_features(features_df, corner_id)
print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
```

## Key Scripts

### Phase 1: Data Integration & Labeling

**`scripts/download_statsbomb_corners.py`**
- Fast pandas-based StatsBomb 360 downloader
- Filters for professional men's competitions only (excludes youth/women's)
- Priority order: Champions League, La Liga, Premier League, etc.
- Extracts corners with 360 player position data
- Output: `data/raw/statsbomb/corners_360.csv` with player positions (JSON format)
- Run via: `scripts/slurm/download_statsbomb_corners.sh`

**`scripts/extract_skillcorner_corners.py`**
- Extracts corner events from SkillCorner dynamic_events.csv
- Matches with tracking data (10fps continuous)
- Output: Corner events with tracking references

**`scripts/extract_soccernet_corners.py`**
- Extracts corner clips from SoccerNet dataset
- Links video clips to tracking data
- Output: Corner events with video references

**`scripts/integrate_corner_datasets.py` (Phase 1.1)**
- Combines StatsBomb, SkillCorner, and SoccerNet data
- Creates unified corner dataset
- Output: `data/processed/unified_corners_dataset.parquet`
- Run via: `sbatch scripts/slurm/phase1_1_complete.sh`

**`scripts/label_statsbomb_outcomes.py` (Phase 1.2)**
- Labels corner outcomes for StatsBomb data
- Analyzes subsequent events (15-20 second window)
- Classifies: Goal/Shot/Clearance/Possession
- Output: Corners with outcome labels

**`scripts/label_skillcorner_outcomes.py` (Phase 1.2)**
- Labels corner outcomes for SkillCorner data
- Uses tracking data and dynamic events
- Output: Corners with outcome labels

**`scripts/label_soccernet_outcomes.py` (Phase 1.2)**
- Labels corner outcomes for SoccerNet data
- Uses video labels and action spotting
- Output: Corners with outcome labels

### Phase 2: Graph Construction

**`scripts/extract_corner_features.py` (Phase 2.1)**
- Extracts 14-dimensional node features for all players
- Processes both StatsBomb 360 and SkillCorner tracking
- Features: spatial, kinematic, contextual, density
- Output: `data/features/node_features/statsbomb_player_features.parquet`
- Run via: `sbatch scripts/slurm/phase2_1_extract_features.sh`

**`scripts/build_graph_dataset.py` (Phase 2.2)**
- Builds graph representations from node features
- Supports 5 adjacency strategies (team/distance/delaunay/ball_centric/zone)
- Computes 6-dimensional edge features
- Output: `data/graphs/adjacency_<strategy>/<dataset>_graphs.pkl`
- Run via: `sbatch scripts/slurm/phase2_2_build_graphs.sh`
- CLI: `python scripts/build_graph_dataset.py --strategy team --dataset all`

**`scripts/visualize_graph_structure.py` (Phase 2.2 Debug)**
- Visualizes adjacency matrices overlaid on pitch
- Compares all 5 strategies side-by-side
- Professional broadcast-style rendering
- Output: `data/results/graphs/strategy_comparison_<corner_id>.png`
- CLI: `python scripts/visualize_graph_structure.py --strategy all --num-samples 1`

### Visualization Scripts

**`scripts/visualization/visualize_corners_with_players.py`**
- Creates 2x2 grid visualization of corner kicks with player positions
- Blue = attacking team, Orange = defending team, Red star = corner kick
- Uses mplsoccer for pitch visualization
- Output: `data/results/statsbomb/corners_with_players_2x2.png`

**`scripts/visualization/visualize_single_corner.py`**
- Individual corner visualization script
- Professional broadcast style: grass pitch, red attacking, blue defending
- Cropped to attacking half (right side of pitch) for focused view
- Transparent players (70% alpha) for overlap visibility
- Dotted line trajectory with heat spot for ball landing
- Useful for quick testing or analyzing specific corners
- Output: `data/results/statsbomb/single_corner_<corner_id>.png`
- Run via: `scripts/visualization/visualize_single_corner.sh`

**`scripts/visualization/visualize_all_corners.py`**
- Batch generation of all corner kick visualizations
- Processes entire dataset (~1,118 corners)
- Same broadcast-style presentation as test script
- Output: `data/results/statsbomb/corner_images/corner_<corner_id>.png`
- Progress bar via tqdm
- Run via: `scripts/visualization/visualize_all_corners.sh`

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

**Phase 1 - Data Pipeline**:
- `phase1_1_complete.sh`: Data integration (StatsBomb + SkillCorner + SoccerNet)
- `phase1_2_label_outcomes.sh`: Outcome labeling for all datasets

**Phase 2 - Graph Construction & Augmentation**:
- `phase2_1_extract_features.sh`: Node feature extraction (14-dim)
- `phase2_2_build_graphs.sh`: Graph dataset construction with adjacency matrices
- `phase2_3_skillcorner_temporal.sh`: SkillCorner temporal feature extraction (10fps tracking)
- `phase2_4_statsbomb_augment.sh`: StatsBomb temporal augmentation (US Soccer Fed approach)

**Phase 3 - GNN Training**:
- `phase3_train_gnn.sh`: Train GNN model on augmented dataset

**Visualization**:
- `visualize_corners_players.sh`: Create corner visualizations (2x2 grid)
- `visualize_single_corner.sh`: Generate single corner visualization with cropped view
- `visualize_all_corners.sh`: Batch generate all 1,118 corner visualizations (4 hours, 16GB RAM)

## Dependencies

Core dependencies (install as needed):
```bash
# Phase 1: Data pipeline
pip install statsbombpy pandas tqdm matplotlib mplsoccer requests

# Phase 2: Graph construction
pip install numpy scipy

# Phase 3: GNN training (upcoming)
pip install torch torch-geometric spektral tensorflow
```

**Key Libraries**:
- **statsbombpy**: StatsBomb data access
- **pandas**: Data processing and analysis
- **numpy**: Numerical operations
- **scipy**: Sparse matrices, spatial algorithms (Delaunay)
- **tqdm**: Progress bars
- **matplotlib**: Plotting and visualization
- **mplsoccer**: Soccer pitch visualizations
- **requests**: HTTP requests for data download
- **torch/torch-geometric**: PyTorch + graph neural networks (Phase 3)
- **spektral/tensorflow**: Alternative GNN framework (Phase 3)

## Development Workflow

### Adding New Data Sources

1. Create download script in `scripts/`
2. Create corresponding SLURM script in `scripts/slurm/`
3. **Save raw data to `data/raw/<source_name>/`**
4. **Save processed/unified data to `data/processed/`**
5. **Save analysis outputs to `data/results/<source_name>/`**
6. Test locally first with small subset
7. Submit SLURM job for full download
8. Add output paths to `.gitignore` if needed

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

## Documentation (`docs/`)

The `docs/` directory contains comprehensive project documentation:

**`docs/CORNER_GNN_PLAN.md`** (Primary Reference)
- Master implementation plan for GNN system
- Phase-by-phase breakdown (Phases 1-6)
- Task checklists with completion status
- Architecture specifications
- Success metrics and publication targets
- **Always check this for implementation roadmap**

**`docs/DATA_FEASIBILITY_ANALYSIS.md`**
- Analysis of available data sources
- Dataset size estimates
- Coverage and quality assessments
- Recommendations for data combinations

**`docs/PROJECT_STATUS.md`**
- Current project progress tracker
- Completed phases and deliverables
- Active work and blockers
- Next steps and priorities

**Usage**: When starting a new phase or task, always consult `CORNER_GNN_PLAN.md` first to understand requirements and dependencies.

## Project Status (as of October 23, 2024)

**Completed Phases**:
- ✅ Phase 1.1: Data Integration (1,118 StatsBomb corners + SkillCorner + SoccerNet)
- ✅ Phase 1.2: Outcome Labeling (goal/shot/clearance/possession)
- ✅ Phase 2.1: Node Feature Engineering (14-dim features, 21,231 players)
- ✅ Phase 2.2: Adjacency Matrix Construction (5 strategies, 6-dim edges)
- ✅ Phase 2.3: SkillCorner Temporal Extraction (1,555 temporal graphs from 317 corners)
- ✅ Phase 2.4: StatsBomb Temporal Augmentation (5,814 augmented graphs)
- ✅ Phase 3.0: GNN Model Implementation (PyTorch Geometric, 28k parameters)

**Current Phase**:
- ⏳ Phase 3.1: Re-training with Expanded Dataset (NEXT)

**Dataset Summary**:
- **Total Graphs**: 7,369 (6.6× increase from original)
  - StatsBomb Augmented: 5,814 graphs (5× temporal + mirrors)
  - SkillCorner Temporal: 1,555 graphs (real 10fps tracking)
- **Dangerous Situations**: ~1,261 (17.1% positive class)
  - Changed target from "goal" (1.3%) to "shot OR goal" for better balance
- **Temporal Augmentation**: US Soccer Federation approach
  - 5 temporal frames: t = -2s, -1s, 0s, +1s, +2s
  - Position perturbations + mirror augmentation

**Previous Training Results** (Original 1,118 corners, goal-only target):
- Best Val AUC: 0.765
- Test AUC: 0.271 (severe overfitting)
- Only 14 goals (1.3%) - extreme class imbalance

**CRITICAL FIX - Data Leakage (October 26, 2024)**:
- ✅ Fixed train/val/test split to prevent temporal frame leakage
- **Issue**: Previous splits randomly assigned temporal frames, allowing model to see same corner at different times across train/test
- **Fix**: Now splits by base corner ID (1,435 unique corners) instead of graphs (7,369 frames)
- **Impact**: All temporal frames from a corner now stay together in same split
- **Verification**: Zero overlap between splits confirmed (see `DATA_LEAKAGE_FIX_NOTES.md`)
- **File Modified**: `src/data_loader.py::get_split_indices()`
- **Test Script**: `scripts/test_split_fix.py`

**Next Immediate Tasks**:
1. ✅ Fix data leakage in train/val/test split
2. ⏳ Re-train with fixed splits (expect AUC 0.70-0.80, not inflated 0.95)
3. ⏳ Update data loader to use augmented dataset
4. ⏳ Re-train with "dangerous situation" target (shot OR goal)
5. ⏳ Evaluate on test set with 17% positive class
