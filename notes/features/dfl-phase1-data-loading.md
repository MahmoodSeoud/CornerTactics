# DFL Phase 1: Data Loading & Validation

## Overview
Implement Phase 1 of the DFL GNN pipeline: load DFL tracking data, count corners, extract one complete corner as proof of concept.

## Goals
1. Load DFL tracking data using kloppy
2. Load event data and count corners across all 7 matches
3. Extract tracking frames around corner kick events (-2s to +6s)
4. Compute velocity vectors using central difference
5. Visualize a corner kick with player positions and velocity arrows

## Design Decisions

### Module Structure
Using `src/dfl/` module structure as decided:
```
src/dfl/
├── __init__.py
├── data_loading.py     # kloppy data loading utilities
└── visualization.py    # Corner visualization functions
```

### Key Dependencies
- `kloppy>=3.0` - Standard library for loading multi-vendor tracking/event data
- Uses `sportec` loader for DFL data (DFL uses Sportec format)

## Progress
- [x] Install kloppy (already installed: v3.17.0)
- [x] Download DFL open data (figshare API, ~2.5GB)
- [x] Write data loading module with tests (17 tests)
- [x] Count corners across 7 matches
- [x] Extract corner sequences with velocities
- [x] Create visualization

## Implementation Notes

### Data Download
- DFL data from figshare (doi.org/10.6084/m9.figshare.28196177)
- Direct browser download blocked by AWS WAF
- Created `scripts/download_dfl_data.py` using figshare API

### Data Statistics
- **7 matches** from Bundesliga 1st and 2nd divisions
- **57 total corners** (exceeds 50 corner threshold)
- Average **8.1 corners per match**
- Per-match breakdown:
  - J03WMX (Bayern vs Köln): 10 corners
  - J03WN1 (Leverkusen vs Bochum): 7 corners
  - J03WOH (Düsseldorf vs Regensburg): 6 corners
  - J03WOY (Düsseldorf vs Rostock): 2 corners
  - J03WPY (Düsseldorf vs Nürnberg): 11 corners
  - J03WQQ (Düsseldorf vs St. Pauli): 15 corners
  - J03WR9 (Düsseldorf vs Kaiserslautern): 6 corners

### Data Format
- Tracking: 25fps, 22 players per frame
- Metrica also available (35 players including substitutes)
- Timestamps are relative to period start (need period filtering)

### Velocity Computation
- Central difference: v(t) = (pos(t+1) - pos(t-1)) / (2*dt)
- Reasonable velocities observed (0-12 m/s range)

## Files Created
- `src/dfl/__init__.py` - Module exports
- `src/dfl/data_loading.py` - Loading functions
- `src/dfl/visualization.py` - Pitch plotting
- `tests/dfl/__init__.py` - Test package
- `tests/dfl/test_data_loading.py` - 14 tests
- `tests/dfl/test_visualization.py` - 3 tests
- `scripts/download_dfl_data.py` - Figshare downloader
- `scripts/slurm/download_dfl.sbatch` - SLURM job script
- `scripts/phase1_demo.py` - Phase 1 demo script
- `results/phase1/corner_visualization.png` - Output visualization

## Decision Point
With 57 corners (>50 threshold), proceed to Phase 2: Graph Construction Pipeline.
