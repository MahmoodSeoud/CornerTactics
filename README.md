# CornerTactics

Clean, simple soccer corner kick analysis pipeline that extracts 20-second video clips from SoccerNet broadcast videos around corner kick moments.

## Status 🎯 READY FOR VIDEO EXTRACTION

- **4,826 corner events identified** from 500 SoccerNet games
- **Ready to extract 20-second video clips** around each corner kick
- **Clean, Carmack-style codebase** - simple functions that do one thing well
- **HPC-ready** with working SLURM scripts

## Quick Start

```bash
# Download SoccerNet dataset (HPC cluster)
sbatch scripts/slurm/download_data.sh

# Download GSR gamestate data (HPC cluster)
sbatch scripts/slurm/download_gsr.sh

# Unzip GSR gamestate data
sbatch scripts/slurm/unzip_gsr_data.sh

# Extract corner frames (HPC cluster)
sbatch scripts/slurm/extract_corner_frames.sh

# Or run locally
python scripts/extract_corners.py --data-dir ./data
```

## Architecture

**Clean and simple** - following John Carmack's principles:

```
CornerTactics/
├── scripts/
│   ├── extract_corners.py          # Main script (Carmack-refactored)
│   └── slurm/                      # HPC job scripts
│       ├── download_data.sh        # SoccerNet dataset download
│       ├── download_gsr.sh         # GSR gamestate data download
│       ├── unzip_gsr_data.sh       # Unzip GSR gamestate data
│       └── extract_corner_frames.sh # Corner frame extraction
└── src/                            # Library code
    ├── data_loader.py              # Load games and corner events
    └── download_soccernet.py       # SoccerNet dataset downloader
```

**Key principles:**
- Each function does one thing well
- Linear data flow, no deep nesting
- Clear variable names, no magic strings
- Early returns, obvious error handling

## Output

**20-second video clips around corner kicks:**

```
data/
├── datasets/soccernet/
│   ├── videos/                     # 720p broadcast videos
│   │   ├── england_epl/
│   │   ├── europe_uefa-champions-league/
│   │   └── france_ligue-1/
│   └── corner_clips/               # 🎯 Ready to extract 4,826 clips
│       ├── visible/                # High-quality visible corners (4,221 clips)
│       └── not_shown/              # Lower quality corners (605 clips)
└── insights/
    └── corner_clips_metadata.csv   # Complete metadata with paths
```

**What you get:**
- 20-second MP4 clips (H.264/AAC) around each corner kick
- Start 5 seconds before corner, end 15 seconds after
- 4,826 total clips from 500 games
- 87.5% are high-quality "visible" corners perfect for tactical analysis
- Complete player movements, set pieces, and outcomes

## External Dependencies

**Current approach: Clone & Ignore**
```bash
# Clone external repos locally (not tracked in git)
git clone https://github.com/SoccerNet/sn-gamestate.git
git clone https://github.com/SoccerNet/SoccerNet-v3.git
```

**Why this approach:**
- Simple setup for research projects
- Fast iteration, allows local modifications
- No complex submodule workflows

**External repositories:**
- `sn-gamestate/` - Advanced player tracking and visualization pipeline
- `SoccerNet-v3/` - Dataset tools and utilities

*For production use or team collaboration, consider git submodules. See CLAUDE.md for details.*

## Requirements

- Python 3.11+ (conda environment recommended)
- ffmpeg (for video frame extraction)
- SoccerNet dataset access

## Next Steps

1. **Extract Video Clips**: Run the extraction to get 4,826 corner video clips
2. **Player Tracking**: Use `sn-gamestate/` for frame-by-frame player tracking
3. **Tactical Analysis**: Analyze corner kick formations, player movements, and outcomes
4. **Machine Learning**: Train models on corner kick success patterns and defensive strategies

*See CLAUDE.md for detailed technical information and implementation notes.*