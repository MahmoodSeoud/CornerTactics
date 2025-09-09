# CornerTactics - AI Assistant Context

## Project Overview

Simple soccer corner kick analysis pipeline that processes ALL games automatically.

## Core Flow

```
Download Data → Run main.py → Get Results
```

1. **Download**: SoccerNet data (one-time setup)
2. **Extract**: Video clips from ALL games
3. **Analyze**: Corner kicks across ALL games
4. **Output**: Combined CSV with all data

## Key Files

- `main.py` (93 lines) - Runs complete pipeline on ALL games
- `src/data_loader.py` (65 lines) - Downloads SoccerNet data
- `src/corner_extractor.py` (93 lines) - Extracts corner clips
- `src/analyzer.py` (92 lines) - Analyzes corner events

Total: ~250 lines of clean code

## Technical Details

### Data Structure
```
data/
└── england_epl/
    └── 2015-2016/
        └── 2015-11-07 - 18-00 Manchester United 2 - 0 West Brom/
            ├── Labels-v2.json  # Match annotations
            ├── 1.mkv          # First half video
            └── 2.mkv          # Second half video
```

### Corner Detection
- Found in Labels-v2.json where `label == "Corner"`
- Includes: gameTime, team, visibility

### Video Extraction
- Default: 30-second clips (10s before, 20s after)
- Output: `corner_1H_28m46s_home.mp4`

## Common Commands

```bash
# Full pipeline
python main.py

# Analysis only (no video)
python main.py --no-clips

# Check what games exist
from src.data_loader import SoccerNetDataLoader
loader = SoccerNetDataLoader('data/')
games = loader.list_games()
```

## Important Notes

- Always processes ALL games (no single game option)
- SoccerNet API downloads entire splits (cannot limit)
- Video quality: 398x224 pixels (intentional)
- Each game ~400MB with videos

## Simplified Design Principles

- No unnecessary parameters
- No complex options
- Always batch process everything
- Clear progress output
- Single entry point (main.py)