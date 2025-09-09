# CornerTactics - AI Context

## Project Overview
CornerTactics is a soccer corner kick analysis toolkit that processes SoccerNet dataset to extract and analyze corner kick events from professional soccer matches.

## Core Pipeline
1. **Download**: Get SoccerNet data (annotations + videos)
2. **Extract**: Create video clips of corner kicks
3. **Analyze**: Generate statistics and label outcomes

## Key Technical Details

### Data Format
- **Annotations**: Labels-v2.json files containing match events with timestamps
- **Videos**: 1.mkv (first half), 2.mkv (second half), 398x224 resolution
- **Game paths**: Format like "england_epl/2015-2016/2015-11-07 - 18-00 Team1 X - Y Team2"

### Corner Kick Events
- Identified by label "Corner" in annotations
- Include metadata: game time, team, visibility
- Typical match has 5-10 corner kicks

### Video Extraction
- Default: 30-second clips (10s before, 20s after corner)
- Output: MP4 files named "corner_[half]H_[time]_[team].mp4"
- Uses FFmpeg for video processing

## Important Limitations

1. **SoccerNet API**: Downloads entire splits, cannot limit number of games
2. **Video Quality**: 398x224 pixels (intentionally low for research)
3. **Storage**: Each game ~400MB with videos

## Code Structure
- `main.py`: Entry point, runs complete pipeline
- `src/data_loader.py`: SoccerNet download and data loading
- `src/corner_extractor.py`: Video clip extraction
- `src/analyzer.py`: Statistical analysis and outcome labeling

## Common Tasks

### Run complete analysis
```bash
python main.py "england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom"
```

### Extract specific duration clips
```python
extractor.extract_all(duration=20, before=5)  # 20s clips, 5s before corner
```

### Analyze multiple games
```python
for game in loader.list_games():
    df = analyzer.analyze_game(game)
```

## Development Notes
- Keep code simple and focused
- No unnecessary abstractions
- Clear error messages for missing data
- Preserve exact game path formats from SoccerNet