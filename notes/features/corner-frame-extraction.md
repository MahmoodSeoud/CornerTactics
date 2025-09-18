# Corner Frame Extraction Feature

## Goal
Extract single frames at the exact timestamp when each corner kick is taken from SoccerNet broadcast videos.

## Implementation Status: ✅ COMPLETED

### Components Implemented
1. **CornerFrameExtractor** - Core frame extraction logic using ffmpeg
2. **CornerFramePipeline** - Batch processing pipeline for all games
3. **CLI Interface** - Command-line tool for easy usage
4. **Comprehensive Tests** - Full test coverage for all components

## Usage

### Command Line Interface
```bash
# Extract frames from all games in data directory
python src/cli.py --data-dir ./data

# Extract frames with custom output CSV
python src/cli.py --data-dir ./data --output ./corner_frames.csv
```

### Programmatic Usage
```python
from corner_frame_pipeline import CornerFramePipeline

pipeline = CornerFramePipeline("./data")
csv_path = pipeline.extract_all_corners()
```

## Technical Implementation
- **Frame extraction**: ffmpeg with precise timestamp positioning
- **Time parsing**: Supports both v2 ("1 - 05:30") and v3 ("05:30") formats
- **Error handling**: Graceful handling of missing videos/invalid timestamps
- **Progress tracking**: Detailed logging with extraction statistics
- **Output structure**: Organized frame storage in soccernet_corner_frames/

## Expected Output
- ~4,836 corner frames (1 per corner)
- Frame filenames: `{game}_{half}_{minutes}m{seconds}s_{team}.jpg`
- Metadata CSV with columns: game_path, game_time, half, team, visibility, frame_path
- Total storage: ~1GB (vs 300GB for full videos)
- Success rate: Typically 95%+ with proper video files

## Next Steps for Phase 2
- Use extracted frames for static formation analysis
- Apply YOLOv8 detection to identify player positions
- Select subset of corners for 3D trajectory analysis with SoccerNet-v3D