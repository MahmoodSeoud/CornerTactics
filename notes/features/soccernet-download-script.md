# SoccerNet Download Script Feature

## Requirements
Create a download script for getting labels and videos from SoccerNet.

## Current Understanding
- SoccerNetDataLoader class already exists in `src/data_loader.py`
- Current functionality includes:
  - `download_annotations(split)` - Downloads all annotation files for a split
  - `download_videos(game_path)` - Downloads video files for a specific game
  - `load_annotations(game_path)` - Load annotations for a game
  - `list_games()` - List all games with annotations

## Task Clarification
The request is for a "download script" but we already have a data loader class. 
Possible interpretations:
1. Create a command-line script that uses the existing data loader
2. Enhance the existing data loader with additional functionality
3. Create a standalone script for bulk downloads

## Implementation Plan
Create a command-line script (`download_soccernet.py`) that provides an easy interface to:
1. Download all labels for specified leagues/splits
2. Download videos for all games or specific games
3. Show download progress and status
4. Handle errors gracefully

## Final Implementation
Created `src/download_soccernet.py` with:
- `SoccerNetDownloadScript` class wrapping `SoccerNetDataLoader`
- `download_all_labels(splits)` - download labels for multiple splits
- `download_videos_for_games(game_paths)` - download videos for specific games
- `download_all_videos()` - download videos for all games with labels
- Command-line interface with argparse supporting:
  - `--labels train test` - download labels for specific splits
  - `--videos game_path1 game_path2` - download videos for specific games
  - `--all-videos` - download videos for all games with labels
  - `--data-dir` and `--password` options

## Test Strategy
✅ Test script argument parsing
✅ Test integration with SoccerNetDataLoader using mocking
✅ Test CLI main function with argument mocking
✅ Test all core functionality with unit tests
- All tests passing with 100% coverage of implemented functionality

## Usage Examples
```bash
# Download labels for train and test splits
python src/download_soccernet.py --labels train test

# Download videos for specific games
python src/download_soccernet.py --videos "england_epl/2015-2016/game1" "spain_laliga/2015-2016/game2"

# Download videos for all games with labels
python src/download_soccernet.py --all-videos

# Show help
python src/download_soccernet.py --help
```