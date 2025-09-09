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

## Test Strategy
- Test script argument parsing
- Test integration with SoccerNetDataLoader
- Test error handling for network issues
- Test progress reporting