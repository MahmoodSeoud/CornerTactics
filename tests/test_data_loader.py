#!/usr/bin/env python3
"""
Tests for SoccerNetDataLoader - focused on data access functionality only.
Tests should verify that the class can load annotations, list games, and extract corner events.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import SoccerNetDataLoader


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with mock game structure."""
    # Create the full soccernet_videos path that data_loader expects
    videos_dir = tmp_path / "datasets" / "soccernet" / "soccernet_videos"
    game_path = videos_dir / "england_epl" / "2015-2016" / "2015-11-07 - 18-00 Manchester United 2 - 0 West Brom"
    game_path.mkdir(parents=True)
    
    # Create mock Labels-v3.json (v3 format has actions dict with imageMetadata)
    labels = {
        "GameMetadata": {"num_actions": 3},
        "actions": {
            "5.png": {
                "imageMetadata": {
                    "gameTime": "1 - 5:23",
                    "label": "Corner",
                    "half": 1,
                    "visibility": "visible"
                }
            },
            "12.png": {
                "imageMetadata": {
                    "gameTime": "2 - 12:45",
                    "label": "Corner",
                    "half": 2,
                    "visibility": "visible"
                }
            },
            "10.png": {
                "imageMetadata": {
                    "gameTime": "1 - 10:30",
                    "label": "Foul",
                    "half": 1,
                    "visibility": "visible"
                }
            }
        }
    }

    with open(game_path / "Labels-v3.json", 'w') as f:
        json.dump(labels, f)
    
    # Create mock video files
    (game_path / "1_720p.mkv").touch()
    (game_path / "2_720p.mkv").touch()
    
    return tmp_path


class TestSoccerNetDataLoader:
    """Test data access functionality of SoccerNetDataLoader."""
    
    def test_init_creates_data_directory(self, tmp_path):
        """Test that DataLoader creates data directory if it doesn't exist."""
        data_dir = tmp_path / "new_data_dir"
        loader = SoccerNetDataLoader(str(data_dir))
        
        assert data_dir.exists()
        assert loader.data_dir == data_dir
    
    def test_list_games_returns_games_with_videos(self, temp_data_dir):
        """Test that list_games finds games with video files."""
        loader = SoccerNetDataLoader(str(temp_data_dir))
        games = loader.list_games()
        
        assert len(games) == 1
        expected_path = "england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom"
        assert expected_path in games[0]
    
    def test_list_games_empty_when_no_videos(self, tmp_path):
        """Test that list_games returns empty list when no video files exist."""
        # Create game directory without videos but in the right structure
        videos_dir = tmp_path / "datasets" / "soccernet" / "soccernet_videos"
        game_path = videos_dir / "league" / "season" / "game"
        game_path.mkdir(parents=True)
        (game_path / "Labels-v3.json").touch()

        loader = SoccerNetDataLoader(str(tmp_path))
        games = loader.list_games()

        assert games == []
    
    def test_load_annotations_success(self, temp_data_dir):
        """Test loading annotations from existing Labels-v3.json file."""
        loader = SoccerNetDataLoader(str(temp_data_dir))
        game_path = "datasets/soccernet/soccernet_videos/england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom"

        annotations = loader.load_annotations(game_path)

        assert "actions" in annotations
        assert len(annotations["actions"]) == 3
    
    def test_load_annotations_file_not_found(self, temp_data_dir):
        """Test that load_annotations raises FileNotFoundError for missing files."""
        loader = SoccerNetDataLoader(str(temp_data_dir))
        
        with pytest.raises(FileNotFoundError):
            loader.load_annotations("non/existent/game")
    
    def test_get_corner_events_extracts_corners_only(self, temp_data_dir):
        """Test that get_corner_events extracts only corner kick events."""
        loader = SoccerNetDataLoader(str(temp_data_dir))
        game_path = "datasets/soccernet/soccernet_videos/england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom"

        corners = loader.get_corner_events(game_path)
        
        # Should have 2 corners (not the 1 foul)
        assert len(corners) == 2
        
        # Check first corner
        assert corners[0]['gameTime'] == "1 - 5:23"
        assert corners[0]['team'] == "unknown"  # Team info not available in v3
        assert corners[0]['half'] == 1  # Should be integer, not string
        assert corners[0]['visibility'] == "visible"

        # Check second corner
        assert corners[1]['gameTime'] == "2 - 12:45"
        assert corners[1]['team'] == "unknown"  # Team info not available in v3
        assert corners[1]['half'] == 2  # Should be integer, not string
    
    def test_get_corner_events_no_corners(self, tmp_path):
        """Test get_corner_events returns empty list when no corners exist."""
        videos_dir = tmp_path / "datasets" / "soccernet" / "soccernet_videos"
        game_path = videos_dir / "league" / "season" / "game"
        game_path.mkdir(parents=True)
        
        # Create annotations with no corners (v3 format)
        labels = {
            "actions": {
                "5.png": {
                    "imageMetadata": {
                        "gameTime": "1 - 5:23",
                        "label": "Foul",
                        "half": 1
                    }
                }
            }
        }
        with open(game_path / "Labels-v3.json", 'w') as f:
            json.dump(labels, f)
        
        loader = SoccerNetDataLoader(str(tmp_path))
        corners = loader.get_corner_events("datasets/soccernet/soccernet_videos/league/season/game")
        
        assert corners == []
    
    def test_data_loader_should_not_have_download_methods(self):
        """Test that SoccerNetDataLoader does not have download methods after refactor."""
        loader = SoccerNetDataLoader("dummy_path")
        
        # These methods should NOT exist after refactor
        assert not hasattr(loader, 'download_broadcast_videos')
        assert not hasattr(loader, 'download_tracklets')
        
        # These methods SHOULD exist (data access)
        assert hasattr(loader, 'load_annotations')
        assert hasattr(loader, 'list_games')
        assert hasattr(loader, 'get_corner_events')