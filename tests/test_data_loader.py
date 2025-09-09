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
    game_path = tmp_path / "england_epl" / "2015-2016" / "2015-11-07 - 18-00 Manchester United 2 - 0 West Brom"
    game_path.mkdir(parents=True)
    
    # Create mock Labels-v2.json
    labels = {
        "annotations": [
            {
                "gameTime": "1 - 5:23",
                "label": "Corner", 
                "team": "home",
                "visibility": "visible"
            },
            {
                "gameTime": "2 - 12:45", 
                "label": "Corner",
                "team": "away",
                "visibility": "visible"
            },
            {
                "gameTime": "1 - 10:30",
                "label": "Foul",
                "team": "home", 
                "visibility": "visible"
            }
        ]
    }
    
    with open(game_path / "Labels-v2.json", 'w') as f:
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
        # Create game directory without videos
        game_path = tmp_path / "league" / "season" / "game"
        game_path.mkdir(parents=True)
        (game_path / "Labels-v2.json").touch()
        
        loader = SoccerNetDataLoader(str(tmp_path))
        games = loader.list_games()
        
        assert games == []
    
    def test_load_annotations_success(self, temp_data_dir):
        """Test loading annotations from existing Labels-v2.json file."""
        loader = SoccerNetDataLoader(str(temp_data_dir))
        game_path = "england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom"
        
        annotations = loader.load_annotations(game_path)
        
        assert "annotations" in annotations
        assert len(annotations["annotations"]) == 3
    
    def test_load_annotations_file_not_found(self, temp_data_dir):
        """Test that load_annotations raises FileNotFoundError for missing files."""
        loader = SoccerNetDataLoader(str(temp_data_dir))
        
        with pytest.raises(FileNotFoundError):
            loader.load_annotations("non/existent/game")
    
    def test_get_corner_events_extracts_corners_only(self, temp_data_dir):
        """Test that get_corner_events extracts only corner kick events."""
        loader = SoccerNetDataLoader(str(temp_data_dir))
        game_path = "england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom"
        
        corners = loader.get_corner_events(game_path)
        
        # Should have 2 corners (not the 1 foul)
        assert len(corners) == 2
        
        # Check first corner
        assert corners[0]['gameTime'] == "1 - 5:23"
        assert corners[0]['team'] == "home"
        assert corners[0]['half'] == "1"
        assert corners[0]['visibility'] == "visible"
        
        # Check second corner 
        assert corners[1]['gameTime'] == "2 - 12:45"
        assert corners[1]['team'] == "away"
        assert corners[1]['half'] == "2"
    
    def test_get_corner_events_no_corners(self, tmp_path):
        """Test get_corner_events returns empty list when no corners exist."""
        game_path = tmp_path / "league" / "season" / "game"
        game_path.mkdir(parents=True)
        
        # Create annotations with no corners
        labels = {"annotations": [{"gameTime": "1 - 5:23", "label": "Foul", "team": "home"}]}
        with open(game_path / "Labels-v2.json", 'w') as f:
            json.dump(labels, f)
        
        loader = SoccerNetDataLoader(str(tmp_path))
        corners = loader.get_corner_events("league/season/game")
        
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