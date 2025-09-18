#!/usr/bin/env python3
"""
Tests for CornerFramePipeline - extracts frames for all corners and generates metadata CSV.
"""

import pytest
import json
import csv
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from corner_frame_pipeline import CornerFramePipeline


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with mock game structure."""
    # Create soccernet directory structure that matches data_loader expectations
    soccernet_dir = tmp_path / "datasets" / "soccernet" / "soccernet_videos"
    soccernet_dir.mkdir(parents=True)

    # Create first game
    game1_path = soccernet_dir / "england_epl" / "2015-2016" / "2015-11-07 - 18-00 Manchester United 2 - 0 West Brom"
    game1_path.mkdir(parents=True)

    # Create mock Labels-v3.json for game 1
    labels1 = {
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
            }
        }
    }
    with open(game1_path / "Labels-v3.json", 'w') as f:
        json.dump(labels1, f)

    # Create mock video files
    (game1_path / "1_720p.mkv").touch()
    (game1_path / "2_720p.mkv").touch()

    # Create second game
    game2_path = soccernet_dir / "france_ligue-1" / "2015-2016" / "2015-11-08 - 20-00 PSG 3 - 1 Marseille"
    game2_path.mkdir(parents=True)

    labels2 = {
        "actions": {
            "8.png": {
                "imageMetadata": {
                    "gameTime": "1 - 8:15",
                    "label": "Corner",
                    "half": 1,
                    "visibility": "visible"
                }
            }
        }
    }
    with open(game2_path / "Labels-v3.json", 'w') as f:
        json.dump(labels2, f)

    (game2_path / "1_720p.mkv").touch()
    (game2_path / "2_720p.mkv").touch()

    return tmp_path


class TestCornerFramePipeline:
    """Test CornerFramePipeline functionality."""

    def test_extract_all_corners_creates_frames_and_csv(self, temp_data_dir):
        """Test that extract_all_corners processes all games and creates metadata CSV."""
        # This test should fail initially since CornerFramePipeline doesn't exist yet
        pipeline = CornerFramePipeline(str(temp_data_dir))

        # Mock the frame extraction to avoid actual ffmpeg calls
        with patch('corner_frame_pipeline.CornerFrameExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor

            # Mock successful frame extractions
            mock_extractor.extract_corner_frame.side_effect = [
                "/path/to/frame1.jpg",  # First corner
                "/path/to/frame2.jpg",  # Second corner
                "/path/to/frame3.jpg"   # Third corner
            ]

            # Run the pipeline
            result_csv = pipeline.extract_all_corners()

            # Should create CSV file
            assert Path(result_csv).exists()

            # Should call extract_corner_frame for each corner (3 total)
            assert mock_extractor.extract_corner_frame.call_count == 3

            # Check CSV content
            with open(result_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3

            # Check first row
            assert rows[0]['game_path'].endswith('Manchester United 2 - 0 West Brom')
            assert rows[0]['game_time'] == '1 - 5:23'
            assert rows[0]['half'] == '1'
            assert rows[0]['frame_path'] == '/path/to/frame1.jpg'
            assert rows[0]['visibility'] == 'visible'

            # Check second row
            assert rows[1]['game_path'].endswith('Manchester United 2 - 0 West Brom')
            assert rows[1]['game_time'] == '2 - 12:45'
            assert rows[1]['half'] == '2'

            # Check third row (PSG game)
            assert rows[2]['game_path'].endswith('PSG 3 - 1 Marseille')
            assert rows[2]['game_time'] == '1 - 8:15'

    def test_extract_all_corners_handles_failed_extractions(self, temp_data_dir):
        """Test that pipeline handles failed frame extractions gracefully."""
        pipeline = CornerFramePipeline(str(temp_data_dir))

        with patch('corner_frame_pipeline.CornerFrameExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor

            # Mock some successful and some failed extractions
            mock_extractor.extract_corner_frame.side_effect = [
                "/path/to/frame1.jpg",  # Success
                None,                   # Failed
                "/path/to/frame3.jpg"   # Success
            ]

            result_csv = pipeline.extract_all_corners()

            # Check CSV content - should include failed extractions with None frame_path
            with open(result_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3
            assert rows[0]['frame_path'] == '/path/to/frame1.jpg'
            assert rows[1]['frame_path'] == ''  # Empty for failed extraction
            assert rows[2]['frame_path'] == '/path/to/frame3.jpg'

    def test_extract_all_corners_skips_games_without_videos(self, temp_data_dir):
        """Test that pipeline skips games that don't have video files."""
        # Create game without videos
        game_no_video = temp_data_dir / "datasets" / "soccernet" / "soccernet_videos" / "italy_serie-a" / "2015-2016" / "Test Game"
        game_no_video.mkdir(parents=True)

        labels = {
            "actions": {
                "1.png": {
                    "imageMetadata": {
                        "gameTime": "1 - 1:00",
                        "label": "Corner",
                        "half": 1,
                        "visibility": "visible"
                    }
                }
            }
        }
        with open(game_no_video / "Labels-v3.json", 'w') as f:
            json.dump(labels, f)

        pipeline = CornerFramePipeline(str(temp_data_dir))

        with patch('corner_frame_pipeline.CornerFrameExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            mock_extractor.extract_corner_frame.return_value = "/path/to/frame.jpg"

            result_csv = pipeline.extract_all_corners()

            # Should only process the 2 games with videos (3 corners total)
            # Should NOT process the game without videos
            assert mock_extractor.extract_corner_frame.call_count == 3

    def test_extract_all_corners_creates_output_directory(self, temp_data_dir):
        """Test that pipeline creates output directory for CSV."""
        pipeline = CornerFramePipeline(str(temp_data_dir))

        with patch('corner_frame_pipeline.CornerFrameExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            mock_extractor.extract_corner_frame.return_value = "/path/to/frame.jpg"

            result_csv = pipeline.extract_all_corners()

            # Output directory should be created
            output_dir = Path(result_csv).parent
            assert output_dir.exists()