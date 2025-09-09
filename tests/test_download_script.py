#!/usr/bin/env python3
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from download_soccernet import SoccerNetDownloadScript


class TestSoccerNetDownloadScript(unittest.TestCase):
    
    def test_script_initialization_with_defaults(self):
        """Test that the download script can be initialized with default parameters."""
        script = SoccerNetDownloadScript()
        
        self.assertEqual(script.data_dir, 'data')
        self.assertEqual(script.password, 's0cc3rn3t')
        self.assertIsNotNone(script.data_loader)
    
    @patch('download_soccernet.SoccerNetDataLoader')
    def test_download_all_labels_calls_data_loader_correctly(self, mock_data_loader_class):
        """Test that download_all_labels method calls the data loader with correct parameters."""
        mock_data_loader = Mock()
        mock_data_loader_class.return_value = mock_data_loader
        
        script = SoccerNetDownloadScript()
        script.download_all_labels(['train', 'test'])
        
        mock_data_loader.download_annotations.assert_any_call('train')
        mock_data_loader.download_annotations.assert_any_call('test')
        self.assertEqual(mock_data_loader.download_annotations.call_count, 2)
    
    @patch('download_soccernet.SoccerNetDataLoader')
    def test_download_videos_for_games_calls_data_loader_correctly(self, mock_data_loader_class):
        """Test that download_videos_for_games method calls the data loader with correct game paths."""
        mock_data_loader = Mock()
        mock_data_loader_class.return_value = mock_data_loader
        
        game_paths = ['england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom',
                     'spain_laliga/2015-2016/2015-09-12 - 17-00 Espanyol 0 - 6 Real Madrid']
        
        script = SoccerNetDownloadScript()
        script.download_videos_for_games(game_paths)
        
        mock_data_loader.download_videos.assert_any_call(game_paths[0])
        mock_data_loader.download_videos.assert_any_call(game_paths[1])
        self.assertEqual(mock_data_loader.download_videos.call_count, 2)
    
    @patch('sys.argv', ['download_soccernet.py', '--labels', 'train', 'test'])
    @patch('download_soccernet.SoccerNetDataLoader')
    def test_main_with_labels_argument(self, mock_data_loader_class):
        """Test main function execution with --labels argument."""
        from download_soccernet import main
        mock_data_loader = Mock()
        mock_data_loader_class.return_value = mock_data_loader
        
        main()
        
        mock_data_loader.download_annotations.assert_any_call('train')
        mock_data_loader.download_annotations.assert_any_call('test')
    
    @patch('download_soccernet.SoccerNetDataLoader')
    def test_download_all_videos_calls_list_games_and_download_videos(self, mock_data_loader_class):
        """Test that download_all_videos gets game list and downloads videos for each game."""
        mock_data_loader = Mock()
        mock_data_loader_class.return_value = mock_data_loader
        mock_data_loader.list_games.return_value = ['game1', 'game2']
        
        script = SoccerNetDownloadScript()
        script.download_all_videos()
        
        mock_data_loader.list_games.assert_called_once()
        mock_data_loader.download_videos.assert_any_call('game1')
        mock_data_loader.download_videos.assert_any_call('game2')
        self.assertEqual(mock_data_loader.download_videos.call_count, 2)


if __name__ == '__main__':
    unittest.main()