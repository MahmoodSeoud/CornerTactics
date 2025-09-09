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


if __name__ == '__main__':
    unittest.main()