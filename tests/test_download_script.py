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


if __name__ == '__main__':
    unittest.main()