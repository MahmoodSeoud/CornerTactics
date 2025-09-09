#!/usr/bin/env python3
"""
SoccerNet Download Script
Command-line interface for downloading SoccerNet data.
"""

from data_loader import SoccerNetDataLoader


class SoccerNetDownloadScript:
    """Command-line script for downloading SoccerNet data."""
    
    def __init__(self, data_dir: str = "data", password: str = "s0cc3rn3t"):
        self.data_dir = data_dir
        self.password = password
        self.data_loader = SoccerNetDataLoader(data_dir, password)
    
    def download_all_labels(self, splits):
        """Download labels for all specified splits."""
        for split in splits:
            self.data_loader.download_annotations(split)