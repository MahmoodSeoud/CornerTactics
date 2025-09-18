#!/usr/bin/env python3
"""
Tests for CLI interface for corner frame extraction.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from io import StringIO

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from cli import main


class TestCLI:
    """Test command-line interface."""

    def test_cli_requires_data_dir_argument(self):
        """Test that CLI requires data-dir argument."""
        with patch('sys.argv', ['cli.py']):
            with pytest.raises(SystemExit):
                main()

    def test_cli_runs_pipeline_with_data_dir(self, tmp_path):
        """Test that CLI runs pipeline with provided data directory."""
        with patch('sys.argv', ['cli.py', '--data-dir', str(tmp_path)]):
            with patch('cli.CornerFramePipeline') as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.extract_all_corners.return_value = "/path/to/output.csv"

                # Capture stdout
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    main()

                # Should create pipeline with provided data dir
                mock_pipeline_class.assert_called_once_with(str(tmp_path), None)
                mock_pipeline.extract_all_corners.assert_called_once()

                # Should print success message
                output = mock_stdout.getvalue()
                assert "Corner frame extraction complete" in output
                assert "/path/to/output.csv" in output

    def test_cli_accepts_custom_output_csv(self, tmp_path):
        """Test that CLI accepts custom output CSV path."""
        output_csv = tmp_path / "custom_output.csv"

        with patch('sys.argv', ['cli.py', '--data-dir', str(tmp_path), '--output', str(output_csv)]):
            with patch('cli.CornerFramePipeline') as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.extract_all_corners.return_value = str(output_csv)

                main()

                # Should create pipeline with custom output path
                mock_pipeline_class.assert_called_once_with(str(tmp_path), str(output_csv))

    def test_cli_handles_pipeline_errors(self, tmp_path):
        """Test that CLI handles pipeline errors gracefully."""
        with patch('sys.argv', ['cli.py', '--data-dir', str(tmp_path)]):
            with patch('cli.CornerFramePipeline') as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.extract_all_corners.side_effect = Exception("Pipeline error")

                with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    # Should exit with error code
                    assert exc_info.value.code == 1

                    # Should print error message
                    error_output = mock_stderr.getvalue()
                    assert "Error during extraction" in error_output
                    assert "Pipeline error" in error_output