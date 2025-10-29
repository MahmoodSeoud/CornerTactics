#!/usr/bin/env python3
"""
Test for BASELINE_RESULTS.md documentation file.

Following TDD principles: This test verifies that the baseline results
documentation exists and contains all required information.
"""

import pytest
from pathlib import Path


def test_baseline_results_md_exists():
    """Test that BASELINE_RESULTS.md file exists."""
    docs_path = Path(__file__).parent.parent / "docs" / "BASELINE_RESULTS.md"
    assert docs_path.exists(), "docs/BASELINE_RESULTS.md should exist"


def test_baseline_results_contains_required_sections():
    """Test that BASELINE_RESULTS.md contains all required sections."""
    docs_path = Path(__file__).parent.parent / "docs" / "BASELINE_RESULTS.md"

    with open(docs_path, 'r') as f:
        content = f.read()

    # Required sections from TacticAI plan
    required_sections = [
        "TacticAI Day 5-6",  # Phase identifier
        "Random Baseline",   # Random baseline results
        "MLP Baseline",      # MLP baseline results
        "Success Criteria",  # Decision criteria
        "Decision",          # Go/No-Go decision
        "Top-1",             # Metrics (case-insensitive)
        "Top-3",             # Metrics (case-insensitive)
        "Top-5",             # Metrics (case-insensitive)
    ]

    for section in required_sections:
        assert section in content, f"Missing required section: {section}"


def test_baseline_results_contains_metrics():
    """Test that BASELINE_RESULTS.md contains actual metric values."""
    docs_path = Path(__file__).parent.parent / "docs" / "BASELINE_RESULTS.md"

    with open(docs_path, 'r') as f:
        content = f.read()

    # Should contain percentage values
    assert "%" in content, "Should contain percentage metrics"

    # Should reference the results file
    assert "baseline_mlp.json" in content, "Should reference results JSON file"


def test_baseline_results_has_decision():
    """Test that BASELINE_RESULTS.md contains a clear decision."""
    docs_path = Path(__file__).parent.parent / "docs" / "BASELINE_RESULTS.md"

    with open(docs_path, 'r') as f:
        content = f.read()

    # Should contain a decision about proceeding to Phase 2
    decision_keywords = ["Proceed", "Phase 2", "GATv2"]
    assert any(keyword in content for keyword in decision_keywords), \
        "Should contain decision about proceeding to Phase 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
