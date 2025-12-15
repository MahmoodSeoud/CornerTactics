#!/usr/bin/env python3
"""
Quick feasibility test for soccersegcal pipeline.

Run this locally to verify the pipeline works on a small subset
before submitting SLURM jobs for full processing.
"""

import subprocess
import sys
from pathlib import Path
import json
import os

PIPELINE_DIR = Path("/home/mseo/CornerTactics/soccersegcal-pipeline")
SCRIPTS_DIR = PIPELINE_DIR / "scripts"


def run_step(name: str, script: str, args: list = None):
    """Run a pipeline step."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print('='*60)

    cmd = [sys.executable, str(SCRIPTS_DIR / script)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: {name} failed with code {result.returncode}")
        return False

    return True


def check_results():
    """Check pipeline results."""
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)

    data_dir = PIPELINE_DIR / "data"

    files_to_check = [
        ("corner_index.csv", "Corner index"),
        ("frames_index.csv", "Frames index"),
        ("camera_calibrations.json", "Camera calibrations"),
        ("player_detections.json", "Player detections"),
        ("corner_positions.json", "Corner positions"),
        ("corner_dataset.json", "Final dataset")
    ]

    for filename, desc in files_to_check:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  [OK] {desc}: {size/1024:.1f} KB")

            # Count records for JSON files
            if filename.endswith('.json'):
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        print(f"       Records: {len(data)}")
                except:
                    pass
        else:
            print(f"  [--] {desc}: NOT FOUND")


def main():
    print("="*60)
    print("  SOCCERSEGCAL PIPELINE - QUICK TEST")
    print("="*60)
    print(f"\nPipeline directory: {PIPELINE_DIR}")

    # Check if venv exists
    venv_dir = PIPELINE_DIR / "venv"
    if not venv_dir.exists():
        print("\nWARNING: Virtual environment not found!")
        print("Run: bash scripts/setup_environment.sh")
        return

    # Ensure data directory exists
    (PIPELINE_DIR / "data").mkdir(exist_ok=True)
    (PIPELINE_DIR / "outputs").mkdir(exist_ok=True)

    # Run pipeline steps with limited data
    steps = [
        ("Step 1: Load corners", "01_load_corners.py", []),
        ("Step 2: Extract frames", "02_extract_frames.py", ["--limit", "10"]),
        ("Step 3: Camera calibration", "03_calibrate_cameras.py", ["--device", "cpu", "--limit", "30"]),
        ("Step 4: Player detection", "04_detect_players.py", ["--model", "yolov8n.pt", "--limit", "30"]),
        ("Step 5: Project to pitch", "05_project_to_pitch.py", ["--min-players", "5"]),
        ("Step 6: Create dataset", "06_create_dataset.py", [])
    ]

    for name, script, args in steps:
        success = run_step(name, script, args)
        if not success:
            print(f"\nPipeline failed at: {name}")
            break

    # Check results
    check_results()

    print("\n" + "="*60)
    print("  QUICK TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
