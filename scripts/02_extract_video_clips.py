#!/usr/bin/env python3
"""
Step 2: Extract video clips for corner kicks.

This script extracts 30-second video clips from SoccerNet match videos:
- 25 seconds before corner (observation window)
- 5 seconds after corner (anticipation window)

Features:
- Parallel extraction with configurable workers
- Resume support (skips existing clips)
- Integrity verification
- Corrupt clip detection and repair

Usage:
    python scripts/02_extract_video_clips.py [options]

    # Extract all clips
    python scripts/02_extract_video_clips.py

    # Extract specific range (for parallel SLURM jobs)
    python scripts/02_extract_video_clips.py --start 0 --end 1000

    # Verify and repair corrupt clips
    python scripts/02_extract_video_clips.py --verify --repair

Output:
    FAANTRA/data/corners/clips/corner_XXXX/720p.mp4
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Default paths (relative to CornerTactics root)
DEFAULT_DATASET_FILE = Path("FAANTRA/data/corners/corner_dataset.json")
DEFAULT_OUTPUT_DIR = Path("FAANTRA/data/corners/clips")

# FFmpeg settings
FFMPEG_PATH = "/opt/itu/easybuild/software/FFmpeg/6.0-GCCcore-12.3.0/bin/ffmpeg"
FFPROBE_PATH = "/opt/itu/easybuild/software/FFmpeg/6.0-GCCcore-12.3.0/bin/ffprobe"

# Clip settings
CLIP_BEFORE_MS = 25000  # 25 seconds before corner
CLIP_AFTER_MS = 5000    # 5 seconds after corner
OUTPUT_RESOLUTION = "720p"


def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-version"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def verify_video_integrity(video_path: Path) -> tuple[bool, str]:
    """
    Verify video file integrity using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not video_path.exists():
        return False, "File not found"

    # Check file size (should be > 100KB for a valid clip)
    if video_path.stat().st_size < 100 * 1024:
        return False, "File too small"

    try:
        # Use ffprobe to check video integrity
        cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration,nb_frames",
            "-of", "json",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode != 0:
            stderr = result.stderr.decode()
            if "moov atom not found" in stderr:
                return False, "Corrupt: moov atom not found"
            return False, f"ffprobe error: {stderr[:100]}"

        # Parse output
        output = json.loads(result.stdout)
        streams = output.get("streams", [])

        if not streams:
            return False, "No video stream found"

        # Check duration (should be ~30 seconds)
        duration = float(streams[0].get("duration", 0))
        if duration < 25:  # Allow some tolerance
            return False, f"Duration too short: {duration:.1f}s"

        return True, "OK"

    except subprocess.TimeoutExpired:
        return False, "Verification timeout"
    except json.JSONDecodeError:
        return False, "Invalid ffprobe output"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


def extract_clip(args: tuple) -> tuple[int, bool, str]:
    """
    Extract a single video clip.

    Args:
        args: Tuple of (corner_idx, corner_data, output_dir, force)

    Returns:
        Tuple of (corner_idx, success, message)
    """
    corner_idx, corner, output_dir, force = args

    video_file = Path(corner["video_file"])
    if not video_file.exists():
        return corner_idx, False, f"Source not found: {video_file.name}"

    # Output path
    output_file = output_dir / f"corner_{corner_idx:04d}" / f"{OUTPUT_RESOLUTION}.mp4"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Skip if exists and not forcing
    if output_file.exists() and not force:
        is_valid, msg = verify_video_integrity(output_file)
        if is_valid:
            return corner_idx, True, "exists"
        # File exists but corrupt, will re-extract

    # Calculate timing
    corner_time_ms = corner["corner_time_ms"]
    start_ms = max(0, corner_time_ms - CLIP_BEFORE_MS)
    duration_ms = CLIP_BEFORE_MS + CLIP_AFTER_MS

    start_sec = start_ms / 1000
    duration_sec = duration_ms / 1000

    # FFmpeg command
    cmd = [
        FFMPEG_PATH,
        "-ss", str(start_sec),
        "-i", str(video_file),
        "-t", str(duration_sec),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # No audio
        "-y",   # Overwrite
        str(output_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120
        )

        if result.returncode != 0:
            stderr = result.stderr.decode()
            return corner_idx, False, f"ffmpeg error: {stderr[-100:]}"

        # Verify the extracted clip
        is_valid, msg = verify_video_integrity(output_file)
        if not is_valid:
            output_file.unlink(missing_ok=True)
            return corner_idx, False, f"Verification failed: {msg}"

        return corner_idx, True, "OK"

    except subprocess.TimeoutExpired:
        return corner_idx, False, "Timeout"
    except Exception as e:
        return corner_idx, False, f"Exception: {str(e)[:50]}"


def extract_clips(
    dataset_file: Path,
    output_dir: Path,
    start: int = 0,
    end: int = None,
    workers: int = 8,
    force: bool = False
) -> dict:
    """
    Extract video clips for corner kicks.

    Args:
        dataset_file: Path to corner_dataset.json
        output_dir: Output directory for clips
        start: Starting corner index
        end: Ending corner index (exclusive)
        workers: Number of parallel workers
        force: Force re-extraction of existing clips

    Returns:
        Dictionary with extraction statistics
    """
    # Load dataset
    print(f"Loading dataset from {dataset_file}")
    with open(dataset_file) as f:
        dataset = json.load(f)

    corners = dataset["corners"]
    total = len(corners)

    # Apply range filter
    if end is None:
        end = total
    corners_to_process = list(enumerate(corners))[start:end]

    print(f"Processing corners {start} to {end} ({len(corners_to_process)} clips)")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {workers}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare extraction tasks
    tasks = [
        (idx, corner, output_dir, force)
        for idx, corner in corners_to_process
    ]

    # Track results
    stats = {
        "total": len(tasks),
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "failed_indices": []
    }

    # Extract in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_clip, task): task[0] for task in tasks}

        with tqdm(total=len(tasks), desc="Extracting clips") as pbar:
            for future in as_completed(futures):
                idx, success, msg = future.result()

                if success:
                    if msg == "exists":
                        stats["skipped"] += 1
                    else:
                        stats["success"] += 1
                else:
                    stats["failed"] += 1
                    stats["failed_indices"].append(idx)
                    if stats["failed"] <= 10:
                        tqdm.write(f"  Failed corner_{idx:04d}: {msg}")

                pbar.update(1)

    # Print summary
    print(f"\nExtraction complete:")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped (existing): {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")

    if stats["failed_indices"]:
        print(f"  Failed indices: {stats['failed_indices'][:20]}{'...' if len(stats['failed_indices']) > 20 else ''}")

    return stats


def verify_all_clips(dataset_file: Path, output_dir: Path) -> list:
    """
    Verify integrity of all extracted clips.

    Returns:
        List of corrupt clip indices
    """
    with open(dataset_file) as f:
        dataset = json.load(f)

    corners = dataset["corners"]
    corrupt = []

    print(f"Verifying {len(corners)} clips...")

    for idx in tqdm(range(len(corners)), desc="Verifying"):
        clip_path = output_dir / f"corner_{idx:04d}" / f"{OUTPUT_RESOLUTION}.mp4"
        is_valid, msg = verify_video_integrity(clip_path)
        if not is_valid:
            corrupt.append((idx, msg))

    print(f"\nVerification complete:")
    print(f"  Valid: {len(corners) - len(corrupt)}")
    print(f"  Corrupt/Missing: {len(corrupt)}")

    if corrupt:
        print("\nCorrupt clips:")
        for idx, msg in corrupt[:20]:
            print(f"  corner_{idx:04d}: {msg}")
        if len(corrupt) > 20:
            print(f"  ... and {len(corrupt) - 20} more")

    return [idx for idx, _ in corrupt]


def main():
    parser = argparse.ArgumentParser(
        description="Extract video clips for corner kicks"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help="Path to corner_dataset.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for clips"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting corner index"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending corner index (exclusive)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction of existing clips"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify integrity of existing clips"
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-extract corrupt clips (use with --verify)"
    )
    args = parser.parse_args()

    # Check FFmpeg
    if not check_ffmpeg():
        print(f"Error: FFmpeg not found at {FFMPEG_PATH}")
        print("Load the module: module load GCCcore/12.3.0 FFmpeg/6.0-GCCcore-12.3.0")
        sys.exit(1)

    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        print("Run step 1 first: python scripts/01_build_corner_dataset.py")
        sys.exit(1)

    if args.verify:
        corrupt = verify_all_clips(args.dataset, args.output)

        if args.repair and corrupt:
            print(f"\nRepairing {len(corrupt)} corrupt clips...")

            # Load dataset
            with open(args.dataset) as f:
                dataset = json.load(f)

            # Prepare repair tasks
            tasks = [
                (idx, dataset["corners"][idx], args.output, True)
                for idx in corrupt
            ]

            # Re-extract
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(extract_clip, task): task[0] for task in tasks}

                repaired = 0
                for future in tqdm(as_completed(futures), total=len(tasks), desc="Repairing"):
                    idx, success, msg = future.result()
                    if success:
                        repaired += 1
                    else:
                        tqdm.write(f"  Still failed corner_{idx:04d}: {msg}")

            print(f"\nRepaired: {repaired}/{len(corrupt)}")
    else:
        extract_clips(
            args.dataset,
            args.output,
            args.start,
            args.end,
            args.workers,
            args.force
        )


if __name__ == "__main__":
    main()
