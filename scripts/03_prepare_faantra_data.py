#!/usr/bin/env python3
"""
Step 3: Prepare data for FAANTRA training.

This script:
1. Creates train/val/test splits (80/10/10)
2. Extracts frames from video clips using decord
3. Creates Labels-ball.json files for FAANTRA
4. Creates split JSON files and class.txt

Features:
- Chunked extraction for parallel SLURM jobs
- Resume support (skips existing frames)
- Progress tracking

Usage:
    # Create splits and metadata only (no frame extraction)
    python scripts/03_prepare_faantra_data.py --splits-only

    # Extract frames for all splits
    python scripts/03_prepare_faantra_data.py --extract-frames

    # Extract frames for specific split and chunk (for SLURM)
    python scripts/03_prepare_faantra_data.py --extract-frames --split train --chunk 0 --total-chunks 10

Output:
    FAANTRA/data/corner_anticipation/
    ├── train/
    │   ├── clip_1/frame0.jpg, frame1.jpg, ...
    │   ├── Labels-ball.json
    │   └── clip_mapping.json
    ├── valid/
    ├── test/
    ├── train.json, val.json, test.json
    └── class.txt
"""

import argparse
import json
import random
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Default paths (relative to CornerTactics root)
DEFAULT_DATASET_FILE = Path("FAANTRA/data/corners/corner_dataset.json")
DEFAULT_CLIPS_DIR = Path("FAANTRA/data/corners/clips")
DEFAULT_OUTPUT_DIR = Path("FAANTRA/data/corner_anticipation")

# Frame extraction settings
FPS = 25
CLIP_DURATION_SEC = 30
EXPECTED_FRAMES = FPS * CLIP_DURATION_SEC  # 750 frames

# Outcome classes for FAANTRA (order matters for class index)
OUTCOME_CLASSES = [
    "GOAL",
    "SHOT_ON_TARGET",
    "SHOT_OFF_TARGET",
    "CORNER_WON",
    "CLEARED",
    "NOT_DANGEROUS",
    "FOUL",
    "OFFSIDE"
]


def extract_frames_from_clip(args: tuple) -> tuple[int, int, str]:
    """
    Extract all frames from a single video clip.

    Args:
        args: Tuple of (clip_num, video_path, output_dir)

    Returns:
        Tuple of (clip_num, frame_count, status)
    """
    clip_num, video_path, output_dir = args

    clip_dir = output_dir / f"clip_{clip_num}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_frames = list(clip_dir.glob("frame*.jpg"))
    if len(existing_frames) >= EXPECTED_FRAMES - 10:  # Allow small tolerance
        return clip_num, len(existing_frames), "skipped"

    try:
        from decord import VideoReader, cpu
        from PIL import Image

        vr = VideoReader(str(video_path), ctx=cpu(0))
        n_frames = len(vr)

        if n_frames == 0:
            return clip_num, 0, "error: no frames"

        # Extract all frames
        for i in range(n_frames):
            frame = vr[i].asnumpy()
            img = Image.fromarray(frame)
            img.save(str(clip_dir / f"frame{i}.jpg"), quality=95)

        return clip_num, n_frames, "success"

    except Exception as e:
        return clip_num, 0, f"error: {str(e)[:50]}"


def create_labels_json(corners: list, outcome_frame: int = 625) -> dict:
    """
    Create Labels-ball.json for FAANTRA.

    Args:
        corners: List of corner dictionaries
        outcome_frame: Frame index where outcome is placed (start of anticipation)

    Returns:
        Labels-ball.json structure
    """
    videos = []

    for corner in corners:
        outcome = corner["outcome"]
        visibility = corner.get("visibility", "visible")

        # Position in milliseconds (frame_index * ms_per_frame)
        position_ms = int(outcome_frame * (1000 / FPS))

        video_entry = {
            "annotations": {
                "observation": [],  # No events during observation
                "anticipation": [{
                    "position": str(position_ms),
                    "label": outcome,
                    "visibility": visibility
                }]
            }
        }
        videos.append(video_entry)

    return {"videos": videos}


def create_splits(
    dataset_file: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> dict:
    """
    Create train/val/test splits and metadata files.

    Returns:
        Dictionary mapping split names to corner indices
    """
    print(f"Loading dataset from {dataset_file}")
    with open(dataset_file) as f:
        dataset = json.load(f)

    corners = dataset["corners"]
    total = len(corners)

    print(f"Total corners: {total}")

    # Shuffle indices
    random.seed(seed)
    indices = list(range(total))
    random.shuffle(indices)

    # Calculate split sizes
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    splits = {
        "train": indices[:n_train],
        "valid": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:]
    }

    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Valid: {len(splits['valid'])}")
    print(f"  Test: {len(splits['test'])}")

    # Create output directories and metadata
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_indices in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_corners = [corners[i] for i in split_indices]

        # Create clip mapping (clip_N -> original corner index)
        mapping = {f"clip_{i+1}": split_indices[i] for i in range(len(split_indices))}
        with open(split_dir / "clip_mapping.json", "w") as f:
            json.dump(mapping, f, indent=2)

        # Create Labels-ball.json
        labels = create_labels_json(split_corners)
        with open(split_dir / "Labels-ball.json", "w") as f:
            json.dump(labels, f, indent=2)

        print(f"Created {split_name} metadata")

    # Create split JSON files
    for split_name in ["train", "valid", "test"]:
        json_name = "val.json" if split_name == "valid" else f"{split_name}.json"
        split_json = [{
            "video": split_name,
            "num_frames": len(splits[split_name]) * EXPECTED_FRAMES,
            "num_clips": len(splits[split_name])
        }]
        with open(output_dir / json_name, "w") as f:
            json.dump(split_json, f)

    # Create class.txt
    with open(output_dir / "class.txt", "w") as f:
        f.write("\n".join(OUTCOME_CLASSES))

    # Create README
    readme = f"""# Corner Kick Outcome Anticipation Dataset

Formatted for FAANTRA training on corner kick outcome prediction.

## Statistics
- Total corners: {total}
- Train: {len(splits['train'])} clips
- Valid: {len(splits['valid'])} clips
- Test: {len(splits['test'])} clips

## Outcome Classes
{chr(10).join(f'- {c}' for c in OUTCOME_CLASSES)}

## Frame Convention
- FPS: {FPS}
- Clip length: {CLIP_DURATION_SEC} seconds ({EXPECTED_FRAMES} frames)
- Observation: frames 0-624 (25 seconds before corner)
- Anticipation: frames 625-749 (5 seconds after corner)
- Outcome event placed at frame 625 (start of anticipation)

## Directory Structure
```
corner_anticipation/
├── train/
│   ├── clip_1/frame0.jpg, frame1.jpg, ...
│   ├── clip_2/
│   ├── Labels-ball.json
│   └── clip_mapping.json
├── valid/
├── test/
├── train.json
├── val.json
├── test.json
└── class.txt
```
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)

    print(f"\nMetadata saved to: {output_dir}")

    return splits


def extract_frames(
    dataset_file: Path,
    clips_dir: Path,
    output_dir: Path,
    split: str = None,
    chunk: int = 0,
    total_chunks: int = 1,
    workers: int = 8
) -> dict:
    """
    Extract frames from video clips.

    Args:
        dataset_file: Path to corner_dataset.json
        clips_dir: Directory containing video clips
        output_dir: Output directory for frames
        split: Split to process (train/valid/test) or None for all
        chunk: Chunk index for parallel processing
        total_chunks: Total number of chunks
        workers: Number of parallel workers

    Returns:
        Extraction statistics
    """
    splits_to_process = [split] if split else ["train", "valid", "test"]

    total_stats = {
        "success": 0,
        "skipped": 0,
        "errors": 0,
        "error_clips": []
    }

    for split_name in splits_to_process:
        split_dir = output_dir / split_name
        mapping_file = split_dir / "clip_mapping.json"

        if not mapping_file.exists():
            print(f"Error: {mapping_file} not found. Run with --splits-only first.")
            continue

        with open(mapping_file) as f:
            clip_mapping = json.load(f)

        # Sort clips by number
        all_clips = sorted(
            clip_mapping.items(),
            key=lambda x: int(x[0].split("_")[1])
        )

        # Apply chunking
        chunk_size = (len(all_clips) + total_chunks - 1) // total_chunks
        start_idx = chunk * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_clips))
        clips_to_process = all_clips[start_idx:end_idx]

        if not clips_to_process:
            continue

        print(f"\n=== {split_name.upper()} split (chunk {chunk}/{total_chunks}) ===")
        print(f"Processing clips {start_idx + 1} to {end_idx} ({len(clips_to_process)} clips)")

        # Prepare extraction tasks
        tasks = []
        for clip_name, corner_idx in clips_to_process:
            clip_num = int(clip_name.split("_")[1])
            video_path = clips_dir / f"corner_{corner_idx:04d}" / "720p.mp4"

            if video_path.exists():
                tasks.append((clip_num, video_path, split_dir))
            else:
                total_stats["errors"] += 1
                total_stats["error_clips"].append(f"{split_name}/clip_{clip_num}")

        # Extract frames in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(extract_frames_from_clip, task): task[0]
                for task in tasks
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{split_name} chunk {chunk}"
            ):
                clip_num, n_frames, status = future.result()

                if status == "success":
                    total_stats["success"] += 1
                elif status == "skipped":
                    total_stats["skipped"] += 1
                else:
                    total_stats["errors"] += 1
                    total_stats["error_clips"].append(f"{split_name}/clip_{clip_num}")
                    if total_stats["errors"] <= 10:
                        tqdm.write(f"  Error clip_{clip_num}: {status}")

    print(f"\n=== Extraction Summary ===")
    print(f"Success: {total_stats['success']}")
    print(f"Skipped: {total_stats['skipped']}")
    print(f"Errors: {total_stats['errors']}")

    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for FAANTRA training"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help="Path to corner_dataset.json"
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=DEFAULT_CLIPS_DIR,
        help="Directory containing video clips"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for FAANTRA data"
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Create splits and metadata only (no frame extraction)"
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames from video clips"
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        help="Process specific split only"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Chunk index for parallel processing"
    )
    parser.add_argument(
        "--total-chunks",
        type=int,
        default=1,
        help="Total number of chunks"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits"
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        print("Run step 1 first: python scripts/01_build_corner_dataset.py")
        sys.exit(1)

    # Create splits if needed
    mapping_file = args.output / "train" / "clip_mapping.json"
    if not mapping_file.exists() or args.splits_only:
        create_splits(args.dataset, args.output, seed=args.seed)

    if args.splits_only:
        return

    # Extract frames
    if args.extract_frames:
        extract_frames(
            args.dataset,
            args.clips_dir,
            args.output,
            split=args.split,
            chunk=args.chunk,
            total_chunks=args.total_chunks,
            workers=args.workers
        )
    else:
        print("\nTo extract frames, run with --extract-frames")
        print("Example: python scripts/03_prepare_faantra_data.py --extract-frames")


if __name__ == "__main__":
    main()
