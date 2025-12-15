# FAANTRA Corner Kick Prediction - Execution Plan

## Goal
Use FAANTRA (Football Action ANticipation TRAnsformer) to predict corner kick outcomes from video clips.

## Repository
https://github.com/MohamadDalal/FAANTRA

## Dataset
https://huggingface.co/datasets/SoccerNet/ActionAnticipation

---

## Phase 1: Setup Environment

```bash
# Create project directory
mkdir -p ~/faantra-corners
cd ~/faantra-corners

# Clone FAANTRA
git clone https://github.com/MohamadDalal/FAANTRA.git
cd FAANTRA

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Additional dependencies
pip install huggingface_hub pyzipper
```

---

## Phase 2: Download Dataset

```bash
# Download Ball Action Anticipation dataset
# Replace {NDA_KEY} with password from soccer-net.org/data NDA form
python setup_dataset_BAA.py --download-key {NDA_KEY}
```

This downloads:
- Pre-clipped 30-second video segments
- Action labels for each clip
- Train/valid/test/challenge splits

---

## Phase 3: Explore Dataset Structure

Create this script to understand what's in the dataset:

```python
# explore_dataset.py
import os
import json
from pathlib import Path

DATA_PATH = "data/BAA"  # Adjust if different

def explore_dataset():
    """Explore the BAA dataset structure and find corner-related clips."""

    # List available splits
    print("Available splits:")
    for split in os.listdir(DATA_PATH):
        split_path = Path(DATA_PATH) / split
        if split_path.is_dir():
            print(f"  {split}/")

    # Load labels and look for action types
    labels_path = Path(DATA_PATH) / "train" / "labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

        # Count action types
        action_counts = {}
        for clip_id, clip_data in labels.items():
            for action in clip_data.get("actions", []):
                action_type = action.get("label", "unknown")
                action_counts[action_type] = action_counts.get(action_type, 0) + 1

        print("\nAction types in dataset:")
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"  {action}: {count}")

    # Check clip structure
    print("\nSample clip structure:")
    train_path = Path(DATA_PATH) / "train"
    clips = [d for d in train_path.iterdir() if d.is_dir()]
    if clips:
        sample_clip = clips[0]
        print(f"  {sample_clip.name}/")
        for item in sample_clip.iterdir():
            print(f"    {item.name}")

if __name__ == "__main__":
    explore_dataset()
```

Run it:
```bash
python explore_dataset.py
```

---

## Phase 4: Identify Corner Clips

Create script to find clips containing corners:

```python
# find_corner_clips.py
import os
import json
from pathlib import Path

DATA_PATH = "data/BAA"

def find_corner_clips():
    """
    Find clips that contain corner kick situations.

    Corners might be identified by:
    1. Action labels containing "corner"
    2. Context before certain actions (cross from corner position)
    3. Spatial position of actions near corner flag
    """

    corner_clips = []
    all_clips = []

    for split in ["train", "valid", "test"]:
        labels_path = Path(DATA_PATH) / split / "labels.json"
        if not labels_path.exists():
            continue

        with open(labels_path) as f:
            labels = json.load(f)

        for clip_id, clip_data in labels.items():
            all_clips.append({"split": split, "clip_id": clip_id, "data": clip_data})

            # Check if any action mentions corner
            actions = clip_data.get("actions", [])
            action_labels = [a.get("label", "").lower() for a in actions]

            # Look for corner indicators
            is_corner = any("corner" in label for label in action_labels)

            # Also check for crosses that might be from corners
            # (This is a heuristic - may need refinement)
            has_cross = any("cross" in label for label in action_labels)

            if is_corner:
                corner_clips.append({
                    "split": split,
                    "clip_id": clip_id,
                    "actions": actions
                })

    print(f"Total clips: {len(all_clips)}")
    print(f"Corner clips found: {len(corner_clips)}")

    # Save corner clips list
    with open("corner_clips.json", "w") as f:
        json.dump(corner_clips, f, indent=2)

    # Show sample
    if corner_clips:
        print("\nSample corner clip:")
        print(json.dumps(corner_clips[0], indent=2))

    return corner_clips

if __name__ == "__main__":
    find_corner_clips()
```

Run it:
```bash
python find_corner_clips.py
```

---

## Phase 5: Create Corner-Specific Dataset

If corners are found, create a filtered dataset:

```python
# create_corner_dataset.py
import os
import json
import shutil
from pathlib import Path

DATA_PATH = "data/BAA"
CORNER_DATA_PATH = "data/BAA_corners"

def create_corner_dataset():
    """Create a corner-only subset of the BAA dataset."""

    # Load corner clips
    with open("corner_clips.json") as f:
        corner_clips = json.load(f)

    if not corner_clips:
        print("No corner clips found!")
        return

    # Create output directory
    os.makedirs(CORNER_DATA_PATH, exist_ok=True)

    # Group by split
    by_split = {}
    for clip in corner_clips:
        split = clip["split"]
        if split not in by_split:
            by_split[split] = []
        by_split[split].append(clip)

    # Copy clips to new location
    for split, clips in by_split.items():
        split_out = Path(CORNER_DATA_PATH) / split
        os.makedirs(split_out, exist_ok=True)

        # Create labels file
        labels = {}
        for clip in clips:
            clip_id = clip["clip_id"]

            # Copy clip folder
            src = Path(DATA_PATH) / split / clip_id
            dst = split_out / clip_id
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)

            # Add to labels
            labels[clip_id] = {"actions": clip["actions"]}

        # Save labels
        with open(split_out / "labels.json", "w") as f:
            json.dump(labels, f, indent=2)

        print(f"{split}: {len(clips)} corner clips")

    print(f"\nCorner dataset created at {CORNER_DATA_PATH}")

if __name__ == "__main__":
    create_corner_dataset()
```

---

## Phase 6: Train FAANTRA on Corners

Option A: Train on full dataset first (baseline)
```bash
python main.py config/BAA_config.json baseline_model
```

Option B: Train on corner-only subset
```bash
# First modify config to point to corner dataset
# Then train
python main.py config/corner_config.json corner_model
```

Create corner config:
```python
# create_corner_config.py
import json

# Load base config
with open("config/BAA_config.json") as f:
    config = json.load(f)

# Modify for corners
config["data_path"] = "data/BAA_corners"
config["model_name"] = "corner_faantra"

# Save
with open("config/corner_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Created config/corner_config.json")
```

---

## Phase 7: Evaluate

```bash
# Evaluate on test set
python test.py config/corner_config.json checkpoints/best.pth corner_model -s test
```

---

## Phase 8: Analysis

Create script to analyze predictions:

```python
# analyze_results.py
import json

def analyze_predictions():
    """Analyze model predictions on corner clips."""

    # Load predictions (generated by test.py)
    with open("predictions.json") as f:
        predictions = json.load(f)

    # Load ground truth
    with open("data/BAA_corners/test/labels.json") as f:
        ground_truth = json.load(f)

    # Compare
    correct = 0
    total = 0

    by_action = {}  # Accuracy per action type

    for clip_id, pred in predictions.items():
        gt = ground_truth.get(clip_id, {})

        pred_actions = set(a["label"] for a in pred.get("actions", []))
        gt_actions = set(a["label"] for a in gt.get("actions", []))

        # Check overlap
        for action in gt_actions:
            total += 1
            if action in pred_actions:
                correct += 1

            if action not in by_action:
                by_action[action] = {"correct": 0, "total": 0}
            by_action[action]["total"] += 1
            if action in pred_actions:
                by_action[action]["correct"] += 1

    print(f"Overall accuracy: {correct}/{total} = {100*correct/total:.1f}%")
    print("\nPer-action accuracy:")
    for action, stats in sorted(by_action.items()):
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {action}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

if __name__ == "__main__":
    analyze_predictions()
```

---

## Decision Points

After Phase 4, check:
- **If >500 corner clips:** Proceed with corner-specific training
- **If 100-500 corner clips:** Use full dataset, evaluate on corner subset only
- **If <100 corner clips:** Ask Discord if corners are in dataset, or pivot approach

After Phase 7, check:
- **If mAP > 30%:** Good baseline, iterate on model
- **If mAP 10-30%:** Task is learnable but hard, expected for anticipation
- **If mAP < 10%:** Check data pipeline, may need different approach

---

## File Structure After Setup

```
~/faantra-corners/FAANTRA/
├── data/
│   ├── BAA/                    # Full dataset
│   │   ├── train/
│   │   ├── valid/
│   │   ├── test/
│   │   └── challenge/
│   └── BAA_corners/            # Corner subset
│       ├── train/
│       ├── valid/
│       └── test/
├── config/
│   ├── BAA_config.json         # Original config
│   └── corner_config.json      # Corner config
├── checkpoints/
│   └── best.pth
├── corner_clips.json
├── explore_dataset.py
├── find_corner_clips.py
├── create_corner_dataset.py
└── analyze_results.py
```

---

## Commands Summary

```bash
# 1. Setup
cd ~/faantra-corners/FAANTRA
source venv/bin/activate

# 2. Download data
python setup_dataset_BAA.py --download-key {KEY}

# 3. Explore
python explore_dataset.py

# 4. Find corners
python find_corner_clips.py

# 5. Create corner dataset (if corners found)
python create_corner_dataset.py

# 6. Create config
python create_corner_config.py

# 7. Train
python main.py config/corner_config.json corner_model

# 8. Evaluate
python test.py config/corner_config.json checkpoints/best.pth corner_model -s test

# 9. Analyze
python analyze_results.py
```
