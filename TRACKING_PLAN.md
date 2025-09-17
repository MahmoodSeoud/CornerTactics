# Player & Ball Tracking Plan for 4,229 Corner Kicks

## Current Situation Summary
- **Available Data**: 500 games with raw video files (1_720p.mkv, 2_720p.mkv)
- **V3 Frames**: Sparse annotations (~23 key frames per game with bounding boxes)
- **SNMOT**: Only 12 sequences with continuous 30-second tracking
- **Target**: Generate tracking data for 4,229 corner kicks

## Recommended Solution: TrackLab + Fine-tuned YOLOv8

### Step 1: Set Up TrackLab Framework
- Clone TrackLab repository: https://github.com/TrackingLaboratory/tracklab
- Install dependencies:
  - Python 3.12
  - PyTorch 2.6
  - CUDA 12.4
- Configure for SoccerNet dataset support

### Step 2: Use Fine-tuned Detection Models
- **Model**: YOLOv8X fine-tuned for soccer (used by SoccerNet 2024 challenge winners)
- **Classes Detected**:
  - Player
  - Goalkeeper
  - Ball
  - Main referee
  - Side referee
- **Tracking Algorithm**: ByteTrack for temporal consistency

### Step 3: Process Corner Kick Sequences

#### 3.1 Extract Corner Clips
- **Duration**: 20 seconds total (10s before, 10s after corner kick)
- **Rationale**: 10 seconds after is sufficient to capture immediate outcome
- **Frame Rate**: 25 FPS × 20s = 500 frames per corner

#### 3.2 Determine Corner Outcomes
Using Labels-v3.json, we can automatically label corner outcomes:
```python
def get_corner_outcome(corner_timestamp, game_labels):
    """
    Check if a goal occurred within ~30 seconds after corner
    by comparing timestamps in Labels-v3.json
    """
    for event in game_labels['actions']:
        if event['label'] == 'Goal':
            time_diff = event['position'] - corner_timestamp
            if 0 < time_diff < 30000:  # Within 30 seconds
                return 'GOAL'
    return 'NO_GOAL'
```

#### 3.3 Run Tracking Pipeline
1. Load corner kick timestamps from Labels-v3.json
2. Extract 20-second video clips
3. Run TrackLab pipeline:
   - Detection: YOLOv8X (fine-tuned)
   - Tracking: ByteTrack
   - Output: Player positions, ball location, team assignments
4. Label with outcome (GOAL/NO_GOAL) based on subsequent events

### Step 4: Alternative Quick Start Options

If TrackLab setup is complex, consider these simpler alternatives:

#### Option A: Anudeep007-hub/soccer-multi-object-tracking (Recommended)
- Pre-configured YOLOv8 + ByteTrack
- Team assignment via jersey color clustering
- Handles occlusions well
- GitHub: https://github.com/Anudeep007-hub/soccer-multi-object-tracking

#### Option B: Darkmyter/Football-Players-Tracking
- Simple YOLOv8 + ByteTrack implementation
- Easy to modify for specific needs
- GitHub: https://github.com/Darkmyter/Football-Players-Tracking

### Step 5: Data Processing Pipeline

```python
# Pseudo-code for processing pipeline
for game in games:
    labels = load_json(f"{game}/Labels-v3.json")
    video = load_video(f"{game}/1_720p.mkv", f"{game}/2_720p.mkv")

    for corner in get_corners(labels):
        # Extract 20-second clip
        clip = extract_clip(video,
                          start=corner.time - 10s,
                          end=corner.time + 10s)

        # Run tracking
        tracking_data = tracklab.process(clip)

        # Determine outcome
        outcome = get_corner_outcome(corner.timestamp, labels)

        # Save results
        save_tracking(tracking_data, outcome, corner.id)
```

## Expected Outputs

### Per Corner Kick:
- **Player Tracking**: Bounding boxes for each player across 500 frames
- **Ball Tracking**: Ball position for each frame
- **Team Assignment**: Players labeled as team_left/team_right
- **Player IDs**: Consistent IDs for tracking across frames
- **Outcome Label**: GOAL/NO_GOAL based on subsequent events

### Dataset Statistics:
- **Total Corners**: 4,229
- **Frames per Corner**: 500 (20s × 25fps)
- **Total Frames**: ~2.1 million frames to process
- **Storage Required**: ~1.5TB for tracking data (JSON format)

## Validation Strategy

1. **Use SNMOT Sequences**: Validate tracking quality against 12 ground-truth sequences
2. **V3 Frame Validation**: Cross-check detected bounding boxes against V3 annotations
3. **Outcome Verification**: Manually verify goal detection logic on sample of games

## Processing Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup & Configuration | 1-2 days | Install dependencies, configure models |
| Test on Sample | 0.5 day | Process 10 games to validate pipeline |
| Full Processing | 2-3 days | Single GPU, can parallelize to reduce time |
| Validation | 1 day | Quality checks and outcome verification |
| **Total** | **4-6 days** | Assuming single GPU setup |

## Advantages of This Approach

1. **Automated Outcome Labeling**: Use existing Labels-v3.json to determine if corner resulted in goal
2. **Efficient Clip Duration**: 20 seconds captures essential action while reducing processing
3. **Pre-trained Models**: Leverage SoccerNet 2024 challenge winners' approaches
4. **Validation Data**: Can verify quality using SNMOT ground truth and V3 annotations

## Next Steps

1. Choose tracking implementation (TrackLab or alternatives)
2. Set up environment with CUDA support
3. Test pipeline on one game to verify:
   - Corner extraction works correctly
   - Tracking produces expected outputs
   - Outcome labeling is accurate
4. Scale to full dataset with parallel processing if available
5. Export tracking data in format suitable for geometric deep learning models