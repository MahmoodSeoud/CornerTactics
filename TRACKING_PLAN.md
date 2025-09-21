# Player & Ball Tracking Plan for Corner Kicks

## Current Situation Summary ✅ EXTRACTION COMPLETED
- **Available Data**: 500 games with raw video files (1_720p.mkv, 2_720p.mkv)
- **Labels Data**: Both Labels-v2.json (ALL corners) and Labels-v3.json (spatial annotations)
- **SNMOT**: 6 corner sequences with ground truth tracking for validation
- **Corner Frames Extracted**: **4,826 total corners successfully processed**
  - ✅ **4,221 visible corners** (87.5%) - Ready for player analysis
  - ❌ **605 "not shown" corners** (12.5%) - Camera on other footage
  - ✅ **100% extraction success rate** (4,826/4,826)
  - ✅ **856MB of frame data** stored
  - Frames saved to: `data/datasets/soccernet/corner_frames/`
  - CSV metadata: `data/insights/corner_frames_metadata.csv`
- **Target**: Generate player positions for 4,221 visible corners

## Recommended Solution: TrackLab + Fine-tuned YOLOv8

### Step 1: Set Up TrackLab Framework
- Clone TrackLab repository: https://github.com/TrackingLaboratory/tracklab
- Install dependencies:
  - Python 3.12
  - PyTorch 2.6
  - CUDA 12.4
- Configure for SoccerNet dataset support

### Step 2: Use Fine-tuned Detection Models
- **Model**: YOLOv11X fine-tuned for soccer 
- **Classes Detected**:
  - Player
  - Goalkeeper
  - Ball
  - Main referee
  - Side referee
- **Tracking Algorithm**: ByteTrack for temporal consistency

### Step 3: Process Corner Kick Sequences

#### 3.1 Extract Corner Clips ✅ COMPLETED (Alternative: Single Frames)
- **Implemented Solution**: Single frame extraction at corner moment
- **Alternative**: 20 seconds clips (10s before, 10s after corner kick)
- **Available Now**: 4,830 corner moment frames ready for analysis
- **Frame Rate**: 25 FPS (if switching to clips: 25 FPS × 20s = 500 frames per corner)

#### 3.2 Determine Corner Outcomes ✅ DATA AVAILABLE
Now using both Labels-v2.json (4,836 corners) and Labels-v3.json for outcome analysis:
```python
def get_corner_outcome(corner_timestamp, game_labels):
    """
    Check if a goal occurred within ~30 seconds after corner
    by comparing timestamps in Labels-v2.json (more complete) or Labels-v3.json
    """
    # Labels-v2.json has 'annotations' array with more events
    for annotation in game_labels.get('annotations', []):
        if annotation['label'] == 'Goal':
            # Parse timestamps to compare
            goal_time = parse_time(annotation['gameTime'])
            corner_time = parse_time(corner_timestamp)
            if 0 < goal_time - corner_time < 30:  # Within 30 seconds
                return 'GOAL'
    return 'NO_GOAL'
```

#### 3.3 Run Tracking Pipeline (NEXT STEPS)
Current status: **Ready to start with 4,830 corner frames**

**Option A: Single Frame Analysis (Quick Start)**
1. Load extracted corner frames from `data/datasets/soccernet/soccernet_corner_frames/`
2. Run detection on single frames:
   - Detection: YOLOv8X (fine-tuned) on each frame
   - Output: Player positions, ball location at corner moment
   - No temporal tracking needed for static analysis

**Option B: Video Clip Analysis (Full Tracking)**
1. Extract 20-second clips around corner moments (using existing infrastructure)
2. Run TrackLab pipeline:
   - Detection: YOLOv8X (fine-tuned)
   - Tracking: ByteTrack for temporal consistency
   - Output: Player trajectories, ball movement, team assignments
3. Label with outcome (GOAL/NO_GOAL) based on Labels-v2.json events

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
- **Player Detection**: Bounding boxes for each player in corner frame
- **Ball Detection**: Ball position in frame
- **Team Assignment**: Players labeled as team_left/team_right via jersey colors
- **Player Positions**: (x, y) coordinates for tactical analysis
- **Outcome Label**: GOAL/NO_GOAL based on subsequent events

### Dataset Statistics:
- **Total Corners**: 4,826 (extracted from 500 games)
- **Visible Corners**: 4,221 (87.5%) - High quality for analysis
- **Not Shown Corners**: 605 (12.5%) - Filter out for ML
- **Successfully Extracted**: 4,826/4,826 (100% success rate)
- **Storage Used**: 856MB for corner frames (JPEG format)
- **Average**: 9.7 corners per game (excellent distribution)

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

## Next Steps (UPDATED)

✅ **COMPLETED**: Corner frame extraction (4,830 frames ready)
✅ **COMPLETED**: Data pipeline with Labels-v2.json support
✅ **COMPLETED**: Outcome labeling infrastructure

**IMMEDIATE NEXT STEPS**:
1. **Choose analysis approach**:
   - Option A: Single frame analysis (faster, good for formation analysis)
   - Option B: Full video tracking (comprehensive, better for tactical sequences)
2. **Set up detection environment** with CUDA support
3. **Test on sample frames**: Run YOLOv8 detection on 10-20 corner frames
4. **Validate results**: Cross-check with V3 annotations where available
5. **Scale processing**: Apply to all 4,830 corner frames
6. **Export for ML**: Format data for geometric deep learning models (graph structures)

**FILES READY FOR PROCESSING**:
- Corner frames: `data/datasets/soccernet/corner_frames/*.jpg` (4,826 frames, 856MB)
- Metadata: `data/insights/corner_frames_metadata.csv`
- Labels: Both Labels-v2.json (ALL corners) and Labels-v3.json (spatial) available
- Visible corners only: Filter `visibility == "visible"` for 4,221 high-quality frames
