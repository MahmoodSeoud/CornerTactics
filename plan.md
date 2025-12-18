# CornerTactics - Corner Kick Outcome Prediction

## Goal
Predict corner kick outcomes (SHOT vs NO_SHOT) from SoccerNet broadcast videos using FAANTRA.

## Approach

Using FAANTRA (Football Action ANticipation TRAnsformer) for binary classification:
- **SHOT**: Corner results in a shot attempt (goal, on target, off target)
- **NO_SHOT**: Corner is cleared, won foul, etc.

Labeling follows TacticAI methodology: classify based on **immediate next event** after corner, not a time window.

---

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Video Clip Extraction | **Complete** | 4,836 corner clips (114GB) |
| 2. Label Processing | **Complete** | SHOT/NO_SHOT labels using immediate next event |
| 3. Frame Extraction | **Complete** | 4,836 clips × 750 frames each |
| 4. Model Training | **In Progress** | Training on V100 (Job 43215) |
| 5. Evaluation | Pending | Evaluate model performance |

---

## Phase 1: Video Clip Extraction (Complete)

### Dataset Statistics

- **Total corners**: 4,836 video clips
- **Clip duration**: 30 seconds (25s observation + 5s anticipation)
- **Resolution**: 720p MP4
- **Total size**: 114GB
- **Valid clips**: 4,690 (97% - some corrupt source videos)

### Source Data

SoccerNet videos from 6 leagues:
- England Premier League, Spain La Liga, Italy Serie A
- Germany Bundesliga, France Ligue 1, UEFA Champions League

---

## Phase 2: Label Processing (Complete)

### Labeling Methodology

Following TacticAI approach: label based on **immediate next event** after corner kick.

**Old approach (15-second window)**: Captured unrelated events like offsides from counter-attacks.

**New approach (immediate next event)**: Clean labels matching what actually happened.

### Label Distribution

| Primary Label | Count | Percentage |
|---------------|-------|------------|
| **SHOT** | 1,476 | 30.5% |
| **NO_SHOT** | 3,360 | 69.5% |

### Shot Breakdown

| Type | Count | % |
|------|-------|---|
| Off Target | 819 | 16.9% |
| On Target | 636 | 13.2% |
| Goal | 21 | 0.4% |

### No-Shot Breakdown

| Type | Count | % |
|------|-------|---|
| Clearance (ball out of play) | 1,560 | 32.3% |
| Timeout (>30s to next event) | 1,100 | 22.7% |
| Foul | 533 | 11.0% |
| Offside | 134 | 2.8% |

### Key Files

- `FAANTRA/data/corners/corner_dataset.json` - Original 8-class labels
- `FAANTRA/data/corners/corner_dataset_v2.json` - New SHOT/NO_SHOT labels

---

## Phase 3: Frame Extraction (Complete)

### Progress

- **Train**: 3,868 / 3,868 clips (100%)
- **Valid**: 483 / 483 clips (100%)
- **Test**: 485 / 485 clips (100%)

### Configuration

- Frame rate: 25 fps
- Frames per clip: 750 (30 seconds)
- Format: JPEG
- Split: 80/10/10 train/valid/test

### Output Structure

```
FAANTRA/data/corner_anticipation/
├── train/
│   ├── clip_1/frame0.jpg ... frame749.jpg
│   ├── clip_2/
│   ├── Labels-ball.json
│   └── clip_mapping.json
├── valid/
├── test/
├── train.json, val.json, test.json
└── class.txt
```

---

## Phase 4: Model Training (In Progress)

### FAANTRA Configuration

- **Architecture**: Transformer with RegNetY backbone + GSF (Gated Shift Fusion)
- **Observation**: 50% of clip (first 375 frames)
- **Anticipation**: 50% of clip (predict outcome)
- **Batch size**: 1 (V100 32GB memory constraint)
- **Epochs**: 50
- **Learning rate**: 0.0001 with linear warmup (5 epochs) + cosine annealing
- **Classes**: 8 corner outcomes (GOAL, SHOT_ON_TARGET, SHOT_OFF_TARGET, CORNER_WON, CLEARED, NOT_DANGEROUS, FOUL, OFFSIDE)

### Training Status

**Current Job**: 43215 on V100 (cn5.hpc.itu.dk)
- Training started: 2025-12-18
- Estimated time: ~25 hours (50 epochs × 30 min/epoch)

### Issues Encountered & Fixed

#### 1. Label Preprocessing Bug (Critical)
**Problem**: FAANTRA's `_store_clips_anticipation()` only processed first ~15 seconds of each 30-second clip, missing our labels at 25 seconds.

```python
# Original (broken): video_len = int(full_video_len * ((clip_len*stride/FPS)+5)/30) = 381 frames
# Fixed: video_len = full_video_len = 750 frames
```

**Fix**: Modified `FAANTRA/dataset/frame.py:132` to use full video length.

**Result**: 78,729 clips now have valid labels (was 0 before).

#### 2. Multi-GPU Incompatibility
**Problem**: FAANTRA's GSF (Gated Shift Fusion) module incompatible with PyTorch DataParallel.
- 4× V100: `CUDA error: misaligned address`
- 2× V100: `cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`

**Solution**: Single GPU training only.

#### 3. Memory Constraints
**Problem**: V100 32GB OOM with batch_size > 1.

**Solution**: batch_size=1 for V100, batch_size=8 planned for A100 80GB.

#### 4. Corrupt Video Clips
**Status**: 146/4836 clips (3%) have "moov atom not found" error.
- Being repaired via `python scripts/02_extract_video_clips.py --verify --repair`
- Does not block current training (frames already extracted)

### Training Commands

```bash
cd /home/mseo/CornerTactics/FAANTRA
source venv/bin/activate

# Submit V100 training job
sbatch scripts/slurm/train_faantra_v100.sbatch

# Or A100 (when available)
sbatch scripts/slurm/train_faantra.sbatch

# Monitor
squeue -u $USER
tail -f logs/train_*.out
```

### Key Config Files

- `config/SoccerNetBall/Corner-Config.json` - V100 config (batch_size=1)
- `config/SoccerNetBall/Corner-Config-v2.json` - A100 config (batch_size=8)

---

## Phase 5: Evaluation (Pending)

### Metrics

- mAP@delta: Temporal precision
- Binary accuracy: SHOT vs NO_SHOT
- Precision/Recall per class

### Baselines to Compare

- Random baseline: 50%
- Most-frequent-class (NO_SHOT): 69.5%
- FAANTRA trained on our data: TBD

---

## Scripts

Clean, refactored pipeline in `scripts/`:

| Script | Purpose |
|--------|---------|
| `01_build_corner_dataset.py` | Build corner metadata (original 8-class) |
| `01_build_corner_dataset_v2.py` | Build SHOT/NO_SHOT labels (TacticAI method) |
| `02_extract_video_clips.py` | Extract 30s video clips from SoccerNet |
| `03_prepare_faantra_data.py` | Create splits + extract frames |
| `slurm/extract_frames.sbatch` | SLURM job for parallel frame extraction |
| `slurm/train_faantra.sbatch` | SLURM job for GPU training |
| `slurm/eval_faantra.sbatch` | SLURM job for evaluation |

---

## Environment Setup

### FAANTRA Environment

```bash
cd /home/mseo/CornerTactics/FAANTRA
source venv/bin/activate
```

### FFmpeg Module (for video processing)

```bash
module load GCCcore/12.3.0 FFmpeg/6.0-GCCcore-12.3.0
```

---

## References

- [FAANTRA Paper (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Dalal_Action_Anticipation_from_SoccerNet_Football_Video_Broadcasts_CVPRW_2025_paper.html)
- [FAANTRA Repository](https://github.com/MohamadDalal/FAANTRA)
- [TacticAI Paper](https://www.nature.com/articles/s41467-024-45965-x) - Labeling methodology reference
- [SoccerNet Challenge](https://www.soccer-net.org/challenges/2026)
