# Phase 2 Plan: SoccerNet GSR Corner Extraction

## Prerequisites (already satisfied)

- [x] sn-gamestate repo cloned at `/home/mseo/CornerTactics/sn-gamestate/`
- [x] tracklab framework at `/home/mseo/CornerTactics/tracklab/`
- [x] Conda env `sn-gamestate` with Python 3.9, PyTorch 1.13.1, mmcv 2.0.1
- [x] 4,836 corner clips at `FAANTRA/data/corners/clips/corner_XXXX/720p.mp4` (115 GB)
- [x] `corner_dataset.json` with timing, outcome, team, visibility for all corners
- [x] `soccernet_gsr_adapter.py` (412 lines) and `extract_soccernet_gsr.py` (138 lines) ready
- [x] SLURM cluster with GPU nodes (cn7: A100 80GB, cn5/cn6: V100s)

## Overview

Run sn-gamestate's TrackLab pipeline on corner kick video clips to get per-frame player pitch
coordinates, then convert to our unified JSON format via the existing adapter.

**Pipeline**: video clips → TrackLab (detect → ReID → track → calibrate → pitch project → team assign) → .pklz state → extract script → JSON → existing adapter → unified format

**Three phases**: pilot (50 clips) → quality review → scale (500-4229 clips)

---

## Step 1: Create a custom dataset config for corner clips

**File:** `sn-gamestate/sn_gamestate/configs/dataset/corner_clips.yaml`

The `ExternalVideo` wrapper (tracklab) already supports a directory of .mp4 files.
However, processing 4836 videos in one go is impractical. We need per-batch processing.

Create a Hydra config override that:
- Uses `_target_: tracklab.wrappers.ExternalVideo`
- Points `video_path` to a batch directory of symlinked clips
- Sets `eval_set: "val"` (required by ExternalVideo which creates a val set)

```yaml
defaults:
  - default

_target_: tracklab.wrappers.ExternalVideo

video_path: "${data_dir}/corner_batch"
```

**Test criteria:** `tracklab -cn soccernet dataset=corner_clips` initializes without error.

---

## Step 2: Create the batch preparation script

**File:** `tracking_extraction/scripts/prepare_gsr_batches.py`

This script:
1. Reads `corner_dataset.json`
2. Filters to visible corners only
3. Creates batch directories under `sn-gamestate/data/corner_batches/batch_XXXX/`
4. Each batch contains N symlinks: `corner_XXXX.mp4 -> FAANTRA/data/corners/clips/corner_XXXX/720p.mp4`
5. Writes a manifest JSON mapping corner_id → batch_id → video_path

**Arguments:**
- `--corner-dataset`: path to `FAANTRA/data/corners/corner_dataset.json`
- `--clips-dir`: path to `FAANTRA/data/corners/clips/`
- `--output-dir`: path to `sn-gamestate/data/corner_batches/`
- `--batch-size`: clips per batch (default: 10 for pilot, 50 for production)
- `--max-corners`: limit for pilot (default: None = all visible)
- `--pilot`: shortcut flag, sets batch-size=10, max-corners=50

**Output structure:**
```
sn-gamestate/data/corner_batches/
├── batch_manifest.json      # {corner_id: batch_idx, ...}
├── batch_0000/
│   ├── corner_0001.mp4 -> /abs/path/clips/corner_0001/720p.mp4
│   ├── corner_0005.mp4 -> /abs/path/clips/corner_0005/720p.mp4
│   └── ...
├── batch_0001/
│   └── ...
```

**Test criteria:**
- Pilot: 50 visible corners across 5 batch dirs (10 each)
- All symlinks resolve to existing .mp4 files
- Manifest JSON is valid and complete

---

## Step 3: Create the SLURM job script for GSR inference

**File:** `scripts/slurm/run_gsr_batch.sbatch`

SLURM array job, one task per batch. Each task:
1. Activates the `sn-gamestate` conda env
2. Loads CUDA module
3. Sets TrackLab config overrides via Hydra CLI:
   - `dataset=corner_clips`
   - `dataset.video_path=sn-gamestate/data/corner_batches/batch_${SLURM_ARRAY_TASK_ID}`
   - `state.save_file=sn-gamestate/outputs/corner_states/batch_${SLURM_ARRAY_TASK_ID}.pklz`
   - `visualization.cfg.save_videos=False` (skip video rendering to save time)
   - `nvid=-1` (process all videos in batch)
4. Runs `tracklab -cn soccernet <overrides>`

**SLURM config:**
```bash
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-4   # pilot: 5 batches
```

**Estimated time per batch (10 clips × 30s × 25fps = 7,500 frames at ~1.1 FPS):**
- ~1.9 hours on A100, ~3-4 hours on V100

**Test criteria:**
- Pilot batch 0 completes without error
- `sn-gamestate/outputs/corner_states/batch_0000.pklz` exists and is non-empty

---

## Step 4: Create the pklz → JSON extraction script

**File:** `tracking_extraction/scripts/extract_pklz_to_json.py`

This is the critical bridge between TrackLab output and our existing adapter.

The script:
1. Opens each `batch_XXXX.pklz` (zipfile containing pickled DataFrames per video)
2. For each video DataFrame, extracts columns:
   - `image_id` (frame index)
   - `track_id` (player tracking ID)
   - `bbox_pitch` (dict with `x_bottom_middle`, `y_bottom_middle`)
   - `role_detection` (from ReID: "player", "goalkeeper", "referee", "other")
   - `team` (from team_side: "left", "right")
3. Converts to our adapter's expected JSON format:
   ```json
   [
     {
       "image_id": 0,
       "track_id": 1,
       "attributes": {
         "role": "player",
         "team": "left"
       },
       "bbox_pitch": {
         "x_bottom_middle": 52.3,
         "y_bottom_middle": 34.1
       }
     }
   ]
   ```
4. Saves one JSON per corner: `tracking_extraction/output/gsr_raw/corner_XXXX.json`

**Arguments:**
- `--pklz-dir`: path to `sn-gamestate/outputs/corner_states/`
- `--batch-manifest`: path to `sn-gamestate/data/corner_batches/batch_manifest.json`
- `--output-dir`: path to `tracking_extraction/output/gsr_raw/`

**Key handling:**
- The pklz stores DataFrames with video_id as key. The video_id maps to the video filename
  (e.g., "corner_0001" from the symlink name)
- The `bbox_pitch` column contains dict values (may be None for failed calibration frames)
- Skip detections where `bbox_pitch` is None
- The `role_detection` or final aggregated role comes from `tracklet_agg` module
- The `team` column comes from `team_side` module, values "left"/"right"

**Test criteria:**
- At least one `corner_XXXX.json` is produced per batch
- JSON structure matches adapter expectation
- Each JSON has detections with valid pitch coordinates

---

## Step 5: Run existing adapter pipeline on GSR JSON output

This uses the **already-implemented** code. No new code needed.

```bash
source FAANTRA/venv/bin/activate
python -m tracking_extraction.scripts.extract_soccernet_gsr parse \
    --gsr-output-dir tracking_extraction/output/gsr_raw \
    --clip-list tracking_extraction/output/gsr_clips.json \
    --output-dir tracking_extraction/output/soccernet_gsr
```

**But first**, we need to prepare the clip list (Step 5a):

```bash
python -m tracking_extraction.scripts.extract_soccernet_gsr prepare \
    --output-list tracking_extraction/output/gsr_clips.json \
    --max-corners 50  # pilot
```

**Adapter modifications needed:**
1. **Team resolution** (line 330 TODO): Implement the corner-taker-near-flag heuristic.
   At the delivery frame, find the player closest to the corner flag coordinates
   (0,0), (0,68), (105,0), or (105,68). That player's GSR team label ("left"/"right")
   = attacking team. Map accordingly.

2. **Match ID normalization**: The `match_dir` field from `corner_dataset.json` is a long
   path like "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley".
   For graph splitting, we need a stable match ID. Use the match directory basename.

**Test criteria:**
- Unified JSON files in `tracking_extraction/output/soccernet_gsr/`
- Each has valid player positions, velocities, outcome labels
- Quality stats logged (mean players/frame, position ranges)

---

## Step 6: Quality validation on pilot

**File:** No new file — use existing `tracking_extraction/validate.py`

Run validation checks on the pilot output:
1. Position ranges: all x in [0, 105], y in [0, 68]
2. Velocity ranges: max player speed < 12 m/s, max ball speed < 35 m/s
3. Player count per frame: log min/max/mean
4. Manual spot-check: verify 5 random corners visually
   - Is the delivery frame approximately correct?
   - Are player positions reasonable (near the goal, not scattered)?
   - Is the outcome label correct?

**Decision gate:** If fewer than 30% of pilot corners pass quality filters,
GSR quality on corners is too poor to proceed. Document and stop.

**Test criteria:**
- Validation report generated
- At least 15/50 pilot corners produce valid unified JSON

---

## Step 7: Scale to full dataset (if pilot passes)

1. Re-run Step 2 with `--batch-size 50` and no `--max-corners` limit
   - 4,229 visible corners → 85 batches
2. Re-run Step 3 SLURM job with `--array=0-84`
   - Estimated: 85 batches × ~10 GPU-hours each = ~850 GPU-hours total
   - With 4 GPU jobs in parallel: ~9 days wall time
   - Or: use scavenge partition for opportunistic scheduling
3. Run Steps 4 and 5 on full output
4. Run Step 6 validation on full output

---

## Step 8: Consolidate all sources and rebuild graphs

```bash
# Consolidate: SkillCorner + DFL + SoccerNet GSR
python -m tracking_extraction.scripts.consolidate_dataset \
    --skillcorner-dir tracking_extraction/output/skillcorner \
    --dfl-dir tracking_extraction/output/dfl \
    --soccernet-gsr-dir tracking_extraction/output/soccernet_gsr \
    --output-dir tracking_extraction/output/unified

# Convert to USSF graphs
python -m tracking_extraction.scripts.build_graph_dataset \
    --input-dir tracking_extraction/output/unified \
    --output-path transfer_learning/data/multi_source_corners_dense.pkl
```

**Consolidation script modification needed:**
- Add `--soccernet-gsr-dir` argument to `consolidate_dataset.py`
- Include GSR corners in the merge logic

**Test criteria:**
- Unified dataset includes all 3 sources
- Feature distribution report shows GSR data quality metrics
- Graph conversion succeeds with correct shapes

---

## Files to create/modify

| Action | File | ~Lines |
|--------|------|--------|
| CREATE | `sn-gamestate/sn_gamestate/configs/dataset/corner_clips.yaml` | ~8 |
| CREATE | `tracking_extraction/scripts/prepare_gsr_batches.py` | ~120 |
| CREATE | `scripts/slurm/run_gsr_batch.sbatch` | ~50 |
| CREATE | `tracking_extraction/scripts/extract_pklz_to_json.py` | ~150 |
| MODIFY | `tracking_extraction/soccernet_gsr_adapter.py` | ~30 lines (team resolution) |
| MODIFY | `tracking_extraction/scripts/consolidate_dataset.py` | ~15 lines (add GSR source) |

**Total new code:** ~350 lines + 15 lines modifications

---

## Execution order

1. Step 1 + Step 2 (config + batch prep) — can run immediately
2. Step 5a (prepare clip list) — can run immediately
3. Step 3 (submit SLURM pilot job) — needs GPU, ~2-4 hours
4. **WAIT for SLURM job to complete**
5. Step 4 (extract pklz → JSON)
6. Step 5 (run adapter) — includes team resolution fix
7. Step 6 (quality validation)
8. **DECISION GATE**: proceed to scale or stop
9. Steps 7 + 8 (scale + consolidate)

---

## Risk mitigations

| Risk | Mitigation |
|------|------------|
| GSR fails on corner clips (different from standard broadcast) | Pilot of 50 clips first; corner clips are standard 720p broadcast footage |
| Camera calibration fails (clustered players occlude lines) | Accept partial data; adapter already filters frames with <10 players |
| Team assignment wrong | Corner-taker-near-flag heuristic; fallback to random 50/50 |
| Insufficient disk space for pklz files | Each pklz ~50-200 MB per batch; 85 batches = ~4-17 GB |
| SLURM queue contention | Use scavenge partition for pilot; acltr for production |
| sn-gamestate env incompatible with our code | pklz→JSON bridge runs in sn-gamestate env; adapter runs in FAANTRA env |
