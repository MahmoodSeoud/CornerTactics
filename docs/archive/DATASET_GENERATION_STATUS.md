# Dataset Generation Status

## Date: November 12, 2025

### Problem Solved
Successfully fixed the corner extraction script to properly detect and extract corner kicks from StatsBomb Open Data.

### Issue Resolution
**Original Problem**: The extraction script was finding 0 corners across all competitions.

**Root Cause**: The script was using `sb.events(match_id, split=True, flatten_attrs=False)` which changes the data format and structure, making the corner detection logic fail.

**Solution**: Modified to use `sb.events(match_id)` for the main events data (keeping the expected format), then separately fetching freeze frames if needed.

### Current Status

**SLURM Job**: #33534 (Running on desktop2)
- Started: 11:34:47 CET
- Status: Successfully extracting corners
- Progress: Over 1,090 corners found so far (processing La Liga 2018/2019)
- Expected: ~34,000 corners total
- Duration: Estimated 8-10 hours

### Files Being Generated

Location: `/home/mseo/CornerTactics/data/analysis/`

1. **corner_sequences_full.json** - Complete corner data with event sequences
2. **corner_sequences_full.parquet** - Parquet format for efficient loading
3. **corner_sequences_full.csv** - CSV format for compatibility
4. **dataset_statistics.json** - Summary statistics

### Next Steps

Once the extraction completes:

1. **Verify Dataset Quality**
   ```bash
   python -c "
   import json
   with open('data/analysis/corner_sequences_full.json', 'r') as f:
       corners = json.load(f)
   print(f'Total corners: {len(corners)}')
   "
   ```

2. **Train Baseline Models**
   ```bash
   sbatch scripts/slurm/train_raw_baseline.sh
   ```

3. **Run Ablation Studies**
   - Feature importance analysis
   - Different feature subsets
   - Model comparison

### Data Structure

Each corner in the dataset contains:
- **corner_event**: The corner kick event with all StatsBomb features
- **next_events**: Sequence of 15 events following the corner
- **freeze_frame**: Player positions at corner time (if available)
- **outcomes**: Labels for goal, shot, clearance, possession
- **receiver_info**: Player who received the corner pass

### Key Features Extracted

**Raw Features (23 dimensions)**:
- Location (x, y)
- Pass details (end_location, angle, height, length)
- Technique and body part
- Cross/cut-back indicators
- Aerial won, outcome
- Team and player IDs

**Event Sequence Features**:
- Next 15 events after corner
- Event types and transitions
- Time to key events (shot, clearance)
- Possession changes

### Monitoring Commands

```bash
# Check job status
squeue -u $USER

# View extraction progress
tail -f logs/gen_34k_corners_33534.out

# Check for errors
tail logs/gen_34k_corners_33534.err

# See corner count in real-time (once JSON starts being written)
ls -lh data/analysis/corner_sequences_full.json
```

### Technical Details

**Fixed Code Segment**:
```python
# OLD (incorrect)
events = sb.events(match_id=match_id, split=True, flatten_attrs=False)
if "threesixty" in events:
    events_df = events["event"]

# NEW (correct)
events_df = sb.events(match_id=match_id)
# Get freeze frames separately if needed
```

This ensures the events DataFrame maintains the expected column structure where corners are Pass events with `pass_type == 'Corner'`.

---

**Status**: Extraction in progress, expected completion in 8-10 hours
**Action Required**: None - wait for completion then proceed with training