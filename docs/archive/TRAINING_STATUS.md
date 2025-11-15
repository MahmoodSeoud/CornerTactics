# Training Status Report

## Current Situation (November 2025)

### Data Availability

**Issue Identified**: The full 34,049 corner dataset is not available in JSON format.

**What we have**:
- `corner_transition_matrix.csv`: Transition probabilities computed from 34,049 corners
- `corner_sequences_detailed.json`: Sample of 100 corners with full event sequences
- `corner_transition_report.md`: Statistical summary of all 34,049 corners

**Why this happened**:
- The full 34K dataset would be ~140MB in JSON format
- Only 100 corners were saved as a representative sample
- The transition matrix captures the aggregate statistics from all 34K

### Options for Training

#### Option 1: Small-Scale Proof of Concept (100 corners)
**Pros**:
- Quick to run (< 5 minutes)
- Tests the full pipeline
- Validates code correctness

**Cons**:
- Not enough data for meaningful learning
- Cannot compare to TacticAI's 7K dataset
- Results will not be generalizable

#### Option 2: Regenerate Full 34K Dataset
**Pros**:
- Full dataset for robust training
- Can compare to TacticAI baseline
- Statistically meaningful results

**Cons**:
- Requires re-running the analysis script (12+ hours)
- Large memory requirements (16GB+)
- Storage of ~140MB JSON file

#### Option 3: Use Existing Graph Dataset (5,814 graphs)
**Location**: `data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl`

**Pros**:
- Already processed with 14-dim features
- Includes receiver labels (100% coverage)
- 3-class outcome labels
- Temporal augmentation (5× data)

**Cons**:
- Contains engineered features (not "raw")
- Different from the event-stream approach
- Already augmented (not original data)

### Recommended Path Forward

1. **Immediate**: Use the 100-corner sample to validate the training pipeline
2. **Short-term**: Regenerate the full 34K dataset overnight
3. **Alternative**: Pivot to using the 5,814 graph dataset for meaningful results

### Training Pipeline Status

✅ **Completed**:
- `scripts/train_raw_baseline.py`: Full training pipeline
- `scripts/slurm/train_raw_baseline.sh`: SLURM submission script
- `docs/RAW_DATA_LOCATIONS.md`: Data location guide

⚠️ **Issue**:
- Training fails due to insufficient data (100 corners)
- All 100 corners have "Unknown" as next event (data quality issue)
- No receiver information in the sample

❌ **Blocked**:
- Cannot train meaningful models without full dataset
- Cannot perform ablation studies
- Cannot compare to baselines

### Error Analysis

The training script failed with:
```
ValueError: a must be greater than 0 unless no samples are taken
```

**Root cause**:
1. Only 100 corners in the dataset
2. All have "Unknown" as next event type
3. No receiver information (-1 for all)
4. Insufficient variety for train/val/test splits

### Next Steps

#### To Generate Full Dataset:

```bash
# Create script to extract all 34K corners
python scripts/generate_full_corner_dataset.py

# Or re-run the original analysis
sbatch scripts/slurm/analyze_statsbomb_raw_all.sh --save-all
```

#### To Use Graph Dataset Instead:

```python
import pickle

# Load the graph dataset
with open('data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl', 'rb') as f:
    graphs = pickle.load(f)

print(f"Total graphs: {len(graphs)}")  # 5,814
print(f"Has receiver labels: {all(g.receiver_node_index >= 0 for g in graphs)}")  # True
print(f"Has outcome labels: {all(hasattr(g, 'outcome_class') for g in graphs)}")  # True
```

### Recommendation

Given time constraints, I recommend:

1. **Use the 5,814 graph dataset** for immediate training
   - Already has all features and labels
   - Sufficient size for meaningful results
   - Can start training immediately

2. **Regenerate 34K raw dataset** in parallel
   - Run overnight on SLURM
   - Use for "raw baseline" comparison
   - Validates the event-stream approach

3. **Document the data limitation** clearly
   - Current results use 5,814 augmented graphs
   - Not the originally planned 34K raw events
   - Still larger than TacticAI's 7K dataset

---

**Updated**: November 12, 2025
**Status**: Awaiting decision on dataset choice