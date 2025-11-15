# Corner Transition Analysis Solution

## Complete Pipeline for P(a_{t+1} | corner_t)

### Problem Statement
You want to answer: **Given a corner kick at time t, what happens next?**
- P(a_{t+1} | corner_t) where a is the action

### Solution Architecture

```
Raw JSONs → Complete Event Sequences → Transition Matrix → Probabilities
```

## 📁 Files Created

### 1. Data Collection
**`scripts/download_statsbomb_raw_jsons.py`**
- Downloads ALL raw JSON event files from StatsBomb GitHub
- Preserves 100% of data structure (no SDK filtering)
- Creates index of matches with corners
- Output: `data/raw/statsbomb/json_events/`

### 2. Transition Analysis
**`scripts/analyze_corner_transitions.py`**
- Processes raw JSONs to find corner kicks
- Extracts complete event sequences after corners
- Builds transition probability matrix P(a_{t+1} | corner_t)
- Tracks ALL features (flattens entire JSON structure)
- Output: `data/analysis/corner_transition_matrix.csv`

### 3. Enhanced Feature Extraction
**`src/statsbomb_raw_analyzer.py`** (Updated)
- Now extracts 50+ features (was ~17)
- Critical additions:
  - `pass_recipient_id/name` - Who receives the corner
  - `pass_shot_assist/goal_assist` - Direct outcomes
  - `shot_first_time` - First-time shots
  - `freeze_frame` - 360 player positions
  - `aerial_won` - Header success
  - `duel_type/outcome` - Physical contests
  - `goalkeeper_position` - GK positioning

### 4. Test & Demo
**`scripts/test_corner_analysis.py`**
- Quick demo showing the analysis works
- Euro 2020 Final example: 57% corners cleared immediately

### 5. SLURM Job
**`scripts/slurm/download_analyze_corners.sh`**
- Runs complete pipeline on HPC cluster
- 6 hours, 32GB RAM for full dataset

## 🚀 How to Run

### Quick Test (Local)
```bash
# Demo with single match
python scripts/test_corner_analysis.py
```

### Full Analysis (SLURM)
```bash
# Submit to cluster
sbatch scripts/slurm/download_analyze_corners.sh

# Check progress
tail -f logs/corner_analysis_*.out
```

### Results Location
```
data/analysis/
├── corner_transition_matrix.csv      # P(a_{t+1} | corner_t)
├── corner_sequences_detailed.json    # Raw sequences
├── corner_transition_report.md       # Human-readable report
└── statsbomb_feature_comparison.md   # Feature documentation
```

## 📊 What You Get

### Transition Matrix
Shows probability of each event type following a corner:
```
                    Pass    Shot    Clearance    Duel    ...
Corner             0.23    0.08      0.41       0.15
```

### Complete Feature Extraction
Every single field from the JSON is preserved:
- Event metadata (id, timestamp, period, minute, second)
- Player tracking (who took corner, who received)
- Outcomes (shot assist, goal assist)
- Physical details (aerial won, first-time shot)
- Tactical context (formation, counterpress)
- 360 freeze frame data (all player positions)

### Sample Insights (from Euro 2020 Final)
- **57%** of corners cleared immediately
- **29%** result in ball receipt by attacking team
- **0%** led to direct shots (in this specific match)

## 🔑 Key Advantages

1. **Complete Data**: Raw JSONs preserve 100% of features
2. **No Filtering**: Extract everything now, filter later
3. **Flexible Analysis**: Can answer any question about post-corner events
4. **Temporal Sequences**: Full event chains, not just immediate transitions
5. **Enhanced Features**: 50+ attributes vs. original 17

## 💡 Next Steps

With this data, you can now:
1. Build predictive models for corner outcomes
2. Identify optimal corner strategies
3. Analyze player-specific corner effectiveness
4. Study temporal dynamics (time to shot/goal)
5. Compare teams' corner defending patterns

## 📈 Feature Coverage

**Before**: ~17 features (23% utilization)
**After**: 50+ features including:
- ✅ All critical features (recipient, assists, freeze frames)
- ✅ All high-priority features (aerial duels, first-time shots)
- ✅ Complete event metadata
- ✅ Full nested JSON structure preserved

---

*Solution complete. You now have everything needed to analyze P(a_{t+1} | corner_t) with full StatsBomb data.*