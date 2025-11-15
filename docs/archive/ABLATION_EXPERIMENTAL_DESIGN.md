# Ablation Study: Feature Configuration Experimental Design

**Date**: November 2025
**Purpose**: Compare different feature sets to understand data leakage impact and feature engineering value

---

## Overview

This experimental design compares three feature configurations for corner kick receiver prediction, enabling:

1. **Leakage Impact Analysis**: Quantify how much data leakage inflates performance
2. **Fair Baseline**: Match TacticAI methodology for apple-to-apple comparison
3. **Feature Engineering Value**: Demonstrate improvements from domain knowledge

---

## Three Configurations

### Configuration 1: "Out-of-the-Box StatsBomb"

**Purpose**: Baseline using raw StatsBomb event features as-is

**Features (23 total)**:
```python
FEATURES_CONFIG_1 = [
    # Temporal
    "minute", "second", "period", "duration",

    # Spatial (INCLUDES LEAKAGE)
    "location_x", "location_y",
    "end_location_x",      # ⚠️ DATA LEAKAGE: Where ball lands
    "end_location_y",      # ⚠️ DATA LEAKAGE: Where ball lands

    # Pass characteristics (INCLUDES LEAKAGE)
    "pass_length",         # ⚠️ Derived from end_location
    "pass_angle",          # ⚠️ Derived from end_location
    "inswinging", "switch", "shot_assist",
    "height_id", "body_part_id", "technique_id",

    # Game context
    "team_id", "player_id", "position_id",
    "possession_team_id", "play_pattern_id",
    "possession", "related_events_count"
]
```

**Why Include This?**
- Shows what naive users get "out of the box"
- Quantifies data leakage impact
- Common mistake in sports analytics research

**Expected Performance**:
- Top-1 Accuracy: ~99% (current results)
- Top-3 Accuracy: ~99.6%

**Limitations**:
- **Not fair comparison** to TacticAI (uses future information)
- **Not deployable** (requires knowing where ball lands before prediction)

---

### Configuration 2: "Fair Comparison (TacticAI-matched)"

**Purpose**: Remove data leakage, match TacticAI methodology for fair comparison

**Features (14 total - NO post-kick information)**:
```python
FEATURES_CONFIG_2 = [
    # Temporal context
    "minute", "second", "period",

    # Corner location ONLY (not destination)
    "location_x",          # Corner kick position (120, 0) or (120, 80)
    "location_y",

    # Corner characteristics (known pre-kick)
    "inswinging",          # Corner type (in/out-swinging)
    "height_id",           # Intended height (ground/low/high)
    "body_part_id",        # Kick technique

    # Game context
    "team_id",             # Which team taking corner
    "player_id",           # Corner taker ID
    "possession_team_id",  # Possession tracking
    "play_pattern_id",     # How play developed
    "possession",          # Possession count
    "related_events_count" # Sequence complexity
]

# REMOVED (data leakage):
# - end_location_x/y     ❌ Where ball lands (not known pre-kick)
# - pass_length          ❌ Derived from end_location
# - pass_angle           ❌ Derived from end_location
# - switch               ❌ May depend on outcome
# - shot_assist          ❌ Post-kick information
# - technique_id         ❌ May contain outcome info
# - position_id          ❌ Not in TacticAI
```

**Why This Matters?**
- **Fair comparison** to TacticAI (only pre-kick information)
- **Deployable** (all features known before kick taken)
- **Standard benchmark** for future work

**Expected Performance**:
- Top-1 Accuracy: ~50-60%
- Top-3 Accuracy: ~70-85% (comparable to TacticAI's 78.2%)

**Comparison to TacticAI**:
| Feature Type | TacticAI | Config 2 (Ours) |
|--------------|----------|-----------------|
| Player positions (freeze frame) | ✅ 22 players × [x,y,vx,vy,height,weight] | ❌ Not in event data |
| Corner kick event features | ❌ Not used | ✅ 14 event features |
| Pre-kick only | ✅ Yes | ✅ Yes |
| Data source | Tracking data | StatsBomb events |

---

### Configuration 3: "Enhanced Features" (Research Contribution)

**Purpose**: Add engineered features based on domain knowledge to improve performance

**Features (Config 2 + 12 engineered = 26 total)**:
```python
FEATURES_CONFIG_3 = FEATURES_CONFIG_2 + [
    # Player positioning features (from freeze frame)
    "num_attackers_in_box",      # Attacking players in penalty area
    "num_defenders_in_box",      # Defending players in penalty area
    "attackers_to_defenders_ratio", # Numerical advantage
    "avg_defender_distance_to_goal", # Defensive line depth

    # Tactical features
    "corner_side",               # Left (0,80) vs Right (0,0) corner
    "defensive_formation",       # Zonal vs man-marking (inferred)
    "attacking_players_near_posts", # Near/far post runners

    # Historical/contextual features
    "corner_taker_success_rate", # Player's historical receiver accuracy
    "team_corner_conversion_rate", # Team's corner efficiency
    "match_score_differential",  # Game state (winning/losing)
    "time_remaining",            # Late game vs early
    "previous_corner_outcome"    # Momentum/pattern
]
```

**Engineering Methodology**:
- Extract from freeze frame player positions (already available)
- Compute from historical event data (StatsBomb)
- Infer from game context and patterns

**Why This Matters?**
- Shows **value of domain knowledge** over raw features
- **Your research contribution** beyond TacticAI
- Tests if event-based features can compensate for lack of tracking data

**Expected Performance**:
- Top-1 Accuracy: ~55-70%
- Top-3 Accuracy: ~75-90% (goal: beat TacticAI's 78.2%)

---

## Experimental Protocol

### Dataset
- **Source**: `data/analysis/corner_sequences_full.json` (21,656 corners)
- **Receiver Labels**: First player to touch ball after corner (fixed extraction)
- **Train/Val/Test Split**: 70/15/15 (stratified by outcome class)

### Models
For each configuration, train:
1. **XGBoost** (gradient boosted trees)
2. **MLP** (multi-layer perceptron)
3. **Random Baseline** (for reference)

### Hyperparameter Search
- Use same search space for all configurations (fairness)
- Validate on validation set
- Report final metrics on held-out test set

### Evaluation Metrics
- **Top-1, Top-3, Top-5 Accuracy** (main metrics for comparison)
- **Macro F1** (handle class imbalance)
- **Per-class accuracy** (which receivers predicted well)

---

## Ablation Analysis

### Primary Comparisons

**1. Leakage Impact Analysis**
```
Config 1 (with leakage) vs Config 2 (fair)
└─> Quantifies performance drop from removing end_location
└─> Expected drop: 99% → 75% Top-3 (~24% inflation from leakage)
```

**2. TacticAI Benchmark**
```
Config 2 (ours) vs TacticAI (78.2% Top-3)
└─> Fair comparison: event data vs tracking data
└─> Can event features compensate for no player velocities?
```

**3. Feature Engineering Value**
```
Config 3 (enhanced) vs Config 2 (fair)
└─> Value of domain knowledge features
└─> Goal: Config 3 > TacticAI > Config 2
```

### Secondary Ablations

**Individual Feature Importance**:
- Remove one feature at a time from Config 3
- Identify which engineered features matter most
- Guide future feature engineering

**Feature Groups**:
- Position features only
- Tactical features only
- Historical features only
- Test interactions between groups

---

## Implementation Plan

### Step 1: Create Feature Extraction Functions

**File**: `src/feature_extractors.py`

```python
class FeatureExtractor:
    """Extract features for different configurations."""

    def extract_config1(self, corner: dict) -> np.ndarray:
        """Out-of-box StatsBomb features (with leakage)."""
        pass

    def extract_config2(self, corner: dict) -> np.ndarray:
        """Fair comparison features (no leakage)."""
        pass

    def extract_config3(self, corner: dict, freeze_frame: dict) -> np.ndarray:
        """Enhanced features (domain knowledge)."""
        pass
```

### Step 2: Training Script

**File**: `scripts/train_ablation_study.py`

```python
configs = {
    "config1_outofbox": extract_config1,
    "config2_fair": extract_config2,
    "config3_enhanced": extract_config3
}

for config_name, extractor in configs.items():
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"{'='*60}")

    # Extract features
    X, y = prepare_data(corners, extractor)

    # Train models
    for model_type in ["xgboost", "mlp"]:
        model = train_model(X, y, model_type)
        results = evaluate_model(model, X_test, y_test)
        save_results(config_name, model_type, results)
```

### Step 3: Results Format

**CSV**: `results/ablation_feature_configs.csv`
```csv
config,model,top1_acc,top3_acc,top5_acc,macro_f1,num_features
config1_outofbox,XGBoost,0.996,0.997,0.997,0.850,23
config2_fair,XGBoost,0.520,0.782,0.850,0.450,14
config3_enhanced,XGBoost,0.650,0.820,0.880,0.520,26
```

**JSON**: `results/ablation_leakage_analysis.json`
```json
{
  "leakage_impact": {
    "top3_drop": 0.215,
    "percentage_drop": "21.5%",
    "interpretation": "end_location features inflate performance by ~22%"
  },
  "tacticai_comparison": {
    "config2_vs_tacticai": "+0.002 (within margin of error)",
    "config3_vs_tacticai": "+0.038 (4% improvement)",
    "conclusion": "Enhanced features beat TacticAI despite no tracking data"
  }
}
```

---

## Expected Results Table

| Configuration | Top-1 Acc | Top-3 Acc | Features | Leakage? | Fair to TacticAI? |
|---------------|-----------|-----------|----------|----------|-------------------|
| **Config 1** (Out-of-box) | 99.6% | 99.7% | 23 | ⚠️ Yes | ❌ No |
| **Config 2** (Fair) | 52.0% | 78.0% | 14 | ✅ No | ✅ Yes |
| **TacticAI** (Baseline) | - | **78.2%** | - | ✅ No | - |
| **Config 3** (Enhanced) | 65.0% | **82.0%** | 26 | ✅ No | ✅ Yes |

**Key Insights**:
1. Removing leakage drops Top-3 from 99.7% → 78.0% (~22% drop)
2. Config 2 matches TacticAI despite different data source
3. Config 3 beats TacticAI by +4% through feature engineering

---

## Research Contributions

### Paper Storyline

**Section 1: Reproducibility Analysis**
- "We first attempt to reproduce results using raw StatsBomb features"
- "Achieve 99.6% Top-1 accuracy - suspiciously high"
- "Identify data leakage from end_location features"

**Section 2: Fair Baseline**
- "Remove leakage, match TacticAI methodology"
- "Achieve 78.0% Top-3 - matches TacticAI's 78.2%"
- "Validates that event data can substitute for tracking data"

**Section 3: Our Contribution**
- "Engineer domain knowledge features"
- "Achieve 82.0% Top-3 - beats TacticAI by 4%"
- "Demonstrates value of tactical features even without player velocities"

---

## Next Steps

1. ✅ Design complete (this document)
2. ⏳ Implement feature extractors for all 3 configs
3. ⏳ Create training script with config switching
4. ⏳ Run experiments (estimate: 2-4 hours on cluster)
5. ⏳ Generate ablation results CSVs/JSONs
6. ⏳ Create comparison visualizations
7. ⏳ Write up findings in paper format

---

## Files to Create

**Code**:
- `src/feature_extractors.py` - Feature extraction functions
- `scripts/train_ablation_study.py` - Main training script
- `scripts/slurm/run_ablation_study.sh` - SLURM job script

**Results**:
- `results/ablation_feature_configs.csv` - Main results table
- `results/ablation_leakage_analysis.json` - Leakage impact quantification
- `results/ablation_feature_importance.csv` - Individual feature contributions

**Documentation**:
- `docs/ABLATION_RESULTS.md` - Full results write-up
- `docs/FEATURE_ENGINEERING_GUIDE.md` - How Config 3 features were designed

---

## Timeline

- **Day 1**: Implement feature extractors (2 hours)
- **Day 2**: Create training script and test locally (2 hours)
- **Day 3**: Submit SLURM job for all configs (4 hours runtime)
- **Day 4**: Analyze results and create visualizations (2 hours)
- **Day 5**: Write up findings (3 hours)

**Total**: ~13 hours of work + 4 hours compute time
