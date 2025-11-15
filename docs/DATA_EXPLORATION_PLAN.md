# Data Exploration Plan: Guide Ablation Design with Analysis

**Date**: November 2025
**Purpose**: Explore dataset systematically to identify which features to ablate
**Principle**: Let the data tell you which configs to test

---

## Overview

Before running ablation experiments, we need to understand:

1. **Which features are leaking information?** (suspiciously high importance)
2. **Which features are redundant?** (highly correlated with each other)
3. **Which features are weak?** (low importance, can be removed)
4. **Which features correlate with receiver?** (important to keep)

This exploration will guide which ablation configs to test.

---

## Exploration Workflow

### Phase 1: Train Baseline Model

**Goal**: Get baseline performance and feature importance

**Input**:
- Dataset: `data/analysis/corner_sequences_full.json` (21,656 corners)
- Features: All 23 current features (including potential leakage)

**Script**: Use existing `scripts/train_raw_baseline.py`

**Run**:
```bash
sbatch scripts/slurm/train_raw_baseline.sh
```

**Outputs**:
- `results/task1_receiver_prediction.csv` - Baseline receiver accuracy
- `results/task2_outcome_classification.csv` - Baseline outcome accuracy
- `results/feature_importance_event.csv` - XGBoost feature importance
- `results/learning_curves_xgboost_event.csv` - Training curves
- `models/final/xgboost_receiver_best.json` - Trained model

**Status**: ✅ Already done (Nov 13)

---

### Phase 2: Analyze Feature Importance

**Goal**: Identify features by importance tier

**Input**: `results/feature_importance_event.csv`

**Analysis**:
```python
import pandas as pd

# Load feature importance
fi = pd.read_csv('results/feature_importance_event.csv')

# Define tiers
tier1 = fi[fi['importance_percentage'] > 10]  # Top features
tier2 = fi[(fi['importance_percentage'] >= 5) & (fi['importance_percentage'] <= 10)]
tier3 = fi[(fi['importance_percentage'] >= 1) & (fi['importance_percentage'] < 5)]
tier4 = fi[fi['importance_percentage'] < 1]  # Weak features

print("Tier 1 (>10%):", list(tier1['feature_name']))
print("Tier 2 (5-10%):", list(tier2['feature_name']))
print("Tier 3 (1-5%):", list(tier3['feature_name']))
print("Tier 4 (<1%):", list(tier4['feature_name']))
```

**Expected Output**:
```
Tier 1 (>10%): ['end_location_x', 'end_location_y', 'player_id']
Tier 2 (5-10%): ['team_id', 'pass_angle', 'pass_length']
Tier 3 (1-5%): ['minute', 'location_x', 'location_y', ...]
Tier 4 (<1%): ['duration', 'second', 'period', ...]
```

**Flags to Investigate**:
- ⚠️ If `end_location_x/y` are in Tier 1 → **Suspected leakage**
- ⚠️ If derived features (pass_length, pass_angle) are high → **Check if redundant**

**Save**:
- `results/exploration/feature_tiers.json`

**Script**: `scripts/analyze_feature_importance.py`

---

### Phase 3: Compute Feature Correlations

**Goal**: Find redundant features (high correlation with each other)

**Analysis**:
```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/processed/corner_features.csv')

# Compute correlation matrix
corr_matrix = df.corr()

# Find high correlations (r > 0.8)
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

print(f"Found {len(high_corr)} highly correlated pairs:")
for pair in high_corr:
    print(f"  {pair['feature1']} <-> {pair['feature2']}: r={pair['correlation']:.3f}")
```

**Expected Findings**:
```
pass_length <-> end_location_x: r=0.95  ← Redundant (derived)
pass_length <-> end_location_y: r=0.88  ← Redundant (derived)
pass_angle <-> end_location_y: r=0.82   ← Redundant (derived)
```

**Save**:
- `results/exploration/feature_correlation_matrix.csv` - Full matrix
- `results/exploration/high_correlations.json` - Pairs with |r| > 0.8

**Script**: `scripts/analyze_feature_correlations.py`

---

### Phase 4: Feature-Receiver Correlation

**Goal**: Which features predict receiver well?

**Analysis**:
```python
import pandas as pd
from scipy.stats import pointbiserialr

# Load dataset
df = pd.read_csv('data/processed/corner_features.csv')

# For each feature, compute correlation with receiver
receiver_corr = []
for feature in feature_columns:
    # Point-biserial correlation for categorical receiver
    r, p_value = pointbiserialr(df['receiver_id'], df[feature])
    receiver_corr.append({
        'feature': feature,
        'correlation': r,
        'abs_correlation': abs(r),
        'p_value': p_value,
        'significant': p_value < 0.05
    })

# Sort by absolute correlation
receiver_corr = sorted(receiver_corr, key=lambda x: x['abs_correlation'], reverse=True)

print("Top 10 features correlated with receiver:")
for item in receiver_corr[:10]:
    print(f"  {item['feature']}: r={item['correlation']:.3f} (p={item['p_value']:.4f})")
```

**Expected Output**:
```
Top 10 features correlated with receiver:
  end_location_x: r=0.85 (p=0.0000)  ← LEAKAGE!
  end_location_y: r=0.78 (p=0.0000)  ← LEAKAGE!
  player_id: r=0.32 (p=0.0000)
  team_id: r=0.18 (p=0.0002)
  minute: r=0.05 (p=0.1234)
  ...
```

**Interpretation**:
- If `end_location` has r > 0.7 → **Strong leakage** (knows where ball lands)
- Features with r < 0.1 → Weak predictors (candidates for removal)

**Save**:
- `results/exploration/feature_receiver_correlation.csv`

**Script**: `scripts/analyze_receiver_correlation.py`

---

### Phase 5: Identify Leakage Suspects

**Goal**: Flag features that likely leak post-kick information

**Method**: Combine evidence from Phase 2-4

**Criteria for Leakage**:
1. **High importance** (>10% in Tier 1) AND
2. **High correlation with receiver** (|r| > 0.5) AND
3. **Conceptually post-kick** (end_location, shot_assist, etc.)

**Analysis**:
```python
import json

leakage_suspects = []

for feature in all_features:
    # Check importance
    high_importance = importance[feature] > 10

    # Check correlation
    high_correlation = abs(receiver_corr[feature]) > 0.5

    # Conceptual check (manual)
    post_kick_features = ['end_location_x', 'end_location_y',
                          'pass_length', 'pass_angle',
                          'shot_assist', 'switch']

    is_post_kick = feature in post_kick_features

    if high_importance and high_correlation:
        leakage_suspects.append({
            'feature': feature,
            'importance': importance[feature],
            'receiver_correlation': receiver_corr[feature],
            'reason': 'High importance + correlation'
        })
    elif is_post_kick and high_importance:
        leakage_suspects.append({
            'feature': feature,
            'importance': importance[feature],
            'reason': 'Post-kick information + high importance'
        })

print(f"Found {len(leakage_suspects)} suspected leakage features:")
for suspect in leakage_suspects:
    print(f"  {suspect['feature']}: {suspect['reason']}")
```

**Expected Output**:
```
Found 4 suspected leakage features:
  end_location_x: High importance (35.2%) + correlation (0.85)
  end_location_y: High importance (28.1%) + correlation (0.78)
  pass_length: Post-kick information + high importance (8.5%)
  pass_angle: Post-kick information + high importance (6.2%)
```

**Save**:
- `results/exploration/leakage_suspects.json`

---

### Phase 6: Design Ablation Configs

**Goal**: Based on exploration, design specific configs to test

**Input**: All exploration results from Phases 2-5

**Decision Tree**:

```
1. Baseline (Config 0):
   - All 23 features
   - Purpose: Reference performance

2. For each leakage suspect (Configs 1-N):
   - Remove ONE suspected feature at a time
   - Measure performance drop
   - If drop > 10% → Confirmed leakage

   Example:
   - Config 1: Remove end_location_x only
   - Config 2: Remove end_location_y only
   - Config 3: Remove pass_length only
   - Config 4: Remove pass_angle only

3. For each redundant pair (Configs N+1 to M):
   - Remove one feature from each correlated pair
   - See if performance stays same

   Example:
   - Config 5: Remove pass_length (keep end_location)
   - Config 6: Remove pass_angle (keep end_location)

4. Clean baseline (Config M+1):
   - Remove ALL confirmed leakage features
   - This becomes the "fair comparison" baseline

   Example:
   - Config 7: Remove end_location_x, end_location_y

5. Add engineered features (Configs M+2 onwards):
   - Starting from clean baseline
   - Add ONE new feature at a time

   Example:
   - Config 8: Clean + num_attackers_in_box
   - Config 9: Clean + num_defenders_in_box
```

**Output**:
- `configs/ablation_plan.json` - List of configs to test

---

## Implementation Scripts

### Script 1: `scripts/explore_dataset.py`

**Purpose**: Run all 5 exploration phases in one script

```python
#!/usr/bin/env python3
"""
Explore dataset to guide ablation design.

Phases:
  1. Train baseline (already done)
  2. Analyze feature importance
  3. Compute feature correlations
  4. Feature-receiver correlation
  5. Identify leakage suspects
  6. Generate ablation configs

Usage:
  python scripts/explore_dataset.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_importance():
    """Phase 2: Analyze feature importance tiers."""
    pass

def compute_correlations():
    """Phase 3: Feature correlation matrix."""
    pass

def receiver_correlation():
    """Phase 4: Which features predict receiver?"""
    pass

def identify_leakage():
    """Phase 5: Flag leakage suspects."""
    pass

def generate_configs():
    """Phase 6: Create ablation config plan."""
    pass

if __name__ == "__main__":
    print("="*60)
    print("Dataset Exploration for Ablation Design")
    print("="*60)

    # Create output directory
    Path("results/exploration").mkdir(parents=True, exist_ok=True)

    # Run phases
    print("\nPhase 2: Analyzing feature importance...")
    analyze_importance()

    print("\nPhase 3: Computing feature correlations...")
    compute_correlations()

    print("\nPhase 4: Feature-receiver correlations...")
    receiver_correlation()

    print("\nPhase 5: Identifying leakage suspects...")
    identify_leakage()

    print("\nPhase 6: Generating ablation configs...")
    generate_configs()

    print("\n" + "="*60)
    print("Exploration complete! Check results/exploration/")
    print("="*60)
```

**Run**:
```bash
python scripts/explore_dataset.py
```

**Outputs**:
```
results/exploration/
├── feature_tiers.json              # Importance tiers
├── feature_correlation_matrix.csv  # Full correlation matrix
├── high_correlations.json          # Highly correlated pairs
├── feature_receiver_correlation.csv # Receiver prediction correlations
├── leakage_suspects.json           # Flagged leakage features
└── ablation_plan.json              # Recommended configs to test
```

---

## Expected Timeline

**Phase 1** (Baseline): ✅ Already done (Nov 13)

**Phase 2-5** (Exploration):
- Script implementation: 2 hours
- Execution time: 10 minutes
- Analysis of results: 30 minutes
- **Total**: ~3 hours

**Phase 6** (Config design):
- Review exploration results: 30 minutes
- Design ablation configs: 1 hour
- **Total**: ~1.5 hours

**Grand Total**: ~4.5 hours from start to ablation configs ready

---

## Deliverables

After completing this plan, you will have:

✅ **Clear understanding** of which features leak information
✅ **Evidence-based** list of configs to test
✅ **Prioritized** ablation experiments (test leakage first)
✅ **Systematic** approach (not guessing which features to test)

---

## Next Steps After Exploration

1. Review `results/exploration/leakage_suspects.json`
2. Check `results/exploration/ablation_plan.json`
3. Implement configs in feature registry
4. Run ablation experiments one-by-one
5. Analyze results and update plan

---

## Integration with Flexible Ablation Framework

This exploration feeds into the framework:

```
DATA_EXPLORATION_PLAN.md (this file)
        ↓
  Identifies leakage suspects
        ↓
FLEXIBLE_ABLATION_FRAMEWORK.md
        ↓
  Systematic one-at-a-time testing
        ↓
  Results & conclusions
```

The exploration **guides** which configs to test.
The framework **defines** how to test them properly.

---

## Status

- [x] Plan created
- [ ] Script implemented (`scripts/explore_dataset.py`)
- [ ] Baseline already trained (Nov 13)
- [ ] Exploration run
- [ ] Leakage suspects identified
- [ ] Ablation configs designed

**Ready to implement**: Yes, all inputs available
