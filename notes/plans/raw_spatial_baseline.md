# Plan: Raw Spatial Feature Baseline

**Date:** 2025-11-27
**Status:** Planned
**Priority:** High

---

## Problem Statement

Current experiments use only **aggregate features** (22 total) derived from freeze frame data:
- Counts (total_attacking, defending_in_box, etc.)
- Densities (attacking_density, defending_density)
- Aggregates (centroids, distances to goal, depth)

**We never tested raw player coordinates as features.**

This is a methodological flaw. A proper baseline should:
1. Use raw coordinates first (simplest approach with full information)
2. Show that aggregates perform similarly or better
3. Justify the feature engineering decision with evidence

---

## Critique (Valid)

> "Your baseline experiments don't baseline anything useful. A baseline should be the simplest reasonable approach that future work can beat. But your approach isn't simple—it's impoverished. You threw away the spatial structure (individual positions) before even trying."

---

## Available Raw Data

Each freeze frame contains **individual player positions**:

```python
{
    "teammate": true,      # Attacker (true) or defender (false)
    "actor": false,        # Is this the corner taker?
    "keeper": false,       # Is this a goalkeeper?
    "location": [102.7, 40.3]  # (x, y) coordinates
}
```

**Statistics across 1,933 corners:**
- Total players: min=5, max=22, mean=18.6
- Attackers: min=1, max=11, mean=8.1
- Defenders: min=4, max=11, mean=10.5

---

## Proposed Experiments

### Experiment 1: Raw Coordinate Baseline

**Features:**
- Pad to max players (11 attackers + 11 defenders = 22 players)
- Sort attackers by distance to goal (ascending)
- Sort defenders by distance to goal (ascending)
- Flatten: 22 players × 2 coords = **44 features**
- Missing players → fill with (-1, -1) or (0, 0)

**Models:** XGBoost, Random Forest, MLP

**Expected outcome:** Establish whether raw coordinates contain more signal than aggregates.

---

### Experiment 2: Pairwise Distance Features

**Features:**
- Distance from each attacker to nearest defender
- Distance from each attacker to goalkeeper
- Min/max/mean attacker-defender distances
- Number of "unmarked" attackers (no defender within 2m)

**Rationale:** Captures marking structure that aggregates miss.

---

### Experiment 3: Spatial Structure Features

**Features:**
- Convex hull area (attackers)
- Convex hull area (defenders)
- Overlap between hulls
- Voronoi cell areas
- Clustering coefficient

**Rationale:** Captures team shape beyond simple density.

---

## Implementation Steps

1. **Create feature extraction script** (`scripts/16_extract_raw_spatial_features.py`)
   - Load freeze frames
   - Sort and pad player positions
   - Compute pairwise distances
   - Save to CSV

2. **Train models** (`scripts/17_train_raw_spatial_baseline.py`)
   - Use same train/val/test splits
   - Train XGBoost, RF, MLP on raw features
   - Compare to aggregate baseline

3. **Document results**
   - Add comparison table to paper
   - Justify feature choice based on evidence

---

## Success Criteria

| Scenario | Interpretation | Action |
|----------|----------------|--------|
| Raw coords >> Aggregates | Aggregates lose information | Use raw coords or richer features |
| Raw coords ≈ Aggregates | Aggregates sufficient | Current approach justified |
| Raw coords << Aggregates | Raw coords add noise | Aggregation helps (unlikely) |

---

## Timeline

- [ ] Step 1: Feature extraction script
- [ ] Step 2: Training script
- [ ] Step 3: Run experiments
- [ ] Step 4: Update paper with results

---

## Files to Create

```
scripts/
├── 16_extract_raw_spatial_features.py
├── 17_train_raw_spatial_baseline.py
data/processed/
├── corners_raw_spatial_features.csv
results/
├── raw_spatial_baseline/
│   ├── training_results.json
│   └── comparison_table.md
```

---

## Notes

- Variable player count (5-22) requires padding strategy
- Sorting by distance to goal provides canonical ordering
- May need to normalize coordinates (already 0-120, 0-80)
- Consider adding goalkeeper position as separate features
