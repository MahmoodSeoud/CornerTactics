# Dataset Expansion Summary

## Overview
This document summarizes the temporal augmentation approach applied to expand the Corner Kick GNN dataset from 1,118 corners to 7,369 graphs (6.6× increase).

## Problem
**Original Dataset**:
- 1,118 StatsBomb corners (single freeze-frames)
- Only 14 goals (1.25% - extreme class imbalance)
- Training Results: Val AUC 0.765, **Test AUC 0.271** (severe overfitting)

## Solution: US Soccer Federation Temporal Augmentation Approach

Inspired by Bekkers & Sahasrabudhe (2024) US Soccer Federation counterattack GNN which used ~20k temporal tracking frames.

### 1. SkillCorner Temporal Extraction (Phase 2.3)
**Script**: `scripts/extract_skillcorner_temporal.py`

- Accessed SkillCorner 10fps tracking data via GitHub media URLs
  - URL format: `https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl`
  - Bypasses Git LFS budget limitations
- Extracted 5 temporal frames per corner:
  - t = -2.0s, -1.0s, 0.0s, +1.0s, +2.0s
- **Output**: 1,555 graphs from 317 corners
- **Dangerous situations**: ~205 shots (13.2%)

### 2. StatsBomb Temporal Augmentation (Phase 2.4)
**Script**: `scripts/augment_statsbomb_temporal.py`

Applied data augmentation to StatsBomb freeze-frames:

**Temporal Variations**:
- 5 temporal offsets: t = -2s, -1s, 0s, +1s, +2s
- Position perturbations with Gaussian noise
  - Noise magnitude proportional to temporal distance
  - Simulates movement uncertainty
- Velocity features augmented for non-zero offsets

**Mirror Augmentation**:
- Flipped y-coordinates for left/right symmetry
- Applied to t=0 frames only
- Doubles geometric diversity

**Output**: 5,814 augmented graphs
- 5,590 temporal variations (5× original)
- 224 mirror augmentations
- **Dangerous situations**: 1,056 (18.2%)

## Target Label Change

**Old Target**: Goal only (1.3% positive class)
**New Target**: **Dangerous Situation** = Shot OR Goal (17.1% positive class)

Rationale:
- Predicting goals from freeze-frames is extremely difficult (low signal)
- Shots are more frequent and still tactically valuable
- 17% positive class provides reasonable balance for training

## Final Dataset

**Total**: 7,369 graphs (6.6× increase)
- StatsBomb Augmented: 5,814 graphs (78.9%)
- SkillCorner Temporal: 1,555 graphs (21.1%)

**Class Distribution**:
- Dangerous situations: ~1,261 (17.1%)
- Non-dangerous: ~6,108 (82.9%)
- **Much better than 1.3% goal-only rate!**

## Comparison to US Soccer Federation

| Metric | US Soccer Fed | CornerTactics |
|--------|--------------|---------------|
| Event Type | Counterattacks | Corner Kicks |
| Total Graphs | 20,863 | 7,369 |
| Approach | Real 10fps tracking | Temporal augmentation + real tracking |
| Temporal Frames | Multiple per sequence | 5 per corner |
| Data Sources | MLS + International | StatsBomb + SkillCorner |

## Implementation Details

### SkillCorner Data Access
```python
# Load tracking data directly from GitHub
tracking_url = f'https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl'
raw_data = pd.read_json(tracking_url, lines=True)
```

### Temporal Augmentation
```python
# Position perturbation
noise_magnitude = abs(temporal_offset) * 0.5
augmented[:, 0] += np.random.normal(0, noise_magnitude, len(augmented))  # x
augmented[:, 1] += np.random.normal(0, noise_magnitude, len(augmented))  # y
```

### Mirror Augmentation
```python
# Flip y-coordinate
mirrored_features[:, 1] = 80 - mirrored_features[:, 1]

# Flip angles
mirrored_features[:, 7] = -mirrored_features[:, 7]  # velocity_angle
mirrored_features[:, 8] = -mirrored_features[:, 8]  # angle_to_goal
```

## SLURM Scripts

- `phase2_3_skillcorner_temporal.sh`: Extract SkillCorner temporal features
- `phase2_4_statsbomb_augment.sh`: Augment StatsBomb with temporal variations
- `phase3_train_gnn.sh`: Train GNN on expanded dataset

## Expected Improvements

**Hypothesis**:
- Larger dataset → Less overfitting
- Better class balance (17% vs 1.3%) → Improved learning
- Temporal variations → Better generalization

**Baseline to Beat**:
- Val AUC: 0.765
- **Test AUC: 0.271** ← Need to significantly improve this!

**Target**:
- Test AUC > 0.60 (reasonable performance)
- Reduced val-test AUC gap (less overfitting)

## Files Created

**Scripts**:
- `scripts/extract_skillcorner_temporal.py`
- `scripts/augment_statsbomb_temporal.py`
- `scripts/slurm/phase2_3_skillcorner_temporal.sh`
- `scripts/slurm/phase2_4_statsbomb_augment.sh`

**Data**:
- `data/features/temporal/skillcorner_temporal_features.parquet`
- `data/graphs/adjacency_team/statsbomb_temporal_augmented.pkl`

**Documentation**:
- Updated `CLAUDE.md` with temporal augmentation details
- Updated project status and file tree

## References

Bekkers, J., & Sahasrabudhe, N. (2024). "A Graph Neural Network Deep-Dive into Successful Counterattacks". *Proceedings of the 11th Workshop on Machine Learning and Data Mining for Sports Analytics*.

GitHub: https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn
