# Comprehensive Project Audit

**Date:** 2026-02-21
**Scope:** Full audit of datasets, features, experiments, backbone, training, bugs, and unused assets.

---

## 1. Dataset Inventory

### Corner counts by source

| Source | Matches | Corners | Shots | Shot Rate | Goals | Goal Rate | Receiver Labels |
|--------|---------|---------|-------|-----------|-------|-----------|-----------------|
| SkillCorner (A-League) | 10 | 86 | 29 | 33.7% | 3 | 3.5% | 66 (76.7%) |
| DFL (Bundesliga) | 7 | 57 | 16 | 28.1% | 0 | 0.0% | 0 (0.0%) |
| **Combined** | **17** | **143** | **45** | **31.5%** | **3** | **2.1%** | **66 (46.2%)** |

No quality filtering is applied -- `min_detection_rate=0.0` in `config.py`. All corners with valid 22-player snapshots are kept.

### Per-match breakdown

| Match ID | Source | Corners | Shots | Shot Rate | Receiver Labels |
|----------|--------|---------|-------|-----------|-----------------|
| 1886347 | SkillCorner | 8 | 2 | 25% | 6 |
| 1899585 | SkillCorner | 7 | 2 | 29% | 5 |
| 1925299 | SkillCorner | 4 | 3 | 75% | 3 |
| 1953632 | SkillCorner | 8 | 4 | 50% | 6 |
| 1996435 | SkillCorner | 14 | 4 | 29% | 12 |
| 2006229 | SkillCorner | 4 | 1 | 25% | 2 |
| 2011166 | SkillCorner | 12 | 5 | 42% | 10 |
| 2013725 | SkillCorner | 13 | 3 | 23% | 10 |
| 2015213 | SkillCorner | 4 | 3 | 75% | 3 |
| 2017461 | SkillCorner | 12 | 2 | 17% | 9 |
| J03WMX | DFL | 10 | 3 | 30% | 0 |
| J03WN1 | DFL | 7 | 2 | 29% | 0 |
| J03WOH | DFL | 6 | 2 | 33% | 0 |
| J03WOY | DFL | 2 | 0 | 0% | 0 |
| J03WPY | DFL | 11 | 1 | 9% | 0 |
| J03WQQ | DFL | 15 | 4 | 27% | 0 |
| J03WR9 | DFL | 6 | 4 | 67% | 0 |

Receiver label extraction uses a 3-tier method in `extract_corners.py:257-383`: (1) `player_targeted_name`, (2) `passing_option` with targeted+received flags, (3) `player_possession` within 3s window. DFL XML has no event-level receiver data.

### Other datasets on disk (unused in final pipeline)

| Dataset | Location | Size | Status |
|---------|----------|------|--------|
| **StatsBomb events** | `data/statsbomb/` | ~2 MB | Abandoned (AUC=0.43). 323 freeze-frame JSONs, train/val/test splits |
| **Metrica tracking** | `data/metrica/` | Small | Downloaded, never integrated |
| **FAANTRA video clips** | `data/corner_clips/` | 9.9 GB (~4,200 clips) | Used for video model (mAP=50%, random). Dead end |
| **USSF counterattacks** | `data/misc/ussf_data_sample.pkl` | 4.5 GB | Used only for backbone pretraining (20,863 graphs) |
| **SoccerNet-GSR** | `sn-gamestate/` + `data/MySoccerNetGS/` | ~854 MB | Downloaded, submodule present. Never used in pipeline |
| **SoccerSegCal** | `soccersegcal-pipeline/` | Small | Submodule with untracked `sskit` changes. Not used |
| **Bundesliga raw** | `data/bundesliga/` | Small | Exploration scaffolding, not integrated |
| **DFL open-play graphs** | `transfer_learning/data/dfl_openplay_graphs.pkl` | ~30 MB | 11,967 graphs for transfer validation (0.86 AUC) |
| **Processed legacy** | `data/processed/` | 33 MB | CSVs/JSONs from earlier experiments. Stale |

---

## 2. Feature Completeness

### Node features (13 base + 1 dynamic)

| Index | Feature | Derivation | Range |
|-------|---------|-----------|-------|
| 0 | `x_norm` | `x / 52.5` (`build_graphs.py:82`) | [-1, 1] |
| 1 | `y_norm` | `y / 34.0` (`build_graphs.py:83`) | [-1, 1] |
| 2 | `vx` | Raw backward difference, not smoothed. SK: `(x_t - x_{t-1}) / 0.1` at 10Hz. DFL: `(x_nearest - x_prev) / dt` at 25Hz (`build_graphs.py:85`) | m/s |
| 3 | `vy` | Same backward difference (`build_graphs.py:86`) | m/s |
| 4 | `speed` | `sqrt(vx^2 + vy^2)` (`build_graphs.py:87`) | m/s |
| 5 | `is_attacking` | Binary: 1.0 if attacking team (`build_graphs.py:89`) | {0, 1} |
| 6 | `is_corner_taker` | Binary: 1.0 for corner taker (`build_graphs.py:90`) | {0, 1} |
| 7 | `is_goalkeeper` | Binary: 1.0 if role == "GK" (`build_graphs.py:91`) | {0, 1} |
| 8 | `is_detected` | Binary: tracked vs extrapolated (`build_graphs.py:92`) | {0, 1} |
| 9-12 | `position_group` | One-hot [GK, DEF, MID, FWD] via `ROLE_TO_GROUP` mapping (`build_graphs.py:60-66, 97`) | {0, 1} |
| **13** | **receiver_indicator** | Appended dynamically in `two_stage.py:45-64`. 0.0 for Stage 1; 1.0 at receiver node for Stage 2 | {0, 1} |

### Computed but unused features

None in main pipeline. The graph construction is lean -- no intermediate features computed and discarded.

`transfer_learning/phase2_engineer_dfl_features.py:172-185` computes 12 USSF-format derived features (`angle_goal`, `dist_goal`, `angle_ball`, `dist_ball`, `vel_angle`, etc.) that are only used in transfer experiments, not the main two-stage pipeline.

### Physical metadata

**Not available.** Neither SkillCorner nor DFL provides height, weight, or anthropometric data. SkillCorner metadata (`extract_corners.py:106-126`) has: team_id, role, number, name. DFL metadata (`extract_dfl_corners.py:281-314`) has: team_id, position, player_id, name. No biometric fields in either.

### Edge features (4 per edge)

| Index | Feature | Computation (`build_graphs.py:133-136`) |
|-------|---------|----------------------------------------|
| 0 | `dx` | `x_j - x_i` (normalized coords) |
| 1 | `dy` | `y_j - y_i` (normalized coords) |
| 2 | `distance` | `sqrt(dx^2 + dy^2)` |
| 3 | `same_team` | 1.0 if same team, else 0.0 |

No alternative edge features were tested (e.g., velocity differences, angle between velocity vectors). Only topology varied: KNN k=6 (default, ~132 edges) vs fully connected (~462 edges).

---

## 3. Experiments Run vs Not Run

### All result files (57 total)

#### Main LOMO evaluations (7 configs)

| File | Dataset | Config |
|------|---------|--------|
| `lomo_pretrained_20260215_140747.pkl` | SkillCorner | pretrained, frozen |
| `lomo_pretrained_20260219_225128.pkl` | SkillCorner | pretrained, frozen (rerun w/ loss curves) |
| `lomo_scratch_20260215_140829.pkl` | SkillCorner | scratch |
| `combined_lomo_pretrained_20260215_184958.pkl` | Combined | pretrained, frozen |
| `combined_lomo_pretrained_20260219_225339.pkl` | Combined | pretrained, frozen (rerun w/ loss curves) |
| `lomo_pretrained_linear_20260220_194133.pkl` | SkillCorner | pretrained, linear heads |
| `combined_lomo_pretrained_linear_20260220_194030.pkl` | Combined | pretrained, linear heads |

#### Ablations (5 configs, SkillCorner-only)

`ablation_{position_only, plus_velocity, plus_detection, full_features, full_fc_edges}_*.pkl` + batch file `ablation_all_*.pkl`

#### Baselines (10 files across SK-only + combined)

Random, heuristic (SK-only); XGBoost (SK + combined); MLP (SK + combined); MLP linear (SK + combined)

#### Permutation tests (7 files)

| File | Target | Dataset | p-value |
|------|--------|---------|---------|
| `perm_shot_*.pkl` | GNN shot | SkillCorner | 0.020 |
| `perm_receiver_*.pkl` | GNN receiver | SkillCorner | 0.406 |
| `combined_perm_shot_*.pkl` | GNN shot | Combined | **0.010** |
| `combined_perm_receiver_*.pkl` | GNN receiver | Combined | **0.050** |
| `perm_mlp_*.pkl` | MLP shot | SkillCorner | **0.010** |
| `perm_xgboost_*.pkl` | XGBoost shot | SkillCorner | **0.010** |

#### Transfer learning (6 files)

Phase 3 (6 conditions x 5 seeds), Phase 4 (velocity ablation), Phase 5 (DFL vs USSF comparison), Phase 5b (0.86 AUC validation)

#### Multi-source experiments (5 files)

exp1-5 on 3,078 USSF graphs

### Experiments defined but never run

| What | Status | Detail |
|------|--------|--------|
| **Multi-seed evaluation** | NOT IMPLEMENTED | `SEEDS = [42, 123, 456, 789, 1234]` defined in `config.py:68` but never looped over. All main results use seed=42 only. Transfer learning Phase 3 does use 5 seeds. |
| **Baseline permutation on combined** | NOT RUN | MLP/XGBoost permutation tests only on SkillCorner (86 corners). Code supports `--combined` flag but it was never invoked. |
| **Hyperparameter sweeps** | NONE | All values hardcoded: dropout=0.3, patience=20, LR=1e-3, weight_decay=1e-3, hidden=[64,32], batch_size=8, pos_weight=2.0. Zero variations tested. |
| **Graph construction variants** | ONLY 2 OF N | Only KNN k=6 and fully connected. Never tried k=3,4,5,8,10 or radius-based graphs. Multi-source exp5 tested k=5 vs dense on USSF data but not on corners. |
| **Different early stopping patience** | NOT TESTED | Only patience=20 ever used |

---

## 4. Backbone Usage

### USSF backbone expectations

- **Weight file:** `transfer_learning/weights/ussf_backbone_dense.pt`
- **Trained on:** 20,863 counterattack graphs, AUC=0.693
- **Expected input:** 12 node features, 6 edge features (USSF schema)
- **Architecture:** CGConv(12, dim=6) -> Linear(12, 128) -> 2x CGConv(128, dim=6)

### Linear adapter (projection layers)

The corner features (14-dim nodes, 4-dim edges) don't match USSF expectations, so trainable projections bridge the gap:

- `node_proj`: Linear(14->12) = 180 weights + 12 bias = **192 params**
- `edge_proj`: Linear(4->6) = 24 weights + 6 bias = **30 params**
- Always trainable even when backbone is frozen

### Parameter counts

| Component | Pretrained+Frozen | Scratch |
|-----------|-------------------|---------|
| Backbone (frozen) | 137,064 | -- |
| Projections (trainable) | 222 | -- |
| Backbone (all trainable) | -- | 35,932 |
| ReceiverHead MLP | 8,321 | 4,225 |
| ShotHead MLP | 4,193 | 2,145 |
| **Total** | **149,788** | **42,302** |
| **Trainable** | **12,724 (8.5%)** | **42,302 (100%)** |

### Freeze vs unfreeze vs scratch

| Config | Dataset | Shot AUC (oracle) |
|--------|---------|-------------------|
| Pretrained + frozen | SkillCorner | 0.751 +/- 0.213 |
| Scratch | SkillCorner | 0.706 +/- 0.273 |
| Pretrained + frozen | Combined | 0.730 +/- 0.202 |

Unfreezing was never tested on the main corner pipeline (only in transfer learning Phase 3, Condition C: 0.56 +/- 0.23 on 57 DFL corners). Scratch was never run on combined.

---

## 5. Training Details -- Per-Fold

### Combined dataset, pretrained+frozen, seed=42 (Feb 19 rerun)

| Fold | Test Match | Train | Val | Test | Shots/NoShots | Shot Rate | Recv Epochs | Recv Best Val | Shot Epochs | Shot Best Val | Shot AUC |
|------|-----------|-------|-----|------|---------------|-----------|-------------|---------------|-------------|---------------|----------|
| 0 | 1886347 | 128 | 7 | 8 | 2/6 | 25% | 26 | 2.1987 | 30 | 0.8061 | 1.000 |
| 1 | 1899585 | 132 | 4 | 7 | 2/5 | 29% | 81 | 0.3940 | 52 | 0.4856 | 0.600 |
| 2 | 1925299 | 131 | 8 | 4 | 3/1 | 75% | 25 | 2.5923 | 56 | 0.8927 | 1.000 |
| 3 | 1953632 | 121 | 14 | 8 | 4/4 | 50% | 36 | 0.9047 | 22 | 1.1374 | 0.875 |
| 4 | 1996435 | 125 | 4 | 14 | 4/10 | 29% | 22 | 2.5238 | 22 | 0.8255 | 0.825 |
| 5 | 2006229 | 127 | 12 | 4 | 1/3 | 25% | 23 | 1.0374 | 45 | 0.8898 | 1.000 |
| 6 | 2011166 | 118 | 13 | 12 | 5/7 | 42% | 72 | 3.4939 | 45 | 0.6155 | 0.657 |
| 7 | 2013725 | 126 | 4 | 13 | 3/10 | 23% | 49 | 0.9676 | 72 | 0.4666 | 0.533 |
| 8 | 2015213 | 127 | 12 | 4 | 3/1 | 75% | 21 | 1.1656 | 23 | 0.6461 | 1.000 |
| **9** | **2017461** | 121 | 10 | 12 | 2/10 | 17% | **21** | **0.0000** | 100 | 0.6953 | 0.700 |
| **10** | **J03WMX** | 126 | 7 | 10 | 3/7 | 30% | **21** | **0.0000** | 21 | 0.8776 | 0.571 |
| **11** | **J03WN1** | 130 | 6 | 7 | 2/5 | 29% | **21** | **0.0000** | 26 | 0.9352 | 0.700 |
| **12** | **J03WOH** | 135 | 2 | 6 | 2/4 | 33% | **21** | **0.0000** | 100 | 0.0482 | 0.625 |
| **13** | **J03WOY** | 130 | 11 | 2 | 0/2 | 0% | **21** | **0.0000** | 21 | 0.7076 | 0.500 |
| **14** | **J03WPY** | 117 | 15 | 11 | 1/10 | 9% | **21** | **0.0000** | 21 | 0.9053 | 0.900 |
| **15** | **J03WQQ** | 122 | 6 | 15 | 4/11 | 27% | **21** | **0.0000** | 24 | 1.1998 | 0.364 |
| 16 | J03WR9 | 129 | 8 | 6 | 4/2 | 67% | 68 | 1.3727 | 58 | 0.4176 | 0.500 |

**Bold rows** = folds where receiver early stopping is broken (see Section 6).

### Receiver evaluation per fold (SkillCorner test folds only)

| Fold | Test Match | Labeled in Test | Top-1 | Top-3 |
|------|-----------|-----------------|-------|-------|
| 0 | 1886347 | 5 | 0.200 | 0.400 |
| 1 | 1899585 | 5 | 0.400 | 0.800 |
| 2 | 1925299 | 2 | 0.000 | 0.000 |
| 3 | 1953632 | 4 | 0.250 | 0.500 |
| 4 | 1996435 | 3 | 0.667 | 1.000 |
| 5 | 2006229 | 1 | 0.000 | 0.000 |
| 6 | 2011166 | 4 | 0.750 | 0.750 |
| 7 | 2013725 | 8 | 0.500 | 0.625 |
| 8 | 2015213 | 2 | 0.500 | 0.500 |
| 9 | 2017461 | 1 | 0.000 | 0.000 |
| 10-16 | DFL | 0 each | N/A | N/A |

### Degenerate folds

- **Fold 13 (J03WOY):** Only 2 test corners, 0 shots -> AUC defaults to 0.500
- **Folds 2, 8 (1925299, 2015213):** 4 test corners with 75% shot rate -- only 1 negative. AUC=1.000 but meaningless with n=4
- **Fold 12 (J03WOH):** Val set of only 2 corners. Shot head ran 100 epochs (hit max, never early stopped properly)

---

## 6. Known Bugs and Issues

### BUG: Receiver early stopping broken for 8 of 17 combined folds

**This is the most significant finding in this audit.**

Folds 9-15 have DFL validation sets with **zero receiver labels**. Since `receiver_loss()` returns 0.0 when `has_receiver_label=False`, the validation loss is permanently 0.0. Early stopping interprets this as "perfect loss, can't improve" and triggers after patience=20 epochs (epoch 21).

| Fold | Val Match | Val Has Recv Labels? | Recv Epochs | Effect |
|------|----------|---------------------|-------------|--------|
| 0-8 | SkillCorner | Yes | 21-81 | Normal training |
| **9** | **DFL-J03WMX** | **No** | **21** | **Early stop on fake 0.0 loss** |
| **10** | **DFL-J03WN1** | **No** | **21** | **Early stop on fake 0.0 loss** |
| **11** | **DFL-J03WOH** | **No** | **21** | **Early stop on fake 0.0 loss** |
| **12** | **DFL-J03WOY** | **No** | **21** | **Early stop on fake 0.0 loss** |
| **13** | **DFL-J03WPY** | **No** | **21** | **Early stop on fake 0.0 loss** |
| **14** | **DFL-J03WQQ** | **No** | **21** | **Early stop on fake 0.0 loss** |
| **15** | **DFL-J03WR9** | **No** | **21** | **Early stop on fake 0.0 loss** |
| 16 | SK-1886347 | Yes | 68 | Normal training |

In these 7 folds, the receiver head trains for exactly 21 epochs with no validation signal. The backbone projections barely update. This affects Stage 2 because the "predicted receiver" mode depends on Stage 1 quality. The "oracle" and "unconditional" modes are unaffected.

**Impact on reported results:** The combined receiver Top-3 of 0.458 is computed over only the 10 SkillCorner test folds (which have labels). Stage 1 in folds 9-15 is undertrained. However, shot AUC (oracle mode) is not directly affected because it uses ground-truth receiver labels, not predictions.

### Silent exception handling (data quality risk)

6x `except ValueError: pass` in `extract_corners.py:311-340` silently drop malformed CSV values during receiver label extraction. No logging of how many records are affected. Same pattern in `extract_dfl_corners.py:652-653` and `build_graphs.py:315-316`.

### Non-determinism

Seeds are set per fold (`torch.manual_seed(seed + fold_idx)`, `np.random.seed(seed + fold_idx)` in `evaluate.py:140-141`) but **CUDA determinism flags are never set**:

```python
# Missing:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

GNN scatter/gather operations on GPU are inherently non-deterministic without these. This explains the 0.730 vs 0.715 AUC gap between separate runs (CLAUDE.md Known Gap #5).

### Receiver mask correctly applied for DFL

DFL corners always have `has_receiver_label=False` (`extract_dfl_corners.py:613`). The receiver mask (which players are valid candidates) is still set correctly -- it's the label that's missing, not the mask. `receiver_loss()` in `receiver_head.py:113-114` correctly returns 0.0 for unlabeled graphs. No bug here, just the early stopping interaction described above.

---

## 7. What's Sitting Unused

### Unused directories in repo

| Directory | Contents | Status |
|-----------|----------|--------|
| `experiments/` | `gnn_baseline/`, `offside_analysis/`, `class_imbalance/`, `interpretability/`, `statistical_tests/` | Pre-pipeline explorations. Not imported by current code. Safe to archive |
| `tracking_extraction/` | Adapters for SK, DFL, SoccerNet-GSR + graph conversion | Superseded by `corner_prediction/data/extract_*.py`. Dead code |
| `corner-diffusion/` | Separate experimental project | Not integrated |
| `src/` | Legacy source | Not imported by pipeline |
| `sn-gamestate/` | SoccerNet GSR fork (~854 MB) | Downloaded, never used for corners |
| `soccersegcal-pipeline/` | Submodule with untracked `sskit` changes | Not used |

### TODO/FIXME comments

None found in any `corner_prediction/` Python files. The codebase is clean of unfinished markers.

### Partially written code

No stubbed-out functions or `pass` bodies found in the main pipeline. The `SEEDS` list in `config.py:68` is the only artifact of planned-but-unimplemented functionality.

### Unused weight files

| File | Purpose | Used? |
|------|---------|-------|
| `ussf_backbone_dense.pt` | Main pretrained backbone | **Yes** (default) |
| `ussf_backbone_normal.pt` | Normal adjacency variant | No (only in transfer learning) |
| `ussf_full_model_dense.pt` | Full USSF model (not just backbone) | No |
| `ussf_full_model_normal.pt` | Full model, normal adjacency | No |
| `dfl_backbone_dense.pt` | DFL open-play pretrained backbone | Only in Phase 5 comparison |

---

## Summary of Gaps

### High priority (affects validity)

1. **Receiver early stopping broken for 8/17 combined folds** -- DFL val has no labels -> fake 0.0 loss -> immediate stop at epoch 21
2. **No multi-seed evaluation** -- all results from seed=42 only
3. **No CUDA determinism flags** -- results vary between runs
4. **Baseline permutation tests missing on combined dataset**

### Medium priority (incomplete exploration)

5. Zero hyperparameter tuning (dropout, LR, patience, pos_weight all at defaults)
6. Only 2 graph construction variants tested (KNN k=6, fully connected)
7. Silent exception handling drops unknown number of records during data extraction

### Low priority (cleanup)

8. ~5 unused directories and legacy code that could be archived
9. Stale submodules (`sn-gamestate`, `soccersegcal-pipeline`)
10. Unused `SEEDS` list suggests unfinished plan
