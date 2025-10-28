# Final Cleanup Plan - TacticAI Receiver Prediction

## TacticAI Requirements vs Our Data

### What TacticAI Needs:
1. ✅ Player positions (x, y)
2. ⚠️ Player velocities (vx, vy)
3. ⚠️ Player profiles (height, weight)
4. ✅ Graph structure
5. ✅ Receiver labels

### What StatsBomb 360 Provides:
- ✅ Player positions (freeze frames at corner moment)
- ❌ Player velocities (static snapshot, no motion)
- ❌ Player profiles (not included in free data)
- ✅ Graph structure (buildable from positions)
- ✅ Receiver labels (via Ball Receipt + pass_recipient)

### What SkillCorner Provides:
- ✅ Player positions (10fps tracking)
- ✅ Player velocities (derived from position changes)
- ❌ Player profiles (not in open data)
- ✅ Graph structure
- ⚠️ Receiver labels (need to add via Ball Receipt matching)

---

## Decision: **Hybrid Approach - Keep Both**

### Reasoning:
1. **StatsBomb**: Larger dataset (1,118 corners), receiver labels ready
2. **SkillCorner**: Provides real velocities (critical for TacticAI method)
3. **Temporal augmentation**: Simulates velocities for StatsBomb data
4. **Combined**: 7,369 graphs total (best of both worlds)

### What This Means:
- Use StatsBomb for bulk training (5,814 augmented graphs)
- Use SkillCorner for velocity validation (1,555 temporal graphs)
- Temporal augmentation fills velocity gap for StatsBomb data

---

## Files to Keep (Minimal Essential Pipeline)

### Core Data Pipeline (8 scripts)
```
✅ scripts/download_statsbomb_corners.py
✅ scripts/label_statsbomb_outcomes.py
✅ scripts/extract_skillcorner_temporal.py
✅ scripts/augment_statsbomb_temporal.py
✅ scripts/build_skillcorner_graphs.py
✅ scripts/preprocessing/add_receiver_labels.py
✅ scripts/train_gnn.py
```

### Core Source Files (10 files)
```
✅ src/statsbomb_loader.py
✅ src/outcome_labeler.py
✅ src/feature_engineering.py
✅ src/graph_builder.py
✅ src/receiver_labeler.py
✅ src/gnn_model.py
✅ src/train_utils.py
✅ src/data_loader.py
✅ src/data/receiver_data_loader.py
✅ src/data/__init__.py
```

### Core SLURM Scripts (8 scripts)
```
✅ scripts/slurm/redownload_statsbomb_corners.sh
✅ scripts/slurm/phase1_2_label_outcomes.sh
✅ scripts/slurm/phase2_3_skillcorner_temporal.sh
✅ scripts/slurm/phase2_4_statsbomb_augment.sh
✅ scripts/slurm/phase2_5_build_skillcorner_graphs.sh
✅ scripts/slurm/phase3_train_gnn.sh
✅ scripts/slurm/tacticai_day1_2_receiver_labels.sh
✅ scripts/slurm/test_receiver_data_loader.sh
```

### Tests (1 file)
```
✅ tests/test_receiver_data_loader.py
```

**Total Keep: 27 files**

---

## Files to Remove (65 files)

### Category 1: Alternative Downloads (4 files)
```
❌ scripts/download_all_statsbomb_data.py
❌ scripts/download_more_corners.py
❌ scripts/download_free_kicks.py
❌ scripts/simple_statsbomb_expand.py
```

### Category 2: Unused Data Sources (5 files)
```
❌ scripts/extract_skillcorner_corners.py (use temporal version instead)
❌ scripts/extract_soccernet_corners.py
❌ scripts/integrate_corner_datasets.py
❌ scripts/label_skillcorner_outcomes.py (outcomes in temporal already)
❌ scripts/label_soccernet_outcomes.py
```

### Category 3: Old Graph Building (1 file)
```
❌ scripts/build_graph_dataset.py (replaced by build_skillcorner_graphs.py)
```

### Category 4: Analysis/Inspection (13 files)
```
❌ scripts/analyze_class_distribution.py
❌ scripts/analyze_dataset_stats.py
❌ scripts/inspect_corner_columns.py
❌ scripts/inspect_receiver_graph.py
❌ scripts/inspect_receiver_source.py
❌ scripts/inspect_statsbomb_raw_360.py
❌ scripts/inspect_ussf_data.py
❌ scripts/test_download_receiver.py
❌ scripts/test_feature_extraction.py
❌ scripts/test_freeze_frame_structure.py
❌ scripts/test_receiver_position_matching.py
❌ scripts/test_statsbomb_360.py
❌ scripts/test_statsbomb_freeze_frame.py
```

### Category 5: Experimental Training (5 files)
```
❌ scripts/train_gnn_balanced.py
❌ scripts/quick_train.py
❌ scripts/evaluate_balanced_model.py
❌ scripts/evaluate_model.py
```

### Category 6: Experimental Data Processing (5 files)
```
❌ scripts/apply_smote.py
❌ scripts/merge_goal_into_shot.py
❌ scripts/merge_goal_into_shot_standalone.py
❌ scripts/test_split_fix.py
❌ scripts/test_imbalance_fixes.py
```

### Category 7: Visualization (4 files)
```
❌ scripts/visualization/visualize_all_corners.py
❌ scripts/visualization/visualize_corners_with_players.py
❌ scripts/visualization/visualize_graph_structure.py
❌ scripts/visualization/visualize_single_corner.py
```

### Category 8: Experimental Source Files (3 files)
```
❌ src/balanced_sampler.py
❌ src/balanced_metrics.py
❌ src/focal_loss.py
```

### Category 9: Bash Scripts (4 files)
```
❌ scripts/build_team_with_ball_graphs.sh
❌ scripts/monitor_training.sh
❌ scripts/train_fixed.sh
❌ scripts/train_gat_improved.sh
```

### Category 10: Experimental SLURM (18 files)
```
❌ scripts/slurm/analyze_class_distribution.sh
❌ scripts/slurm/apply_smote.sh
❌ scripts/slurm/download_all_statsbomb.sh
❌ scripts/slurm/download_more_data.sh
❌ scripts/slurm/download_ussf_data.sh
❌ scripts/slurm/evaluate_balanced_model.sh
❌ scripts/slurm/inspect_receiver_graph.sh
❌ scripts/slurm/merge_goal_into_shot.sh
❌ scripts/slurm/phase1_1_complete.sh
❌ scripts/slurm/phase2_1_extract_features.sh (not used for temporal)
❌ scripts/slurm/phase2_2_build_graphs.sh (not used for temporal)
❌ scripts/slurm/quick_test.sh
❌ scripts/slurm/test_ball_centric_adjacency.sh
❌ scripts/slurm/test_distance_adjacency.sh
❌ scripts/slurm/test_freeze_frame.sh
❌ scripts/slurm/test_gat_model.sh
❌ scripts/slurm/test_imbalance_fixes.sh
❌ scripts/slurm/test_split_fix.sh
❌ scripts/slurm/train_balanced_gnn.sh
❌ scripts/slurm/train_gat_improved.sh
❌ scripts/slurm/visualize_graphs.sh
❌ scripts/slurm/RUN_FULL_PIPELINE.sh (outdated)
```

### Category 11: Old Feature Scripts (Not Used in Temporal Pipeline)
```
❌ scripts/extract_corner_features.py (temporal extracts features directly)
```

**Total Remove: 65 files**

---

## Summary

**Keep**: 27 files (minimal TacticAI receiver prediction pipeline)
**Remove**: 65 files (experimental, analysis, alternatives)

**Pipeline**: StatsBomb (bulk data) + SkillCorner (velocity validation) → Temporal augmentation → Receiver prediction

Ready to proceed with cleanup?
