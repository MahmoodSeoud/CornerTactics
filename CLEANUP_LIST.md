# Cleanup List - Non-Essential Files to Remove

## Current Pipeline (TacticAI Days 3-4: Receiver Prediction)

### ✅ KEEP - Essential Core Pipeline

#### Core Scripts (scripts/)
- `download_statsbomb_corners.py` - Download corners with receiver info
- `label_statsbomb_outcomes.py` - Label corner outcomes
- `extract_corner_features.py` - Extract 14-dim node features (Phase 2.1)
- `build_skillcorner_graphs.py` - Build graph structures
- `extract_skillcorner_temporal.py` - Extract temporal features (Phase 2.3)
- `augment_statsbomb_temporal.py` - Temporal augmentation (Phase 2.4)
- `preprocessing/add_receiver_labels.py` - Add receiver labels (TacticAI Day 1-2)
- `train_gnn.py` - Main training script

#### Core Source (src/)
- `statsbomb_loader.py` - StatsBomb data loading
- `outcome_labeler.py` - Outcome labeling logic
- `feature_engineering.py` - Feature extraction
- `graph_builder.py` - Graph construction
- `receiver_labeler.py` - Receiver labeling logic
- `gnn_model.py` - GNN model implementation
- `train_utils.py` - Training utilities
- `data_loader.py` - Base data loader
- `data/receiver_data_loader.py` - Receiver prediction data loader
- `data/__init__.py`

#### Core SLURM (scripts/slurm/)
- `redownload_statsbomb_corners.sh` - Re-download with receiver info
- `phase1_2_label_outcomes.sh` - Label outcomes
- `phase2_1_extract_features.sh` - Feature extraction
- `phase2_3_skillcorner_temporal.sh` - SkillCorner temporal extraction
- `phase2_4_statsbomb_augment.sh` - StatsBomb temporal augmentation
- `phase2_5_build_skillcorner_graphs.sh` - Build graphs
- `phase3_train_gnn.sh` - GNN training
- `tacticai_day1_2_receiver_labels.sh` - Receiver label mapping
- `test_receiver_data_loader.sh` - Test receiver data loader
- `RUN_FULL_PIPELINE.sh` - Master pipeline script

#### Core Tests (tests/)
- `test_receiver_data_loader.py` - Main test for TacticAI Day 3-4

---

## ❌ DELETE - Non-Essential Files

### Category 1: Alternative/Experimental Download Scripts (4 files)
```
scripts/download_all_statsbomb_data.py - Alternative comprehensive download
scripts/download_more_corners.py - Experimental expansion
scripts/download_free_kicks.py - Free kicks (not corners)
scripts/simple_statsbomb_expand.py - Old expansion approach
```

### Category 2: Analysis & Inspection Scripts (13 files)
```
scripts/analyze_class_distribution.py - Class distribution analysis
scripts/analyze_dataset_stats.py - Dataset statistics
scripts/inspect_corner_columns.py - Column inspection
scripts/inspect_receiver_graph.py - Receiver graph inspection
scripts/inspect_receiver_source.py - Receiver source inspection
scripts/inspect_statsbomb_raw_360.py - Raw 360 data inspection
scripts/inspect_ussf_data.py - USSF data inspection
scripts/test_download_receiver.py - Test download receiver logic
scripts/test_feature_extraction.py - Test feature extraction
scripts/test_freeze_frame_structure.py - Test freeze frame structure
scripts/test_receiver_position_matching.py - Test position matching
scripts/test_statsbomb_360.py - Test StatsBomb 360 data
scripts/test_statsbomb_freeze_frame.py - Test freeze frame
```

### Category 3: Experimental Training/Model Scripts (5 files)
```
scripts/train_gnn_balanced.py - Alternative balanced training
scripts/quick_train.py - Quick test training
scripts/evaluate_balanced_model.py - Evaluate balanced model
scripts/evaluate_model.py - General evaluation script
```

### Category 4: Experimental Data Processing (5 files)
```
scripts/apply_smote.py - SMOTE balancing experiment
scripts/merge_goal_into_shot.py - Merge goal into shot labels
scripts/merge_goal_into_shot_standalone.py - Standalone version
scripts/test_split_fix.py - Test data split fix
scripts/test_imbalance_fixes.py - Test imbalance fixes
```

### Category 5: Unused Data Sources (6 files)
```
scripts/extract_skillcorner_corners.py - SkillCorner extraction (if not used)
scripts/extract_soccernet_corners.py - SoccerNet extraction (if not used)
scripts/integrate_corner_datasets.py - Dataset integration (if not needed)
scripts/label_skillcorner_outcomes.py - SkillCorner outcome labeling
scripts/label_soccernet_outcomes.py - SoccerNet outcome labeling
scripts/build_graph_dataset.py - Old graph building approach
```

### Category 6: Visualization Scripts (4 files)
```
scripts/visualization/visualize_all_corners.py - Batch corner visualization
scripts/visualization/visualize_corners_with_players.py - Corner visualization
scripts/visualization/visualize_graph_structure.py - Graph structure visualization
scripts/visualization/visualize_single_corner.py - Single corner visualization
```

### Category 7: Experimental Source Files (3 files)
```
src/balanced_sampler.py - Experimental balanced sampling
src/balanced_metrics.py - Experimental balanced metrics
src/focal_loss.py - Focal loss implementation (if not used in train_gnn.py)
```

### Category 8: Bash Scripts (4 files)
```
scripts/build_team_with_ball_graphs.sh - Build team graphs
scripts/monitor_training.sh - Monitor training progress
scripts/train_fixed.sh - Fixed training script
scripts/train_gat_improved.sh - Improved GAT training
```

### Category 9: Experimental SLURM Scripts (19 files)
```
scripts/slurm/analyze_class_distribution.sh
scripts/slurm/apply_smote.sh
scripts/slurm/download_all_statsbomb.sh
scripts/slurm/download_more_data.sh
scripts/slurm/download_ussf_data.sh
scripts/slurm/evaluate_balanced_model.sh
scripts/slurm/inspect_receiver_graph.sh
scripts/slurm/merge_goal_into_shot.sh
scripts/slurm/phase1_1_complete.sh - Dataset integration (not needed for StatsBomb-only)
scripts/slurm/phase2_2_build_graphs.sh - Old graph building approach
scripts/slurm/quick_test.sh
scripts/slurm/test_ball_centric_adjacency.sh
scripts/slurm/test_distance_adjacency.sh
scripts/slurm/test_freeze_frame.sh
scripts/slurm/test_gat_model.sh
scripts/slurm/test_imbalance_fixes.sh
scripts/slurm/test_split_fix.sh
scripts/slurm/train_balanced_gnn.sh
scripts/slurm/train_gat_improved.sh
scripts/slurm/visualize_graphs.sh
```

---

## Summary

**KEEP**: 34 files (core pipeline)
**DELETE**: 63 files (experimental/analysis/alternatives)

**Space saved**: Removes ~63 non-essential files
**Clarity gained**: Clear linear pipeline visible

Would you like me to proceed with deletion?
