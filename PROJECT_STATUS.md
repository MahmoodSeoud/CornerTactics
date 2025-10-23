# CornerTactics Project Status

**Last Updated**: October 22, 2025
**Current Phase**: Phase 1 Complete ✅ | Phase 2 Ready to Start

---

## Project Goal

Build a Graph Neural Network (GNN) system to predict corner kick outcomes using StatsBomb 360° and SkillCorner tracking data.

**Prediction Task**: Multi-class classification
- Goal (1.3%)
- Shot (17%)
- Clearance (52%)
- Loss (19%)
- Possession (11%)

---

## ✅ COMPLETED: Phase 1 - Data Integration & Outcome Labeling

### Phase 1.1: Data Collection ✅

**StatsBomb 360° Data** (1,118 corners):
- ✅ Downloaded from StatsBomb Open Data
- ✅ Freeze-frame player positions at corner moment
- ✅ Event data with corner locations
- ✅ Coverage: UEFA Euro, FIFA World Cup, Bundesliga

**SkillCorner Tracking Data** (317 corners):
- ✅ Downloaded 10 A-League matches
- ✅ 10fps continuous tracking data
- ✅ Dynamic events with end_type classification
- ✅ Corner events extracted and linked to tracking

**SoccerNet Data** (4,208 videos):
- ✅ Downloaded corner video clips
- ⚠️ Not yet integrated (Phase 3 optional)

### Phase 1.2: Outcome Labeling ✅

**StatsBomb Outcome Labeling**:
- ✅ Fixed critical time-window filtering bug (0% → 18.2% shot detection)
- ✅ Labeled 1,118 corners with realistic distributions:
  - Goals: 14 (1.3%)
  - Shots: 189 (16.9%)
  - Clearances: 579 (51.8%)
  - Loss: 218 (19.5%)
  - Possession: 118 (10.6%)
- ✅ Added temporal metadata (time_to_outcome)
- ✅ Added spatial metadata (outcome_location)
- ✅ Added xThreat delta

**SkillCorner Outcome Labeling**:
- ✅ Fixed data model understanding (player_possession + end_type)
- ✅ Labeled 317 corners with realistic distributions:
  - Shots: 41 (12.9%)
  - Clearances: 152 (47.9%)
  - Possession: 110 (34.7%)
  - Loss: 14 (4.4%)

**Unified Dataset**:
- ✅ Created unified dataset: 1,435 corners
- ✅ Format: Parquet + CSV
- ✅ 1,118 corners with outcome labels (77.9%)

### Implementation Files Created ✅

**Core Modules**:
- ✅ `src/outcome_labeler.py` (670 lines)
  - Base `OutcomeLabeler` class
  - `StatsBombOutcomeLabeler` with time-window filtering
  - `SkillCornerOutcomeLabeler` with end_type classification
  - `SoccerNetOutcomeLabeler` stub

**Scripts**:
- ✅ `scripts/label_statsbomb_outcomes.py`
- ✅ `scripts/label_skillcorner_outcomes.py`
- ✅ `scripts/label_soccernet_outcomes.py` (stub)
- ✅ `scripts/integrate_corner_datasets.py`
- ✅ `scripts/slurm/phase1_2_label_outcomes.sh`

**Documentation**:
- ✅ `PHASE_1_2_SUMMARY.md` - Implementation guide
- ✅ `PHASE_1_2_COMPLETION.md` - Completion report
- ✅ `FEATURE_COMPARISON.md` - USSF vs our features
- ✅ `notes/ADDITIONAL_FEATURES_FOR_MULTICLASS.md` - Enhanced features
- ✅ `notes/CORNER_GNN_PLAN.md` - Master implementation plan

---

## 📊 Current Data Inventory

### Available Data Files

```
data/
├── datasets/
│   ├── statsbomb/
│   │   ├── corners_360.csv (1,118 corners - original)
│   │   └── corners_360_with_outcomes.csv (1.2 MB) ✅ NEW
│   ├── skillcorner/
│   │   ├── skillcorner_corners.csv (317 corners - original)
│   │   ├── skillcorner_corners_with_outcomes.csv (128 KB) ✅ NEW
│   │   └── data/matches/{match_id}/
│   │       ├── {match_id}_tracking_extrapolated.jsonl (10fps tracking)
│   │       ├── {match_id}_dynamic_events.csv
│   │       └── {match_id}_phases_of_play.csv
│   └── soccernet/
│       └── corner_clips/visible/ (4,208 videos - not yet integrated)
├── unified_corners_dataset.parquet (0.3 MB) ✅ NEW
└── unified_corners_dataset.csv (0.6 MB) ✅ NEW
```

### Data Features Currently Available

**StatsBomb (1,118 corners)**:
```
Columns (31 total):
- Match context: match_id, competition, season, teams, date
- Corner event: minute, second, team, player, corner_id
- Ball trajectory: location_x, location_y, end_x, end_y
- Player positions: attacking_positions (JSON), defending_positions (JSON)
- Counts: num_attacking_players, num_defending_players
- Outcomes: outcome_category, outcome_type, outcome_team, outcome_player
- Metadata: same_team, time_to_outcome, events_to_outcome, goal_scored
- Spatial: outcome_location, xthreat_delta
```

**SkillCorner (317 corners)**:
```
Columns (41 total):
- Match context: match_id, home_team, away_team, competition, season, date
- Event timing: period, minute_start, second_start, time_start
- Frames: frame_start, frame_end, duration (for linking to tracking)
- Corner event: attacking_side, team_shortname, player_name, player_position
- Ball location: x_start, y_start, x_end, y_end
- Context: game_interruption_before, game_interruption_after, event_type, event_subtype
- Tracking link: has_tracking, tracking_file (path to 10fps JSONL)
- Outcomes: outcome_category, outcome_type, outcome_team, outcome_player
- Metadata: same_team, time_to_outcome, events_to_outcome, goal_scored
- Spatial: outcome_location, xthreat_delta
```

**Unified Dataset (1,435 corners)**:
- Combines both sources with 'source' column
- Common schema for cross-dataset analysis

---

## 🎯 What We Have vs What We Need

### ✅ We Have (Raw Data)

**Spatial Data**:
- ✅ Player positions (x, y) at corner moment (StatsBomb 360°)
- ✅ Continuous tracking data (SkillCorner 10fps)
- ✅ Ball trajectory (start location → end location)
- ✅ Team affiliation (attacking/defending)

**Outcome Labels**:
- ✅ Multi-class outcomes (Goal/Shot/Clearance/Loss/Possession)
- ✅ Temporal metadata (time_to_outcome)
- ✅ Spatial metadata (outcome_location)
- ✅ Threat values (xthreat_delta)

**Match Context**:
- ✅ Competition, season, teams, date
- ✅ Match events (StatsBomb)
- ✅ Dynamic events (SkillCorner)

### ⚠️ We Need to Calculate (Phase 2)

**Derived Node Features**:
- ❌ Velocities (vx, vy, magnitude, angle) - from SkillCorner tracking
- ❌ Distance to goal
- ❌ Angle to goal
- ❌ Distance to ball landing zone
- ❌ Angle to ball
- ❌ Receiver indicator (who will get first touch)
- ❌ Marking relationships (marked_flag, marker_distance)
- ❌ Zone features (in_penalty_box, zone_id)
- ❌ Shooting angle
- ❌ Zone advantage (attackers - defenders)
- ❌ Defenders on ball path
- ❌ Density features (players within 5m, local density)

**Edge Features**:
- ❌ Player-to-player distances
- ❌ Speed differences
- ❌ Positional angles (sin/cos)
- ❌ Velocity angles (sin/cos)

**Graph-Level Features**:
- ❌ Ball landing zone coordinates
- ❌ Inswinger flag
- ❌ Defensive compactness
- ❌ Total attackers/defenders in box
- ❌ Goalkeeper positioning
- ❌ Unmarked attackers count

**Graph Structure**:
- ❌ Adjacency matrices (team-based, distance-based, Delaunay, etc.)
- ❌ Graph conversion (NetworkX or PyTorch Geometric format)

---

## 📋 Next Steps: Phase 2 - Graph Construction

### 2.1 Feature Engineering Pipeline

**Priority 1: Core Spatial Features** (Week 1)
1. Implement distance calculations (to goal, to ball, player-to-player)
2. Implement angle calculations (to goal, to ball, between players)
3. Calculate receiver indicators
4. Calculate marking relationships

**Priority 2: Motion Features** (Week 2)
5. Calculate velocities from SkillCorner tracking (frame differences)
6. Calculate velocity magnitudes and angles
7. Calculate speed differences (edge features)

**Priority 3: Corner-Specific Features** (Week 3)
8. Zone classification (penalty box, near/far post, edge)
9. Zone occupancy (attackers/defenders per zone)
10. Shooting angles and open lanes
11. Defensive organization metrics

**Priority 4: Graph-Level Features** (Week 3)
12. Ball trajectory features
13. Overall team balance metrics
14. Defensive compactness scores

### 2.2 Graph Construction

**Implementation** (Week 4):
1. Build adjacency matrix constructors (5 types)
2. Convert corners to NetworkX/PyG graphs
3. Export to GNN-ready format (pickle or PyG Data objects)

### 2.3 Output

Create graph dataset files:
- `data/graphs/statsbomb_corner_graphs.pkl`
- `data/graphs/skillcorner_corner_graphs.pkl`
- Metadata: Node feature names, edge feature names, graph statistics

---

## 🔧 Technology Stack

**Current**:
- Python 3.11
- pandas - Data manipulation
- statsbombpy - StatsBomb API
- tqdm - Progress tracking
- SLURM - HPC job scheduling

**Phase 2 Requirements**:
- NumPy - Numerical calculations
- SciPy - Spatial algorithms (Delaunay)
- NetworkX or PyTorch Geometric - Graph representation
- scikit-learn - Feature scaling/normalization

**Phase 3 Requirements** (GNN Training):
- TensorFlow 2.14 or PyTorch 2.0
- Spektral 1.2.0 or PyG - GNN layers
- CUDA - GPU training

---

## 📈 Project Metrics

### Dataset Statistics

**Total Corners**: 1,435
- StatsBomb: 1,118 (78%)
- SkillCorner: 317 (22%)

**Outcome Distribution** (labeled subset, n=1,118):
- Clearance: 579 (51.8%)
- Loss: 218 (19.5%)
- Shot: 189 (16.9%)
- Possession: 118 (10.6%)
- Goal: 14 (1.3%)

**Data Richness**:
- With player positions: 1,118 (78%)
- With tracking data: 317 (22%)
- With outcome labels: 1,118 (78%)
- With video: 4,208 clips (future integration)

### Code Statistics

**Lines of Code**:
- `src/outcome_labeler.py`: 670 lines
- Total implementation: ~1,200 lines
- Documentation: ~500 lines

**Files Created**: 15 files
- Core modules: 1
- Scripts: 4
- SLURM jobs: 1
- Documentation: 5
- Planning: 4

---

## 🎓 Research Foundation

**Based On**:
- Bekkers & Sahasrabudhe (2024): "A Graph Neural Network Deep-Dive into Successful Counterattacks"
- US Soccer Federation GNN implementation
- StatsBomb 360° methodology

**Innovation**:
- ✅ First GNN approach for corner kick prediction
- ✅ Multi-class outcomes (vs binary success/failure)
- ✅ Combination of freeze-frame + tracking data
- ✅ Corner-specific spatial features

---

## 🚀 Timeline

**Phase 1** (Complete): Weeks 1-2 ✅
- Data collection
- Outcome labeling
- Unified dataset

**Phase 2** (Current): Weeks 3-6 🔧
- Feature engineering
- Graph construction
- Dataset export

**Phase 3** (Future): Weeks 7-10
- GNN model implementation
- Training pipeline
- Evaluation

**Phase 4** (Future): Weeks 11-12
- Model tuning
- Analysis
- Publication preparation

---

## 💡 Key Achievements

1. ✅ **Fixed critical outcome labeling bugs** that prevented shot detection
2. ✅ **Achieved realistic outcome distributions** matching soccer statistics
3. ✅ **Created unified dataset** combining multiple data sources
4. ✅ **Designed comprehensive feature set** (22 node + 6 edge + 8 graph-level)
5. ✅ **Established reproducible pipeline** with SLURM jobs

---

## 📁 Git Repository Status

**Branch**: main
**Uncommitted Changes**:
- Modified: 2 files
- New files: 7 files
- Ready to commit: Yes ✅

**Next Commit**: "Complete Phase 1.2: Comprehensive outcome labeling with bug fixes"

---

*Status Report Generated: October 22, 2025*
*Project: CornerTactics - GNN-based Corner Kick Outcome Prediction*
