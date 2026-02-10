# Phase 2: Graph Construction Pipeline

## Goal
Convert each tracking frame into a graph. Build the full dataset of labeled corner kick graphs.

## Key Design Decisions

### Graph Structure (per frame)
- **Nodes**: 22 players + 1 ball = 23 nodes
- **Node features** (8 total):
  - x, y: position
  - vx, vy: velocity
  - team_flag: 1.0 for attacking, 0.0 for defending
  - is_kicker: 1.0 for corner taker
  - dist_to_goal: Euclidean distance to goal center
  - dist_to_ball: Euclidean distance to ball

### Edge Types
1. **Proximity edges (kNN)**: k=4 nearest neighbors
2. **Marking edges**: Attacker-to-nearest-defender (bidirectional)
3. **Ball edges**: Ball connected to all players

### Labels (Multi-head)
- shot_binary: 1 if shot within next 5 events
- goal_binary: 1 if goal within next 5 events
- first_contact_team: 'attacking' or 'defending'
- outcome_class: 6-class categorical

## Implementation Steps
1. Install torch-geometric (Step 2.1)
2. frame_to_graph function (Step 2.2)
3. corner_to_temporal_graphs function (Step 2.3)
4. label_corner function (Step 2.4)
5. build_corner_dataset function (Step 2.5)

## Dependencies
- torch
- torch-geometric
- scipy (for cdist)
