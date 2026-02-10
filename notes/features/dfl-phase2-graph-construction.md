# Phase 2: Graph Construction Pipeline

## Status: Complete

All tests passing (21 tests).

## Goal
Convert each tracking frame into a graph. Build the full dataset of labeled corner kick graphs.

## Implemented Functions

### graph_construction.py

1. **frame_to_graph(frame, velocities, corner_event, k_neighbors=4)**
   - Converts single tracking frame to PyTorch Geometric Data
   - 8 node features per player/ball
   - kNN edges (k=4), marking edges, ball edges

2. **corner_to_temporal_graphs(tracking_dataset, corner_event, ...)**
   - Converts corner kick to sequence of graphs
   - Adds frame_idx and relative_time metadata
   - Covers -2s to +6s around delivery

3. **label_corner(corner_event, event_dataset, n_subsequent_events=5)**
   - Creates multi-head labels from subsequent events
   - shot_binary, goal_binary, first_contact_team, outcome_class

4. **build_corner_dataset_from_match(tracking, events, match_id, ...)**
   - Processes all corners in a match
   - Returns list of {graphs, labels, match_id, corner_time}

5. **save_corner_dataset(dataset, path)** / **load_corner_dataset(path)**
   - Pickle serialization

6. **get_dataset_summary(dataset)**
   - Returns statistics: total, shot_rate, goal_rate, distributions

## Graph Structure (per frame)
- **Nodes**: num_players + 1 (ball) nodes
- **Node features** (8 total):
  - x, y: position (meters)
  - vx, vy: velocity (m/s)
  - team_flag: 1.0 (attacking), 0.0 (defending), -1.0 (ball)
  - is_kicker: 1.0 for corner taker (currently always 0)
  - dist_to_goal: Euclidean distance to goal center
  - dist_to_ball: Euclidean distance to ball

## Edge Types
1. **Proximity edges (kNN)**: k=4 nearest neighbors
2. **Marking edges**: Attacker-to-nearest-defender (bidirectional)
3. **Ball edges**: Ball connected to all players (bidirectional)

## Labels (Multi-head)
- shot_binary: 1 if shot within next 5 events
- goal_binary: 1 if goal within next 5 events
- first_contact_team: 'attacking', 'defending', or 'unknown'
- outcome_class: 'goal', 'shot_saved', 'shot_blocked', 'clearance', 'other'

## Dependencies
- torch
- torch-geometric
- scipy (for cdist)
- kloppy (from Phase 1)
