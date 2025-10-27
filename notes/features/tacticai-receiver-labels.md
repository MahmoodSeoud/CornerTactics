# TacticAI Receiver Label Extraction (Day 1-2)

## Feature Goal
Extract receiver_player_id for corner kicks by identifying which player touches the ball 0-5 seconds after the corner event.

## Requirements from Implementation Plan
- [ ] Extract receiver_player_id from StatsBomb events (player who touches ball 0-5s after corner)
- [ ] Add receiver_player_id to CornerGraph.metadata
- [ ] Add player_ids list to CornerGraph.metadata (maps node index 0-21 to StatsBomb player IDs)
- [ ] Run script on existing graphs
- [ ] Verify coverage: Target 85%+ of corners have valid receiver labels (expect ~950/1118)
- [ ] Save updated graphs: `data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl`

## Success Criteria
- ✅ At least 900 corners with valid receiver labels
- ✅ Receiver distribution: strikers (30-40%), midfielders (30-40%), defenders (20-30%)

## Current Understanding

### Existing Data Structures
- CornerGraph dataclass in `src/graph_builder.py`
- StatsBomb event data loaded via `src/statsbomb_loader.py`
- Existing graphs: `data/graphs/adjacency_team/combined_temporal_graphs.pkl`

### Key Questions
1. What is the current CornerGraph metadata structure?
2. How do we access StatsBomb events for a given corner?
3. How do we identify the receiver from subsequent events?
4. What events count as "touching the ball"?

## Implementation Notes

### Key Decisions Made
1. **Receiver Definition**: First player (excluding corner taker) who touches ball within 0-5 seconds
2. **Valid Event Types**: Pass, Shot, Duel, Interception, Clearance, Miscontrol, Ball Receipt*
3. **Time Window**: 5.0 seconds (per TacticAI plan)
4. **Exclusion**: Corner taker excluded (handles short corners correctly)

### Files Created
- `src/receiver_labeler.py`: Core receiver identification logic
- `scripts/preprocessing/add_receiver_labels.py`: Script to add receiver labels to graphs
- `scripts/slurm/tacticai_day1_2_receiver_labels.sh`: SLURM job script
- `tests/test_receiver_labeler.py`: Unit tests for ReceiverLabeler

### CornerGraph Updates
Added three new optional fields:
- `receiver_player_id: Optional[int]`: StatsBomb player ID
- `receiver_player_name: Optional[str]`: Player name
- `receiver_node_index: Optional[int]`: Node index (0-21) in graph

### Next Steps
1. Submit SLURM job: `sbatch scripts/slurm/tacticai_day1_2_receiver_labels.sh`
2. Check logs: `tail -f logs/receiver_labels_*.out`
3. Verify coverage >= 85% (900+ receivers out of ~1118 corners)
4. Analyze receiver distribution by position
