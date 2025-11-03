# Feature: Expand Receiver Labeling to Match TacticAI

## Goal
Increase receiver label coverage from 60% (3,492/5,814 corners) to 85-90% by labeling ANY first touch (attacking OR defending player), matching TacticAI's methodology.

## Current State Analysis

### TacticAI Approach (from paper)
- **Receiver**: "First player to touch the ball after corner was taken"
- **Can be**: Either attacking OR defensive player
- **Dataset**: 7,176 corners from 2020-2021 Premier League
- **No filtering**: All first touches count

### Our Current Approach
- **Receiver**: Found via `ReceiverLabeler.find_receiver()`
- **Valid events**: Already includes `'Clearance'` in VALID_RECEIVER_EVENTS
- **Problem**: Only 60% coverage despite including clearances

### Coverage Analysis (from investigation)
- **Total graphs**: 5,814
- **With receiver**: 3,492 (60.1%)
- **Without receiver**: 2,322 (39.9%)
  - **73.1% are Clearances** ← Key finding!

### Root Cause (CONFIRMED)
The receiver labeling script (`scripts/preprocessing/add_receiver_labels.py`) uses **location-based matching** from StatsBomb CSV, which requires:
1. `receiver_name` field (from Ball Receipt event)
2. `receiver_location_x/y` fields

**Problem**: Clearances don't create Ball Receipt events → no receiver data in CSV → can't match to graph nodes.

**Key Finding**: `ReceiverLabeler` class ALREADY handles defensive players correctly! The issue is that `add_receiver_labels.py` doesn't USE it - it just reads from the pre-processed CSV.

## Solution Approach

### Strategy (REFINED)
Re-label receivers by using `ReceiverLabeler` + event location matching:

1. **For each corner**:
   - Use `ReceiverLabeler.find_receiver()` to get first touch player (ANY team)
   - Extract that player's **event location** from the event stream
     - Ball Receipt events have `location`
     - Clearance events have `location` ✓
     - Interception events have `location` ✓
     - Duel events have `location` ✓
   - Match event location to closest freeze frame position
   - Support BOTH attacking and defending players

2. **Expected outcome**:
   - Increase coverage from 60% → 85-90% (matching TacticAI)
   - Recover ~2,000 missing clearance corners

### Key Insight
ALL StatsBomb events (not just Ball Receipt) have `location` field!

## Implementation Plan

### Phase 1: Test Infrastructure ✅ COMPLETE
- [x] Write test for ReceiverLabeler with clearance events
- [x] Write test for ReceiverLabeler returning location
- [x] Update ReceiverLabeler to return (player_id, player_name, location)
- [x] Update all existing tests
- [x] All tests passing

### Phase 2: Update Receiver Labeling Script ✅ COMPLETE
- [x] Create new script `add_receiver_labels_v2.py`
- [x] Implement `extract_receiver_from_events()` function
- [x] Implement `match_location_to_node()` function
- [x] Implement `add_receiver_labels_from_events()` main function
- [x] Write comprehensive tests (all passing)
- [x] Create SLURM script

### Phase 3: Full Dataset Re-labeling ⚠️ PARTIAL SUCCESS
- [x] Create SLURM job script
- [x] Submit job (Job ID: 30905, running on cn14)
- [x] Monitor execution (completed in ~2.5 minutes)
- [x] Fixed time window bug: increased from 5s → 15s
- [x] Results with 15s window: **67.7% coverage** (3,937/5,814)
  - Attacking receivers: 3,025 (76.8%)
  - Defending receivers: 912 (23.2%) ← **NEW CAPTURES!**
  - Comparison to old: gained 445 more receivers (3,937 vs old 3,492)
- [ ] ⚠️ Still below 85% target (need investigation)
- [ ] ⚠️ Avg distance to match: 32.61m (should be <5m) - position matching issue?

### Phase 4: Integration (PENDING)
- [ ] Verify data loader still works with v2 labels
- [ ] Re-train baselines with expanded dataset
- [ ] Compare results: old (3,492 corners) vs new (5,000+ corners)
- [ ] Create pull request

## Key Files
- `src/receiver_labeler.py` - Core receiver finding logic (already works!)
- `scripts/preprocessing/add_receiver_labels.py` - Needs update
- `data/graphs/adjacency_team/statsbomb_temporal_augmented.pkl` - Input graphs

## Questions/Clarifications
- Q: Should we include goalkeeper as potential receiver?
  - A: Yes, TacticAI includes all 22 players

- Q: What if player_id doesn't match any node?
  - A: Skip that corner (should be rare)

## Success Criteria
- Coverage increases from 60% → 85%+
- Clearance corners now have receiver labels
- All tests pass
