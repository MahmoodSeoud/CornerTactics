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

### Phase 1: Test Infrastructure
- [ ] Write test for ReceiverLabeler with clearance events
- [ ] Write test for matching player_id to node index

### Phase 2: Update Receiver Labeling Script
- [ ] Modify `add_receiver_labels.py` to use event stream
- [ ] Load StatsBomb events for each match
- [ ] Use ReceiverLabeler.find_receiver() for all corners
- [ ] Match player_id → node_index using graph.player_ids

### Phase 3: Validation
- [ ] Run on full dataset
- [ ] Verify coverage increases to 85%+
- [ ] Check that clearances now have receiver labels
- [ ] Validate that receiver predictions still work

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
