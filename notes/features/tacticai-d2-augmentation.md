# TacticAI Days 8-9: D2 Augmentation Implementation

## Feature Overview
Implement D2 symmetry augmentation for corner kick graphs to make the model equivariant to field reflections (horizontal flip, vertical flip, both).

## Requirements (from TACTICAI_IMPLEMENTATION_PLAN.md)

### Day 8-9 Tasks:
1. Create `src/data/augmentation.py` with:
   - `D2Augmentation` class
   - `apply_transform(x, transform_type)`: h-flip, v-flip, both-flip
     - H-flip: `x[:, 0] = 120 - x[:, 0]`, `x[:, 4] = -x[:, 4]` (flip vx)
     - V-flip: `x[:, 1] = 80 - x[:, 1]`, `x[:, 5] = -x[:, 5]` (flip vy)
     - Both-flip: Apply both transformations
   - `get_all_views(x, edge_index)`: Generate 4 D2 views
2. Unit tests: `tests/test_augmentation.py`
   - Test: Apply h-flip twice = identity
   - Test: Edge structure unchanged across transforms
3. Visual test: Plot all 4 views of a corner kick (use mplsoccer)
   - Save to `data/results/d2_augmentation_demo.png`

### Success Criteria:
- All 4 D2 transforms implemented correctly
- Unit tests pass
- Visual inspection: 4 views look geometrically correct

## Understanding Existing Code

### Coordinate System (StatsBomb):
- Pitch dimensions: 120 x 80 units
- X: 0 (defensive) to 120 (attacking)
- Y: 0 (bottom) to 80 (top)

### Feature Vector (14 dimensions from feature_engineering.py):
Looking at existing code to understand feature indices...
- [0]: x position
- [1]: y position
- [2-3]: distance_to_goal, distance_to_ball_target
- [4]: vx (velocity x)
- [5]: vy (velocity y)
- [6-7]: velocity_magnitude, velocity_angle
- [8-9]: angle_to_goal, angle_to_ball
- [10]: team_flag
- [11]: in_penalty_box
- [12-13]: num_players_within_5m, local_density_score

### D2 Group (Dihedral group of order 2):
- Identity: No transformation
- H-flip: Horizontal reflection (flip along x-axis at x=60)
- V-flip: Vertical reflection (flip along y-axis at y=40)
- Both-flip: 180° rotation (both reflections)

## Implementation Notes

### Key Design Decisions:
1. Keep transformations in-place for memory efficiency
2. Preserve edge structure (adjacency unchanged by geometric transforms)
3. Update velocity components (vx, vy) to match coordinate flip
4. Keep angle features as-is (they're relative, not absolute)
5. Team flag and density features unchanged

### Questions to Clarify:
- Q: Should angle features (angle_to_goal, angle_to_ball, velocity_angle) be transformed?
- A: velocity_angle should be updated when velocities flip. angle_to_goal and angle_to_ball depend on ball/goal positions which also flip, so they should be recomputed or adjusted.

### Testing Strategy:
1. Unit test: Double h-flip returns to identity
2. Unit test: Edge indices unchanged
3. Unit test: Feature ranges preserved (x in [0,120], y in [0,80])
4. Visual test: 4 views plotted on pitch

## Progress Tracking
- [x] Create augmentation.py module
- [x] Implement D2Augmentation class
- [x] Write unit tests (17/17 passing)
- [x] Create visual demo script
- [x] Verify all success criteria

## Final Summary

**Completed**: October 29, 2025

**Deliverables**:
1. `src/data/augmentation.py` - D2Augmentation class with 4 transformations
2. `tests/test_augmentation.py` - 17 comprehensive unit tests (all passing)
3. `scripts/visualization/visualize_d2_augmentation.py` - Visual demo script
4. `data/results/d2_augmentation_demo.png` - 2x2 visualization of 4 D2 views

**Key Implementation Details**:
- Algebraic angle transformations (θ → π - θ for h_flip, θ → -θ for v_flip)
- Angle normalization to [-π, π] using atan2(sin(θ), cos(θ)) for perfect involution
- Edge structure preserved across all transformations
- Spatial coordinates and velocities correctly transformed
- All transformations satisfy involution property (double flip = identity)

**Test Coverage**:
- Identity transformation
- Position flips (h, v, both)
- Velocity flips (h, v, both)
- Involution properties (flip twice = identity)
- Edge structure preservation
- Feature range validation
- Team flag preservation
- Batch transformations
- Error handling (invalid transform types)

**All Success Criteria Met** ✅
