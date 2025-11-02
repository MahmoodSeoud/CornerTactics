# TacticAI Baseline Models Implementation (Days 5-6)

## Objective
Implement baseline models for receiver prediction to establish performance benchmarks before training the full GATv2 model.

## Requirements (from TACTICAI_IMPLEMENTATION_PLAN.md)

### Three Baseline Models:
1. **RandomReceiverBaseline**: Uniform random predictions over 22 players
2. **XGBoostReceiverBaseline**: Engineered spatial/contextual features
3. **MLPReceiverBaseline**: Flattened positions through simple neural network

### XGBoost Feature Engineering (22 players × ~15 features each):
- **Spatial**: distance to ball, distance to goal, x-position, y-position
- **Relative**: closest opponent distance, teammates within 5m radius
- **Zonal**: binary flags (in 6-yard box? in penalty area? near/far post?)
- **Team context**: average team x-position, defensive line compactness
- **Player role**: is_goalkeeper, is_corner_taker (binary flags)

### Success Criteria:
- ✅ Random baseline: top-1=4.5%, top-3=13.6% (sanity check)
- ✅ XGBoost baseline: top-1 > 25%, top-3 > 42%
- ✅ MLP baseline: top-1 > 22%, top-3 > 45%
- ❌ **If MLP top-3 < 40%**: STOP and debug data pipeline

### Deliverables:
- `src/models/baselines.py` (all three models)
- `scripts/training/train_baseline.py` (training script)
- `results/baseline_xgboost.json` (results)
- `results/baseline_mlp.json` (results)

## Implementation Notes

### Data Source:
- Using `data/graphs/adjacency_team/combined_temporal_graphs_with_receiver.pkl`
- Expected: ~996 corners with valid receiver labels (89.1% coverage)

### Architecture Details:
- **XGBoost**: max_depth=6, n_estimators=500, learning_rate=0.05
- **MLP**: 308 (22×14) → 256 → 128 → 22, Dropout=0.3, ReLU activations

### Existing Code Status:
- ✅ RandomReceiverBaseline already implemented
- ✅ MLPReceiverBaseline already implemented
- ❌ XGBoostReceiverBaseline missing - NEED TO IMPLEMENT
- ✅ evaluate_baseline() helper function exists
- ✅ train_mlp_baseline() training function exists

### Interface Design:
- Models use `forward(x, batch)` where `batch` is PyG batch tensor
- Models have `predict(x, batch)` method that returns softmax probabilities
- Need to update XGBoost to match this interface

## Questions/Clarifications:
- None currently - requirements are clear from implementation plan
