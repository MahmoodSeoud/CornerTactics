# Flexible Ablation Framework for Corner Kick Receiver Prediction

**Date**: November 2025
**Principle**: Change ONE thing at a time for proper causal analysis

---

## Methodology

### Core Principle: Controlled Ablation

**❌ Bad Ablation (What NOT to do)**:
```
Config A: [23 features]
Config B: [14 features]  ← Removed 9 features at once!
```
*Can't tell which of the 9 features caused performance change*

**✅ Good Ablation (One change at a time)**:
```
Config 0: [23 features] (baseline)
Config 1: [22 features] (remove end_location_x only)
Config 2: [22 features] (remove end_location_y only)
Config 3: [22 features] (remove pass_length only)
...
Config N: [23 features] (baseline + new_feature_1)
```
*Can isolate each feature's individual contribution*

---

## Ablation Strategy

### Phase 1: Data Exploration (Do This First!)

Before designing configs, explore the data to identify:

1. **Feature Correlations**
   - Which features are highly correlated? (candidates for removal)
   - Which features correlate with receiver? (important to keep)
   - Which features are redundant?

2. **Feature Importance** (from baseline XGBoost)
   - Top 10 most important features → investigate why
   - Bottom features with <1% importance → candidates for removal
   - Surprising features (high/low importance)

3. **Leakage Suspects**
   - Features that use post-kick information
   - Features derived from outcome
   - Suspiciously high importance features

**Output**:
- `results/exploration/feature_correlation_matrix.csv`
- `results/exploration/feature_importance_baseline.csv`
- `results/exploration/leakage_suspects.json`

### Phase 2: Systematic Feature Removal

Based on exploration, test removal of:

**A. Confirmed Leakage Features** (one at a time)
- Remove `end_location_x` only
- Remove `end_location_y` only
- Remove `pass_length` only (derived from end_location)
- Remove `pass_angle` only (derived from end_location)

**B. Low Importance Features** (one at a time)
- Remove each feature with <1% importance
- Test if performance stays same (if yes, feature is redundant)

**C. Redundant Features** (one at a time)
- Remove one of each highly correlated pair (r > 0.9)

### Phase 3: Systematic Feature Addition

After finding minimal clean feature set, add features one-by-one:

**A. Position-based Features**
- Add `num_attackers_in_box` only
- Add `num_defenders_in_box` only
- Add `attackers_to_defenders_ratio` only

**B. Tactical Features**
- Add `corner_side` only
- Add `defensive_formation` only

**C. Historical Features**
- Add `corner_taker_success_rate` only
- Add `team_corner_conversion_rate` only

### Phase 4: Interaction Effects

After finding individually helpful features, test combinations:
- Add best 2 features together
- Add best 3 features together
- etc.

---

## Implementation Design

### 1. Feature Registry

**File**: `configs/feature_registry.yaml`

```yaml
features:
  # Temporal features
  minute:
    type: temporal
    source: event
    leakage: false
    always_include: true
    description: "Match minute when corner taken"

  second:
    type: temporal
    source: event
    leakage: false
    always_include: true
    description: "Second within minute"

  # Spatial features (LEAKAGE SUSPECTS)
  end_location_x:
    type: spatial
    source: event
    leakage: true  # ⚠️ Where ball lands (not known pre-kick)
    always_include: false
    description: "X coordinate where ball lands"
    ablation_priority: high  # Test removal first

  end_location_y:
    type: spatial
    source: event
    leakage: true  # ⚠️ Where ball lands
    always_include: false
    description: "Y coordinate where ball lands"
    ablation_priority: high

  pass_length:
    type: derived
    source: event
    leakage: true  # Derived from end_location
    depends_on: [end_location_x, end_location_y]
    always_include: false
    ablation_priority: high

  # Game context (clean features)
  team_id:
    type: categorical
    source: event
    leakage: false
    always_include: true
    description: "Team taking corner"

  player_id:
    type: categorical
    source: event
    leakage: false
    always_include: true
    description: "Corner taker ID"

  # Engineered features (to be added)
  num_attackers_in_box:
    type: engineered
    source: freeze_frame
    leakage: false
    implemented: false
    description: "Attacking players in penalty area"
    ablation_priority: medium

  # ... more features
```

### 2. Config Generator

**File**: `src/ablation_config_generator.py`

```python
class AblationConfigGenerator:
    """Generate feature configurations for ablation studies."""

    def __init__(self, registry_path: str):
        self.registry = self.load_registry(registry_path)
        self.baseline_features = self.get_baseline_features()

    def get_baseline_features(self) -> List[str]:
        """Get all current features (Config 0)."""
        return [f for f, meta in self.registry.items()
                if meta.get('source') == 'event']

    def remove_feature(self, feature: str) -> dict:
        """Generate config with ONE feature removed."""
        config = {
            'name': f'remove_{feature}',
            'features': [f for f in self.baseline_features if f != feature],
            'change': f'Removed: {feature}',
            'purpose': f'Test if {feature} is necessary'
        }
        return config

    def add_feature(self, feature: str, base_config: List[str]) -> dict:
        """Generate config with ONE feature added."""
        config = {
            'name': f'add_{feature}',
            'features': base_config + [feature],
            'change': f'Added: {feature}',
            'purpose': f'Test if {feature} improves performance'
        }
        return config

    def generate_removal_ablations(self) -> List[dict]:
        """Generate all single-feature removal configs."""
        configs = []
        for feature in self.baseline_features:
            if not self.registry[feature].get('always_include'):
                configs.append(self.remove_feature(feature))
        return configs

    def generate_addition_ablations(self, base_config: List[str]) -> List[dict]:
        """Generate all single-feature addition configs."""
        configs = []
        for feature, meta in self.registry.items():
            if meta.get('type') == 'engineered' and meta.get('implemented'):
                configs.append(self.add_feature(feature, base_config))
        return configs

    def generate_prioritized_ablations(self) -> List[dict]:
        """Generate configs ordered by priority."""
        # High priority: Suspected leakage features
        high_priority = [f for f, m in self.registry.items()
                        if m.get('ablation_priority') == 'high']

        configs = []
        for feature in high_priority:
            configs.append(self.remove_feature(feature))

        return configs
```

### 3. Flexible Training Script

**File**: `scripts/train_flexible_ablation.py`

```python
from src.ablation_config_generator import AblationConfigGenerator

def main():
    # Load feature registry
    generator = AblationConfigGenerator('configs/feature_registry.yaml')

    # Baseline (Config 0)
    baseline_config = {
        'name': 'baseline',
        'features': generator.baseline_features,
        'change': 'None (baseline)',
        'purpose': 'Reference performance'
    }

    # Phase 1: Explore data with baseline
    print("Phase 1: Data Exploration")
    results_baseline = train_and_evaluate(baseline_config)
    analyze_feature_importance(results_baseline)
    compute_feature_correlations(results_baseline)

    # Phase 2: Systematic removal (one at a time)
    print("\nPhase 2: Feature Removal Ablations")
    removal_configs = generator.generate_prioritized_ablations()

    for config in removal_configs:
        print(f"\nTesting: {config['name']}")
        print(f"  Change: {config['change']}")
        results = train_and_evaluate(config)
        compare_to_baseline(results, results_baseline)

    # Phase 3: Add engineered features (one at a time)
    print("\nPhase 3: Feature Addition Ablations")
    # Get best clean config from Phase 2
    best_clean_config = find_best_config_without_leakage()

    addition_configs = generator.generate_addition_ablations(
        best_clean_config['features']
    )

    for config in addition_configs:
        print(f"\nTesting: {config['name']}")
        results = train_and_evaluate(config)
        compare_to_baseline(results, best_clean_config)
```

---

## Example Workflow

### Step 1: Explore Baseline

```bash
# Train baseline model
python scripts/train_flexible_ablation.py --phase exploration

# Output:
# - results/exploration/baseline_performance.json
# - results/exploration/feature_importance.csv
# - results/exploration/correlation_matrix.csv
```

**Look at results**:
```
Feature Importance:
  1. end_location_x: 35.2% ⚠️ SUSPICIOUS (leakage?)
  2. end_location_y: 28.1% ⚠️ SUSPICIOUS
  3. player_id: 12.3%
  4. team_id: 8.5%
  ...

Correlation Matrix:
  pass_length <-> end_location_x: r=0.95 (redundant?)
  pass_angle <-> end_location_y: r=0.88 (redundant?)
```

### Step 2: Test Leakage Suspects (One at a Time)

```bash
# Config 1: Remove end_location_x ONLY
python scripts/train_flexible_ablation.py --config remove_end_location_x

# Config 2: Remove end_location_y ONLY
python scripts/train_flexible_ablation.py --config remove_end_location_y

# Config 3: Remove pass_length ONLY
python scripts/train_flexible_ablation.py --config remove_pass_length
```

**Results**:
```
Config 0 (baseline): Top-3 = 99.6%
Config 1 (- end_location_x): Top-3 = 87.2% (-12.4%) ← Big drop!
Config 2 (- end_location_y): Top-3 = 85.1% (-14.5%) ← Big drop!
Config 3 (- pass_length): Top-3 = 98.8% (-0.8%) ← Small drop (redundant)

Conclusion: end_location_x/y are major leakage sources!
```

### Step 3: Remove All Leakage Features

```bash
# Config 4: Remove end_location_x AND end_location_y
python scripts/train_flexible_ablation.py --config remove_end_locations

# Results: Top-3 = 78.2% (matches TacticAI!)
```

### Step 4: Add Engineered Features (One at a Time)

```bash
# Start from clean config (Config 4)
# Config 5: + num_attackers_in_box
python scripts/train_flexible_ablation.py --config add_num_attackers_in_box

# Config 6: + num_defenders_in_box
python scripts/train_flexible_ablation.py --config add_num_defenders_in_box
```

**Results**:
```
Config 4 (clean baseline): Top-3 = 78.2%
Config 5 (+ attackers): Top-3 = 79.8% (+1.6%) ← Helps!
Config 6 (+ defenders): Top-3 = 78.5% (+0.3%) ← Marginal
```

---

## Results Format

### Ablation Results Table

**CSV**: `results/ablation_one_feature_at_a_time.csv`

```csv
config_id,config_name,change_type,feature_changed,num_features,top1_acc,top3_acc,top5_acc,delta_top3,interpretation
0,baseline,none,none,23,0.996,0.996,0.997,0.000,Reference performance
1,remove_end_location_x,removal,end_location_x,22,0.850,0.872,0.891,-0.124,Major leakage feature
2,remove_end_location_y,removal,end_location_y,22,0.832,0.851,0.879,-0.145,Major leakage feature
3,remove_pass_length,removal,pass_length,22,0.988,0.988,0.992,-0.008,Redundant (derived)
4,remove_both_end_locs,removal,end_location_x/y,21,0.520,0.782,0.850,-0.214,Clean baseline
5,add_attackers_in_box,addition,num_attackers_in_box,22,0.650,0.798,0.862,+0.016,Helpful feature
```

### Feature Impact Summary

**JSON**: `results/ablation_feature_impact.json`

```json
{
  "leakage_features": {
    "end_location_x": {
      "importance": 0.352,
      "removal_impact": -0.124,
      "conclusion": "Major data leakage - causes 12.4% inflation"
    },
    "end_location_y": {
      "importance": 0.281,
      "removal_impact": -0.145,
      "conclusion": "Major data leakage - causes 14.5% inflation"
    }
  },
  "helpful_features": {
    "num_attackers_in_box": {
      "addition_impact": +0.016,
      "conclusion": "Modest improvement (+1.6%)"
    }
  },
  "redundant_features": {
    "pass_length": {
      "correlation_with": "end_location_x",
      "removal_impact": -0.008,
      "conclusion": "Redundant - can be removed without harm"
    }
  }
}
```

---

## Implementation Priority

### What to Build First

1. **Feature Registry** (`configs/feature_registry.yaml`)
   - Define all 23 current features with metadata
   - Mark leakage suspects
   - Takes 30 minutes

2. **Config Generator** (`src/ablation_config_generator.py`)
   - Auto-generate single-feature removal configs
   - Auto-generate single-feature addition configs
   - Takes 1 hour

3. **Flexible Training Script** (`scripts/train_flexible_ablation.py`)
   - Accept config as input
   - Train model
   - Save standardized results
   - Takes 2 hours

4. **Batch Runner** (`scripts/run_all_ablations.sh`)
   - Loop through all configs
   - Submit as SLURM array job
   - Takes 30 minutes

### What to Run First

1. **Baseline** (Config 0) → Get feature importance + correlations
2. **Remove suspected leakage** (Configs 1-5) → Quantify leakage
3. **Pause and analyze** → Decide which features to add
4. **Add engineered features** (Configs 6+) → One at a time based on analysis

---

## Advantages of This Approach

✅ **Proper causality**: Change one thing at a time
✅ **Data-driven**: Let exploration guide which configs to try
✅ **Flexible**: Easy to add new configs as you learn
✅ **Reproducible**: Config files document exactly what changed
✅ **Publishable**: Can show systematic ablation table in paper

---

## Next Steps

**Immediate**:
1. Review current 23 features - mark leakage suspects
2. Create feature registry YAML file
3. Run baseline to get feature importance

**Then**:
4. Design removal ablations based on importance scores
5. Implement config generator
6. Run systematic ablations

Would you like me to start with **Step 1: Creating the feature registry**?
