# CornerTactics Documentation

**Last Updated**: November 2025

---

## Core Documentation

### 1. [FLEXIBLE_ABLATION_FRAMEWORK.md](FLEXIBLE_ABLATION_FRAMEWORK.md)
**Primary methodology document** for systematic feature ablation studies.

**Purpose**:
- Design ablation experiments (change one feature at a time)
- Identify data leakage
- Test feature engineering improvements

**Key Sections**:
- One-at-a-time ablation methodology
- Feature registry system
- Config generator design
- Systematic removal/addition workflow

**Status**: ✅ Current (Nov 2025)

---

### 2. [raw_dataset_plan.md](raw_dataset_plan.md)
**Training plan for 34K corner kick dataset** using raw StatsBomb features.

**Purpose**:
- Phase 1: Data preparation & baseline training
- Phase 2: Feature importance analysis
- Phase 3: Ablation study
- Phase 4: Analysis & documentation

**Dataset**: 21,656 corners (raw StatsBomb event data)

**Status**: ⏳ In Progress
- ✅ Phases 1-3 complete
- ⚠️ Receiver extraction fix needed (data leakage identified)
- ⏳ Phase 4 pending

---

### 3. [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md)
**GNN dataset specification** (5,814 temporally augmented graphs).

**Purpose**:
- Complete dataset documentation
- Receiver labeling methodology (100% coverage)
- 3-class outcome system
- Statistical summary and quality metrics

**Dataset**: 5,814 graphs from 1,118 base corners (5.2× augmentation)

**Status**: ✅ Complete

**Use For**:
- GNN model training (TacticAI replication)
- Graph-based receiver prediction
- Multi-class outcome prediction

---

### 4. [OUTCOME_BASELINE_DOCUMENTATION.md](OUTCOME_BASELINE_DOCUMENTATION.md)
**Baseline results** for multi-class outcome prediction (Shot/Clearance/Possession).

**Purpose**:
- GNN vs MLP performance comparison
- Baseline training procedures
- Evaluation metrics (Macro F1, per-class F1)

**Status**: ✅ Complete (reference results)

---

### 5. [RAW_DATA_LOCATIONS.md](RAW_DATA_LOCATIONS.md)
**Data file locations** and directory structure.

**Purpose**:
- Where raw data is stored
- Processed data locations
- Result file paths

**Status**: ✅ Reference

---

## Archived Documentation

Older documentation has been moved to `docs/archive/` for reference:

- `ABLATION_EXPERIMENTAL_DESIGN.md` - Superseded by FLEXIBLE_ABLATION_FRAMEWORK.md
- `TRAINING_STATUS.md` - Outdated status document
- `DATASET_GENERATION_STATUS.md` - Outdated status document
- `CORNER_*` analysis files - Early exploratory analysis
- `DATA_AND_FEATURES_SUMMARY.md` - Outdated summary
- Other outdated docs

These can be deleted if not needed for historical reference.

---

## Documentation Hierarchy

```
Current Work (Ablation Studies):
├── FLEXIBLE_ABLATION_FRAMEWORK.md  ← Start here
├── raw_dataset_plan.md             ← Dataset context
└── DATASET_DOCUMENTATION.md        ← If using GNN dataset

Reference:
├── OUTCOME_BASELINE_DOCUMENTATION.md  ← Baseline results
└── RAW_DATA_LOCATIONS.md              ← Data paths

Archive:
└── archive/                           ← Historical docs
```

---

## Quick Start

**For Ablation Studies**: Read `FLEXIBLE_ABLATION_FRAMEWORK.md`

**For Dataset Info**: Read `raw_dataset_plan.md` (34K raw) or `DATASET_DOCUMENTATION.md` (5.8K graphs)

**For Baselines**: Read `OUTCOME_BASELINE_DOCUMENTATION.md`

---

## Maintenance

**When to Update**:
- After completing major phases in `raw_dataset_plan.md`
- When dataset statistics change
- When new methodologies are added

**What to Archive**:
- Status documents older than 1 month
- Superseded methodology docs
- Exploratory analysis that's been finalized

**Keep docs/** clean and focused on current work!
