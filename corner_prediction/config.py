"""Centralized configuration for corner kick prediction pipeline.

All hyperparameters, paths, and constants in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "corner_prediction" / "data"
PRETRAINED_PATH = PROJECT_ROOT / "transfer_learning" / "weights" / "ussf_backbone_dense.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "corner_prediction"

# ---------------------------------------------------------------------------
# Match IDs (10 A-League matches from SkillCorner)
# ---------------------------------------------------------------------------

MATCH_IDS = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]

# ---------------------------------------------------------------------------
# Training — Stage 1 (Receiver Prediction)
# ---------------------------------------------------------------------------

RECEIVER_LR = 1e-3
RECEIVER_EPOCHS = 100
RECEIVER_PATIENCE = 20
RECEIVER_WEIGHT_DECAY = 1e-3
RECEIVER_HIDDEN = 64
RECEIVER_DROPOUT = 0.3

# ---------------------------------------------------------------------------
# Training — Stage 2 (Shot Prediction)
# ---------------------------------------------------------------------------

SHOT_LR = 1e-3
SHOT_EPOCHS = 100
SHOT_PATIENCE = 20
SHOT_WEIGHT_DECAY = 1e-3
SHOT_HIDDEN = 32
SHOT_DROPOUT = 0.3
SHOT_POS_WEIGHT = 2.0  # upweight shots (33.7% minority)

# ---------------------------------------------------------------------------
# General Training
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
SEEDS = [42, 123, 456, 789, 1234]

# ---------------------------------------------------------------------------
# Permutation Test
# ---------------------------------------------------------------------------

N_PERMUTATIONS = 100

# ---------------------------------------------------------------------------
# Ablation Configurations
#
# Each config specifies which of the 13 node feature indices to keep active.
# Inactive features are zeroed out. The 14th feature (receiver indicator) is
# always handled by TwoStageModel and not part of the mask.
#
# Feature indices:
#   0: x_norm, 1: y_norm, 2: vx, 3: vy, 4: speed,
#   5: is_attacking, 6: is_corner_taker, 7: is_goalkeeper,
#   8: is_detected, 9: group_GK, 10: group_DEF, 11: group_MID, 12: group_FWD
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "position_only": {
        "description": "x, y, team flags, role (no velocity, no detection)",
        "active_features": [0, 1, 5, 6, 7, 9, 10, 11, 12],
        "edge_type": "knn",
        "k": 6,
    },
    "plus_velocity": {
        "description": "Position + velocity features",
        "active_features": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
        "edge_type": "knn",
        "k": 6,
    },
    "plus_detection": {
        "description": "Position + velocity + detection flag (= all 13)",
        "active_features": list(range(13)),
        "edge_type": "knn",
        "k": 6,
    },
    "full_features": {
        "description": "All 13 features, KNN edges (default model)",
        "active_features": list(range(13)),
        "edge_type": "knn",
        "k": 6,
    },
    "full_fc_edges": {
        "description": "All 13 features, fully connected edges",
        "active_features": list(range(13)),
        "edge_type": "dense",
        "k": 6,
    },
}
