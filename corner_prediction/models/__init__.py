"""Corner kick prediction models.

Two-stage pipeline:
    Stage 1: Receiver prediction (per-node classification)
    Stage 2: Conditional shot prediction (graph-level classification)
"""

from .backbone import CornerBackbone
from .receiver_head import ReceiverHead, masked_softmax, receiver_loss
from .shot_head import ShotHead
from .two_stage import TwoStageModel

__all__ = [
    "CornerBackbone",
    "ReceiverHead",
    "masked_softmax",
    "receiver_loss",
    "ShotHead",
    "TwoStageModel",
]
