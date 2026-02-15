"""Training and evaluation for corner kick prediction."""

from .train import (
    build_model,
    train_receiver_epoch,
    train_shot_epoch,
    eval_receiver,
    eval_shot,
)
from .evaluate import lomo_cv, compute_receiver_metrics, compute_shot_metrics

__all__ = [
    "build_model",
    "train_receiver_epoch",
    "train_shot_epoch",
    "eval_receiver",
    "eval_shot",
    "lomo_cv",
    "compute_receiver_metrics",
    "compute_shot_metrics",
]
