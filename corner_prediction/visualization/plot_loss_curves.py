"""Figure 7: Training loss curves across LOMO folds.

Shows per-epoch train and validation loss for both stages (receiver, shot),
with individual fold curves in light color and the mean overlaid.
Useful for diagnosing overfitting, convergence, and early stopping behavior.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _pad_to_length(curves: List[List[float]], length: int) -> np.ndarray:
    """Pad variable-length curves with NaN to a uniform length.

    Returns [n_folds, length] array.
    """
    padded = np.full((len(curves), length), np.nan)
    for i, curve in enumerate(curves):
        padded[i, :len(curve)] = curve
    return padded


def _compute_ylim(train_arr: np.ndarray, val_arr: np.ndarray) -> tuple:
    """Compute y-axis limits using the 5th-95th percentile range.

    Prevents outlier folds (e.g. first-epoch loss spikes from linear head
    initialization) from blowing up the axis and hiding meaningful dynamics.
    """
    all_vals = np.concatenate([train_arr.ravel(), val_arr.ravel()])
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) == 0:
        return (0, 1)
    p5 = np.percentile(all_vals, 5)
    p95 = np.percentile(all_vals, 95)
    margin = (p95 - p5) * 0.15
    ymin = max(0, p5 - margin)
    ymax = p95 + margin
    return (ymin, ymax)


def plot_loss_curves(
    fold_results: List[Dict],
    model_name: str = "GNN",
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot train/val loss curves for receiver and shot stages.

    Args:
        fold_results: List of per-fold dicts, each containing 'loss_history'
            with keys 'receiver' and 'shot', each having 'train' and 'val'.
        model_name: Label for the figure title.
        output_path: Path to save PDF.
        show: Whether to display interactively.

    Returns:
        matplotlib Figure.
    """
    # Collect curves from folds that have loss history
    recv_train, recv_val = [], []
    shot_train, shot_val = [], []

    for fold in fold_results:
        lh = fold.get("loss_history")
        if lh is None:
            continue
        if lh["receiver"]["train"]:
            recv_train.append(lh["receiver"]["train"])
            recv_val.append(lh["receiver"]["val"])
        if lh["shot"]["train"]:
            shot_train.append(lh["shot"]["train"])
            shot_val.append(lh["shot"]["val"])

    has_receiver = len(recv_train) > 0
    has_shot = len(shot_train) > 0
    n_panels = has_receiver + has_shot

    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No loss history available",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
        return fig

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    for stage_name, train_curves, val_curves, has_data in [
        ("Receiver (Stage 1)", recv_train, recv_val, has_receiver),
        ("Shot (Stage 2)", shot_train, shot_val, has_shot),
    ]:
        if not has_data:
            continue

        ax = axes[panel_idx]
        max_len = max(len(c) for c in train_curves)

        train_arr = _pad_to_length(train_curves, max_len)
        val_arr = _pad_to_length(val_curves, max_len)
        ymin, ymax = _compute_ylim(train_arr, val_arr)
        epochs = np.arange(1, max_len + 1)

        # Individual fold curves (light)
        for i in range(len(train_curves)):
            n = len(train_curves[i])
            ax.plot(epochs[:n], train_arr[i, :n],
                    color="#D6604D", alpha=0.15, linewidth=0.8)
            ax.plot(epochs[:n], val_arr[i, :n],
                    color="#4393C3", alpha=0.15, linewidth=0.8)

        # Mean curves (bold)
        train_mean = np.nanmean(train_arr, axis=0)
        val_mean = np.nanmean(val_arr, axis=0)
        # Only plot mean where at least 2 folds have data
        n_valid = np.sum(~np.isnan(train_arr), axis=0)
        mask = n_valid >= 2

        ax.plot(epochs[mask], train_mean[mask],
                color="#D6604D", linewidth=2.5, label="Train (mean)")
        ax.plot(epochs[mask], val_mean[mask],
                color="#4393C3", linewidth=2.5, label="Val (mean)")

        # Early stopping marker: median epoch where folds stopped
        stop_epochs = [len(c) for c in train_curves]
        median_stop = int(np.median(stop_epochs))
        ax.axvline(x=median_stop, color="gray", linestyle="--",
                   linewidth=1.0, alpha=0.6,
                   label=f"Median stop (ep {median_stop})")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"{model_name} — {stage_name}", fontsize=13,
                     fontweight="bold")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_xlim(1, max_len)
        ax.set_ylim(ymin, ymax)

        # Annotate if any folds were clipped by the y-axis range
        train_maxes = np.nanmax(train_arr, axis=1)
        val_maxes = np.nanmax(val_arr, axis=1)
        n_clipped = int(np.sum((train_maxes > ymax) | (val_maxes > ymax)))
        if n_clipped > 0:
            ax.annotate(
                f"{n_clipped} fold(s) clipped",
                xy=(0.02, 0.98), xycoords="axes fraction",
                fontsize=8, color="gray", va="top",
            )

        panel_idx += 1

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {output_path}")

    if show:
        plt.show()

    return fig


def generate(
    output_dir: Path,
    lomo_results: Optional[dict] = None,
    mlp_results: Optional[dict] = None,
    show: bool = False,
) -> List[Path]:
    """Generate loss curve figures and return list of output paths."""
    paths: List[Path] = []

    if lomo_results is not None:
        folds = lomo_results.get("per_fold", [])
        if any(f.get("loss_history") for f in folds):
            output_path = output_dir / "fig7_loss_curves_gnn.pdf"
            plot_loss_curves(folds, model_name="GNN (Pretrained)",
                             output_path=str(output_path), show=show)
            plt.close("all")
            paths.append(output_path)
        else:
            print("  Skipping GNN loss curves: no loss_history in fold results")
    else:
        print("  Skipping GNN loss curves: no LOMO results provided")

    if mlp_results is not None:
        folds = mlp_results.get("per_fold", [])
        if any(f.get("loss_history") for f in folds):
            output_path = output_dir / "fig7b_loss_curves_mlp.pdf"
            plot_loss_curves(folds, model_name="MLP Baseline",
                             output_path=str(output_path), show=show)
            plt.close("all")
            paths.append(output_path)
        else:
            print("  Skipping MLP loss curves: no loss_history in fold results")
    else:
        print("  Skipping MLP loss curves: no MLP results provided")

    return paths
