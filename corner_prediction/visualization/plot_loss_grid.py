#!/usr/bin/env python3
"""Per-fold loss curve grid with shared y-axis and val-loss overlay.

Generates a publication-ready figure with:
  - One subplot per LOMO fold showing train_loss and val_loss vs epoch
  - Shared y-axis across all subplots for visual comparability
  - Min val_loss marked with a dot on each subplot
  - A final overlay subplot with all val_loss curves (one line per fold)

Usage:
    python -m corner_prediction.visualization.plot_loss_grid
    python -m corner_prediction.visualization.plot_loss_grid --stage receiver
    python -m corner_prediction.visualization.plot_loss_grid --seed 42 --stage shot
"""

import argparse
import math
import pickle
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from corner_prediction.config import RESULTS_DIR

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

TRAIN_COLOR = "#D6604D"
VAL_COLOR = "#4393C3"
MIN_DOT_COLOR = "#2ECC71"


def _set_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path, seed: int = 42) -> dict:
    pattern = f"combined_lomo_ussf_aligned_seed{seed}_*.pkl"
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No results matching {pattern} in {results_dir}")
    path = matches[-1]
    print(f"Loading: {path.name}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _fold_label(fold: dict) -> str:
    match_id = str(fold["held_out_match"])
    source = "DFL" if match_id.startswith("DFL") else "SK"
    if match_id.startswith("DFL"):
        match_id = match_id.replace("DFL-MAT-", "")
    return f"Fold {fold['fold_idx']} [{source}]"


# ---------------------------------------------------------------------------
# Shared y-axis limits
# ---------------------------------------------------------------------------

def _compute_shared_ylim(
    fold_results: List[dict], stage: str
) -> tuple:
    """Compute shared y-axis limits from all folds using 2nd-98th percentile."""
    all_vals = []
    for fold in fold_results:
        lh = fold.get("loss_history")
        if lh is None or not lh[stage]["train"]:
            continue
        all_vals.extend(lh[stage]["train"])
        val = lh[stage]["val"]
        all_vals.extend([v for v in val if not (isinstance(v, float) and math.isnan(v))])

    if not all_vals:
        return (0, 1)

    arr = np.array(all_vals)
    arr = arr[~np.isnan(arr)]
    p2, p98 = np.percentile(arr, [2, 98])
    margin = (p98 - p2) * 0.10
    return (max(0, p2 - margin), p98 + margin)


# ---------------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------------

def plot_loss_grid(
    fold_results: List[dict],
    stage: str = "shot",
    model_name: str = "GNN",
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """Create per-fold loss grid + val overlay figure.

    Returns the matplotlib Figure.
    """
    stage_label = "Shot (Stage 2)" if stage == "shot" else "Receiver (Stage 1)"

    # Filter folds with actual loss data
    valid_folds = [
        f for f in fold_results
        if f.get("loss_history") and f["loss_history"][stage]["train"]
    ]
    n_folds = len(valid_folds)
    if n_folds == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No loss history available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig

    # Grid layout: n_folds subplots + 1 overlay = n_folds + 1 total
    total_panels = n_folds + 1
    ncols = 4
    nrows = math.ceil(total_panels / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        squeeze=False,
        sharey=True,
    )

    ymin, ymax = _compute_shared_ylim(fold_results, stage)

    # Collect val curves for overlay panel
    overlay_data = []  # list of (label, epochs, val_array)

    for idx, fold in enumerate(valid_folds):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        lh = fold["loss_history"]
        train = np.array(lh[stage]["train"])
        val = np.array(lh[stage]["val"])
        epochs = np.arange(1, len(train) + 1)

        has_val = not np.all(np.isnan(val))

        # Train curve
        ax.plot(epochs, train, color=TRAIN_COLOR, linewidth=1.3,
                linestyle="-", label="Train")

        # Val curve
        if has_val:
            ax.plot(epochs, val, color=VAL_COLOR, linewidth=1.3,
                    linestyle="--", label="Val")

            # Mark minimum val loss
            val_clean = np.where(np.isnan(val), np.inf, val)
            min_epoch_idx = int(np.argmin(val_clean))
            min_val = val_clean[min_epoch_idx]
            if np.isfinite(min_val):
                ax.axvline(x=min_epoch_idx + 1, color="gray", linestyle=":",
                           linewidth=0.7, alpha=0.5)
                ax.plot(min_epoch_idx + 1, min_val, "o",
                        color=MIN_DOT_COLOR, markersize=5, zorder=5,
                        markeredgecolor="black", markeredgewidth=0.5)

            # Save for overlay
            overlay_data.append((_fold_label(fold), epochs, val))
        else:
            ax.annotate(
                "No val labels", xy=(0.5, 0.5),
                xycoords="axes fraction", fontsize=8,
                color="gray", ha="center", va="center", fontstyle="italic",
            )

        ax.set_title(_fold_label(fold), fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Loss")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(1, None)

        # Only show legend on first subplot
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # --- Overlay panel: all val curves on one axis ---
    overlay_idx = n_folds
    overlay_row, overlay_col = divmod(overlay_idx, ncols)
    ax_overlay = axes[overlay_row][overlay_col]

    cmap = plt.cm.tab20
    for i, (label, ep, val) in enumerate(overlay_data):
        color = cmap(i / max(len(overlay_data) - 1, 1))
        ax_overlay.plot(ep, val, linewidth=1.0, alpha=0.8, color=color,
                        label=label)

    ax_overlay.set_title("All Folds Val Loss", fontsize=9, fontweight="bold")
    ax_overlay.set_xlabel("Epoch")
    if overlay_col == 0:
        ax_overlay.set_ylabel("Loss")
    ax_overlay.set_ylim(ymin, ymax)
    ax_overlay.set_xlim(1, None)

    # Legend outside if many folds
    if len(overlay_data) > 8:
        ax_overlay.legend(
            fontsize=5.5, loc="upper right", ncol=2,
            framealpha=0.9, handlelength=1.2,
        )
    else:
        ax_overlay.legend(fontsize=6, loc="upper right", framealpha=0.9)

    # Hide unused axes
    for panel_idx in range(overlay_idx + 1, nrows * ncols):
        r, c = divmod(panel_idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"Per-Fold Loss Curves with Shared Y-Axis — {stage_label} ({model_name})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    # Export
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"loss_grid_{stage}"
        for ext in ("pdf", "png"):
            p = output_dir / f"{stem}.{ext}"
            fig.savefig(p, bbox_inches="tight", dpi=300)
            print(f"  Saved: {p}")
    plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-fold loss grid with shared y-axis and val-loss overlay",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=["shot", "receiver"], default="shot")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    _set_style()

    results_dir = Path(args.results_dir)
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "figures" / "loss_curves"

    results = load_results(results_dir, seed=args.seed)
    fold_results = results["per_fold"]

    print(f"Generating loss grid: stage={args.stage}, seed={args.seed}, "
          f"{len(fold_results)} folds")

    plot_loss_grid(
        fold_results,
        stage=args.stage,
        model_name="GNN",
        output_dir=output_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()
