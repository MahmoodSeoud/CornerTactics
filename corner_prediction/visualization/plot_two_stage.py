"""Figure 4: Two-stage benefit scatter plot.

Shows whether better receiver prediction correlates with better conditional
shot prediction across LOMO folds.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_two_stage_benefit(
    lomo_results: dict,
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot scatter: receiver Top-3 accuracy vs conditional shot AUC per fold.

    Args:
        lomo_results: LOMO pretrained results with per_fold data.
        output_path: Path to save PDF.
        show: Whether to display interactively.

    Returns:
        matplotlib Figure.
    """
    per_fold = lomo_results["per_fold"]

    # Extract per-fold metrics
    recv_top3 = []
    shot_pred_auc = []
    shot_uncond_auc = []
    fold_labels = []

    for fold in per_fold:
        r = fold["receiver"]
        if r["n_labeled"] == 0:
            continue
        recv_top3.append(r["top3_acc"])
        shot_pred_auc.append(fold["shot_predicted"]["auc"])
        shot_uncond_auc.append(fold["shot_unconditional"]["auc"])
        fold_labels.append(fold.get("held_out_match", str(fold["fold_idx"])))

    recv_top3 = np.array(recv_top3)
    shot_pred_auc = np.array(shot_pred_auc)
    shot_uncond_auc = np.array(shot_uncond_auc)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter: receiver acc vs predicted-receiver shot AUC
    ax.scatter(recv_top3, shot_pred_auc, s=100, c="#2166AC",
               edgecolors="black", linewidth=0.8, zorder=5,
               label="Predicted receiver")

    # Scatter: receiver acc vs unconditional shot AUC
    ax.scatter(recv_top3, shot_uncond_auc, s=80, c="#92C5DE",
               edgecolors="black", linewidth=0.6, zorder=4,
               marker="s", alpha=0.7, label="Unconditional")

    # Fold labels
    for i, label in enumerate(fold_labels):
        # Truncate match ID for readability
        short_label = label[-4:] if len(label) > 4 else label
        ax.annotate(short_label, (recv_top3[i], shot_pred_auc[i]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, alpha=0.7)

    # Trend line for predicted receiver
    if len(recv_top3) > 2:
        z = np.polyfit(recv_top3, shot_pred_auc, 1)
        x_line = np.linspace(0, 1, 100)
        y_line = np.polyval(z, x_line)
        ax.plot(x_line, y_line, "--", color="#2166AC", alpha=0.4, linewidth=1.0)

        # Correlation
        corr = np.corrcoef(recv_top3, shot_pred_auc)[0, 1]
        ax.text(0.02, 0.98, f"r = {corr:.2f}", transform=ax.transAxes,
                fontsize=10, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8))

    # Reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.0,
               alpha=0.5, label="Random AUC = 0.50")
    ax.axvline(x=0.441, color="gray", linestyle=":", linewidth=1.0,
               alpha=0.5, label="Random Top-3 ~0.441")

    ax.set_xlabel("Receiver Top-3 Accuracy (per fold)", fontsize=12)
    ax.set_ylabel("Shot Prediction AUC (per fold)", fontsize=12)
    ax.set_title("Two-Stage Benefit: Receiver Accuracy vs Shot AUC",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)

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
    show: bool = False,
) -> Optional[Path]:
    """Generate Figure 4 and return the output path."""
    if lomo_results is None:
        print("  Skipping Figure 4: no LOMO results provided")
        return None

    output_path = output_dir / "fig4_two_stage_benefit.pdf"
    plot_two_stage_benefit(lomo_results, output_path=str(output_path),
                           show=show)
    plt.close("all")
    return output_path
