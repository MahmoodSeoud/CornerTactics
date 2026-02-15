"""Bonus Figure: Permutation test null distributions.

Histograms of shuffled-label metrics with the real metric marked,
showing whether the model's performance is statistically significant.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_permutation_tests(
    perm_receiver: Optional[dict] = None,
    perm_shot: Optional[dict] = None,
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot permutation test null distributions.

    Args:
        perm_receiver: Permutation test results for receiver prediction.
        perm_shot: Permutation test results for shot prediction.
        output_path: Path to save PDF.
        show: Whether to display interactively.

    Returns:
        matplotlib Figure.
    """
    n_panels = sum(1 for p in [perm_receiver, perm_shot] if p is not None)
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No permutation test results",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    for perm, title_prefix, metric_label in [
        (perm_receiver, "Receiver", "Top-3 Accuracy"),
        (perm_shot, "Shot", "AUC-ROC"),
    ]:
        if perm is None:
            continue

        ax = axes[panel_idx]
        null_dist = np.array(perm["null_distribution"])
        real_val = perm["real_metric"]
        p_value = perm["p_value"]
        significant = perm["significant"]

        # Histogram
        ax.hist(null_dist, bins=20, color="#92C5DE", edgecolor="black",
                linewidth=0.6, alpha=0.8, label="Null distribution")

        # Real metric line
        ax.axvline(x=real_val, color="#D6604D", linewidth=2.5,
                   linestyle="-", label=f"Observed = {real_val:.3f}")

        # Null mean line
        if perm.get("null_mean") is not None:
            ax.axvline(x=perm["null_mean"], color="gray", linewidth=1.2,
                       linestyle="--", alpha=0.7,
                       label=f"Null mean = {perm['null_mean']:.3f}")

        # P-value annotation
        sig_str = "Significant" if significant else "Not significant"
        color = "#D6604D" if significant else "gray"
        ax.text(0.98, 0.95, f"p = {p_value:.3f}\n({sig_str})",
                transform=ax.transAxes, fontsize=11, va="top", ha="right",
                fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor=color))

        ax.set_xlabel(metric_label, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{title_prefix} Prediction: Permutation Test",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)

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
    perm_receiver: Optional[dict] = None,
    perm_shot: Optional[dict] = None,
    show: bool = False,
) -> Optional[Path]:
    """Generate bonus permutation figure and return the output path."""
    if perm_receiver is None and perm_shot is None:
        print("  Skipping permutation figure: no results provided")
        return None

    output_path = output_dir / "fig_bonus_permutation_tests.pdf"
    plot_permutation_tests(perm_receiver, perm_shot,
                            output_path=str(output_path), show=show)
    plt.close("all")
    return output_path
