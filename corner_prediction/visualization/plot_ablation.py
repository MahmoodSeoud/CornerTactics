"""Figure 2: Ablation comparison bar chart.

Grouped bar chart showing receiver Top-1 and Top-3 accuracy
across all ablation configurations, with random baseline reference lines.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Readable labels for ablation configs
ABLATION_LABELS = {
    "position_only": "Position\nOnly",
    "plus_velocity": "+ Velocity",
    "plus_detection": "+ Detection",
    "full_features": "Full\nFeatures",
    "full_fc_edges": "Full + FC\nEdges",
}

# Ordered display sequence
ABLATION_ORDER = [
    "position_only", "plus_velocity", "plus_detection",
    "full_features", "full_fc_edges",
]


def plot_ablation_comparison(
    ablation_results: dict,
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot grouped bar chart of ablation receiver accuracy.

    Args:
        ablation_results: Dict from ablation_all pickle.
            Keys are ablation names, values have 'aggregated.receiver'.
        output_path: Path to save PDF.
        show: Whether to display interactively.

    Returns:
        matplotlib Figure.
    """
    configs = [c for c in ABLATION_ORDER if c in ablation_results]
    labels = [ABLATION_LABELS.get(c, c) for c in configs]

    top1_means = []
    top1_stds = []
    top3_means = []
    top3_stds = []

    for c in configs:
        r = ablation_results[c]["aggregated"]["receiver"]
        top1_means.append(r["top1_mean"])
        top1_stds.append(r["top1_std"])
        top3_means.append(r["top3_mean"])
        top3_stds.append(r["top3_std"])

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width / 2, top1_means, width, yerr=top1_stds,
                   label="Top-1 Accuracy", color="#D6604D",
                   edgecolor="black", linewidth=0.6,
                   capsize=3, error_kw={"linewidth": 1.0})
    bars2 = ax.bar(x + width / 2, top3_means, width, yerr=top3_stds,
                   label="Top-3 Accuracy", color="#4393C3",
                   edgecolor="black", linewidth=0.6,
                   capsize=3, error_kw={"linewidth": 1.0})

    # Random baselines
    ax.axhline(y=0.147, color="#D6604D", linestyle="--", linewidth=1.0,
               alpha=0.6, label="Random Top-1 (~0.147)")
    ax.axhline(y=0.441, color="#4393C3", linestyle="--", linewidth=1.0,
               alpha=0.6, label="Random Top-3 (~0.441)")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Feature Configuration", fontsize=12)
    ax.set_title("Receiver Prediction: Ablation Comparison", fontsize=13,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {output_path}")

    if show:
        plt.show()

    return fig


def generate(
    output_dir: Path,
    ablation_results: Optional[dict] = None,
    show: bool = False,
) -> Optional[Path]:
    """Generate Figure 2 and return the output path."""
    if ablation_results is None:
        print("  Skipping Figure 2: no ablation results provided")
        return None

    output_path = output_dir / "fig2_ablation_comparison.pdf"
    plot_ablation_comparison(ablation_results, output_path=str(output_path),
                             show=show)
    plt.close("all")
    return output_path
