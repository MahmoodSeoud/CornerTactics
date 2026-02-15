"""Figure 3: Shot prediction AUC comparison.

Bar chart comparing shot AUC across receiver modes (unconditional, oracle,
predicted), ablation position-only, and optional baselines.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_shot_auc_comparison(
    lomo_results: dict,
    ablation_results: Optional[dict] = None,
    baseline_results: Optional[Dict[str, dict]] = None,
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot bar chart of shot prediction AUC across conditions.

    Args:
        lomo_results: LOMO pretrained results (has aggregated.shot_*).
        ablation_results: Optional ablation_all results.
        baseline_results: Optional dict of baseline name -> results.
        output_path: Path to save PDF.
        show: Whether to display interactively.

    Returns:
        matplotlib Figure.
    """
    agg = lomo_results["aggregated"]

    # Build condition list: (label, mean, std, color)
    conditions = []

    # GNN conditions
    gnn_color = "#4393C3"
    conditions.append((
        "Unconditional",
        agg["shot_unconditional"]["auc_mean"],
        agg["shot_unconditional"]["auc_std"],
        gnn_color,
    ))
    conditions.append((
        "Oracle\nReceiver",
        agg["shot_oracle"]["auc_mean"],
        agg["shot_oracle"]["auc_std"],
        "#2166AC",
    ))
    conditions.append((
        "Predicted\nReceiver",
        agg["shot_predicted"]["auc_mean"],
        agg["shot_predicted"]["auc_std"],
        "#053061",
    ))

    # Ablation: position only
    if ablation_results and "position_only" in ablation_results:
        pos_agg = ablation_results["position_only"]["aggregated"]
        conditions.append((
            "Position\nOnly",
            pos_agg["shot_oracle"]["auc_mean"],
            pos_agg["shot_oracle"]["auc_std"],
            "#92C5DE",
        ))

    # Baselines
    baseline_color = "#D6604D"
    if baseline_results:
        for name, res in baseline_results.items():
            bagg = res["aggregated"]
            shot_key = "shot_oracle"
            if shot_key not in bagg:
                # Some baselines may store shot metrics differently
                for k in ("shot_oracle", "shot_unconditional", "shot"):
                    if k in bagg:
                        shot_key = k
                        break
                else:
                    continue

            conditions.append((
                name.replace("_", "\n").title(),
                bagg[shot_key]["auc_mean"],
                bagg[shot_key]["auc_std"],
                baseline_color,
            ))

    # Plot
    labels = [c[0] for c in conditions]
    means = [c[1] for c in conditions]
    stds = [c[2] for c in conditions]
    colors = [c[3] for c in conditions]

    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(x, means, yerr=stds, color=colors,
                  edgecolor="black", linewidth=0.6,
                  capsize=4, error_kw={"linewidth": 1.0})

    # Random baseline line
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.2,
               alpha=0.7, label="Random (AUC = 0.50)")

    # Value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("Shot Prediction: AUC Comparison", fontsize=13,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
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
    lomo_results: Optional[dict] = None,
    ablation_results: Optional[dict] = None,
    baseline_results: Optional[Dict[str, dict]] = None,
    show: bool = False,
) -> Optional[Path]:
    """Generate Figure 3 and return the output path."""
    if lomo_results is None:
        print("  Skipping Figure 3: no LOMO results provided")
        return None

    output_path = output_dir / "fig3_shot_auc_comparison.pdf"
    plot_shot_auc_comparison(
        lomo_results,
        ablation_results=ablation_results,
        baseline_results=baseline_results,
        output_path=str(output_path),
        show=show,
    )
    plt.close("all")
    return output_path
