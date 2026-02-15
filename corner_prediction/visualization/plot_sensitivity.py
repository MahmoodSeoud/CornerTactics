"""Figure 5: Detection rate sensitivity analysis.

Shows how model performance changes when filtering corners by minimum
detection rate threshold.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from corner_prediction.config import DATA_DIR


def _reconstruct_fold_predictions(lomo_results: dict, dataset) -> list:
    """Reconstruct per-corner predictions paired with detection rates.

    Each LOMO fold holds out one match. We reconstruct which corners
    belonged to each fold's test set and pair predictions with metadata.

    Returns:
        List of (detection_rate, shot_prob, shot_label) tuples.
    """
    from corner_prediction.data.dataset import get_match_ids

    match_ids = get_match_ids(dataset)
    all_pairs = []

    for fold in lomo_results["per_fold"]:
        held_out = fold["held_out_match"]

        # Get test graphs for this fold
        test_graphs = [g for g in dataset if str(g.match_id) == held_out]

        probs = fold["shot_oracle"]["probs"]
        labels = fold["shot_oracle"]["labels"]

        if len(test_graphs) != len(probs):
            # Mismatch — skip this fold
            continue

        for g, prob, label in zip(test_graphs, probs, labels):
            all_pairs.append((g.detection_rate, prob, label))

    return all_pairs


def plot_detection_sensitivity(
    lomo_results: dict,
    dataset,
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot performance vs detection rate threshold.

    Args:
        lomo_results: LOMO pretrained results with per_fold data.
        dataset: PyG dataset with detection_rate metadata.
        output_path: Path to save PDF.
        show: Whether to display interactively.

    Returns:
        matplotlib Figure.
    """
    pairs = _reconstruct_fold_predictions(lomo_results, dataset)

    if not pairs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        if output_path:
            fig.savefig(output_path, bbox_inches="tight", dpi=300)
        return fig

    det_rates = np.array([float(p[0]) for p in pairs])
    probs = np.array([float(p[1]) for p in pairs])
    labels = np.array([int(p[2]) for p in pairs])

    # Thresholds to test
    thresholds = np.arange(0.3, 1.01, 0.05)

    aucs = []
    n_samples_list = []
    n_positive_list = []

    for threshold in thresholds:
        mask = det_rates >= threshold
        n = mask.sum()
        n_pos = labels[mask].sum() if n > 0 else 0

        if n >= 5 and n_pos >= 1 and n_pos < n:
            try:
                auc = roc_auc_score(labels[mask], probs[mask])
            except ValueError:
                auc = np.nan
        else:
            auc = np.nan

        aucs.append(auc)
        n_samples_list.append(n)
        n_positive_list.append(int(n_pos))

    aucs = np.array(aucs)
    n_samples_arr = np.array(n_samples_list)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # AUC line
    valid = ~np.isnan(aucs)
    ax1.plot(thresholds[valid], aucs[valid], "o-", color="#2166AC",
             linewidth=2.0, markersize=6, label="Shot AUC")
    ax1.set_xlabel("Minimum Detection Rate Threshold", fontsize=12)
    ax1.set_ylabel("Shot AUC-ROC", fontsize=12, color="#2166AC")
    ax1.tick_params(axis="y", labelcolor="#2166AC")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.0,
                alpha=0.5, label="Random (AUC = 0.50)")

    # Sample count on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(thresholds, n_samples_arr, width=0.04, alpha=0.3,
            color="#D6604D", label="Sample count")
    ax2.set_ylabel("Number of Corners", fontsize=12, color="#D6604D")
    ax2.tick_params(axis="y", labelcolor="#D6604D")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
               loc="upper right", framealpha=0.9)

    ax1.set_title("Detection Rate Sensitivity: Shot Prediction AUC",
                  fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

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
    dataset=None,
    show: bool = False,
) -> Optional[Path]:
    """Generate Figure 5 and return the output path."""
    if lomo_results is None or dataset is None:
        print("  Skipping Figure 5: missing LOMO results or dataset")
        return None

    output_path = output_dir / "fig5_detection_sensitivity.pdf"
    plot_detection_sensitivity(lomo_results, dataset,
                               output_path=str(output_path), show=show)
    plt.close("all")
    return output_path
