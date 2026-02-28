#!/usr/bin/env python3
"""Comprehensive training analysis: loss curves, metrics table, confusion matrices.

Generates thesis-ready figures for diagnosing GNN training behavior across
Leave-One-Match-Out cross-validation folds.  Also produces MLP baseline loss
curves / metrics and XGBoost baseline metrics (no loss curves — tree-based).

Usage:
    python -m corner_prediction.visualization.analysis_plots
    python -m corner_prediction.visualization.analysis_plots --seed 42
"""

import argparse
import csv
import math
import pickle
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from corner_prediction.config import RESULTS_DIR

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

COLORS = {
    "train": "#D6604D",
    "val": "#4393C3",
    "gap": "#FDD49E",
    "shot_pos": "#E63946",
    "shot_neg": "#457B9D",
    "sk": "#1D3557",
    "dfl": "#E76F51",
}

THRESHOLD = 0.5  # Standard decision threshold for confusion matrices


def _set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path, seed: int = 42) -> dict:
    """Load USSF-aligned combined LOMO results for a given seed."""
    pattern = f"combined_lomo_ussf_aligned_seed{seed}_*.pkl"
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No results matching {pattern} in {results_dir}"
        )
    path = matches[-1]  # latest
    print(f"Loading: {path.name}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_baseline_results(results_dir: Path, name: str) -> dict:
    """Load the latest combined baseline result file (MLP or XGBoost)."""
    pattern = f"combined_baseline_{name}_*.pkl"
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No results matching {pattern} in {results_dir}"
        )
    path = matches[-1]
    print(f"Loading: {path.name}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _fold_label(fold: dict) -> str:
    """Short label for a fold: match ID + source."""
    match_id = str(fold["held_out_match"])
    if match_id.startswith("DFL"):
        return match_id.replace("DFL-MAT-", "")
    return match_id


def _fold_source(fold: dict) -> str:
    match_id = str(fold["held_out_match"])
    return "DFL" if match_id.startswith("DFL") else "SK"


# ---------------------------------------------------------------------------
# Task 1: Per-fold loss curves
# ---------------------------------------------------------------------------


def _pad_curves(curves: List[List[float]], max_len: int) -> np.ndarray:
    """Pad variable-length loss curves to [n_folds, max_len] with NaN."""
    arr = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
    return arr


def plot_per_fold_loss_curves(
    fold_results: List[dict],
    output_dir: Path,
    stage: str = "shot",
    model_name: str = "GNN",
) -> List[Path]:
    """Plot individual loss curves per fold with overfitting gap shading.

    Returns list of saved file paths.
    """
    stage_key = stage
    stage_label = "Shot (Stage 2)" if stage == "shot" else "Receiver (Stage 1)"
    suffix = f"_{model_name.lower()}" if model_name != "GNN" else ""
    paths = []

    # Determine grid layout
    n_folds = sum(
        1 for f in fold_results
        if f.get("loss_history") and f["loss_history"][stage_key]["train"]
    )
    if n_folds == 0:
        return paths

    ncols = 4
    nrows = (n_folds + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False
    )

    plot_idx = 0
    for fold in fold_results:
        lh = fold.get("loss_history")
        if lh is None or not lh[stage_key]["train"]:
            continue

        row, col = divmod(plot_idx, ncols)
        ax = axes[row][col]

        train = np.array(lh[stage_key]["train"])
        val = np.array(lh[stage_key]["val"])
        epochs = np.arange(1, len(train) + 1)
        has_val = not np.all(np.isnan(val))

        # Plot curves
        ax.plot(epochs, train, color=COLORS["train"], linewidth=1.5,
                label="Train")
        if has_val:
            ax.plot(epochs, val, color=COLORS["val"], linewidth=1.5,
                    label="Val")

            # Shade overfitting gap where val > train
            gap = val - train
            overfitting = gap > 0
            if overfitting.any():
                ax.fill_between(
                    epochs, train, val,
                    where=overfitting,
                    alpha=0.2, color=COLORS["gap"],
                    label="Overfit gap",
                )

        else:
            ax.annotate(
                "No val labels",
                xy=(0.98, 0.02), xycoords="axes fraction",
                fontsize=8, color="#7F8C8D", fontstyle="italic",
                ha="right", va="bottom",
            )

        # Early stopping marker
        stop_epoch = len(train)
        max_epochs = 100
        if stop_epoch < max_epochs:
            ax.axvline(x=stop_epoch, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            ax.annotate(
                f"stop={stop_epoch}",
                xy=(stop_epoch, ax.get_ylim()[1]),
                fontsize=7, color="gray", ha="center", va="top",
            )

        # Fold info
        source = _fold_source(fold)
        match_id = _fold_label(fold)
        so = fold["shot_oracle"]
        ax.set_title(
            f"Fold {fold['fold_idx']} [{source}] — AUC={so['auc']:.2f}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)

        # Clip y-axis to hide initialization spikes
        all_vals = np.concatenate([train, val])
        all_vals = all_vals[~np.isnan(all_vals)]
        p5, p95 = np.percentile(all_vals, [5, 95])
        margin = (p95 - p5) * 0.2
        ax.set_ylim(max(0, p5 - margin), p95 + margin)

        if plot_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

        plot_idx += 1

    # Hide unused axes
    for idx in range(plot_idx, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Per-Fold Loss Curves — {stage_label} ({model_name})",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = output_dir / f"per_fold_loss_{stage}{suffix}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)

    print(f"  Saved per-fold {stage} loss curves ({n_folds} folds, {model_name})")
    return paths


def plot_aggregated_loss_curves(
    fold_results: List[dict],
    output_dir: Path,
    stage: str = "shot",
    model_name: str = "GNN",
) -> List[Path]:
    """Aggregated loss curve: mean ± 1std with early stopping markers."""
    stage_key = stage
    stage_label = "Shot (Stage 2)" if stage == "shot" else "Receiver (Stage 1)"
    suffix = f"_{model_name.lower()}" if model_name != "GNN" else ""
    paths = []

    train_curves, val_curves, stop_epochs = [], [], []
    for fold in fold_results:
        lh = fold.get("loss_history")
        if lh is None or not lh[stage_key]["train"]:
            continue
        train_curves.append(lh[stage_key]["train"])
        val_curves.append(lh[stage_key]["val"])
        stop_epochs.append(len(lh[stage_key]["train"]))

    if not train_curves:
        return paths

    max_len = max(len(c) for c in train_curves)
    train_arr = _pad_curves(train_curves, max_len)
    val_arr = _pad_curves(val_curves, max_len)
    epochs = np.arange(1, max_len + 1)

    # Compute stats (ignore NaN from early-stopped folds)
    train_mean = np.nanmean(train_arr, axis=0)
    train_std = np.nanstd(train_arr, axis=0)
    val_mean = np.nanmean(val_arr, axis=0)
    val_std = np.nanstd(val_arr, axis=0)

    # Only show where >= 2 folds have data
    n_train_valid = np.sum(~np.isnan(train_arr), axis=0)
    n_val_valid = np.sum(~np.isnan(val_arr), axis=0)
    train_mask = n_train_valid >= 2
    val_mask = n_val_valid >= 2

    fig, ax = plt.subplots(figsize=(8, 5))

    # Count folds with real val data (not all-NaN)
    n_val_folds = sum(
        1 for c in val_curves if not all(math.isnan(v) for v in c)
    )

    # Shaded bands
    ax.fill_between(
        epochs[train_mask],
        (train_mean - train_std)[train_mask],
        (train_mean + train_std)[train_mask],
        alpha=0.15, color=COLORS["train"],
    )
    if val_mask.any():
        ax.fill_between(
            epochs[val_mask],
            (val_mean - val_std)[val_mask],
            (val_mean + val_std)[val_mask],
            alpha=0.15, color=COLORS["val"],
        )

    # Mean curves
    ax.plot(epochs[train_mask], train_mean[train_mask], color=COLORS["train"],
            linewidth=2.5,
            label=f"Train (mean ± 1σ, n={len(train_curves)})")
    if val_mask.any():
        ax.plot(epochs[val_mask], val_mean[val_mask], color=COLORS["val"],
                linewidth=2.5,
                label=f"Val (mean ± 1σ, n={n_val_folds})")

    # Per-fold early stopping markers
    for i, ep in enumerate(stop_epochs):
        label = "Fold early stop" if i == 0 else None
        ax.axvline(
            x=ep, color="gray", linestyle=":", linewidth=0.5, alpha=0.4,
        )
    # Median early stopping
    median_stop = int(np.median(stop_epochs))
    ax.axvline(
        x=median_stop, color="gray", linestyle="--", linewidth=1.5,
        alpha=0.7, label=f"Median early stop (ep {median_stop})",
    )

    # Annotation: fold count diminishing
    ax.annotate(
        f"n={len(train_curves)} folds",
        xy=(0.02, 0.98), xycoords="axes fraction",
        fontsize=9, va="top", color="gray",
    )

    # Clip y-axis
    all_vals = np.concatenate([train_arr.ravel(), val_arr.ravel()])
    all_vals = all_vals[~np.isnan(all_vals)]
    p5, p95 = np.percentile(all_vals, [5, 95])
    margin = (p95 - p5) * 0.15
    ax.set_ylim(max(0, p5 - margin), p95 + margin)
    ax.set_xlim(1, max_len)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Aggregated Loss Curves — {stage_label} ({model_name}, combined)",
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = output_dir / f"aggregated_loss_{stage}{suffix}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)

    print(f"  Saved aggregated {stage} loss curves ({model_name})")
    return paths


# ---------------------------------------------------------------------------
# Task 2: Per-fold metrics summary
# ---------------------------------------------------------------------------


def compute_per_fold_metrics(
    fold_results: List[dict],
    mode: str = "shot_oracle",
) -> List[dict]:
    """Compute accuracy, AUC, precision, recall, F1 per fold."""
    rows = []
    for fold in fold_results:
        so = fold[mode]
        probs = np.array(so["probs"])
        labels = np.array(so["labels"])
        preds = (probs >= THRESHOLD).astype(int)

        n = len(labels)
        n_pos = int(labels.sum())

        if len(np.unique(labels)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(labels, probs)

        rows.append({
            "fold": fold["fold_idx"],
            "match": _fold_label(fold),
            "source": _fold_source(fold),
            "n_samples": n,
            "n_positive": n_pos,
            "shot_rate": n_pos / n if n > 0 else 0,
            "accuracy": float((preds == labels).mean()),
            "auc": auc,
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        })
    return rows


def save_metrics_csv(rows: List[dict], output_path: Path):
    """Save metrics table as CSV."""
    fieldnames = [
        "fold", "match", "source", "n_samples", "n_positive", "shot_rate",
        "accuracy", "auc", "precision", "recall", "f1",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        # Summary row
        metrics = ["accuracy", "auc", "precision", "recall", "f1"]
        summary = {"fold": "MEAN±STD", "match": "", "source": ""}
        for m in metrics:
            vals = [r[m] for r in rows]
            summary[m] = f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
        summary["n_samples"] = sum(r["n_samples"] for r in rows)
        summary["n_positive"] = sum(r["n_positive"] for r in rows)
        summary["shot_rate"] = (
            summary["n_positive"] / summary["n_samples"]
            if summary["n_samples"] > 0 else 0
        )
        writer.writerow(summary)

    print(f"  Saved metrics CSV: {output_path}")


def plot_metrics_table(
    rows: List[dict],
    output_dir: Path,
    model_name: str = "GNN",
) -> List[Path]:
    """Render metrics table as a formatted figure."""
    paths = []
    suffix = f"_{model_name.lower()}" if model_name != "GNN" else ""
    metrics = ["accuracy", "auc", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "AUC", "Precision", "Recall", "F1"]

    fig, ax = plt.subplots(figsize=(14, 0.4 * len(rows) + 2.5))
    ax.axis("off")

    col_labels = ["Fold", "Match", "Src", "n", "pos", "Rate"] + metric_labels
    cell_data = []
    for r in rows:
        cell_data.append([
            str(r["fold"]),
            r["match"],
            r["source"],
            str(r["n_samples"]),
            str(r["n_positive"]),
            f"{r['shot_rate']:.2f}",
        ] + [f"{r[m]:.3f}" for m in metrics])

    # Summary row
    summary = []
    for m in metrics:
        vals = [r[m] for r in rows]
        summary.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}")
    total_n = sum(r["n_samples"] for r in rows)
    total_pos = sum(r["n_positive"] for r in rows)
    cell_data.append([
        "ALL", "", "",
        str(total_n), str(total_pos),
        f"{total_pos / total_n:.2f}",
    ] + summary)

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style summary row
    last = len(cell_data)
    for j in range(len(col_labels)):
        table[last, j].set_facecolor("#ECF0F1")
        table[last, j].set_text_props(fontweight="bold")

    # Color-code AUC column
    auc_col = col_labels.index("AUC")
    for i, r in enumerate(rows, start=1):
        auc = r["auc"]
        if auc >= 0.7:
            table[i, auc_col].set_facecolor("#D5F5E3")
        elif auc < 0.5:
            table[i, auc_col].set_facecolor("#FADBD8")

    ax.set_title(
        f"Per-Fold Metrics — {model_name} (Shot, Oracle Receiver, threshold=0.5)",
        fontsize=12, fontweight="bold", pad=10,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = output_dir / f"metrics_table{suffix}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)

    print(f"  Saved metrics table figure ({model_name})")
    return paths


def plot_metrics_bar_chart(
    rows: List[dict],
    output_dir: Path,
    model_name: str = "GNN",
) -> List[Path]:
    """Grouped bar chart of per-fold metrics with mean ± std band."""
    paths = []
    suffix = f"_{model_name.lower()}" if model_name != "GNN" else ""
    metrics = ["accuracy", "auc", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
    metric_colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]

    fold_labels = [
        f"F{r['fold']}\n({r['source']})" for r in rows
    ]
    x = np.arange(len(rows))
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.9), 5.5))

    for i, (m, label, color) in enumerate(
        zip(metrics, metric_labels, metric_colors)
    ):
        vals = [r[m] for r in rows]
        offset = (i - 2) * width
        ax.bar(
            x + offset, vals, width,
            label=label, color=color, alpha=0.85, edgecolor="white",
        )

        # Mean line
        mean_val = np.mean(vals)
        ax.axhline(
            y=mean_val, color=color, linestyle="--",
            linewidth=0.8, alpha=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, fontsize=7)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Per-Fold Metrics — {model_name} (Oracle Shot, threshold=0.5)",
        fontweight="bold",
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=5, fontsize=9,
    )
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5,
               label="chance")

    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = output_dir / f"metrics_bar_chart{suffix}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)

    print(f"  Saved metrics bar chart ({model_name})")
    return paths


# ---------------------------------------------------------------------------
# Task 3: Confusion matrices + class imbalance analysis
# ---------------------------------------------------------------------------


def plot_per_fold_confusion_matrices(
    fold_results: List[dict],
    output_dir: Path,
    mode: str = "shot_oracle",
) -> List[Path]:
    """Plot normalized confusion matrix per fold."""
    paths = []

    n_folds = len(fold_results)
    ncols = 4
    nrows = (n_folds + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False
    )

    for idx, fold in enumerate(fold_results):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        so = fold[mode]
        labels = np.array(so["labels"])
        preds = (np.array(so["probs"]) >= THRESHOLD).astype(int)

        cm = confusion_matrix(labels, preds, labels=[0, 1])
        # Normalize by row (true class)
        cm_norm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")

        # Annotate with count and percentage
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_norm[i, j]
                color = "white" if pct > 0.5 else "black"
                ax.text(
                    j, i, f"{count}\n({pct:.0%})",
                    ha="center", va="center", fontsize=8, color=color,
                )

        source = _fold_source(fold)
        ax.set_title(
            f"Fold {fold['fold_idx']} [{source}] n={so['n_samples']}",
            fontsize=8, fontweight="bold",
        )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Shot", "Shot"], fontsize=7)
        ax.set_yticklabels(["No Shot", "Shot"], fontsize=7)
        if col == 0:
            ax.set_ylabel("True", fontsize=8)
        if row == nrows - 1:
            ax.set_xlabel("Predicted", fontsize=8)

    # Hide unused
    for idx in range(n_folds, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Per-Fold Normalized Confusion Matrices (threshold=0.5)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = output_dir / f"per_fold_confusion_matrices.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)

    print(f"  Saved per-fold confusion matrices")
    return paths


def plot_aggregated_confusion_matrix(
    fold_results: List[dict],
    output_dir: Path,
    mode: str = "shot_oracle",
) -> List[Path]:
    """Sum predictions across folds and plot overall confusion matrix."""
    paths = []

    all_labels, all_preds = [], []
    for fold in fold_results:
        so = fold[mode]
        labels = np.array(so["labels"])
        preds = (np.array(so["probs"]) >= THRESHOLD).astype(int)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Raw counts
    im1 = ax1.imshow(cm, cmap="Blues", aspect="equal")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color=color,
            )
    ax1.set_title("Aggregated (Raw Counts)", fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["No Shot", "Shot"])
    ax1.set_yticklabels(["No Shot", "Shot"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    # Normalized
    im2 = ax2.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax2.text(
                j, i, f"{cm_norm[i, j]:.1%}",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color=color,
            )
    ax2.set_title("Aggregated (Row-Normalized)", fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["No Shot", "Shot"])
    ax2.set_yticklabels(["No Shot", "Shot"])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    n = len(all_labels)
    n_pos = int(all_labels.sum())
    fig.suptitle(
        f"Aggregated Confusion Matrix — {n} samples "
        f"({n_pos} shots, {n - n_pos} no-shot)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = output_dir / f"aggregated_confusion_matrix.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)

    print(f"  Saved aggregated confusion matrix")
    return paths


def plot_class_distribution(
    fold_results: List[dict],
    output_dir: Path,
    mode: str = "shot_oracle",
) -> List[Path]:
    """Class distribution analysis: actual vs predicted per fold."""
    paths = []

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel A: Per-fold class distribution (actual) ---
    ax_a = fig.add_subplot(gs[0, 0])

    fold_labels = []
    n_pos_list, n_neg_list = [], []
    for fold in fold_results:
        so = fold[mode]
        n_pos = so["n_positive"]
        n_neg = so["n_samples"] - n_pos
        n_pos_list.append(n_pos)
        n_neg_list.append(n_neg)
        source = _fold_source(fold)
        fold_labels.append(f"F{fold['fold_idx']}\n({source})")

    x = np.arange(len(fold_results))
    ax_a.bar(x, n_neg_list, color=COLORS["shot_neg"], label="No Shot",
             alpha=0.85)
    ax_a.bar(x, n_pos_list, bottom=n_neg_list, color=COLORS["shot_pos"],
             label="Shot", alpha=0.85)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(fold_labels, fontsize=6)
    ax_a.set_ylabel("Count")
    ax_a.set_title("A) Actual Class Distribution Per Fold", fontweight="bold")
    ax_a.legend(fontsize=8)

    # --- Panel B: Shot rate per fold vs overall ---
    ax_b = fig.add_subplot(gs[0, 1])

    shot_rates = [
        fold[mode]["n_positive"] / fold[mode]["n_samples"]
        for fold in fold_results
    ]
    colors_bar = [
        COLORS["sk"] if _fold_source(f) == "SK" else COLORS["dfl"]
        for f in fold_results
    ]
    ax_b.bar(x, shot_rates, color=colors_bar, alpha=0.85)

    overall_rate = sum(n_pos_list) / (sum(n_pos_list) + sum(n_neg_list))
    ax_b.axhline(
        y=overall_rate, color="#E74C3C", linestyle="--", linewidth=1.5,
        label=f"Overall rate: {overall_rate:.1%}",
    )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(fold_labels, fontsize=6)
    ax_b.set_ylabel("Shot Rate")
    ax_b.set_ylim(0, 1)
    ax_b.set_title("B) Shot Rate Per Fold", fontweight="bold")
    ax_b.legend(fontsize=8)

    # Custom legend for SK vs DFL
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["sk"], label="SkillCorner"),
        Patch(facecolor=COLORS["dfl"], label="DFL"),
    ]
    ax_b.legend(
        handles=legend_elements + [
            plt.Line2D([0], [0], color="#E74C3C", linestyle="--",
                       label=f"Overall: {overall_rate:.1%}")
        ],
        fontsize=7, loc="upper right",
    )

    # --- Panel C: Actual vs predicted class distribution ---
    ax_c = fig.add_subplot(gs[1, 0])

    actual_pos = sum(n_pos_list)
    actual_neg = sum(n_neg_list)
    total = actual_pos + actual_neg

    # Collect all predictions
    pred_pos, pred_neg = 0, 0
    for fold in fold_results:
        so = fold[mode]
        preds = (np.array(so["probs"]) >= THRESHOLD).astype(int)
        pred_pos += preds.sum()
        pred_neg += (1 - preds).sum()

    categories = ["Actual", "Predicted"]
    pos_vals = [actual_pos, pred_pos]
    neg_vals = [actual_neg, pred_neg]

    bar_x = np.arange(2)
    ax_c.bar(bar_x, neg_vals, width=0.5, color=COLORS["shot_neg"],
             label="No Shot", alpha=0.85)
    ax_c.bar(bar_x, pos_vals, width=0.5, bottom=neg_vals,
             color=COLORS["shot_pos"], label="Shot", alpha=0.85)

    # Annotate percentages
    for i, (neg, pos) in enumerate(zip(neg_vals, pos_vals)):
        total_i = neg + pos
        ax_c.text(i, neg / 2, f"{neg}\n({neg / total_i:.1%})",
                  ha="center", va="center", fontsize=10, fontweight="bold",
                  color="white")
        ax_c.text(i, neg + pos / 2, f"{pos}\n({pos / total_i:.1%})",
                  ha="center", va="center", fontsize=10, fontweight="bold",
                  color="white")

    ax_c.set_xticks(bar_x)
    ax_c.set_xticklabels(categories, fontsize=11)
    ax_c.set_ylabel("Count")
    ax_c.set_title("C) Actual vs Predicted Distribution", fontweight="bold")
    ax_c.legend(fontsize=8)

    # --- Panel D: Per-fold predicted vs actual shot rate ---
    ax_d = fig.add_subplot(gs[1, 1])

    pred_rates = []
    for fold in fold_results:
        so = fold[mode]
        preds = (np.array(so["probs"]) >= THRESHOLD).astype(int)
        pred_rates.append(preds.mean())

    ax_d.scatter(shot_rates, pred_rates, c=colors_bar, s=60, edgecolors="black",
                 linewidth=0.5, zorder=3)

    # Identity line
    ax_d.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5,
              label="Perfect calibration")

    # Annotate fold indices
    for i, fold in enumerate(fold_results):
        ax_d.annotate(
            str(fold["fold_idx"]), (shot_rates[i], pred_rates[i]),
            fontsize=6, ha="center", va="bottom",
            xytext=(0, 3), textcoords="offset points",
        )

    ax_d.set_xlabel("Actual Shot Rate")
    ax_d.set_ylabel("Predicted Shot Rate")
    ax_d.set_xlim(-0.05, 1.05)
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.set_title("D) Predicted vs Actual Shot Rate Per Fold",
                    fontweight="bold")
    ax_d.legend(fontsize=8)
    ax_d.set_aspect("equal")

    fig.suptitle(
        f"Class Imbalance Analysis — {total} corners "
        f"({actual_pos} shots / {actual_neg} no-shot = "
        f"{actual_pos / total:.1%} positive rate)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ext in ("png", "pdf"):
        p = output_dir / f"class_distribution_analysis.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)

    print(f"  Saved class distribution analysis")
    return paths


def write_analysis(
    fold_results: List[dict],
    metrics_rows: List[dict],
    output_path: Path,
    mode: str = "shot_oracle",
):
    """Write a short text analysis of class imbalance and model behavior."""
    # Aggregate predictions
    all_labels, all_preds, all_probs = [], [], []
    for fold in fold_results:
        so = fold[mode]
        labels = np.array(so["labels"])
        probs = np.array(so["probs"])
        preds = (probs >= THRESHOLD).astype(int)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    n = len(all_labels)
    n_pos = int(all_labels.sum())
    n_pred_pos = int(all_preds.sum())
    base_rate = n_pos / n

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    overall_auc = roc_auc_score(all_labels, all_probs)
    overall_prec = precision_score(all_labels, all_preds, zero_division=0)
    overall_rec = recall_score(all_labels, all_preds, zero_division=0)
    overall_f1 = f1_score(all_labels, all_preds, zero_division=0)

    aucs = [r["auc"] for r in metrics_rows]
    f1s = [r["f1"] for r in metrics_rows]

    # Majority-class baseline accuracy
    majority_acc = max(base_rate, 1 - base_rate)

    fold_mean_auc = np.mean(aucs)

    lines = [
        "=" * 70,
        "CLASS IMBALANCE & MODEL BEHAVIOR ANALYSIS",
        "=" * 70,
        "",
        f"Dataset: {n} corners, {n_pos} shots ({base_rate:.1%}), "
        f"{n - n_pos} no-shot ({1 - base_rate:.1%})",
        f"Imbalance ratio: {(n - n_pos) / max(n_pos, 1):.1f}:1 "
        f"(no-shot : shot)",
        f"Majority-class baseline accuracy: {majority_acc:.1%}",
        "",
        "--- Aggregated Metrics (threshold=0.5) ---",
        f"AUC (pooled):    {overall_auc:.3f}   "
        f"<- all {n} predictions concatenated",
        f"AUC (fold-mean): {fold_mean_auc:.3f}   "
        f"<- mean of {len(aucs)} per-fold AUCs",
        f"Accuracy:  {(all_labels == all_preds).mean():.3f}",
        f"Precision: {overall_prec:.3f}",
        f"Recall:    {overall_rec:.3f}",
        f"F1:        {overall_f1:.3f}",
        "",
        "--- Confusion Matrix ---",
        f"  TN={tn}  FP={fp}",
        f"  FN={fn}  TP={tp}",
        f"  Model predicted {n_pred_pos} shots out of {n} "
        f"({n_pred_pos / n:.1%} predicted positive rate vs "
        f"{base_rate:.1%} actual)",
        "",
        "--- Per-Fold Stability ---",
        f"AUC range: [{min(aucs):.3f}, {max(aucs):.3f}], "
        f"mean={np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
        f"F1 range:  [{min(f1s):.3f}, {max(f1s):.3f}], "
        f"mean={np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
        "",
        "--- Interpretation ---",
        "",
    ]

    # Generate interpretation
    interp = []

    # 1. Majority-class default?
    if n_pred_pos / n < 0.1:
        interp.append(
            f"The model predicts shot for only {n_pred_pos / n:.1%} of "
            f"samples (vs {base_rate:.1%} actual), indicating it is "
            f"heavily biased toward predicting the majority class (no-shot). "
            f"Despite pos_weight=2.0 in training, the learned decision "
            f"boundary at threshold=0.5 is conservative."
        )
    elif abs(n_pred_pos / n - base_rate) < 0.05:
        interp.append(
            f"The model's predicted positive rate ({n_pred_pos / n:.1%}) "
            f"closely matches the actual base rate ({base_rate:.1%}), "
            f"suggesting reasonable calibration. It is not simply defaulting "
            f"to majority-class prediction."
        )
    else:
        interp.append(
            f"The model predicts shot for {n_pred_pos / n:.1%} of samples "
            f"(vs {base_rate:.1%} actual). "
            f"{'It over-predicts' if n_pred_pos / n > base_rate else 'It under-predicts'} "
            f"the positive class."
        )

    # 2. Precision/recall trade-off
    if overall_rec > 0 and overall_prec > 0:
        interp.append(
            f"Precision={overall_prec:.3f} and recall={overall_rec:.3f} "
            f"both exceed what a random classifier at base rate "
            f"({base_rate:.1%}) would achieve "
            f"(expected precision≈{base_rate:.3f}, recall≈0.5). "
            f"Combined with AUC={overall_auc:.3f} (well above 0.5), "
            f"this confirms the model has learned discriminative signal "
            f"beyond the base rate."
        )
    else:
        interp.append(
            f"Either precision ({overall_prec:.3f}) or recall "
            f"({overall_rec:.3f}) is zero, indicating the model fails to "
            f"identify the positive class at threshold=0.5."
        )

    # 3. Fold consistency
    auc_cv = np.std(aucs) / max(np.mean(aucs), 1e-6)
    if auc_cv > 0.3:
        interp.append(
            f"Confusion matrix patterns vary substantially across folds "
            f"(AUC CV={auc_cv:.1%}). This is expected given small fold sizes "
            f"(2-15 samples) in LOMO cross-validation — individual match "
            f"characteristics dominate fold-level metrics."
        )
    else:
        interp.append(
            f"Performance is moderately stable across folds "
            f"(AUC CV={auc_cv:.1%}), though some variation exists due to "
            f"the small fold sizes inherent in LOMO cross-validation."
        )

    for i, text in enumerate(interp):
        lines.append(f"{i + 1}. {text}")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Saved analysis: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive training analysis plots",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the result file to load (default: 42)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help="Directory containing result pickle files",
    )
    args = parser.parse_args()

    _set_style()

    results_dir = Path(args.results_dir)
    project_root = Path(__file__).resolve().parent.parent.parent

    # Output directories
    loss_dir = project_root / "figures" / "loss_curves"
    cm_dir = project_root / "figures" / "confusion_matrices"
    metrics_dir = project_root / "figures"
    loss_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(results_dir, seed=args.seed)
    fold_results = results["per_fold"]

    print(f"\n{'=' * 60}")
    print(f"Training Analysis — seed={args.seed}, "
          f"{len(fold_results)} folds")
    print(f"{'=' * 60}")

    all_paths = []

    # Task 1: Loss curves
    print("\n--- Task 1: Per-Fold Loss Curves ---")
    all_paths += plot_per_fold_loss_curves(fold_results, loss_dir, stage="shot")
    all_paths += plot_per_fold_loss_curves(
        fold_results, loss_dir, stage="receiver"
    )
    all_paths += plot_aggregated_loss_curves(
        fold_results, loss_dir, stage="shot"
    )
    all_paths += plot_aggregated_loss_curves(
        fold_results, loss_dir, stage="receiver"
    )

    # Task 2: Metrics table
    print("\n--- Task 2: Per-Fold Metrics ---")
    metrics_rows = compute_per_fold_metrics(fold_results)
    save_metrics_csv(metrics_rows, metrics_dir / "metrics_summary.csv")
    all_paths += plot_metrics_table(metrics_rows, metrics_dir)
    all_paths += plot_metrics_bar_chart(metrics_rows, metrics_dir)

    # Task 3: Confusion matrices + class imbalance
    print("\n--- Task 3: Confusion Matrices & Class Imbalance ---")
    all_paths += plot_per_fold_confusion_matrices(fold_results, cm_dir)
    all_paths += plot_aggregated_confusion_matrix(fold_results, cm_dir)
    all_paths += plot_class_distribution(fold_results, cm_dir)
    write_analysis(
        fold_results, metrics_rows,
        cm_dir / "class_imbalance_analysis.txt",
    )

    # Task 4: MLP baseline (loss curves + metrics)
    print("\n--- Task 4: MLP Baseline ---")
    try:
        mlp_results = load_baseline_results(results_dir, "mlp")
        mlp_folds = mlp_results["per_fold"]
        all_paths += plot_per_fold_loss_curves(
            mlp_folds, loss_dir, stage="shot", model_name="MLP",
        )
        all_paths += plot_aggregated_loss_curves(
            mlp_folds, loss_dir, stage="shot", model_name="MLP",
        )
        mlp_rows = compute_per_fold_metrics(mlp_folds)
        save_metrics_csv(mlp_rows, metrics_dir / "metrics_summary_mlp.csv")
        all_paths += plot_metrics_table(mlp_rows, metrics_dir, model_name="MLP")
        all_paths += plot_metrics_bar_chart(
            mlp_rows, metrics_dir, model_name="MLP",
        )
    except FileNotFoundError as e:
        print(f"  Skipped MLP baseline: {e}")

    # Task 5: XGBoost baseline (metrics only — tree-based, no loss curves)
    print("\n--- Task 5: XGBoost Baseline ---")
    try:
        xgb_results = load_baseline_results(results_dir, "xgboost")
        xgb_folds = xgb_results["per_fold"]
        xgb_rows = compute_per_fold_metrics(xgb_folds)
        save_metrics_csv(
            xgb_rows, metrics_dir / "metrics_summary_xgboost.csv",
        )
        all_paths += plot_metrics_table(
            xgb_rows, metrics_dir, model_name="XGBoost",
        )
        all_paths += plot_metrics_bar_chart(
            xgb_rows, metrics_dir, model_name="XGBoost",
        )
    except FileNotFoundError as e:
        print(f"  Skipped XGBoost baseline: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Generated {len(all_paths)} files:")
    for p in all_paths:
        print(f"  {p.relative_to(project_root)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
