#!/usr/bin/env python3
"""Generate per-fold loss curve plots and overfitting summary CSV.

Produces 5 plots:
1. Frozen backbone GNN — Stage 2 (shot)
2. Frozen backbone GNN — Stage 1 (receiver)
3. Random-init GNN — Stage 2 (shot)
4. Random-init GNN — Stage 1 (receiver)
5. MLP — Stage 2 (shot)
(MLP has no receiver head, so no 6th plot.)

Plus: results/overfitting_summary.csv
"""

import pickle
import math
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# --- Config ---
MODELS = {
    'frozen_backbone': {
        'pkl': 'results/corner_prediction/combined_lomo_ussf_aligned_seed42_20260221_164251.pkl',
        'label': 'Frozen USSF Backbone',
        'has_receiver': True,
    },
    'random_init': {
        'pkl': 'results/corner_prediction/combined_lomo_ussf_random_init_seed42_20260313_062515.pkl',
        'label': 'Random-Init GNN',
        'has_receiver': True,
    },
    'mlp': {
        'pkl': 'results/corner_prediction/combined_baseline_mlp_20260221_163943.pkl',
        'label': 'MLP Baseline',
        'has_receiver': False,
    },
}

OUTDIR = 'figures/loss_curves'
os.makedirs(OUTDIR, exist_ok=True)

NROWS, NCOLS = 4, 5  # 20 subplots for 17 folds + 3 blank


def fold_source(match_id):
    """Return 'SK' or 'DFL' based on match_id."""
    s = str(match_id)
    return 'DFL' if 'DFL' in s or s.startswith('J0') else 'SK'


def has_valid_val(vals):
    """Check if val loss list has any non-NaN values."""
    return any(not math.isnan(v) for v in vals)


def plot_perfold(data, stage, model_label, save_path, is_receiver=False):
    """Plot per-fold loss curves in a 4x5 grid.

    For receiver plots: folds with all-NaN val show train-only.
    """
    per_fold = data['per_fold']
    n_folds = len(per_fold)

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, 16))
    fig.suptitle(f'{model_label} — {stage.capitalize()} Loss (Per Fold)', fontsize=16, y=0.98)

    # Compute shared y-axis range using robust percentiles (handles outliers)
    all_losses = []
    for fold in per_fold:
        lh = fold['loss_history'].get(stage, {})
        train = lh.get('train', [])
        val = lh.get('val', [])
        all_losses.extend([v for v in train if not math.isnan(v)])
        all_losses.extend([v for v in val if not math.isnan(v)])

    if not all_losses:
        plt.close(fig)
        return

    arr = np.array(all_losses)
    y_min = max(0, np.percentile(arr, 1) * 0.90)
    y_max = np.percentile(arr, 99) * 1.10

    for idx in range(NROWS * NCOLS):
        ax = axes[idx // NCOLS][idx % NCOLS]

        if idx >= n_folds:
            ax.set_visible(False)
            continue

        fold = per_fold[idx]
        lh = fold['loss_history'].get(stage, {})
        train = lh.get('train', [])
        val = lh.get('val', [])
        match_id = fold.get('held_out_match', '?')
        n_test = fold.get('n_test', '?')
        source = fold_source(match_id)

        # Get metric for title
        if stage == 'shot':
            metric_val = fold.get('shot_oracle', {}).get('auc', None)
            metric_name = 'AUC'
        else:
            recv = fold.get('receiver', {})
            if isinstance(recv, dict) and recv.get('n_labeled', 0) > 0:
                metric_val = recv.get('top3_acc', None)
            else:
                metric_val = None
            metric_name = 'Top3'

        if not train:
            ax.set_visible(False)
            continue

        n_epochs = len(train)
        epochs = np.arange(1, n_epochs + 1)

        # Early stopping epoch = last epoch recorded
        es_epoch = n_epochs

        # X-axis: early stopping + 5 buffer
        x_max = es_epoch + 5

        # Plot train
        ax.plot(epochs, train, 'r-', linewidth=1.2, label='Train')

        # Plot val (handle NaN / no-val-labels case)
        val_valid = has_valid_val(val) if val else False

        if is_receiver and not val_valid:
            subtitle_extra = '\n(No val labels)'
        else:
            if val and val_valid:
                ax.plot(epochs, val, 'b--', linewidth=1.2, label='Val')
            subtitle_extra = ''

        # Green dot at early stopping epoch
        ax.plot(es_epoch, train[-1], 'go', markersize=6, zorder=5)

        # Title
        metric_str = f'{metric_val:.3f}' if metric_val is not None else 'N/A'
        ax.set_title(f'Fold {idx} [{source}] n={n_test}, {metric_name}={metric_str}{subtitle_extra}',
                     fontsize=8)

        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=6, loc='upper right')

        if idx // NCOLS == NROWS - 1 or idx >= n_folds - NCOLS:
            ax.set_xlabel('Epoch', fontsize=8)
        if idx % NCOLS == 0:
            ax.set_ylabel('Loss', fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    fig.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_path}')


def build_overfitting_csv(models_data, save_path):
    """Build overfitting summary CSV."""
    rows = []
    for model_name, (data, cfg) in models_data.items():
        for stage in ['shot', 'receiver']:
            if stage == 'receiver' and not cfg['has_receiver']:
                continue
            for idx, fold in enumerate(data['per_fold']):
                lh = fold['loss_history'].get(stage, {})
                train = lh.get('train', [])
                val = lh.get('val', [])
                if not train:
                    continue

                es_epoch = len(train)
                final_train = train[-1]
                final_val = val[-1] if val and not math.isnan(val[-1]) else float('nan')
                gap = final_val - final_train if not math.isnan(final_val) else float('nan')

                rows.append({
                    'model': cfg['label'],
                    'stage': stage,
                    'fold': idx,
                    'match_id': str(fold.get('held_out_match', '?')).replace('DFL-MAT-', ''),
                    'source': fold_source(fold.get('held_out_match', '')),
                    'fold_size_n': fold.get('n_test', '?'),
                    'early_stopping_epoch': es_epoch,
                    'final_train_loss': round(final_train, 6),
                    'final_val_loss': round(final_val, 6) if not math.isnan(final_val) else 'NaN',
                    'gap': round(gap, 6) if not math.isnan(gap) else 'NaN',
                })

    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {save_path}')


def main():
    # Load all data
    models_data = {}
    for name, cfg in MODELS.items():
        with open(cfg['pkl'], 'rb') as f:
            data = pickle.load(f)
        models_data[name] = (data, cfg)

    # Plot 1: Frozen backbone — shot
    plot_perfold(models_data['frozen_backbone'][0], 'shot',
                 'Frozen USSF Backbone', f'{OUTDIR}/frozen_backbone_shot_perfold.pdf')

    # Plot 2: Frozen backbone — receiver
    plot_perfold(models_data['frozen_backbone'][0], 'receiver',
                 'Frozen USSF Backbone', f'{OUTDIR}/frozen_backbone_receiver_perfold.pdf',
                 is_receiver=True)

    # Plot 3: Random-init — shot
    plot_perfold(models_data['random_init'][0], 'shot',
                 'Random-Init GNN', f'{OUTDIR}/random_init_shot_perfold.pdf')

    # Plot 4: Random-init — receiver
    plot_perfold(models_data['random_init'][0], 'receiver',
                 'Random-Init GNN', f'{OUTDIR}/random_init_receiver_perfold.pdf',
                 is_receiver=True)

    # Plot 5: MLP — shot (no receiver head)
    plot_perfold(models_data['mlp'][0], 'shot',
                 'MLP Baseline', f'{OUTDIR}/mlp_shot_perfold.pdf')

    # Overfitting summary CSV
    os.makedirs('results', exist_ok=True)
    build_overfitting_csv(models_data, 'results/overfitting_summary.csv')


if __name__ == '__main__':
    main()
