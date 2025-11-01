#!/usr/bin/env python3
"""
Visualize TacticAI Baseline Comparison
Generates bar charts and tables comparing different baseline configurations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Results data from experiments
RESULTS = {
    'receiver': {
        'models': ['Random', 'XGBoost', 'MLP-Config1\n(shot_wt=0.3)', 'MLP-Config2\n(weighted BCE)', 'MLP-Config3\n(combined)'],
        'top1': [4.52, 61.77, 31.64, 27.50, 19.21],
        'top3': [13.18, 88.14, 71.56, 72.13, 55.74],
        'top5': [22.60, 95.10, None, None, None],
    },
    'shot': {
        'models': ['Random', 'XGBoost', 'MLP-Config1\n(shot_wt=0.3)', 'MLP-Config2\n(weighted BCE)', 'MLP-Config3\n(combined)'],
        'f1': [0.33, 0.22, 0.00, 0.4522, 0.00],
        'auroc': [0.46, 0.56, 0.5383, 0.5631, 0.5396],
        'auprc': [0.27, 0.34, None, None, None],
    },
    'training': {
        'configs': ['Config 1\n(V100)', 'Config 2\n(A100)', 'Config 3\n(H100)'],
        'gpu': ['V100 32GB', 'A100 40GB', 'H100 NVL'],
        'time_minutes': [8.4, 8.4, 4.17],
        'best_val_top3': [78.4, 77.4, 76.9],
        'test_top3': [71.56, 72.13, 55.74],
        'test_shot_f1': [0.00, 0.4522, 0.00],
    }
}

def create_receiver_comparison():
    """Create receiver prediction comparison bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = RESULTS['receiver']['models']
    top1 = RESULTS['receiver']['top1']
    top3 = RESULTS['receiver']['top3']

    x = np.arange(len(models))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, top1, width, label='Top-1 Accuracy',
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, top3, width, label='Top-3 Accuracy',
                   color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height is not None and not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Styling
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Prediction Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha='center')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    # Add success criteria line
    ax.axhline(y=45, color='red', linestyle='--', linewidth=2, alpha=0.6, label='MLP Target (45%)')
    ax.axhline(y=42, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='XGBoost Target (42%)')

    # Add legend for target lines
    target_patch1 = mpatches.Patch(color='red', alpha=0.6, label='MLP Target (45%)')
    target_patch2 = mpatches.Patch(color='orange', alpha=0.6, label='XGBoost Target (42%)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [target_patch2, target_patch1], labels + ['XGBoost Target (42%)', 'MLP Target (45%)'],
             loc='upper left', framealpha=0.9)

    plt.tight_layout()
    return fig

def create_shot_comparison():
    """Create shot prediction comparison bar chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = RESULTS['shot']['models']
    f1 = RESULTS['shot']['f1']
    auroc = RESULTS['shot']['auroc']

    x = np.arange(len(models))

    # F1 Score comparison
    bars1 = ax1.bar(x, f1, color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Shot Prediction: F1 Score', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=0, ha='center')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 0.6)
    ax1.axhline(y=0.33, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Random Baseline')
    ax1.legend(loc='upper right', framealpha=0.9)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # AUROC comparison
    bars2 = ax2.bar(x, auroc, color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax2.set_title('Shot Prediction: AUROC', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=0, ha='center')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 0.7)
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Random Chance')
    ax2.legend(loc='upper right', framealpha=0.9)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig

def create_training_comparison():
    """Create training configuration comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    configs = RESULTS['training']['configs']
    x = np.arange(len(configs))

    # Training time comparison
    time_minutes = RESULTS['training']['time_minutes']
    bars1 = ax1.bar(x, time_minutes, color=['#3A506B', '#5BC0BE', '#6FDC8C'],
                    alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Time (minutes)', fontsize=11, fontweight='bold')
    ax1.set_title('Training Time Comparison', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}m',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Best validation Top-3
    best_val = RESULTS['training']['best_val_top3']
    bars2 = ax2.bar(x, best_val, color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                    alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Best Val Top-3 (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Best Validation Performance', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Test receiver Top-3
    test_top3 = RESULTS['training']['test_top3']
    bars3 = ax3.bar(x, test_top3, color=['#F67280', '#C06C84', '#6C5B7B'],
                    alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Test Top-3 (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Test Receiver Prediction Performance', fontsize=12, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 100)
    ax3.axhline(y=45, color='red', linestyle='--', linewidth=2, alpha=0.6)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Test shot F1
    test_shot = RESULTS['training']['test_shot_f1']
    bars4 = ax4.bar(x, test_shot, color=['#355C7D', '#6C5B7B', '#C06C84'],
                    alpha=0.8, edgecolor='black', linewidth=1.2)
    ax4.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Test Shot F1', fontsize=11, fontweight='bold')
    ax4.set_title('Test Shot Prediction Performance', fontsize=12, fontweight='bold', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(configs)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 0.6)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig

def create_summary_table():
    """Create comprehensive summary table visualization"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data for table
    table_data = [
        ['Model', 'Config', 'GPU', 'Receiver\nTop-1', 'Receiver\nTop-3', 'Shot\nF1', 'Shot\nAUROC', 'Status'],
        ['Random', '-', '-', '4.5%', '13.2%', '0.33', '0.46', 'Baseline ✓'],
        ['XGBoost', '-', '-', '61.8%', '88.1%', '0.22', '0.56', 'Best Receiver ⭐'],
        ['MLP', 'Config 1\n(shot_wt=0.3)', 'V100', '31.6%', '71.6%', '0.00', '0.54', 'Failed Shot ✗'],
        ['MLP', 'Config 2\n(weighted BCE)', 'A100', '27.5%', '72.1%', '0.45', '0.56', 'Best Dual-Task ⭐'],
        ['MLP', 'Config 3\n(combined)', 'H100', '19.2%', '55.7%', '0.00', '0.54', 'Unstable ✗'],
    ]

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.15, 0.1, 0.1, 0.1, 0.1, 0.12, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=11)

    # Style data rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('white')

            # Highlight best performances
            if '⭐' in table_data[i][-1]:
                cell.set_facecolor('#E8F4EA')
            elif '✗' in table_data[i][-1]:
                cell.set_facecolor('#FFEAEA')

    plt.title('TacticAI Baseline Models: Comprehensive Performance Summary',
             fontsize=16, fontweight='bold', pad=20)

    return fig

def create_dual_task_scatter():
    """Create scatter plot showing receiver vs shot performance trade-off"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Data points (Top-3 Receiver, Shot F1)
    models_scatter = {
        'Random': (13.18, 0.33, '#808080'),
        'XGBoost': (88.14, 0.22, '#2E86AB'),
        'MLP-Config1': (71.56, 0.00, '#FF6B6B'),
        'MLP-Config2': (72.13, 0.4522, '#06A77D'),
        'MLP-Config3': (55.74, 0.00, '#F18F01'),
    }

    # Plot points
    for model, (top3, f1, color) in models_scatter.items():
        size = 300 if 'Config2' in model else 200
        marker = 's' if 'XGBoost' in model else 'o'
        ax.scatter(top3, f1, s=size, c=color, alpha=0.7,
                  edgecolors='black', linewidth=2, marker=marker, label=model)

        # Add labels
        offset_x = 2 if 'Config2' in model else 1
        offset_y = 0.02 if 'Config2' in model else 0.01
        ax.annotate(model, (top3, f1), xytext=(offset_x, offset_y),
                   textcoords='offset points', fontsize=10, fontweight='bold')

    # Styling
    ax.set_xlabel('Receiver Top-3 Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shot Prediction F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Dual-Task Performance Trade-off:\nReceiver Accuracy vs Shot Prediction',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 0.6)

    # Add quadrant lines
    ax.axvline(x=45, color='red', linestyle='--', linewidth=1.5, alpha=0.4, label='MLP Target (45%)')
    ax.axhline(y=0.33, color='orange', linestyle='--', linewidth=1.5, alpha=0.4, label='Random F1 (0.33)')

    # Add ideal region annotation
    ax.fill_between([45, 100], 0.33, 0.6, alpha=0.1, color='green', label='Ideal Region')

    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    return fig

def main():
    """Generate all comparison visualizations"""
    print("Generating TacticAI Baseline Comparison Visualizations...")

    # Create output directory
    output_dir = Path("data/results/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\n1. Creating receiver comparison...")
    fig1 = create_receiver_comparison()
    fig1.savefig(output_dir / "receiver_comparison.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/receiver_comparison.png")

    print("\n2. Creating shot comparison...")
    fig2 = create_shot_comparison()
    fig2.savefig(output_dir / "shot_comparison.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/shot_comparison.png")

    print("\n3. Creating training comparison...")
    fig3 = create_training_comparison()
    fig3.savefig(output_dir / "training_comparison.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/training_comparison.png")

    print("\n4. Creating summary table...")
    fig4 = create_summary_table()
    fig4.savefig(output_dir / "summary_table.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/summary_table.png")

    print("\n5. Creating dual-task scatter plot...")
    fig5 = create_dual_task_scatter()
    fig5.savefig(output_dir / "dual_task_tradeoff.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/dual_task_tradeoff.png")

    print("\n✅ All visualizations generated successfully!")
    print(f"\nOutput directory: {output_dir.absolute()}")

    plt.close('all')

if __name__ == "__main__":
    main()
