#!/usr/bin/env python3
"""
Batch generate individual corner kick visualizations.
Creates one PNG per corner kick with broadcast-style presentation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.lines import Line2D
import json
from tqdm import tqdm

print("="*70)
print("BATCH CORNER VISUALIZATION")
print("="*70)

# Load the CSV
csv_path = "data/statsbomb/corners_360.csv"
df = pd.read_csv(csv_path)

print(f"\nTotal corners to process: {len(df)}")

# Create output directory
output_dir = Path("data/statsbomb/corner_images")
output_dir.mkdir(exist_ok=True, parents=True)

print(f"Output directory: {output_dir}")
print("\nGenerating visualizations...")

# Process each corner
for idx, corner_data in tqdm(df.iterrows(), total=len(df), desc="Generating"):

    # Parse player positions
    try:
        attacking_positions = json.loads(corner_data['attacking_positions'])
        defending_positions = json.loads(corner_data['defending_positions'])
    except:
        print(f"Warning: Skipping corner {corner_data['corner_id']} - invalid player data")
        continue

    # Create figure - Professional Broadcast Style
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('#ffffff')

    # Create pitch - Professional broadcast grass with stripes
    pitch = Pitch(pitch_type='statsbomb',
                  pitch_color='#195905',    # Rich grass green
                  line_color='#ffffff',      # White lines (classic)
                  linewidth=2.5,
                  line_zorder=2,
                  stripe=True,               # Broadcast-style mowed stripes
                  stripe_color='#1a6d08',    # Alternating grass shade
                  half=True)                 # Show only half pitch

    pitch.draw(ax=ax)

    # Corner kick location
    corner_loc = [corner_data['location_x'], corner_data['location_y']]

    # Plot attacking players
    if attacking_positions:
        attacking_x = [p[0] for p in attacking_positions]
        attacking_y = [p[1] for p in attacking_positions]

        pitch.scatter(attacking_x, attacking_y,
                     s=150, color='#E31E24', edgecolors='#FFFFFF',
                     linewidth=1, zorder=9, ax=ax, label='Attacking', alpha=0.75)

    # Plot defending players
    if defending_positions:
        defending_x = [p[0] for p in defending_positions]
        defending_y = [p[1] for p in defending_positions]

        pitch.scatter(defending_x, defending_y,
                     s=150, color='#0047AB', edgecolors='#FFFFFF',
                     linewidth=1, zorder=9, ax=ax, label='Defending', alpha=0.75)

    # Plot corner trajectory with heat spot and dotted line
    if pd.notna(corner_data['end_x']) and pd.notna(corner_data['end_y']):
        end_loc = [corner_data['end_x'], corner_data['end_y']]

        # Heat spot where ball is targeted (layered circles for blur effect)
        pitch.scatter(end_loc[0], end_loc[1],
                     s=800, color='#FFD700', edgecolors='none',
                     linewidth=0, zorder=6, ax=ax, alpha=0.15)
        pitch.scatter(end_loc[0], end_loc[1],
                     s=500, color='#FFD700', edgecolors='none',
                     linewidth=0, zorder=6, ax=ax, alpha=0.25)
        pitch.scatter(end_loc[0], end_loc[1],
                     s=250, color='#FFD700', edgecolors='none',
                     linewidth=0, zorder=6, ax=ax, alpha=0.4)

        # Dotted line showing trajectory
        pitch.plot([corner_loc[0], end_loc[0]], [corner_loc[1], end_loc[1]],
                   color='#ffffff', linestyle=':', linewidth=3, zorder=7, ax=ax,
                   alpha=0.8)

    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Attacking Player',
               markerfacecolor='#E31E24', markeredgecolor='#FFFFFF', markersize=8,
               markeredgewidth=0.5, linestyle='None', alpha=0.75),
        Line2D([0], [0], marker='o', color='w', label='Defending Player',
               markerfacecolor='#0047AB', markeredgecolor='#FFFFFF', markersize=8,
               markeredgewidth=0.5, linestyle='None', alpha=0.75),
        Line2D([0], [0], marker='o', color='w', label='Ball Landing',
               markerfacecolor='#FFD700', markeredgecolor='none', markersize=8,
               markeredgewidth=0, linestyle='None', alpha=0.4)
    ]

    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=14,
                       frameon=True, fancybox=False, shadow=True, framealpha=1.0,
                       edgecolor='#333333', facecolor='#ffffff', labelspacing=0.9)
    plt.setp(legend.get_texts(), color='#000000', fontweight='bold')

    plt.tight_layout()

    # Save with unique filename based on corner_id
    output_filename = f"corner_{corner_data['corner_id']}.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')

    plt.close()

print("\n" + "="*70)
print(f"✓ Successfully generated {len(df)} corner visualizations")
print(f"✓ Output directory: {output_dir}")
print("="*70)
print("Done!")
