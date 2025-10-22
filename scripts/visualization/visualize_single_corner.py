#!/usr/bin/env python3
"""
Visualize a single corner kick with player positions.
- Attacking team: RED
- Defending team: BLUE
- Cropped to attacking half of the pitch (right side)
- Professional broadcast style presentation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import json

print("Loading corners with 360 player position data...")

# Load the CSV
csv_path = "data/statsbomb/corners_360.csv"
df = pd.read_csv(csv_path)

print(f"Total corners in dataset: {len(df)}")

# Get first corner for testing
corner_data = df.iloc[0]

print(f"\nTest corner:")
print(f"  Match: {corner_data['home_team']} vs {corner_data['away_team']}")
print(f"  Team: {corner_data['team']}")
print(f"  Time: {corner_data['minute']}:{corner_data['second']:02d}")
print(f"  Location: ({corner_data['location_x']}, {corner_data['location_y']})")

# Parse player positions
attacking_positions = json.loads(corner_data['attacking_positions'])
defending_positions = json.loads(corner_data['defending_positions'])

print(f"  Players: {len(attacking_positions)} attacking, {len(defending_positions)} defending")

# Create figure - Professional Broadcast Style
fig, ax = plt.subplots(figsize=(16, 12))
fig.patch.set_facecolor('#ffffff')  # Clean white background

# Create pitch - Professional broadcast grass with stripes
# StatsBomb pitch is 120x80, attacking half is x: 60-120
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

# Determine corner side for labeling
corner_side = "Bottom" if corner_data['location_y'] < 40 else "Top"

# Plot ATTACKING players (Bright Red - broadcast style with transparency)
if attacking_positions:
    attacking_x = [p[0] for p in attacking_positions]
    attacking_y = [p[1] for p in attacking_positions]

    # Main player markers - transparent for overlap visibility
    pitch.scatter(attacking_x, attacking_y,
                 s=500, color='#E31E24', edgecolors='#000000',
                 linewidth=3.5, zorder=9, ax=ax, label='Attacking', alpha=0.7)

# Plot DEFENDING players (Royal Blue - broadcast style with transparency)
if defending_positions:
    defending_x = [p[0] for p in defending_positions]
    defending_y = [p[1] for p in defending_positions]

    # Main player markers - transparent for overlap visibility
    pitch.scatter(defending_x, defending_y,
                 s=500, color='#0047AB', edgecolors='#000000',
                 linewidth=3.5, zorder=9, ax=ax, label='Defending', alpha=0.7)

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

# Create simple legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Attacking',
           markerfacecolor='#E31E24', markeredgecolor='#000000', markersize=12,
           markeredgewidth=2, linestyle='None', alpha=0.7),
    Line2D([0], [0], marker='o', color='w', label='Defending',
           markerfacecolor='#0047AB', markeredgecolor='#000000', markersize=12,
           markeredgewidth=2, linestyle='None', alpha=0.7),
    Line2D([0], [0], marker='o', color='w', label='Ball Landing',
           markerfacecolor='#FFD700', markeredgecolor='none', markersize=10,
           markeredgewidth=0, linestyle='None', alpha=0.4)
]

legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=14,
                   frameon=True, fancybox=False, shadow=True, framealpha=1.0,
                   edgecolor='#333333', facecolor='#ffffff', labelspacing=0.9)
plt.setp(legend.get_texts(), color='#000000', fontweight='bold')

plt.tight_layout()

# Save with unique filename based on corner_id
output_filename = f"single_corner_{corner_data['corner_id'][:8]}.png"
output_path = f'data/statsbomb/{output_filename}'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')  # Clean white background, high DPI
print(f"\n✓ Saved visualization to: {output_path}")

plt.close()
print("Done!")
