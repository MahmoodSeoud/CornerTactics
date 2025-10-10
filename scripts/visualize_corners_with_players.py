#!/usr/bin/env python3
"""
Visualize corner kicks with player positions (2x2 grid).
Attacking team in one color, defending team in another.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import requests
import json

print("Fetching corners with 360 player position data...")
print("="*70)

# Get 360 data from GitHub
github_360_url = "https://api.github.com/repos/statsbomb/open-data/contents/data/three-sixty"
response = requests.get(github_360_url)

corners_with_360 = []

if response.status_code == 200:
    files = response.json()
    print(f"Total 360 files: {len(files)}")

    # Check first 20 files to find 4 good corners
    for idx, file_info in enumerate(files[:20]):
        if len(corners_with_360) >= 4:
            break

        match_id = file_info['name'].replace('.json', '')
        print(f"\nChecking match {idx+1}: {match_id}")

        # Download 360 data
        threesixty_response = requests.get(file_info['download_url'])
        if threesixty_response.status_code != 200:
            continue

        threesixty_data = threesixty_response.json()

        # Get corresponding events
        events_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
        events_response = requests.get(events_url)

        if events_response.status_code != 200:
            continue

        events_data = events_response.json()

        # Find corners
        corners = [e for e in events_data if e.get('type', {}).get('name') == 'Pass'
                  and e.get('pass', {}).get('type', {}).get('name', '').lower().find('corner') != -1]

        # Match corners with 360 data
        for corner in corners:
            corner_360 = next((e for e in threesixty_data if e['event_uuid'] == corner['id']), None)

            if corner_360 and len(corner_360.get('freeze_frame', [])) >= 10:
                corners_with_360.append({
                    'corner': corner,
                    'freeze_frame': corner_360['freeze_frame'],
                    'match_id': match_id
                })
                print(f"  ✓ Found corner with {len(corner_360['freeze_frame'])} players")

                if len(corners_with_360) >= 4:
                    break

print(f"\n✓ Found {len(corners_with_360)} corners with player positions")

# Create 2x2 visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
fig.patch.set_facecolor('white')

for idx, (ax, corner_data) in enumerate(zip(axes.flatten(), corners_with_360)):
    corner = corner_data['corner']
    freeze_frame = corner_data['freeze_frame']

    # Create pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#f4f4f4',
                  line_zorder=2, line_color='black', linewidth=2)
    pitch.draw(ax=ax)

    # Get corner info
    team = corner.get('team', {}).get('name', 'Unknown')
    minute = corner.get('minute', 0)
    second = corner.get('second', 0)
    location = corner.get('location', [120, 0])

    # Plot corner kick starting point
    pitch.scatter(location[0], location[1],
                 s=400, color='red', edgecolors='black',
                 linewidth=3, zorder=10, ax=ax, marker='*', label='Corner')

    # Plot corner arrow if end location exists
    end_location = corner.get('pass', {}).get('end_location', None)
    if end_location:
        pitch.arrows(location[0], location[1],
                    end_location[0], end_location[1],
                    width=3, headwidth=8, headlength=6,
                    color='red', ax=ax, zorder=8, alpha=0.7)

    # Plot players
    attacking_players = []
    defending_players = []

    for player in freeze_frame:
        pos = player['location']
        is_teammate = player['teammate']
        is_keeper = player.get('keeper', False)

        if is_teammate:
            attacking_players.append(pos)
        else:
            defending_players.append(pos)

    # Plot attacking team (blue)
    if attacking_players:
        attacking_x = [p[0] for p in attacking_players]
        attacking_y = [p[1] for p in attacking_players]
        pitch.scatter(attacking_x, attacking_y,
                     s=250, color='#1e90ff', edgecolors='white',
                     linewidth=2.5, zorder=9, ax=ax, label='Attacking', alpha=0.9)

    # Plot defending team (orange)
    if defending_players:
        defending_x = [p[0] for p in defending_players]
        defending_y = [p[1] for p in defending_players]
        pitch.scatter(defending_x, defending_y,
                     s=250, color='#ff6b35', edgecolors='white',
                     linewidth=2.5, zorder=9, ax=ax, label='Defending', alpha=0.9)

    # Add legend
    ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True,
             shadow=True, framealpha=0.9)

    # Set title
    title = f"{team}\n{minute}:{second:02d} | {len(attacking_players)} ATK vs {len(defending_players)} DEF"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

# Add main title
fig.suptitle('StatsBomb 360: Corner Kicks with Player Positions\n' +
            'Blue = Attacking Team | Orange = Defending Team | Red Star = Corner Kick',
            fontsize=20, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save
output_path = 'data/statsbomb/corners_with_players_2x2.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved visualization to: {output_path}")

plt.close()
print("Done!")
