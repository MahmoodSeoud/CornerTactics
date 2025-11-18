"""Create corner_labels.csv from corners_with_labels.json and corners_with_shot_labels.json"""

import pandas as pd
import json
from pathlib import Path

# Load corners with outcome labels
with open('data/processed/corners_with_labels.json') as f:
    corners_labeled = json.load(f)

# Load corners with shot labels
with open('data/processed/corners_with_shot_labels.json') as f:
    corners_shot = json.load(f)

# Create mapping from event_id to shot label
shot_labels = {}
for corner in corners_shot:
    event_id = corner['event']['id']
    shot_labels[event_id] = corner.get('shot_outcome', 0)  # Already encoded as 0/1

# Extract labels
labels = []
for corner in corners_labeled:
    event_id = corner['event']['id']
    outcome = corner.get('outcome', 'Unknown')

    # Map to 4-class outcome
    if outcome == 'Ball Receipt':
        next_event_type = 0
    elif outcome == 'Clearance':
        next_event_type = 1
    elif outcome == 'Goalkeeper':
        next_event_type = 2
    else:
        next_event_type = 3

    # Get shot label (0 = No Shot, 1 = Shot)
    leads_to_shot = shot_labels.get(event_id, 0)

    labels.append({
        'event_id': event_id,
        'next_event_name': outcome,
        'next_event_type': next_event_type,
        'leads_to_shot': leads_to_shot
    })

labels_df = pd.DataFrame(labels)
output_file = Path('data/processed/corner_labels.csv')
labels_df.to_csv(output_file, index=False)

print(f'✓ Created corner_labels.csv with {len(labels_df)} labels')
print(f'\nLabel distribution (4-class):')
print(labels_df['next_event_name'].value_counts())
print(f'\nShot distribution (binary):')
print(labels_df['leads_to_shot'].value_counts())
print(f'\n4-class encoding:')
print(labels_df.groupby(['next_event_name', 'next_event_type']).size())
