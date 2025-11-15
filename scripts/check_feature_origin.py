#!/usr/bin/env python3
"""Check where features come from in the dataset."""

import json

# Load data
with open('data/analysis/corner_sequences_full.json', 'r') as f:
    data = json.load(f)

corner = data[0]

print("="*60)
print("Feature Origin Analysis")
print("="*60)
print()

print("1. Corner Event (the kick itself):")
print(f"   - timestamp: {corner['corner_event']['timestamp']}")
print(f"   - minute: {corner['corner_event']['minute']}")
print(f"   - second: {corner['corner_event']['second']}")
print(f"   - location: {corner['corner_event']['location']}")
print()

print("2. Pass details (from StatsBomb):")
pass_info = corner['corner_event']['pass']
print(f"   - end_location: {pass_info['end_location']} <- WHERE BALL LANDS")
print(f"   - length: {pass_info['length']} <- CALCULATED BY STATSBOMB")
print(f"   - angle: {pass_info['angle']} <- CALCULATED BY STATSBOMB")
print(f"   - shot_assist: {pass_info['shot_assist']} <- POST-KICK LABEL")
print(f"   - recipient: {pass_info['recipient']['name']} <- WHO RECEIVES")
print()

print("3. Next events (what happened after corner):")
for i, evt in enumerate(corner['next_events'][:3]):
    print(f"   Event {i+1}: {evt['type']['name']} at {evt['timestamp']} (second: {evt['second']})")
print()

print("="*60)
print("CONCLUSION:")
print("="*60)
print()
print("StatsBomb PROVIDES these features:")
print("  - end_location_x/y: Where the ball landed")
print("  - pass_length: Distance = sqrt((end_x - start_x)^2 + (end_y - start_y)^2)")
print("  - pass_angle: Angle = atan2(end_y - start_y, end_x - start_x)")
print("  - shot_assist: Boolean - did this pass lead to a shot?")
print()
print("These are NOT derived by us - they come from StatsBomb event data.")
print("However, they are STILL LEAKAGE because:")
print("  - end_location: Reveals WHERE ball went (outcome of kick)")
print("  - pass_length/angle: Derived FROM end_location")
print("  - shot_assist: Reveals IF a shot happened (prediction target)")
print()
print("We should NOT use these for prediction - they contain outcome info!")
print("="*60)
