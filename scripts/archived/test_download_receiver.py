#!/usr/bin/env python3
"""
Test the updated download script with receiver extraction on a single match.
"""

import json
import pandas as pd
from statsbombpy import sb

# Test with a specific match that we know has corners with 360 data
match_id = 3795506  # Champions League Final 2021

print(f"Testing receiver extraction for match {match_id}")

# Get 360 frames
frames_df = sb.frames(match_id=match_id, fmt='dataframe')
print(f"✓ Loaded {len(frames_df)} freeze frame records")

# Get events
events_df = sb.events(match_id=match_id, fmt='dataframe')
print(f"✓ Loaded {len(events_df)} events")

# Filter for corners
corner_events = events_df[
    (events_df['type'] == 'Pass') &
    (events_df['pass_type'].fillna('').str.contains('Corner', case=False, na=False))
].copy()

print(f"✓ Found {len(corner_events)} corner kicks")

# Test extraction on first few corners
corners_with_receiver = []
corners_without_receiver = []

for idx, (_, corner) in enumerate(corner_events.head(10).iterrows()):
    corner_frames = frames_df[frames_df['id'] == corner['id']]

    print(f"\nCorner {idx+1}: {corner['player']} - 360 frames: {len(corner_frames)}")

    if len(corner_frames) > 0:
        # Extract receiver
        receiver_name = None
        receiver_location_x = None
        receiver_location_y = None

        if 'pass_recipient' in corner and pd.notna(corner['pass_recipient']):
            receiver_name = corner['pass_recipient']
            print(f"  Recipient: {receiver_name}")

            # Find Ball Receipt
            try:
                ball_receipts = events_df[
                    (events_df['type'] == 'Ball Receipt*') &
                    (events_df['player'] == receiver_name)
                ]

                print(f"  Found {len(ball_receipts)} Ball Receipts for {receiver_name}")

                # Filter to ones related to this corner event
                corner_id = corner['id']
                related_receipts = ball_receipts[ball_receipts['related_events'].apply(
                    lambda x: corner_id in x if isinstance(x, list) else False
                )]

                print(f"  Related to this corner: {len(related_receipts)}")

                if len(related_receipts) > 0:
                    nearby_receipts = related_receipts

                    if len(nearby_receipts) > 0:
                        receipt = nearby_receipts.iloc[0]
                        if 'location' in receipt and isinstance(receipt['location'], list) and len(receipt['location']) >= 2:
                            receiver_location_x = receipt['location'][0]
                            receiver_location_y = receipt['location'][1]
                            corners_with_receiver.append({
                                'corner_id': corner['id'],
                                'kicker': corner['player'],
                                'receiver': receiver_name,
                                'location': [receiver_location_x, receiver_location_y]
                            })
                        else:
                            corners_without_receiver.append({'corner_id': corner['id'], 'reason': 'no_location'})
                else:
                    corners_without_receiver.append({'corner_id': corner['id'], 'reason': 'no_related_receipt'})
            except Exception as e:
                corners_without_receiver.append({'corner_id': corner['id'], 'reason': f'error:{e}'})
        else:
            corners_without_receiver.append({'corner_id': corner['id'], 'reason': 'no_recipient'})

print(f"\n✓ Corners with receiver location: {len(corners_with_receiver)}")
print(f"✗ Corners without receiver location: {len(corners_without_receiver)}")

if corners_with_receiver:
    print(f"\nSample corners with receiver:")
    for c in corners_with_receiver[:3]:
        print(f"  {c['kicker']} → {c['receiver']} at {c['location']}")

if corners_without_receiver:
    print(f"\nReasons for missing receiver:")
    reasons = {}
    for c in corners_without_receiver:
        reason = c['reason']
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in reasons.items():
        print(f"  {reason}: {count}")
