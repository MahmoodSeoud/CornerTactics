#!/usr/bin/env python3
"""
Download StatsBomb 360 freeze frame data for set pieces.

This script downloads the separate 360 data files that contain
player positions at the moment of corner kicks (and other set pieces).
This is different from the shot freeze frames in regular event data.
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


def download_json(url: str, max_retries: int = 3) -> Optional[Dict]:
    """Download JSON with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # 404 means no 360 data for this match
                return None
            if attempt == max_retries - 1:
                print(f"Failed to download {url}: {e}")
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download {url}: {e}")
                return None
            time.sleep(2 ** attempt)


def main():
    """Download StatsBomb 360 freeze frame data."""

    # Base URL for StatsBomb open data
    BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

    # Create output directory
    output_dir = Path("data/statsbomb/freeze-frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("StatsBomb 360 Freeze Frame Downloader")
    print("=" * 60)

    # Load competitions to find which ones have 360 data
    print("\n📥 Loading competitions...")
    competitions_file = Path("data/statsbomb/competitions.json")

    if not competitions_file.exists():
        print("❌ Error: competitions.json not found!")
        print("   Run download_statsbomb_raw_jsons.py first")
        return

    with open(competitions_file) as f:
        competitions = json.load(f)

    # Filter competitions with 360 data
    comps_with_360 = [c for c in competitions if c.get('match_available_360')]

    print(f"✓ Found {len(comps_with_360)} competitions with 360 data:")
    for comp in comps_with_360:
        print(f"   - {comp['competition_name']} {comp['season_name']}")

    # Get match IDs for competitions with 360 data
    print("\n📋 Getting matches for 360 competitions...")
    comp_season_360 = {(c['competition_id'], c['season_id']) for c in comps_with_360}

    matches_with_360 = []
    for comp in comps_with_360:
        comp_id = comp['competition_id']
        season_id = comp['season_id']

        # Download matches for this competition/season
        matches_url = f"{BASE_URL}/matches/{comp_id}/{season_id}.json"
        matches = download_json(matches_url)

        if matches:
            for match in matches:
                match['competition_id'] = comp_id
                match['season_id'] = season_id
                matches_with_360.append(match)

    print(f"✓ Found {len(matches_with_360)} matches with potential 360 data")

    # Download 360 data for each match
    print(f"\n⬇️ Downloading 360 freeze frame data...")

    downloaded = 0
    not_found = 0
    errors = 0
    total_freeze_frames = 0
    total_corners_with_360 = 0

    # Build event data path
    events_dir = Path("data/statsbomb/events")

    for match in tqdm(matches_with_360, desc="Downloading 360 data"):
        match_id = match['match_id']

        # Load event data to find corner UUIDs
        event_file = events_dir / f"{match_id}.json"
        corner_uuids = set()

        if event_file.exists():
            with open(event_file) as f:
                events = json.load(f)
                for event in events:
                    if (event.get('type', {}).get('name') == 'Pass' and
                        event.get('pass', {}).get('type', {}).get('name') == 'Corner'):
                        corner_uuids.add(event['id'])

        # Skip if already downloaded
        output_file = output_dir / f"{match_id}.json"
        if output_file.exists():
            with open(output_file) as f:
                data_360 = json.load(f)
                downloaded += 1
                total_freeze_frames += len(data_360)

                # Count corner kick freeze frames by matching UUIDs
                for frame in data_360:
                    if frame.get('event_uuid') in corner_uuids:
                        total_corners_with_360 += 1
                continue

        # Download 360 data
        url = f"{BASE_URL}/three-sixty/{match_id}.json"
        data_360 = download_json(url)

        if data_360 is None:
            not_found += 1
            continue

        if isinstance(data_360, dict) and 'error' in data_360:
            errors += 1
            continue

        # Save 360 data
        with open(output_file, 'w') as f:
            json.dump(data_360, f, indent=2)

        downloaded += 1
        total_freeze_frames += len(data_360)

        # Count corner kick freeze frames by matching UUIDs
        for frame in data_360:
            if frame.get('event_uuid') in corner_uuids:
                total_corners_with_360 += 1

        # Be nice to GitHub
        time.sleep(0.1)

    print(f"\n✅ Download Complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Statistics:")
    print(f"   - Total matches checked: {len(matches_with_360)}")
    print(f"   - 360 files downloaded: {downloaded}")
    print(f"   - Matches without 360 data: {not_found}")
    print(f"   - Errors: {errors}")
    print(f"   - Total freeze frames: {total_freeze_frames:,}")
    print(f"   - Corner kick freeze frames: {total_corners_with_360:,}")

    if total_corners_with_360 > 0:
        print(f"\n🎉 SUCCESS! You now have {total_corners_with_360:,} corners with 360 freeze frames!")
        print(f"\n💡 Next steps:")
        print(f"   1. Validate the 360 data structure")
        print(f"   2. Match corner events with their 360 freeze frames")
        print(f"   3. Extract features from player positions")
    else:
        print(f"\n⚠️  No corner kick freeze frames found")
        print(f"   This may be expected - 360 data is only for specific events")
        print(f"   Check the freeze frame data to see what's available")


if __name__ == "__main__":
    main()
