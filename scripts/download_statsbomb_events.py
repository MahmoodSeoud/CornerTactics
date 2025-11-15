#!/usr/bin/env python3
"""
Download ALL raw JSON event files from StatsBomb open-data.
This preserves 100% of the data structure for complete analysis.
"""

import json
import requests
import time
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm


def download_json(url: str, max_retries: int = 3) -> Dict:
    """Download JSON with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download {url}: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff


def main():
    """Download all StatsBomb raw event JSONs."""

    # Base URL for raw GitHub content
    BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

    # Create output directory
    output_dir = Path("data/statsbomb")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("StatsBomb Raw JSON Downloader")
    print("=" * 60)

    # Step 1: Get competitions
    print("\n📥 Downloading competitions...")
    competitions = download_json(f"{BASE_URL}/competitions.json")
    if not competitions:
        print("Failed to download competitions")
        return

    # Save competitions
    with open(output_dir / "competitions.json", "w") as f:
        json.dump(competitions, f, indent=2)

    print(f"✓ Found {len(competitions)} competitions")

    # Step 2: Get all match IDs
    all_matches = []
    print("\n📋 Collecting match IDs...")

    for comp in tqdm(competitions, desc="Competitions"):
        comp_id = comp['competition_id']
        season_id = comp['season_id']

        # Get matches for this competition/season
        matches_url = f"{BASE_URL}/matches/{comp_id}/{season_id}.json"
        matches = download_json(matches_url)

        if matches:
            for match in matches:
                match['competition_name'] = comp['competition_name']
                match['season_name'] = comp['season_name']
                all_matches.append(match)

    print(f"✓ Found {len(all_matches)} total matches")

    # Save match index
    matches_df = pd.DataFrame(all_matches)
    matches_df.to_csv(output_dir / "match_index.csv", index=False)

    # Step 3: Download ALL event files
    print(f"\n⬇️ Downloading {len(all_matches)} event files...")
    print("This will take a while but gives you COMPLETE data.")

    events_dir = output_dir / "events"
    events_dir.mkdir(exist_ok=True)

    # Track statistics
    total_events = 0
    total_corners = 0
    files_with_corners = []

    for match in tqdm(all_matches, desc="Downloading events"):
        match_id = match['match_id']

        # Skip if already downloaded
        event_file = events_dir / f"{match_id}.json"
        if event_file.exists():
            with open(event_file) as f:
                events = json.load(f)
        else:
            # Download events
            events_url = f"{BASE_URL}/events/{match_id}.json"
            events = download_json(events_url)

            if events:
                # Save raw JSON
                with open(event_file, "w") as f:
                    json.dump(events, f, indent=2)

                # Small delay to be nice to GitHub
                time.sleep(0.1)

        if events:
            total_events += len(events)

            # Count corners in this match
            corners_in_match = 0
            for event in events:
                # Check for corner kicks (various ways it might be stored)
                if event.get('type', {}).get('name') == 'Pass':
                    pass_obj = event.get('pass', {})
                    if pass_obj.get('type', {}).get('name') == 'Corner':
                        corners_in_match += 1

            if corners_in_match > 0:
                total_corners += corners_in_match
                files_with_corners.append({
                    'match_id': match_id,
                    'competition': match.get('competition_name'),
                    'home_team': match.get('home_team', {}).get('home_team_name'),
                    'away_team': match.get('away_team', {}).get('away_team_name'),
                    'num_corners': corners_in_match
                })

    # Save corner index
    if files_with_corners:
        corners_df = pd.DataFrame(files_with_corners)
        corners_df.to_csv(output_dir / "matches_with_corners.csv", index=False)

    # Step 4: Create a combined dataset of ALL events (for sequence analysis)
    print("\n📊 Creating master event sequence file...")

    all_events = []
    for match in tqdm(all_matches[:100], desc="Building sequence dataset"):  # Start with 100 matches
        match_id = match['match_id']
        event_file = events_dir / f"{match_id}.json"

        if event_file.exists():
            with open(event_file) as f:
                events = json.load(f)

                # Add match context to each event
                for i, event in enumerate(events):
                    event['match_id'] = match_id
                    event['event_index'] = i
                    event['next_event_same_match'] = i < len(events) - 1
                    all_events.append(event)

    # Save master sequence file
    with open(output_dir / "master_event_sequence.json", "w") as f:
        json.dump(all_events, f)

    print(f"\n✅ Download Complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Statistics:")
    print(f"   - Total matches: {len(all_matches)}")
    print(f"   - Total events: {total_events:,}")
    print(f"   - Total corners: {total_corners:,}")
    print(f"   - Matches with corners: {len(files_with_corners)}")
    print(f"   - Events in sequence file: {len(all_events):,}")

    print(f"\n💡 Next steps:")
    print(f"   1. Run analyze_corner_transitions.py to compute P(a_{{t+1}} | corner_t)")
    print(f"   2. All raw data preserved - filter later as needed")


if __name__ == "__main__":
    main()