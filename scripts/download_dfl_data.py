#!/usr/bin/env python3
"""
Download DFL tracking data from figshare.

The DFL dataset is hosted on figshare and requires handling of redirects
and WAF challenges. This script attempts multiple download strategies.
"""

import os
import sys
import time
import requests
from pathlib import Path


def download_figshare_article(article_id: str, output_dir: Path) -> bool:
    """
    Download all files from a figshare article.

    Args:
        article_id: The figshare article ID (e.g., "28196177")
        output_dir: Directory to save downloaded files

    Returns:
        True if download succeeded, False otherwise
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a session with browser-like headers
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })

    print(f"Fetching file list from figshare article {article_id}...")

    # Get full article details (includes all files)
    article_url = f"https://api.figshare.com/v2/articles/{article_id}"
    try:
        response = session.get(article_url, timeout=30)
        response.raise_for_status()
        article_data = response.json()
        files = article_data.get("files", [])
    except requests.RequestException as e:
        print(f"Error fetching article details: {e}")
        return False

    if not files:
        print("No files found in article")
        return False

    print(f"Found {len(files)} file(s) to download:")
    for f in files:
        print(f"  - {f['name']} ({f['size'] / 1024 / 1024:.1f} MB)")

    # Download each file
    success = True
    for file_info in files:
        file_name = file_info["name"]
        file_url = file_info["download_url"]
        file_size = file_info["size"]
        output_path = output_dir / file_name

        # Skip if already downloaded
        if output_path.exists() and output_path.stat().st_size == file_size:
            print(f"Skipping {file_name} (already downloaded)")
            continue

        print(f"\nDownloading {file_name} ({file_size / 1024 / 1024:.1f} MB)...")

        try:
            # Stream download with progress
            download_response = session.get(file_url, stream=True, timeout=60)
            download_response.raise_for_status()

            downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in download_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Print progress every 10MB
                        if downloaded % (10 * 1024 * 1024) < 8192:
                            pct = 100 * downloaded / file_size
                            print(f"  Progress: {downloaded / 1024 / 1024:.1f} MB ({pct:.1f}%)")

            print(f"  Downloaded: {output_path}")

        except requests.RequestException as e:
            print(f"  Error downloading {file_name}: {e}")
            success = False
            continue

    return success


def main():
    """Main entry point."""
    # DFL dataset article ID
    article_id = "28196177"

    # Output directory
    output_dir = Path(__file__).parent.parent / "data" / "dfl"

    print("=" * 60)
    print("DFL Dataset Downloader")
    print("=" * 60)
    print(f"Article ID: {article_id}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    success = download_figshare_article(article_id, output_dir)

    if success:
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)

        # List downloaded files
        print("\nDownloaded files:")
        for f in output_dir.iterdir():
            if f.is_file():
                print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")

        return 0
    else:
        print("\n" + "=" * 60)
        print("Download failed or incomplete")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
