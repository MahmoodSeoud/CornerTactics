"""Merge SkillCorner and DFL corner kick records into a combined dataset.

Adds a 'source' field to each record and concatenates both datasets.
The combined dataset can be used for cross-source evaluation.

Usage:
    python -m corner_prediction.data.merge_datasets
    python -m corner_prediction.data.merge_datasets \
        --skillcorner corner_prediction/data/extracted_corners.pkl \
        --dfl corner_prediction/data/dfl_extracted_corners.pkl \
        --output corner_prediction/data/combined_corners.pkl
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def merge_records(
    skillcorner_records: List[Dict[str, Any]],
    dfl_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge SkillCorner and DFL corner records.

    Adds 'source' field to each record if not already present.
    Validates no duplicate corner_ids across sources.

    Returns combined list of records.
    """
    # Tag source
    for r in skillcorner_records:
        if "source" not in r:
            r["source"] = "skillcorner"

    for r in dfl_records:
        if "source" not in r:
            r["source"] = "dfl"

    # Check for duplicate corner_ids
    sc_ids = {r["corner_id"] for r in skillcorner_records}
    dfl_ids = {r["corner_id"] for r in dfl_records}
    overlap = sc_ids & dfl_ids
    if overlap:
        logger.warning("Duplicate corner_ids across sources: %s", overlap)

    combined = skillcorner_records + dfl_records

    logger.info("Merged: %d SkillCorner + %d DFL = %d combined",
                len(skillcorner_records), len(dfl_records), len(combined))
    return combined


def print_summary(records: List[Dict[str, Any]]) -> None:
    """Print summary of the combined dataset."""
    n = len(records)
    if n == 0:
        print("Empty combined dataset.")
        return

    # Per-source stats
    sources = {}
    for r in records:
        src = r.get("source", "unknown")
        if src not in sources:
            sources[src] = {"n": 0, "shots": 0, "goals": 0, "receiver": 0, "matches": set()}
        sources[src]["n"] += 1
        if r.get("lead_to_shot"):
            sources[src]["shots"] += 1
        if r.get("lead_to_goal"):
            sources[src]["goals"] += 1
        if r.get("has_receiver_label"):
            sources[src]["receiver"] += 1
        sources[src]["matches"].add(str(r["match_id"]))

    print(f"\n{'='*60}")
    print("Combined Dataset Summary")
    print(f"{'='*60}")
    print(f"Total corners: {n}")
    print(f"Total matches: {len(set(str(r['match_id']) for r in records))}")

    for src, stats in sorted(sources.items()):
        print(f"\n  Source: {src}")
        print(f"    Corners:         {stats['n']}")
        print(f"    Matches:         {len(stats['matches'])}")
        print(f"    Shot rate:       {stats['shots']}/{stats['n']} "
              f"({100*stats['shots']/stats['n']:.1f}%)")
        print(f"    Goal rate:       {stats['goals']}/{stats['n']} "
              f"({100*stats['goals']/stats['n']:.1f}%)")
        print(f"    Receiver labels: {stats['receiver']}/{stats['n']} "
              f"({100*stats['receiver']/stats['n']:.1f}%)")

    n_shot = sum(1 for r in records if r.get("lead_to_shot"))
    n_recv = sum(1 for r in records if r.get("has_receiver_label"))
    print(f"\n  Combined shot rate:     {n_shot}/{n} ({100*n_shot/n:.1f}%)")
    print(f"  Combined receiver rate: {n_recv}/{n} ({100*n_recv/n:.1f}%)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge SkillCorner and DFL corner records",
    )
    parser.add_argument(
        "--skillcorner", type=str,
        default="corner_prediction/data/extracted_corners.pkl",
        help="Path to SkillCorner extracted corners pickle",
    )
    parser.add_argument(
        "--dfl", type=str,
        default="corner_prediction/data/dfl_extracted_corners.pkl",
        help="Path to DFL extracted corners pickle",
    )
    parser.add_argument(
        "--output", type=str,
        default="corner_prediction/data/combined_corners.pkl",
        help="Output path for combined records",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load
    sc_path = Path(args.skillcorner)
    dfl_path = Path(args.dfl)

    with open(sc_path, "rb") as f:
        sc_records = pickle.load(f)
    logger.info("Loaded %d SkillCorner records from %s", len(sc_records), sc_path)

    with open(dfl_path, "rb") as f:
        dfl_records = pickle.load(f)
    logger.info("Loaded %d DFL records from %s", len(dfl_records), dfl_path)

    # Merge
    combined = merge_records(sc_records, dfl_records)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(combined, f)
    logger.info("Saved combined dataset to %s", output_path)

    # JSON summary
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info("Saved JSON: %s", json_path)

    # Summary
    print_summary(combined)


if __name__ == "__main__":
    main()
