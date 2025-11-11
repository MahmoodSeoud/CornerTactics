#!/usr/bin/env python3
"""
Comprehensive StatsBomb Raw Data Analysis Script

This script analyzes raw StatsBomb data to:
1. Calculate transition probabilities P(Event at t+1 | Event at t)
2. Build complete transition matrices
3. Document all available features
4. Generate detailed analysis reports
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.statsbomb_raw_analyzer import (
    StatsBombRawAnalyzer,
    TransitionMatrixBuilder,
    FeatureExtractor,
    ReportGenerator
)


def main():
    """Main execution function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Analyze raw StatsBomb data for transition probabilities and features"
    )
    parser.add_argument(
        '--matches', '-m',
        type=int,
        default=10,
        help='Number of matches to analyze (default: 10)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/analysis',
        help='Output directory for results (default: data/analysis)'
    )
    parser.add_argument(
        '--window-size', '-w',
        type=int,
        default=10,
        help='Event window size after corners (default: 10)'
    )
    parser.add_argument(
        '--focus-event',
        type=str,
        default='Corner',
        help='Event type to focus analysis on (default: Corner)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("StatsBomb Raw Data Analysis")
    print(f"{'='*60}\n")

    # Initialize analyzer
    analyzer = StatsBombRawAnalyzer()

    # Fetch and analyze data
    print(f"Fetching data from {args.matches} matches...")
    events = analyzer.fetch_multiple_matches(args.matches)

    if not events:
        print("ERROR: Could not fetch any events")
        sys.exit(1)

    print(f"✓ Loaded {len(events)} events from {len(analyzer.matches)} matches")

    # Identify corners
    corners = analyzer.identify_corner_kicks()
    print(f"✓ Found {len(corners)} corner kicks")

    # Build transition matrix
    print("\nBuilding transition matrix...")
    builder = TransitionMatrixBuilder()
    builder.build_from_events(events, track_corners=True)
    matrix = builder.calculate_probability_matrix()
    print(f"✓ Matrix size: {matrix.shape[0]}x{matrix.shape[1]}")

    # Get corner-specific transitions
    corner_transitions = builder.get_corner_transitions()
    if corner_transitions:
        print(f"\nTransitions from {args.focus_event}:")
        for event, prob in sorted(corner_transitions.items(),
                                 key=lambda x: x[1], reverse=True)[:5]:
            print(f"  → {event}: {prob:.3f}")

    # Extract features
    print("\nExtracting features...")
    extractor = FeatureExtractor()
    feature_summary = extractor.summarize_features(events)
    print(f"✓ Found {len(feature_summary['event_types'])} event types")
    print(f"✓ Location coverage: {feature_summary['location_coverage']*100:.1f}%")
    print(f"✓ Pressure rate: {feature_summary['pressure_rate']*100:.1f}%")

    # Extract corner sequences for detailed analysis
    print(f"\nExtracting corner sequences (window={args.window_size})...")
    sequences = analyzer.extract_corner_sequences(args.window_size)

    # Analyze sequence patterns
    if sequences:
        first_events = {}
        three_chains = {}

        for seq in sequences:
            # First event after corner
            if seq['following_events']:
                first = seq['following_events'][0].get('type', {}).get('name', 'Unknown')
                first_events[first] = first_events.get(first, 0) + 1

            # Three-event chains
            if len(seq['following_events']) >= 3:
                chain = ' → '.join([
                    e.get('type', {}).get('name', 'Unknown')
                    for e in seq['following_events'][:3]
                ])
                three_chains[chain] = three_chains.get(chain, 0) + 1

        print("\nMost common first events after corners:")
        for event, count in sorted(first_events.items(),
                                  key=lambda x: x[1], reverse=True)[:5]:
            pct = (count / len(sequences)) * 100
            print(f"  {event}: {count} ({pct:.1f}%)")

        print("\nMost common 3-event chains:")
        for chain, count in sorted(three_chains.items(),
                                  key=lambda x: x[1], reverse=True)[:3]:
            pct = (count / len(sequences)) * 100
            print(f"  {chain}: {pct:.1f}%")

    # Generate reports
    print("\nGenerating reports...")
    generator = ReportGenerator()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comprehensive report
    analysis_results = {
        'num_matches': len(analyzer.matches),
        'num_events': len(events),
        'num_corners': len(corners),
        'transition_matrix': matrix,
        'feature_summary': feature_summary,
        'corner_sequences': sequences
    }

    report = generator.generate_complete_report(analysis_results)

    # Add timestamp to report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"*Generated: {timestamp}*\n\n{report}"

    # Save report
    report_path = output_dir / "statsbomb_raw_analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✓ Report saved to: {report_path}")

    # Save transition matrix
    matrix_path = output_dir / "transition_matrix_complete.csv"
    matrix.to_csv(matrix_path)
    print(f"✓ Matrix saved to: {matrix_path}")

    # Save corner transitions specifically
    if corner_transitions:
        corner_path = output_dir / "corner_transitions.json"
        with open(corner_path, "w") as f:
            json.dump(corner_transitions, f, indent=2)
        print(f"✓ Corner transitions saved to: {corner_path}")

    # Save feature summary
    feature_path = output_dir / "feature_summary.json"
    with open(feature_path, "w") as f:
        json.dump(feature_summary, f, indent=2)
    print(f"✓ Feature summary saved to: {feature_path}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"Matches analyzed: {len(analyzer.matches)}")
    print(f"Total events: {len(events)}")
    print(f"Corner kicks: {len(corners)}")
    print(f"Event types: {len(feature_summary['event_types'])}")
    print(f"Output directory: {output_dir}")
    print("")

    return analysis_results


if __name__ == "__main__":
    main()