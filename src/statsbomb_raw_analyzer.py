#!/usr/bin/env python3
"""
StatsBomb Raw Data Analyzer

Analyzes raw StatsBomb data to:
1. Calculate transition probabilities P(Event at t+1 | Event at t)
2. Build complete transition matrices
3. Document all available features
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path


class StatsBombRawAnalyzer:
    """Main analyzer for StatsBomb raw data using statsbombpy SDK."""

    def __init__(self):
        """Initialize the analyzer with StatsBomb SDK."""
        try:
            from statsbombpy import sb
            self.sb = sb
        except ImportError:
            raise ImportError("statsbombpy not installed. Run: pip install statsbombpy")

        self.events = []
        self.competitions = []
        self.matches = []

    def fetch_competitions(self) -> pd.DataFrame:
        """Fetch all available competitions from StatsBomb."""
        self.competitions = self.sb.competitions()
        return self.competitions

    def fetch_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Fetch matches for a specific competition and season."""
        return self.sb.matches(competition_id=competition_id, season_id=season_id)

    def fetch_match_events(self, match_id: int) -> pd.DataFrame:
        """Fetch all events for a specific match."""
        return self.sb.events(match_id=match_id)

    def fetch_multiple_matches(self, num_matches: int = 10) -> List[Dict]:
        """Fetch events from multiple matches for better statistics."""
        all_events = []
        match_count = 0

        # Try to get competitions if not already fetched
        if len(self.competitions) == 0:
            comps_df = self.fetch_competitions()
        else:
            comps_df = self.competitions

        # Focus on major competitions
        target_comps = ['Champions League', 'La Liga', 'Premier League']

        for _, comp in comps_df.iterrows():
            if comp['competition_name'] in target_comps:
                comp_id = comp['competition_id']
                season_id = comp['season_id']

                try:
                    matches_df = self.fetch_matches(comp_id, season_id)

                    for _, match in matches_df.head(5).iterrows():  # Try first 5 matches
                        try:
                            events_df = self.fetch_match_events(match['match_id'])

                            # Convert DataFrame to list of dicts for compatibility
                            if not events_df.empty:
                                events_list = events_df.to_dict('records')
                                all_events.extend(events_list)
                                self.matches.append(match.to_dict())
                                match_count += 1

                                if match_count >= num_matches:
                                    self.events = all_events
                                    return all_events
                        except Exception as e:
                            continue
                except Exception as e:
                    continue

        self.events = all_events
        return all_events

    def identify_corner_kicks(self) -> List[Dict]:
        """Identify all corner kicks in the loaded events."""
        corners = []
        for event in self.events:
            # StatsBomb SDK flattens the data structure
            # Check for 'type' == 'Pass' and 'pass_type' contains 'Corner'
            event_type = event.get('type', '')
            pass_type = event.get('pass_type', '')

            if event_type == 'Pass' and isinstance(pass_type, str) and 'Corner' in pass_type:
                corners.append(event)
        return corners

    def extract_corner_sequences(self, window_size: int = 10) -> List[Dict]:
        """Extract sequences of events following corner kicks."""
        sequences = []

        for i, event in enumerate(self.events):
            # Check if this is a corner kick (SDK flattened format)
            event_type = event.get('type', '')
            pass_type = event.get('pass_type', '')

            if event_type == 'Pass' and isinstance(pass_type, str) and 'Corner' in pass_type:
                sequence = {
                    'corner': event,
                    'corner_index': i,
                    'following_events': []
                }

                # Get next events
                for j in range(1, min(window_size + 1, len(self.events) - i)):
                    sequence['following_events'].append(self.events[i + j])

                sequences.append(sequence)

        return sequences

    def analyze(self, num_matches: int = 10) -> Dict:
        """Complete analysis pipeline."""
        # Fetch data
        self.fetch_multiple_matches(num_matches)

        # Build transition matrix
        builder = TransitionMatrixBuilder()
        builder.build_from_events(self.events, track_corners=True)
        matrix = builder.calculate_probability_matrix()

        # Extract features
        extractor = FeatureExtractor()
        feature_summary = extractor.summarize_features(self.events)

        # Generate reports
        generator = ReportGenerator()

        return {
            'num_events': len(self.events),
            'num_matches': len(self.matches),
            'num_corners': len(self.identify_corner_kicks()),
            'transition_matrix': matrix,
            'feature_summary': feature_summary,
            'corner_sequences': self.extract_corner_sequences()
        }


class TransitionMatrixBuilder:
    """Builds transition probability matrices from event sequences."""

    def __init__(self):
        """Initialize the builder."""
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.event_counts = defaultdict(int)

    def add_transition(self, from_event: str, to_event: str):
        """Add a transition from one event to another."""
        self.transitions[from_event][to_event] += 1
        self.event_counts[from_event] += 1

    def build_from_events(self, events: List[Dict], track_corners: bool = False):
        """Build transition matrix from a sequence of events."""
        for i in range(len(events) - 1):
            curr_event = events[i]
            next_event = events[i + 1]

            # Get event types (SDK flattens to just 'type' field)
            curr_type = curr_event.get('type', 'Unknown')
            next_type = next_event.get('type', 'Unknown')

            # Handle nested structure if it exists (for backward compatibility)
            if isinstance(curr_type, dict):
                curr_type = curr_type.get('name', 'Unknown')
            if isinstance(next_type, dict):
                next_type = next_type.get('name', 'Unknown')

            # Special handling for corners (SDK uses 'pass_type' field)
            if track_corners and curr_type == 'Pass':
                pass_type = curr_event.get('pass_type', '')
                if isinstance(pass_type, str) and 'Corner' in pass_type:
                    curr_type = 'Corner'

            self.add_transition(curr_type, next_type)

    def calculate_probability_matrix(self) -> pd.DataFrame:
        """Calculate probability matrix from counts."""
        # Get all unique events
        all_events = set()
        for from_event in self.transitions:
            all_events.add(from_event)
            for to_event in self.transitions[from_event]:
                all_events.add(to_event)

        # Create matrix
        all_events = sorted(all_events)
        matrix = pd.DataFrame(0.0, index=all_events, columns=all_events)

        # Calculate probabilities
        for from_event in self.transitions:
            total = sum(self.transitions[from_event].values())
            if total > 0:
                for to_event in self.transitions[from_event]:
                    matrix.loc[from_event, to_event] = (
                        self.transitions[from_event][to_event] / total
                    )

        return matrix

    def get_corner_transitions(self) -> Dict:
        """Get transitions specifically from corner kicks."""
        corner_transitions = self.transitions.get('Corner', {})
        total = sum(corner_transitions.values())

        if total == 0:
            return {}

        return {
            event: count / total
            for event, count in corner_transitions.items()
        }


class FeatureExtractor:
    """Extracts and documents features from raw StatsBomb data."""

    def __init__(self):
        """Initialize the extractor."""
        self.features = {}

    def extract_event_features(self, event: Dict) -> Dict:
        """Extract features from a single event."""
        # SDK flattens structure, so 'type' is just a string
        event_type = event.get('type', 'Unknown')
        if isinstance(event_type, dict):
            event_type = event_type.get('name', 'Unknown')

        features = {
            'event_type': event_type,
            'has_location': 'location' in event,
            'has_timestamp': 'timestamp' in event,
            'has_under_pressure': 'under_pressure' in event,
            'has_duration': 'duration' in event,
            'has_related_events': 'related_events' in event and len(event.get('related_events', [])) > 0
        }

        # Extract type-specific features (SDK flattens these as pass_*, shot_*, etc.)
        if event_type == 'Pass':
            features['pass_features'] = {
                'length': event.get('pass_length'),
                'angle': event.get('pass_angle'),
                'height': event.get('pass_height'),
                'has_end_location': 'pass_end_location' in event,
                'outcome': event.get('pass_outcome', 'Complete')
            }

        elif event_type == 'Shot':
            features['shot_features'] = {
                'statsbomb_xg': event.get('shot_statsbomb_xg'),
                'outcome': event.get('shot_outcome'),
                'technique': event.get('shot_technique'),
                'body_part': event.get('shot_body_part')
            }

        elif event_type == 'Clearance':
            features['clearance_features'] = {
                'aerial_won': event.get('clearance_aerial_won', False),
                'head': event.get('clearance_head', False),
                'body_part': event.get('clearance_body_part')
            }

        return features

    def summarize_features(self, events: List[Dict]) -> Dict:
        """Summarize features across all events."""
        summary = {
            'event_types': set(),
            'location_coverage': 0,
            'timestamp_coverage': 0,
            'pressure_rate': 0,
            'type_specific_features': defaultdict(set),
            'universal_fields': None,
            'field_coverage': defaultdict(int)
        }

        location_count = 0
        timestamp_count = 0
        pressure_count = 0

        for event in events:
            # Track event types (SDK format)
            event_type = event.get('type', 'Unknown')
            if isinstance(event_type, dict):
                event_type = event_type.get('name', 'Unknown')
            summary['event_types'].add(event_type)

            # Track field coverage
            for field in event.keys():
                summary['field_coverage'][field] += 1

            # Track specific features
            if 'location' in event:
                location_count += 1
            if 'timestamp' in event:
                timestamp_count += 1
            if event.get('under_pressure', False):
                pressure_count += 1

            # Type-specific features (SDK flattens these as pass_*, shot_*, etc.)
            if event_type == 'Pass':
                for key in event.keys():
                    if key.startswith('pass_'):
                        summary['type_specific_features']['pass'].add(key)

            if event_type == 'Shot':
                for key in event.keys():
                    if key.startswith('shot_'):
                        summary['type_specific_features']['shot'].add(key)

            if event_type == 'Clearance':
                for key in event.keys():
                    if key.startswith('clearance_'):
                        summary['type_specific_features']['clearance'].add(key)

        # Calculate coverage rates
        if events:
            summary['location_coverage'] = location_count / len(events)
            summary['timestamp_coverage'] = timestamp_count / len(events)
            summary['pressure_rate'] = pressure_count / len(events)

        # Convert sets to lists for JSON serialization
        summary['event_types'] = list(summary['event_types'])
        for key in summary['type_specific_features']:
            summary['type_specific_features'][key] = list(summary['type_specific_features'][key])

        return summary


class ReportGenerator:
    """Generates human-readable reports from analysis results."""

    def __init__(self):
        """Initialize the generator."""
        pass

    def generate_transition_report(self, matrix: pd.DataFrame, focus_event: str = None) -> str:
        """Generate a report on transition probabilities."""
        report = "# Transition Probability Report\n\n"

        if focus_event and focus_event in matrix.index:
            report += f"## Transitions from {focus_event}\n\n"

            transitions = matrix.loc[focus_event]
            transitions = transitions[transitions > 0].sort_values(ascending=False)

            for event, prob in transitions.items():
                report += f"- {event}: {prob:.3f}\n"

        report += "\n## Top Overall Transitions\n\n"

        # Find highest probability transitions
        top_transitions = []
        for from_event in matrix.index:
            for to_event in matrix.columns:
                prob = matrix.loc[from_event, to_event]
                if prob > 0.1:
                    top_transitions.append((from_event, to_event, prob))

        top_transitions.sort(key=lambda x: x[2], reverse=True)

        for from_e, to_e, prob in top_transitions[:10]:
            report += f"- {from_e} → {to_e}: {prob:.3f}\n"

        return report

    def generate_feature_report(self, feature_summary: Dict) -> str:
        """Generate a report on available features."""
        report = "# Feature Documentation Report\n\n"

        report += "## Event Types Found\n\n"
        for event_type in sorted(feature_summary.get('event_types', [])):
            report += f"- {event_type}\n"

        report += "\n## Data Coverage\n\n"
        report += f"- Location Coverage: {feature_summary.get('location_coverage', 0) * 100:.1f}%\n"
        report += f"- Timestamp Coverage: {feature_summary.get('timestamp_coverage', 0) * 100:.1f}%\n"
        report += f"- Pressure Rate: {feature_summary.get('pressure_rate', 0) * 100:.1f}%\n"

        report += "\n## Type-Specific Features\n\n"
        for event_type, features in feature_summary.get('type_specific_features', {}).items():
            report += f"\n### {event_type.title()} Features\n"
            for feature in sorted(features):
                report += f"- {feature}\n"

        return report

    def generate_complete_report(self, analysis_results: Dict) -> str:
        """Generate a complete analysis report."""
        report = "# StatsBomb Raw Data Analysis Report\n\n"

        # Overview
        report += "## Overview\n\n"
        report += f"- Matches Analyzed: {analysis_results.get('num_matches', 0)}\n"
        report += f"- Total Events: {analysis_results.get('num_events', 0)}\n"
        report += f"- Corner Kicks: {analysis_results.get('num_corners', 0)}\n\n"

        # Transition analysis
        if 'transition_matrix' in analysis_results:
            matrix = analysis_results['transition_matrix']
            report += self.generate_transition_report(matrix, 'Corner')

        # Feature documentation
        if 'feature_summary' in analysis_results:
            report += "\n" + self.generate_feature_report(analysis_results['feature_summary'])

        return report


def main():
    """Main execution function."""
    print("Starting StatsBomb Raw Data Analysis...")

    # Initialize analyzer
    analyzer = StatsBombRawAnalyzer()

    # Run analysis
    results = analyzer.analyze(num_matches=5)

    # Generate report
    generator = ReportGenerator()
    report = generator.generate_complete_report(results)

    # Save outputs
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    with open(output_dir / "statsbomb_raw_analysis_report.md", "w") as f:
        f.write(report)

    # Save transition matrix
    if 'transition_matrix' in results:
        results['transition_matrix'].to_csv(output_dir / "transition_matrix_complete.csv")

    print(f"\nAnalysis complete!")
    print(f"- Report saved to: {output_dir / 'statsbomb_raw_analysis_report.md'}")
    print(f"- Matrix saved to: {output_dir / 'transition_matrix_complete.csv'}")

    return results


if __name__ == "__main__":
    main()