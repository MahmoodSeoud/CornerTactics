#!/usr/bin/env python3
"""
Analyze what happens after corner kicks: P(a_{t+1} | corner_t)
Works with raw JSON files to preserve ALL features.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm


class CornerTransitionAnalyzer:
    """Analyze transitions after corner kicks with COMPLETE data preservation."""

    def __init__(self, json_dir: str = "data/raw/statsbomb/json_events"):
        self.json_dir = Path(json_dir)
        self.events_dir = self.json_dir / "events"
        self.corner_sequences = []
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.all_features = set()  # Track ALL unique features we find

    def load_match_events(self, match_id: int) -> List[Dict]:
        """Load events from a match JSON file."""
        event_file = self.events_dir / f"{match_id}.json"
        if event_file.exists():
            with open(event_file) as f:
                return json.load(f)
        return []

    def is_corner_kick(self, event: Dict) -> bool:
        """Check if an event is a corner kick."""
        # Check type is Pass
        if event.get('type', {}).get('name') != 'Pass':
            return False

        # Check pass type is Corner
        pass_obj = event.get('pass', {})
        pass_type = pass_obj.get('type', {})
        if isinstance(pass_type, dict) and pass_type.get('name') == 'Corner':
            return True

        return False

    def extract_all_features(self, event: Dict) -> Dict:
        """Extract EVERY feature from an event - no filtering."""
        features = {}

        # Recursively flatten the entire event structure
        def flatten_dict(d, prefix=''):
            for key, value in d.items():
                new_key = f"{prefix}{key}" if prefix else key

                if isinstance(value, dict):
                    # Recurse into nested dicts
                    flatten_dict(value, f"{new_key}.")
                elif isinstance(value, list):
                    # Store list length and first item if exists
                    features[f"{new_key}_count"] = len(value)
                    if value and isinstance(value[0], dict):
                        # Sample first item of list
                        flatten_dict(value[0], f"{new_key}[0].")
                else:
                    # Store the actual value
                    features[new_key] = value
                    self.all_features.add(new_key)  # Track this feature

        flatten_dict(event)
        return features

    def get_event_summary(self, event: Dict) -> str:
        """Get a human-readable summary of an event."""
        event_type = event.get('type', {}).get('name', 'Unknown')

        # Add context based on event type
        if event_type == 'Pass':
            pass_obj = event.get('pass', {})
            pass_type = pass_obj.get('type', {}).get('name', '')
            outcome = pass_obj.get('outcome', {}).get('name', 'Complete')
            recipient = pass_obj.get('recipient', {}).get('name', 'Unknown')
            return f"Pass({pass_type}) -> {recipient} ({outcome})"

        elif event_type == 'Shot':
            shot_obj = event.get('shot', {})
            outcome = shot_obj.get('outcome', {}).get('name', 'Unknown')
            xg = shot_obj.get('statsbomb_xg', 0)
            return f"Shot({outcome}, xG={xg:.2f})"

        elif event_type == 'Clearance':
            return f"Clearance"

        elif event_type == 'Ball Receipt*':
            return f"Ball Receipt"

        elif event_type == 'Carry':
            return f"Carry"

        elif event_type == 'Duel':
            duel_obj = event.get('duel', {})
            duel_type = duel_obj.get('type', {}).get('name', 'Unknown')
            outcome = duel_obj.get('outcome', {}).get('name', 'Unknown')
            return f"Duel({duel_type}): {outcome}"

        else:
            return event_type

    def analyze_corner_sequences(self, max_events_after: int = 10):
        """Analyze what happens after each corner kick."""

        # Load matches with corners
        corner_matches_file = self.json_dir / "matches_with_corners.csv"
        if corner_matches_file.exists():
            matches_df = pd.read_csv(corner_matches_file)
            match_ids = matches_df['match_id'].tolist()
        else:
            # Use all matches
            event_files = list(self.events_dir.glob("*.json"))
            match_ids = [int(f.stem) for f in event_files]

        print(f"Analyzing {len(match_ids)} matches...")

        for match_id in tqdm(match_ids, desc="Processing matches"):
            events = self.load_match_events(match_id)

            for i, event in enumerate(events):
                if self.is_corner_kick(event):
                    # Extract corner features
                    corner_features = self.extract_all_features(event)

                    # Get sequence of following events
                    sequence = {
                        'match_id': match_id,
                        'corner_index': i,
                        'corner_event': event,
                        'corner_features': corner_features,
                        'corner_team': event.get('team', {}).get('name'),
                        'corner_player': event.get('player', {}).get('name'),
                        'corner_minute': event.get('minute'),
                        'corner_second': event.get('second'),
                        'following_events': []
                    }

                    # Collect next events (up to max_events_after)
                    for j in range(1, min(max_events_after + 1, len(events) - i)):
                        next_event = events[i + j]
                        next_features = self.extract_all_features(next_event)

                        sequence['following_events'].append({
                            'event': next_event,
                            'features': next_features,
                            'summary': self.get_event_summary(next_event),
                            'time_diff': (next_event.get('minute', 0) * 60 + next_event.get('second', 0)) -
                                       (event.get('minute', 0) * 60 + event.get('second', 0))
                        })

                        # Track immediate transition (t+1)
                        if j == 1:
                            from_type = "Corner"
                            to_type = self.get_event_summary(next_event)
                            self.transition_counts[from_type][to_type] += 1

                    self.corner_sequences.append(sequence)

    def compute_transition_matrix(self) -> pd.DataFrame:
        """Compute P(a_{t+1} | corner_t) as a probability matrix."""

        # Calculate probabilities
        transition_probs = {}
        for from_event, to_events in self.transition_counts.items():
            total = sum(to_events.values())
            transition_probs[from_event] = {
                to_event: count / total
                for to_event, count in to_events.items()
            }

        # Convert to DataFrame
        df = pd.DataFrame(transition_probs).T.fillna(0)
        return df

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("CORNER KICK TRANSITION ANALYSIS")
        report.append("What happens after a corner? P(a_{t+1} | corner_t)")
        report.append("=" * 80)

        # Overall statistics
        report.append(f"\n📊 Dataset Statistics:")
        report.append(f"   Total corner kicks analyzed: {len(self.corner_sequences)}")
        report.append(f"   Unique features tracked: {len(self.all_features)}")

        # Top transitions after corners
        report.append(f"\n🎯 Most Common Next Events After Corners:")
        corner_transitions = self.transition_counts.get('Corner', {})
        total_corners = sum(corner_transitions.values())

        for event, count in sorted(corner_transitions.items(), key=lambda x: x[1], reverse=True)[:10]:
            prob = count / total_corners
            report.append(f"   {event}: {prob:.1%} ({count} times)")

        # Time analysis
        report.append(f"\n⏱️ Temporal Analysis:")
        time_to_outcome = defaultdict(list)

        for seq in self.corner_sequences:
            for event in seq['following_events']:
                summary = event['summary']
                time_diff = event['time_diff']
                if 'Shot' in summary and time_diff >= 0:
                    time_to_outcome['Shot'].append(time_diff)
                elif 'Goal' in summary and time_diff >= 0:
                    time_to_outcome['Goal'].append(time_diff)

        for outcome, times in time_to_outcome.items():
            if times:
                report.append(f"   Average time to {outcome}: {np.mean(times):.1f} seconds")

        # Success metrics
        report.append(f"\n✅ Corner Outcomes:")
        shot_within_10s = 0
        goal_within_20s = 0
        cleared_immediately = 0

        for seq in self.corner_sequences:
            if seq['following_events']:
                first_event = seq['following_events'][0]['summary']
                if 'Clearance' in first_event:
                    cleared_immediately += 1

                for event in seq['following_events']:
                    if event['time_diff'] <= 10 and 'Shot' in event['summary']:
                        shot_within_10s += 1
                        break
                    if event['time_diff'] <= 20 and 'Goal' in event['summary']:
                        goal_within_20s += 1
                        break

        if self.corner_sequences:
            report.append(f"   Cleared immediately: {cleared_immediately/len(self.corner_sequences):.1%}")
            report.append(f"   Shot within 10s: {shot_within_10s/len(self.corner_sequences):.1%}")
            report.append(f"   Goal within 20s: {goal_within_20s/len(self.corner_sequences):.1%}")

        # Feature discovery
        report.append(f"\n🔍 Unique Features Found (Sample):")
        for feature in sorted(list(self.all_features)[:20]):
            report.append(f"   - {feature}")

        return "\n".join(report)

    def save_results(self, output_dir: str = "data/analysis"):
        """Save all analysis results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save transition matrix
        matrix = self.compute_transition_matrix()
        matrix.to_csv(output_path / "corner_transition_matrix.csv")

        # Save detailed sequences (JSON)
        with open(output_path / "corner_sequences_detailed.json", "w") as f:
            # Convert to serializable format
            sequences_serializable = []
            for seq in self.corner_sequences[:100]:  # First 100 for file size
                seq_copy = seq.copy()
                # Keep only essential fields for JSON
                seq_copy['following_events'] = [
                    {'summary': e['summary'], 'time_diff': e['time_diff']}
                    for e in seq['following_events']
                ]
                sequences_serializable.append(seq_copy)
            json.dump(sequences_serializable, f, indent=2)

        # Save report
        report = self.generate_report()
        with open(output_path / "corner_transition_report.md", "w") as f:
            f.write(report)

        print(f"✅ Results saved to {output_path}")


def main():
    """Run the complete corner transition analysis."""
    print("🎯 Corner Transition Analyzer")
    print("Analyzing: P(a_{t+1} | corner_t)")
    print("-" * 40)

    # Initialize analyzer
    analyzer = CornerTransitionAnalyzer()

    # Run analysis
    analyzer.analyze_corner_sequences(max_events_after=15)

    # Save results
    analyzer.save_results()

    # Print summary
    print(analyzer.generate_report())


if __name__ == "__main__":
    main()