#!/usr/bin/env python3
"""
Phase 0: USSF Data Inspection
=============================
Download and analyze the USSF counterattack dataset to understand feature distributions
before implementing transfer learning to DFL corner kick data.

Output: ussf_feature_distribution_report.md
"""

import pickle
import numpy as np
import requests
import os
from pathlib import Path
from collections import Counter

# Configuration
DATA_DIR = Path(__file__).parent / "data"
REPORT_DIR = Path(__file__).parent / "reports"
COMBINED_PKL = DATA_DIR / "combined.pkl"
S3_URL = "https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/combined.pkl"


def download_data():
    """Download combined.pkl from USSF S3 bucket."""
    if COMBINED_PKL.exists():
        print(f"✓ Data already exists at {COMBINED_PKL}")
        return

    print(f"Downloading from {S3_URL}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(S3_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(COMBINED_PKL, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\rDownloading: {pct:.1f}% ({downloaded / 1e6:.1f} MB)", end='')

    print(f"\n✓ Downloaded to {COMBINED_PKL}")


def load_data():
    """Load the pickle file."""
    print(f"Loading {COMBINED_PKL}...")
    with open(COMBINED_PKL, 'rb') as f:
        data = pickle.load(f)
    return data


def analyze_structure(data):
    """Analyze top-level structure of the data."""
    report = []
    report.append("# USSF Counterattack Dataset: Feature Distribution Report\n")
    report.append("## 1. Data Structure Overview\n")

    report.append(f"**Data type:** `{type(data).__name__}`\n")
    report.append(f"**Top-level keys:** `{list(data.keys())}`\n")

    # Analyze each adjacency type
    for key in data.keys():
        if key == 'binary':
            continue
        report.append(f"\n### Adjacency Type: `{key}`\n")
        if isinstance(data[key], dict):
            report.append(f"**Sub-keys:** `{list(data[key].keys())}`\n")
            for subkey in data[key].keys():
                val = data[key][subkey]
                if isinstance(val, list):
                    report.append(f"- `{subkey}`: List of {len(val)} items\n")
                    if len(val) > 0:
                        first = val[0]
                        if hasattr(first, 'shape'):
                            report.append(f"  - First item shape: `{first.shape}`\n")
                        else:
                            report.append(f"  - First item type: `{type(first).__name__}`\n")

    return report


def analyze_node_features(data, adj_type='normal'):
    """Analyze node feature distributions."""
    report = []
    report.append("\n## 2. Node Feature Distributions\n")

    # Feature names (from USSF documentation)
    feature_names = [
        'x', 'y', 'vx', 'vy', 'velocity_mag', 'velocity_angle',
        'dist_goal', 'angle_goal', 'dist_ball', 'angle_ball',
        'attacking_team_flag', 'potential_receiver'
    ]

    # Get node feature key
    x_key = f"{adj_type}_x" if f"{adj_type}_x" in data[adj_type] else 'x'
    if x_key not in data[adj_type]:
        # Try alternative naming
        for k in data[adj_type].keys():
            if 'x' in k.lower():
                x_key = k
                break

    node_features = data[adj_type].get(x_key, data[adj_type].get('x', None))

    if node_features is None:
        report.append(f"**Warning:** Could not find node features in `data['{adj_type}']`\n")
        report.append(f"Available keys: `{list(data[adj_type].keys())}`\n")
        return report

    # Stack all node features
    all_nodes = np.vstack(node_features)
    n_features = all_nodes.shape[1]

    report.append(f"**Total graphs:** {len(node_features)}\n")
    report.append(f"**Total nodes across all graphs:** {all_nodes.shape[0]}\n")
    report.append(f"**Features per node:** {n_features}\n")

    # Adjust feature names if needed
    if n_features != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(n_features)]
        report.append(f"\n**Note:** Expected 12 features but found {n_features}. Using generic names.\n")

    report.append("\n| Feature | Mean | Std | Min | Max | Description |\n")
    report.append("|---------|------|-----|-----|-----|-------------|\n")

    descriptions = {
        'x': 'X position on pitch',
        'y': 'Y position on pitch',
        'vx': 'X velocity component',
        'vy': 'Y velocity component',
        'velocity_mag': 'Speed (magnitude of velocity)',
        'velocity_angle': 'Direction of movement (radians)',
        'dist_goal': 'Euclidean distance to goal',
        'angle_goal': 'Angle toward goal',
        'dist_ball': 'Euclidean distance to ball',
        'angle_ball': 'Angle toward ball',
        'attacking_team_flag': 'Binary: 1=attacking, 0=defending',
        'potential_receiver': 'Binary: potential pass receiver'
    }

    feature_stats = {}
    for i, name in enumerate(feature_names):
        vals = all_nodes[:, i]
        mean = vals.mean()
        std = vals.std()
        vmin = vals.min()
        vmax = vals.max()
        desc = descriptions.get(name, '')

        feature_stats[name] = {'mean': mean, 'std': std, 'min': vmin, 'max': vmax}
        report.append(f"| {name} | {mean:.4f} | {std:.4f} | {vmin:.4f} | {vmax:.4f} | {desc} |\n")

    # Additional analysis: coordinate system detection
    report.append("\n### Coordinate System Analysis\n")

    x_vals = all_nodes[:, 0]
    y_vals = all_nodes[:, 1]

    if x_vals.max() <= 1.0 and y_vals.max() <= 1.0:
        report.append("**Coordinate system:** Normalized [0, 1] pitch-relative coordinates\n")
    elif x_vals.max() > 100:
        report.append("**Coordinate system:** Meters (likely 105x68 pitch)\n")
    else:
        report.append(f"**Coordinate system:** Unknown (x range: [{x_vals.min():.2f}, {x_vals.max():.2f}], y range: [{y_vals.min():.2f}, {y_vals.max():.2f}])\n")

    return report, feature_stats


def analyze_edge_features(data, adj_type='normal'):
    """Analyze edge feature distributions."""
    report = []
    report.append("\n## 3. Edge Feature Distributions\n")

    edge_feature_names = [
        'player_distance', 'speed_difference',
        'positional_sine_angle', 'positional_cosine_angle',
        'velocity_sine_angle', 'velocity_cosine_angle'
    ]

    # Get edge feature key
    e_key = f"{adj_type}_e" if f"{adj_type}_e" in data[adj_type] else 'e'
    if e_key not in data[adj_type]:
        for k in data[adj_type].keys():
            if 'e' in k.lower() and 'edge' not in k.lower():
                e_key = k
                break

    edge_features = data[adj_type].get(e_key, data[adj_type].get('e', None))

    if edge_features is None:
        report.append(f"**Warning:** Could not find edge features in `data['{adj_type}']`\n")
        return report

    # Stack all edge features
    all_edges = np.vstack(edge_features)
    n_features = all_edges.shape[1]

    report.append(f"**Total edges across all graphs:** {all_edges.shape[0]}\n")
    report.append(f"**Features per edge:** {n_features}\n")

    if n_features != len(edge_feature_names):
        edge_feature_names = [f"edge_feature_{i}" for i in range(n_features)]

    report.append("\n| Feature | Mean | Std | Min | Max |\n")
    report.append("|---------|------|-----|-----|-----|\n")

    for i, name in enumerate(edge_feature_names):
        vals = all_edges[:, i]
        report.append(f"| {name} | {vals.mean():.4f} | {vals.std():.4f} | {vals.min():.4f} | {vals.max():.4f} |\n")

    return report


def analyze_adjacency(data):
    """Analyze adjacency matrix structure across types."""
    report = []
    report.append("\n## 4. Adjacency Matrix Analysis\n")

    adj_types = [k for k in data.keys() if k != 'binary']
    report.append(f"**Adjacency types available:** `{adj_types}`\n")

    report.append("\n| Adjacency Type | Avg Nodes | Avg Edges | Avg Density | Description |\n")
    report.append("|----------------|-----------|-----------|-------------|-------------|\n")

    descriptions = {
        'normal': 'Team-based connectivity through ball',
        'delaunay': 'Delaunay triangulation',
        'dense': 'Fully connected (all-to-all)',
        'dense_ap': 'Attackers fully connected + defenders',
        'dense_dp': 'Defenders fully connected + attackers'
    }

    for adj_type in adj_types:
        a_key = f"{adj_type}_a" if f"{adj_type}_a" in data[adj_type] else 'a'
        if a_key not in data[adj_type]:
            for k in data[adj_type].keys():
                if 'a' in k.lower():
                    a_key = k
                    break

        adj_matrices = data[adj_type].get(a_key, data[adj_type].get('a', None))
        if adj_matrices is None:
            continue

        nodes_list = []
        edges_list = []
        densities = []

        for adj in adj_matrices:
            n = adj.shape[0]
            edges = adj.sum()
            density = edges / (n * n) if n > 0 else 0
            nodes_list.append(n)
            edges_list.append(edges)
            densities.append(density)

        desc = descriptions.get(adj_type, '')
        report.append(f"| {adj_type} | {np.mean(nodes_list):.1f} | {np.mean(edges_list):.1f} | {np.mean(densities):.3f} | {desc} |\n")

    return report


def analyze_graph_sizes(data, adj_type='normal'):
    """Analyze variable graph sizes."""
    report = []
    report.append("\n## 5. Graph Size Distribution\n")

    x_key = f"{adj_type}_x" if f"{adj_type}_x" in data[adj_type] else 'x'
    if x_key not in data[adj_type]:
        for k in data[adj_type].keys():
            if 'x' in k.lower():
                x_key = k
                break

    node_features = data[adj_type].get(x_key, data[adj_type].get('x', None))
    if node_features is None:
        return report

    sizes = [x.shape[0] for x in node_features]

    report.append(f"**Min players per graph:** {min(sizes)}\n")
    report.append(f"**Max players per graph:** {max(sizes)}\n")
    report.append(f"**Mean players per graph:** {np.mean(sizes):.1f}\n")
    report.append(f"**Std players per graph:** {np.std(sizes):.1f}\n")

    # Distribution
    size_counts = Counter(sizes)
    report.append("\n| Players | Count | Percentage |\n")
    report.append("|---------|-------|------------|\n")
    for size in sorted(size_counts.keys()):
        count = size_counts[size]
        pct = (count / len(sizes)) * 100
        report.append(f"| {size} | {count} | {pct:.1f}% |\n")

    return report


def analyze_labels(data):
    """Analyze class balance."""
    report = []
    report.append("\n## 6. Class Balance (Labels)\n")

    labels = np.array(data['binary']).flatten()

    report.append(f"**Total samples:** {len(labels)}\n")
    report.append(f"**Positive (successful counterattacks):** {labels.sum()} ({labels.mean()*100:.1f}%)\n")
    report.append(f"**Negative (unsuccessful):** {len(labels) - labels.sum()} ({(1-labels.mean())*100:.1f}%)\n")

    return report


def main():
    """Run Phase 0 inspection."""
    # Download data
    download_data()

    # Load data
    data = load_data()

    # Build report
    full_report = []

    # Structure analysis
    full_report.extend(analyze_structure(data))

    # Node features
    node_report, feature_stats = analyze_node_features(data, 'normal')
    full_report.extend(node_report)

    # Edge features
    full_report.extend(analyze_edge_features(data, 'normal'))

    # Adjacency analysis
    full_report.extend(analyze_adjacency(data))

    # Graph sizes
    full_report.extend(analyze_graph_sizes(data, 'normal'))

    # Labels
    full_report.extend(analyze_labels(data))

    # Compatibility checklist for DFL
    full_report.append("\n## 7. DFL Feature Engineering Checklist\n")
    full_report.append("\nBased on this analysis, DFL corner kick features must match:\n\n")
    full_report.append("- [ ] Coordinate system: Check if pitch-relative [0,1] or meters\n")
    full_report.append("- [ ] Velocity units: Match range and scale\n")
    full_report.append("- [ ] Goal position convention: Same reference point\n")
    full_report.append("- [ ] Angle conventions: atan2(y, x) vs atan2(x, y)\n")
    full_report.append("- [ ] Binary flags: Same encoding (0/1)\n")
    full_report.append("- [ ] Handle `potential_receiver`: Drop or set to 0\n")

    # Add date
    from datetime import datetime
    full_report.append(f"\n---\n\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Write report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "ussf_feature_distribution_report.md"

    with open(report_path, 'w') as f:
        f.write(''.join(full_report))

    print(f"\n✓ Report written to {report_path}")

    # Also print to stdout
    print("\n" + "="*60)
    print(''.join(full_report))

    return data, feature_stats


if __name__ == "__main__":
    data, stats = main()
