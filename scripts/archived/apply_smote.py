#!/usr/bin/env python3
"""
Apply SMOTE to generate synthetic dangerous corner examples.
Works on node features before graph construction.
"""
import pickle
import numpy as np
from pathlib import Path
from imblearn.over_sampling import SMOTE
from collections import Counter
import copy


def apply_smote_to_graphs(input_path, output_path, target_ratio=0.3):
    """
    Apply SMOTE to increase minority class (dangerous corners).

    Args:
        input_path: Path to original graphs
        output_path: Path to save augmented graphs
        target_ratio: Target ratio for positive class (default 30%)
    """
    print("Loading graphs...")
    with open(input_path, 'rb') as f:
        graphs = pickle.load(f)

    # Separate by class based on outcome_label (Shot or Goal = dangerous)
    dangerous_graphs = []
    safe_graphs = []

    for g in graphs:
        if hasattr(g, 'outcome_label') and g.outcome_label in ['Shot', 'Goal']:
            dangerous_graphs.append(g)
        elif hasattr(g, 'outcome_label') and g.outcome_label:
            safe_graphs.append(g)

    print(f"Original distribution:")
    print(f"  Safe: {len(safe_graphs)} ({len(safe_graphs)/len(graphs)*100:.1f}%)")
    print(f"  Dangerous: {len(dangerous_graphs)} ({len(dangerous_graphs)/len(graphs)*100:.1f}%)")

    # Extract feature matrices
    print("\nExtracting features for SMOTE...")

    # Get average node features per graph (since graphs have variable nodes)
    def get_graph_embedding(graph):
        """Create fixed-size embedding from variable-size graph"""
        features = graph.node_features  # Use node_features attribute
        # Use mean and std of node features as graph embedding
        mean_features = np.mean(features, axis=0)
        std_features = np.std(features, axis=0)
        max_features = np.max(features, axis=0)
        min_features = np.min(features, axis=0)
        return np.concatenate([mean_features, std_features, max_features, min_features])

    # Create feature matrix
    dangerous_embeddings = np.array([get_graph_embedding(g) for g in dangerous_graphs])
    safe_embeddings = np.array([get_graph_embedding(g) for g in safe_graphs])

    X = np.vstack([safe_embeddings, dangerous_embeddings])
    y = np.array([0] * len(safe_graphs) + [1] * len(dangerous_graphs))

    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: {Counter(y)}")

    # Apply SMOTE
    print(f"\nApplying SMOTE to achieve {target_ratio*100:.1f}% positive class...")

    # Calculate sampling strategy
    n_safe = len(safe_graphs)
    n_dangerous_target = int(n_safe * target_ratio / (1 - target_ratio))

    smote = SMOTE(
        sampling_strategy={0: n_safe, 1: n_dangerous_target},
        random_state=42,
        k_neighbors=min(5, len(dangerous_graphs) - 1)  # Adjust k_neighbors if too few samples
    )

    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"After SMOTE: {Counter(y_resampled)}")

    # Generate synthetic graphs from new embeddings
    print("\nGenerating synthetic graphs...")

    # Find new synthetic samples
    n_original = len(X)
    synthetic_embeddings = X_resampled[n_original:]

    # Create synthetic graphs by finding nearest dangerous graph and perturbing
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(dangerous_embeddings)

    synthetic_graphs = []
    for i, embed in enumerate(synthetic_embeddings):
        # Find nearest dangerous graphs
        distances, indices = nn.kneighbors([embed[:len(embed)//4]])  # Use mean features

        # Use nearest graph as template
        template_graph = dangerous_graphs[indices[0][0]]

        # Create synthetic graph (copy structure, perturb features slightly)
        synthetic = copy.deepcopy(template_graph)

        # Add small noise to node features
        noise_scale = 0.1
        features = synthetic.node_features
        noise = np.random.normal(0, noise_scale, features.shape)
        synthetic.node_features = features + noise

        # Mark as synthetic for tracking
        synthetic.is_synthetic = True
        synthetic.corner_id = f"{synthetic.corner_id}_synthetic_{i}"

        synthetic_graphs.append(synthetic)

    # Combine all graphs
    augmented_graphs = graphs + synthetic_graphs

    print(f"\nFinal dataset:")
    print(f"  Original graphs: {len(graphs)}")
    print(f"  Synthetic graphs: {len(synthetic_graphs)}")
    print(f"  Total graphs: {len(augmented_graphs)}")

    # Verify new distribution
    new_dangerous = len(dangerous_graphs) + len(synthetic_graphs)
    new_total = len(augmented_graphs)
    print(f"  New positive rate: {new_dangerous/new_total*100:.1f}%")

    # Save augmented dataset
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(augmented_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✅ SMOTE augmentation complete!")

    return len(synthetic_graphs)


if __name__ == "__main__":
    input_path = Path("data/graphs/adjacency_team/combined_temporal_graphs.pkl")
    output_path = Path("data/graphs/adjacency_team/combined_temporal_graphs_smote.pkl")

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        exit(1)

    apply_smote_to_graphs(input_path, output_path, target_ratio=0.35)