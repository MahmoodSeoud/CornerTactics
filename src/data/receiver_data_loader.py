#!/usr/bin/env python3
"""
Receiver Data Loader Module for TacticAI-Style Corner Prediction

Implements Day 3-4: Data Loader Extension
Extends CornerDataset to include receiver prediction labels and shot labels.

Based on TacticAI Implementation Plan:
- Load graphs with receiver labels (player who receives ball 0-5s after corner)
- Mask velocity features (vx, vy = 0) to acknowledge missing data
- Add receiver_label field: torch.LongTensor (0-21)
- Add shot_label field: torch.FloatTensor (1.0 if dangerous else 0.0)
- Skip corners without valid receiver labels

Author: mseo
Date: October 2024
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch_geometric.data import Data, DataLoader
import warnings

# Import parent data loader
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import CornerDataset
from src.graph_builder import CornerGraph


class ReceiverCornerDataset(CornerDataset):
    """
    Extended dataset for TacticAI-style receiver prediction and outcome classification.

    Extends CornerDataset to:
    1. Mask velocity features (vx, vy = 0)
    2. Add receiver_label (0-21 node index)
    3. Add shot_label (1.0 if shot/goal, 0.0 otherwise)
    4. Add outcome_class_label (0=Goal, 1=Shot, 2=Clearance, 3=Possession)
    5. Filter out corners without receiver labels
    """

    def __init__(self, graph_path: str, mask_velocities: bool = True):
        """
        Initialize the receiver prediction dataset.

        Args:
            graph_path: Path to pickled graph file with receiver labels
            mask_velocities: If True, set vx, vy features to 0 (acknowledge missing data)
        """
        self.mask_velocities = mask_velocities

        # Load graphs (don't call parent __init__ yet)
        print(f"Loading graphs from {graph_path}")
        with open(graph_path, 'rb') as f:
            all_graphs = pickle.load(f)
        print(f"Loaded {len(all_graphs)} total graphs")

        # Filter: Keep only graphs with receiver labels
        self.graphs = [g for g in all_graphs if g.receiver_node_index is not None]
        print(f"Filtered to {len(self.graphs)} graphs with receiver labels "
              f"({len(self.graphs)/len(all_graphs)*100:.1f}% coverage)")

        if len(self.graphs) == 0:
            raise ValueError("No graphs with receiver labels found! "
                           "Run scripts/preprocessing/add_receiver_labels.py first")

        # Store path for parent compatibility
        self.graph_path = graph_path
        self.outcome_type = "receiver"  # Custom outcome type

        # Convert to PyG data with receiver labels
        self.data_list = self._convert_to_pyg_data_with_receiver()

        # Compute statistics
        self._compute_statistics()

    def _convert_to_pyg_data_with_receiver(self) -> List[Data]:
        """
        Convert CornerGraph objects to PyTorch Geometric Data objects
        with receiver, shot, and outcome class labels.

        Returns:
            List of PyG Data objects with receiver_label, shot_label, and outcome_class_label
        """
        data_list = []

        for graph in self.graphs:
            # Convert node features to tensor
            x = torch.FloatTensor(graph.node_features)

            # MASK VELOCITIES: Set vx (column 4) and vy (column 5) to 0
            if self.mask_velocities:
                x[:, 4:6] = 0.0

            # Convert edge index to tensor
            edge_index = torch.LongTensor(graph.edge_index)

            # Convert edge features to tensor
            edge_attr = None
            if graph.edge_features is not None and len(graph.edge_features) > 0:
                edge_attr = torch.FloatTensor(graph.edge_features)

            # RECEIVER LABEL: Node index (0-21) of receiver
            receiver_label = torch.LongTensor([graph.receiver_node_index])

            # SHOT LABEL: Binary (1.0 if shot/goal, 0.0 otherwise)
            is_dangerous = (graph.outcome_label == "Shot") or graph.goal_scored
            shot_label = torch.FloatTensor([1.0 if is_dangerous else 0.0])

            # OUTCOME CLASS LABEL: Multi-class (0=Goal, 1=Shot, 2=Clearance, 3=Possession)
            outcome_mapping = {
                "Goal": 0,
                "Shot": 1,
                "Clearance": 2,
                "Possession": 3,
                "Loss": 3  # Merge Loss into Possession
            }
            outcome_class = outcome_mapping.get(graph.outcome_label, 3)  # Default to Possession
            outcome_class_label = torch.LongTensor([outcome_class])

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                receiver_label=receiver_label,         # Receiver node index
                shot_label=shot_label,                 # Shot prediction label
                outcome_class_label=outcome_class_label,  # NEW: Multi-class outcome label
                corner_id=graph.corner_id,
                num_nodes=x.size(0)
            )

            # Add team information
            if graph.teams:
                team_labels = torch.LongTensor([
                    1 if team == 'attacking' else 0 for team in graph.teams
                ])
                data.team = team_labels

            # Add receiver metadata for analysis
            data.receiver_name = graph.receiver_player_name

            data_list.append(data)

        return data_list

    def _compute_statistics(self):
        """Compute and print dataset statistics."""
        self.num_graphs = len(self.data_list)
        self.num_features = self.data_list[0].x.size(1) if self.data_list else 0

        # Receiver distribution statistics
        receiver_indices = [data.receiver_label.item() for data in self.data_list]
        unique_receivers = len(set(receiver_indices))

        print(f"\nDataset: {self.num_graphs} corners with receiver labels")
        print(f"Node features: {self.num_features} dimensions "
              f"(velocities {'masked' if self.mask_velocities else 'included'})")

        # Shot label statistics
        positive_shots = sum(data.shot_label.item() for data in self.data_list)
        shot_rate = positive_shots / self.num_graphs if self.num_graphs > 0 else 0
        print(f"Dangerous situations (shot/goal): {int(positive_shots)} "
              f"({shot_rate*100:.1f}%)")

        # Node and edge statistics
        node_counts = [data.num_nodes for data in self.data_list]
        edge_counts = [data.edge_index.size(1) for data in self.data_list]

        self.avg_nodes = np.mean(node_counts)
        self.avg_edges = np.mean(edge_counts)

        print(f"Avg nodes per graph: {self.avg_nodes:.1f}")
        print(f"Avg edges per graph: {self.avg_edges:.1f}")

        # Receiver position distribution
        receiver_dist = {}
        for idx in receiver_indices:
            receiver_dist[idx] = receiver_dist.get(idx, 0) + 1
        print(f"Unique receiver positions: {unique_receivers}")

    def get_split_indices(self, test_size: float = 0.15,
                         val_size: float = 0.15,
                         random_state: int = 42) -> Dict[str, List[int]]:
        """
        Get train/val/test split indices by corner ID (not by graph).

        Override parent method to use shot_label instead of y.

        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'val', 'test' indices
        """
        from sklearn.model_selection import train_test_split

        # Helper function to extract base corner ID
        def get_base_corner_id(corner_id):
            """Extract base corner ID by removing temporal suffix (_t...)"""
            if '_t' in corner_id:
                return corner_id.split('_t')[0]
            return corner_id

        # Step 1: Create corner_id to graph_indices mapping
        corner_to_graphs = {}
        corner_to_label = {}

        for idx, data in enumerate(self.data_list):
            base_corner_id = get_base_corner_id(data.corner_id)

            if base_corner_id not in corner_to_graphs:
                corner_to_graphs[base_corner_id] = []
                # Use shot_label for stratification
                corner_to_label[base_corner_id] = data.shot_label.numpy()[0]
            corner_to_graphs[base_corner_id].append(idx)

        # Step 2: Get unique corners and their labels
        unique_corners = list(corner_to_graphs.keys())
        corner_labels = np.array([corner_to_label[c] for c in unique_corners])

        # Step 3: Split corners (not graphs!)
        train_val_corners, test_corners = train_test_split(
            unique_corners,
            test_size=test_size,
            random_state=random_state,
            stratify=corner_labels
        )

        # Second split: train vs val
        train_val_labels = np.array([corner_to_label[c] for c in train_val_corners])
        val_relative_size = val_size / (1 - test_size)
        train_corners, val_corners = train_test_split(
            train_val_corners,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=train_val_labels
        )

        # Step 4: Map corner_ids back to graph indices
        train_idx = [idx for c in train_corners for idx in corner_to_graphs[c]]
        val_idx = [idx for c in val_corners for idx in corner_to_graphs[c]]
        test_idx = [idx for c in test_corners for idx in corner_to_graphs[c]]

        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }

    def get_data_loaders(self, batch_size: int = 32,
                        test_size: float = 0.15,
                        val_size: float = 0.15,
                        random_state: int = 42,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.

        Uses corner-based splitting (not graph-based) to prevent temporal leakage.

        Args:
            batch_size: Batch size for training
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            num_workers: Number of workers for data loading

        Returns:
            train_loader, val_loader, test_loader
        """
        # Use parent's split method (prevents temporal leakage)
        splits = self.get_split_indices(test_size, val_size, random_state)

        # Create data subsets
        train_data = [self.data_list[i] for i in splits['train']]
        val_data = [self.data_list[i] for i in splits['val']]
        test_data = [self.data_list[i] for i in splits['test']]

        # Create data loaders
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        # Print split statistics
        print(f"\n{'='*60}")
        print("DATA SPLIT STATISTICS")
        print(f"{'='*60}")
        print(f"Train: {len(train_data):4d} ({len(train_data)/len(self.data_list)*100:.1f}%)")
        print(f"Val:   {len(val_data):4d} ({len(val_data)/len(self.data_list)*100:.1f}%)")
        print(f"Test:  {len(test_data):4d} ({len(test_data)/len(self.data_list)*100:.1f}%)")

        # Check shot rates in each split
        train_shot_rate = sum(d.shot_label.item() for d in train_data) / len(train_data)
        val_shot_rate = sum(d.shot_label.item() for d in val_data) / len(val_data)
        test_shot_rate = sum(d.shot_label.item() for d in test_data) / len(test_data)

        print(f"\nDangerous situation rates:")
        print(f"  Train: {train_shot_rate*100:.1f}%")
        print(f"  Val:   {val_shot_rate*100:.1f}%")
        print(f"  Test:  {test_shot_rate*100:.1f}%")
        print(f"{'='*60}\n")

        return train_loader, val_loader, test_loader


def load_receiver_dataset(
    graph_path: str = "data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl",
    batch_size: int = 32,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    mask_velocities: bool = True
) -> Tuple[ReceiverCornerDataset, DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to load receiver dataset and create data loaders.

    Args:
        graph_path: Path to graph pickle file with receiver labels
        batch_size: Batch size
        test_size: Test set fraction
        val_size: Validation set fraction
        random_state: Random seed
        mask_velocities: Whether to mask velocity features (vx, vy = 0)

    Returns:
        dataset, train_loader, val_loader, test_loader
    """
    dataset = ReceiverCornerDataset(graph_path, mask_velocities=mask_velocities)
    train_loader, val_loader, test_loader = dataset.get_data_loaders(
        batch_size=batch_size,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    return dataset, train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the receiver data loader
    print("="*60)
    print("TESTING RECEIVER CORNER DATASET LOADER")
    print("="*60)

    # Check if graph file exists
    graph_path = Path("data/graphs/adjacency_team/statsbomb_temporal_augmented_with_receiver.pkl")
    if not graph_path.exists():
        print(f"❌ Graph file not found: {graph_path}")
        print("Please run scripts/preprocessing/add_receiver_labels.py first")
        sys.exit(1)

    # Load dataset
    print("\n1. Loading dataset with masked velocities...")
    dataset = ReceiverCornerDataset(str(graph_path), mask_velocities=True)

    # Get data loaders
    print("\n2. Creating train/val/test data loaders...")
    train_loader, val_loader, test_loader = dataset.get_data_loaders(
        batch_size=32,
        test_size=0.15,
        val_size=0.15
    )

    # Test iteration through loader
    print("\n3. Testing data loader iteration...")
    for i, batch in enumerate(train_loader):
        if i == 0:
            print(f"\nBatch 1 shapes:")
            print(f"  x (node features):    {batch.x.shape}")
            print(f"  edge_index:           {batch.edge_index.shape}")
            print(f"  receiver_label:       {batch.receiver_label.shape}")
            print(f"  shot_label:           {batch.shot_label.shape}")
            print(f"  batch size:           {batch.batch.max().item() + 1}")
            print(f"  total nodes in batch: {batch.num_nodes}")

            # Verify velocities are masked
            print(f"\nVelocity masking check:")
            print(f"  vx (col 4) sum: {batch.x[:, 4].sum().item():.6f}")
            print(f"  vy (col 5) sum: {batch.x[:, 5].sum().item():.6f}")
            assert batch.x[:, 4:6].abs().sum() == 0, "Velocities should be masked!"

            # Show some labels
            print(f"\nFirst 5 receiver labels: {batch.receiver_label[:5].squeeze().tolist()}")
            print(f"First 5 shot labels: {batch.shot_label[:5].squeeze().tolist()}")

        if i >= 2:
            break

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print(f"Ready for TacticAI receiver prediction training with {len(dataset.data_list)} corners")
    print(f"Dataset file: {graph_path.name}")
    print("="*60)
