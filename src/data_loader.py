#!/usr/bin/env python3
"""
Data Loader Module for Corner Kick GNN

Implements Phase 3.3: PyTorch Geometric data loading utilities.
Handles conversion from CornerGraph objects to PyG Data objects
and provides train/val/test data loaders.

Author: mseo
Date: October 2024
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataListLoader
from sklearn.model_selection import train_test_split
import warnings

# Import our custom graph structure
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.graph_builder import CornerGraph


class CornerDataset:
    """
    Dataset class for corner kick graphs.

    Handles loading graphs from pickle files and converting
    to PyTorch Geometric Data objects.
    """

    def __init__(self, graph_path: str, outcome_type: str = "goal"):
        """
        Initialize the dataset.

        Args:
            graph_path: Path to pickled graph file
            outcome_type: Type of outcome to predict
                - "goal": Binary goal prediction
                - "shot": Binary shot prediction
                - "multi": Multi-class outcome prediction
        """
        self.graph_path = graph_path
        self.outcome_type = outcome_type
        self.graphs = self._load_graphs()
        self.data_list = self._convert_to_pyg_data()

        # Compute dataset statistics
        self._compute_statistics()

    def _load_graphs(self) -> List[CornerGraph]:
        """Load graphs from pickle file."""
        print(f"Loading graphs from {self.graph_path}")
        with open(self.graph_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")
        return graphs

    def _convert_to_pyg_data(self) -> List[Data]:
        """
        Convert CornerGraph objects to PyTorch Geometric Data objects.

        Returns:
            List of PyG Data objects
        """
        data_list = []

        for graph in self.graphs:
            # Convert node features to tensor
            x = torch.FloatTensor(graph.node_features)

            # Convert edge index to tensor
            # Note: edge_index should be [2, num_edges] with source->target pairs
            edge_index = torch.LongTensor(graph.edge_index)

            # Convert edge features to tensor (if available)
            edge_attr = None
            if graph.edge_features is not None and len(graph.edge_features) > 0:
                edge_attr = torch.FloatTensor(graph.edge_features)

            # Prepare labels based on outcome type
            if self.outcome_type == "goal":
                # Binary goal prediction
                y = torch.FloatTensor([1.0 if graph.goal_scored else 0.0])
            elif self.outcome_type == "shot":
                # Binary shot prediction (includes goals)
                # FIXED: Check both outcome_label AND goal_scored flag
                # Some goals have outcome_label=None, so we must check goal_scored
                is_shot = (graph.outcome_label == "Shot") or graph.goal_scored
                y = torch.FloatTensor([1.0 if is_shot else 0.0])
            elif self.outcome_type == "multi":
                # Multi-class prediction
                outcome_map = {
                    "Goal": 0,
                    "Shot": 1,
                    "Clearance": 2,
                    "Loss": 3,
                    "Possession": 4,
                    "Other": 5
                }
                outcome_idx = outcome_map.get(graph.outcome_label, 5)
                y = torch.LongTensor([outcome_idx])
            else:
                raise ValueError(f"Unknown outcome type: {self.outcome_type}")

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                corner_id=graph.corner_id,
                num_nodes=x.size(0)
            )

            # Add team information for potential analysis
            if graph.teams:
                # Create a tensor indicating attacking (1) vs defending (0) team
                team_labels = torch.LongTensor([
                    1 if team == 'attacking' else 0 for team in graph.teams
                ])
                data.team = team_labels

            data_list.append(data)

        return data_list

    def _compute_statistics(self):
        """Compute and print dataset statistics."""
        self.num_graphs = len(self.data_list)
        self.num_features = self.data_list[0].x.size(1) if self.data_list else 0

        # Compute outcome statistics
        if self.outcome_type in ["goal", "shot"]:
            positive_count = sum(data.y.item() for data in self.data_list)
            self.positive_rate = positive_count / self.num_graphs if self.num_graphs > 0 else 0
            print(f"Dataset: {self.num_graphs} corners, "
                  f"{positive_count} positive ({self.positive_rate*100:.1f}%)")
        else:
            # Multi-class statistics
            outcome_counts = {}
            for data in self.data_list:
                label = data.y.item()
                outcome_counts[label] = outcome_counts.get(label, 0) + 1
            print(f"Dataset: {self.num_graphs} corners")
            print("Outcome distribution:", outcome_counts)

        # Node and edge statistics
        node_counts = [data.num_nodes for data in self.data_list]
        edge_counts = [data.edge_index.size(1) for data in self.data_list]

        self.avg_nodes = np.mean(node_counts)
        self.avg_edges = np.mean(edge_counts)

        print(f"Avg nodes per graph: {self.avg_nodes:.1f}")
        print(f"Avg edges per graph: {self.avg_edges:.1f}")

    def get_split_indices(self, test_size: float = 0.15,
                         val_size: float = 0.15,
                         random_state: int = 42) -> Dict[str, List[int]]:
        """
        Get train/val/test split indices by corner ID (not by graph).

        CRITICAL FIX: Ensures all temporal frames from the same corner
        stay together in the same split, preventing data leakage.

        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'val', 'test' indices
        """
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
                corner_to_label[base_corner_id] = data.y.numpy()[0]
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

        Args:
            batch_size: Batch size for training
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            num_workers: Number of workers for data loading

        Returns:
            train_loader, val_loader, test_loader
        """
        # Get split indices
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
        print(f"\nData split:")
        print(f"  Train: {len(train_data)} ({len(train_data)/len(self.data_list)*100:.1f}%)")
        print(f"  Val:   {len(val_data)} ({len(val_data)/len(self.data_list)*100:.1f}%)")
        print(f"  Test:  {len(test_data)} ({len(test_data)/len(self.data_list)*100:.1f}%)")

        # Check positive rates in each split
        if self.outcome_type in ["goal", "shot"]:
            train_pos = sum(d.y.item() for d in train_data) / len(train_data)
            val_pos = sum(d.y.item() for d in val_data) / len(val_data)
            test_pos = sum(d.y.item() for d in test_data) / len(test_data)
            print(f"\nPositive rates:")
            print(f"  Train: {train_pos*100:.1f}%")
            print(f"  Val:   {val_pos*100:.1f}%")
            print(f"  Test:  {test_pos*100:.1f}%")

        return train_loader, val_loader, test_loader


def load_corner_dataset(
    graph_path: str = "data/graphs/adjacency_team/statsbomb_graphs.pkl",
    outcome_type: str = "goal",
    batch_size: int = 32,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[CornerDataset, DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to load corner dataset and create data loaders.

    Args:
        graph_path: Path to graph pickle file
        outcome_type: Outcome to predict ("goal", "shot", "multi")
        batch_size: Batch size
        test_size: Test set fraction
        val_size: Validation set fraction
        random_state: Random seed

    Returns:
        dataset, train_loader, val_loader, test_loader
    """
    dataset = CornerDataset(graph_path, outcome_type)
    train_loader, val_loader, test_loader = dataset.get_data_loaders(
        batch_size=batch_size,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    return dataset, train_loader, val_loader, test_loader


def collate_fn(batch: List[Data]) -> Data:
    """
    Custom collate function for handling variable-sized graphs.

    This is handled automatically by PyG DataLoader, but provided
    here for reference and potential custom batching logic.
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


if __name__ == "__main__":
    # Test the data loader
    print("Testing Corner Dataset Loader...")
    print("=" * 60)

    # Check if graph file exists
    graph_path = Path("data/graphs/adjacency_team/statsbomb_graphs.pkl")
    if not graph_path.exists():
        print(f"Graph file not found: {graph_path}")
        print("Please run scripts/build_graph_dataset.py first")
        sys.exit(1)

    # Load dataset
    dataset = CornerDataset(str(graph_path), outcome_type="goal")

    # Get data loaders
    train_loader, val_loader, test_loader = dataset.get_data_loaders(
        batch_size=32,
        test_size=0.15,
        val_size=0.15
    )

    # Test iteration through loader
    print("\nTesting data loader iteration...")
    for i, batch in enumerate(train_loader):
        if i == 0:
            print(f"Batch shape: x={batch.x.shape}, edge_index={batch.edge_index.shape}")
            print(f"Batch size: {batch.batch.max().item() + 1}")
            print(f"Labels: {batch.y[:5].squeeze()}")
            print(f"Num graphs in batch: {batch.num_graphs}")
        if i >= 2:
            break

    print("\n✅ Data loader tests passed!")
    print(f"Ready for training with {len(dataset.data_list)} graphs")