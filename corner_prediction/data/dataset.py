"""CornerKickDataset — PyTorch Geometric dataset for corner kick graphs.

Wraps the graph list from build_graphs.py into a PyG InMemoryDataset with
support for leave-one-match-out (LOMO) cross-validation splits.

Usage:
    from corner_prediction.data.dataset import CornerKickDataset, lomo_split

    dataset = CornerKickDataset(root="corner_prediction/data")
    train, test = lomo_split(dataset, held_out_match_id="2017461")
"""

import logging
import os.path as osp
import pickle
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data, InMemoryDataset

from .build_graphs import build_graph_dataset

logger = logging.getLogger(__name__)


class CornerKickDataset(InMemoryDataset):
    """In-memory PyG dataset for corner kick graphs.

    Loads extracted corner records, builds PyG graphs, and caches the
    processed result for fast subsequent loading.

    Args:
        root: Directory containing extracted_corners.pkl.
              Processed graphs are cached in root/processed/.
        records: Pre-loaded corner records (skips file loading if provided).
        edge_type: "knn" or "dense" edge construction.
        k: Number of neighbors for KNN edges.
        transform: PyG transform applied per-access.
        pre_transform: PyG transform applied once during processing.
    """

    def __init__(
        self,
        root: str,
        records: list = None,
        edge_type: str = "knn",
        k: int = 6,
        transform=None,
        pre_transform=None,
    ):
        self._records = records
        self._edge_type = edge_type
        self._k = k
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        # extracted_corners.pkl lives directly in root, not root/raw/
        return self.root

    @property
    def raw_file_names(self) -> List[str]:
        if self._records is not None:
            return []  # no raw files needed when records are provided
        return ["extracted_corners.pkl"]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"graphs_{self._edge_type}{self._k}.pt"]

    def process(self):
        if self._records is not None:
            records = self._records
        else:
            raw_path = Path(self.raw_dir) / "extracted_corners.pkl"
            with open(raw_path, "rb") as f:
                records = pickle.load(f)

        graphs = build_graph_dataset(records, edge_type=self._edge_type, k=self._k)

        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]

        self.save(graphs, self.processed_paths[0])


def lomo_split(
    dataset: CornerKickDataset,
    held_out_match_id: str,
) -> Tuple[List[Data], List[Data]]:
    """Leave-one-match-out split.

    Args:
        dataset: CornerKickDataset or list of Data objects.
        held_out_match_id: Match ID to hold out for testing.

    Returns:
        (train_data, test_data) — two lists of Data objects.
    """
    held_out = str(held_out_match_id)
    train_data = [g for g in dataset if str(g.match_id) != held_out]
    test_data = [g for g in dataset if str(g.match_id) == held_out]
    return train_data, test_data


def get_match_ids(dataset) -> List[str]:
    """Return sorted unique match IDs in the dataset."""
    return sorted(set(str(g.match_id) for g in dataset))
