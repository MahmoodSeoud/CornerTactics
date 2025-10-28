#!/usr/bin/env python3
"""
Balanced batch sampling for imbalanced datasets.
"""
import torch
import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """
    Samples batches with equal representation of positive and negative classes.
    """
    def __init__(self, labels, batch_size, oversample=True):
        """
        Args:
            labels: List of binary labels (0 or 1)
            batch_size: Size of each batch
            oversample: If True, oversample minority class
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.oversample = oversample

        # Get indices for each class
        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]

        # Calculate samples per class in each batch
        self.pos_per_batch = batch_size // 2
        self.neg_per_batch = batch_size - self.pos_per_batch

        # Calculate number of batches
        if oversample:
            # Oversample minority to match majority
            self.n_batches = len(self.neg_indices) // self.neg_per_batch
        else:
            # Limited by minority class
            self.n_batches = len(self.pos_indices) // self.pos_per_batch

        print(f"BalancedBatchSampler initialized:")
        print(f"  Positive samples: {len(self.pos_indices)} ({len(self.pos_indices)/len(labels)*100:.1f}%)")
        print(f"  Negative samples: {len(self.neg_indices)} ({len(self.neg_indices)/len(labels)*100:.1f}%)")
        print(f"  Batch size: {batch_size} ({self.pos_per_batch} pos, {self.neg_per_batch} neg)")
        print(f"  Number of batches: {self.n_batches}")
        print(f"  Oversample: {oversample}")

    def __iter__(self):
        for _ in range(self.n_batches):
            # Sample positive indices
            pos_batch = np.random.choice(
                self.pos_indices,
                size=self.pos_per_batch,
                replace=self.oversample  # Allow replacement if oversampling
            )

            # Sample negative indices
            neg_batch = np.random.choice(
                self.neg_indices,
                size=self.neg_per_batch,
                replace=False
            )

            # Combine and shuffle
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)

            yield batch.tolist()

    def __len__(self):
        return self.n_batches