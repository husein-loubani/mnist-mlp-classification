"""
The Lightning data module.

Holds the three tensor datasets and the loaders that feed them. Normalization
statistics arrive from the training split and are applied unchanged everywhere
else, so the transform is fitted exactly once on exactly the right data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from mnist_mlp.config import BATCH_SIZE, NUM_WORKERS, PIXEL_MAX
from mnist_mlp.dataset import normalization_stats, to_arrays


class MNISTDataModule(LightningDataModule):
    """
    Wraps the three splits as tensor datasets.

    The images are already in memory as a dense matrix, so `TensorDataset` is
    the right structure: no per-item file reads, no decoding, and the whole
    thing sits on the device in one transfer. Loading 42,000 small PNGs one at a
    time through a `Dataset` would be slower for no benefit.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Fitted on train, applied to all three. This is the whole leakage story
        # for the input pipeline.
        self.mean, self.std = normalization_stats(train)

        self.datasets = {
            name: self._build(part)
            for name, part in [("train", train), ("validation", validation), ("test", test)]
        }

    def _build(self, part: pd.DataFrame) -> TensorDataset:
        x, y = to_arrays(part)
        x = (x / PIXEL_MAX - self.mean) / self.std
        return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

    def _loader(self, name: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.datasets[name],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        # Shuffled so consecutive batches are not correlated by label order.
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("validation", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    def class_weights(self, train: pd.DataFrame) -> torch.Tensor:
        """
        Inverse-frequency class weights, for the imbalanced case.

        MNIST is close to balanced, so these come out near 1 and the notebook
        reports that rather than applying them for show. The method exists
        because the review asks what you would do about imbalance, and the
        answer should be runnable rather than described.
        """
        counts = train["label"].value_counts().sort_index().to_numpy(dtype=np.float64)
        weights = counts.sum() / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32)
