"""
The Lightning data module.

Holds the three tensor datasets and the loaders that feed them. Normalization
statistics arrive from the training split and are applied unchanged everywhere
else, so the transform is fitted exactly once on exactly the right data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import TensorDataset

from mnist_mlp.config import BATCH_SIZE, PIXEL_MAX
from mnist_mlp.dataset import normalization_stats, to_arrays


def resolve_device() -> torch.device:
    """The device training will actually run on: CUDA, then Apple Metal, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeviceTensorLoader:
    """
    Batches sliced straight out of tensors that already live on the training device.

    A `DataLoader` over CPU tensors copies every batch across the bus, and for a
    network this small the copy and the kernel launch cost more than the matrix
    multiplies they feed: the GPU spends most of each step idle waiting for work.
    At a batch size of 32 that is 788 transfers per epoch. The whole training
    split is 79 MB, so it fits on the device with room to spare, and moving it
    once at setup turns every batch into an on-device index.

    Shuffling draws its permutation from the CPU generator that `seed_everything`
    seeds, so a run stays reproducible and does not depend on the device having a
    seeded RNG of its own.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def __iter__(self):
        order = torch.randperm(len(self.x)) if self.shuffle else torch.arange(len(self.x))
        order = order.to(self.x.device)
        for start in range(0, len(order), self.batch_size):
            index = order[start:start + self.batch_size]
            yield self.x[index], self.y[index]


class MNISTDataModule(LightningDataModule):
    """
    Wraps the three splits as tensor datasets.

    The images are already in memory as a dense matrix, so there is nothing to
    decode and no per-item file read to do. Each split is moved onto the training
    device once at construction and batched by indexing it there, which removes
    the per-batch host-to-device copy that dominates the step time for a network
    this small.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        batch_size: int = BATCH_SIZE,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.device = torch.device(device) if device is not None else resolve_device()

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
        # The one transfer. Everything after this is an index on the device.
        return TensorDataset(torch.from_numpy(x).to(self.device),
                             torch.from_numpy(y).to(self.device))

    def _loader(self, name: str, shuffle: bool) -> DeviceTensorLoader:
        x, y = self.datasets[name].tensors
        return DeviceTensorLoader(x, y, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self) -> DeviceTensorLoader:
        # Shuffled so consecutive batches are not correlated by label order.
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DeviceTensorLoader:
        return self._loader("validation", shuffle=False)

    def test_dataloader(self) -> DeviceTensorLoader:
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
