"""The Lightning data module: loader shapes, batching, and shuffling."""

from __future__ import annotations

import torch

from mnist_mlp.config import N_PIXELS
from mnist_mlp.datamodule import MNISTDataModule
from tests.conftest import make_images


def module(n: int = 200, batch_size: int = 32) -> MNISTDataModule:
    train, validation, test = make_images(n, 0), make_images(60, 1), make_images(60, 2)
    return MNISTDataModule(train, validation, test, batch_size=batch_size)


def test_every_split_becomes_a_dataset():
    assert set(module().datasets) == {"train", "validation", "test"}


def test_batches_have_the_expected_shape_and_dtype():
    x, y = next(iter(module(batch_size=16).train_dataloader()))
    assert x.shape == (16, N_PIXELS)
    assert y.shape == (16,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_training_loader_shuffles_and_evaluation_loaders_do_not():
    """
    Shuffling train decorrelates consecutive batches. Leaving validation and
    test in order keeps their metrics comparable between runs.
    """
    data = module(300)
    assert data.train_dataloader().sampler.__class__.__name__ == "RandomSampler"
    assert data.val_dataloader().sampler.__class__.__name__ == "SequentialSampler"
    assert data.test_dataloader().sampler.__class__.__name__ == "SequentialSampler"


def test_no_images_are_dropped_by_batching():
    data = module(205, batch_size=32)          # deliberately not a multiple
    seen = sum(len(y) for _, y in data.train_dataloader())
    assert seen == 205


def test_normalized_pixels_are_centered_near_zero():
    x = module(400).datasets["train"].tensors[0]
    assert abs(float(x.mean())) < 0.15
    assert 0.5 < float(x.std()) < 2.0
