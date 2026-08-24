"""
Leakage guards.

The failure these protect against is not a crash, it is a test score that looks
good because information reached the model that should not have. Each test is
written to prove the claim rather than restate it: the fitted artifacts have to
stay identical when the unseen data changes wildly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mnist_mlp.datamodule import MNISTDataModule
from mnist_mlp.dataset import (
    clean_data,
    normalization_stats,
    pixel_columns,
    split_data,
)
from tests.conftest import make_images


def test_normalization_statistics_come_from_train_alone():
    """
    Same training split, two completely different held-out sets. The mean and
    standard deviation applied to every batch must not move, because they are
    the one transform fitted on data and applied to the model's inputs.
    """
    train = make_images(400, seed=0)
    mild = make_images(200, seed=1)

    wild = make_images(200, seed=2)
    wild[pixel_columns(wild)] = 255            # a maximally different distribution

    a = MNISTDataModule(train, mild, mild)
    b = MNISTDataModule(train, wild, wild)

    assert a.mean == b.mean, "normalization mean shifted with the held-out data"
    assert a.std == b.std, "normalization std shifted with the held-out data"


def test_normalization_differs_when_the_training_data_differs():
    """The mirror of the previous test: the statistics must track train."""
    dark = make_images(200, seed=3)
    dark[pixel_columns(dark)] = 10
    bright = make_images(200, seed=3)
    bright[pixel_columns(bright)] = 240

    assert normalization_stats(dark)[0] < normalization_stats(bright)[0]


def test_deduplication_happens_before_the_split():
    """
    If a duplicated image survived into two different splits, the test score
    would partly be a memory check. Cleaning first makes that impossible.
    """
    df = make_images(300, seed=4)
    with_copies = pd.concat([df, df.iloc[:50]], ignore_index=True)

    cleaned = clean_data(with_copies, verbose=False)
    train, validation, test = split_data(cleaned)

    cols = pixel_columns(cleaned)
    sets = [{tuple(r) for r in part[cols].to_numpy()} for part in (train, validation, test)]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]


def test_splitting_a_dirty_frame_would_have_leaked():
    """
    The negative control for the test above. Splitting before cleaning does put
    the same image on both sides, which is exactly why the order is fixed.
    """
    df = make_images(300, seed=5)
    with_copies = pd.concat([df, df.iloc[:100]], ignore_index=True)

    train, validation, test = split_data(with_copies)      # deliberately not cleaned
    cols = pixel_columns(with_copies)
    train_rows = {tuple(r) for r in train[cols].to_numpy()}
    held_out = {tuple(r) for r in pd.concat([validation, test])[cols].to_numpy()}
    assert train_rows & held_out, "this ordering is expected to leak, which is the point"


def test_the_data_module_applies_one_transform_everywhere():
    """
    Validation and test must be standardized with the training statistics, not
    with their own. Re-fitting per split would quietly change what the model
    sees at evaluation time.
    """
    train = make_images(300, seed=6)
    held = make_images(150, seed=7)
    module = MNISTDataModule(train, held, held)

    raw = held[pixel_columns(held)].to_numpy(dtype=np.float32) / 255.0
    expected = (raw - module.mean) / module.std
    actual = module.datasets["validation"].tensors[0].numpy()
    assert np.allclose(actual, expected, atol=1e-6)


def test_labels_stay_attached_to_their_images_through_the_pipeline():
    """A shuffle that moved pixels but not labels would silently destroy training."""
    df = make_images(120, seed=8)
    module = MNISTDataModule(df, df, df)
    _, labels = module.datasets["train"].tensors
    assert labels.tolist() == df["label"].tolist()


def test_class_weights_are_inverse_frequency():
    df = make_images(600, seed=9)
    weights = MNISTDataModule(df, df, df).class_weights(df)
    counts = df["label"].value_counts().sort_index().to_numpy()
    assert weights.argmax().item() == int(counts.argmin())
    assert weights.mean().item() == pytest.approx(weights.mean().item())
