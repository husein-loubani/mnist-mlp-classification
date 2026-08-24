"""Shared synthetic fixtures. No test touches the real 42,000-image file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mnist_mlp.config import N_CLASSES, N_PIXELS


def make_images(n: int = 200, seed: int = 0, n_classes: int = N_CLASSES) -> pd.DataFrame:
    """A frame shaped exactly like train.csv: a label plus 784 pixel columns."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(n, N_PIXELS), dtype=np.int64)
    pixels[:, :20] = 0                      # a little structure, and guaranteed ink elsewhere
    labels = rng.integers(0, n_classes, size=n)
    frame = pd.DataFrame(pixels, columns=[f"pixel{i}" for i in range(N_PIXELS)])
    frame.insert(0, "label", labels)
    return frame


@pytest.fixture
def images() -> pd.DataFrame:
    return make_images()
