"""
Shared fixtures, and the warning setup for the whole suite.

The filters live here rather than at the top of a test module so that every
module can keep its imports at the top: pytest imports conftest before any test
file, and these filters act on warnings raised at run time rather than at import
time, so nothing depends on the order the libraries load in.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from lightning.fabric.utilities.warnings import PossibleUserWarning

from mnist_mlp.config import N_CLASSES, N_PIXELS

# Two named filters, not a blanket ignore. Lightning's operational advice is not
# a finding here, and the pytree deprecation belongs to the library. Anything
# else, including a numerical or convergence warning from our own code, still
# reaches the output.
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", message=r".*treespec.*", category=FutureWarning)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

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
