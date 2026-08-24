"""
Constants.

A typo in this module fails silently rather than loudly: split fractions that do
not sum to one, or a grid that no longer matches what the model accepts, would
produce a plausible-looking but wrong project. These are cheap guards against
that.
"""

from __future__ import annotations

import pytest

from mnist_mlp import config
from mnist_mlp.models import ACTIVATIONS


def test_split_fractions_sum_to_one():
    total = config.TRAIN_FRACTION + config.VALIDATION_FRACTION + config.TEST_FRACTION
    assert total == pytest.approx(1.0)


def test_split_matches_the_sixty_twenty_twenty_the_brief_asks_for():
    assert pytest.approx(0.60) == config.TRAIN_FRACTION
    assert pytest.approx(0.20) == config.VALIDATION_FRACTION
    assert pytest.approx(0.20) == config.TEST_FRACTION


def test_image_geometry_is_self_consistent():
    assert config.N_PIXELS == config.IMAGE_SIZE**2 == 784
    assert config.N_CLASSES == 10


def test_every_activation_in_the_grid_is_implemented():
    """A grid entry the model cannot build would fail mid-sweep, not at import."""
    assert set(config.ACTIVATION_GRID) <= set(ACTIVATIONS)


def test_every_optimizer_in_the_grid_is_implemented():
    from mnist_mlp.models import LitMLP

    for name in config.OPTIMIZER_GRID:
        LitMLP(hidden_layers=(4,), optimizer=name).configure_optimizers()


def test_every_loss_in_the_grid_is_implemented():
    from mnist_mlp.models import LitMLP

    for name in config.LOSS_GRID:
        LitMLP(hidden_layers=(4,), loss=name)


def test_dropout_grid_stays_in_the_valid_range():
    assert all(0.0 <= d < 1.0 for d in config.DROPOUT_GRID)


def test_learning_rate_grid_is_positive_and_ordered():
    grid = list(config.LEARNING_RATE_GRID)
    assert all(lr > 0 for lr in grid)
    assert grid == sorted(grid)


def test_architecture_grid_entries_are_non_empty_tuples_of_positive_widths():
    for architecture in config.ARCHITECTURE_GRID:
        assert len(architecture) >= 1
        assert all(isinstance(w, int) and w > 0 for w in architecture)
