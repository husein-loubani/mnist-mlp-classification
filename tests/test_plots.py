"""
Figure tests.

These do not judge whether a chart looks good, which no test can. They check the
contract every figure in this project is supposed to meet: it returns a Figure,
it never calls plt.show(), and its axes carry a title and labels so a reader is
never guessing what an axis measures.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")   # no display in a test run
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from mnist_mlp.dataset import class_balance  # noqa: E402
from mnist_mlp.plots import (
    plot_class_balance,
    plot_confusion_matrix,
    plot_digit_grid,
    plot_ink_distribution,
    plot_joint_grid,  # noqa: E402
    plot_learning_curves,
    plot_misclassified,
    plot_sweep,
    save_figure,
)
from tests.conftest import make_images  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_class_balance_figure_is_labeled():
    fig = plot_class_balance(class_balance(make_images(300)))
    ax = fig.axes[0]
    assert isinstance(fig, Figure)
    assert ax.get_title()
    assert ax.get_xlabel() and ax.get_ylabel()


def test_class_balance_bars_start_at_zero():
    """Bars must be proportional to their values, so the axis cannot be truncated."""
    fig = plot_class_balance(class_balance(make_images(300)))
    assert fig.axes[0].get_ylim()[0] == 0


def test_ink_distribution_is_labeled():
    fig = plot_ink_distribution(make_images(120))
    ax = fig.axes[0]
    assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()


def test_digit_grid_draws_a_row_per_class():
    fig = plot_digit_grid(make_images(400, seed=1), n_per_class=3)
    assert len(fig.axes) == 10 * 3


def test_sweep_figure_has_both_panels_labeled():
    results = pd.DataFrame({
        "optimizer": ["adam", "sgd"],
        "val_acc": [0.97, 0.91],
        "val_loss": [0.10, 0.30],
    })
    fig = plot_sweep(results, "optimizer", "Optimizers")
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()


def test_learning_curves_plot_train_and_validation():
    history = pd.DataFrame({
        "epoch": [0, 1, 2],
        "train_loss": [0.9, 0.5, 0.3],
        "val_loss": [1.0, 0.6, 0.5],
        "train_acc": [0.6, 0.8, 0.9],
        "val_acc": [0.5, 0.75, 0.85],
    })
    fig = plot_learning_curves(history, "Baseline")
    assert len(fig.axes) == 2
    assert fig.axes[0].get_legend() is not None


def test_confusion_matrix_is_labeled():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 10, 200)
    y_pred = y_true.copy()
    y_pred[:20] = (y_pred[:20] + 1) % 10
    ax = plot_confusion_matrix(y_true, y_pred).axes[0]
    assert ax.get_xlabel() and ax.get_ylabel() and ax.get_title()


def test_misclassified_grid_only_shows_mistakes():
    df = make_images(60, seed=2)
    y_true = df["label"].to_numpy()
    y_pred = y_true.copy()
    y_pred[:5] = (y_pred[:5] + 1) % 10
    fig = plot_misclassified(df, y_true, y_pred, n=5)
    titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titled) == 5
    assert all("→" in t for t in titled)


def test_save_figure_writes_a_png(tmp_path):
    fig = plot_class_balance(class_balance(make_images(100)))
    save_figure(fig, "example", tmp_path)
    written = tmp_path / "example.png"
    assert written.exists() and written.stat().st_size > 0


def test_joint_grid_heatmap_is_labeled():
    table = pd.DataFrame([[0.90, 0.95], [0.92, 0.91]],
                         index=[32, 64], columns=[1e-3, 3e-3])
    fig = plot_joint_grid(table)
    ax = fig.axes[0]
    assert ax.get_xlabel() and ax.get_ylabel() and ax.get_title()
    assert [t.get_text() for t in ax.get_yticklabels()] == ["32", "64"]
    plt.close(fig)
