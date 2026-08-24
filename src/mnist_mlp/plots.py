"""
All Matplotlib / Seaborn visualization functions.

Design rules:
  - Every function returns a Figure without calling plt.show().
  - apply_global_style() sets project-wide aesthetics; call once at notebook start.
  - No hardcoded colors: palettes come from mnist_mlp.config.
  - Axes always carry title, x-label, and y-label.
  - Bars start at zero so their lengths stay proportional to the values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from mnist_mlp.config import (
    CMAP_SEQ,
    IMAGE_SIZE,
    PALETTE_ACCENT,
    PALETTE_LIST,
    PALETTE_PRIMARY,
)


def apply_global_style() -> None:
    """Apply project-wide styling. Call once at notebook start."""
    sns.set_theme(style="whitegrid", palette=PALETTE_LIST, font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "font.size": 11,
    })


def save_figure(fig: Figure, name: str, figures_dir) -> None:
    """Save a figure as PNG at 150 dpi."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figures_dir) / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_digit_grid(df: pd.DataFrame, n_per_class: int = 8, seed: int = 0) -> Figure:
    """
    A row per digit, sampled at random.

    Worth looking at before any modeling: it shows the handwriting variation
    the model has to absorb, and it is the fastest way to spot images that are
    corrupted rather than merely untidy.
    """
    rng = np.random.default_rng(seed)
    pixels = [c for c in df.columns if c != "label"]
    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 0.9, 10))

    for digit in range(10):
        pool = df.index[df["label"] == digit].to_numpy()
        picks = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)
        for col, idx in enumerate(picks):
            ax = axes[digit, col]
            ax.imshow(df.loc[idx, pixels].to_numpy().reshape(IMAGE_SIZE, IMAGE_SIZE),
                      cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if col == 0:
                ax.set_ylabel(str(digit), rotation=0, labelpad=12, fontsize=11)
    fig.suptitle("Random samples of each digit", fontsize=13)
    fig.tight_layout()
    return fig


def plot_class_balance(balance: pd.DataFrame) -> Figure:
    """Images per digit, to show whether class weighting is warranted."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(balance.index.astype(str), balance["images"], color=PALETTE_PRIMARY)
    mean = balance["images"].mean()
    ax.axhline(mean, color=PALETTE_ACCENT, ls="--", lw=1.3,
               label=f"mean = {mean:,.0f}")
    ax.set_title("Images per digit", fontsize=12)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Images")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ink_distribution(df: pd.DataFrame) -> Figure:
    """
    How many pixels each image actually uses.

    The left tail is where corrupted or near-empty frames hide, so this is the
    diagnostic behind the cleaning threshold rather than a decorative histogram.
    """
    pixels = [c for c in df.columns if c != "label"]
    ink = (df[pixels] > 0).sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(ink, bins=60, color=PALETTE_PRIMARY, alpha=0.85)
    ax.set_title("Non-zero pixels per image", fontsize=12)
    ax.set_xlabel("Pixels with ink")
    ax.set_ylabel("Images")
    fig.tight_layout()
    return fig


def plot_sweep(results: pd.DataFrame, axis: str, title: str,
               metric: str = "val_acc", log_x: bool = False) -> Figure:
    """
    One sweep axis against validation accuracy and loss.

    Accuracy and loss sit side by side because they disagree in an informative
    way: a configuration can keep predicting the right class while growing less
    confident about it, and only the loss panel shows that.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels = results[axis].astype(str)

    for ax, column, name, better in [
        (axes[0], metric, "Validation accuracy", "higher is better"),
        (axes[1], "val_loss", "Validation loss", "lower is better"),
    ]:
        best = results[column].idxmax() if column == metric else results[column].idxmin()
        colors = [PALETTE_ACCENT if i == best else PALETTE_PRIMARY for i in results.index]
        ax.bar(labels, results[column], color=colors)
        ax.set_title(f"{name}\n{better}", fontsize=11)
        ax.set_xlabel(axis.replace("_", " "))
        ax.set_ylabel(name)
        if log_x:
            ax.tick_params(axis="x", rotation=45)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_learning_curves(history: pd.DataFrame, title: str) -> Figure:
    """
    Training and validation loss and accuracy per epoch.

    The gap between the two curves is the overfitting diagnostic: training loss
    falling while validation loss turns back up is the picture dropout and
    weight decay exist to flatten.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (train_col, val_col, name) in zip(
        axes,
        [("train_loss", "val_loss", "Loss"), ("train_acc", "val_acc", "Accuracy")],
        strict=True,
    ):
        if train_col in history:
            ax.plot(history["epoch"], history[train_col], label="train",
                    color=PALETTE_PRIMARY, lw=1.6)
        ax.plot(history["epoch"], history[val_col], label="validation",
                color=PALETTE_ACCENT, lw=1.6)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.legend()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
    """
    Counts, not proportions, so the rare confusions stay visible.

    The interesting cells are off the diagonal: which digits this model mistakes
    for which is a more useful description of its weakness than one accuracy
    number.
    """
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap=CMAP_SEQ, ax=ax,
                linewidths=0.4, linecolor="white", cbar_kws={"label": "Images"})
    ax.set_title("Confusion matrix on the test split", fontsize=12)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    fig.tight_layout()
    return fig


def plot_misclassified(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                       n: int = 24) -> Figure:
    """
    The images the model got wrong, labeled true then predicted.

    Many of them turn out to be genuinely ambiguous handwriting, which is the
    honest ceiling on this task and worth showing rather than asserting.
    """
    wrong = np.flatnonzero(y_true != y_pred)[:n]
    pixels = [c for c in df.columns if c != "label"]
    cols = 8
    rows = int(np.ceil(len(wrong) / cols)) or 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.5))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, idx in zip(np.atleast_1d(axes).ravel(), wrong, strict=False):
        ax.imshow(df.iloc[idx][pixels].to_numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray_r")
        ax.set_title(f"{y_true[idx]} → {y_pred[idx]}", fontsize=9)
    fig.suptitle("Misclassified test images (true → predicted)", fontsize=13)
    fig.tight_layout()
    return fig
