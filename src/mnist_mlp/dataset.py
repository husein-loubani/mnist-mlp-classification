"""
Download, audit, clean, and split the Kaggle MNIST digits.

Two ordering rules matter here and both are about leakage rather than tidiness:

  1. Duplicate images are removed *before* the split. If the same picture sat in
     train and test, the test score would partly be a memory check.
  2. Normalization statistics come from the training split only, and are then
     applied unchanged to validation and test. Standardizing with statistics
     computed over the whole file would let the test set influence the inputs
     the model trains on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mnist_mlp.config import (
    KAGGLE_COMPETITION,
    MIN_INK_PIXELS,
    N_CLASSES,
    N_PIXELS,
    PIXEL_MAX,
    RANDOM_SEED,
    TEST_FRACTION,
    TRAIN_FILE,
    TRAIN_FILE_GZ,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)


def cached_train_file(raw_dir: str | Path) -> Path | None:
    """
    The cached training data, plain or gzipped, whichever is on disk.

    A fresh Kaggle download writes `train.csv`; the repository ships
    `train.csv.gz` instead so a clone stays small. Either satisfies the loader,
    and the plain file wins when both exist because it is the newer download.
    """
    raw_dir = Path(raw_dir)
    for name in (TRAIN_FILE, TRAIN_FILE_GZ):
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    return None


def download_data(raw_dir: str | Path, force: bool = False) -> Path:
    """
    Fetch the Digit Recognizer competition data with the Kaggle API and cache it.

    Needs an API token at ~/.kaggle/kaggle.json and the competition rules
    accepted on the website. The cache means later runs never touch the network,
    so the notebook reproduces offline.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / TRAIN_FILE

    cached = cached_train_file(raw_dir)
    if cached is not None and not force:
        print(f"  {cached.name} cached ({cached.stat().st_size / 1e6:.1f} MB)")
        return cached

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(KAGGLE_COMPETITION, path=str(raw_dir), quiet=False)

    zip_path = raw_dir / f"{KAGGLE_COMPETITION}.zip"
    if zip_path.exists():
        import zipfile

        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(raw_dir)
        zip_path.unlink()

    print(f"  downloaded {TRAIN_FILE} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    """
    Read train.csv: one row per image, a `label` column plus 784 pixel columns.

    Kaggle's test.csv is deliberately not used. It carries no labels because it
    scores the leaderboard, so it cannot measure anything here. All three splits
    come out of this labeled file.
    """
    cached = cached_train_file(raw_dir)
    if cached is None:
        raise FileNotFoundError(
            f"no {TRAIN_FILE} or {TRAIN_FILE_GZ} in {raw_dir}; run download_data first"
        )
    return pd.read_csv(cached)


def pixel_columns(df: pd.DataFrame) -> list[str]:
    """The 784 pixel column names, in order."""
    return [c for c in df.columns if c != "label"]


def audit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the things that quietly ruin an image dataset: duplicates, blank
    frames, out-of-range pixels, impossible labels, and class imbalance.
    """
    pixels = df[pixel_columns(df)]
    ink = (pixels > 0).sum(axis=1)
    counts = df["label"].value_counts()

    summary = {
        "images": len(df),
        "pixel_columns": pixels.shape[1],
        "duplicate_images": int(df.duplicated(subset=pixel_columns(df)).sum()),
        "duplicate_rows_incl_label": int(df.duplicated().sum()),
        "blank_images": int((ink == 0).sum()),
        "near_blank_images": int((ink < MIN_INK_PIXELS).sum()),
        "missing_values": int(df.isna().sum().sum()),
        "pixel_min": int(pixels.to_numpy().min()),
        "pixel_max": int(pixels.to_numpy().max()),
        "labels_out_of_range": int((~df["label"].between(0, N_CLASSES - 1)).sum()),
        "classes_present": int(df["label"].nunique()),
        "most_common_class_%": round(counts.max() / len(df) * 100, 2),
        "least_common_class_%": round(counts.min() / len(df) * 100, 2),
    }
    return pd.DataFrame([summary]).T.rename(columns={0: "value"})


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Per-digit counts and shares, which decide whether class weights are needed."""
    counts = df["label"].value_counts().sort_index()
    out = pd.DataFrame({"images": counts})
    out["share_%"] = (counts / counts.sum() * 100).round(2)
    out.index.name = "digit"
    return out


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove corrupted and duplicated images.

    Duplicates are dropped on the pixels alone, not on pixels plus label. Two
    identical pictures carrying different labels are a labeling contradiction,
    and keeping either copy would teach the model that the same input has two
    right answers, so both go.

    This runs before the split on purpose: deduplicating afterwards would leave
    copies of the same image on both sides of the boundary.
    """
    pixels = pixel_columns(df)
    before = len(df)
    out = df.copy()

    out = out[out["label"].between(0, N_CLASSES - 1)]
    out = out.dropna()

    in_range = (out[pixels] >= 0).all(axis=1) & (out[pixels] <= PIXEL_MAX).all(axis=1)
    out = out[in_range]

    ink = (out[pixels] > 0).sum(axis=1)
    out = out[ink >= MIN_INK_PIXELS]

    contradictory = out.duplicated(subset=pixels, keep=False) & out.duplicated(
        subset=[*pixels, "label"], keep=False
    ).eq(False)
    out = out[~contradictory]
    out = out.drop_duplicates(subset=pixels, keep="first")

    out = out.reset_index(drop=True)
    if verbose:
        print(f"  cleaned: {before:,} -> {len(out):,} images ({before - len(out):,} removed)")
    return out


def split_data(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 60 / 20 / 20 split.

    Stratifying keeps every digit at the same share in all three parts, so a
    validation score is not moved by one class being over-represented there.
    The split is seeded, so the same rows land in the same part on every run and
    the test set stays genuinely untouched between experiments.
    """
    from sklearn.model_selection import train_test_split

    total = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")

    train, rest = train_test_split(
        df, train_size=train_fraction, stratify=df["label"], random_state=seed
    )
    relative_validation = validation_fraction / (validation_fraction + test_fraction)
    validation, test = train_test_split(
        rest, train_size=relative_validation, stratify=rest["label"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def split_summary(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Row counts and per-class shares, to show the stratification held."""
    rows = []
    total = len(train) + len(validation) + len(test)
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        shares = part["label"].value_counts(normalize=True).sort_index() * 100
        rows.append({
            "split": name,
            "images": len(part),
            "share_of_all_%": round(len(part) / total * 100, 1),
            "min_class_%": round(shares.min(), 2),
            "max_class_%": round(shares.max(), 2),
        })
    return pd.DataFrame(rows).set_index("split")


def to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split a frame into a float32 pixel matrix and an int64 label vector."""
    # copy=True because pandas hands back a read-only view of its own block,
    # and torch.from_numpy warns on a non-writable array rather than copying it.
    x = df[pixel_columns(df)].to_numpy(dtype=np.float32, copy=True)
    y = df["label"].to_numpy(dtype=np.int64, copy=True)
    if x.shape[1] != N_PIXELS:
        raise ValueError(f"expected {N_PIXELS} pixel columns, found {x.shape[1]}")
    return x, y


def normalization_stats(train: pd.DataFrame) -> tuple[float, float]:
    """
    Mean and standard deviation of the *training* pixels, scaled to [0, 1].

    These are the only statistics allowed to touch the model's inputs. Computing
    them over the full file instead would leak the test distribution into every
    training batch, which is the quietest leak available in an image pipeline.
    """
    x, _ = to_arrays(train)
    x = x / PIXEL_MAX
    return float(x.mean()), float(x.std())
