"""Cleaning, splitting, and array conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mnist_mlp.config import N_CLASSES, N_PIXELS, PIXEL_MAX
from mnist_mlp.dataset import (
    audit_data,
    cached_train_file,
    class_balance,
    clean_data,
    load_raw,
    normalization_stats,
    pixel_columns,
    split_data,
    split_summary,
    to_arrays,
)
from tests.conftest import make_images


def test_pixel_columns_excludes_the_label():
    cols = pixel_columns(make_images(10))
    assert "label" not in cols
    assert len(cols) == N_PIXELS


def test_audit_counts_duplicates_and_blanks():
    df = make_images(50)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)      # one exact duplicate
    blank = df.iloc[[1]].copy()
    blank[pixel_columns(df)] = 0
    df = pd.concat([df, blank], ignore_index=True)             # one blank frame

    report = audit_data(df)["value"]
    assert report["duplicate_images"] == 1
    assert report["blank_images"] == 1


def test_clean_removes_duplicate_images():
    df = make_images(40)
    doubled = pd.concat([df, df.iloc[[3, 7]]], ignore_index=True)
    cleaned = clean_data(doubled, verbose=False)
    assert len(cleaned) == 40
    assert cleaned.duplicated(subset=pixel_columns(cleaned)).sum() == 0


def test_clean_drops_contradictory_labels():
    """
    The same picture carrying two different labels is a labeling error. Keeping
    either copy would teach the model that one input has two right answers, so
    both go.
    """
    df = make_images(30)
    clash = df.iloc[[5]].copy()
    clash["label"] = (int(df.loc[5, "label"]) + 1) % N_CLASSES
    cleaned = clean_data(pd.concat([df, clash], ignore_index=True), verbose=False)

    original = df.loc[5, pixel_columns(df)].to_numpy()
    surviving = cleaned[pixel_columns(cleaned)].to_numpy()
    assert not (surviving == original).all(axis=1).any(), "the contradictory pair must be dropped"


def test_clean_removes_blank_and_near_blank_frames():
    df = make_images(30)
    blank = df.iloc[[0]].copy()
    blank[pixel_columns(df)] = 0
    cleaned = clean_data(pd.concat([df, blank], ignore_index=True), verbose=False)
    ink = (cleaned[pixel_columns(cleaned)] > 0).sum(axis=1)
    assert ink.min() > 0


def test_clean_removes_impossible_labels_and_pixels():
    df = make_images(30)
    bad_label = df.iloc[[0]].copy()
    bad_label["label"] = 99
    bad_pixel = df.iloc[[1]].copy()
    bad_pixel["pixel400"] = 999
    cleaned = clean_data(pd.concat([df, bad_label, bad_pixel], ignore_index=True), verbose=False)

    assert cleaned["label"].between(0, N_CLASSES - 1).all()
    assert cleaned[pixel_columns(cleaned)].to_numpy().max() <= PIXEL_MAX


def test_split_is_exactly_sixty_twenty_twenty():
    df = make_images(1000)
    train, validation, test = split_data(df)
    assert len(train) == 600
    assert len(validation) == 200
    assert len(test) == 200
    assert len(train) + len(validation) + len(test) == len(df)


def test_split_is_stratified_across_all_three_parts():
    df = make_images(2000, seed=3)
    train, validation, test = split_data(df)
    whole = df["label"].value_counts(normalize=True).sort_index()
    for part in (train, validation, test):
        share = part["label"].value_counts(normalize=True).sort_index()
        assert np.allclose(share.to_numpy(), whole.to_numpy(), atol=0.02)


def test_splits_are_disjoint():
    """No image may appear in more than one split."""
    df = make_images(600, seed=4)
    train, validation, test = split_data(df)
    cols = pixel_columns(df)
    as_rows = [{tuple(r) for r in part[cols].to_numpy()} for part in (train, validation, test)]
    assert not as_rows[0] & as_rows[1]
    assert not as_rows[0] & as_rows[2]
    assert not as_rows[1] & as_rows[2]


def test_split_is_reproducible_under_the_same_seed():
    df = make_images(500, seed=5)
    a = split_data(df)[0]["label"].tolist()
    b = split_data(df)[0]["label"].tolist()
    assert a == b


def test_split_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        split_data(make_images(100), 0.5, 0.3, 0.3)


def test_split_summary_reports_every_part():
    train, validation, test = split_data(make_images(500))
    out = split_summary(train, validation, test)
    assert list(out.index) == ["train", "validation", "test"]
    assert out["share_of_all_%"].sum() == pytest.approx(100.0, abs=0.2)


def test_to_arrays_returns_the_right_shapes_and_dtypes():
    x, y = to_arrays(make_images(37))
    assert x.shape == (37, N_PIXELS)
    assert y.shape == (37,)
    assert x.dtype == np.float32
    assert y.dtype == np.int64


def test_class_balance_shares_sum_to_one_hundred():
    out = class_balance(make_images(400))
    assert out["share_%"].sum() == pytest.approx(100.0, abs=0.1)


def test_normalization_stats_are_on_the_zero_one_scale():
    mean, std = normalization_stats(make_images(300))
    assert 0.0 < mean < 1.0
    assert 0.0 < std < 1.0


def test_cached_train_file_finds_nothing_in_an_empty_directory(tmp_path):
    assert cached_train_file(tmp_path) is None


def test_cached_train_file_accepts_the_gzipped_copy(tmp_path):
    """A fresh clone ships only train.csv.gz, and that must be enough."""
    (tmp_path / "train.csv.gz").write_bytes(b"")
    assert cached_train_file(tmp_path).name == "train.csv.gz"


def test_plain_csv_wins_when_both_are_present(tmp_path):
    """The plain file is the newer Kaggle download, so it takes precedence."""
    (tmp_path / "train.csv.gz").write_bytes(b"")
    (tmp_path / "train.csv").write_text("")
    assert cached_train_file(tmp_path).name == "train.csv"


def test_load_raw_reads_a_gzipped_file(tmp_path):
    """pandas decompresses by extension, so the gzipped cache round-trips."""
    frame = make_images(12, seed=1)
    frame.to_csv(tmp_path / "train.csv.gz", index=False, compression="gzip")
    loaded = load_raw(tmp_path)
    assert loaded.shape == frame.shape
    assert loaded["label"].tolist() == frame["label"].tolist()


def test_load_raw_says_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw(tmp_path)
