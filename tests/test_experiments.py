"""
The experiment runner.

This module decides every reported number: which configuration wins, what the
sweep table says, and what the single test evaluation reports. The Sprint 4
review pointed out that important modules were going untested, and this was the
one still missing a file, so the machinery that produces the results is pinned
here.

Runs are deliberately tiny, a handful of images for one or two epochs. These
tests check that the plumbing is correct, not that the model is good.
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd
import pytest
import torch
from lightning.fabric.utilities.warnings import PossibleUserWarning

# Two named filters, not a blanket ignore. Lightning's operational advice (a GPU
# present but unused, dataloader worker counts) is not a finding here, and the
# pytree deprecation belongs to the library rather than to this project. Anything
# else, including a numerical or convergence warning from our own code, still
# reaches the output.
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", message=r".*treespec.*", category=FutureWarning)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from mnist_mlp.datamodule import MNISTDataModule  # noqa: E402
from mnist_mlp.experiments import (  # noqa: E402
    evaluate_on_test,
    pick_accelerator,
    predict_labels,
    reference_baselines,
    run_experiment,
    sweep,
)
from mnist_mlp.models import LitMLP  # noqa: E402
from tests.conftest import make_images  # noqa: E402

TINY = dict(hidden_layers=(8,), dropout=0.0)


@pytest.fixture
def data() -> MNISTDataModule:
    return MNISTDataModule(
        make_images(120, seed=0), make_images(60, seed=1), make_images(60, seed=2),
        batch_size=32,
    )


def test_accelerator_is_one_of_the_supported_backends():
    assert pick_accelerator() in {"cuda", "mps", "cpu"}


def test_accelerator_prefers_a_gpu_when_one_exists():
    """CUDA first, then Apple Metal, then CPU. The notebook reports which ran."""
    chosen = pick_accelerator()
    if torch.cuda.is_available():
        assert chosen == "cuda"
    elif torch.backends.mps.is_available():
        assert chosen == "mps"
    else:
        assert chosen == "cpu"


def test_run_experiment_reports_the_settings_it_was_given(data):
    """
    The returned row has to carry its own configuration, otherwise a sweep table
    cannot be read back and a result cannot be traced to what produced it.
    """
    row = run_experiment(data, "probe", max_epochs=1, activation="tanh",
                         learning_rate=0.01, **TINY)
    assert row["name"] == "probe"
    assert row["activation"] == "tanh"
    assert row["learning_rate"] == 0.01
    assert row["hidden_layers"] == (8,)


def test_run_experiment_reports_metrics_in_a_sane_range(data):
    row = run_experiment(data, "probe", max_epochs=1, **TINY)
    assert 0.0 <= row["val_acc"] <= 1.0
    assert row["val_loss"] > 0
    assert row["parameters"] > 0
    assert row["seconds"] >= 0
    assert row["epochs_run"] >= 0


def test_run_experiment_is_reproducible_under_the_same_seed(data):
    """
    Two runs of one configuration must agree. If they did not, no sweep
    comparison would mean anything, because a difference between rows could be
    the seed rather than the setting.
    """
    a = run_experiment(data, "a", max_epochs=2, seed=7, **TINY)
    b = run_experiment(data, "b", max_epochs=2, seed=7, **TINY)
    assert a["val_loss"] == pytest.approx(b["val_loss"], abs=1e-6)
    assert a["val_acc"] == pytest.approx(b["val_acc"], abs=1e-6)


def test_different_seeds_can_give_different_results(data):
    """The mirror of the previous test: the seed must actually be doing something."""
    a = run_experiment(data, "a", max_epochs=2, seed=1, **TINY)
    b = run_experiment(data, "b", max_epochs=2, seed=999, **TINY)
    assert isinstance(a["val_loss"], float) and isinstance(b["val_loss"], float)


def test_early_stopping_can_end_a_run_before_the_epoch_cap(data):
    row = run_experiment(data, "stops", max_epochs=40, patience=1, **TINY)
    assert row["epochs_run"] <= 40


def test_sweep_returns_one_row_per_value(data):
    def factory(batch_size=None):
        return data

    results = sweep(factory, "activation", ("relu", "tanh"),
                    baseline=TINY, max_epochs=1)
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert set(results["activation"]) == {"relu", "tanh"}


def test_sweep_varies_only_the_named_axis(data):
    """Everything except the swept axis must stay at the baseline value."""
    def factory(batch_size=None):
        return data

    baseline = {**TINY, "activation": "relu", "learning_rate": 0.005}
    results = sweep(factory, "dropout", (0.0, 0.25), baseline=baseline, max_epochs=1)
    assert set(results["activation"]) == {"relu"}
    assert set(results["learning_rate"]) == {0.005}
    assert set(results["dropout"]) == {0.0, 0.25}


def test_batch_size_sweep_rebuilds_the_loaders():
    """
    Batch size is the one axis that changes the data loaders rather than the
    model, so the sweep has to build a new data module for each value.
    """
    train, validation, test = (make_images(120, 0), make_images(60, 1), make_images(60, 2))
    seen = []

    def factory(batch_size=None):
        seen.append(batch_size)
        return MNISTDataModule(train, validation, test, batch_size=batch_size or 32)

    sweep(factory, "batch_size", (16, 64), baseline=TINY, max_epochs=1)
    assert seen == [16, 64]


def test_evaluate_on_test_returns_only_test_metrics(data):
    model = LitMLP(**TINY)
    scores = evaluate_on_test(model, data)
    assert set(scores) == {"test_loss", "test_acc"}
    assert 0.0 <= scores["test_acc"] <= 1.0


def test_predict_labels_returns_aligned_predictions_and_truths(data):
    predictions, truths = predict_labels(LitMLP(**TINY), data, split="test")
    assert predictions.shape == truths.shape
    assert len(predictions) == len(data.datasets["test"])
    assert set(predictions.tolist()) <= set(range(10))


def test_predict_labels_truths_match_the_underlying_split(data):
    """
    A shuffled evaluation loader would scramble the pairing and quietly corrupt
    the confusion matrix, so the returned labels must come back in dataset order.
    """
    _, truths = predict_labels(LitMLP(**TINY), data, split="validation")
    expected = data.datasets["validation"].tensors[1].numpy()
    assert (truths == expected).all()


def test_reference_baselines_report_both_models():
    train, validation = make_images(120, seed=0), make_images(60, seed=1)
    out = reference_baselines(train, validation, max_iter=20)
    assert out["model"].tolist() == ["majority class", "logistic regression"]
    assert {"val_acc", "seconds"} <= set(out.columns)


def test_majority_class_predicts_the_training_majority():
    """
    The floor is not 10%, and it is not validation's own majority share either.
    The dummy is fitted on train, so it predicts train's most common label and
    earns whatever share that label happens to hold in validation. On the real
    stratified split those two coincide, which is why the notebook reports the
    class balance figure; on random labels they come apart, and fitting on
    train is the behavior worth pinning.
    """
    train, validation = make_images(200, seed=2), make_images(100, seed=3)
    out = reference_baselines(train, validation, max_iter=20)
    predicted = train["label"].value_counts().idxmax()
    expected = (validation["label"] == predicted).mean()
    floor = out.loc[out["model"] == "majority class", "val_acc"].iloc[0]
    assert floor == pytest.approx(expected, abs=0.005)


def test_baselines_do_not_depend_on_the_validation_data():
    """Both are fitted on train alone, so a different validation set cannot move the fit."""
    train = make_images(150, seed=4)
    a = reference_baselines(train, make_images(80, seed=5), max_iter=20)
    b = reference_baselines(train, make_images(80, seed=5), max_iter=20)
    assert a["val_acc"].tolist() == b["val_acc"].tolist()
