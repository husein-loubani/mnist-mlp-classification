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

import pandas as pd
import pytest
import torch

from mnist_mlp.datamodule import MNISTDataModule
from mnist_mlp.experiments import (
    evaluate_on_test,
    joint_grid_table,
    joint_sweep,
    load_best_weights,
    per_class_report,
    pick_accelerator,
    predict_labels,
    reference_baselines,
    run_experiment,
    sweep,
    top_confusions,
)
from mnist_mlp.models import LitMLP
from tests.conftest import make_images

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
    expected = data.datasets["validation"].tensors[1].cpu().numpy()
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


def test_run_experiment_reports_the_best_epoch_not_the_last():
    """
    Early stopping trains `patience` epochs past the minimum before it gives
    up, so the weights in memory when fit() returns are not the ones that were
    saved. The row must describe the checkpoint, which means best_epoch can sit
    strictly earlier than the last epoch run.
    """
    data = MNISTDataModule(make_images(200, seed=0), make_images(100, seed=1),
                           make_images(100, seed=2), batch_size=32)
    row = run_experiment(data, "probe", max_epochs=6, patience=2)
    assert "best_epoch" in row
    assert row["best_epoch"] is not None
    assert row["best_epoch"] <= row["epochs_run"]


def test_returned_model_carries_the_checkpoint_weights(tmp_path):
    """
    Every parameter of the returned model must equal the saved checkpoint.

    This is the test that actually guards the fix. Asserting that `best_epoch`
    is reported, or that the metrics are finite, passes just as happily when the
    weights are never loaded, because those values are read from the checkpoint
    file rather than from the network. Only comparing the live parameters
    against the file can tell the two situations apart: with early stopping the
    in-memory weights are `patience` epochs later than the ones on disk, so if
    the load is removed these tensors diverge.
    """
    import torch

    data = MNISTDataModule(make_images(200, seed=3), make_images(100, seed=4),
                           make_images(100, seed=5), batch_size=32)
    row, model = run_experiment(data, "probe", max_epochs=8, patience=2,
                                return_model=True, checkpoint_dir=str(tmp_path))

    saved = list(tmp_path.glob("*.ckpt"))
    assert saved, "no checkpoint was written"
    state = torch.load(saved[0], map_location="cpu", weights_only=False)["state_dict"]

    live = model.state_dict()
    assert set(live) == set(state)
    for name, tensor in live.items():
        assert torch.equal(tensor.cpu(), state[name].cpu()), f"{name} is not the saved epoch"

    scores = evaluate_on_test(model, data)
    assert set(scores) == {"test_loss", "test_acc"}
    assert row["best_epoch"] is not None


def test_load_best_weights_is_a_no_op_without_a_checkpoint():
    from lightning.pytorch.callbacks import ModelCheckpoint

    from mnist_mlp.models import LitMLP

    empty = ModelCheckpoint()
    assert load_best_weights(LitMLP(), empty) is None


def test_joint_sweep_crosses_both_axes():
    """Every batch size meets every learning rate, which one-factor sweeps cannot do."""
    def factory(batch_size=None):
        return MNISTDataModule(make_images(120, seed=6), make_images(60, seed=7),
                               make_images(60, seed=8), batch_size=batch_size or 32)

    out = joint_sweep(factory, batch_sizes=(32, 64), learning_rates=(1e-3, 3e-3),
                      max_epochs=2)
    assert len(out) == 4
    assert set(out["batch_size"]) == {32, 64}
    assert set(out["learning_rate"]) == {1e-3, 3e-3}


def test_joint_grid_table_is_batch_by_learning_rate():
    def factory(batch_size=None):
        return MNISTDataModule(make_images(120, seed=9), make_images(60, seed=10),
                               make_images(60, seed=11), batch_size=batch_size or 32)

    table = joint_grid_table(joint_sweep(factory, batch_sizes=(32, 64),
                                         learning_rates=(1e-3, 3e-3), max_epochs=2))
    assert list(table.index) == [32, 64]
    assert list(table.columns) == [1e-3, 3e-3]


def test_per_class_report_counts_errors_as_integers():
    """
    The error column must be a difference of counts, not a rounded accuracy
    multiplied back by the support. Nine of ten correct rounds to 0.9 exactly,
    but a support that does not divide cleanly is where the derivation slips.
    """
    import numpy as np

    y_true = np.array([0] * 7 + [1] * 3)
    y_pred = np.array([0] * 6 + [1] + [1] * 3)      # one 0 predicted as 1
    out = per_class_report(y_true, y_pred)
    assert out.loc[0, "images"] == 7
    assert out.loc[0, "errors"] == 1
    assert out.loc[1, "errors"] == 0
    assert out["errors"].dtype.kind in "iu"


def test_per_class_report_accuracy_is_recall():
    import numpy as np

    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    out = per_class_report(y_true, y_pred)
    assert out.loc[0, "accuracy"] == pytest.approx(0.75)
    assert out.loc[1, "accuracy"] == pytest.approx(1.0)


def test_top_confusions_excludes_the_diagonal_and_sorts():
    import numpy as np

    y_true = np.array([3] * 5 + [8] * 3 + [1] * 4)
    y_pred = np.array([5] * 4 + [3] + [1] * 3 + [1] * 4)
    out = top_confusions(y_true, y_pred, n=5)
    assert (out["true"] != out["predicted"]).all(), "a correct prediction is not a confusion"
    assert out["count"].is_monotonic_decreasing
    assert out.iloc[0].to_dict() == {"true": 3, "predicted": 5, "count": 4}
