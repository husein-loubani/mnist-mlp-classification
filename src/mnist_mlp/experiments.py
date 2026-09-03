"""
The sweep runner.

The brief asks for several hyperparameters to be tested. Doing that by editing a
cell and rerunning it by hand is how results get mixed up, so one function
trains one configuration and returns a row, and the sweeps are loops over that
function. Every run starts from the same seed and the same data, so a difference
between two rows is caused by the thing that was varied.

Model selection happens on validation. The test split is opened once, at the
end, for the single configuration that validation chose.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from mnist_mlp.config import (
    EARLY_STOPPING_PATIENCE,
    JOINT_BATCH_GRID,
    JOINT_LR_GRID,
    LOGREG_MAX_ITER,
    MAX_EPOCHS,
    MONITOR_METRIC,
    MONITOR_MODE,
    OPTUNA_MAX_EPOCHS,
    OPTUNA_TRIALS,
    PIXEL_MAX,
    RANDOM_SEED,
    SEED_GRID,
)
from mnist_mlp.datamodule import MNISTDataModule, resolve_device
from mnist_mlp.dataset import to_arrays
from mnist_mlp.models import LitMLP


def pick_accelerator() -> str:
    """
    Prefer a GPU, whichever kind is present.

    On Apple Silicon that is Metal (`mps`); on a Colab or Kaggle box it would be
    `cuda`. Both are GPU training, and the notebook prints which one actually
    ran rather than claiming one.
    """
    return resolve_device().type



def load_best_weights(model: LitMLP, checkpoint: ModelCheckpoint) -> int | None:
    """
    Copy the best saved epoch's weights back into `model`, in place.

    Returns the epoch that checkpoint came from, or None when nothing was
    saved (a run of a single epoch that never completed a validation pass).
    Loading in place rather than returning a fresh object keeps every later
    reference to `model` pointing at the network that was actually scored.
    """
    path = checkpoint.best_model_path
    if not path:
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])
    return saved.get("epoch")

def run_experiment(
    data: MNISTDataModule,
    name: str,
    max_epochs: int = MAX_EPOCHS,
    seed: int = RANDOM_SEED,
    patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_dir: str | None = None,
    verbose: bool = False,
    return_model: bool = False,
    **model_kwargs,
) -> dict:
    """
    Train one configuration and report its validation result.

    Early stopping ends a run once validation loss stops improving, which keeps
    a sweep from spending GPU time on configurations that already peaked. The
    returned row carries the settings alongside the metrics so the sweep table
    is self-describing.
    """
    seed_everything(seed, workers=True)

    model = LitMLP(**model_kwargs)
    # Checkpointing is unconditional. Early stopping deliberately trains past the
    # best epoch before it gives up, so the weights left in memory when fit()
    # returns are `patience` epochs the wrong side of the minimum. Scoring those
    # would report a model nobody would choose to keep.
    with ExitStack() as stack:
        target = checkpoint_dir or stack.enter_context(TemporaryDirectory())
        checkpoint = ModelCheckpoint(
            dirpath=target, filename=name, monitor=MONITOR_METRIC,
            mode=MONITOR_MODE, save_top_k=1,
        )
        callbacks = [
            EarlyStopping(monitor=MONITOR_METRIC, mode=MONITOR_MODE, patience=patience),
            checkpoint,
        ]

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=pick_accelerator(),
            devices=1,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=verbose,
            enable_model_summary=False,
            deterministic=False,   # MPS lacks deterministic kernels for some ops
        )

        started = time.perf_counter()
        trainer.fit(model, datamodule=data)
        elapsed = time.perf_counter() - started

        # Load the saved best epoch back over the in-memory weights, so the row
        # below and any later use of `model` describe the same network.
        best_epoch = load_best_weights(model, checkpoint)
        scores = trainer.validate(model, datamodule=data, verbose=False)[0]

    row = {
        "name": name,
        **{k: v for k, v in model_kwargs.items() if k != "class_weights"},
        "val_loss": round(scores["val_loss"], 4),
        "val_acc": round(scores["val_acc"], 4),
        "epochs_run": trainer.current_epoch,
        "best_epoch": best_epoch,
        "parameters": model.count_parameters(),
        "seconds": round(elapsed, 1),
    }
    return (row, model) if return_model else row


def sweep(
    data_factory,
    axis: str,
    values,
    baseline: dict | None = None,
    max_epochs: int = MAX_EPOCHS,
    **kwargs,
) -> pd.DataFrame:
    """
    Vary one axis against a fixed baseline and collect the results.

    `data_factory(batch_size)` returns a data module, because batch size is the
    one axis that changes the loaders rather than the model, and it has to be
    swept the same way as the rest.
    """
    baseline = dict(baseline or {})
    rows = []
    for value in values:
        settings = {**baseline, axis: value}
        batch_size = settings.pop("batch_size", None)
        data = data_factory(batch_size) if batch_size else data_factory(None)
        label = f"{axis}={value}"
        row = run_experiment(data, label, max_epochs=max_epochs, **settings, **kwargs)
        row[axis] = value
        if batch_size:
            row["batch_size"] = batch_size
        rows.append(row)
        print(f"  {label:38s} val_acc={row['val_acc']:.4f}  val_loss={row['val_loss']:.4f}  "
              f"{row['epochs_run']} epochs  {row['seconds']}s")
    return pd.DataFrame(rows)



def reference_baselines(train: pd.DataFrame, validation: pd.DataFrame,
                        max_iter: int = LOGREG_MAX_ITER) -> pd.DataFrame:
    """
    Two non-neural reference points, so the network's accuracy has a scale.

    Predicting the most common class is the floor any classifier has to clear,
    and it lands at the majority share rather than at 10% because the classes
    are not perfectly even. Multinomial logistic regression is the strongest
    model that can only draw straight boundaries in pixel space, so the gap
    above it is what the hidden layers are actually buying.

    Pixels are scaled by a constant rather than by train statistics, so neither
    baseline depends on anything fitted to data.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    x_train, y_train = to_arrays(train)
    x_val, y_val = to_arrays(validation)
    x_train, x_val = x_train / PIXEL_MAX, x_val / PIXEL_MAX

    rows = []
    for name, model in (
        ("majority class", DummyClassifier(strategy="most_frequent")),
        ("logistic regression", LogisticRegression(max_iter=max_iter, n_jobs=-1)),
    ):
        started = time.perf_counter()
        model.fit(x_train, y_train)
        rows.append({
            "model": name,
            "val_acc": round(float(model.score(x_val, y_val)), 4),
            "seconds": round(time.perf_counter() - started, 1),
        })
    return pd.DataFrame(rows)


def joint_sweep(data_factory, batch_sizes=JOINT_BATCH_GRID, learning_rates=JOINT_LR_GRID,
                baseline: dict | None = None, max_epochs: int = MAX_EPOCHS) -> pd.DataFrame:
    """
    Every batch size against every learning rate, rather than one axis at a time.

    A one-factor sweep holds the learning rate fixed while the batch size moves,
    so it measures the batch size *at that one rate* and silently attributes the
    interaction to the batch. Crossing the two axes separates them: each row of
    the returned table is one batch size, each column one rate, and the best
    rate is allowed to differ between rows.
    """
    baseline = dict(baseline or {})
    baseline.pop("learning_rate", None)
    rows = []
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            row = run_experiment(
                data_factory(batch_size), f"b{batch_size}_lr{learning_rate:g}",
                max_epochs=max_epochs, learning_rate=learning_rate, **baseline,
            )
            row["batch_size"] = batch_size
            rows.append(row)
    return pd.DataFrame(rows)


def joint_grid_table(results: pd.DataFrame, metric: str = "val_acc") -> pd.DataFrame:
    """The joint sweep as batch size by learning rate, for reading off interactions."""
    return results.pivot(index="batch_size", columns="learning_rate", values=metric)


def optuna_search(data_factory, n_trials: int = OPTUNA_TRIALS, seed: int = RANDOM_SEED,
                  max_epochs: int = OPTUNA_MAX_EPOCHS) -> tuple[pd.DataFrame, dict]:
    """
    Search the axes jointly with Optuna instead of one at a time.

    A grid over six axes is the product of their lengths and mostly spends its
    budget in regions the first few trials already ruled out. Optuna's TPE
    sampler models which regions produced good scores and draws later trials
    from there, so the same number of runs covers the promising part of the
    space far more densely. The seed is fixed so the search is reproducible.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", list(JOINT_BATCH_GRID))
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "elu"]),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd_momentum"]),
            "hidden_layers": trial.suggest_categorical("hidden_layers", [(128,), (256, 128), (512, 256)]),
        }
        row = run_experiment(data_factory(batch_size), f"trial{trial.number}",
                             max_epochs=max_epochs, **params)
        trial.set_user_attr("val_acc", row["val_acc"])
        trial.set_user_attr("epochs_run", row["epochs_run"])
        return row["val_loss"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    frame = pd.DataFrame([
        {"trial": t.number, "val_loss": round(t.value, 4),
         "val_acc": t.user_attrs.get("val_acc"), **t.params}
        for t in study.trials if t.value is not None
    ]).sort_values("val_loss").reset_index(drop=True)
    return frame, study.best_params

def evaluate_on_test(model: LitMLP, data: MNISTDataModule) -> dict:
    """
    The single test-set evaluation, run once for the chosen configuration.

    Everything before this point was selected on validation. Calling this more
    than once, on more than one candidate, would turn the test split into a
    second validation set and the reported accuracy into an optimistic one.
    """
    trainer = Trainer(
        accelerator=pick_accelerator(), devices=1, logger=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    scores = trainer.test(model, datamodule=data, verbose=False)[0]
    return {"test_loss": round(scores["test_loss"], 4), "test_acc": round(scores["test_acc"], 4)}


def predict_labels(model: LitMLP, data: MNISTDataModule, split: str = "test"):
    """Predicted and true labels for a split, for the confusion matrix."""
    loader = {"train": data.train_dataloader, "validation": data.val_dataloader,
              "test": data.test_dataloader}[split]()
    device = pick_accelerator()
    model = model.to(device).eval()

    predictions, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            predictions.append(logits.argmax(dim=1).cpu())
            # The labels now come off the training device too, so they need the
            # same trip back to the host before numpy will touch them.
            truths.append(y.cpu())
    return torch.cat(predictions).numpy(), torch.cat(truths).numpy()


def seed_stability(
    data,
    settings: dict,
    seeds: tuple = SEED_GRID,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    name: str = "candidate",
) -> pd.DataFrame:
    """
    Retrain one configuration under several seeds and report the spread.

    A sweep that runs each configuration once cannot tell a real difference from
    run-to-run noise, and on this task the candidates are separated by fractions
    of a point. Training the finalists repeatedly turns "A beat B" into a
    statement with a standard deviation attached, which is the difference
    between a ranking and a coincidence.
    """
    rows = [
        run_experiment(data, f"{name}_seed{seed}", max_epochs=max_epochs,
                       seed=seed, patience=patience, **settings)
        for seed in seeds
    ]
    frame = pd.DataFrame(rows)
    return frame[["name", "val_acc", "val_loss", "epochs_run", "seconds"]]


def stability_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Mean and spread of validation accuracy per configuration.

    Two candidates whose intervals overlap are tied on this evidence, however
    their single-run numbers happened to order them.
    """
    rows = []
    for name, frame in frames.items():
        accuracy = frame["val_acc"]
        rows.append({
            "configuration": name,
            "seeds": len(accuracy),
            "mean_val_acc": round(float(accuracy.mean()), 4),
            "std_val_acc": round(float(accuracy.std(ddof=1)), 4),
            "min_val_acc": round(float(accuracy.min()), 4),
            "max_val_acc": round(float(accuracy.max()), 4),
        })
    return (
        pd.DataFrame(rows)
        .set_index("configuration")
        .sort_values("mean_val_acc", ascending=False)
    )
