# Handwritten digit classification with a multi-layer perceptron

**Module 4 · Sprint 1: Deep Learning Fundamentals** | Turing College · Data Science Program

A fully connected network that classifies MNIST digits, built and tuned with PyTorch Lightning. The point of the sprint is to see how far a model with no spatial prior gets on an image task, and what each training choice actually buys.

---

## Result

**98.2% accuracy on the held-out test split**, from a two-layer network with about 235,000 parameters trained in under a minute on a laptop GPU.

Two non-neural reference points put that number on a scale:

| Model | Validation accuracy |
|---|---|
| Predicting the most common digit | 11.2% |
| Multinomial logistic regression | 91.4% |
| Tuned two-layer MLP | 97.8% |

The hidden layers are worth about six and a half points over the best straight-line model in pixel space, which is the part of the result that the architecture actually earns.

The tuning moved the result by well under a point over a sensible default, which is the honest scale of what hyperparameter search bought here. The sweeps were more informative than the final number:

| Sweep | Finding |
|---|---|
| Optimizer | Adam beats plain SGD; per-parameter step sizes suit an input where most pixels are always zero |
| Batch size | Strongly affects throughput; the smallest batch is the *slowest*. Its apparent flatness in accuracy turned out to be an artifact of holding the learning rate fixed, see below |
| Activation | Modern activations tie; sigmoid needs far more epochs, a vanishing gradient visible at two layers |
| Learning rate | The sharpest axis. At 0.1 training collapses to near the 10% random floor |
| Dropout | A moderate rate helps; heavy dropout costs accuracy and epochs |
| Architecture | Capacity is not the constraint. One layer of 128 units matches a network six times larger |
| Loss | Cross-entropy and NLL are identical by construction; label smoothing wins on accuracy but its loss is not comparable |

### One axis at a time is not enough

Those sweeps each move one axis and hold the rest at a baseline, which cannot see an interaction. Crossing learning rate with batch size finds one: a gradient averaged over 512 images is quieter than one averaged over 32 and tolerates a longer step, so the best rate is not the same at both ends of the range. The interaction runs in both directions: raising the rate costs the smallest batch about nine tenths of a point and *gains* the largest batch about seven tenths. The flat batch-size reading above is a property of the single rate that sweep held fixed, and the best combination turns out to sit at the small-batch end it could not reach.

The project therefore searches jointly as well: a full batch-size by learning-rate grid, and an Optuna study that varies six axes at once using a TPE sampler rather than a grid.

### Best epoch, not last epoch

Early stopping trains a few epochs past the minimum before it gives up, so the weights sitting in memory when training returns are not the ones that were checkpointed. Every run now loads its saved best epoch back before being scored, which means the validation number, the test number, and the confusion matrix all describe the same network.

What limits the model is the `Flatten` layer. Turning the image into 784 independent numbers discards the fact that neighboring pixels are related, and no width or depth recovers it.

---

## Repository structure

```
.
├── data/raw/train.csv.gz           <- the labeled data, gzipped: 9 MB rather than 77 MB
├── notebooks/mnist_mlp.ipynb       <- the deliverable
├── src/mnist_mlp/                  <- all logic; the notebook stays thin
│   ├── config.py                   <- every constant and all seven sweep grids
│   ├── dataset.py                  <- download, audit, clean, stratified split
│   ├── datamodule.py               <- Lightning data module, train-only normalization
│   ├── models.py                   <- the configurable MLP and its Lightning wrapper
│   ├── experiments.py              <- one-config runner, sweep loop, test evaluation
│   └── plots.py                    <- every figure
├── tests/                          <- 68 tests, including leakage proofs
├── references/data_dictionary.md
├── reports/figures/
├── pyproject.toml                  <- dependencies and ruff config
└── uv.lock                         <- exact pinned resolution, committed
```

---

## Guarding against leakage

Two orderings in this pipeline are correctness matters rather than tidiness, and both are enforced by tests:

1. **Deduplication happens before the split.** If a duplicated image sat in train and test, part of the test score would be a memory check. `tests/test_leakage.py` proves this, including a negative control showing the wrong order does leak.
2. **Normalization statistics come from the training split only** and apply unchanged to validation and test. Deriving them from the whole file would let the test distribution shape every training batch, silently.

Model selection uses validation throughout. The test split is opened once, for the single configuration validation chose.

---

## How to run

```bash
cd "Module 4/Sprint 1"
uv sync --extra dev
uv run pytest -q          # 68 tests
uv run ruff check .       # package, tests, and notebook
uv run jupyter lab notebooks/mnist_mlp.ipynb
```

The first run downloads from Kaggle and needs an API token at `~/.kaggle/access_token` (or `~/.kaggle/kaggle.json`) plus the competition rules accepted. Later runs read the CSV cache, so no network is needed.

**On the GPU.** `pick_accelerator()` prefers CUDA, falls back to Apple's Metal backend (`mps`), then CPU, and the notebook prints which one ran. These results were produced on Metal; the same code runs unchanged on a Colab or Kaggle CUDA box.

**On Kaggle's `test.csv`.** It ships 28,000 unlabeled images for leaderboard scoring, so it cannot measure anything. All three splits come out of the 42,000 labeled rows in `train.csv`, which is also the only way to get a genuine 60/20/20.

---

## Sprint coverage

- Three-partition split, 60/20/20, stratified and seeded
- A training pipeline where architecture, activation, dropout, optimizer, learning rate, batch size, and loss are all constructor arguments
- Sweeps across all seven of those axes, each varied against a fixed baseline
- Early stopping and best-epoch checkpointing
- Class weighting available for imbalance, measured and reported as unnecessary here
- GPU training, with the device reported rather than assumed
- Error analysis: confusion matrix, per-digit accuracy, and the misclassified images
