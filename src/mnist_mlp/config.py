"""
Global constants for the MNIST multi-layer perceptron project.

Every magic number and string lives here so the notebook carries no hardcoded
literals and one change propagates everywhere.
"""

RANDOM_SEED = 42

# Kaggle's Digit Recognizer competition. Its test.csv is unlabeled because it
# is the leaderboard set, so every split below is carved out of train.csv.
KAGGLE_COMPETITION = "digit-recognizer"
TRAIN_FILE = "train.csv"
# The repository ships the gzipped copy: 9 MB instead of 77 MB, and pandas
# decompresses it on read. A fresh Kaggle download produces the plain CSV.
TRAIN_FILE_GZ = "train.csv.gz"

IMAGE_SIZE = 28
N_PIXELS = IMAGE_SIZE * IMAGE_SIZE   # 784 input features
N_CLASSES = 10
PIXEL_MAX = 255.0

# Three-way split, as the brief requires. Stratified so every digit keeps its
# share in all three parts, and deterministic so a rerun reproduces it exactly.
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20

# Cleaning thresholds.
BLANK_PIXEL_TOLERANCE = 0     # an image whose pixels are all this value is empty
MIN_INK_PIXELS = 10           # fewer non-zero pixels than this is not a digit

# Training defaults. Every one of these is a knob the sweep varies, so they are
# starting points rather than final answers.
BATCH_SIZE = 128
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
HIDDEN_LAYERS = (256, 128)
ACTIVATION = "relu"
OPTIMIZER = "adam"
LOSS = "cross_entropy"

# Early stopping and checkpointing, so a sweep does not waste GPU time on runs
# that stopped improving.
# Seeds for the stability check. One run per configuration cannot separate a
# real difference from run-to-run noise, and the finalists here sit fractions of
# a point apart, so the shortlist is retrained across all of these.
SEED_GRID = (42, 43, 44, 45, 46)

EARLY_STOPPING_PATIENCE = 5
MONITOR_METRIC = "val_loss"
MONITOR_MODE = "min"

NUM_WORKERS = 0   # 0 avoids worker-startup overhead dominating small MNIST batches

# Sweep grids. Each axis is varied against the same baseline so the effect of
# one choice is not confounded with another.
OPTIMIZER_GRID = ("sgd", "sgd_momentum", "adam", "adamw", "rmsprop")
BATCH_SIZE_GRID = (32, 64, 128, 256, 512)
ACTIVATION_GRID = ("relu", "leaky_relu", "tanh", "sigmoid", "gelu", "elu")
LEARNING_RATE_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1)
DROPOUT_GRID = (0.0, 0.1, 0.2, 0.3, 0.5)
ARCHITECTURE_GRID = (
    (128,),
    (256, 128),
    (512, 256, 128),
    (1024, 512, 256, 128),
    (256, 256, 256),
)
LOSS_GRID = ("cross_entropy", "nll", "label_smoothing")

# Plotting.
PALETTE_PRIMARY = "#4C72B0"
PALETTE_ACCENT = "#DD8452"
PALETTE_LIST = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                "#CCB974", "#64B5CD", "#8C8C8C", "#937860", "#DA8BC3"]
CMAP_SEQ = "Blues"

# Reference baselines. 1000 iterations is where multinomial logistic regression
# stops moving on this data; 200 leaves it short and raises a ConvergenceWarning.
LOGREG_MAX_ITER = 1000

# Joint search. The one-factor-at-a-time sweeps above hold every other axis at
# the baseline, which cannot see an interaction: learning rate and batch size
# are the textbook pair, since a larger batch averages more samples per step and
# needs a proportionally larger step to cover the same ground per epoch.
JOINT_BATCH_GRID = (32, 64, 128, 256, 512)
JOINT_LR_GRID = (3e-4, 1e-3, 3e-3)
OPTUNA_TRIALS = 20
# A search pass runs on a shorter budget than the final fit: it only has to rank
# configurations, not squeeze the last fraction out of the winner, which is then
# retrained at full length in section 9.
OPTUNA_MAX_EPOCHS = 15

# The Optuna search space, here rather than inside the search function so every
# grid in this project lives in one file and can be read without opening code.
OPTUNA_SPACE = {
    "batch_size": JOINT_BATCH_GRID,
    "learning_rate": (1e-4, 1e-2),
    "dropout": (0.0, 0.5),
    "dropout_step": 0.1,
    "activation": ("relu", "gelu", "elu"),
    "optimizer": ("adam", "adamw", "sgd_momentum"),
    "hidden_layers": ((128,), (256, 128), (512, 256)),
}
