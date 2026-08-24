"""The MLP builder and the Lightning module: shapes, wiring, and rejected inputs."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mnist_mlp.config import N_CLASSES, N_PIXELS
from mnist_mlp.models import ACTIVATIONS, LitMLP, build_mlp


def test_architecture_matches_the_requested_widths():
    model = build_mlp(hidden_layers=(64, 32), activation="relu", dropout=0.0)
    widths = [m.out_features for m in model if isinstance(m, nn.Linear)]
    assert widths == [64, 32, N_CLASSES]


def test_dropout_layers_appear_only_when_requested():
    with_dropout = build_mlp(hidden_layers=(32, 16), dropout=0.3)
    without = build_mlp(hidden_layers=(32, 16), dropout=0.0)
    assert sum(isinstance(m, nn.Dropout) for m in with_dropout) == 2
    assert sum(isinstance(m, nn.Dropout) for m in without) == 0


def test_every_hidden_layer_gets_an_activation():
    """
    A stack of Linear layers without activations collapses to a single linear
    map, which is the whole reason activations exist.
    """
    model = build_mlp(hidden_layers=(32, 16, 8), activation="tanh", dropout=0.0)
    assert sum(isinstance(m, nn.Tanh) for m in model) == 3


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_each_activation_builds_and_runs(name):
    model = build_mlp(hidden_layers=(16,), activation=name, dropout=0.0)
    out = model(torch.randn(4, N_PIXELS))
    assert out.shape == (4, N_CLASSES)
    assert torch.isfinite(out).all()


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="unknown activation"):
        build_mlp(activation="banana")


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_impossible_dropout_is_rejected(bad):
    with pytest.raises(ValueError, match="dropout"):
        build_mlp(dropout=bad)


def test_forward_returns_logits_not_probabilities():
    """
    The head is linear on purpose: cross-entropy expects logits and folds the
    log-softmax in with it, which is numerically safer than exponentiating first.
    """
    out = LitMLP(hidden_layers=(16,))(torch.randn(8, N_PIXELS))
    assert out.shape == (8, N_CLASSES)
    assert not torch.allclose(out.exp().sum(dim=1), torch.ones(8)), "output must not be normalized"


@pytest.mark.parametrize("optimizer,expected", [
    ("sgd", torch.optim.SGD),
    ("sgd_momentum", torch.optim.SGD),
    ("adam", torch.optim.Adam),
    ("adamw", torch.optim.AdamW),
    ("rmsprop", torch.optim.RMSprop),
])
def test_each_optimizer_is_constructed(optimizer, expected):
    opt = LitMLP(hidden_layers=(16,), optimizer=optimizer).configure_optimizers()
    assert isinstance(opt, expected)


def test_momentum_variant_actually_sets_momentum():
    plain = LitMLP(hidden_layers=(16,), optimizer="sgd").configure_optimizers()
    with_momentum = LitMLP(hidden_layers=(16,), optimizer="sgd_momentum").configure_optimizers()
    assert plain.param_groups[0]["momentum"] == 0
    assert with_momentum.param_groups[0]["momentum"] == pytest.approx(0.9)


def test_weight_decay_reaches_the_optimizer():
    """Weight decay is L2 regularization applied at the update, not added to the loss."""
    opt = LitMLP(hidden_layers=(16,), weight_decay=0.05).configure_optimizers()
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.05)


def test_learning_rate_reaches_the_optimizer():
    opt = LitMLP(hidden_layers=(16,), learning_rate=0.007).configure_optimizers()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.007)


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="unknown optimizer"):
        LitMLP(hidden_layers=(16,), optimizer="rocket").configure_optimizers()


@pytest.mark.parametrize("loss,expected", [
    ("cross_entropy", nn.CrossEntropyLoss),
    ("label_smoothing", nn.CrossEntropyLoss),
    ("nll", nn.NLLLoss),
])
def test_each_loss_is_constructed(loss, expected):
    assert isinstance(LitMLP(hidden_layers=(16,), loss=loss).criterion, expected)


def test_label_smoothing_is_actually_applied():
    plain = LitMLP(hidden_layers=(16,), loss="cross_entropy").criterion
    smoothed = LitMLP(hidden_layers=(16,), loss="label_smoothing").criterion
    assert plain.label_smoothing == 0.0
    assert smoothed.label_smoothing > 0.0


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="unknown loss"):
        LitMLP(hidden_layers=(16,), loss="hinge")


def test_class_weights_are_handed_to_the_criterion():
    weights = torch.arange(1, N_CLASSES + 1, dtype=torch.float32)
    model = LitMLP(hidden_layers=(16,), class_weights=weights)
    assert torch.equal(model.criterion.weight, weights)


def test_nll_path_receives_log_probabilities():
    """
    NLLLoss expects log-probabilities while cross-entropy expects logits. Feeding
    raw logits to NLL would train against a wrong objective without erroring, so
    the step has to apply log_softmax for that branch only.
    """
    model = LitMLP(hidden_layers=(16,), loss="nll")
    batch = (torch.randn(6, N_PIXELS), torch.randint(0, N_CLASSES, (6,)))
    loss = model._shared_step(batch, "train")
    assert torch.isfinite(loss) and loss > 0


def test_parameter_count_matches_the_hand_calculation():
    # 784*32 + 32 for the first layer, 32*10 + 10 for the head.
    model = LitMLP(hidden_layers=(32,), dropout=0.0)
    assert model.count_parameters() == (N_PIXELS * 32 + 32) + (32 * N_CLASSES + N_CLASSES)


def test_a_training_step_reduces_the_loss_on_one_repeated_batch():
    """A model that cannot overfit a single batch is wired wrong."""
    torch.manual_seed(0)
    model = LitMLP(hidden_layers=(64,), dropout=0.0, learning_rate=0.05)
    opt = model.configure_optimizers()
    x = torch.randn(32, N_PIXELS)
    y = torch.randint(0, N_CLASSES, (32,))

    first = model.criterion(model(x), y).item()
    for _ in range(60):
        opt.zero_grad()
        model.criterion(model(x), y).backward()
        opt.step()
    assert model.criterion(model(x), y).item() < first
