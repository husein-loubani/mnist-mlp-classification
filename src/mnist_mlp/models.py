"""
The configurable multi-layer perceptron.

Every axis the brief asks to vary (hidden architecture, activation, dropout,
optimizer, learning rate, loss) is a constructor argument, so a sweep changes
one keyword rather than editing a model definition. That is the "easily
customized pipeline" requirement taken literally.
"""

from __future__ import annotations

import torch
import torchmetrics
from lightning import LightningModule
from torch import nn

from mnist_mlp.config import (
    ACTIVATION,
    DROPOUT,
    HIDDEN_LAYERS,
    LEARNING_RATE,
    LOSS,
    N_CLASSES,
    N_PIXELS,
    OPTIMIZER,
    WEIGHT_DECAY,
)

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}


def build_mlp(
    hidden_layers: tuple[int, ...] = HIDDEN_LAYERS,
    activation: str = ACTIVATION,
    dropout: float = DROPOUT,
    n_inputs: int = N_PIXELS,
    n_classes: int = N_CLASSES,
) -> nn.Sequential:
    """
    Stack Linear -> activation -> dropout blocks and finish with a linear head.

    The head emits raw logits rather than probabilities. Cross-entropy expects
    logits and folds the log-softmax in with it, which is both numerically
    safer than exponentiating first and one fewer thing to get wrong.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}, expected one of {sorted(ACTIVATIONS)}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")

    layers: list[nn.Module] = [nn.Flatten()]
    in_features = n_inputs
    for width in hidden_layers:
        layers.append(nn.Linear(in_features, width))
        layers.append(ACTIVATIONS[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_features = width
    layers.append(nn.Linear(in_features, n_classes))
    return nn.Sequential(*layers)


class LitMLP(LightningModule):
    """
    Lightning wrapper: the training loop, the metrics, and the optimizer choice.

    Metrics come from torchmetrics rather than hand-counting correct predictions,
    so accumulation across batches and devices is handled by a tested component.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = HIDDEN_LAYERS,
        activation: str = ACTIVATION,
        dropout: float = DROPOUT,
        optimizer: str = OPTIMIZER,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        loss: str = LOSS,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.1,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = build_mlp(hidden_layers, activation, dropout)
        self.loss_name = loss
        self.criterion = self._build_criterion(loss, class_weights, label_smoothing)

        metrics = {
            stage: torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
            for stage in ("train", "val", "test")
        }
        self.train_accuracy, self.val_accuracy, self.test_accuracy = metrics.values()

    @staticmethod
    def _build_criterion(loss: str, class_weights, label_smoothing: float) -> nn.Module:
        """
        Map a loss name onto a module.

        `nll` expects log-probabilities, so it is paired with a log-softmax in
        the step rather than being handed raw logits. Cross-entropy applies that
        transform internally, which is why the two need different treatment even
        though they optimize the same objective.
        """
        if loss == "cross_entropy":
            return nn.CrossEntropyLoss(weight=class_weights)
        if loss == "label_smoothing":
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        if loss == "nll":
            return nn.NLLLoss(weight=class_weights)
        raise ValueError(f"unknown loss {loss!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        inputs = torch.log_softmax(logits, dim=1) if self.loss_name == "nll" else logits
        loss = self.criterion(inputs, y)

        accuracy = {"train": self.train_accuracy, "val": self.val_accuracy,
                    "test": self.test_accuracy}[stage]
        accuracy(logits, y)
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"), on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x).argmax(dim=1)

    def configure_optimizers(self):
        """
        Weight decay is passed to the optimizer rather than added to the loss by
        hand, which is what the review question about weight decay is really
        asking: it is L2 regularization applied at the update step.
        """
        name, params = self.hparams.optimizer, self.parameters()
        lr, wd = self.hparams.learning_rate, self.hparams.weight_decay

        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd)
        if name == "sgd_momentum":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
        raise ValueError(f"unknown optimizer {name!r}")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
