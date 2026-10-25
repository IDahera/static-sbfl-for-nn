"""Training and evaluation loops.

Cross-entropy loss with Adam throughout -- one optimizer for all six models, so
that differences between them come from the architecture rather than from the
optimization recipe. The optimizer name and learning rate are carried in
:class:`TrainingReport` and written to the training log, because "which
optimizer produced these weights" is exactly the kind of detail that becomes
unrecoverable later.

Reported quality is accuracy, mean cross-entropy loss and the macro-averaged
precision / recall / F1. Accuracy alone hides per-class collapse, which is a
real failure mode for a 5-epoch run on CIFAR-10.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

#: The optimizer every model in this project is trained with.
OPTIMIZER_NAME = "Adam"


@dataclass
class EvalResult:
    """Quality of a model on one split."""

    accuracy: float           # fraction in [0, 1]
    loss: float               # mean cross-entropy per sample
    macro_precision: float
    macro_recall: float
    macro_f1: float
    n_samples: int
    duration_s: float = 0.0

    def summary(self) -> str:
        """One line for the console log."""
        return (
            f"acc {self.accuracy:.4f}  loss {self.loss:.4f}  "
            f"macro-F1 {self.macro_f1:.4f}  "
            f"(P {self.macro_precision:.4f} / R {self.macro_recall:.4f})  "
            f"on {self.n_samples} samples in {self.duration_s:.1f}s"
        )


@dataclass
class EpochResult:
    """Outcome of a single training epoch."""

    epoch: int          # 1-based
    train_loss: float
    duration_s: float


@dataclass
class TrainingReport:
    """Everything the training script needs to log about one model."""

    key: str
    label: str
    dataset: str
    epochs: List[EpochResult] = field(default_factory=list)
    optimizer: str = OPTIMIZER_NAME
    learning_rate: float = 1e-3
    n_parameters: int = 0
    size_bytes: int = 0
    evaluation: EvalResult = None  # type: ignore[assignment]

    @property
    def total_train_seconds(self) -> float:
        """Wall-clock seconds spent training, summed over epochs."""
        return sum(e.duration_s for e in self.epochs)

    @property
    def avg_epoch_seconds(self) -> float:
        """Mean wall-clock seconds per training epoch."""
        return self.total_train_seconds / len(self.epochs) if self.epochs else 0.0

    @property
    def final_train_loss(self) -> float:
        """Training loss of the last epoch; NaN if no epoch ran."""
        return self.epochs[-1].train_loss if self.epochs else float("nan")


def confusion_matrix(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """``(num_classes, num_classes)`` counts, rows = true class, cols = predicted.

    Built with a single ``bincount`` over the flattened ``true * K + pred``
    index, which is one kernel launch instead of a Python loop over classes.
    """
    index = targets.to(torch.long) * num_classes + predictions.to(torch.long)
    counts = torch.bincount(index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def macro_scores(matrix: torch.Tensor) -> tuple:
    """Macro-averaged ``(precision, recall, f1)`` from a confusion matrix.

    Classes absent from both the predictions and the labels contribute 0 rather
    than NaN, matching scikit-learn's ``zero_division=0``.
    """
    matrix = matrix.double()
    true_positive = matrix.diagonal()
    predicted = matrix.sum(dim=0)
    actual = matrix.sum(dim=1)

    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(true_positive))
    recall = torch.where(actual > 0, true_positive / actual, torch.zeros_like(true_positive))
    denominator = precision + recall
    f1 = torch.where(
        denominator > 0, 2 * precision * recall / denominator, torch.zeros_like(denominator)
    )
    return float(precision.mean()), float(recall.mean()), float(f1.mean())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> float:
    """Train for one epoch; return the mean per-sample loss."""
    model.to(device)
    model.train()

    loss_sum = 0.0
    n_samples = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * targets.size(0)
        n_samples += targets.size(0)

    return loss_sum / n_samples if n_samples else 0.0


def train_epochs(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    device: torch.device,
    learning_rate: float = 1e-3,
) -> Iterator[EpochResult]:
    """Train epoch by epoch, yielding an :class:`EpochResult` after each one.

    Yielding rather than returning lets the caller log progress -- or snapshot
    the model -- between epochs without this function knowing about either.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        loss = train_one_epoch(model, loader, optimizer, device=device)
        yield EpochResult(
            epoch=epoch, train_loss=loss, duration_s=time.perf_counter() - started
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
) -> EvalResult:
    """Evaluate *model* on *loader* without touching its parameters."""
    model.to(device)
    model.eval()
    started = time.perf_counter()

    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    loss_sum = 0.0
    n_samples = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        matrix += confusion_matrix(logits.argmax(dim=1), targets, num_classes)
        n_samples += targets.size(0)

    matrix = matrix.cpu()
    precision, recall, f1 = macro_scores(matrix)
    correct = int(matrix.diagonal().sum())

    return EvalResult(
        accuracy=correct / n_samples if n_samples else 0.0,
        loss=loss_sum / n_samples if n_samples else 0.0,
        macro_precision=precision,
        macro_recall=recall,
        macro_f1=f1,
        n_samples=n_samples,
        duration_s=time.perf_counter() - started,
    )
