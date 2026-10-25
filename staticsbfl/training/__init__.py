"""Training and evaluation."""

from staticsbfl.training.trainer import (
    OPTIMIZER_NAME,
    EpochResult,
    EvalResult,
    TrainingReport,
    confusion_matrix,
    evaluate,
    macro_scores,
    train_epochs,
    train_one_epoch,
)

__all__ = [
    "OPTIMIZER_NAME",
    "EpochResult",
    "EvalResult",
    "TrainingReport",
    "confusion_matrix",
    "evaluate",
    "macro_scores",
    "train_epochs",
    "train_one_epoch",
]
