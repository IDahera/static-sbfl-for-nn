"""Dataset loading: CIFAR-10, and Fashion-MNIST as the control population.

Both datasets are downloaded into ``cache/data/`` on first use and read from
there afterwards. Loaders are seeded (``seed=42`` by default) so batch order --
and therefore training -- is reproducible.

No data augmentation is applied. Two reasons: the pipeline stays deterministic,
and the resulting models keep a healthy misclassification rate on the test set.
That second point matters more than it looks. The hit spectrum is built from the
contrast between failing and successful samples, so a model with almost no
failures produces ``a_f = 0`` almost everywhere and a degenerate, all-zero
ranking. Some headroom in the error rate is what makes the spectrum informative.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from staticsbfl.utils.paths import DATA_DIR, ensure_dir
from staticsbfl.utils.seeding import DEFAULT_SEED, seeded_generator

logger = logging.getLogger(__name__)

#: Datasets this module can produce.
DATASET_NAMES = ("cifar10", "fashion_mnist")

#: The datasets a dense architecture population can be built on: all of them.
IMAGE_DATASET_NAMES = DATASET_NAMES

# CIFAR-10 channel statistics, used to standardise the images.
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Fashion-MNIST, single channel. Present as a *control* rather than a second
# benchmark: a dense model plateaus near 50% on CIFAR-10 but reaches roughly
# 88-90% here, so a population trained on it is properly fitted rather than
# half-fitted. That separates two explanations for dead units that CIFAR-10
# confounds -- "the task is too hard for this capacity" from "deep narrow ReLU
# stacks do this anyway". Note what it does NOT separate: dataset and reached
# accuracy move together here, which is the point but must be said out loud.
_FASHION_MNIST_MEAN = (0.2860,)
_FASHION_MNIST_STD = (0.3530,)

@dataclass
class DatasetBundle:
    """Everything a script needs about a prepared dataset."""

    name: str
    train_loader: DataLoader
    test_loader: DataLoader
    input_shape: Tuple[int, ...]
    num_classes: int
    class_names: Tuple[str, ...]

    @property
    def n_train(self) -> int:
        """Number of samples in the training split."""
        return len(self.train_loader.dataset)

    @property
    def n_test(self) -> int:
        """Number of samples in the test split."""
        return len(self.test_loader.dataset)


def load_dataset(
    name: str,
    *,
    batch_size: int = 128,
    seed: int = DEFAULT_SEED,
    root: Path = DATA_DIR,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> DatasetBundle:
    """Prepare *name* and return its loaders.

    Args:
        name: one of :data:`DATASET_NAMES`.
        batch_size: loader batch size for both splits.
        seed: seeds the training shuffle.
        root: download/cache directory.
        max_train_samples: cap the training split (smoke runs and tests).
        max_test_samples: cap the test split.

    Raises:
        ValueError: for an unknown dataset name.
    """
    key = name.lower()
    if key == "cifar10":
        return _load_cifar10(
            batch_size, seed, root, max_train_samples, max_test_samples
        )
    if key in ("fashion_mnist", "fashionmnist", "fashion-mnist"):
        return _load_fashion_mnist(
            batch_size, seed, root, max_train_samples, max_test_samples
        )
    raise ValueError(f"Unknown dataset {name!r}. Choose from {list(DATASET_NAMES)}.")


# --------------------------------------------------------------------------
# Fashion-MNIST
# --------------------------------------------------------------------------


def _load_fashion_mnist(
    batch_size: int,
    seed: int,
    root: Path,
    max_train_samples: Optional[int],
    max_test_samples: Optional[int],
) -> DatasetBundle:
    """Fashion-MNIST as standardised ``(1, 28, 28)`` float tensors.

    Same ten-class shape as CIFAR-10 and the same loader contract, so a
    population definition transfers unchanged; only the input shape differs, and
    every architecture in :mod:`staticsbfl.variation` derives its first layer from it.
    """
    from torchvision import transforms
    from torchvision.datasets import FashionMNIST

    ensure_dir(root)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(_FASHION_MNIST_MEAN, _FASHION_MNIST_STD)]
    )
    train = FashionMNIST(str(root), train=True, download=True, transform=transform)
    test = FashionMNIST(str(root), train=False, download=True, transform=transform)

    train = _subset(train, max_train_samples)
    test = _subset(test, max_test_samples)

    return DatasetBundle(
        name="fashion_mnist",
        train_loader=_loader(train, batch_size, shuffle=True, seed=seed),
        test_loader=_loader(test, batch_size, shuffle=False, seed=seed),
        input_shape=(1, 28, 28),
        num_classes=10,
        class_names=(
            "t-shirt", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle boot",
        ),
    )


# --------------------------------------------------------------------------
# CIFAR-10
# --------------------------------------------------------------------------


def _load_cifar10(
    batch_size: int,
    seed: int,
    root: Path,
    max_train_samples: Optional[int],
    max_test_samples: Optional[int],
) -> DatasetBundle:
    """CIFAR-10 as standardised ``(3, 32, 32)`` float tensors."""
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    ensure_dir(root)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)]
    )
    train = CIFAR10(str(root), train=True, download=True, transform=transform)
    test = CIFAR10(str(root), train=False, download=True, transform=transform)

    train = _subset(train, max_train_samples)
    test = _subset(test, max_test_samples)

    return DatasetBundle(
        name="cifar10",
        train_loader=_loader(train, batch_size, shuffle=True, seed=seed),
        test_loader=_loader(test, batch_size, shuffle=False, seed=seed),
        input_shape=(3, 32, 32),
        num_classes=10,
        class_names=(
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ),
    )


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def _subset(dataset, limit: Optional[int]):
    """The first *limit* samples of *dataset*, or the dataset itself."""
    if limit is None or limit >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, range(limit))


def _loader(dataset, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    """A DataLoader with a seeded generator and no worker processes.

    ``num_workers=0`` keeps ordering deterministic and sidesteps the macOS spawn
    overhead; these datasets are small enough that the loader is not the
    bottleneck.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=seeded_generator(seed) if shuffle else None,
    )
