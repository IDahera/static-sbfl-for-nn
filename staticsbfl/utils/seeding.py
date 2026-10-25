"""Reproducibility.

Seed 42 everywhere, fixed by :data:`DEFAULT_SEED`, so a rerun of the training
script reproduces the cached models bit-for-bit on the same machine and backend.
"""

import random

import numpy as np
import torch

#: The seed used for every model, split and shuffle in this project.
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and every torch backend."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """A CPU generator for DataLoader shuffling, so batch order is reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
