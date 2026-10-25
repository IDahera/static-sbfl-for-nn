"""Dataset preparation tests.

Only the parts that run without a network. The download paths are exercised by
the debug run (``make debug``), not by the unit suite.
"""

import pytest

from staticsbfl.data import DATASET_NAMES, IMAGE_DATASET_NAMES, load_dataset


def test_an_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("imagenet")


def test_every_dataset_can_carry_a_dense_population():
    assert set(IMAGE_DATASET_NAMES) == set(DATASET_NAMES) == {"cifar10", "fashion_mnist"}
