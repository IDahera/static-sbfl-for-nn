"""Shared fixtures.

The models here are deliberately *scripted*: their weights are set by hand so
that which neurons fire, and on which samples, is known before the code runs.
That is what makes it possible to assert exact expected counts rather than
merely self-consistent ones.
"""

from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from staticsbfl.suspiciousness.spectrum import LayerSpectrum


@pytest.fixture
def cpu() -> torch.device:
    """The device every test defaults to, so the suite never needs a GPU."""
    return torch.device("cpu")


class ScriptedMLP(nn.Module):
    """Two hidden dense layers plus a head, with hand-set weights.

    ``hidden_bias`` drives every hidden pre-activation to a constant, so the
    'active' decision is fully determined:

        ``+1`` every hidden neuron fires on every sample;
        ``-1`` no hidden neuron ever fires.

    The head is the identity on the first ``num_classes`` inputs of a zeroed
    hidden state plus ``head_bias``, so the predicted class is fixed by
    ``head_bias`` and is therefore independent of the hidden activations. That
    independence is the point: it lets a test choose the success/failure split
    and the activation pattern separately.
    """

    def __init__(
        self,
        in_features: int = 4,
        hidden: Tuple[int, int] = (3, 2),
        num_classes: int = 2,
        hidden_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_shape = (in_features,)
        self.output_dim = num_classes
        self.fc1 = nn.Linear(in_features, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.head = nn.Linear(hidden[1], num_classes)
        self.relu = nn.ReLU()
        self.hidden_layer_names = ("fc1", "fc2")

        with torch.no_grad():
            for layer, bias in ((self.fc1, hidden_bias), (self.fc2, hidden_bias)):
                layer.weight.zero_()
                layer.bias.fill_(bias)
            self.head.weight.zero_()
            self.head.bias.zero_()

    def set_prediction(self, class_index: int) -> None:
        """Force every sample to be predicted as *class_index*."""
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[class_index] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.head(x)


class PassthroughNet(nn.Module):
    """A net whose hidden activations are exactly its input.

    ``fc1`` is the identity, so ``active_mask`` fires precisely where the input
    is positive -- a test can therefore write down the activation pattern
    directly. The head reads a *separate* trailing feature of the input, so the
    prediction can be steered independently of the hidden pattern.

    Input is ``(n_hidden + 1,)``: the first ``n_hidden`` entries are the hidden
    activations, the last entry is the class to predict.
    """

    def __init__(self, n_hidden: int = 5, num_classes: int = 2) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.input_shape = (n_hidden + 1,)
        self.output_dim = num_classes
        self.fc1 = nn.Linear(n_hidden + 1, n_hidden)
        self.head = nn.Linear(n_hidden, num_classes)
        self.hidden_layer_names = ("fc1",)

        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc1.weight[:, :n_hidden] = torch.eye(n_hidden)
            self.fc1.bias.zero_()
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)  # == x[:, :n_hidden]
        # One-hot logits taken straight from the trailing control feature.
        control = x[:, -1].long().clamp(0, self.output_dim - 1)
        logits = torch.zeros(x.shape[0], self.output_dim, device=x.device)
        logits.scatter_(1, control.unsqueeze(1), 1.0)
        return logits + 0.0 * self.head(hidden)  # keep the head in the graph


@pytest.fixture
def scripted_mlp():
    """Factory for :class:`ScriptedMLP`."""
    return ScriptedMLP


@pytest.fixture
def passthrough_net():
    """Factory for :class:`PassthroughNet`."""
    return PassthroughNet


def make_loader(
    inputs: torch.Tensor, targets: torch.Tensor, batch_size: int = 2
) -> DataLoader:
    """A non-shuffling loader over the given tensors."""
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)


def make_spectrum(a_f, a_s, n_failing: int, n_successful: int, name: str = "layer"):
    """Build a :class:`LayerSpectrum` from plain lists, for metric tests."""
    return LayerSpectrum(
        name=name,
        a_f=torch.tensor(a_f, dtype=torch.int32),
        a_s=torch.tensor(a_s, dtype=torch.int32),
        n_failing=n_failing,
        n_successful=n_successful,
    )


@pytest.fixture
def loader_factory():
    return make_loader


@pytest.fixture
def spectrum_factory():
    return make_spectrum
