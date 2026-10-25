"""Turning a configuration into a model.

The one rule here: **the builder invents no dimension.** Every width, the depth
and the activation come from the :class:`~staticsbfl.variation.config.ArchitectureConfig`
it is handed. Anything the builder decided for itself would be invisible to the
config, and therefore missing from the feature table that later describes the
model -- which is exactly the failure mode this separation exists to prevent.

Only the dense family is implemented. The dispatch table at the bottom is the
extension point: adding a convolutional or recurrent family means adding a
builder and a config field, not changing the callers.
"""

from typing import Callable, Dict, Sequence, Tuple

import math

import torch
import torch.nn as nn

from staticsbfl.variation.config import ArchitectureConfig

#: Activation name -> the module that implements it.
ACTIVATION_MODULES: Dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def make_activation(name: str) -> nn.Module:
    """Instantiate an activation by name.

    Raises:
        ValueError: for an unknown name.
    """
    if name not in ACTIVATION_MODULES:
        raise ValueError(
            f"Unknown activation {name!r}. Known: {list(ACTIVATION_MODULES)}."
        )
    return ACTIVATION_MODULES[name]()


class VariableMLP(nn.Module):
    """A dense classifier whose depth, widths and activation come from a config.

    Hidden layers are named ``fc0 .. fc{depth-1}`` and the output layer is
    ``head``, so ``hidden_layer_names`` is derivable from the config alone and
    the spectrum module needs no special case.

    The activation is a single shared module applied between layers rather than
    one instance per layer. It is stateless, so this changes nothing numerically,
    and it keeps the module list free of entries the spectrum would have to skip.
    """

    def __init__(
        self,
        hidden_widths: Sequence[int],
        *,
        num_classes: int,
        input_shape: Sequence[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_shape: Tuple[int, ...] = tuple(int(d) for d in input_shape)
        self.output_dim = int(num_classes)
        self.activation_name = activation
        self.hidden_widths = tuple(int(w) for w in hidden_widths)

        if not self.hidden_widths:
            raise ValueError("A VariableMLP needs at least one hidden layer.")

        self.flatten = nn.Flatten()
        self.activation = make_activation(activation)

        previous = int(math.prod(self.input_shape))
        for index, width in enumerate(self.hidden_widths):
            setattr(self, f"fc{index}", nn.Linear(previous, width))
            previous = width
        self.head = nn.Linear(previous, num_classes)

        self.hidden_layer_names: Tuple[str, ...] = tuple(
            f"fc{index}" for index in range(len(self.hidden_widths))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        for name in self.hidden_layer_names:
            x = self.activation(getattr(self, name)(x))
        return self.head(x)


def build_mlp(
    config: ArchitectureConfig, *, num_classes: int, input_shape: Sequence[int]
) -> nn.Module:
    """Build the dense model *config* describes."""
    return VariableMLP(
        config.hidden_widths,
        num_classes=num_classes,
        input_shape=input_shape,
        activation=config.activation,
    )


#: Family -> builder. The extension point for convolutional and recurrent
#: variants: add an entry, do not touch the callers.
BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "mlp": build_mlp,
}


def build_model(
    config: ArchitectureConfig, *, num_classes: int, input_shape: Sequence[int]
) -> nn.Module:
    """Build the model *config* describes, dispatching on its family.

    Raises:
        NotImplementedError: for a family that has no builder yet.
    """
    if config.family not in BUILDERS:
        raise NotImplementedError(
            f"No builder for family {config.family!r}. Implemented: "
            f"{list(BUILDERS)}. Add one to staticsbfl.variation.builder.BUILDERS."
        )
    return BUILDERS[config.family](
        config, num_classes=num_classes, input_shape=input_shape
    )
