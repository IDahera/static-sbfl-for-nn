"""Two fixed dense reference models and parameter accounting."""

from staticsbfl.models.architectures import (
    SmallMLP,
    WideMLP,
    count_parameters,
    parameter_bytes,
)

__all__ = [
    "SmallMLP",
    "WideMLP",
    "count_parameters",
    "parameter_bytes",
]
