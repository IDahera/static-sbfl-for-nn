"""Device selection.

Every compute entry point in this package takes an explicit ``device`` argument
rather than reaching for a module-level global, so the same function can be
driven from a CLI flag in production and pinned to CPU in a unit test.

``resolve_device`` is the single place that turns the *string* coming off the
command line into a :class:`torch.device`. ``"auto"`` prefers Apple MPS, then
CUDA, then CPU; the explicit names are honoured as given so that a user asking
for a device that is not there gets a clear error instead of a silent downgrade.
"""

from typing import Callable

import click
import torch

#: Accepted values for ``--device``.
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


def best_available_device() -> torch.device:
    """Return the best device present on this machine (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(name: str = "auto") -> torch.device:
    """Turn a ``--device`` string into a concrete :class:`torch.device`.

    Args:
        name: one of :data:`DEVICE_CHOICES`. ``"auto"`` picks the best backend
            that is actually available.

    Raises:
        ValueError: if *name* is not a known choice, or if an explicitly named
            accelerator is unavailable. Failing loudly matters here: a run that
            silently fell back to CPU would still produce numbers, just much
            more slowly and with a misleading log line.
    """
    key = str(name).lower()
    if key not in DEVICE_CHOICES:
        raise ValueError(
            f"Unknown device {name!r}. Choose from {list(DEVICE_CHOICES)}."
        )

    if key == "auto":
        return best_available_device()
    if key == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available on this machine.")
    if key == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this machine.")
    return torch.device(key)


def device_option(default: str = "auto") -> Callable:
    """The shared ``--device`` click option, so every script spells it the same."""
    return click.option(
        "--device",
        "device_name",
        type=click.Choice(DEVICE_CHOICES, case_sensitive=False),
        default=default,
        show_default=True,
        help="Compute device. 'auto' prefers MPS, then CUDA, then CPU.",
    )


def same_device(a: torch.device, b: torch.device) -> bool:
    """Compare two devices ignoring an unset index.

    ``torch.device("cuda")`` and ``torch.device("cuda:0")`` refer to the same
    place, but ``==`` says otherwise; tensors report the indexed form. Used by
    the tests that assert a tensor really landed on the requested device.
    """
    if a.type != b.type:
        return False
    if a.index is None or b.index is None:
        return True
    return a.index == b.index
