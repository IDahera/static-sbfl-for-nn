"""Reading and writing the cache.

Two artefact kinds, both ``torch.save`` blobs under ``cache/variants*/``
(see :mod:`staticsbfl.utils.paths` for the exact layout):

    model checkpoint    weights plus the metadata needed to rebuild the
                        architecture without guessing
    suspiciousness      the hit spectrum and the per-neuron score tensors
                        derived from it

Model checkpoints record the dataset shape they were built for, so the scoring
step can reconstruct an architecture from the cache alone rather than
re-deriving it from a freshly loaded dataset.

Everything is stored on the CPU. A checkpoint written on MPS has to be loadable
on a CUDA box, and a device-tagged tensor is not.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from staticsbfl.suspiciousness.spectrum import HitSpectrum, LayerSpectrum
from staticsbfl.utils.paths import ensure_dir


@dataclass
class ModelCheckpoint:
    """A cached model: its weights and enough metadata to rebuild it."""

    key: str
    state_dict: Dict[str, torch.Tensor]
    dataset: str
    input_shape: Tuple[int, ...]
    num_classes: int
    metadata: Dict[str, Any] = None  # type: ignore[assignment]


@dataclass
class SuspiciousnessBundle:
    """Cached suspiciousness for one model.

    ``scores`` is ``{metric_name: {layer_name: tensor}}``, each tensor shaped
    like its layer's neurons, so a score lines up 1:1 with the neuron it
    describes without any reshaping at the consumer end.
    """

    key: str
    hit_spectrum: HitSpectrum
    scores: Dict[str, Dict[str, torch.Tensor]]
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    @property
    def metric_names(self) -> Tuple[str, ...]:
        """The metrics this bundle holds scores for."""
        return tuple(self.scores)

    @property
    def layer_names(self) -> Tuple[str, ...]:
        """The counted layers, in forward order."""
        return self.hit_spectrum.layer_names


def save_model(path: Path, checkpoint: ModelCheckpoint) -> Path:
    """Write a :class:`ModelCheckpoint` to *path*, creating parent directories."""
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "key": checkpoint.key,
            "state_dict": {k: v.cpu() for k, v in checkpoint.state_dict.items()},
            "dataset": checkpoint.dataset,
            "input_shape": tuple(checkpoint.input_shape),
            "num_classes": checkpoint.num_classes,
            "metadata": checkpoint.metadata or {},
        },
        path,
    )
    return Path(path)


def load_model(path: Path) -> ModelCheckpoint:
    """Read a :class:`ModelCheckpoint` written by :func:`save_model`.

    Raises:
        FileNotFoundError: if *path* does not exist, with a hint at the script
            that would create it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached model at {path}. Run `make train` first."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return ModelCheckpoint(
        key=blob["key"],
        state_dict=blob["state_dict"],
        dataset=blob["dataset"],
        input_shape=tuple(blob["input_shape"]),
        num_classes=blob["num_classes"],
        metadata=blob.get("metadata", {}),
    )


def save_suspiciousness(path: Path, bundle: SuspiciousnessBundle) -> Path:
    """Write a :class:`SuspiciousnessBundle` to *path*.

    The hit spectrum is stored alongside the scores. It is the expensive part to
    recompute, and keeping it means a new metric can be added later and scored
    from the cache without another pass over the dataset.
    """
    ensure_dir(Path(path).parent)
    spectrum = bundle.hit_spectrum
    torch.save(
        {
            "key": bundle.key,
            "n_failing": spectrum.n_failing,
            "n_successful": spectrum.n_successful,
            "threshold": spectrum.threshold,
            "counts": {
                name: {"a_f": layer.a_f.cpu(), "a_s": layer.a_s.cpu()}
                for name, layer in spectrum.layers.items()
            },
            "scores": {
                metric: {layer: tensor.cpu() for layer, tensor in per_layer.items()}
                for metric, per_layer in bundle.scores.items()
            },
            "metadata": bundle.metadata or {},
        },
        path,
    )
    return Path(path)


def load_suspiciousness(path: Path) -> SuspiciousnessBundle:
    """Read a :class:`SuspiciousnessBundle` written by :func:`save_suspiciousness`.

    Raises:
        FileNotFoundError: if *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached suspiciousness at {path}. Run `make score` first."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)

    spectrum = HitSpectrum(
        layers={
            name: LayerSpectrum(
                name=name,
                a_f=counts["a_f"],
                a_s=counts["a_s"],
                n_failing=blob["n_failing"],
                n_successful=blob["n_successful"],
            )
            for name, counts in blob["counts"].items()
        },
        n_failing=blob["n_failing"],
        n_successful=blob["n_successful"],
        threshold=blob["threshold"],
    )
    return SuspiciousnessBundle(
        key=blob["key"],
        hit_spectrum=spectrum,
        scores=blob["scores"],
        metadata=blob.get("metadata", {}),
    )
