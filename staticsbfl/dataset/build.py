"""Turning trained models and their spectra into a table of neurons.

One row per hidden neuron. The features describe the neuron and the
architecture it sits in and are computable **without executing the model**,
which is the premise of the contribution; the targets come from the hit spectrum
and are what executing the model would have told you.

The construction is deliberately mechanical, so that a test can predict its
output exactly: for a model with hidden widths ``(w_0, ..., w_{d-1})`` the table
gains ``sum(w_i)`` rows, in layer order and within a layer in unit order, with
the columns of :data:`~staticsbfl.dataset.schema.FEATURE_NAMES` in that order.

Grouping
--------
Every row carries the key of the model it came from. That column is the group
for a ranking objective and the unit for any train/test split or confidence
interval: neurons of one model share its architecture, initialisation and
training run, so they are not independent observations.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from staticsbfl.dataset.schema import (
    ACTIVITY_TARGET,
    FEATURE_NAMES,
    TARGET_METRICS,
    TARGET_NAMES,
    WIDTH_PROFILE_IDS,
    target_names,
)
from staticsbfl.persistence import SuspiciousnessBundle
from staticsbfl.variation.config import ArchitectureConfig

logger = logging.getLogger(__name__)

#: Incoming weights below this magnitude count as "small" for the
#: ``in_weight_frac_small`` feature.
SMALL_WEIGHT = 1e-3


@dataclass
class NeuronDataset:
    """A table of neurons: features, targets, and the grouping columns.

    ``features`` is ``(n_rows, len(FEATURE_NAMES))``. ``targets`` maps each name
    in :data:`~staticsbfl.dataset.schema.TARGET_NAMES` to a ``(n_rows,)`` array.
    ``model_keys``, ``layer_names`` and ``splits`` are parallel string arrays
    identifying where each row came from.
    """

    features: np.ndarray
    targets: Dict[str, np.ndarray]
    model_keys: np.ndarray
    layer_names: np.ndarray
    splits: np.ndarray
    feature_names: Tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        n = self.features.shape[0]
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"{self.features.shape[1]} feature columns for "
                f"{len(self.feature_names)} declared names."
            )
        for name, values in self.targets.items():
            if values.shape[0] != n:
                raise ValueError(f"target {name!r} has {values.shape[0]} rows, expected {n}.")
        for label, column in (
            ("model_keys", self.model_keys),
            ("layer_names", self.layer_names),
            ("splits", self.splits),
        ):
            if column.shape[0] != n:
                raise ValueError(f"{label} has {column.shape[0]} rows, expected {n}.")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_models(self) -> int:
        """Independent observations, which is the model count -- not ``len(self)``."""
        return int(np.unique(self.model_keys).size)

    def select(self, mask: np.ndarray) -> "NeuronDataset":
        """A new dataset holding only the rows where *mask* is True."""
        return NeuronDataset(
            features=self.features[mask],
            targets={k: v[mask] for k, v in self.targets.items()},
            model_keys=self.model_keys[mask],
            layer_names=self.layer_names[mask],
            splits=self.splits[mask],
            feature_names=self.feature_names,
        )

    def for_split(self, split: str) -> "NeuronDataset":
        """Only the rows belonging to one population (e.g. ``"base"``)."""
        return self.select(self.splits == split)

    def feature_subset(self, names: Sequence[str]) -> np.ndarray:
        """The columns named, in the order given -- for feature-group ablations.

        Raises:
            KeyError: if a name is not a declared feature.
        """
        missing = [n for n in names if n not in self.feature_names]
        if missing:
            raise KeyError(f"Unknown feature(s) {missing}.")
        index = [self.feature_names.index(n) for n in names]
        return self.features[:, index]

    def save(self, path: Union[str, Path]) -> None:
        """Write the table to a compressed ``.npz``."""
        np.savez_compressed(
            path,
            features=self.features,
            feature_names=np.array(self.feature_names),
            model_keys=self.model_keys,
            layer_names=self.layer_names,
            splits=self.splits,
            **{f"target__{k}": v for k, v in self.targets.items()},
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NeuronDataset":
        """Read a table written by :meth:`save`."""
        blob = np.load(path, allow_pickle=False)
        targets = {
            k[len("target__"):]: blob[k] for k in blob.files if k.startswith("target__")
        }
        return cls(
            features=blob["features"],
            targets=targets,
            model_keys=blob["model_keys"],
            layer_names=blob["layer_names"],
            splits=blob["splits"],
            feature_names=tuple(str(n) for n in blob["feature_names"]),
        )


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks of *values* rescaled to ``[0, 1]``, largest value ranking 1.

    Ties take the average of the ranks they span, so the result does not depend
    on the input order. A single value ranks 1.0. Non-finite entries (D* is
    ``+inf`` for a neuron firing exactly on the failures) sort above every finite
    value, which is the intended ordering.
    """
    n = values.shape[0]
    if n == 1:
        return np.ones(1, dtype=np.float64)

    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)

    # Average the ranks of tied values.
    sorted_values = values[order]
    start = 0
    for stop in range(1, n + 1):
        if stop == n or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = np.arange(start, stop).mean()
            start = stop

    return ranks / (n - 1)


def _weight_statistics(vector: torch.Tensor) -> List[float]:
    """Mean, std, L1, L2 and max-abs of one weight vector, plus the small fraction."""
    absolute = vector.abs()
    return [
        float(vector.mean()),
        float(vector.std(unbiased=False)),
        float(absolute.sum()),
        float(vector.norm(p=2)),
        float(absolute.max()),
        float((absolute < SMALL_WEIGHT).float().mean()),
    ]


def _outgoing_statistics(vector: torch.Tensor) -> List[float]:
    """Mean, std, L1, L2 and max-abs of one outgoing weight vector."""
    absolute = vector.abs()
    return [
        float(vector.mean()),
        float(vector.std(unbiased=False)),
        float(absolute.sum()),
        float(vector.norm(p=2)),
        float(absolute.max()),
    ]


def _incoming_statistics_matrix(weight: torch.Tensor) -> torch.Tensor:
    """:func:`_weight_statistics` for every row of *weight* at once, ``(width, 6)``.

    Bit-identical to calling it per row: each row is contiguous, so the
    row-wise reductions see the same elements in the same order.
    """
    absolute = weight.abs()
    return torch.stack([
        weight.mean(dim=1),
        weight.std(dim=1, unbiased=False),
        absolute.sum(dim=1),
        weight.norm(p=2, dim=1),
        absolute.amax(dim=1),
        (absolute < SMALL_WEIGHT).float().mean(dim=1),
    ], dim=1)


def _outgoing_statistics_matrix(next_weight: torch.Tensor) -> torch.Tensor:
    """:func:`_outgoing_statistics` for every column of *next_weight*, ``(width, 5)``.

    Bit-identical to calling it per column ``next_weight[:, unit]``. That
    column is a strided view, and float32 reductions depend on the order they
    visit elements: std, L1 and max-abs match on the contiguous transpose, the
    L2 norm matches reduced along dim 0, and the mean of a strided view matches
    no vectorised form -- it is the one statistic still taken column by column,
    so that tables built before and after this change are identical.
    """
    columns = next_weight.t().contiguous()
    absolute = columns.abs()
    mean = torch.stack([next_weight[:, unit].mean() for unit in range(next_weight.shape[1])])
    return torch.stack([
        mean,
        columns.std(dim=1, unbiased=False),
        absolute.sum(dim=1),
        next_weight.norm(p=2, dim=0),
        absolute.amax(dim=1),
    ], dim=1)


def model_feature_rows(
    model: nn.Module,
    config: ArchitectureConfig,
    *,
    input_size: int,
    num_classes: int,
) -> Tuple[np.ndarray, List[str]]:
    """Build the feature rows of one dense model.

    Returns ``(features, layer_names)`` where ``features`` is
    ``(total_hidden_units, len(FEATURE_NAMES))`` and ``layer_names`` names the
    layer each row belongs to. Rows are ordered by layer, then by unit within
    the layer.

    Raises:
        ValueError: if the model declares no hidden layers.
    """
    hidden = list(model.hidden_layer_names)
    if not hidden:
        raise ValueError(f"{config.key}: the model declares no hidden layers.")

    state = model.state_dict()
    depth = len(hidden)
    widths = [int(state[f"{name}.weight"].shape[0]) for name in hidden]
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    profile_id = WIDTH_PROFILE_IDS[config.width_profile]

    blocks: List[np.ndarray] = []
    row_layers: List[str] = []

    for layer_index, name in enumerate(hidden):
        weight = state[f"{name}.weight"]          # (width, fan_in)
        bias = state[f"{name}.bias"]              # (width,)
        following = hidden[layer_index + 1] if layer_index + 1 < depth else "head"
        next_weight = state[f"{following}.weight"]  # (next_width, width)

        width = widths[layer_index]
        fan_in = int(weight.shape[1])
        fan_out = int(next_weight.shape[0])
        previous_width = input_size if layer_index == 0 else widths[layer_index - 1]
        next_width = num_classes if layer_index + 1 == depth else widths[layer_index + 1]
        layer_position = layer_index / (depth - 1) if depth > 1 else 0.0

        architecture = [
            float(depth),
            float(layer_index),
            float(layer_position),
            float(width),
            float(previous_width),
            float(next_width),
            float(config.activation_id),
            float(config.total_hidden_units),
            float(profile_id),
            float(n_parameters),
        ]

        # One block per layer instead of one Python iteration per unit; the
        # columns and their values are those of the per-unit helpers above.
        constant = np.tile(np.asarray([float(fan_in), float(fan_out)] + architecture, dtype=np.float64), (width, 1))
        blocks.append(np.concatenate([
            _incoming_statistics_matrix(weight).double().numpy(),
            bias.double().numpy()[:, None],
            _outgoing_statistics_matrix(next_weight).double().numpy(),
            constant,
        ], axis=1))
        row_layers.extend([name] * width)

    features = np.concatenate(blocks, axis=0)
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"built {features.shape[1]} columns for {len(FEATURE_NAMES)} declared "
            "features; schema and builder are out of step."
        )
    return features, row_layers


def model_target_rows(
    bundle: SuspiciousnessBundle, layer_order: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Build the target columns of one model, aligned with its feature rows.

    For each metric this produces the raw value, its rank among all neurons of
    the model, and its rank within its own layer.

    Raises:
        KeyError: if the bundle lacks a layer the features cover.
    """
    targets: Dict[str, np.ndarray] = {}

    for metric in TARGET_METRICS:
        per_layer = bundle.scores[metric]
        missing = [name for name in layer_order if name not in per_layer]
        if missing:
            raise KeyError(f"{bundle.key}: no {metric} scores for layer(s) {missing}.")

        raw_parts, layer_rank_parts = [], []
        for name in layer_order:
            values = per_layer[name].flatten().to(torch.float64).numpy()
            raw_parts.append(values)
            layer_rank_parts.append(normalized_ranks(values))

        raw = np.concatenate(raw_parts)
        raw_name, model_rank_name, layer_rank_name = target_names(metric)
        targets[raw_name] = raw
        targets[model_rank_name] = normalized_ranks(raw)
        targets[layer_rank_name] = np.concatenate(layer_rank_parts)

    # How often each neuron fired, independent of the outcome. Carried alongside
    # the metrics so a later evaluation can exclude never-active units without
    # guessing at them from a zero score: a score of zero also arises for a unit
    # that fired only on successes, which is evidence, not the absence of it.
    spectrum = bundle.hit_spectrum
    rate_parts = []
    for name in layer_order:
        layer = spectrum.layers[name]
        active = (layer.a_f + layer.a_s).flatten().to(torch.float64).numpy()
        rate_parts.append(active / spectrum.n_samples if spectrum.n_samples else active)
    targets[ACTIVITY_TARGET] = np.concatenate(rate_parts)

    return targets


def build_model_rows(
    model: nn.Module,
    config: ArchitectureConfig,
    bundle: SuspiciousnessBundle,
    *,
    input_size: int,
    num_classes: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """Features, targets and layer names for one model, aligned row for row.

    Raises:
        ValueError: if the spectrum covers a different number of neurons than
            the architecture does, which means the two are not the same model.
    """
    features, row_layers = model_feature_rows(
        model, config, input_size=input_size, num_classes=num_classes
    )

    # Deduplicated, order-preserving list of the layers the rows cover.
    layer_order: List[str] = []
    for name in row_layers:
        if name not in layer_order:
            layer_order.append(name)

    targets = model_target_rows(bundle, layer_order)
    expected = features.shape[0]
    for name, values in targets.items():
        if values.shape[0] != expected:
            raise ValueError(
                f"{config.key}: the spectrum has {values.shape[0]} neurons for "
                f"{name} but the architecture has {expected}. The cached model "
                "and the cached spectrum are not the same model."
            )
    return features, targets, row_layers


def combine(
    parts: Sequence[Tuple[np.ndarray, Dict[str, np.ndarray], List[str], str, str]]
) -> NeuronDataset:
    """Stack per-model parts into one dataset.

    Each part is ``(features, targets, layer_names, model_key, split)``.

    Raises:
        ValueError: if *parts* is empty.
    """
    if not parts:
        raise ValueError("No models to combine; nothing was built.")

    features = np.concatenate([p[0] for p in parts], axis=0)
    targets = {
        name: np.concatenate([p[1][name] for p in parts], axis=0)
        for name in TARGET_NAMES
    }
    model_keys = np.concatenate(
        [np.full(len(p[2]), p[3], dtype=object) for p in parts]
    ).astype(str)
    layer_names = np.concatenate([np.asarray(p[2], dtype=str) for p in parts])
    splits = np.concatenate(
        [np.full(len(p[2]), p[4], dtype=object) for p in parts]
    ).astype(str)

    return NeuronDataset(
        features=features,
        targets=targets,
        model_keys=model_keys,
        layer_names=layer_names,
        splits=splits,
    )
