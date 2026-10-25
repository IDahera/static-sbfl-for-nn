"""Shared spectrum engine: layer responses -> activity masks -> outcome counts.

Suspiciousness formulas are in metrics.py. The hooks capture the raw output of
every selected ``nn.Linear`` layer (the pre-activation) and compare it with a
fixed threshold, zero by default.

Selecting layers restricts measurement, not forward inference. Hidden layers
are an explicit experimental scope, not a theorem about output-layer scores.
Two active counters replace four stored counters; this does not halve inference
cost. int32 counters require at most 2**31-1 scored samples.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

#: Module types treated as carrying countable neurons.
CANDIDATE_LAYER_TYPES = (nn.Linear,)

#: dtype of the accumulators. Counts are bounded by the sample count.
COUNT_DTYPE = torch.int32

@dataclass
class LayerSpectrum:
    """The hit spectrum of a single layer.

    ``a_f`` and ``a_s`` are ``int32`` tensors shaped like the layer's neurons
    (no batch dimension): ``(hidden,)`` for a dense layer.

    ``n_f`` and ``n_s`` are **not stored**. A sample is either failing or
    successful, and on each sample a neuron is either active or inactive, so::

        n_f = n_failing    - a_f
        n_s = n_successful - a_s

    They are exposed as properties so the metric formulae can still be written
    in the familiar four-count form.
    """

    name: str
    a_f: torch.Tensor
    a_s: torch.Tensor
    n_failing: int
    n_successful: int

    @property
    def neuron_shape(self) -> Tuple[int, ...]:
        """Shape of one sample's activations for this layer."""
        return tuple(self.a_f.shape)

    @property
    def n_neurons(self) -> int:
        """How many neurons this layer contributes to the ranking."""
        return int(self.a_f.numel())

    @property
    def n_samples(self) -> int:
        """Total samples the spectrum was accumulated over."""
        return self.n_failing + self.n_successful

    @property
    def n_f(self) -> torch.Tensor:
        """Inactive-and-failing count, derived from the failing total."""
        return self.n_failing - self.a_f

    @property
    def n_s(self) -> torch.Tensor:
        """Inactive-and-successful count, derived from the successful total."""
        return self.n_successful - self.a_s

    def to(self, device: torch.device) -> "LayerSpectrum":
        """A copy with both counter tensors moved to *device*."""
        return LayerSpectrum(
            name=self.name,
            a_f=self.a_f.to(device),
            a_s=self.a_s.to(device),
            n_failing=self.n_failing,
            n_successful=self.n_successful,
        )


@dataclass
class HitSpectrum:
    """The hit spectra of every counted layer of one model.

    ``layers`` is ordered the way the layers were hooked, i.e. in forward order
    for the reference architectures, so iterating it walks the model from input
    to output.
    """

    layers: Dict[str, LayerSpectrum] = field(default_factory=dict)
    n_failing: int = 0
    n_successful: int = 0
    threshold: float = 0.0

    @property
    def n_samples(self) -> int:
        """Total samples the spectra were accumulated over."""
        return self.n_failing + self.n_successful

    @property
    def accuracy(self) -> float:
        """Fraction of samples the model handled correctly during the pass."""
        return self.n_successful / self.n_samples if self.n_samples else 0.0

    @property
    def layer_names(self) -> Tuple[str, ...]:
        """The counted layers, in the order they were hooked (forward order)."""
        return tuple(self.layers)

    @property
    def n_neurons(self) -> int:
        """Total counted neurons across all layers."""
        return sum(layer.n_neurons for layer in self.layers.values())

    def to(self, device: torch.device) -> "HitSpectrum":
        """A copy with every layer's counters moved to *device*."""
        return HitSpectrum(
            layers={name: layer.to(device) for name, layer in self.layers.items()},
            n_failing=self.n_failing,
            n_successful=self.n_successful,
            threshold=self.threshold,
        )


@dataclass
class ActivationSummary:
    """How often a layer's neurons fired, independently of the outcome.

    Recorded for every scored layer because the activation *rate* is what
    decides whether a spectrum can discriminate at all. A layer whose neurons
    fire on nearly every sample yields ``a_f ~ N_f`` and ``a_s ~ N_s`` for all of
    them, which is the degenerate 'always active' case: every metric collapses
    to the same value and the ranking carries no information. The rate depends
    on the activation threshold, so it is the diagnostic for choosing one.
    """

    layer: str
    n_neurons: int
    mean_rate: float           # mean over neurons of (a_f + a_s) / N
    median_rate: float
    saturated_fraction: float  # neurons active on every single sample
    dead_fraction: float       # neurons never active at all
    score_spread: float = 0.0  # std of a bounded metric across the layer

    def summary(self) -> str:
        """One line for the console log."""
        return (
            f"activity mean {self.mean_rate:.3f} / median {self.median_rate:.3f}  "
            f"saturated {self.saturated_fraction:.1%}  dead {self.dead_fraction:.1%}"
        )


def activation_rates(layer: LayerSpectrum) -> torch.Tensor:
    """Per-neuron fraction of samples on which the neuron fired, in ``[0, 1]``."""
    if layer.n_samples == 0:
        return torch.zeros_like(layer.a_f, dtype=torch.float32)
    return (layer.a_f + layer.a_s).float() / layer.n_samples


def summarize_activity(
    hit_spectrum: "HitSpectrum",
    scores: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    spread_metric: str = "tarantula",
) -> Dict[str, ActivationSummary]:
    """Per-layer activation statistics, optionally with the score spread.

    Args:
        hit_spectrum: the counted spectrum.
        scores: ``{metric: {layer: tensor}}``; when given, the standard
            deviation of *spread_metric* across each layer is recorded too. A
            spread near zero means the layer ranks every neuron alike.
        spread_metric: which metric's spread to record.
    """
    summaries: Dict[str, ActivationSummary] = {}
    for name, layer in hit_spectrum.layers.items():
        rates = activation_rates(layer)
        spread = 0.0
        if scores and spread_metric in scores and name in scores[spread_metric]:
            spread = float(scores[spread_metric][name].float().std())
        summaries[name] = ActivationSummary(
            layer=name,
            n_neurons=layer.n_neurons,
            mean_rate=float(rates.mean()),
            median_rate=float(rates.median()),
            saturated_fraction=float((rates >= 1.0).float().mean()),
            dead_fraction=float((rates <= 0.0).float().mean()),
            score_spread=spread,
        )
    return summaries


# --------------------------------------------------------------------------
# Layer selection
# --------------------------------------------------------------------------


def candidate_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Every ``Linear`` submodule, in registration order."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, CANDIDATE_LAYER_TYPES)
    ]


def select_hidden_layers(
    model: nn.Module,
    layer_names: Optional[Sequence[str]] = None,
) -> List[Tuple[str, nn.Module]]:
    """Choose the layers whose neurons take part in the spectrum.

    Resolution order:

    1. an explicit *layer_names* argument, if given;
    2. a ``hidden_layer_names`` attribute declared by the model class -- the
       reliable route, and the one every generated architecture takes;
    3. otherwise all candidate layers except the last, which is assumed to be
       the classifier head.

    Raises:
        ValueError: if a requested name is not a candidate layer, or if the
            model has no hidden layer at all (a single-layer model has nothing
            to localize into).
    """
    candidates = candidate_layers(model)
    by_name = dict(candidates)

    if layer_names is None:
        layer_names = getattr(model, "hidden_layer_names", None)
    else:
        layer_names = tuple(layer_names)

    if layer_names is not None:
        missing = [n for n in layer_names if n not in by_name]
        if missing:
            raise ValueError(
                f"Layer(s) {missing} are not Linear submodules of "
                f"{type(model).__name__}. Available: {sorted(by_name)}."
            )
        return [(n, by_name[n]) for n in layer_names]

    if len(candidates) < 2:
        raise ValueError(
            f"{type(model).__name__} has {len(candidates)} candidate layer(s); at "
            "least two are needed so that one can be excluded as the output layer. "
            "Declare 'hidden_layer_names' on the model to select explicitly."
        )

    hidden = candidates[:-1]
    logger.debug(
        "%s does not declare hidden_layer_names; falling back to registration "
        "order and treating %r as the output layer.",
        type(model).__name__,
        candidates[-1][0],
    )
    return hidden


# --------------------------------------------------------------------------
# Activation capture
# --------------------------------------------------------------------------


class ActivationRecorder:
    """Capture the activations of selected layers for one forward pass.

    Used as a context manager so the forward hooks are always removed, even if
    the pass raises::

        with ActivationRecorder(model, layers) as recorder:
            logits = model(inputs)
            acts = recorder.activations   # {layer_name: (B, *neuron_shape)}

    Activations are detached. Nothing is retained between passes: :meth:`clear` runs at the
    start of every ``__enter__`` and the caller is expected to fold each batch
    into the counters before the next one.

    Responses are cloned to preserve the module-boundary measurement site.
    Repeated invocations of a selected module fail explicitly: choosing first,
    last, or aggregated invocation responses requires a separate definition.
    Call clear() before each forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[Tuple[str, nn.Module]],
    ) -> None:
        self.model = model
        self.layers = list(layers)
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if name in self.activations:
                raise RuntimeError(
                    f"Selected layer {name!r} executed more than once in one forward pass; "
                    "define an invocation aggregation policy before scoring shared modules."
                )
            response = output[0] if isinstance(output, (tuple, list)) else output
            # detach alone aliases output: a later in-place ReLU could otherwise
            # silently change the measurement site after the hook fired.
            self.activations[name] = response.detach().clone()

        return hook

    def clear(self) -> None:
        """Drop the activations captured by the previous pass."""
        self.activations = {}

    def __enter__(self) -> "ActivationRecorder":
        self.clear()
        for name, module in self.layers:
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


# --------------------------------------------------------------------------
# Counter allocation and update
# --------------------------------------------------------------------------


def infer_layer_shapes(
    model: nn.Module,
    sample_inputs: torch.Tensor,
    layers: Sequence[Tuple[str, nn.Module]],
    *,
    device: torch.device,
) -> Dict[str, Tuple[int, ...]]:
    """Probe the model once to learn each hooked layer's neuron shape.

    A single forward pass over *sample_inputs* (in practice the first batch) is
    enough: the per-sample activation shape of a layer does not depend on the
    batch. The batch dimension is stripped from what the hooks saw, so the
    result is exactly the shape the counters must have.

    Raises:
        RuntimeError: if a selected layer did not fire during the probe pass,
            which means it is not on the model's forward path.
    """
    model.to(device)
    model.eval()
    with torch.no_grad(), ActivationRecorder(model, layers) as recorder:
        model(sample_inputs.to(device))
        captured = dict(recorder.activations)

    missing = [name for name, _ in layers if name not in captured]
    if missing:
        raise RuntimeError(
            f"Layer(s) {missing} were selected but never executed during the probe "
            "forward pass; they are not on this model's forward path."
        )
    # Preserve the selection order rather than the (identical) capture order.
    return {name: tuple(captured[name].shape[1:]) for name, _ in layers}


def init_counters(
    shapes: Dict[str, Tuple[int, ...]],
    *,
    device: torch.device,
    dtype: torch.dtype = COUNT_DTYPE,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Allocate the ``a_f`` / ``a_s`` accumulators, zeroed, once per run.

    Returns ``(a_f, a_s)``, two dicts of tensors living on *device* and shaped
    like their layer's neurons. These tensors are the only per-neuron state the
    counting pass keeps; the batch loop updates them in place and never
    reallocates.
    """
    a_f = {
        name: torch.zeros(shape, dtype=dtype, device=device)
        for name, shape in shapes.items()
    }
    a_s = {
        name: torch.zeros(shape, dtype=dtype, device=device)
        for name, shape in shapes.items()
    }
    return a_f, a_s


def active_mask(activations: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """``True`` where a neuron fired, i.e. where its activation exceeds *threshold*.

    Hooks capture a layer's raw output, which for a ``Linear`` followed by ReLU
    is the *pre*-activation. With the default threshold of 0 the two agree:
    ``pre > 0`` exactly when the ReLU output is positive. The comparison is
    strict, so a response equal to the threshold is inactive.
    """
    return activations > threshold


def success_mask(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """``True`` where the model's arg-max prediction matches the label.

    This is the "test outcome" of the SBFL analogy: a successful sample is a
    passing test, a misclassified sample a failing one.
    """
    return logits.argmax(dim=1) == targets


def update_counts(
    a_f: torch.Tensor,
    a_s: torch.Tensor,
    active: torch.Tensor,
    success: torch.Tensor,
) -> None:
    """Fold one batch into the two accumulators, **in place**.

    Args:
        a_f: ``(*neuron_shape)`` int accumulator for active-and-failing.
        a_s: ``(*neuron_shape)`` int accumulator for active-and-successful.
        active: ``(batch, *neuron_shape)`` bool, output of :func:`active_mask`.
        success: ``(batch,)`` bool, output of :func:`success_mask`.

    The sample mask is broadcast against the neuron axes rather than used to
    index, which avoids materialising two sliced copies of the activation batch
    on every layer of every step.
    """
    if active.shape[1:] != a_f.shape:
        raise ValueError(
            f"Activation shape {tuple(active.shape[1:])} does not match the "
            f"counter shape {tuple(a_f.shape)}."
        )
    if success.shape[0] != active.shape[0]:
        raise ValueError(
            f"Got {success.shape[0]} outcome flags for {active.shape[0]} samples."
        )

    # (batch, 1, 1, ...) so it broadcasts across whatever neuron axes exist.
    per_sample = success.view(-1, *([1] * (active.dim() - 1)))
    a_s += (active & per_sample).sum(dim=0, dtype=a_s.dtype)
    a_f += (active & ~per_sample).sum(dim=0, dtype=a_f.dtype)


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def compute_hit_spectrum(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float = 0.0,
    layer_names: Optional[Iterable[str]] = None,
    max_batches: Optional[int] = None,
    batch_observer: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
) -> HitSpectrum:
    """Accumulate the per-neuron hit spectrum of *model* over *loader*.

    One streaming pass. Counters are allocated lazily from the first batch's
    measured response shapes, without a separate probe. batch_observer optionally fingerprints the original CPU batches for cache
    validation without executing the network or loading the dataset again.

    Args:
        model: the network to analyse. Moved to *device* and put in eval mode.
        loader: yields ``(inputs, targets)``; usually the **test** loader, since
            suspiciousness is about behaviour on data the model did not fit.
        device: where the forward pass and the counters live.
        threshold: a neuron is active when its activation exceeds this.
        layer_names: explicit layer selection; defaults to the model's hidden
            layers (see :func:`select_hidden_layers`).
        max_batches: stop after this many batches (for smoke runs and tests).

    Returns:
        A :class:`HitSpectrum` whose counters sit on *device*.

    Raises:
        ValueError: if the loader is empty.
    """
    model.to(device)
    model.eval()

    layers = select_hidden_layers(model, layer_names)
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive.")

    a_f: Dict[str, torch.Tensor] = {}
    a_s: Dict[str, torch.Tensor] = {}
    n_failing = 0
    n_successful = 0

    with torch.no_grad(), ActivationRecorder(model, layers) as recorder:
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            if batch_observer is not None:
                batch_observer(inputs, targets)
            inputs = inputs.to(device)
            targets = targets.to(device)

            recorder.clear()
            logits = model(inputs)
            success = success_mask(logits, targets)

            missing = {name for name, _ in layers} - recorder.activations.keys()
            if missing:
                raise RuntimeError(f"Selected layers did not execute: {sorted(missing)}")
            if n_failing + n_successful + len(targets) > torch.iinfo(COUNT_DTYPE).max:
                raise OverflowError("Sample count exceeds int32 spectrum capacity.")
            if not a_f:  # first batch: allocate the accumulators, once.
                shapes = {
                    name: tuple(recorder.activations[name].shape[1:])
                    for name, _ in layers
                }
                a_f, a_s = init_counters(shapes, device=device)

            for name, _module in layers:
                update_counts(
                    a_f[name],
                    a_s[name],
                    active_mask(recorder.activations[name], threshold),
                    success,
                )

            n_successful += int(success.sum().item())
            n_failing += int((~success).sum().item())

    if not a_f:
        raise ValueError("The data loader yielded no batches; nothing to count.")

    return HitSpectrum(
        layers={
            name: LayerSpectrum(
                name=name,
                a_f=a_f[name],
                a_s=a_s[name],
                n_failing=n_failing,
                n_successful=n_successful,
            )
            for name, _ in layers
        },
        n_failing=n_failing,
        n_successful=n_successful,
        threshold=threshold,
    )
