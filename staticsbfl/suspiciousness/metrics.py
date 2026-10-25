"""Suspiciousness metrics over a per-neuron hit spectrum.

Classical spectrum-based fault localization ranks program statements by how
strongly their execution coincides with failing tests. Transferred to a neural
network, a *statement* becomes a neuron, *executed* becomes "activation above a
threshold", and a *failing test* becomes a sample the model predicts wrongly.
Every metric below is therefore a function of the same four counts, taken per
neuron over the evaluated dataset:

    a_f  active   on a failing sample
    a_s  active   on a successful sample
    n_f  inactive on a failing sample
    n_s  inactive on a successful sample

Only ``a_f`` and ``a_s`` are ever accumulated. A sample is either failing or
successful, and a neuron is either active or inactive on it, so with ``N_f``
failing and ``N_s`` successful samples in total::

    n_f = N_f - a_f        n_s = N_s - a_s

Two scalars therefore replace two full per-neuron tensors -- see
:class:`~staticsbfl.suspiciousness.spectrum.LayerSpectrum`, which exposes ``n_f`` and
``n_s`` as derived properties so the formulae below still read like the
textbook ones.

Every function here is pure and elementwise: it takes one ``LayerSpectrum`` and
returns a float tensor shaped exactly like that layer's neurons. Division by
zero is mapped to ``0.0`` rather than left as NaN -- a neuron that never fired
carries no evidence and must not out-rank one that did.
"""

from typing import TYPE_CHECKING, Dict, Iterable, Optional

import torch

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from staticsbfl.suspiciousness.spectrum import HitSpectrum, LayerSpectrum

#: Default exponent of the D* metric (i.e. D^3), overridable from the CLI.
DEFAULT_STAR = 3


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Elementwise division that maps 0/0 and x/0 to 0 instead of NaN/inf."""
    result = torch.div(numerator, denominator)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def tarantula(spectrum: "LayerSpectrum") -> torch.Tensor:
    """Tarantula suspiciousness, bounded to ``[0, 1]``.

    .. math::

        \\mathrm{tarantula} =
            \\frac{\\frac{a_f}{a_f + n_f}}
                  {\\frac{a_f}{a_f + n_f} + \\frac{a_s}{a_s + n_s}}

    The two fractions are the neuron's activation rate among failing and among
    successful samples. Because ``a_f + n_f = N_f`` and ``a_s + n_s = N_s``,
    both denominators are constants of the run -- the ratio is simply "how much
    of this neuron's firing is concentrated on the failures".

    A neuron that never fires yields 0/0 and scores 0.
    """
    failing_rate = _safe_div(spectrum.a_f.float(), spectrum.a_f.float() + spectrum.n_f.float())
    success_rate = _safe_div(spectrum.a_s.float(), spectrum.a_s.float() + spectrum.n_s.float())
    return _safe_div(failing_rate, failing_rate + success_rate)


def ochiai(spectrum: "LayerSpectrum") -> torch.Tensor:
    """Ochiai suspiciousness, bounded to ``[0, 1]``.

    .. math::

        \\mathrm{ochiai} = \\frac{a_f}{\\sqrt{(a_f + n_f)\\,(a_f + a_s)}}

    The cosine similarity between "this neuron fired" and "the model failed".
    Since ``a_f \\le a_f + n_f`` and ``a_f \\le a_f + a_s``, the geometric mean in
    the denominator is at least ``a_f``, so the score never exceeds 1.
    """
    a_f = spectrum.a_f.float()
    denominator = torch.sqrt((a_f + spectrum.n_f.float()) * (a_f + spectrum.a_s.float()))
    return _safe_div(a_f, denominator)


def dstar(spectrum: "LayerSpectrum", star: int = DEFAULT_STAR) -> torch.Tensor:
    """D* suspiciousness with exponent *star* (default 3, i.e. D^3).

    .. math::

        D^{*} = \\frac{a_f^{\\,*}}{a_s + n_f}

    Raising ``a_f`` to a power sharpens the ranking towards neurons that fire on
    many failures. Unlike Tarantula and Ochiai this is **unbounded above**: it
    grows with the number of failing samples, so D* values are comparable within
    one model and dataset but not across runs of different size.

    The denominator vanishes for a neuron with ``a_s = 0`` and ``n_f = 0``, i.e.
    one that fires on *every* failing sample and on *no* successful one. The two
    zero-denominator cases are deliberately kept apart:

        ``0 / 0``  the neuron never fired at all -- no evidence, score ``0``;
        ``x / 0``  the neuron fired exactly on the failures -- maximal evidence,
                   score ``+inf``.

    Collapsing the second case to 0 (as a naive ``nan_to_num`` would) inverts
    the ranking precisely where the metric is most confident, so it is left as
    ``+inf`` and consumers handle it: the neuron table and the evaluation treat
    it as the top of the ranking. On real data the case is rare;
    on small crafted inputs it is not, which is why it is tested explicitly.

    Raises:
        ValueError: if *star* is not a positive integer.
    """
    if not isinstance(star, int) or isinstance(star, bool) or star < 1:
        raise ValueError(f"star must be a positive integer, got {star!r}.")
    numerator = torch.pow(spectrum.a_f.float(), star)
    denominator = spectrum.a_s.float() + spectrum.n_f.float()
    result = torch.div(numerator, denominator)
    # 0/0 -> NaN -> 0 (no evidence); x/0 -> +inf is kept (maximal evidence).
    return torch.nan_to_num(result, nan=0.0, posinf=float("inf"), neginf=0.0)


#: Metric name -> function. Iterate this rather than hard-coding metric lists.
METRICS = {
    "tarantula": tarantula,
    "ochiai": ochiai,
    "dstar": dstar,
}

METRIC_NAMES = tuple(METRICS)

#: Metrics whose values are mathematically confined to ``[0, 1]``. D* is not,
#: which the property tests rely on.
BOUNDED_METRICS = ("tarantula", "ochiai")


def score_layer(
    spectrum: "LayerSpectrum",
    *,
    metrics: Optional[Iterable[str]] = None,
    star: int = DEFAULT_STAR,
) -> Dict[str, torch.Tensor]:
    """Apply every requested metric to one layer's spectrum.

    Returns ``{metric_name: tensor}``, each tensor shaped like the layer's
    neurons.

    Raises:
        ValueError: if an unknown metric name is requested.
    """
    names = tuple(metrics) if metrics is not None else METRIC_NAMES
    unknown = sorted(set(names) - set(METRICS))
    if unknown:
        raise ValueError(f"Unknown metric(s) {unknown}. Choose from {list(METRICS)}.")

    scores: Dict[str, torch.Tensor] = {}
    for name in names:
        fn = METRICS[name]
        scores[name] = fn(spectrum, star) if name == "dstar" else fn(spectrum)
    return scores


def score_spectrum(
    hit_spectrum: "HitSpectrum",
    *,
    metrics: Optional[Iterable[str]] = None,
    star: int = DEFAULT_STAR,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Score every layer of a whole-model hit spectrum.

    This is the cheap half of the pipeline: the counting pass over the dataset
    already happened, so all three metrics come out of a single set of counts
    without touching the model again.

    Returns ``{metric_name: {layer_name: tensor}}`` where each tensor keeps the
    shape of its layer's neurons -- ``(hidden,)`` for a dense layer,
    ``(channels, height, width)`` for a convolution.
    """
    names = tuple(metrics) if metrics is not None else METRIC_NAMES
    scored: Dict[str, Dict[str, torch.Tensor]] = {name: {} for name in names}
    for layer_name, layer in hit_spectrum.layers.items():
        per_metric = score_layer(layer, metrics=names, star=star)
        for name, values in per_metric.items():
            scored[name][layer_name] = values
    return scored
