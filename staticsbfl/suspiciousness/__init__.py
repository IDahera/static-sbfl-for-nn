"""Separate spectrum counting from scoring.

spectrum.py applies the activity threshold and accumulates outcome counts.
metrics.py scores those same counts without further model execution.
"""

from staticsbfl.suspiciousness.metrics import (
    BOUNDED_METRICS,
    DEFAULT_STAR,
    METRICS,
    METRIC_NAMES,
    dstar,
    ochiai,
    score_spectrum,
    tarantula,
)
from staticsbfl.suspiciousness.spectrum import (
    ActivationRecorder,
    ActivationSummary,
    activation_rates,
    HitSpectrum,
    LayerSpectrum,
    active_mask,
    compute_hit_spectrum,
    infer_layer_shapes,
    init_counters,
    select_hidden_layers,
    success_mask,
    summarize_activity,
    update_counts,
)

__all__ = [
    "BOUNDED_METRICS",
    "DEFAULT_STAR",
    "METRICS",
    "METRIC_NAMES",
    "dstar",
    "ochiai",
    "score_spectrum",
    "tarantula",
    "ActivationRecorder",
    "ActivationSummary",
    "activation_rates",
    "HitSpectrum",
    "LayerSpectrum",
    "active_mask",
    "compute_hit_spectrum",
    "infer_layer_shapes",
    "init_counters",
    "select_hidden_layers",
    "success_mask",
    "summarize_activity",
    "update_counts",
]
