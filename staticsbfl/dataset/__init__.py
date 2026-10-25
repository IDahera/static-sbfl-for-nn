"""The neuron-level table: what a predictor is fitted on."""

from staticsbfl.dataset.build import (
    NeuronDataset,
    build_model_rows,
    combine,
    model_feature_rows,
    model_target_rows,
    normalized_ranks,
)
from staticsbfl.dataset.schema import (
    ARCHITECTURE,
    EXCLUDED_FEATURES,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURES,
    FEATURES_BY_NAME,
    NEURON_STATIC,
    ACTIVITY_TARGET,
    TARGET_METRICS,
    TARGET_NAMES,
    WIDTH_PROFILE_IDS,
    FeatureSpec,
    target_names,
)

__all__ = [
    "NeuronDataset", "build_model_rows", "combine", "model_feature_rows",
    "model_target_rows", "normalized_ranks",
    "ARCHITECTURE", "EXCLUDED_FEATURES", "FEATURE_GROUPS", "FEATURE_NAMES",
    "FEATURES", "FEATURES_BY_NAME", "NEURON_STATIC", "TARGET_METRICS",
    "TARGET_NAMES", "WIDTH_PROFILE_IDS", "FeatureSpec", "target_names",
]
