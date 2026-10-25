"""What a neuron row looks like: its features and its prediction targets.

The schema is declared here as data rather than being implied by whatever
:mod:`staticsbfl.dataset.build` happens to append. Three things follow from that:
a unit test can assert the built table matches the declaration column for
column; a feature can be described once and that description reaches the
write-up; and features carry a *group* label, so an ablation between
"architecture only" and "neuron-static only" is a filter rather than a rewrite.

Why statistics of the weights, and not the weights themselves
-------------------------------------------------------------
A neuron's incoming weights are a vector of length ``fan_in``, which differs
between layers and between models, so they cannot be fixed-width columns of one
table. Summarising each weight vector is what makes the rows commensurable.

Permutation equivariance
------------------------
Every neuron-static feature here is a function of the neuron's own weights, so
it travels with the unit if the units of a layer are reordered. The neuron's
*index* is deliberately not a feature: hidden units are exchangeable, so an
index means nothing across independently initialised models. See
:data:`EXCLUDED_FEATURES` for what was left out and why.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

#: Feature groups, for ablations.
NEURON_STATIC = "neuron_static"
ARCHITECTURE = "architecture"
FEATURE_GROUPS: Tuple[str, ...] = (NEURON_STATIC, ARCHITECTURE)


@dataclass(frozen=True)
class FeatureSpec:
    """One column of the feature table."""

    name: str
    group: str
    description: str


#: The feature columns, in table order.
FEATURES: Tuple[FeatureSpec, ...] = (
    # --- the neuron's own incoming weights: row j of this layer's matrix -----
    FeatureSpec("in_weight_mean", NEURON_STATIC, "Mean of the neuron's incoming weight vector."),
    FeatureSpec("in_weight_std", NEURON_STATIC, "Standard deviation of the incoming weights."),
    FeatureSpec("in_weight_l1", NEURON_STATIC, "L1 norm of the incoming weights."),
    FeatureSpec("in_weight_l2", NEURON_STATIC, "L2 norm of the incoming weights."),
    FeatureSpec("in_weight_absmax", NEURON_STATIC, "Largest absolute incoming weight."),
    FeatureSpec("in_weight_frac_small", NEURON_STATIC, "Fraction of incoming weights below 1e-3 in absolute value."),
    FeatureSpec("bias", NEURON_STATIC, "The neuron's own bias term."),
    # --- the neuron's outgoing weights: column j of the NEXT layer's matrix --
    FeatureSpec("out_weight_mean", NEURON_STATIC, "Mean of the weights the next layer applies to this neuron."),
    FeatureSpec("out_weight_std", NEURON_STATIC, "Standard deviation of those outgoing weights."),
    FeatureSpec("out_weight_l1", NEURON_STATIC, "L1 norm of the outgoing weights."),
    FeatureSpec("out_weight_l2", NEURON_STATIC, "L2 norm of the outgoing weights: how strongly the next layer reads this neuron."),
    FeatureSpec("out_weight_absmax", NEURON_STATIC, "Largest absolute outgoing weight."),
    FeatureSpec("fan_in", NEURON_STATIC, "Number of incoming connections."),
    FeatureSpec("fan_out", NEURON_STATIC, "Number of outgoing connections."),
    # --- the architecture the neuron sits in --------------------------------
    FeatureSpec("depth", ARCHITECTURE, "Number of hidden layers in the model."),
    FeatureSpec("layer_index", ARCHITECTURE, "Zero-based index of the neuron's layer."),
    FeatureSpec("layer_position", ARCHITECTURE, "layer_index / (depth - 1), or 0 for a single-layer model."),
    FeatureSpec("layer_width", ARCHITECTURE, "Number of neurons in the neuron's layer."),
    FeatureSpec("prev_layer_width", ARCHITECTURE, "Width of the preceding layer; the flattened input size for the first."),
    FeatureSpec("next_layer_width", ARCHITECTURE, "Width of the following layer; the class count for the last."),
    FeatureSpec("activation_id", ARCHITECTURE, "Integer id of the model's activation function."),
    FeatureSpec("total_hidden_units", ARCHITECTURE, "Sum of all hidden layer widths in the model."),
    FeatureSpec("width_profile_id", ARCHITECTURE, "0 flat, 1 narrowing, 2 widening, 3 mixed."),
    FeatureSpec("n_parameters", ARCHITECTURE, "Total trainable parameters of the model."),
)

FEATURE_NAMES: Tuple[str, ...] = tuple(f.name for f in FEATURES)

FEATURES_BY_NAME: Dict[str, FeatureSpec] = {f.name: f for f in FEATURES}

#: Deliberately absent, with the reason. Kept as data so the write-up can cite
#: it and a test can assert none of them leaked into the table.
EXCLUDED_FEATURES: Dict[str, str] = {
    "neuron_index": (
        "Hidden units are exchangeable, so an index carries no meaning across "
        "independently initialised models. Phase 1 measured the per-index rank "
        "correlation across seeds at approximately zero."
    ),
    "relative_neuron_index": "Same reason as neuron_index.",
    "activation_rate": (
        "Requires executing the model, which is exactly what the predictor is "
        "meant to avoid. Useful as a diagnostic and as a baseline, not as a "
        "feature."
    ),
    "layer_weight_statistics": (
        "Constant within a layer, so it cannot distinguish two neurons of the "
        "same layer. Superseded by the per-neuron incoming/outgoing statistics."
    ),
}

#: Ordinal encoding of ArchitectureConfig.width_profile.
WIDTH_PROFILE_IDS: Dict[str, int] = {
    "flat": 0, "narrowing": 1, "widening": 2, "mixed": 3,
}

#: Suspiciousness metrics that become targets.
TARGET_METRICS: Tuple[str, ...] = ("tarantula", "ochiai", "dstar")


def target_names(metric: str) -> Tuple[str, str, str]:
    """The three target columns derived from one metric.

    ``<metric>``             the raw suspiciousness value;
    ``<metric>_rank_model``  its rank among all neurons of the model, in [0, 1];
    ``<metric>_rank_layer``  its rank among the neurons of its own layer.

    Both a raw value and a rank are produced because they answer different
    questions: the raw value is what a regressor reproduces, the rank is what a
    localization result is actually judged on. The layer-wise rank additionally
    removes the between-layer differences that dominate a pooled comparison.
    """
    return (metric, f"{metric}_rank_model", f"{metric}_rank_layer")


#: Fraction of scored samples on which the neuron fired, in ``[0, 1]``.
#:
#: Not a suspiciousness metric and never a prediction target -- it is recorded so
#: that "this neuron was never active" can be told apart from "this neuron was
#: active and the evidence pointed away from it". All three metrics map both
#: cases to exactly 0.0, so without this column an evaluation cannot separate
#: the two, and a predictor that has merely learned to recognise dead units
#: scores the same as one that has learned to rank live ones.
ACTIVITY_TARGET: str = "activity_rate"

#: Every target column, in table order.
TARGET_NAMES: Tuple[str, ...] = (
    *(name for metric in TARGET_METRICS for name in target_names(metric)),
    ACTIVITY_TARGET,
)
